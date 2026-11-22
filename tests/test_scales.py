"""Scale scoring on small hand-built response tables."""

import numpy as np
import pandas as pd
import pytest

from dyad_sync_metrics.scales import (
    cronbach_alpha,
    dyad_scores,
    reliability,
    reverse_score,
    scale_scores,
    score_items,
    validate_item_plan,
    with_subset_scale,
)

PLAN = pd.DataFrame(
    {
        "item": ["A1", "A2", "A3", "B1", "B2"],
        "scale": ["ScaleA", "ScaleA", "ScaleA", "ScaleB", "ScaleB"],
        "reverse": [False, True, False, False, True],
    }
)


def responses(values: dict[int, list[float]], condition: str = "a") -> pd.DataFrame:
    """Rows for persons 0..n (dyad person // 2), values in PLAN's item order."""
    rows = [
        (person, person // 2, condition, item, value)
        for person, answers in values.items()
        for item, value in zip(PLAN["item"], answers, strict=True)
    ]
    return pd.DataFrame(rows, columns=["person", "dyad", "condition", "item", "value"])


ANSWERS = {0: [7, 1, 6, 5, 2], 1: [2, 6, 3, 4, 4], 2: [5, 3, 5, 1, 7], 3: [4, 4, 4, 6, 1]}


def scored() -> pd.DataFrame:
    return score_items(responses(ANSWERS), PLAN, scale_min=1, scale_max=7)


def test_reverse_score_mirrors_the_scale():
    assert reverse_score(np.array([1, 4, 7]), 1, 7).tolist() == [7, 4, 1]
    assert reverse_score(np.array([0, 3]), 0, 4).tolist() == [4, 1]


def test_score_items_reverses_exactly_the_reverse_keyed_items():
    out = scored()
    reversed_rows = out["reverse"]
    assert (out.loc[reversed_rows, "value"] == 8 - out.loc[reversed_rows, "value_raw"]).all()
    assert (out.loc[~reversed_rows, "value"] == out.loc[~reversed_rows, "value_raw"]).all()
    assert out.loc[(out["person"] == 0) & (out["item"] == "A2"), "value"].item() == 7


def test_missing_answers_stay_missing():
    answers = {**ANSWERS, 1: [2, np.nan, 3, 4, 4]}
    out = score_items(responses(answers), PLAN, scale_min=1, scale_max=7)
    assert np.isnan(out.loc[(out["person"] == 1) & (out["item"] == "A2"), "value"].item())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r[r["item"] != "B2"], "without any response"),
        (lambda r: r.replace({"item": {"B2": "C9"}}), "item plan lacks"),
        (lambda r: pd.concat([r, r.iloc[:1]]), "repeated answers"),
        (lambda r: r.assign(value=r["value"].where(r.index != 0, 9)), "outside 1..7"),
        (lambda r: r.drop(columns="dyad"), "lacks column"),
    ],
)
def test_score_items_fails_loudly(mutate, message):
    with pytest.raises(ValueError, match=message):
        score_items(mutate(responses(ANSWERS)), PLAN, scale_min=1, scale_max=7)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (PLAN.drop(columns="reverse"), "lacks column"),
        (pd.concat([PLAN, PLAN.iloc[:1]]), "more than once"),
        (PLAN.assign(reverse=["no", "yes", "no", "no", "yes"]), "booleans"),
        (PLAN.assign(scale=[None, "ScaleA", "ScaleA", "ScaleB", "ScaleB"]), "without a scale"),
    ],
)
def test_invalid_item_plans_are_rejected(plan, message):
    with pytest.raises(ValueError, match=message):
        validate_item_plan(plan)


def test_scale_scores_are_item_means_after_reverse_scoring():
    out = scale_scores(scored()).set_index("person")
    # Person 0: ScaleA items 7, 8 - 1, 6; ScaleB items 5, 8 - 2.
    assert out.loc[0, "ScaleA"] == pytest.approx((7 + 7 + 6) / 3)
    assert out.loc[0, "ScaleB"] == pytest.approx((5 + 6) / 2)
    assert out.loc[3, "dyad"] == 1


def test_dyad_scores_report_partner_mean_and_absolute_difference():
    out = dyad_scores(scale_scores(scored()), ["ScaleA", "ScaleB"]).set_index("dyad")
    person = scale_scores(scored()).set_index("person")
    assert out.loc[0, "ScaleA"] == pytest.approx(
        (person.loc[0, "ScaleA"] + person.loc[1, "ScaleA"]) / 2
    )
    assert out.loc[0, "ScaleA_absdiff"] == pytest.approx(
        abs(person.loc[0, "ScaleA"] - person.loc[1, "ScaleA"])
    )
    assert list(out.columns) == [
        "condition",
        "ScaleA",
        "ScaleB",
        "ScaleA_absdiff",
        "ScaleB_absdiff",
    ]


def test_dyad_scores_need_exactly_two_partners():
    person = scale_scores(scored())
    with pytest.raises(ValueError, match="other than two persons"):
        dyad_scores(person[person["person"] != 3], ["ScaleA"])


def test_cronbach_alpha_matches_the_formula():
    m = np.array([[1, 2, 2], [2, 2, 3], [3, 4, 3], [4, 5, 5], [5, 4, 4]], dtype=float)
    k = m.shape[1]
    expected = k / (k - 1) * (1 - m.var(axis=0, ddof=1).sum() / m.sum(axis=1).var(ddof=1))
    assert cronbach_alpha(m) == pytest.approx(expected)
    assert cronbach_alpha(np.column_stack([m[:, 0]] * 3)) == pytest.approx(1.0)


def test_cronbach_alpha_drops_incomplete_persons_and_handles_degenerate_input():
    m = np.array([[1, 2], [2, 3], [3, 3], [np.nan, 1]])
    assert cronbach_alpha(m) == pytest.approx(cronbach_alpha(m[:3]))
    assert np.isnan(cronbach_alpha(m[:, :1]))
    assert np.isnan(cronbach_alpha(m[:1]))
    assert np.isnan(cronbach_alpha(np.full((3, 2), 4.0)))


def test_reliability_reports_alpha_per_scale_and_condition():
    both = pd.concat(
        [score_items(responses(ANSWERS, c), PLAN, scale_min=1, scale_max=7) for c in ("a", "b")]
    )
    out = reliability(both)
    assert set(zip(out["scale"], out["condition"], strict=True)) == {
        ("ScaleA", "a"),
        ("ScaleA", "b"),
        ("ScaleB", "a"),
        ("ScaleB", "b"),
    }
    row = out[(out["scale"] == "ScaleA") & (out["condition"] == "a")].iloc[0]
    assert (row["n_items"], row["n_persons"]) == (3, 4)
    matrix = (
        scored().query("scale == 'ScaleA'").pivot(index="person", columns="item", values="value")
    )
    assert row["alpha"] == pytest.approx(cronbach_alpha(matrix))


def test_subset_scale_is_scored_from_exactly_the_named_items():
    out = with_subset_scale(scored(), ["A1", "A3"], "ScaleA_short")
    scores = scale_scores(out).set_index("person")
    assert scores.loc[0, "ScaleA_short"] == pytest.approx((7 + 6) / 2)
    assert scores.loc[0, "ScaleA"] == pytest.approx((7 + 7 + 6) / 3)
    rel = reliability(out).set_index("scale")
    assert rel.loc["ScaleA_short", "n_items"] == 2
    assert len(out) == len(scored()) + 2 * 4


@pytest.mark.parametrize(
    ("items", "name", "message"),
    [
        (["A1", "A9"], "short", "without scored rows"),
        ([], "short", "distinct items"),
        (["A1", "A1"], "short", "distinct items"),
        (["A1", "A3"], "ScaleA", "already exists"),
    ],
)
def test_subset_scale_fails_loudly(items, name, message):
    with pytest.raises(ValueError, match=message):
        with_subset_scale(scored(), items, name)
