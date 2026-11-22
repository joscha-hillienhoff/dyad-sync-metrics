"""The generators are deterministic, and the measures recover their ground truth."""

import numpy as np
import pytest

from dyad_sync_metrics import gaze as g
from dyad_sync_metrics.intervals import build_timelines, lagged_overlap, resolve_overlaps
from dyad_sync_metrics.scales import reliability, score_items
from dyad_sync_metrics.signal import speed
from dyad_sync_metrics.surrogates import circular_shift_correction, pseudo_dyad_correction
from dyad_sync_metrics.sync import windowed_xcorr
from dyad_sync_metrics.synthetic import (
    EDGE_OVERLAP_SEC,
    LABELS,
    ConditionParams,
    example_item_plan,
    expected_alpha,
    gaze,
    interval_codes,
    survey,
    synthetic_study,
    tracking,
)

FS = 30.0
MAX_DIST_M = 2.5
HEAD_EXCLUSION_M = 0.30


def head_sync(p1, p2):
    return windowed_xcorr(speed(p1["head"], p1["t"], FS), speed(p2["head"], p2["t"], FS), FS)


def angles(p1, p2):
    return (
        g.angle_to_target(p1["gaze_origin"], p1["gaze_direction"], p2["head"]),
        g.angle_to_target(p2["gaze_origin"], p2["gaze_direction"], p1["head"]),
    )


def convergence(p1, p2):
    return g.gaze_convergence_series(
        p1["gaze_origin"],
        p1["gaze_direction"],
        p1["head"],
        p2["gaze_origin"],
        p2["gaze_direction"],
        p2["head"],
        max_dist_m=MAX_DIST_M,
        head_exclusion_m=HEAD_EXCLUSION_M,
    )


def test_study_is_deterministic_for_a_given_seed():
    conditions = {"a": ConditionParams(coupling=0.5)}
    first = synthetic_study(3, conditions, n_dyads=2, n_frames=200, fs=FS)
    again = synthetic_study(3, conditions, n_dyads=2, n_frames=200, fs=FS)
    other = synthetic_study(4, conditions, n_dyads=2, n_frames=200, fs=FS)
    assert set(first) == {(0, "a"), (1, "a")}
    for key in first:
        for person in ("p1", "p2"):
            for name, values in first[key][person].items():
                assert np.array_equal(values, again[key][person][name])
    assert not np.array_equal(first[(0, "a")]["p1"]["head"], other[(0, "a")]["p1"]["head"])


def test_full_coupling_makes_person_2_a_delayed_copy_of_person_1():
    people = tracking(np.random.default_rng(0), n_frames=300, fs=FS, coupling=1.0, lag_sec=0.5)
    lag = round(0.5 * FS)
    moved_1 = people["p1"]["head"] - people["p1"]["head"].mean(axis=0)
    moved_2 = people["p2"]["head"] - people["p2"]["head"].mean(axis=0)
    assert moved_2[lag:] == pytest.approx(moved_1[:-lag], abs=0.01)


def test_tracking_rejects_a_coupling_outside_the_unit_interval():
    with pytest.raises(ValueError):
        tracking(np.random.default_rng(0), n_frames=10, fs=FS, coupling=1.5, lag_sec=0)


def test_gaze_rejects_probabilities_that_do_not_fit():
    head = np.zeros((10, 3))
    with pytest.raises(ValueError):
        gaze(np.random.default_rng(0), head, head + 1, p_mutual=0.6, p_shared=0.6)


def test_coupled_condition_shows_higher_corrected_synchrony_and_control_corrects_to_zero():
    """Ground truth: condition a is coupled, condition b is an independent control.

    The raw synchrony of the independent pairs is clearly above zero, since a lag
    search over noise always finds some correlation; the pseudo-dyad baseline removes
    exactly that share.
    """
    study = synthetic_study(
        0,
        {"a": ConditionParams(coupling=0.8), "b": ConditionParams(coupling=0.0)},
        n_dyads=5,
        n_frames=int(120 * FS),
        fs=FS,
    )
    result = pseudo_dyad_correction(study, head_sync, conditions=["a", "b"])
    by_condition = {c: [r for (_, cond), r in result.items() if cond == c] for c in "ab"}

    raw_b = np.mean([r.observed for r in by_condition["b"]])
    corrected_a = np.mean([r.corrected for r in by_condition["a"]])
    corrected_b = np.mean([r.corrected for r in by_condition["b"]])
    assert raw_b > 0.1
    assert corrected_b == pytest.approx(0.0, abs=0.03)
    assert corrected_a > 0.25
    assert min(r.corrected for r in by_condition["a"]) > max(r.corrected for r in by_condition["b"])


@pytest.mark.parametrize(("p_mutual", "p_shared"), [(0.15, 0.4), (0.05, 0.1)])
def test_gaze_rates_match_the_generator_probabilities(p_mutual, p_shared):
    study = synthetic_study(
        1,
        {"a": ConditionParams(p_mutual=p_mutual, p_shared=p_shared)},
        n_dyads=4,
        n_frames=20_000,
        fs=FS,
    )
    mutual, rates = [], []
    for s in study.values():
        p1, p2 = s["p1"], s["p2"]
        ang_1, ang_2 = angles(p1, p2)
        mutual_frames = (p1["target"] == -1) & (p2["target"] == -1)
        shared_frames = (p1["target"] == p2["target"]) & (p1["target"] >= 0)
        dist, excluded = convergence(p1, p2)

        # Exact against the realised gaze states...
        assert g.mutual_gaze_fraction(ang_1, ang_2, 10) == pytest.approx(mutual_frames.mean())
        assert np.array_equal(excluded, mutual_frames)
        assert g.gaze_convergence_rate(dist, excluded, 0.10) == pytest.approx(
            shared_frames.sum() / (~mutual_frames).sum()
        )
        mutual.append(g.mutual_gaze_fraction(ang_1, ang_2, 10))
        rates.append(g.gaze_convergence_rate(dist, excluded, 0.10))

    # ...and, within sampling tolerance, against the generator's probabilities.
    assert np.mean(mutual) == pytest.approx(p_mutual, abs=0.03)
    assert np.mean(rates) == pytest.approx(p_shared / (1 - p_mutual), abs=0.04)


# ---------------------------------------------------------------- interval codes


def lagged(a, b):
    return lagged_overlap(a, b, resolution_sec=0.1, max_lag_sec=2.0)


def test_interval_codes_are_deterministic_and_well_formed():
    first = interval_codes(np.random.default_rng(5), duration_sec=120, p_shared=0.5)
    again = interval_codes(np.random.default_rng(5), duration_sec=120, p_shared=0.5)
    assert first.equals(again)
    assert list(first.columns) == ["start", "end", "participant", "label"]
    assert set(first["participant"]) == {"p1", "p2"}
    assert set(first["label"]) <= set(LABELS)
    assert (first["end"] > first["start"]).all()
    p1 = first[first["participant"] == "p1"]
    assert p1["start"].iloc[1:].tolist() == p1["end"].iloc[:-1].tolist()
    assert (p1["label"].iloc[1:].to_numpy() != p1["label"].iloc[:-1].to_numpy()).all()
    assert p1["end"].max() >= 120


def test_full_sharing_makes_person_2_a_delayed_copy():
    codes = interval_codes(np.random.default_rng(0), duration_sec=300, p_shared=1.0, lag_sec=1.0)
    p1, p2 = (codes[codes["participant"] == p].reset_index(drop=True) for p in ("p1", "p2"))
    assert p2["label"].equals(p1["label"])
    assert np.allclose(p2["start"] - p1["start"], 1.0)
    timelines = build_timelines(codes)
    assert lagged(timelines.grids["p1"], timelines.grids["p2"]) == 1.0


@pytest.mark.parametrize("p_shared", [0.5, 0.9])
def test_lagged_overlap_recovers_the_sharing_probability(p_shared):
    codes = interval_codes(np.random.default_rng(2), duration_sec=1800, p_shared=p_shared)
    timelines = build_timelines(codes)
    assert lagged(timelines.grids["p1"], timelines.grids["p2"]) == pytest.approx(p_shared, abs=0.05)


def test_circular_shift_correction_orders_dyads_by_sharing_and_zeroes_chance():
    # With p_shared = 1 / len(LABELS), person 2 matches no more often than an
    # independent coder with the same label distribution.
    corrected = {}
    for p_shared in (1 / len(LABELS), 0.5, 0.9):
        codes = interval_codes(np.random.default_rng(3), duration_sec=1200, p_shared=p_shared)
        grids = build_timelines(codes).grids
        corrected[p_shared] = circular_shift_correction(
            grids["p1"], grids["p2"], lagged, n_iter=50, seed=0
        ).corrected
    assert corrected[1 / len(LABELS)] == pytest.approx(0.0, abs=0.06)
    assert corrected[1 / len(LABELS)] < corrected[0.5] < corrected[0.9]


def test_edge_cases_exercise_overlap_resolution_and_label_matching():
    codes = interval_codes(
        np.random.default_rng(4), duration_sec=120, p_shared=1.0, edge_cases=True
    )
    p1 = codes[codes["participant"] == "p1"]
    resolved, excluded = resolve_overlaps(p1)
    assert excluded == pytest.approx(EDGE_OVERLAP_SEC)
    # The nested segment splits its container in two: one more segment, same time.
    assert len(resolved) == len(p1) + 1
    p2_labels = codes.loc[codes["participant"] == "p2", "label"]
    assert not set(p2_labels) <= set(LABELS)
    timelines = build_timelines(codes)
    assert set(timelines.grids["p2"]) - {None} == set(LABELS)
    assert 0 < timelines.coverage["p1"] < 1


def test_lag_jitter_creates_adjacent_overlaps_that_resolve():
    codes = interval_codes(
        np.random.default_rng(6), duration_sec=300, p_shared=0.5, lag_jitter_sec=0.3
    )
    excluded = build_timelines(codes).excluded_sec
    assert excluded["p1"] == 0.0
    assert excluded["p2"] > 0.0


# ---------------------------------------------------------------- survey


def test_survey_is_deterministic_and_on_the_scale():
    plan = example_item_plan()
    first = survey(np.random.default_rng(0), plan, n_dyads=3, condition_shift={"a": 0, "b": 1})
    again = survey(np.random.default_rng(0), plan, n_dyads=3, condition_shift={"a": 0, "b": 1})
    assert first.equals(again)
    assert len(first) == 6 * 2 * len(plan)
    assert (first["dyad"] == first["person"] // 2).all()
    assert set(first["value"]) <= set(range(1, 8))


def test_reverse_keyed_items_correlate_positively_only_after_reverse_scoring():
    plan = example_item_plan()
    answers = survey(np.random.default_rng(1), plan, n_dyads=200, condition_shift={"a": 0.0})
    scored = score_items(answers, plan, scale_min=1, scale_max=7)

    def corr(column, a, b):
        wide = scored.pivot(index="person", columns="item", values=column)
        return np.corrcoef(wide[a], wide[b])[0, 1]

    assert corr("value_raw", "A1", "A3") < -0.3
    assert corr("value", "A1", "A3") > 0.3


def test_alpha_recovers_the_value_the_loadings_imply():
    plan = example_item_plan()
    answers = survey(
        np.random.default_rng(2), plan, n_dyads=1000, condition_shift={"a": 0.0, "b": 0.5}
    )
    rel = reliability(score_items(answers, plan, scale_min=1, scale_max=7))
    # Rounding to whole points adds error of variance about 1 / 12 to every item.
    noise_sd = np.sqrt(0.8**2 + 1 / 12)
    for row in rel.itertuples():
        loadings = plan.loc[plan["scale"] == row.scale, "loading"]
        assert row.alpha == pytest.approx(expected_alpha(loadings, noise_sd), abs=0.04)
    # The condition shift moves the scale mean, not the scale's internal consistency.
    by_condition = rel.pivot(index="scale", columns="condition", values="alpha")
    assert (by_condition["a"] - by_condition["b"]).abs().max() < 0.05
