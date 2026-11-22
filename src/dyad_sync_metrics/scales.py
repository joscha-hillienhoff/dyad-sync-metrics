"""Questionnaire scale scoring for dyads: reverse scoring, scale means, Cronbach's
alpha, and dyad-level partner mean and absolute difference.

Responses come as a long table with one row per answer and the columns person, dyad,
condition, item and value (the response as given, on a numeric scale from scale_min to
scale_max). Which item belongs to which scale, and which items are reverse-keyed, is an
item plan supplied by the caller: a table with the columns item, scale and reverse.

The usual order is score_items (validate, attach the plan, reverse score), then
scale_scores per person and dyad_scores per dyad, with reliability alongside.
with_subset_scale adds a scale scored from a named subset of another scale's items, for
example to check that a result does not hinge on items whose wording overlaps with
another instrument.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

RESPONSE_COLUMNS = ("person", "dyad", "condition", "item", "value")
"""Columns of a long response table."""
PLAN_COLUMNS = ("item", "scale", "reverse")
"""Columns of an item plan."""


def _require_columns(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"{name} lacks column(s) {missing}")


def validate_item_plan(plan: pd.DataFrame) -> None:
    """Rejects an item plan with missing columns, a repeated item, a missing scale name
    or a reverse flag that is not boolean."""
    _require_columns(plan, PLAN_COLUMNS, "item plan")
    repeated = plan.loc[plan["item"].duplicated(), "item"].tolist()
    if repeated:
        raise ValueError(f"item plan lists item(s) {repeated} more than once")
    if plan["scale"].isna().any():
        raise ValueError("item plan has items without a scale")
    if not plan["reverse"].map(lambda v: isinstance(v, bool | np.bool_)).all():
        raise ValueError("item plan column reverse must hold booleans")


def reverse_score(values, scale_min: float, scale_max: float):
    """Mirror responses on a scale_min..scale_max scale: scale_min + scale_max - value."""
    return scale_min + scale_max - values


def score_items(
    responses: pd.DataFrame, plan: pd.DataFrame, *, scale_min: float, scale_max: float
) -> pd.DataFrame:
    """The response table joined with the item plan, with reverse-keyed items reverse
    scored: columns of responses plus scale, reverse and value_raw (the response as
    given); value is the scored response.

    Raises rather than scoring a narrower basis than the plan describes: on an item
    missing from the plan (it would drop out of every scale) or from the responses (its
    scale would silently lose an item), on a value outside scale_min..scale_max, and on
    a repeated (person, condition, item) answer. Missing values (NaN) are allowed and
    stay missing.
    """
    _require_columns(responses, RESPONSE_COLUMNS, "response table")
    validate_item_plan(plan)
    unplanned = sorted(set(responses["item"]) - set(plan["item"]))
    if unplanned:
        raise ValueError(f"responses contain item(s) {unplanned} that the item plan lacks")
    unanswered = sorted(set(plan["item"]) - set(responses["item"]))
    if unanswered:
        raise ValueError(f"item plan lists item(s) {unanswered} without any response")
    repeated = responses.duplicated(["person", "condition", "item"], keep=False)
    if repeated.any():
        first = responses[repeated].iloc[0]
        raise ValueError(
            f"{int(repeated.sum())} repeated answers, first: person {first['person']!r}, "
            f"condition {first['condition']!r}, item {first['item']!r}"
        )
    values = pd.to_numeric(responses["value"])
    out_of_range = values.notna() & ((values < scale_min) | (values > scale_max))
    if out_of_range.any():
        raise ValueError(
            f"{int(out_of_range.sum())} value(s) outside {scale_min}..{scale_max}, "
            f"first: {values[out_of_range].iloc[0]}"
        )
    scored = responses.assign(value_raw=values).merge(
        plan[list(PLAN_COLUMNS)], on="item", how="left", validate="many_to_one"
    )
    reverse = scored["reverse"].astype(bool)
    scored["value"] = scored["value_raw"].where(
        ~reverse, reverse_score(scored["value_raw"], scale_min, scale_max)
    )
    return scored


def with_subset_scale(scored: pd.DataFrame, items: Sequence[str], name: str) -> pd.DataFrame:
    """scored with the rows of items repeated under the new scale name, so that
    scale_scores and reliability report the subset like any other scale.

    Raises when items is empty or repeats an item, when an item has no scored rows
    (the subset would silently shrink, or the scale come out all missing), and when
    name is already a scale.
    """
    items = list(items)
    if not items or len(set(items)) != len(items):
        raise ValueError(f"subset {name!r} needs distinct items, got {items}")
    absent = [i for i in items if i not in set(scored["item"])]
    if absent:
        raise ValueError(f"subset {name!r} names item(s) {absent} without scored rows")
    if name in set(scored["scale"]):
        raise ValueError(f"scale {name!r} already exists")
    subset = scored[scored["item"].isin(items)].assign(scale=name)
    return pd.concat([scored, subset], ignore_index=True)


def scale_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per person and condition, with the dyad and one column per scale holding
    the mean of the scored items the person answered."""
    return (
        scored.groupby(["dyad", "person", "condition", "scale"])["value"]
        .mean()
        .unstack("scale")
        .reset_index()
        .rename_axis(columns=None)
    )


def dyad_scores(person_scores: pd.DataFrame, scales: Sequence[str]) -> pd.DataFrame:
    """One row per dyad and condition: for each scale, the partners' mean and, as
    <scale>_absdiff, the absolute difference between them.

    Raises unless every dyad and condition has exactly two persons, since a mean over
    one partner would be reported as a dyad value.
    """
    _require_columns(person_scores, ["dyad", "person", "condition", *scales], "person scores")
    sizes = person_scores.groupby(["dyad", "condition"])["person"].nunique()
    wrong = sizes[sizes != 2]
    if len(wrong):
        raise ValueError(f"dyad and condition with other than two persons: {wrong.to_dict()}")
    grouped = person_scores.groupby(["dyad", "condition"])[list(scales)]
    absdiff = (grouped.max() - grouped.min()).add_suffix("_absdiff")
    return grouped.mean().join(absdiff).reset_index()


def cronbach_alpha(items: pd.DataFrame | np.ndarray) -> float:
    """Cronbach's alpha of a persons x items matrix:
    k / (k - 1) * (1 - sum of item variances / variance of the sum score).

    Persons with a missing item are dropped (listwise), so every variance is taken over
    the same persons. NaN with fewer than two items or two complete persons, or when
    the sum score does not vary.
    """
    matrix = np.asarray(items, dtype=float)
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    n, k = matrix.shape
    if k < 2 or n < 2:
        return np.nan
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float(k / (k - 1) * (1 - matrix.var(axis=0, ddof=1).sum() / total_var))


def reliability(scored: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha per scale and condition, with the number of items and of
    complete persons it rests on."""
    rows = []
    for (scale, condition), group in scored.groupby(["scale", "condition"]):
        matrix = group.pivot(index="person", columns="item", values="value")
        rows.append(
            {
                "scale": scale,
                "condition": condition,
                "n_items": matrix.shape[1],
                "n_persons": int(matrix.notna().all(axis=1).sum()),
                "alpha": cronbach_alpha(matrix),
            }
        )
    return pd.DataFrame(rows)
