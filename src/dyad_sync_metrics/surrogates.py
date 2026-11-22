"""Chance baselines for dyadic measures.

Two people sharing a task produce some synchrony, mutual gaze and gaze-ray convergence
without any coordination behind it: they move at similar tempos, sit facing each other,
and look into the same workspace. A raw dyadic value therefore mixes coordination with
that structural share. A surrogate baseline estimates the structural share from pairs
that cannot be coordinated, and the corrected value is observed minus baseline.

Two surrogates are provided, and they answer different null models:

- pseudo-dyads (between dyads): person 1 of one dyad is paired with person 2 of another
  dyad recorded in the same condition. This keeps everything the task and the setting
  impose and removes the interaction itself.
- circular shift (within a dyad): one person's series is rotated in time against the
  partner's. This keeps each series' own distribution and autocorrelation exactly and
  removes only their alignment in time.

Sessions are passed as a mapping {(dyad, condition): session}, where a session is a
mapping with the keys "p1" and "p2" holding whatever per-person data the metric needs.
A metric is a function metric(p1, p2) -> float. The observed value of a dyad is
metric(session["p1"], session["p2"]); a pseudo-pair is metric(x["p1"], y["p2"]) for two
different dyads x and y. Because both run through the same function, the baseline is
always computed with the same metric, preprocessing and NaN handling as the value it is
subtracted from.
"""

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from itertools import permutations
from typing import Any, NamedTuple

import numpy as np

Session = Mapping[str, Any]
Metric = Callable[[Any, Any], float]


class Baseline(NamedTuple):
    mean: float
    """Mean metric value over the surrogate pairs, ignoring NaN values; NaN if all are."""
    n_pairs: int
    """Number of surrogate pairs the mean was formed over, NaN-valued pairs included."""


class Correction(NamedTuple):
    observed: float
    baseline: float
    corrected: float


def _nan_mean(values: list[float]) -> float:
    """Mean ignoring NaN; NaN without a RuntimeWarning when nothing is left."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else np.nan


def _by_condition(
    sessions: Mapping[tuple[Hashable, str], Session], conditions: Iterable[str]
) -> dict[str, dict[Hashable, Session]]:
    conditions = list(conditions)
    grouped: dict[str, dict[Hashable, Session]] = {c: {} for c in conditions}
    for (dyad, condition), session in sessions.items():
        if condition not in grouped:
            raise ValueError(
                f"session ({dyad!r}, {condition!r}) has a condition not in {conditions!r}"
            )
        grouped[condition][dyad] = session
    for condition, dyads in grouped.items():
        if len(dyads) < 2:
            raise ValueError(
                f"condition {condition!r} has {len(dyads)} dyad(s); pseudo-dyads need at least 2"
            )
    return grouped


def pseudo_pairs(dyads: Mapping[Hashable, Session]) -> Iterator[tuple[Any, Any]]:
    """All ordered pseudo-pairs of one condition, as (person 1 of x, person 2 of y) for
    every two different dyads x and y.

    Ordered, because person 1 of x with person 2 of y and person 1 of y with person 2 of
    x are two different pairs: the roles are not interchangeable.
    """
    return ((dyads[x]["p1"], dyads[y]["p2"]) for x, y in permutations(dyads, 2))


def pseudo_dyad_baseline(
    sessions: Mapping[tuple[Hashable, str], Session],
    metric: Metric,
    conditions: Iterable[str],
) -> dict[str, Baseline]:
    """The pseudo-dyad baseline of metric, separately for each condition.

    Pseudo-pairs are only formed within one condition, so each condition gets its own
    baseline. Every session's condition must be one of conditions, and each condition
    needs at least two dyads. The baseline is formed over the sessions passed in, so a
    missing recording shifts it for every remaining dyad of that condition; check
    completeness before calling.

    For measures based on absolute positions (gaze-ray convergence) all sessions must be
    expressed in a comparable world frame. Speed magnitudes and angles to the partner do
    not depend on where the people sat, but rays and head points do: recordings with
    different spawn points would put every pseudo-pair's rays far apart, push the
    baseline toward zero and silently inflate every corrected value.
    """
    baselines = {}
    for condition, dyads in _by_condition(sessions, conditions).items():
        values = [metric(p1, p2) for p1, p2 in pseudo_pairs(dyads)]
        baselines[condition] = Baseline(mean=_nan_mean(values), n_pairs=len(values))
    return baselines


def pseudo_dyad_correction(
    sessions: Mapping[tuple[Hashable, str], Session],
    metric: Metric,
    conditions: Iterable[str],
) -> dict[tuple[Hashable, str], Correction]:
    """Observed, baseline and corrected value of metric for every dyad and condition.

    Each dyad is corrected with the pseudo-dyad baseline of its own condition.
    """
    baselines = pseudo_dyad_baseline(sessions, metric, conditions)
    result = {}
    for (dyad, condition), session in sessions.items():
        observed = metric(session["p1"], session["p2"])
        baseline = baselines[condition].mean
        result[(dyad, condition)] = Correction(observed, baseline, observed - baseline)
    return result


def circular_shift_correction(
    a: np.ndarray, b: np.ndarray, metric: Metric, *, n_iter: int, seed: int
) -> Correction:
    """Within-dyad baseline: shift b circularly against a, n_iter times.

    metric(a, b) is computed on the real pair and on n_iter surrogates in which b is
    replaced by np.roll(b, shift), with shift drawn uniformly from [1, len(b) - 1]. A
    shift of 0 is never drawn, so no surrogate is identical to the observed pair. The
    rotation keeps b's values, distribution and gap pattern exactly and destroys only
    its alignment with a. Deterministic for a given seed.

    The baseline is NaN when b has fewer than two samples or every surrogate is NaN.
    """
    observed = metric(a, b)
    n = len(b)
    if n < 2:
        return Correction(observed, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, n, size=n_iter)
    baseline = _nan_mean([metric(a, np.roll(b, int(s))) for s in shifts])
    return Correction(observed, baseline, observed - baseline)
