"""Pseudo-dyad and circular-shift baselines on hand-built sessions."""

import numpy as np
import pytest

from dyad_sync_metrics.gaze import (
    gaze_convergence_excluded_share,
    gaze_convergence_rate,
    gaze_convergence_series,
    mutual_gaze_fraction,
)
from dyad_sync_metrics.surrogates import (
    circular_shift_correction,
    pseudo_dyad_baseline,
    pseudo_dyad_correction,
    pseudo_pairs,
)
from dyad_sync_metrics.sync import windowed_xcorr

FS = 30.0
TWO_WINDOWS = int(30 * FS) * 2
MAX_DIST_M = 2.5
HEAD_EXCLUSION_M = 0.30


def session(p1, p2):
    return {"p1": p1, "p2": p2}


def angle(*deg, n: int = 20) -> np.ndarray:
    """Angle-to-partner series made of constant segments, each n frames long."""
    return np.concatenate([np.full(n, float(g)) for g in deg])


def mutual_gaze_10(p1, p2):
    return mutual_gaze_fraction(p1, p2, threshold_deg=10)


def strength(p1, p2):
    return windowed_xcorr(p1, p2, FS, use_abs=True)


def direction(p1, p2):
    return windowed_xcorr(p1, p2, FS, use_abs=False, max_lag_sec=0)


# ---------------------------------------------------------------- Pairing and validation


def test_pseudo_pairs_are_all_ordered_wrong_partner_pairs():
    dyads = {d: session(f"{d}.1", f"{d}.2") for d in ("x", "y", "z")}
    pairs = set(pseudo_pairs(dyads))
    assert len(pairs) == 6
    assert ("x.1", "y.2") in pairs and ("y.1", "x.2") in pairs
    assert all(p1[0] != p2[0] for p1, p2 in pairs)


def test_a_session_outside_the_named_conditions_is_rejected():
    sessions = {
        (0, "a"): session(angle(0), angle(0)),
        (1, "a"): session(angle(0), angle(0)),
        (0, "c"): session(angle(0), angle(0)),
    }
    with pytest.raises(ValueError, match="'c'"):
        pseudo_dyad_baseline(sessions, mutual_gaze_10, conditions=["a", "b"])


def test_a_condition_with_fewer_than_two_dyads_is_rejected():
    sessions = {
        (0, "a"): session(angle(0), angle(0)),
        (1, "a"): session(angle(0), angle(0)),
        (0, "b"): session(angle(0), angle(0)),
    }
    with pytest.raises(ValueError, match="'b' has 1"):
        pseudo_dyad_baseline(sessions, mutual_gaze_10, conditions=["a", "b"])


# ---------------------------------------------------------------- Mutual gaze baseline


def test_pseudo_pair_without_shared_gaze_gives_zero_and_with_constant_gaze_one():
    never = {(d, "a"): session(angle(0), angle(90)) for d in range(2)}
    always = {(d, "a"): session(angle(0), angle(0)) for d in range(2)}
    assert pseudo_dyad_baseline(never, mutual_gaze_10, ["a"])["a"].mean == 0.0
    assert pseudo_dyad_baseline(always, mutual_gaze_10, ["a"])["a"].mean == 1.0


def test_baseline_pairs_the_wrong_partner():
    """In both real dyads only one person looks at the other, so real mutual gaze is 0.
    Cross-paired, one pseudo-pair matches fully and the other not at all."""
    sessions = {(0, "a"): session(angle(0), angle(90)), (1, "a"): session(angle(90), angle(0))}
    baseline = pseudo_dyad_baseline(sessions, mutual_gaze_10, ["a"])["a"]
    assert baseline.n_pairs == 2
    assert baseline.mean == pytest.approx(0.5)


def test_conditions_get_separate_baselines():
    sessions = {
        (0, "a"): session(angle(0), angle(0)),
        (1, "a"): session(angle(0), angle(0)),
        (0, "b"): session(angle(0), angle(90)),
        (1, "b"): session(angle(0), angle(90)),
    }
    baselines = pseudo_dyad_baseline(sessions, mutual_gaze_10, ["a", "b"])
    assert baselines["a"].mean == 1.0
    assert baselines["b"].mean == 0.0


def test_correction_subtracts_the_baseline_of_the_dyads_own_condition():
    sessions = {
        (0, "a"): session(angle(0), angle(0)),
        (1, "a"): session(angle(0), angle(90)),
        (0, "b"): session(angle(0), angle(90)),
        (1, "b"): session(angle(90), angle(0)),
    }
    result = pseudo_dyad_correction(sessions, mutual_gaze_10, ["a", "b"])
    # Condition a: pairs (0.p1, 1.p2) = 0 and (1.p1, 0.p2) = 1, baseline 0.5.
    assert result[(0, "a")] == pytest.approx((1.0, 0.5, 0.5))
    assert result[(1, "a")] == pytest.approx((0.0, 0.5, -0.5))
    # Condition b: pairs (0.p1, 1.p2) = 1 and (1.p1, 0.p2) = 0, baseline 0.5.
    assert result[(0, "b")] == pytest.approx((0.0, 0.5, -0.5))


def test_unequal_lengths_and_nan_frames_follow_the_metric():
    long, short = angle(0, 90), angle(0)
    sessions = {(0, "a"): session(long, long), (1, "a"): session(short, short)}
    assert pseudo_dyad_baseline(sessions, mutual_gaze_10, ["a"])["a"].mean == 1.0

    ang_1 = np.array([0.0, 0.0, np.nan, 0.0])
    ang_2 = np.array([0.0, 90.0, 0.0, np.nan])
    sessions = {(0, "a"): session(ang_1, ang_1), (1, "a"): session(ang_2, ang_2)}
    assert pseudo_dyad_baseline(sessions, mutual_gaze_10, ["a"])["a"].mean == pytest.approx(0.5)


def test_an_all_nan_baseline_is_nan_and_counts_its_pairs():
    nan = np.full(5, np.nan)
    sessions = {(d, "a"): session(nan, nan) for d in range(3)}
    baseline = pseudo_dyad_baseline(sessions, mutual_gaze_10, ["a"])["a"]
    assert np.isnan(baseline.mean)
    assert baseline.n_pairs == 6


# ---------------------------------------------------------------- Gaze coupling baselines


def converging_angles():
    rng = np.random.default_rng(2)
    shared = rng.random(TWO_WINDOWS)
    p1 = 5.0 + 40.0 * shared
    p2 = 8.0 + 30.0 * (0.8 * shared + 0.2 * rng.random(TWO_WINDOWS))
    return p1, p2


def test_each_coupling_variant_needs_a_baseline_with_its_own_metric():
    """One pseudo-pair converges and the other alternates: the signed baselines cancel
    to about zero while both |r| values stay near 1. Subtracting the |r| baseline from
    the signed value would be a large, meaningless correction."""
    p1, p2 = converging_angles()
    sessions = {(0, "a"): session(p1, p2), (1, "a"): session(p1, 45.0 - p2)}
    assert pseudo_dyad_baseline(sessions, direction, ["a"])["a"].mean == pytest.approx(
        0.0, abs=0.05
    )
    assert pseudo_dyad_baseline(sessions, strength, ["a"])["a"].mean > 0.9


def test_coupling_baseline_pairs_the_wrong_partner():
    rng = np.random.default_rng(3)
    a, b = rng.random(TWO_WINDOWS) * 45.0, rng.random(TWO_WINDOWS) * 45.0
    sessions = {(0, "a"): session(a, b), (1, "a"): session(b, a)}
    assert pseudo_dyad_baseline(sessions, direction, ["a"])["a"].mean > 0.99
    assert abs(direction(a, b)) < 0.5


# ---------------------------------------------------------------- Gaze convergence baseline


def convergence_rate(threshold_m):
    def metric(p1, p2):
        dist, excluded = gaze_convergence_series(
            p1["origin"],
            p1["direction"],
            p1["head"],
            p2["origin"],
            p2["direction"],
            p2["head"],
            max_dist_m=MAX_DIST_M,
            head_exclusion_m=HEAD_EXCLUSION_M,
        )
        return gaze_convergence_rate(dist, excluded, threshold_m)

    return metric


def facing_session(x_1: float, x_2: float, n: int = 8, head_y: float = 1.0) -> dict:
    """Two people facing each other across 2 m, displaced sideways to x_1 and x_2, so
    their rays are |x_1 - x_2| apart. The head points sit head_y above the gaze origins,
    clear of the exclusion unless head_y is lowered."""
    zeros, heads = np.zeros(n), np.full(n, head_y)
    return session(
        {
            "origin": np.stack([np.full(n, x_1), zeros, zeros], axis=1),
            "direction": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "head": np.stack([np.full(n, x_1), heads, zeros], axis=1),
        },
        {
            "origin": np.stack([np.full(n, x_2), zeros, np.full(n, 2.0)], axis=1),
            "direction": np.tile([0.0, 0.0, -1.0], (n, 1)),
            "head": np.stack([np.full(n, x_2), heads, np.full(n, 2.0)], axis=1),
        },
    )


def test_convergence_baseline_pairs_the_wrong_partner():
    sessions = {(0, "a"): facing_session(0.0, 1.0), (1, "a"): facing_session(1.0, 0.0)}
    metric = convergence_rate(0.10)
    assert pseudo_dyad_baseline(sessions, metric, ["a"])["a"].mean == pytest.approx(1.0)
    assert metric(sessions[(0, "a")]["p1"], sessions[(0, "a")]["p2"]) == pytest.approx(0.0)


def test_convergence_baseline_applies_the_exclusion_with_the_pseudo_pairs_own_heads():
    """With the heads on the sightlines every pseudo-pair frame is excluded, so no
    eligible frame is left and the baseline is NaN rather than 1.0."""
    metric = convergence_rate(0.10)
    clear = {(d, "a"): facing_session(0.0, 0.0) for d in range(2)}
    on_sightline = {(d, "a"): facing_session(0.0, 0.0, head_y=0.0) for d in range(2)}
    assert pseudo_dyad_baseline(clear, metric, ["a"])["a"].mean == pytest.approx(1.0)
    assert np.isnan(pseudo_dyad_baseline(on_sightline, metric, ["a"])["a"].mean)


def test_each_threshold_gets_its_own_baseline():
    sessions = {(d, "a"): facing_session(0.0, 0.07) for d in range(2)}
    means = {
        thr: pseudo_dyad_baseline(sessions, convergence_rate(thr), ["a"])["a"].mean
        for thr in (0.02, 0.05, 0.10, 0.20)
    }
    assert means == pytest.approx({0.02: 0.0, 0.05: 0.0, 0.10: 1.0, 0.20: 1.0})


def co_attention_session(look_frames_1, look_frames_2, n: int = 100) -> dict:
    """Both people attend the same off-axis object in every frame, except that person 1
    looks at person 2 in look_frames_1 and person 2 at person 1 in look_frames_2."""
    obj = np.array([2.0, 0.0, 1.0])
    o1 = np.tile([0.0, 0.0, 0.0], (n, 1))
    o2 = np.tile([0.0, 0.0, 2.0], (n, 1))
    d1 = np.tile(obj / np.linalg.norm(obj), (n, 1))
    d2 = np.tile((obj - o2[0]) / np.linalg.norm(obj - o2[0]), (n, 1))
    d1[list(look_frames_1)] = [0.0, 0.0, 1.0]
    d2[list(look_frames_2)] = [0.0, 0.0, -1.0]
    return session(
        {"origin": o1, "direction": d1, "head": o1}, {"origin": o2, "direction": d2, "head": o2}
    )


def test_corrected_convergence_does_not_depend_on_when_partners_look_at_each_other():
    """Regression: two dyads with identical co-attention that differ only in whether
    their looks at each other coincide must get the same corrected rate. Under an
    unconditional denominator their excluded shares would scale the raw rates
    differently from the pseudo-pair baseline."""
    sessions = {
        (0, "a"): co_attention_session(range(20), range(20)),
        (1, "a"): co_attention_session(range(20), range(20, 40)),
    }

    def share(s):
        dist, excluded = gaze_convergence_series(
            s["p1"]["origin"],
            s["p1"]["direction"],
            s["p1"]["head"],
            s["p2"]["origin"],
            s["p2"]["direction"],
            s["p2"]["head"],
            max_dist_m=MAX_DIST_M,
            head_exclusion_m=HEAD_EXCLUSION_M,
        )
        return gaze_convergence_excluded_share(dist, excluded)

    assert share(sessions[(0, "a")]) != pytest.approx(share(sessions[(1, "a")]))
    for thr in (0.02, 0.05, 0.10, 0.20):
        result = pseudo_dyad_correction(sessions, convergence_rate(thr), ["a"])
        assert result[(0, "a")].corrected == pytest.approx(0.0)
        assert result[(1, "a")].corrected == pytest.approx(0.0)


# ---------------------------------------------------------------- Circular shift


def agreement(a, b):
    return float(np.mean(a == b))


def test_circular_shift_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(5)
    a, b = rng.integers(0, 3, 40), rng.integers(0, 3, 40)
    first = circular_shift_correction(a, b, agreement, n_iter=50, seed=1)
    assert first == circular_shift_correction(a, b, agreement, n_iter=50, seed=1)
    assert first != circular_shift_correction(a, b, agreement, n_iter=50, seed=2)


def test_circular_shift_subtracts_a_chance_level_baseline():
    a = np.random.default_rng(7).integers(0, 3, 300)
    observed, baseline, corrected = circular_shift_correction(
        a, a.copy(), agreement, n_iter=200, seed=0
    )
    assert observed == pytest.approx(1.0)
    assert baseline == pytest.approx(1 / 3, abs=0.05)
    assert corrected == pytest.approx(observed - baseline)


def test_circular_shift_never_draws_a_zero_shift():
    """With two samples the only non-zero shift swaps them, so agreement drops to 0."""
    a = np.array([0, 1])
    observed, baseline, _ = circular_shift_correction(a, a.copy(), agreement, n_iter=10, seed=0)
    assert observed == 1.0
    assert baseline == 0.0


def test_circular_shift_of_a_single_sample_has_no_baseline():
    result = circular_shift_correction(np.array([1]), np.array([1]), agreement, n_iter=10, seed=0)
    assert result.observed == 1.0
    assert np.isnan(result.baseline) and np.isnan(result.corrected)
