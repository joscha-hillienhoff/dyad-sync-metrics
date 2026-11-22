"""The generators are deterministic, and the measures recover their ground truth."""

import numpy as np
import pytest

from dyad_sync_metrics import gaze as g
from dyad_sync_metrics.signal import speed
from dyad_sync_metrics.surrogates import pseudo_dyad_correction
from dyad_sync_metrics.sync import windowed_xcorr
from dyad_sync_metrics.synthetic import (
    ConditionParams,
    gaze,
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
