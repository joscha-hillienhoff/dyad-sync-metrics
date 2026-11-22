import numpy as np
import pytest

from dyad_sync_metrics.signal import MIN_VALID_SAMPLES, interpolate_gaps, lowpass, speed


def test_interpolate_gaps_fills_interior_gaps_linearly_and_edges_with_nearest():
    x = np.array([np.nan, 10, np.nan, 20, 30, np.nan, 40, 50, 60, 70, 80, 90, 100, 110, np.nan])
    out = interpolate_gaps(x)
    valid = ~np.isnan(x)
    assert not np.isnan(out).any()
    assert out[valid] == pytest.approx(x[valid])
    assert out[0] == 10.0 and out[-1] == 110.0
    assert out[2] == pytest.approx(15.0)
    assert out[5] == pytest.approx(35.0)


def test_interpolate_gaps_leaves_a_near_empty_series_alone():
    x = np.array([np.nan, 10.0, np.nan, 20.0])
    assert (~np.isnan(x)).sum() < MIN_VALID_SAMPLES
    assert np.array_equal(interpolate_gaps(x), x, equal_nan=True)


def test_lowpass_removes_a_component_above_the_cutoff():
    fs = 30.0
    t = np.arange(600) / fs
    slow = np.sin(2 * np.pi * 0.5 * t)
    fast = 0.5 * np.sin(2 * np.pi * 12.0 * t)
    out = lowpass(slow + fast, fs=fs, cutoff_hz=6.0)
    # Away from the edges the slow component passes and the fast one is gone.
    assert out[60:-60] == pytest.approx(slow[60:-60], abs=0.02)


def test_lowpass_fills_gaps_before_filtering():
    fs = 30.0
    x = np.sin(2 * np.pi * 0.5 * np.arange(300) / fs)
    x[100:105] = np.nan
    assert not np.isnan(lowpass(x, fs=fs)).any()


def test_speed_of_constant_velocity_is_its_magnitude():
    fs = 30.0
    t = np.arange(300) / fs
    velocity = np.array([0.3, 0.0, 0.4])
    positions = t[:, None] * velocity
    assert speed(positions, t, fs=fs) == pytest.approx(np.full(300, 0.5))


def test_speed_uses_the_actual_timestamps():
    """An irregular frame interval must not show up as a speed change."""
    fs = 30.0
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.8, 1.2, 300)) / fs
    positions = t[:, None] * np.array([0.0, 0.2, 0.0])
    assert speed(positions, t, fs=fs) == pytest.approx(np.full(300, 0.2), abs=1e-9)
