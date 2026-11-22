import numpy as np
import pytest

from dyad_sync_metrics.sync import fisher_z_mean, windowed_xcorr

FS = 30.0
WINDOW_SEC = 30.0
TWO_WINDOWS = int(WINDOW_SEC * FS) * 2


def anticorrelated_pair():
    """Two non-negative, anticorrelated signals (like speed magnitudes). White noise, so
    the lag search finds the relationship only at lag 0."""
    a = np.random.default_rng(0).random(TWO_WINDOWS)
    return a, 1.0 - a


def test_fisher_z_mean_averages_in_z_space():
    assert fisher_z_mean([0.2, 0.6]) == pytest.approx(
        np.tanh((np.arctanh(0.2) + np.arctanh(0.6)) / 2)
    )
    assert fisher_z_mean([0.5]) == pytest.approx(0.5)
    assert np.isnan(fisher_z_mean([]))


def test_fisher_z_mean_clips_perfect_correlations():
    assert np.isfinite(fisher_z_mean([1.0, -1.0, 1.0]))


def test_anticorrelated_pair_is_high_with_abs():
    a, b = anticorrelated_pair()
    assert 0.9 < windowed_xcorr(a, b, FS, use_abs=True) < 1.0


def test_anticorrelated_pair_is_negative_with_sign():
    a, b = anticorrelated_pair()
    assert -1.0 < windowed_xcorr(a, b, FS, use_abs=False) < -0.9


def test_default_is_the_absolute_variant():
    a, b = anticorrelated_pair()
    assert windowed_xcorr(a, b, FS) == pytest.approx(windowed_xcorr(a, b, FS, use_abs=True))


def test_zero_lag_value_is_the_fisher_z_mean_of_per_window_correlations():
    """Pins the averaging down against a direct computation: two non-overlapping
    windows, lag 0, so each window contributes its plain Pearson r."""
    rng = np.random.default_rng(1)
    n = TWO_WINDOWS + 1  # one extra sample so that range() reaches the second window
    a = rng.random(n)
    b = 0.6 * a + 0.4 * rng.random(n)
    win = int(WINDOW_SEC * FS)
    r1 = np.corrcoef(a[:win], b[:win])[0, 1]
    r2 = np.corrcoef(a[win : 2 * win], b[win : 2 * win])[0, 1]
    got = windowed_xcorr(a, b, FS, use_abs=False, overlap=0.0, max_lag_sec=0)
    assert got == pytest.approx(np.tanh((np.arctanh(r1) + np.arctanh(r2)) / 2))


def test_positively_correlated_pair_agrees_in_both_variants():
    rng = np.random.default_rng(1)
    a = rng.random(TWO_WINDOWS)
    b = 0.8 * a + 0.2 * rng.random(TWO_WINDOWS)
    assert windowed_xcorr(a, b, FS, use_abs=True) == pytest.approx(
        windowed_xcorr(a, b, FS, use_abs=False)
    )


def test_lagged_follower_is_found_by_the_lag_search():
    rng = np.random.default_rng(2)
    a = rng.random(TWO_WINDOWS + 30)
    lag = int(0.5 * FS)
    b = np.roll(a, lag)
    assert windowed_xcorr(a, b, FS, max_lag_sec=1.0) > 0.95
    assert windowed_xcorr(a, b, FS, max_lag_sec=0) < 0.3


def test_unequal_lengths_are_truncated_to_the_shorter():
    a, b = anticorrelated_pair()
    assert windowed_xcorr(np.concatenate([a, a]), b, FS) == pytest.approx(windowed_xcorr(a, b, FS))


def test_constant_windows_are_skipped_and_no_window_gives_nan():
    a, b = anticorrelated_pair()
    assert np.isnan(windowed_xcorr(np.ones_like(a), b, FS))
    assert np.isnan(windowed_xcorr(a[:10], b[:10], FS))


def test_window_and_lag_parameters_change_the_windowing():
    rng = np.random.default_rng(4)
    shared = rng.random(TWO_WINDOWS)
    a = 0.5 * shared + 0.5 * rng.random(TWO_WINDOWS)
    b = 0.5 * shared + 0.5 * rng.random(TWO_WINDOWS)
    short = windowed_xcorr(a, b, FS, use_abs=False, window_sec=10, max_lag_sec=2)
    assert not np.isnan(short)
    assert short != pytest.approx(windowed_xcorr(a, b, FS, use_abs=False))


def test_lag_search_can_invert_the_sign_which_is_why_direction_is_taken_at_lag_0():
    """Regression test for the documented sign-at-max-abs-lag defect.

    Two series alternating at a 6 s rhythm are in anti-phase, so the honest direction
    is negative, and lag 0 reports that. The lag search picks its lag by maximum |r|
    and reads the sign off afterwards; half a period (3 s) lies inside the +/- 5 s
    search range, so the winning lag carries a correlation of near-equal magnitude and
    the opposite sign.

    A weaker, slower component that the two series share with a 2 s offset tips the
    half-period lag to a strictly larger |r| than lag 0, so the winner is settled by the
    construction rather than by floating-point rounding between two exactly equal
    candidates.
    """
    t = np.arange(TWO_WINDOWS) / FS
    rhythm = np.sin(2 * np.pi * t / 6.0)
    shared = np.sin(2 * np.pi * t / 11.0)
    shared_offset = np.sin(2 * np.pi * (t + 2.0) / 11.0)
    a = 25.0 + 20.0 * (rhythm + 0.15 * shared)
    b = 25.0 + 20.0 * (-rhythm + 0.15 * shared_offset)

    assert windowed_xcorr(a, b, FS, use_abs=False) > 0.9  # inverted
    assert windowed_xcorr(a, b, FS, use_abs=False, max_lag_sec=0) < -0.9  # correct
    # The strength is unaffected: the two candidate lags differ only in sign.
    assert windowed_xcorr(a, b, FS, use_abs=True) > 0.9
    assert windowed_xcorr(a, b, FS, use_abs=True, max_lag_sec=0) > 0.9


def test_rejects_a_degenerate_window():
    a, b = anticorrelated_pair()
    with pytest.raises(ValueError):
        windowed_xcorr(a, b, FS, window_sec=0.01)
