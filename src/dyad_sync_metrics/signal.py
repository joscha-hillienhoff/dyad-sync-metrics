"""Preprocessing of per-frame tracking signals: gap filling, low-pass filtering, speed."""

import numpy as np
from scipy.signal import butter, filtfilt

MIN_VALID_SAMPLES = 10
"""Below this many valid samples a series is returned unchanged: there is too little
to interpolate from, and the downstream measures report NaN for it anyway."""


def interpolate_gaps(x: np.ndarray) -> np.ndarray:
    """Fill NaN gaps by linear interpolation; leading and trailing gaps take the
    nearest valid value.

    Windowed cross-correlation has no NaN branch, so a gap left in place removes every
    window it touches. When a real pair is compared against pseudo-pairs built from two
    different recordings, the pseudo-pair unions the gaps of both and would lose more
    windows than the real pair. Filling gaps the same way on both paths keeps the
    baseline comparable to the value it is subtracted from.
    """
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() < MIN_VALID_SAMPLES:
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[valid], x[valid])


def lowpass(x: np.ndarray, fs: float, cutoff_hz: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter, applied after gap interpolation.

    fs is the sampling rate in Hz. The 6 Hz default suits body and head speed, which
    carry little behavioural signal above that; it is not a sensible default for gaze,
    whose fast saccadic transitions are signal rather than noise.
    """
    x = np.asarray(x, dtype=float)
    if (~np.isnan(x)).sum() < MIN_VALID_SAMPLES:
        return x
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, interpolate_gaps(x))


def speed(positions: np.ndarray, t: np.ndarray, fs: float, cutoff_hz: float = 6.0) -> np.ndarray:
    """Low-pass filtered speed (magnitude of velocity) of one tracked point.

    positions is an (n, 3) array of coordinates, t the n timestamps in seconds. The
    gradient uses the actual timestamps, so irregular frame intervals are handled; the
    filter uses the nominal sampling rate fs.
    """
    positions = np.asarray(positions, dtype=float)
    t = np.asarray(t, dtype=float)
    velocity = np.gradient(positions, t, axis=0)
    return lowpass(np.linalg.norm(velocity, axis=1), fs=fs, cutoff_hz=cutoff_hz)
