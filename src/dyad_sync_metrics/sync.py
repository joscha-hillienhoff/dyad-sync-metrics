"""Windowed cross-correlation synchrony between two continuous signals."""

import numpy as np

R_CLIP = 0.999
"""Correlations are clipped to +/- R_CLIP before the Fisher z-transform, which is
infinite at +/- 1."""


def fisher_z_mean(rs) -> float:
    """Average correlations in Fisher z space and transform back.

    Returns NaN for an empty input.
    """
    rs = np.asarray(rs, dtype=float)
    if rs.size == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -R_CLIP, R_CLIP)))))


def windowed_xcorr(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: float,
    *,
    use_abs: bool = True,
    window_sec: float = 30.0,
    overlap: float = 0.5,
    max_lag_sec: float = 5.0,
) -> float:
    """Windowed lagged cross-correlation synchrony of two signals.

    The two series are cut into windows of window_sec seconds overlapping by the
    fraction overlap. Within each window both are z-scored, the lag within
    +/- max_lag_sec with the largest |r| is selected, and the per-window values are
    Fisher-z averaged and transformed back. Windows in which either signal is constant
    are skipped. Series of unequal length are truncated to the shorter one, so a
    pseudo-pair built from two recordings of different length runs through exactly
    this code. Returns NaN when no window can be formed.

    The window, overlap and lag defaults are common choices for body-movement
    synchrony, not tuned values; other channels may need shorter ones.

    use_abs=True (the default): the absolute value |r| at the best lag is averaged.
    For non-negative signals such as speed magnitudes a negative correlation does not
    mean anti-phase oscillation the way it would for a position signal; it means the
    partners move reciprocally, one active while the other is still. That is
    coordination just as much as moving together is, so clipping negative values to
    zero would misrepresent the signal, and |r| is the established treatment for this
    signal class.

    use_abs=False: the signed r at the lag of maximum |r| is averaged. That keeps the
    direction of the relationship, at the price that simultaneous and reciprocal
    windows partially cancel each other out.

    THE SIGN RETURNED UNDER use_abs=False IS NOT RELIABLE WHEN max_lag_sec > 0. The lag
    is selected by maximum |r| and the sign is then read off at that lag, so for any
    rhythm whose half-period falls inside the search range the two lags +/- T/2 apart
    carry correlations of near-equal magnitude and opposite sign, and which one wins is
    decided by small perturbations rather than by the coupling itself. A truly
    alternating pair can therefore report a positive value, and a simultaneous pair a
    negative one; on synthetic series both inversions occur. Narrowing max_lag_sec makes
    it rarer, not safe. |r| is unaffected, since the two candidate lags have the same
    magnitude.

    Where the direction of a relationship is the quantity of interest, call this
    function with max_lag_sec=0: the search collapses to the zero-lag correlation, the
    sign is then the sign of the data, and the Fisher-z averaging across windows is
    unchanged.
    """
    win = int(window_sec * fs)
    step = int(win * (1 - overlap))
    max_lag = int(max_lag_sec * fs)
    n = min(len(sig_a), len(sig_b))
    if win < 2 or step < 1:
        raise ValueError(
            "window_sec and overlap must give a window of at least 2 samples "
            "and a step of at least 1 sample"
        )

    rs = []
    for start in range(0, n - win, step):
        a = np.asarray(sig_a[start : start + win], dtype=float)
        b = np.asarray(sig_b[start : start + win], dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        best_r = 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                r = np.corrcoef(a[-lag:], b[: len(b) + lag])[0, 1]
            elif lag > 0:
                r = np.corrcoef(a[: len(a) - lag], b[lag:])[0, 1]
            else:
                r = np.corrcoef(a, b)[0, 1]
            if not np.isnan(r) and abs(r) > abs(best_r):
                best_r = r
        rs.append(abs(best_r) if use_abs else best_r)

    return fisher_z_mean(rs)
