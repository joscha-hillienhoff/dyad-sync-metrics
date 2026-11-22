"""Gaze measures for two people sharing a 3-D scene.

All functions take per-frame arrays: positions and directions are (n, 3) arrays in one
world coordinate frame and metres, angles are length-n arrays in degrees. Series of
unequal length are truncated to the shortest, so a pseudo-pair built from two
different recordings runs through exactly the same code as a real pair (see
dyad_sync_metrics.surrogates).

Three measures with different questions behind them:

- mutual gaze: in how many frames do both people look at each other?
  (mutual_gaze_fraction, on the angle_to_target series of both)
- gaze coupling: does one person's looking-toward track the other's over time?
  (windowed cross-correlation of the two angle series; see gaze_coupling_series)
- gaze-ray convergence: in how many frames do the two gaze rays pass close to each
  other somewhere in space, while neither looks at the other?
  (gaze_convergence_series and the summaries built on it)

The scene-dependent thresholds (ray length, head exclusion radius) are required
arguments. They should be derived from the recorded scene: the ray length from the
largest distance at which anything in the scene can be looked at, the head exclusion
radius from the size of the head as the eye tracker's raycaster sees it. Example values
for a seated, table-sized setting are 2.5 m and 0.30 m.
"""

import numpy as np

from dyad_sync_metrics.signal import interpolate_gaps

PARALLEL_EPS = 1e-6
"""Guard on near-parallel rays. With unit directions 1 - (d1.d2)^2 is the squared sine
of the angle between them and the denominator of the closest-approach solution; below
this value the solution is numerically unusable and the parallel branch is taken."""


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- Mutual gaze


def angle_to_target(origin: np.ndarray, direction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame angle in degrees between a gaze direction and the line from the gaze
    origin to a target point, typically the partner's head.

    Directions need not be unit length. NaN coordinates propagate.
    """
    n = min(len(origin), len(direction), len(target))
    to_target = _unit(np.asarray(target[:n], dtype=float) - np.asarray(origin[:n], dtype=float))
    cos = np.clip(np.sum(to_target * _unit(direction[:n]), axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def mutual_gaze_fraction(ang_1: np.ndarray, ang_2: np.ndarray, threshold_deg: float) -> float:
    """Share of frames in which both people look at each other, i.e. both angles to the
    partner are below threshold_deg, over the frames in which both angles are valid.

    Takes the two angle series rather than the raw geometry, so that a pseudo-pair
    (angles from two different recordings) is scored by exactly this function. Returns
    NaN when no frame is valid. Common choices for threshold_deg lie around 8 to 12
    degrees; reporting several is a useful sensitivity check.
    """
    n = min(len(ang_1), len(ang_2))
    ang_1, ang_2 = np.asarray(ang_1[:n], dtype=float), np.asarray(ang_2[:n], dtype=float)
    valid = ~np.isnan(ang_1) & ~np.isnan(ang_2)
    if not valid.any():
        return np.nan
    both = (ang_1[valid] < threshold_deg) & (ang_2[valid] < threshold_deg)
    return float(both.mean())


# ---------------------------------------------------------------- Gaze coupling


def gaze_coupling_series(ang: np.ndarray) -> np.ndarray:
    """Prepare one angle-to-partner series for dyad_sync_metrics.sync.windowed_xcorr.

    Gaze coupling is the windowed cross-correlation of the two people's angle-to-partner
    series. It is a different quantity from mutual gaze, which is a co-occurrence rate
    and says nothing about whether the two series track one another.

    The only preprocessing is gap interpolation. No low-pass filter is applied: gaze is
    saccadic and its fast transitions are signal, not noise.

    Report strength and direction separately. Strength is windowed_xcorr with
    use_abs=True and a lag search; direction is windowed_xcorr with use_abs=False and
    max_lag_sec=0, because a lag search makes the sign unreliable (see windowed_xcorr).
    Correct each against a baseline computed with its own metric: pseudo-pair |r| is
    clearly positive by construction, while pseudo-pair zero-lag r averages toward zero.

    Read a negative direction with care. Alternating gaze produces it, but so do both
    people turning simultaneously toward an object that lies roughly behind one of
    them: one angle to the partner rises while the other falls purely from the seating
    geometry. The angle series cannot tell the two apart; gaze-ray convergence
    approaches the question without using the partner's head as a reference.
    """
    return interpolate_gaps(ang)


# ---------------------------------------------------------------- Gaze-ray convergence


def gaze_ray_distance(
    origin_1: np.ndarray,
    dir_1: np.ndarray,
    origin_2: np.ndarray,
    dir_2: np.ndarray,
    *,
    max_dist_m: float,
) -> np.ndarray:
    """Per-frame minimum distance between two gaze rays.

    Each gaze is a segment from the person's gaze origin to max_dist_m metres forward,
    not an infinite line, for two reasons. The closest approach must lie in front of
    both people: on infinite lines it falls behind at least one of them in many frames,
    which is impossible as shared attention yet passes any distance threshold. And it
    must lie within reach: near-parallel gazes meet far outside the scene, an artefact of
    two almost-equal directions rather than a place either person can be looking at.

    Solved as the closest-points-between-two-segments problem. With unit directions the
    unconstrained optimum is t1 = (b*e - d) / (1 - b^2) for b = d1.d2, d = d1.(o1 - o2),
    e = d2.(o1 - o2); t1 is clamped into [0, max_dist_m], t2 = e + t1*b is clamped in
    turn, and t1 is re-solved for a clamped t2. Near-parallel frames
    (1 - b^2 <= PARALLEL_EPS) start the clamping from t1 = 0, the correct limit for
    parallel segments. Directions are normalised here, so the metre scale of the cap
    does not depend on the input being unit length.

    This is pure geometry and knows nothing about who is looking at whom. In particular
    mutual gaze is a geometry in which the rays very nearly meet; use
    gaze_convergence_series to exclude it. Frames with any NaN coordinate stay NaN.
    """
    n = min(len(origin_1), len(origin_2), len(dir_1), len(dir_2))
    o1, o2 = np.asarray(origin_1[:n], dtype=float), np.asarray(origin_2[:n], dtype=float)
    d1, d2 = _unit(dir_1[:n]), _unit(dir_2[:n])

    w0 = o1 - o2
    b = np.sum(d1 * d2, axis=1)
    d = np.sum(d1 * w0, axis=1)
    e = np.sum(d2 * w0, axis=1)
    denom = 1.0 - b * b

    with np.errstate(invalid="ignore", divide="ignore"):
        unconstrained = (b * e - d) / denom
    # NaN frames take the parallel branch too and stay NaN through the arithmetic below.
    t1 = np.clip(np.where(denom > PARALLEL_EPS, unconstrained, 0.0), 0.0, max_dist_m)

    t2 = e + t1 * b
    below, above = t2 < 0.0, t2 > max_dist_m
    t2 = np.clip(t2, 0.0, max_dist_m)
    # t2 hit a bound, so re-solve t1 for it: given t2, the optimal t1 is t2*b - d.
    t1 = np.where(below, np.clip(-d, 0.0, max_dist_m), t1)
    t1 = np.where(above, np.clip(max_dist_m * b - d, 0.0, max_dist_m), t1)

    return np.linalg.norm(w0 + t1[:, None] * d1 - t2[:, None] * d2, axis=1)


def ray_point_distance(
    origin: np.ndarray, direction: np.ndarray, point: np.ndarray, *, max_dist_m: float
) -> np.ndarray:
    """Per-frame distance from a capped gaze ray to a point.

    Same ray convention as gaze_ray_distance: the parameter is clamped to
    [0, max_dist_m], so a point behind the person or beyond the cap is measured from the
    nearer end of the segment. A projection, so well-conditioned for every direction.
    NaN coordinates propagate.
    """
    n = min(len(origin), len(direction), len(point))
    d = _unit(direction[:n])
    w = np.asarray(point[:n], dtype=float) - np.asarray(origin[:n], dtype=float)
    t = np.clip(np.sum(w * d, axis=1), 0.0, max_dist_m)
    return np.linalg.norm(w - t[:, None] * d, axis=1)


def gaze_convergence_series(
    origin_1: np.ndarray,
    dir_1: np.ndarray,
    head_1: np.ndarray,
    origin_2: np.ndarray,
    dir_2: np.ndarray,
    head_2: np.ndarray,
    *,
    max_dist_m: float,
    head_exclusion_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The two per-frame series every convergence summary is computed from:
    (distance, excluded).

    distance is gaze_ray_distance. excluded marks the frames in which EITHER person's
    gaze ray passes within head_exclusion_m of the OTHER person's head point.

    Why the exclusion exists: two people looking at each other produce near-antiparallel
    rays a few centimetres apart, so without it nearly every mutual-gaze frame would
    count as convergence and the measure would re-express mutual gaze. The rule is
    behavioural: if somebody is looking at their partner, the frame is not co-attention
    on a third object, whatever the rays do elsewhere. The test is made against the
    partner's head rather than against the convergence point, because for
    near-antiparallel rays the closest-approach point is ill-conditioned and scatters
    along the sightline.

    What it guarantees: no frame survives in which a ray passes within head_exclusion_m
    of the partner's head. That is a perpendicular distance, not an angle, so the
    excluded half-angle is arcsin(head_exclusion_m / separation) and narrows as the
    people sit further apart; check it against any angular mutual-gaze threshold in use.

    Limitations, all deliberate:

    - It is broader than mutual gaze: one-sided looking at the partner excludes the
      frame too.
    - It carries no notion of gaze depth. Any object near the line between the two heads
      is excluded, at any distance from the looker, so genuine co-attention on such an
      object is not counted. This is the safe direction.
    - head_exclusion_m should be the head's size as the raycaster sees it, measured
      against the same head point that is passed in here.

    A frame whose head coordinates are missing cannot be tested, so its distance is set
    to NaN and it leaves the denominator with the other gaps.
    """
    n = min(len(origin_1), len(dir_1), len(head_1), len(origin_2), len(dir_2), len(head_2))
    origin_1, dir_1, head_1 = origin_1[:n], dir_1[:n], head_1[:n]
    origin_2, dir_2, head_2 = origin_2[:n], dir_2[:n], head_2[:n]

    dist = gaze_ray_distance(origin_1, dir_1, origin_2, dir_2, max_dist_m=max_dist_m)
    to_head_2 = ray_point_distance(origin_1, dir_1, head_2, max_dist_m=max_dist_m)
    to_head_1 = ray_point_distance(origin_2, dir_2, head_1, max_dist_m=max_dist_m)

    untestable = np.isnan(to_head_2) | np.isnan(to_head_1)
    dist = np.where(untestable, np.nan, dist)
    excluded = (to_head_2 <= head_exclusion_m) | (to_head_1 <= head_exclusion_m)
    return dist, excluded


def eligible_frames(dist: np.ndarray, excluded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two frame masks every convergence summary is taken over: (valid, eligible).

    valid is every frame with a computable distance; eligible narrows it to the frames
    the head exclusion left. Defining both once keeps the rate, median and excluded
    share consistent, so that rate * (1 - excluded_share) is exactly the whole-session
    rate with excluded frames counted as misses.
    """
    valid = ~np.isnan(dist)
    return valid, valid & ~np.asarray(excluded, dtype=bool)


def gaze_convergence_rate(dist: np.ndarray, excluded: np.ndarray, threshold_m: float) -> float:
    """Share of the eligible frames in which the two rays pass within threshold_m.

    Conditional on the eligible frames: an excluded frame is not an observed failure to
    share attention but a frame the measure cannot speak about, so it leaves numerator
    and denominator. Counting it as a miss would scale each pair's rate by its own
    excluded share, and that share is not comparable between a real dyad and its
    pseudo-pairs: in a real dyad the two people's looks at each other coincide in time
    to the extent they engage in mutual gaze, in a pseudo-pair they are independent.
    That would bring mutual gaze back into a measure built to be orthogonal to it.

    Returns NaN when no frame is eligible.
    """
    _, eligible = eligible_frames(dist, excluded)
    if not eligible.any():
        return np.nan
    return float((dist[eligible] < threshold_m).mean())


def gaze_convergence_median(dist: np.ndarray, excluded: np.ndarray) -> float:
    """Median ray distance over the eligible frames; NaN when there are none."""
    _, eligible = eligible_frames(dist, excluded)
    return float(np.median(dist[eligible])) if eligible.any() else np.nan


def gaze_convergence_excluded_share(dist: np.ndarray, excluded: np.ndarray) -> float:
    """Share of the valid frames that the head exclusion ruled out.

    An audit value: without it a low rate cannot be told apart from a heavily excluded
    recording. Its denominator is deliberately the wider valid-frame set; over the
    eligible frames it would be identically zero. Returns NaN when no frame is valid.
    """
    valid, _ = eligible_frames(dist, excluded)
    return float(np.asarray(excluded, dtype=bool)[valid].mean()) if valid.any() else np.nan
