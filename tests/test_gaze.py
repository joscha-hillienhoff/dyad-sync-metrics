"""Gaze measures on hand-built geometry whose answer is known analytically."""

import numpy as np
import pytest

from dyad_sync_metrics.gaze import (
    PARALLEL_EPS,
    angle_to_target,
    gaze_convergence_excluded_share,
    gaze_convergence_median,
    gaze_convergence_rate,
    gaze_convergence_series,
    gaze_coupling_series,
    gaze_ray_distance,
    mutual_gaze_fraction,
    ray_point_distance,
)

MAX_DIST_M = 2.5
HEAD_EXCLUSION_M = 0.30
THRESHOLDS_M = (0.02, 0.05, 0.10, 0.20)


def ray_distance(o1, d1, o2, d2, max_dist_m=MAX_DIST_M):
    return gaze_ray_distance(o1, d1, o2, d2, max_dist_m=max_dist_m)


def point_distance(origin, direction, point):
    return ray_point_distance(origin, direction, point, max_dist_m=MAX_DIST_M)


def convergence(o1, d1, h1, o2, d2, h2):
    return gaze_convergence_series(
        o1, d1, h1, o2, d2, h2, max_dist_m=MAX_DIST_M, head_exclusion_m=HEAD_EXCLUSION_M
    )


def unexcluded(dist):
    return np.zeros(len(dist), dtype=bool)


def rays(*frames):
    """The four (n, 3) arrays from a list of (o1, d1, o2, d2) tuples."""
    return tuple(np.array([f[i] for f in frames], dtype=float) for i in range(4))


def head_frame(o1, d1, h1, o2, d2, h2):
    """The six (1, 3) arrays of a single frame for gaze_convergence_series."""
    return tuple(np.array([v], dtype=float) for v in (o1, d1, h1, o2, d2, h2))


# ---------------------------------------------------------------- Angle and mutual gaze


def test_angle_to_target_is_zero_when_looking_straight_at_it():
    origin = np.array([[0.0, 0, 0], [0.0, 0, 0]])
    direction = np.array([[0.0, 0, 2.0], [1.0, 0, 0]])
    target = np.array([[0.0, 0, 1.5], [0.0, 0, 1.5]])
    assert angle_to_target(origin, direction, target) == pytest.approx([0.0, 90.0])


def test_mutual_gaze_counts_frames_in_which_both_are_under_the_threshold():
    ang_1 = np.array([2.0, 2.0, 30.0, 2.0])
    ang_2 = np.array([3.0, 40.0, 3.0, 9.0])
    assert mutual_gaze_fraction(ang_1, ang_2, threshold_deg=10) == pytest.approx(0.5)
    assert mutual_gaze_fraction(ang_1, ang_2, threshold_deg=5) == pytest.approx(0.25)


def test_mutual_gaze_ignores_nan_frames():
    ang_1 = np.array([0.0, 0.0, np.nan, 0.0])
    ang_2 = np.array([0.0, 90.0, 0.0, np.nan])
    assert mutual_gaze_fraction(ang_1, ang_2, 10) == pytest.approx(0.5)
    assert np.isnan(mutual_gaze_fraction(np.array([np.nan]), np.array([0.0]), 10))


def test_mutual_gaze_truncates_to_the_shorter_series():
    long = np.concatenate([np.zeros(20), np.full(20, 90.0)])
    short = np.zeros(20)
    assert mutual_gaze_fraction(long, short, 10) == 1.0
    assert mutual_gaze_fraction(short, long, 10) == 1.0


def test_gaze_coupling_series_fills_gaps_but_does_not_filter():
    ang = np.array([np.nan, 10, np.nan, 20, 30, np.nan, 40, 50, 60, 70, 80, 90, 100, 110, np.nan])
    out = gaze_coupling_series(ang)
    valid = ~np.isnan(ang)
    assert not np.isnan(out).any()
    assert out[valid] == pytest.approx(ang[valid])  # a filter would move every sample


# ---------------------------------------------------------------- Ray-to-ray distance


def test_skew_rays_give_their_perpendicular_separation():
    assert ray_distance(*rays(([0, 0, 0], [0, 0, 1], [0, 0.5, 2], [1, 0, 0]))) == (
        pytest.approx([0.5])
    )


def test_intersecting_rays_give_zero():
    dist = ray_distance(*rays(([0, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0])))
    assert dist == pytest.approx([0.0], abs=1e-12)


def test_parallel_rays_fall_back_to_the_parallel_branch():
    dist = ray_distance(*rays(([0, 0, 0], [0, 0, 1], [0.3, 0, 1.5], [0, 0, 1])))
    assert dist == pytest.approx([0.3])


def test_antiparallel_rays_also_fall_back():
    dist = ray_distance(*rays(([0, 0, 0], [0, 0, 1], [0.1, 0, 2], [0, 0, -1])))
    assert dist == pytest.approx([0.1])


def test_near_parallel_rays_use_the_fallback_and_stay_finite():
    tiny = np.sqrt(PARALLEL_EPS) / 10
    d2 = np.array([tiny, 0.0, 1.0])
    d2 /= np.linalg.norm(d2)
    assert 1.0 - np.dot([0, 0, 1], d2) ** 2 < PARALLEL_EPS
    dist = ray_distance(*rays(([0, 0, 0], [0, 0, 1], [0.4, 0, 1.5], list(d2))))
    assert np.isfinite(dist).all()
    assert dist == pytest.approx([0.4], abs=1e-4)


def test_result_is_invariant_to_direction_magnitude():
    """Directions are normalised inside, so a non-unit direction cannot rescale the cap."""
    base = ray_distance(*rays(([0, 0, 0], [0, 0, 1], [0, 0.5, 2], [1, 0, 0])))
    scaled = ray_distance(*rays(([0, 0, 0], [0, 0, 3], [0, 0.5, 2], [7, 0, 0])))
    assert scaled == pytest.approx(base)


def test_nan_frames_stay_nan_and_do_not_contaminate_others():
    dist = ray_distance(
        *rays(
            ([0, 0, 0], [0, 0, 1], [0, 0.5, 2], [1, 0, 0]),
            ([0, 0, 0], [0, 0, 1], [0, np.nan, 2], [1, 0, 0]),
            ([0, 0, 0], [0, 0, 1], [0, 0.25, 2], [1, 0, 0]),
        )
    )
    assert dist[0] == pytest.approx(0.5)
    assert np.isnan(dist[1])
    assert dist[2] == pytest.approx(0.25)


def test_many_frames_are_computed_elementwise():
    n = 50
    offsets = np.linspace(0.0, 1.0, n)
    o1 = np.zeros((n, 3))
    d1 = np.tile([0.0, 0.0, 1.0], (n, 1))
    o2 = np.stack([np.zeros(n), offsets, np.full(n, 2.0)], axis=1)
    d2 = np.tile([1.0, 0.0, 0.0], (n, 1))
    assert ray_distance(o1, d1, o2, d2) == pytest.approx(offsets)


def test_closest_approach_behind_a_person_does_not_count():
    """The lines meet 2 m behind person 1, which is impossible as shared attention."""
    behind = ray_distance(*rays(([0, 0, 0], [0, 0, -1], [0, 0, 2], [1, 0, 0])))
    assert behind == pytest.approx([2.0])
    ahead = ray_distance(*rays(([0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0])))
    assert ahead == pytest.approx([0.0], abs=1e-12)


def test_closest_approach_behind_the_partner_does_not_count_either():
    """As infinite lines these pass within 0.5 m, 1 m behind person 2's own gaze; on the
    true rays the nearest person 2 gets is its own origin."""
    dist = ray_distance(*rays(([0, 0, 0], [1, 0, 0], [2, 0.5, 1], [0, 0, 1])))
    assert dist == pytest.approx([np.hypot(0.5, 1.0)])
    ahead = ray_distance(*rays(([0, 0, 0], [1, 0, 0], [2, 0.5, 1], [0, 0, -1])))
    assert ahead == pytest.approx([0.5])


def test_convergence_beyond_the_cap_does_not_count():
    far = MAX_DIST_M * 4
    frame = ([0, 0, 0], [0, 0, 1], [1.0, 0, far], [0, 0, -1])
    assert ray_distance(*rays(frame)) > 1.0
    assert ray_distance(*rays(frame), max_dist_m=far) == pytest.approx([1.0])


def test_matches_a_brute_force_search_over_the_capped_rays():
    rng = np.random.default_rng(11)
    grid = np.linspace(0.0, MAX_DIST_M, 401)
    for _ in range(40):
        o1, o2 = rng.normal(size=3), rng.normal(size=3)
        d1, d2 = rng.normal(size=3), rng.normal(size=3)
        d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
        pts1 = o1 + grid[:, None] * d1
        pts2 = o2 + grid[:, None] * d2
        brute = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2).min()
        got = ray_distance(*rays((o1, d1, o2, d2)))[0]
        assert got <= brute + 1e-9
        assert got == pytest.approx(brute, abs=2e-2)


def test_unequal_length_rays_truncate_to_the_shorter():
    o1, d1, o2, d2 = rays(
        ([0, 0, 0], [0, 0, 1], [0, 0.5, 2], [1, 0, 0]),
        ([0, 0, 0], [0, 0, 1], [0, 0.25, 2], [1, 0, 0]),
    )
    assert ray_distance(o1[:1], d1[:1], o2, d2) == pytest.approx([0.5])
    assert ray_distance(o1, d1, o2[:1], d2[:1]) == pytest.approx([0.5])


# ---------------------------------------------------------------- Head exclusion


def test_mutual_gaze_is_excluded_even_though_the_rays_nearly_meet():
    dist, excluded = convergence(
        *head_frame([0, 0, 0], [0, 0, 1], [0, 0, -0.03], [0.04, 0, 2], [0, 0, -1], [0.04, 0, 2.03])
    )
    assert dist == pytest.approx([0.04])
    assert bool(excluded[0])
    for thr in THRESHOLDS_M:
        assert np.isnan(gaze_convergence_rate(dist, excluded, thr))


def test_co_attention_on_a_third_object_is_not_excluded():
    dist, excluded = convergence(
        *head_frame([0, 0, 0], [1, 0, 1], [0, 0, -0.03], [0, 0, 2], [1, 0, -1], [0, 0, 2.03])
    )
    assert dist == pytest.approx([0.0], abs=1e-12)
    assert not bool(excluded[0])
    for thr in THRESHOLDS_M:
        assert gaze_convergence_rate(dist, excluded, thr) == pytest.approx(1.0)


def test_an_object_near_the_head_to_head_line_is_excluded_at_any_depth():
    """The documented angular limitation, pinned as behaviour: both attend an object
    1 m in front of person 1 and 0.1 m off the head-to-head line, yet person 1's ray
    continues past it within the exclusion radius of person 2's head."""
    obj = np.array([0.10, 0.0, 1.0])
    o1 = h1 = np.array([[0.0, 0, 0]])
    o2 = h2 = np.array([[0.0, 0, 2.0]])
    d1 = np.array([obj / np.linalg.norm(obj)])
    d2 = np.array([(obj - o2[0]) / np.linalg.norm(obj - o2[0])])
    dist, excluded = convergence(o1, d1, h1, o2, d2, h2)
    assert dist == pytest.approx([0.0], abs=1e-9)
    assert point_distance(o1, d1, h2) < HEAD_EXCLUSION_M
    assert bool(excluded[0])


def test_either_person_looking_at_the_other_is_enough_to_exclude():
    _, excluded = convergence(
        *head_frame([0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 0, 2], [0, 0, -1], [0, 0, 2])
    )
    assert bool(excluded[0])


def test_the_exclusion_radius_is_head_exclusion_m():
    for offset, expected in ((HEAD_EXCLUSION_M * 0.9, True), (HEAD_EXCLUSION_M * 1.1, False)):
        _, excluded = convergence(
            *head_frame([0, 0, 0], [0, 0, 1], [0, 0, 0], [offset, 0, 2], [1, 0, 0], [offset, 0, 2])
        )
        assert bool(excluded[0]) is expected


def test_the_exclusion_uses_the_capped_ray_not_an_infinite_line():
    origin, forward = np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 1]])
    behind = point_distance(origin, forward, np.array([[0.2, 0, -3.0]]))
    assert behind == pytest.approx([np.hypot(0.2, 3.0)])
    beyond = point_distance(origin, forward, np.array([[0.2, 0, MAX_DIST_M + 2.0]]))
    assert beyond == pytest.approx([np.hypot(0.2, 2.0)])


def test_a_frame_without_head_coordinates_leaves_the_denominator():
    dist, excluded = convergence(
        *head_frame([0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 0, 2], [1, 0, -1], [np.nan, 0, 2])
    )
    assert np.isnan(dist[0])
    assert np.isnan(gaze_convergence_rate(dist, excluded, 0.2))


def test_nan_ray_frames_are_unchanged_by_the_exclusion():
    o1 = h1 = np.zeros((3, 3))
    d1 = np.tile([1.0, 0, 1], (3, 1))
    o2 = np.array([[0.0, 0, 2], [np.nan, 0, 2], [0.0, 0.25, 2]])
    d2 = np.tile([1.0, 0, -1], (3, 1))
    h2 = np.array([[0.0, 0, 2], [0.0, 0, 2], [0.0, 0.25, 2]])
    dist, excluded = convergence(o1, d1, h1, o2, d2, h2)
    assert dist[0] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(dist[1])
    assert dist[2] == pytest.approx(0.25)
    assert not excluded.any()
    assert gaze_convergence_rate(dist, excluded, 0.1) == pytest.approx(0.5)


def test_series_truncates_every_input_to_the_shortest():
    n = 5
    o1 = np.zeros((n, 3))
    d1 = np.tile([0.0, 0.0, 1.0], (n, 1))
    h1 = np.tile([0.0, 1.0, 0.0], (n, 1))
    o2 = np.stack([np.full(n, 0.5), np.zeros(n), np.full(n, 2.0)], axis=1)
    d2 = np.tile([0.0, 0.0, -1.0], (n, 1))
    h2 = np.tile([0.5, 1.0, 2.0], (n, 1))
    dist, excluded = convergence(o1, d1, h1[:2], o2, d2, h2)
    assert len(dist) == len(excluded) == 2


# ---------------------------------------------------------------- Summaries


def test_rate_is_the_share_of_frames_below_the_threshold():
    dist = np.array([0.01, 0.03, 0.07, 0.5])
    expected = {0.02: 0.25, 0.05: 0.5, 0.10: 0.75, 0.20: 0.75}
    for thr, share in expected.items():
        assert gaze_convergence_rate(dist, unexcluded(dist), thr) == pytest.approx(share)


def test_rate_ignores_nan_frames_rather_than_counting_them_as_misses():
    dist = np.array([0.01, np.nan, np.nan, 0.5])
    assert gaze_convergence_rate(dist, unexcluded(dist), 0.02) == pytest.approx(0.5)


def test_an_excluded_frame_leaves_the_rate_rather_than_counting_as_a_miss():
    dist = np.array([0.01, 0.01, 0.5, 0.5])
    excluded = np.array([False, True, False, False])
    assert gaze_convergence_rate(dist, excluded, 0.02) == pytest.approx(1 / 3)


def test_rate_is_nan_when_no_frame_is_eligible():
    assert np.isnan(gaze_convergence_rate(np.array([0.01, 0.5]), np.ones(2, dtype=bool), 0.02))


def test_median_distance_leaves_excluded_and_nan_frames_out():
    dist = np.array([0.1, 0.2, 0.3, np.nan])
    excluded = np.array([False, False, True, False])
    assert gaze_convergence_median(dist, excluded) == pytest.approx(0.15)
    assert np.isnan(gaze_convergence_median(np.array([0.1]), np.array([True])))


def test_excluded_share_is_the_share_of_valid_frames_ruled_out():
    dist = np.array([0.1, np.nan, 0.3, 0.4])
    excluded = np.array([True, False, True, False])
    assert gaze_convergence_excluded_share(dist, excluded) == pytest.approx(2 / 3)
    assert np.isnan(gaze_convergence_excluded_share(np.array([np.nan]), np.array([False])))


def test_conditional_rate_and_share_recover_the_whole_session_rate():
    """rate * (1 - excluded_share) is the rate with excluded frames counted as misses."""
    dist = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, np.nan, np.nan])
    excluded = np.array([True, True, True, True, False, False, False, False, False, True])
    share = gaze_convergence_excluded_share(dist, excluded)
    rate = gaze_convergence_rate(dist, excluded, 0.02)
    assert share == pytest.approx(0.5)
    assert rate == pytest.approx(0.25)
    valid = ~np.isnan(dist)
    assert rate * (1 - share) == pytest.approx(((dist < 0.02) & ~excluded)[valid].mean())
