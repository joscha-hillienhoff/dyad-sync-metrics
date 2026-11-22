"""Interval coding: parsing, overlap resolution, timelines, label matching and overlap
measures, on hand-built segment tables."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from dyad_sync_metrics.intervals import (
    build_timeline,
    build_timelines,
    canonical_label,
    canonicalize,
    from_segment_export,
    lagged_overlap,
    lagged_partner_share,
    parse_timestamp,
    resolve_overlaps,
    simultaneous_overlap,
    simultaneous_partner_share,
    validate_aliases,
    validate_segments,
)


def segs(*rows) -> pd.DataFrame:
    """One participant's segments from (start, end, label) tuples."""
    return pd.DataFrame(rows, columns=["start", "end", "label"])


def table(*rows) -> pd.DataFrame:
    """A segment table from (start, end, participant, label) tuples."""
    return pd.DataFrame(rows, columns=["start", "end", "participant", "label"])


def grid(*labels) -> np.ndarray:
    return np.array(labels, dtype=object)


def spans(resolved: pd.DataFrame) -> list[tuple]:
    return [(pytest.approx(r.start), pytest.approx(r.end), r.label) for r in resolved.itertuples()]


# ---------------------------------------------------------------- input


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("0:01:02,5", 62.5), ("1:00:00.25", 3600.25), ("0:00:07", 7.0), (" 0:10:00,0 ", 600.0)],
)
def test_parse_timestamp(text, seconds):
    assert parse_timestamp(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["1:2:3", "01:02", "abc", "0:00:01;5"])
def test_parse_timestamp_rejects_other_formats(text):
    with pytest.raises(ValueError, match="unrecognised timestamp"):
        parse_timestamp(text)


def test_validate_segments_parses_text_timestamps():
    out = validate_segments(table(("0:00:01,5", "0:00:03,0", "p1", "door")))
    assert out["start"].tolist() == [1.5]
    assert out["end"].tolist() == [3.0]


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        (table((0.0, 1.0, "p1", "door")).drop(columns="label"), "lacks column"),
        (table((2.0, 2.0, "p1", "door")), "do not end after they start"),
        (table((0.0, 1.0, "p1", "  ")), "without a label"),
        (table((0.0, 1.0, "p1", None)), "without a label"),
        (table((None, 1.0, "p1", "door")), "missing start"),
    ],
)
def test_validate_segments_fails_loudly(segments, message):
    with pytest.raises(ValueError, match=message):
        validate_segments(segments)


def test_from_segment_export_maps_columns_and_keeps_the_rest():
    export = pd.DataFrame(
        {
            "Begin": ["0:00:00,0", "0:00:02,5"],
            "End": ["0:00:02,5", "0:00:04,0"],
            "Track": ["p1", "p1"],
            "Code": ["table", "window"],
            "Session": ["s1", "s1"],
        }
    )
    columns = {"start": "Begin", "end": "End", "participant": "Track", "label": "Code"}
    out = from_segment_export(export, columns)
    assert out[["start", "end"]].to_numpy().tolist() == [[0.0, 2.5], [2.5, 4.0]]
    assert out["label"].tolist() == ["table", "window"]
    assert out["Session"].tolist() == ["s1", "s1"]


def test_from_segment_export_rejects_unknown_or_absent_columns():
    export = pd.DataFrame({"Begin": [0.0], "End": [1.0], "Track": ["p1"], "Code": ["door"]})
    with pytest.raises(ValueError, match="unknown name"):
        from_segment_export(export, {"onset": "Begin"})
    with pytest.raises(ValueError, match="lacks column"):
        from_segment_export(
            export, {"start": "Begin", "end": "Stop", "participant": "Track", "label": "Code"}
        )


# ---------------------------------------------------------------- overlap resolution


def test_non_overlapping_segments_are_unchanged():
    resolved, excluded = resolve_overlaps(segs((0, 2, "table"), (2, 3, "door"), (5, 6, "table")))
    assert excluded == 0.0
    assert spans(resolved) == [(0, 2, "table"), (2, 3, "door"), (5, 6, "table")]


def test_adjacent_overlap_is_removed_from_both_segments():
    resolved, excluded = resolve_overlaps(segs((0.0, 2.0, "table"), (1.5, 4.0, "door")))
    assert excluded == pytest.approx(0.5)
    assert spans(resolved) == [(0.0, 1.5, "table"), (2.0, 4.0, "door")]


def test_a_chain_of_adjacent_overlaps_excludes_each_window():
    resolved, excluded = resolve_overlaps(
        segs((0.0, 2.0, "table"), (1.5, 4.0, "door"), (3.5, 6.0, "window"))
    )
    assert excluded == pytest.approx(1.0)
    assert spans(resolved) == [(0.0, 1.5, "table"), (2.0, 3.5, "door"), (4.0, 6.0, "window")]


def test_contained_segment_wins_and_splits_the_outer_one():
    resolved, excluded = resolve_overlaps(segs((10.0, 16.0, "window"), (12.0, 14.0, "partner")))
    assert excluded == 0.0
    assert spans(resolved) == [
        (10.0, 12.0, "window"),
        (12.0, 14.0, "partner"),
        (14.0, 16.0, "window"),
    ]


@pytest.mark.parametrize(
    "rows",
    list(permutations([(10.0, 16.0, "window"), (12.0, 14.0, "partner"), (20.0, 21.0, "door")])),
)
def test_containment_is_resolved_whatever_the_row_order(rows):
    resolved, excluded = resolve_overlaps(segs(*rows))
    assert excluded == 0.0
    assert [r.label for r in resolved.itertuples()] == ["window", "partner", "window", "door"]


def test_containment_sharing_the_outer_start_leaves_one_remnant():
    resolved, _ = resolve_overlaps(segs((10.0, 16.0, "window"), (10.0, 12.0, "door")))
    assert spans(resolved) == [(10.0, 12.0, "door"), (12.0, 16.0, "window")]


def test_containment_sharing_the_outer_end_keeps_later_segments():
    resolved, excluded = resolve_overlaps(
        segs((0.0, 10.0, "table"), (6.0, 10.0, "door"), (12.0, 15.0, "window"))
    )
    assert excluded == 0.0
    assert spans(resolved) == [(0.0, 6.0, "table"), (6.0, 10.0, "door"), (12.0, 15.0, "window")]


def test_containment_nests_to_any_depth():
    resolved, excluded = resolve_overlaps(
        segs((0.0, 10.0, "table"), (2.0, 8.0, "door"), (3.0, 6.0, "window"))
    )
    assert excluded == 0.0
    assert [r.label for r in resolved.itertuples()] == ["table", "door", "window", "door", "table"]
    assert (resolved["end"] - resolved["start"]).sum() == pytest.approx(10.0)


def test_a_container_with_two_children_raises_naming_them():
    with pytest.raises(ValueError, match="more than one segment") as info:
        resolve_overlaps(segs((0.0, 10.0, "table"), (1.0, 2.0, "door"), (5.0, 6.0, "window")))
    assert "0.0-10.0 'table'" in str(info.value)
    assert "1.0-2.0 'door'" in str(info.value)
    assert "5.0-6.0 'window'" in str(info.value)


def test_a_container_whose_children_overlap_each_other_raises():
    with pytest.raises(ValueError, match="more than one segment"):
        resolve_overlaps(segs((0.0, 10.0, "table"), (1.0, 4.0, "door"), (3.0, 6.0, "window")))


def test_a_segment_overlapping_more_than_its_successor_raises():
    with pytest.raises(ValueError, match="only adjacent-boundary overlaps"):
        resolve_overlaps(segs((0.0, 5.0, "table"), (1.0, 6.0, "door"), (2.0, 7.0, "window")))


@pytest.mark.parametrize("labels", [("table", "door"), ("door", "table")])
def test_identical_spans_raise_whatever_the_row_order(labels):
    with pytest.raises(ValueError, match="identical span"):
        resolve_overlaps(segs((1.0, 3.0, labels[0]), (1.0, 3.0, labels[1])))


def test_identical_spans_nested_inside_a_container_raise():
    with pytest.raises(ValueError, match="identical span"):
        resolve_overlaps(segs((0.0, 10.0, "table"), (2.0, 4.0, "door"), (2.0, 4.0, "window")))


def test_a_span_derived_by_clipping_is_not_an_identical_span():
    # Clipping the first overlap moves the second segment's start to 10, which equals
    # the third segment's coded span; that is containment sharing the end, not a
    # coded duplicate.
    resolved, excluded = resolve_overlaps(
        segs((0.0, 10.0, "table"), (5.0, 15.0, "door"), (10.0, 15.0, "window"))
    )
    assert excluded == pytest.approx(5.0)
    assert spans(resolved) == [(0.0, 5.0, "table"), (10.0, 15.0, "window")]


def test_errors_name_the_span_as_coded_not_as_clipped():
    with pytest.raises(ValueError, match="more than one segment") as info:
        resolve_overlaps(
            segs((0.0, 10.0, "table"), (5.0, 20.0, "door"), (12.0, 15.0, "x"), (16.0, 18.0, "y"))
        )
    assert "5.0-20.0" in str(info.value)
    assert "10.0-20.0" not in str(info.value)


# ---------------------------------------------------------------- timelines and labels


def test_build_timeline_marks_bins_and_leaves_gaps_uncoded():
    out = build_timeline(
        segs((0.0, 0.2, "Table"), (0.3, 0.5, "door")), start=0.0, end=0.6, resolution_sec=0.1
    )
    assert out.tolist() == ["table", "table", None, "door", "door", None]


def test_build_timeline_clamps_segments_to_the_window():
    out = build_timeline(
        segs((-1.0, 0.2, "table"), (0.4, 9.0, "door"), (20.0, 21.0, "window")),
        start=0.0,
        end=0.5,
        resolution_sec=0.1,
    )
    assert out.tolist() == ["table", "table", None, None, "door"]


def test_build_timelines_aligns_participants_and_reports_audit_values():
    result = build_timelines(
        table(
            (0.0, 1.0, "p1", "table"),
            (0.5, 2.0, "p1", "door"),
            (1.0, 2.0, "p2", "door"),
        ),
        resolution_sec=0.5,
    )
    assert (result.start, result.end) == (0.0, 2.0)
    assert result.grids["p1"].tolist() == ["table", None, "door", "door"]
    assert result.grids["p2"].tolist() == [None, None, "door", "door"]
    assert result.excluded_sec == {"p1": pytest.approx(0.5), "p2": 0.0}
    assert result.coverage == {"p1": 0.75, "p2": 0.5}


def test_canonical_label_folds_case_and_surrounding_space_only():
    assert canonical_label("  Window ") == "window"
    assert canonical_label("WINDOW") == canonical_label("window")
    assert canonical_label("windows") != canonical_label("window")
    assert canonical_label("window-left") != canonical_label("window left")


def test_aliases_merge_exactly_the_named_pair():
    aliases = {"doorway": "door"}
    assert canonical_label("Doorway", aliases) == "door"
    assert canonical_label("door", aliases) == "door"
    assert canonical_label("table", aliases) == "table"
    out = canonicalize(grid("Doorway", None, " DOOR", "table"), aliases)
    assert out.tolist() == ["door", None, "door", "table"]


@pytest.mark.parametrize(
    ("aliases", "message"),
    [
        ({"Doorway": "door"}, "not canonical"),
        ({"doorway": "Door"}, "not canonical"),
        ({"doorway": "entrance", "entrance": "door"}, "points at another alias"),
    ],
)
def test_aliases_that_could_not_work_as_written_are_rejected(aliases, message):
    with pytest.raises(ValueError, match=message):
        validate_aliases(aliases)
    with pytest.raises(ValueError, match=message):
        canonicalize(grid("door"), aliases)


def test_case_variants_match_and_different_labels_do_not():
    a = grid("Table", "window", " door", "door")
    b = grid("table", "WINDOW", "door ", "doors")
    assert simultaneous_overlap(a, b) == pytest.approx(3 / 4)


# ---------------------------------------------------------------- overlap measures


def test_identical_and_disjoint_series():
    a = grid("table", "door", "window")
    assert simultaneous_overlap(a, a) == 1.0
    assert simultaneous_overlap(a, grid("door", "window", "table")) == 0.0


def test_uncoded_frames_leave_the_denominator_by_default():
    a = grid("table", None, "door", "door")
    b = grid("table", "table", None, "window")
    # Eligible: frames 0 and 3; match: frame 0.
    assert simultaneous_overlap(a, b) == pytest.approx(1 / 2)


def test_counting_gaps_keeps_them_in_the_denominator_but_never_as_a_match():
    a = grid("table", None, None, "door")
    b = grid("table", "table", None, "window")
    assert simultaneous_overlap(a, b, gaps="count") == pytest.approx(1 / 4)
    assert simultaneous_overlap(grid(None, None), grid(None, None), gaps="count") == 0.0


def test_without_eligible_frames_the_measure_is_undefined():
    assert np.isnan(simultaneous_overlap(grid("table", None), grid(None, "door")))
    assert np.isnan(simultaneous_overlap(grid(), grid()))


def test_unknown_gap_handling_is_rejected():
    with pytest.raises(ValueError, match="gaps must be"):
        simultaneous_overlap(grid("table"), grid("table"), gaps="drop")


def test_partner_share_is_the_mutual_gaze_part_of_the_matches():
    a = grid("partner", "partner", "table", "door")
    b = grid("Partner", "partner", "table", "window")
    assert simultaneous_overlap(a, b) == pytest.approx(3 / 4)
    assert simultaneous_partner_share(a, b) == pytest.approx(2 / 3)
    assert np.isnan(simultaneous_partner_share(grid("table"), grid("door")))


def test_excluding_partner_drops_mutual_gaze_frames_from_eligibility():
    a = grid("partner", "partner", "table", "door")
    b = grid("partner", "table", "table", "window")
    # Frame 0 drops out; frame 1 (only one on partner) stays a non-match.
    assert simultaneous_overlap(a, b, exclude_partner=True) == pytest.approx(1 / 3)
    only_mutual = grid("partner", "partner")
    assert np.isnan(simultaneous_overlap(only_mutual, only_mutual, exclude_partner=True))


def test_partner_label_is_a_parameter():
    a = grid("other person", "table")
    assert simultaneous_partner_share(a, a, partner="Other Person") == pytest.approx(1 / 2)


def test_lagged_overlap_finds_a_delay_inside_the_window():
    leader = grid("table", "door", "window", "partner", "door", "table")
    follower = grid(None, None, "table", "door", "window", "partner")
    assert simultaneous_overlap(leader, follower) == 0.0
    assert lagged_overlap(leader, follower, resolution_sec=0.5, max_lag_sec=1.0) == 1.0
    assert lagged_overlap(leader, follower, resolution_sec=0.5, max_lag_sec=0.5) == 0.0
    # Leading is directional.
    assert lagged_overlap(follower, leader, resolution_sec=0.5, max_lag_sec=1.0) == 1 / 3


def test_lagged_overlap_is_never_below_simultaneous_overlap():
    rng = np.random.default_rng(0)
    labels = np.array(["table", "door", "window", "partner", None], dtype=object)
    for _ in range(20):
        a, b = rng.choice(labels, 50), rng.choice(labels, 50)
        assert lagged_overlap(a, b, resolution_sec=0.1, max_lag_sec=0.5) >= simultaneous_overlap(
            a, b
        )


def test_lagged_partner_share_is_read_off_the_chosen_lag():
    leader = grid("partner", "door", "door")
    follower = grid("table", "partner", "door")
    assert lagged_overlap(leader, follower, resolution_sec=1.0, max_lag_sec=2.0) == 1.0
    assert lagged_partner_share(
        leader, follower, resolution_sec=1.0, max_lag_sec=2.0
    ) == pytest.approx(1 / 2)
