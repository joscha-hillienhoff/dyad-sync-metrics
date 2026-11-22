"""Co-attention from interval coding.

Input is a table of coded segments, one row per segment, with the columns start and
end (seconds), participant and label: what that participant was looking at from start
to end. A coder working from video produces such a table per recording;
from_segment_export maps the columns of an existing export onto these names.

The pipeline has three steps, each with its own function:

1. resolve_overlaps makes one participant's segments mutually exclusive. Coding tools
   let segments of the same channel overlap, and an overlap leaves open which code
   applies. Two shapes are resolved: an adjacent-boundary overlap (the later segment
   starts before the earlier one ends) is ambiguous, so the contested window is removed
   from both segments and becomes uncoded time; a fully contained segment is not
   ambiguous, so the inner one wins for its duration and the outer one is split around
   it. Every other shape raises instead of being guessed at.
2. build_timelines puts each participant's resolved segments on a regular time grid.
   Labels are canonicalised on the way (casefolded and stripped, plus an optional alias
   map), and uncoded time is None.
3. simultaneous_overlap and lagged_overlap compare two grids: the share of frames in
   which both participants carry the same label, at the same instant or with the best
   lag inside a window.

Uncoded time means the coder could not tell where the participant looked, not that
they looked at nothing. The headline gap handling (gaps="exclude") therefore drops
frames in which either participant is uncoded from the denominator. The alternative
(gaps="count") keeps them in the denominator as non-matches; a frame in which both are
uncoded is never a match, since shared ignorance is not shared attention.

A frame in which both participants look at each other is a match under the label
given as partner. That is mutual gaze rather than attention on a third thing, so it
can be reported separately (the partner_share functions) or dropped from eligibility
(exclude_partner=True).

All measures take (a, b) and return a float, so they plug directly into
dyad_sync_metrics.surrogates.circular_shift_correction for a chance baseline.
"""

import re
from collections.abc import Hashable, Mapping
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

COLUMNS = ("start", "end", "participant", "label")
"""Columns of a segment table."""

Gaps = Literal["exclude", "count"]

_TIMESTAMP = re.compile(r"^(\d+):(\d{2}):(\d{2})(?:[.,](\d+))?$")


def parse_timestamp(text: str) -> float:
    """ "h:mm:ss", "h:mm:ss.d" or "h:mm:ss,d" (decimal point or comma, any number of
    fractional digits) -> seconds."""
    match = _TIMESTAMP.match(text.strip())
    if not match:
        raise ValueError(f"unrecognised timestamp {text!r}; expected h:mm:ss[.f]")
    hours, minutes, seconds, fraction = match.groups()
    value = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if fraction:
        value += int(fraction) / 10 ** len(fraction)
    return value


def _seconds(values: pd.Series) -> pd.Series:
    """A start or end column as float seconds; strings are parsed as timestamps."""
    return values.map(lambda v: parse_timestamp(v) if isinstance(v, str) else float(v))


def validate_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """A checked copy of a segment table, with start and end as float seconds.

    Raises when a column of COLUMNS is missing, a start or end is missing, a segment
    does not end after it starts, or a label is not a non-blank string. A blank label
    names no target: it would enter the denominator without ever being able to match.
    """
    missing = [c for c in COLUMNS if c not in segments.columns]
    if missing:
        raise ValueError(f"segment table lacks column(s) {missing}")
    out = segments.copy()
    for column in ("start", "end"):
        if out[column].isna().any():
            raise ValueError(f"segment table has missing {column} values")
        out[column] = _seconds(out[column])
    bad = out[out["end"] <= out["start"]]
    if len(bad):
        first = bad.iloc[0]
        raise ValueError(
            f"{len(bad)} segment(s) do not end after they start, first: "
            f"{first['participant']!r} {first['start']}-{first['end']} s"
        )
    blank = out[~out["label"].map(lambda v: isinstance(v, str) and bool(v.strip()))]
    if len(blank):
        first = blank.iloc[0]
        raise ValueError(
            f"{len(blank)} segment(s) without a label, first: "
            f"{first['participant']!r} {first['start']}-{first['end']} s"
        )
    return out


def from_segment_export(export: pd.DataFrame, columns: Mapping[str, str]) -> pd.DataFrame:
    """A segment table from an export with other column names.

    columns maps each name in COLUMNS to the export's column, for example
    {"start": "Begin", "end": "End", "participant": "Coder track", "label": "Code"}.
    Other export columns are kept unchanged. Timestamps given as text are parsed with
    parse_timestamp; the result is checked with validate_segments.
    """
    unknown = set(columns) - set(COLUMNS)
    if unknown:
        raise ValueError(f"columns maps unknown name(s) {sorted(unknown)}; use {COLUMNS}")
    absent = [source for source in columns.values() if source not in export.columns]
    if absent:
        raise ValueError(f"export lacks column(s) {absent}")
    return validate_segments(export.rename(columns={v: k for k, v in columns.items()}))


# ---------------------------------------------------------------- overlap resolution


def resolve_overlaps(segments: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """One participant's segments (columns start, end, label) made mutually exclusive.

    Returns (resolved segments, seconds excluded). Two overlap shapes are resolved:

    - Adjacent boundary: the later segment starts before the earlier one ends and ends
      after it. The window between the later start and the earlier end is ambiguous, so
      it is removed from both segments and becomes uncoded time; its length is added to
      the excluded seconds.
    - Full containment: one segment lies entirely inside another. The inner segment
      wins for its duration and the outer one is split into the parts before and after
      it, so no time is excluded and no zero-length remnant is kept. Containment may
      nest to any depth.

    Every other shape raises ValueError rather than being resolved by a rule nobody
    chose: a container holding more than one segment, a segment overlapping more than
    its immediate successor or reaching past the end of its container, and two
    segments with identical spans (which code survived would depend on row order).
    Checks run on the spans as coded, and error messages name those spans.
    """
    ordered = segments.sort_values(["start", "end"], ascending=[True, False])
    items = ordered[["start", "end", "label"]].to_dict("records")
    _require_no_identical_spans(items)
    resolved, excluded = _resolve_run(items)
    out = pd.DataFrame(resolved, columns=["start", "end", "label"])
    return out[out["end"] > out["start"]].reset_index(drop=True), excluded


def _span(item: dict) -> str:
    return f"{item.get('coded_start', item['start'])}-{item['end']} {item['label']!r}"


def _require_no_identical_spans(items: list[dict]) -> None:
    """Runs on the sorted, unresolved segments: a span derived by clipping during
    resolution may coincide with a coded one without being a duplicate."""
    for cur, nxt in zip(items, items[1:], strict=False):
        if cur["start"] == nxt["start"] and cur["end"] == nxt["end"]:
            raise ValueError(
                f"two segments with the identical span {_span(cur)} and {_span(nxt)}: "
                "which code applies is undetermined"
            )


def _resolve_run(items: list[dict]) -> tuple[list[dict], float]:
    """Recursive core of resolve_overlaps; items sorted by start, then by end descending,
    so a container precedes the segments it holds."""
    resolved: list[dict] = []
    excluded = 0.0
    i = 0
    while i < len(items):
        cur = items[i]
        if i == len(items) - 1 or cur["end"] <= items[i + 1]["start"]:
            if cur["end"] > cur["start"]:
                resolved.append(cur)
            i += 1
            continue
        nxt = items[i + 1]
        if cur["end"] >= nxt["end"]:
            # cur contains nxt; run is everything starting before cur ends.
            j = i + 1
            while j < len(items) and items[j]["start"] < cur["end"]:
                j += 1
            run = items[i + 1 : j]
            leak = next((it for it in run if it["end"] > cur["end"]), None)
            if leak is not None:
                raise ValueError(
                    f"segment {_span(cur)} overlaps {_span(nxt)} and {_span(leak)}; only "
                    "adjacent-boundary overlaps and full containment are resolved"
                )
            # Anything in the run not nested inside nxt is a second child of cur.
            siblings = [
                it for it in run[1:] if not (it["start"] < nxt["end"] and it["end"] <= nxt["end"])
            ]
            if siblings:
                held = ", ".join(_span(it) for it in [nxt, *siblings])
                raise ValueError(
                    f"segment {_span(cur)} contains more than one segment ({held}); "
                    "containment is resolved for a single nested segment only"
                )
            inner, inner_excluded = _resolve_run(run)
            excluded += inner_excluded
            if cur["start"] < nxt["start"]:
                resolved.append({"start": cur["start"], "end": nxt["start"], "label": cur["label"]})
            resolved.extend(inner)
            rest = items[j:]
            if nxt["end"] < cur["end"]:
                coded = cur.get("coded_start", cur["start"])
                rest = [{**cur, "start": nxt["end"], "coded_start": coded}, *rest]
            more, more_excluded = _resolve_run(rest)
            return resolved + more, excluded + more_excluded
        # Adjacent-boundary overlap.
        if i + 2 < len(items) and cur["end"] > items[i + 2]["start"]:
            raise ValueError(
                f"segment {_span(cur)} overlaps {_span(nxt)} and {_span(items[i + 2])}; "
                "only adjacent-boundary overlaps and full containment are resolved"
            )
        excluded += cur["end"] - nxt["start"]
        if nxt["start"] > cur["start"]:
            resolved.append({"start": cur["start"], "end": nxt["start"], "label": cur["label"]})
        pushed = {**nxt, "start": cur["end"], "coded_start": nxt.get("coded_start", nxt["start"])}
        items = [*items[: i + 1], pushed, *items[i + 2 :]]
        i += 1
    return resolved, excluded


# ---------------------------------------------------------------- labels and timelines


def validate_aliases(aliases: Mapping[str, str]) -> None:
    """Rejects an alias map that would silently do less than it says.

    Aliases are looked up after casefolding and stripping, so a key or value not in that
    form could never match, and a value that is itself a key would resolve only one of
    two steps. Both raise.
    """
    for alias, target in aliases.items():
        for entry in (alias, target):
            if entry != entry.strip().casefold():
                raise ValueError(
                    f"alias entry {entry!r} is not canonical; write it as "
                    f"{entry.strip().casefold()!r}"
                )
        if target in aliases:
            raise ValueError(
                f"alias {alias!r} -> {target!r} points at another alias; lookup does not "
                "chain, so map both to the final label"
            )


def canonical_label(label: str, aliases: Mapping[str, str] | None = None) -> str:
    """The form two labels are compared in: casefolded, stripped of surrounding
    whitespace, then mapped through aliases. Nothing else is merged: no punctuation
    stripping, transliteration or fuzzy matching."""
    canon = label.strip().casefold()
    return aliases.get(canon, canon) if aliases else canon


def canonicalize(grid: np.ndarray, aliases: Mapping[str, str] | None = None) -> np.ndarray:
    """canonical_label over a grid, leaving uncoded frames (None) as they are.
    Idempotent when aliases is empty or already applied."""
    if aliases:
        validate_aliases(aliases)
    grid = np.asarray(grid, dtype=object)
    lookup = {v: canonical_label(v, aliases) for v in set(grid.tolist()) if v is not None}
    return np.array([None if v is None else lookup[v] for v in grid], dtype=object)


def _bin_span(start: float, end: float, origin: float, resolution_sec: float, n_bins: int):
    """Half-open bin range [first, last) covered by a segment, clamped to the grid at
    both ends; a segment wholly outside the grid covers no bins."""
    first = min(max(int(round((start - origin) / resolution_sec)), 0), n_bins)
    last = min(max(int(round((end - origin) / resolution_sec)), first), n_bins)
    return first, last


def build_timeline(
    resolved: pd.DataFrame,
    *,
    start: float,
    end: float,
    resolution_sec: float,
    aliases: Mapping[str, str] | None = None,
) -> np.ndarray:
    """One participant's resolved segments on a grid of resolution_sec bins from start
    to end: an object array of canonical labels, with None where nothing was coded.

    The segments must already be mutually exclusive (resolve_overlaps); a later
    segment would otherwise overwrite an earlier one without notice.
    """
    n_bins = int(round((end - start) / resolution_sec))
    grid = np.full(n_bins, None, dtype=object)
    for row in resolved.itertuples():
        first, last = _bin_span(row.start, row.end, start, resolution_sec, n_bins)
        grid[first:last] = row.label
    return canonicalize(grid, aliases)


class Timelines(NamedTuple):
    grids: dict[Hashable, np.ndarray]
    """Per participant, the canonical label grid (None where uncoded)."""
    excluded_sec: dict[Hashable, float]
    """Per participant, seconds removed by resolve_overlaps as ambiguous."""
    coverage: dict[Hashable, float]
    """Per participant, share of grid frames that carry a label."""
    start: float
    end: float
    resolution_sec: float


def build_timelines(
    segments: pd.DataFrame,
    *,
    resolution_sec: float = 0.1,
    aliases: Mapping[str, str] | None = None,
    window: tuple[float, float] | None = None,
) -> Timelines:
    """Validated, overlap-resolved, canonicalised grids for every participant in a
    segment table of one recording.

    All grids share one window, by default from the earliest start to the latest end
    over all segments, so they are aligned frame by frame. resolution_sec should match
    the precision of the coding; a finer grid adds no information. aliases maps
    canonical labels onto other canonical labels that name the same target; it is empty
    by default.
    """
    segments = validate_segments(segments)
    if aliases:
        validate_aliases(aliases)
    start, end = window if window else (segments["start"].min(), segments["end"].max())
    grids, excluded, coverage = {}, {}, {}
    for participant, own in segments.groupby("participant", sort=True):
        resolved, excluded[participant] = resolve_overlaps(own)
        grid = build_timeline(
            resolved, start=start, end=end, resolution_sec=resolution_sec, aliases=aliases
        )
        grids[participant] = grid
        coverage[participant] = float(np.mean(grid != None)) if len(grid) else np.nan  # noqa: E711
    return Timelines(grids, excluded, coverage, float(start), float(end), resolution_sec)


# ---------------------------------------------------------------- overlap measures


def _pair_stats(
    a: np.ndarray, b: np.ndarray, gaps: Gaps, exclude_partner: bool, partner: str
) -> tuple[int, int, int]:
    """(eligible frames, matching frames, matching frames on partner) of two aligned,
    canonical grids of equal length: the one place numerator and denominator of every
    measure here are defined."""
    if gaps not in ("exclude", "count"):
        raise ValueError(f"gaps must be 'exclude' or 'count', not {gaps!r}")
    coded = (a != None) & (b != None)  # noqa: E711 - elementwise on object arrays
    eligible = coded if gaps == "exclude" else np.ones(len(a), dtype=bool)
    on_partner = (a == partner) & (b == partner)
    if exclude_partner:
        eligible = eligible & ~on_partner
    match = eligible & coded & (a == b)
    return int(eligible.sum()), int(match.sum()), int((match & on_partner).sum())


def _prepare(a, b, partner: str) -> tuple[np.ndarray, np.ndarray, str]:
    a, b = canonicalize(a), canonicalize(b)
    n = min(len(a), len(b))
    return a[:n], b[:n], canonical_label(partner)


def simultaneous_overlap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    gaps: Gaps = "exclude",
    exclude_partner: bool = False,
    partner: str = "partner",
) -> float:
    """Share of eligible frames in which both grids carry the same label at the same
    instant; NaN without an eligible frame. Grids of unequal length are compared over
    the shorter one; labels are compared in canonical form."""
    a, b, partner = _prepare(a, b, partner)
    n_eligible, n_match, _ = _pair_stats(a, b, gaps, exclude_partner, partner)
    return n_match / n_eligible if n_eligible else np.nan


def simultaneous_partner_share(a: np.ndarray, b: np.ndarray, *, partner: str = "partner") -> float:
    """Of the frames simultaneous_overlap counts as a match, the share in which both
    carry the partner label, that is mutual gaze; NaN without a match. The match set
    is the same under either gap handling."""
    a, b, partner = _prepare(a, b, partner)
    _, n_match, n_partner = _pair_stats(a, b, "exclude", False, partner)
    return n_partner / n_match if n_match else np.nan


def _best_lag(
    leader, follower, resolution_sec, max_lag_sec, gaps, exclude_partner, partner
) -> tuple[float, float]:
    """(best match share over lags 0..max_lag_sec, partner share of the matches at that
    lag). The first lag reaching the best share wins."""
    leader, follower, partner = _prepare(leader, follower, partner)
    best, best_share, found = 0.0, np.nan, False
    for lag in range(int(round(max_lag_sec / resolution_sec)) + 1):
        if lag >= len(leader):
            break
        n_eligible, n_match, n_partner = _pair_stats(
            leader[: len(leader) - lag], follower[lag:], gaps, exclude_partner, partner
        )
        if not n_eligible:
            continue
        found = True
        if n_match / n_eligible > best:
            best = n_match / n_eligible
            best_share = n_partner / n_match
    return (best, best_share) if found else (np.nan, np.nan)


def lagged_overlap(
    leader: np.ndarray,
    follower: np.ndarray,
    *,
    resolution_sec: float,
    max_lag_sec: float = 2.0,
    gaps: Gaps = "exclude",
    exclude_partner: bool = False,
    partner: str = "partner",
) -> float:
    """The highest share of matching frames between leader(t) and follower(t + lag)
    over lags from 0 to max_lag_sec in steps of one frame.

    A single best lag is searched, the same principle as the best-lag search in
    windowed cross-correlation. Counting a match whenever the labels agree at any lag
    in the window would be the alternative; with segments longer than the window it
    scores long steady segments as matches almost everywhere. Lag 0 is included, so the
    value is never below simultaneous_overlap. Swap the arguments for the other
    direction of leading.
    """
    return _best_lag(leader, follower, resolution_sec, max_lag_sec, gaps, exclude_partner, partner)[
        0
    ]


def lagged_partner_share(
    leader: np.ndarray,
    follower: np.ndarray,
    *,
    resolution_sec: float,
    max_lag_sec: float = 2.0,
    gaps: Gaps = "exclude",
    partner: str = "partner",
) -> float:
    """The partner share of the matches at the lag lagged_overlap picks with the same
    arguments, so the two are always read off the same lag."""
    return _best_lag(leader, follower, resolution_sec, max_lag_sec, gaps, False, partner)[1]
