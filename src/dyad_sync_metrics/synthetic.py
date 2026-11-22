"""Seeded synthetic dyad recordings with known ground truth.

Nothing here resembles recorded data; every value is generated at run time from the
parameters given. Two generators:

- tracking: head and body positions for two seated people facing each other. Person
  1's motion is smooth random displacement; person 2's is a mix of person 1's
  displacement, delayed by lag_sec and weighted by coupling, and independent motion of
  the same kind. coupling=0 gives an independent pair.
- gaze: a sticky Markov chain over joint gaze states. In each frame the pair is either
  in mutual gaze, looking at the same scene object, or looking at two different
  objects. The long-run shares of the first two states are p_mutual and p_shared
  exactly, which fixes the expected mutual-gaze fraction and gaze-ray convergence rate.

synthetic_study combines both into the {(dyad, condition): session} mapping that
dyad_sync_metrics.surrogates takes, with per-condition parameters.

Two further generators produce coded and questionnaire data:

- interval_codes: a segment table as a coder would produce it from video. Person 1
  moves between labelled targets; person 2 copies each of person 1's targets with
  probability p_shared after a known lag, and otherwise looks at a different target.
  The share of matching time at the generating lag is therefore p_shared in
  expectation. Optional edge cases exercise overlap resolution and label matching.
- survey: questionnaire answers from a one-factor model per scale, with
  reverse-keyed items. Cronbach's alpha of the model follows from the loadings and the
  noise (expected_alpha).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

SEAT_1 = np.array([0.0, 1.2, 0.0])
"""Head position of person 1 at rest, in metres; y is up."""
SEAT_2 = np.array([0.0, 1.2, 1.5])
"""Head position of person 2 at rest, facing person 1 across a table along z."""
BODY_OFFSET = np.array([0.0, -0.5, 0.0])
"""Body (upper torso) point relative to the head at rest."""

OBJECTS = {
    "left": np.array([-0.5, 0.75, 0.75]),
    "centre": np.array([0.0, 0.75, 0.75]),
    "right": np.array([0.5, 0.75, 0.75]),
}
"""Scene objects on the table between the two people, in the same frame as the seats.
From the rest positions, two rays aimed at different objects pass at least 0.4 m apart,
and a ray aimed at an object passes at least 0.7 m from the partner's head, so neither
case is mistaken for convergence or for looking at the partner."""


@dataclass(frozen=True)
class ConditionParams:
    """Generator parameters of one condition."""

    coupling: float = 0.0
    """Weight of person 1's delayed motion in person 2's motion, in [0, 1]."""
    lag_sec: float = 0.5
    """Delay with which person 2 follows person 1."""
    p_mutual: float = 0.1
    """Long-run share of frames in mutual gaze."""
    p_shared: float = 0.2
    """Long-run share of frames in which both look at the same object."""


def _smooth_noise(rng: np.random.Generator, n: int, fs: float, cutoff_hz: float) -> np.ndarray:
    """(n, 3) low-pass filtered Gaussian noise, scaled to unit standard deviation."""
    b, a = butter(2, cutoff_hz / (fs / 2), btype="low")
    x = filtfilt(b, a, rng.standard_normal((n, 3)), axis=0)
    return x / x.std(axis=0)


def tracking(
    rng: np.random.Generator,
    *,
    n_frames: int,
    fs: float,
    coupling: float,
    lag_sec: float,
    amplitude_m: float = 0.03,
    cutoff_hz: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Head and body trajectories of one dyad.

    Returns {"p1": {...}, "p2": {...}}, each with "t" (seconds), "head" and "body"
    ((n_frames, 3) positions in metres). Displacements around the rest positions are
    smooth noise of standard deviation amplitude_m per axis; head and body move
    independently of each other. Person 2's displacement is
    coupling * (person 1's, delayed by lag_sec) + sqrt(1 - coupling^2) * (own noise).
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must lie in [0, 1]")
    lag = round(lag_sec * fs)
    t = np.arange(n_frames) / fs
    own = np.sqrt(1.0 - coupling**2)
    people = {"p1": {"t": t}, "p2": {"t": t}}
    for part, offset in (("head", np.zeros(3)), ("body", BODY_OFFSET)):
        leader = _smooth_noise(rng, n_frames + lag, fs, cutoff_hz)
        follower = coupling * leader[:n_frames] + own * _smooth_noise(rng, n_frames, fs, cutoff_hz)
        people["p1"][part] = SEAT_1 + offset + amplitude_m * leader[lag:]
        people["p2"][part] = SEAT_2 + offset + amplitude_m * follower
    return people


def _gaze_states(
    rng: np.random.Generator, n_frames: int, p_mutual: float, p_shared: float, stay: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame gaze targets of both people as indices: -1 is the partner, 0..k-1 an
    object. In each frame the joint state is kept with probability stay, otherwise
    redrawn from (mutual: p_mutual, same object: p_shared, two different objects: rest),
    so the redraw distribution is also the stationary one."""
    if p_mutual < 0 or p_shared < 0 or p_mutual + p_shared > 1:
        raise ValueError("p_mutual and p_shared must be non-negative and sum to at most 1")
    k = len(OBJECTS)
    target_1 = np.empty(n_frames, dtype=int)
    target_2 = np.empty(n_frames, dtype=int)
    current = (-1, -1)
    for i in range(n_frames):
        if i == 0 or rng.random() >= stay:
            u = rng.random()
            if u < p_mutual:
                current = (-1, -1)
            elif u < p_mutual + p_shared:
                obj = int(rng.integers(k))
                current = (obj, obj)
            else:
                obj_1, obj_2 = rng.choice(k, size=2, replace=False)
                current = (int(obj_1), int(obj_2))
        target_1[i], target_2[i] = current
    return target_1, target_2


def _aim(
    rng: np.random.Generator,
    origin: np.ndarray,
    partner_head: np.ndarray,
    targets: np.ndarray,
    jitter_deg: float,
) -> np.ndarray:
    """Unit gaze directions from origin toward each frame's target, with angular noise."""
    objects = np.stack(list(OBJECTS.values()))
    points = np.where((targets == -1)[:, None], partner_head, objects[np.maximum(targets, 0)])
    direction = points - origin
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction += np.tan(np.radians(jitter_deg)) * rng.standard_normal(direction.shape)
    return direction / np.linalg.norm(direction, axis=1, keepdims=True)


def gaze(
    rng: np.random.Generator,
    head_1: np.ndarray,
    head_2: np.ndarray,
    *,
    p_mutual: float,
    p_shared: float,
    stay: float = 0.95,
    jitter_deg: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Gaze rays of one dyad, given both people's head trajectories.

    Returns {"p1": {...}, "p2": {...}}, each with "gaze_origin" (the head position),
    "gaze_direction" (unit vectors) and "target" (per-frame index into OBJECTS, or -1
    for the partner). With jitter of about a degree, mutual-gaze frames lie well inside
    common angular thresholds and same-object frames well inside a 10 cm convergence
    threshold, so the expected mutual-gaze fraction is p_mutual and the expected
    convergence rate over the frames in which neither looks at the other is
    p_shared / (1 - p_mutual).
    """
    head_1, head_2 = np.asarray(head_1, dtype=float), np.asarray(head_2, dtype=float)
    target_1, target_2 = _gaze_states(rng, len(head_1), p_mutual, p_shared, stay)
    return {
        "p1": {
            "gaze_origin": head_1,
            "target": target_1,
            "gaze_direction": _aim(rng, head_1, head_2, target_1, jitter_deg),
        },
        "p2": {
            "gaze_origin": head_2,
            "target": target_2,
            "gaze_direction": _aim(rng, head_2, head_1, target_2, jitter_deg),
        },
    }


def synthetic_study(
    seed: int,
    conditions: Mapping[str, ConditionParams],
    *,
    n_dyads: int,
    n_frames: int,
    fs: float = 30.0,
) -> dict[tuple[int, str], dict[str, dict[str, np.ndarray]]]:
    """A complete synthetic study: every dyad recorded once in every condition.

    Returns {(dyad, condition): {"p1": {...}, "p2": {...}}} with dyads numbered from 0.
    Each person carries "t", "head", "body", "gaze_origin", "gaze_direction" and
    "target"; see tracking and gaze. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    study = {}
    for dyad in range(n_dyads):
        for condition, params in conditions.items():
            people = tracking(
                rng, n_frames=n_frames, fs=fs, coupling=params.coupling, lag_sec=params.lag_sec
            )
            rays = gaze(
                rng,
                people["p1"]["head"],
                people["p2"]["head"],
                p_mutual=params.p_mutual,
                p_shared=params.p_shared,
            )
            study[(dyad, condition)] = {p: {**people[p], **rays[p]} for p in ("p1", "p2")}
    return study


LABELS = ("table", "door", "window", "partner")
"""Coded gaze targets of interval_codes; "partner" is the other person."""

EDGE_OVERLAP_SEC = 0.3
"""Width of the adjacent-boundary overlap interval_codes adds with edge_cases."""


def _grid(x: float, precision_sec: float) -> float:
    return round(round(x / precision_sec) * precision_sec, 10)


def interval_codes(
    rng: np.random.Generator,
    *,
    duration_sec: float,
    p_shared: float,
    lag_sec: float = 1.0,
    lag_jitter_sec: float = 0.0,
    mean_dwell_sec: float = 3.0,
    min_dwell_sec: float = 1.0,
    labels: Sequence[str] = LABELS,
    precision_sec: float = 0.1,
    edge_cases: bool = False,
) -> pd.DataFrame:
    """Coded gaze segments of one dyad: a table with the columns start, end,
    participant ("p1" or "p2") and label.

    Person 1's segments follow each other without gaps; each lasts min_dwell_sec plus
    an exponential share with mean mean_dwell_sec - min_dwell_sec, and moves to a
    label other than the previous one. Segments are generated until duration_sec is
    reached, so the recording is at least that long. Person 2 has one segment per
    segment of person 1, delayed by a lag drawn uniformly from lag_sec +- lag_jitter_sec;
    it carries person 1's label with probability p_shared and otherwise one of the
    other labels, uniformly. A jitter makes neighbouring segments of person 2 overlap
    or leave gaps of up to 2 * lag_jitter_sec, which must stay below min_dwell_sec.
    All boundaries lie on a grid of precision_sec.

    With edge_cases, a fixed set of irregularities is added: person 1's second segment
    runs EDGE_OVERLAP_SEC into the third (an adjacent-boundary overlap), the fifth
    contains a nested segment with another label over its middle third, and the eighth
    is removed (an uncoded gap); person 2's third label is upper-case and the fourth
    is title-case with surrounding spaces.
    """
    if not 0.0 <= p_shared <= 1.0:
        raise ValueError("p_shared must lie in [0, 1]")
    if not 2 * lag_jitter_sec < min_dwell_sec <= mean_dwell_sec:
        raise ValueError("need 2 * lag_jitter_sec < min_dwell_sec <= mean_dwell_sec")
    labels = list(labels)
    if len(labels) < 2:
        raise ValueError("need at least two labels")

    bounds, sequence = [0.0], []
    while bounds[-1] < duration_sec:
        dwell = min_dwell_sec + rng.exponential(mean_dwell_sec - min_dwell_sec)
        bounds.append(_grid(bounds[-1] + dwell, precision_sec))
        options = [x for x in labels if not sequence or x != sequence[-1]]
        sequence.append(options[rng.integers(len(options))])
    p1 = [
        {"start": s, "end": e, "label": x}
        for s, e, x in zip(bounds[:-1], bounds[1:], sequence, strict=True)
    ]

    p2 = []
    for segment in p1:
        lag = _grid(lag_sec + rng.uniform(-lag_jitter_sec, lag_jitter_sec), precision_sec)
        if rng.random() < p_shared:
            label = segment["label"]
        else:
            others = [x for x in labels if x != segment["label"]]
            label = others[rng.integers(len(others))]
        p2.append({"start": segment["start"] + lag, "end": segment["end"] + lag, "label": label})

    if edge_cases:
        if len(p1) < 8:
            raise ValueError("edge_cases needs at least eight segments; raise duration_sec")
        p1[1]["end"] = _grid(p1[1]["end"] + EDGE_OVERLAP_SEC, precision_sec)
        outer = p1[4]
        third = (outer["end"] - outer["start"]) / 3
        inner_label = next(x for x in labels if x != outer["label"])
        inner = {
            "start": _grid(outer["start"] + third, precision_sec),
            "end": _grid(outer["start"] + 2 * third, precision_sec),
            "label": inner_label,
        }
        p1 = [*p1[:5], inner, *p1[5:7], *p1[8:]]
        p2[2]["label"] = p2[2]["label"].upper()
        p2[3]["label"] = f"  {p2[3]['label'].title()} "

    rows = [{**s, "participant": "p1"} for s in p1] + [{**s, "participant": "p2"} for s in p2]
    return pd.DataFrame(rows, columns=["start", "end", "participant", "label"])


def example_item_plan() -> pd.DataFrame:
    """An item plan for two fictional scales, with a loading per item for survey:
    ScaleA with four items, one reverse-keyed, and ScaleB with three, one reverse-keyed."""
    return pd.DataFrame(
        {
            "item": ["A1", "A2", "A3", "A4", "B1", "B2", "B3"],
            "scale": ["ScaleA"] * 4 + ["ScaleB"] * 3,
            "reverse": [False, False, True, False, False, True, False],
            "loading": [0.9, 0.8, 0.8, 0.7, 0.9, 0.8, 0.7],
        }
    )


def expected_alpha(loadings: Sequence[float], noise_sd: float) -> float:
    """Cronbach's alpha of the one-factor model survey draws from, before rounding and
    clipping: with item variances l_j^2 + noise_sd^2 and sum-score variance
    (sum l_j)^2 + k * noise_sd^2, k / (k - 1) * (1 - sum of item variances / sum-score
    variance)."""
    loadings = np.asarray(loadings, dtype=float)
    k = len(loadings)
    item_var = np.sum(loadings**2) + k * noise_sd**2
    total_var = loadings.sum() ** 2 + k * noise_sd**2
    return float(k / (k - 1) * (1 - item_var / total_var))


def survey(
    rng: np.random.Generator,
    plan: pd.DataFrame,
    *,
    n_dyads: int,
    condition_shift: Mapping[str, float],
    noise_sd: float = 0.8,
    scale_min: int = 1,
    scale_max: int = 7,
) -> pd.DataFrame:
    """Questionnaire answers of every person of n_dyads dyads in every condition: a long
    table with the columns person, dyad, condition, item and value.

    plan has the columns item, scale, reverse and loading. Each person has one
    standard-normal trait per scale, the same in every condition; condition_shift adds
    a fixed amount per condition. An item's keyed response is the scale midpoint plus
    loading * (trait + shift) plus normal noise of noise_sd. A reverse-keyed item is
    answered mirrored (scale_min + scale_max - keyed response), so reverse scoring
    recovers the keyed direction. Answers are rounded to whole points and clipped to
    the scale. Persons are numbered from 0, two per dyad: person p is in dyad p // 2.
    """
    midpoint = (scale_min + scale_max) / 2
    scales = plan["scale"].unique()
    rows = []
    for person in range(2 * n_dyads):
        trait = dict(zip(scales, rng.standard_normal(len(scales)), strict=True))
        for condition, shift in condition_shift.items():
            for item in plan.itertuples():
                keyed = midpoint + item.loading * (trait[item.scale] + shift)
                keyed += noise_sd * rng.standard_normal()
                given = scale_min + scale_max - keyed if item.reverse else keyed
                value = float(np.clip(np.rint(given), scale_min, scale_max))
                rows.append((person, person // 2, condition, item.item, value))
    return pd.DataFrame(rows, columns=["person", "dyad", "condition", "item", "value"])
