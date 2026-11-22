# dyad-sync-metrics

Measures of interpersonal coordination for two people recorded together with motion
and eye tracking, plus the chance baselines needed to interpret them.

## What it measures

| Measure | Question | Module |
|---|---|---|
| Movement synchrony | Do the two people's head or body movements co-vary over time, possibly with a lag? | `sync.windowed_xcorr` on `signal.speed` |
| Mutual gaze | In how many frames do both look at each other? | `gaze.mutual_gaze_fraction` |
| Gaze coupling | Does one person's looking-toward track the other's over time? | `sync.windowed_xcorr` on `gaze.gaze_coupling_series` |
| Gaze-ray convergence | In how many frames do the two gaze rays meet somewhere in space while neither looks at the other? | `gaze.gaze_convergence_series`, `gaze.gaze_convergence_rate` |
| Co-attention overlap | In video-coded gaze, how often do both people look at the same target, at once or with a short lag? | `intervals.simultaneous_overlap`, `intervals.lagged_overlap` |

Raw values mix coordination with structure the setting imposes: people doing the same
task move at similar tempos, and people seated face to face look toward each other
and into the same workspace. `surrogates` estimates that structural share and
subtracts it:

- **Pseudo-dyads** pair person 1 of one dyad with person 2 of another dyad from the
  same condition (`pseudo_dyad_baseline`, `pseudo_dyad_correction`).
- **Circular shift** rotates one person's series against the partner's within a dyad
  (`circular_shift_correction`).

Both take any metric `metric(p1, p2) -> float`, so the baseline is always computed
with exactly the function, preprocessing and NaN handling of the value it corrects.
Sessions are passed as a mapping `{(dyad, condition): {"p1": ..., "p2": ...}}`, with
condition names chosen by the caller.

## Interval coding

`intervals` works on coded segments with the columns `start`, `end`, `participant`
and `label`; `from_segment_export` maps the columns of an existing export onto these.
`build_timelines` resolves overlapping segments of the same person (an
adjacent-boundary overlap becomes uncoded time, a fully contained segment wins over
its container, every other shape raises), puts each person on a common time grid and
canonicalises labels (case and surrounding whitespace, plus an optional alias map,
empty by default). The overlap measures then exclude uncoded frames from the
denominator (`gaps="exclude"`) or count them as non-matches (`gaps="count"`), and can
report or drop the frames in which both look at each other (`simultaneous_partner_share`,
`lagged_partner_share`, `exclude_partner=True`).

## Questionnaire scales

`scales` takes answers as a long table (`person`, `dyad`, `condition`, `item`,
`value`) and an item plan (`item`, `scale`, `reverse`). `score_items` validates both
and reverse scores, `scale_scores` gives per-person scale means, `dyad_scores` the
partners' mean and absolute difference, and `reliability` Cronbach's alpha per scale
and condition. `with_subset_scale` adds a scale scored from a named subset of items.

## Small-sample inference

`exact` enumerates permutation distributions instead of relying on asymptotics:
`exact_spearman` (two-sided exact p over the distinct permutations of tied data) and
`exact_paired_test` (exact sign test and sign-flip permutation test). Both refuse an
enumeration above a ceiling, return NaN rather than a p-value for an undefined
statistic, and add a `nesting_caveat` when observations share a group, for example two
conditions of one dyad.

## Synthetic data

`synthetic` generates seeded data with known ground truth: coupled and independent
movement; gaze driven by a Markov chain with set probabilities of mutual gaze and
shared attention; coded gaze segments in which one person follows the other's target
with a set probability and lag, optionally with overlaps, a nested segment, a gap and
case variants; and questionnaire answers from a one-factor model whose Cronbach's
alpha follows from the loadings. The test suite uses it to check that the measures
recover what was put in.

## Status

Early development. A fuller methods description and a worked example are planned.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```python
from dyad_sync_metrics.signal import speed
from dyad_sync_metrics.surrogates import pseudo_dyad_correction
from dyad_sync_metrics.sync import windowed_xcorr
from dyad_sync_metrics.synthetic import ConditionParams, synthetic_study

FS = 30.0
study = synthetic_study(
    seed=0,
    conditions={"a": ConditionParams(coupling=0.8), "b": ConditionParams(coupling=0.0)},
    n_dyads=5,
    n_frames=int(120 * FS),
    fs=FS,
)


def head_sync(p1, p2):
    return windowed_xcorr(speed(p1["head"], p1["t"], FS), speed(p2["head"], p2["t"], FS), FS)


for (dyad, condition), r in pseudo_dyad_correction(study, head_sync, ["a", "b"]).items():
    print(dyad, condition, round(r.observed, 3), round(r.baseline, 3), round(r.corrected, 3))
```

## Tests

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

All test input is synthetic and generated at run time.

## License

MIT, see [LICENSE](LICENSE).
