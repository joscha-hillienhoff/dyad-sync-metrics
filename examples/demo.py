"""Worked example on a small synthetic sample: every measure, its surrogate baseline, the
questionnaire scales and the exact tests.

Everything is generated at run time from fixed seeds. Run it with

    uv run python examples/demo.py [figure.png]

It prints the results and saves the raw-versus-baseline figure (default
docs/raw_vs_baseline.png). It needs no display.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dyad_sync_metrics import gaze as g  # noqa: E402
from dyad_sync_metrics.exact import exact_paired_test, exact_spearman  # noqa: E402
from dyad_sync_metrics.intervals import (  # noqa: E402
    build_timelines,
    lagged_overlap,
    lagged_partner_share,
    simultaneous_overlap,
    simultaneous_partner_share,
)
from dyad_sync_metrics.scales import (  # noqa: E402
    dyad_scores,
    reliability,
    scale_scores,
    score_items,
)
from dyad_sync_metrics.signal import speed  # noqa: E402
from dyad_sync_metrics.surrogates import (  # noqa: E402
    circular_shift_correction,
    pseudo_dyad_correction,
)
from dyad_sync_metrics.sync import windowed_xcorr  # noqa: E402
from dyad_sync_metrics.synthetic import (  # noqa: E402
    ConditionParams,
    example_item_plan,
    interval_codes,
    survey,
    synthetic_study,
)

FS = 30.0
N_DYADS = 6
DURATION_SEC = 120
MAX_DIST_M = 2.5
HEAD_EXCLUSION_M = 0.30
MUTUAL_THRESHOLD_DEG = 10.0
CONVERGENCE_THRESHOLD_M = 0.10

# Condition "coupled": person 2 follows person 1's movement and the pair looks at each
# other and at shared objects more often than chance. Condition "independent": no
# coupling, and both people pick one of the three scene objects independently, so they
# land on the same one a third of the time.
CONDITIONS = {
    "coupled": ConditionParams(coupling=0.8, p_mutual=0.2, p_shared=0.4),
    "independent": ConditionParams(coupling=0.0, p_mutual=0.0, p_shared=1 / 3),
}
COLORS = {"raw": "#0072B2", "baseline": "#E69F00"}


# ---------------------------------------------------------------- metrics
# A metric takes the data of person 1 and person 2 and returns one number. The same
# function scores a real pair and every surrogate pair.


def head_sync(p1, p2):
    return windowed_xcorr(speed(p1["head"], p1["t"], FS), speed(p2["head"], p2["t"], FS), FS)


def angles(p1, p2):
    return (
        g.angle_to_target(p1["gaze_origin"], p1["gaze_direction"], p2["head"]),
        g.angle_to_target(p2["gaze_origin"], p2["gaze_direction"], p1["head"]),
    )


def mutual_gaze(p1, p2):
    return g.mutual_gaze_fraction(*angles(p1, p2), MUTUAL_THRESHOLD_DEG)


def gaze_coupling_strength(p1, p2):
    a, b = (g.gaze_coupling_series(x) for x in angles(p1, p2))
    return windowed_xcorr(a, b, FS)


def gaze_coupling_direction(p1, p2):
    # Signed value at lag 0: the sign under a lag search is not reliable.
    a, b = (g.gaze_coupling_series(x) for x in angles(p1, p2))
    return windowed_xcorr(a, b, FS, use_abs=False, max_lag_sec=0)


def convergence(p1, p2):
    return g.gaze_convergence_series(
        p1["gaze_origin"],
        p1["gaze_direction"],
        p1["head"],
        p2["gaze_origin"],
        p2["gaze_direction"],
        p2["head"],
        max_dist_m=MAX_DIST_M,
        head_exclusion_m=HEAD_EXCLUSION_M,
    )


def convergence_rate(p1, p2):
    return g.gaze_convergence_rate(*convergence(p1, p2), CONVERGENCE_THRESHOLD_M)


MEASURES = {
    "head synchrony": head_sync,
    "mutual gaze": mutual_gaze,
    "gaze coupling (strength)": gaze_coupling_strength,
    "gaze coupling (direction)": gaze_coupling_direction,
    "gaze convergence rate": convergence_rate,
}


def correct_all(study):
    """Observed, pseudo-dyad baseline and corrected value per dyad, condition, measure."""
    rows = []
    for measure, metric in MEASURES.items():
        for (dyad, condition), r in pseudo_dyad_correction(study, metric, list(CONDITIONS)).items():
            rows.append((measure, condition, dyad, r.observed, r.baseline, r.corrected))
    return pd.DataFrame(
        rows, columns=["measure", "condition", "dyad", "observed", "baseline", "corrected"]
    )


def plot_raw_vs_baseline(results: pd.DataFrame, path: Path) -> None:
    """Per measure and condition: the raw value of each dyad against the pseudo-dyad
    baseline of its condition."""
    panels = ["head synchrony", "mutual gaze", "gaze convergence rate"]
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.4), sharey=False)
    rng = np.random.default_rng(0)
    for ax, measure in zip(axes, panels, strict=True):
        for x, condition in enumerate(CONDITIONS):
            rows = results[(results["measure"] == measure) & (results["condition"] == condition)]
            jitter = rng.uniform(-0.08, 0.08, len(rows))
            ax.scatter(
                x - 0.15 + jitter, rows["observed"], s=22, color=COLORS["raw"], alpha=0.8,
                label="raw, one point per dyad" if x == 0 else None,
            )  # fmt: skip
            baseline = rows["baseline"].iloc[0]
            ax.hlines(baseline, x + 0.02, x + 0.3, color=COLORS["baseline"], lw=3,
                      label="pseudo-dyad baseline" if x == 0 else None)  # fmt: skip
        ax.set_xticks(range(len(CONDITIONS)), list(CONDITIONS))
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_title(measure, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Raw value against its surrogate baseline (synthetic data)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main(figure: Path) -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.3f}".format)

    study = synthetic_study(0, CONDITIONS, n_dyads=N_DYADS, n_frames=int(DURATION_SEC * FS), fs=FS)
    print(f"{N_DYADS} dyads x {len(CONDITIONS)} conditions, {DURATION_SEC} s at {FS:.0f} Hz")

    section("Motion and gaze: raw value, pseudo-dyad baseline, corrected value")
    results = correct_all(study)
    print(results.groupby(["measure", "condition"])[["observed", "baseline", "corrected"]].mean())
    plot_raw_vs_baseline(results, figure)
    print(f"figure written to {figure}")

    section("Circular-shift baseline within one dyad (head synchrony, 20 shifts)")
    session = study[(0, "coupled")]
    a = speed(session["p1"]["head"], session["p1"]["t"], FS)
    b = speed(session["p2"]["head"], session["p2"]["t"], FS)
    shifted = circular_shift_correction(
        a, b, lambda x, y: windowed_xcorr(x, y, FS), n_iter=20, seed=0
    )
    print(shifted)

    section("Gaze-ray convergence audit values, dyad 0, coupled")
    dist, excluded = convergence(session["p1"], session["p2"])
    print(f"median ray distance (m): {g.gaze_convergence_median(dist, excluded):.3f}")
    print(f"excluded share:          {g.gaze_convergence_excluded_share(dist, excluded):.3f}")

    section("Interval-coded co-attention, one dyad, with irregular segments")
    codes = interval_codes(
        np.random.default_rng(1), duration_sec=600, p_shared=0.6, edge_cases=True
    )
    timelines = build_timelines(codes)
    p1, p2 = timelines.grids["p1"], timelines.grids["p2"]
    res = timelines.resolution_sec
    print(f"seconds excluded as ambiguous overlap: {timelines.excluded_sec}")
    print(f"coverage: {timelines.coverage}")
    print(f"simultaneous overlap:   {simultaneous_overlap(p1, p2):.3f}")
    print(f"lagged overlap:         {lagged_overlap(p1, p2, resolution_sec=res):.3f}")
    print(f"partner share (simult.): {simultaneous_partner_share(p1, p2):.3f}")
    print(f"partner share (lagged):  {lagged_partner_share(p1, p2, resolution_sec=res):.3f}")
    print(
        f"overlap without partner frames: {simultaneous_overlap(p1, p2, exclude_partner=True):.3f}"
    )
    chance = circular_shift_correction(
        p1, p2, lambda x, y: lagged_overlap(x, y, resolution_sec=res), n_iter=30, seed=0
    )
    print(f"lagged overlap against circular shifts: {chance}")

    section("Questionnaire scales")
    plan = example_item_plan()
    answers = survey(
        np.random.default_rng(2),
        plan,
        n_dyads=N_DYADS,
        condition_shift={"coupled": 0.5, "independent": 0.0},
    )
    scored = score_items(answers, plan, scale_min=1, scale_max=7)
    persons = scale_scores(scored)
    dyads = dyad_scores(persons, ["ScaleA", "ScaleB"])
    print(dyads.groupby("condition")[["ScaleA", "ScaleB", "ScaleA_absdiff"]].mean())
    print(reliability(scored))

    section("Exact tests on a handful of dyads")
    wide = results[results["measure"] == "head synchrony"].pivot(
        index="dyad", columns="condition", values="corrected"
    )
    paired = exact_paired_test((wide["coupled"] - wide["independent"]).to_numpy())
    print(f"paired test, one difference per dyad (no nesting): {paired}")

    merged = dyads.merge(results[results["measure"] == "head synchrony"], on=["dyad", "condition"])
    first = merged[merged["dyad"] < 4]  # 8 rows keep the enumeration small
    rho = exact_spearman(
        first["ScaleA"], first["corrected"], groups=first["dyad"].tolist(), name="corrected"
    )
    print(f"Spearman, both conditions of 4 dyads as 8 rows: {rho}")
    print("The caveat above is the point: these rows are not independent units.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/raw_vs_baseline.png"))
