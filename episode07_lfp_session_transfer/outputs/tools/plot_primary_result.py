"""Plot EP07's completed primary report without fitting or new inference.

Run inside an attended/task-owned Slurm allocation after the single primary
opening is complete. Only the fixed operational opening record and completed
primary JSON report are read. Existing day and aggregate differences are
displayed directly; bootstrap samples, neural arrays and model state are not
used. PNG and PDF destinations are exclusive and never overwritten.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket


EPISODE = Path(__file__).resolve().parents[2]
RUN = EPISODE / "outputs/runs/primary_evaluation/attempt001"
REPORT = RUN / "primary_report.json"
DESTINATION = EPISODE / "outputs/figures/primary_evaluation"
DAYS = [["Mihili", "20140304"], ["Mihili", "20140307"],
        ["Chewie-L", "20161007"], ["Chewie-L", "20161014"], ["Chewie-L", "20161021"]]
TITLE = "Neither region meets the full prespecified support rule"


def _compute_admission():
    job = os.environ.get("SLURM_JOB_ID", "")
    host = socket.gethostname()
    if (not job.isdigit() or "-ln" in host
            or not os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit()
            or int(os.environ["SLURM_CPUS_PER_TASK"]) < 1):
        raise SystemExit("plotting requires the actual execution shell to be in its task-owned Slurm allocation")
    temporary = os.environ.get("L_SCRATCH_JOB", "")
    if not temporary or not Path(temporary).is_absolute() or not Path(temporary).is_dir():
        raise SystemExit("the job's existing local scratch directory is required for the plotting cache")
    os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "ep07_primary_matplotlib")
    os.environ["MPLBACKEND"] = "Agg"


def _load_completed_report(path):
    if path.resolve() != REPORT:
        raise ValueError("only the fixed completed primary report is permitted")
    # Admit the operational completion record before reading scientific values.
    opening = json.loads((RUN / "opening.json").read_text())
    if (opening.get("state") != "primary_evaluation_complete"
            or any(opening.get(name) is not True for name in
                   ("complete", "held_out_spike_contact", "analysis_freeze_complete",
                    "scientific_checks_complete", "ten_paired_controls_complete", "four_fixed_recoveries_complete"))):
        raise PermissionError("the single primary opening and all scientific prerequisites must be complete")
    try:
        frozen, started, ended = (datetime.fromisoformat(opening[name]) for name in
                                 ("analysis_frozen_utc", "started_utc", "ended_utc"))
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("completed opening requires explicit freeze/start/end timestamps") from error
    if any(value.tzinfo is None for value in (frozen, started, ended)) or not frozen <= started <= ended:
        raise PermissionError("the completed primary opening must follow its analysis freeze")
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("unexpected primary report size")
    report = json.loads(path.read_text())
    if (report.get("evidence_stage") != "single_primary_held_out_evaluation"
            or report.get("scientific_checks_certified") is not True
            or report.get("authorizes_opening") is not False
            or report.get("regions") != ["M1", "PMd"]
            or report.get("comparators") != ["today_only", "calibration_mean"]
            or report.get("day_order") != DAYS
            or report.get("supported_regions") != []
            or report.get("arithmetic_pass") != [False, False]
            or report.get("regional_decisions") != {
                "M1": "prespecified_support_rule_not_met", "PMd": "prespecified_support_rule_not_met"}):
        raise ValueError("this figure is scoped to the completed five-day result with neither region supported")
    choices = report["choices"]
    if (choices.get("weighting") != "equal_days_within_animal_equal_animals"
            or choices.get("margin") != .01
            or choices.get("residual_followup") != "not_performed_in_this_primary_evaluation"
            or choices.get("claim_limit") != "prediction_transfer_only_no_coupling_or_mean_only_mechanism_claim"):
        raise ValueError("reported weighting, margin or interpretation differs from the plotted result")
    observed, inference = report["observed"], report["margins"]
    if (observed.get("margin_pass") != [True, False]
            or observed.get("day_frequency_pass") != [False, True]
            or observed.get("positive_day_counts") != [[2, 2], [1, 3]]
            or observed.get("omission_pass") != [[True, True], [True, True], [False, True], [True, True], [True, True]]
            or inference.get("available") != [True, True]
            or inference.get("holm_rejected") != [True, False]):
        raise ValueError("the recorded decision criteria differ from this figure's explanatory text")
    return report, opening


def _finite_matrix(values, shape, name):
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} requires complete finite reported values with shape{shape}")
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--out-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    _compute_admission()
    if args.out_dir.resolve() != DESTINATION:
        raise ValueError("use the explicitly scoped EP07 primary figure directory")
    outputs = tuple(args.out_dir / f"ep07_primary_result.{extension}" for extension in ("png", "pdf"))
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("primary result figure already exists; refuse to overwrite either format")
    report, opening = _load_completed_report(args.report)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    day_differences = _finite_matrix(report["day_differences"], (5, 2, 2), "day differences")
    aggregates = _finite_matrix(report["observed"]["aggregate_differences"], (2, 2), "reported aggregates")
    adjusted_p = _finite_matrix(report["margins"]["holm_adjusted_p"], (2,), "existing Holm-adjusted p values")
    if not math.isfinite(float(report["choices"]["margin"])):
        raise ValueError("finite reported meaningful margin required")
    margin = float(report["choices"]["margin"])
    colors, markers = ("#0072B2", "#D55E00"), ("o", "s")
    ys = np.asarray((5., 4., 2.7, 1.7, .7))
    aggregate_y = -1.
    offsets = (.12, -.12)
    labels = [f"{animal}  ·  {day[:4]}-{day[4:6]}-{day[6:]}" for animal, day in DAYS]
    labels.append("Aggregate  ·  equal animal weights")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.235, right=.97, bottom=.32, top=.79, wspace=.095)
    fig.suptitle(TITLE, x=.53, y=.975, fontsize=17, fontweight="bold")
    fig.text(.53, .936, "EP07  |  Single completed primary evaluation  |  Five target days, two animals",
             ha="center", fontsize=11, color="#4F5661")
    legend = [Line2D([], [], marker=markers[index], color=colors[index], linestyle="none", markersize=7,
                     label=label) for index, label in enumerate(("History − today-only", "History − calibration mean"))]
    legend.extend((Line2D([], [], marker="D", color="#484848", linestyle="none", markersize=7,
                          label="Reported aggregate"),
                   Line2D([], [], color="#454545", linewidth=1.1, label="Zero: day sign criterion"),
                   Line2D([], [], color="#8E7400", linestyle="--", linewidth=1.5,
                          label="+0.01: aggregate threshold only")))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.59, .903), ncol=3,
               frameon=False, columnspacing=1.8, handletextpad=.6, fontsize=10)

    for region, ax in enumerate(axes):
        ax.set_title(("M1  ·  Movement", "PMd  ·  Preparation")[region], pad=15, fontweight="bold")
        ax.set_xlim(-.027, .258)
        ax.set_ylim(-1.62, 5.72)
        ax.axvspan(-.027, 0., color="#F8EFED", zorder=0)
        ax.axhspan(-1.5, -.5, color="#F1F4F7", zorder=.2)
        ax.axvline(0., color="#454545", linewidth=1.1, zorder=1)
        # The practical margin is an aggregate requirement, not a day threshold.
        ax.vlines(margin, -1.49, -.51, color="#8E7400", linestyle="--", linewidth=1.6, zorder=2)
        ax.axhline(3.35, color="#D7DCE0", linewidth=.8)
        ax.axhline(-.22, color="#D7DCE0", linewidth=.8)
        for comparator, (color, marker, offset) in enumerate(zip(colors, markers, offsets)):
            ax.scatter(day_differences[:, region, comparator], ys + offset,
                       s=48, marker=marker, color=color, edgecolors="white", linewidths=.7, zorder=4)
            value = aggregates[region, comparator]
            ax.scatter([value], [aggregate_y + offset], s=76, marker="D", color=color,
                       edgecolors="white", linewidths=.7, zorder=5)
            ax.annotate(f"{value:+.4f}", (value, aggregate_y + offset), xytext=(7, 0),
                        textcoords="offset points", va="center", color=color, fontsize=9, fontweight="bold")
        ax.set_yticks(tuple(ys) + (aggregate_y,), labels)
        ax.tick_params(axis="y", length=0, pad=9, labelsize=10)
        ax.set_xticks((0., .05, .10, .15, .20, .25))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("Difference in joint native-scale R²", labelpad=9)
        ax.grid(axis="x", color="#E8EAED", linewidth=.7)
        ax.set_axisbelow(True)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color("#A0A6AD")

    summaries = (
        "M1: gains vary across Chewie days\n"
        f"Aggregate margin test passes (Holm p = {adjusted_p[0]:.4f}).\n"
        "Chewie: 1/3 days positive versus both; at least 2/3 required.\n"
        "The day criterion and the October 7 omission check fail.",
        "PMd: improvement falls below the practical margin\n"
        f"Aggregate history − today-only = {aggregates[1, 0]:+.4f} < +0.0100.\n"
        f"Aggregate margin test does not pass (Holm p = {adjusted_p[1]:.4f}).\n"
        "The reported day-sign and omission checks pass.")
    for ax, summary in zip(axes, summaries):
        position = ax.get_position()
        fig.text(position.x0, .223, summary, ha="left", va="top", fontsize=9.5,
                 linespacing=1.5, color="#333B45")
    caption = (
        "Dots/squares are existing day estimates; diamonds are the existing aggregate: equal days within each animal, then equal animals.\n"
        "Per-day requirements use zero; the +0.01 practical margin applies only to the aggregate. No new intervals or inference are calculated here.\n"
        "Inference is conditional on five fixed days, two animals and fixed fits in a previously exposed public corpus. No trial-coupling claim.\n"
        "Source: outputs/runs/primary_evaluation/attempt001/primary_report.json")
    fig.text(.235, .096, caption, ha="left", va="top", fontsize=8.2, linespacing=1.5, color="#505862")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    description = TITLE + "\n" + caption + f"\nCompleted opening: {opening['ended_utc']}"
    # Claim both outputs before rendering either; any partial files remain
    # visible after failure and are never silently overwritten on another run.
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("xb")) for path in outputs]
        fig.savefig(handles[0], format="png", dpi=220, facecolor="white",
                    metadata={"Title": TITLE, "Description": description})
        fig.savefig(handles[1], format="pdf", facecolor="white",
                    metadata={"Title": TITLE, "Subject": description, "Creator": "EP07 completed-report plot"})
    plt.close(fig)
    print(json.dumps({"state": "completed_report_figure_written", "source": str(REPORT),
                      "outputs": [str(path) for path in outputs], "new_inference": False}), flush=True)


if __name__ == "__main__":
    main()
