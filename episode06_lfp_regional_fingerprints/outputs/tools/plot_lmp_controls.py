"""Plot completed EP06 LMP comparisons without new fitting or inference.

Only the fixed representation001 completion marker, three named comparison
reports and their physical-score summary are read. No neural arrays, model
pickles, calibration or reserved outcomes are opened. Rendering belongs in
the coordinator's existing Slurm allocation; both output formats are exclusive.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import socket


EPISODE = Path(__file__).resolve().parents[2]
SCORES = EPISODE / "outputs/runs/residual_fingerprint_development/representation001/scores"
DESTINATION = EPISODE / "outputs/figures/lmp_controls"
TITLE = "EP06: exploratory LMP signal; full study remains unsupported"
ANIMALS = ("Mihili", "Chewie-L")
SESSIONS = (
    ("Mihili", "Mihili_CO_20140217.mat", "Feb 17"),
    ("Mihili", "Mihili_CO_20140218.mat", "Feb 18"),
    ("Mihili", "Mihili_CO_20140304.mat", "Mar 04"),
    ("Chewie-L", "Chewie_CO_20160929.mat", "Sep 29"),
    ("Chewie-L", "Chewie_CO_20161005.mat", "Oct 05"),
    ("Chewie-L", "Chewie_CO_20161007.mat", "Oct 07"),
    ("Chewie-L", "Chewie_CO_20161014.mat", "Oct 14"),
)
COMPARISONS = (
    ("control07lmp", "Original LMP", "original"),
    ("control08selectedunits", "Selected-electrode\nunits", "selected_units"),
    ("control09mismatch", "Fixed trial\nmismatch", "mismatch"),
)
SERIES = (
    ("original_lmp", "Original LMP", "#0072B2", "o"),
    ("mismatched_lmp", "Fixed trial mismatch", "#D55E00", "X"),
    ("elapsed_time", "Linear time", "#747D87", "s"),
)


def _compute_admission():
    job = os.environ.get("SLURM_JOB_ID", "")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if (not job.isdigit() or "-ln" in socket.gethostname()
            or not cpus.isdigit() or int(cpus) < 1):
        raise SystemExit("render in the actual task-owned Slurm compute shell")
    temporary = Path(os.environ.get("L_SCRATCH_JOB", ""))
    if not temporary.is_absolute() or not temporary.is_dir():
        raise SystemExit("existing job-local scratch is required for the plotting cache")
    os.environ["MPLCONFIGDIR"] = str(temporary / "ep06_lmp_matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"


def _read(path):
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("unexpected size for a fixed completed JSON report")
    return json.loads(path.read_text())


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value))


def _load_completed():
    completion_path = SCORES / "complete.json"
    done = _read(completion_path)
    if (not done.get("completed_utc") or done.get("valid_scientific_comparison_count") != 6
            or done.get("valid_control_count") != 4 or done.get("valid_successor_count") != 2
            or done.get("new_neural_fits") != 0 or done.get("new_outcome_reads") != 0
            or done.get("calibration_values_used") is not False
            or done.get("reserved_sessions_opened") is not False
            or done.get("scientific_acceptance") is not None):
        raise PermissionError("the fixed exploratory representation score stage must be complete")
    summaries = {row["candidate_id"]: row for row in done["comparisons"]}
    accuracies, source_paths, reports = [], [completion_path], []
    for identifier, _, source in COMPARISONS:
        path = SCORES / f"{identifier}.json"
        report = _read(path)
        candidate = report["candidate"]
        reference = report["joint_label_reference"]
        values = report["directional_balanced_accuracy"]
        summary = summaries[identifier]
        if (candidate.get("candidate_id") != identifier or candidate.get("source") != source
                or candidate.get("block_indices") != [0]
                or candidate.get("representation") != "signed_mean_only"
                or candidate.get("classifier") != {"method": "l2_logistic", "penalty": 1.}
                or reference.get("complete") is not True or summary.get("reference_complete") is not True
                or summary.get("directional_balanced_accuracy") != values
                or not isinstance(values, list) or len(values) != 2
                or any(not _finite(value) or not 0 <= value <= 1 for value in values)):
            raise ValueError("three complete fixed singleton-LMP classifier comparisons are required")
        directions = reference.get("directions", [])
        if (len(directions) != 2 or any(
                row.get("source_animal") != ANIMALS[index]
                or row.get("target_animal") != ANIMALS[1 - index]
                or row.get("observed_balanced_accuracy") != values[index]
                for index, row in enumerate(directions))):
            raise ValueError("classification directions must retain their source and target animals")
        accuracies.append(tuple(values))
        reports.append(report)
        source_paths.append(path)
    original_signed = reports[0]["signed_agreement"]
    selected_signed = reports[1]["signed_agreement"]
    if (original_signed.get("raw_available") is not True
            or original_signed.get("raw_all_positive") is not True
            or selected_signed.get("raw_available") is not True
            or selected_signed.get("raw_all_positive") is not False):
        raise ValueError("the caption requires the recorded original-pass/selected-unit-fail signed checks")
    physical_path = SCORES / "physical_score_comparisons.json"
    physical = _read(physical_path)
    cells = physical.get("cells", [])
    expected = [(animal, session, region) for animal, session, _ in SESSIONS for region in ("M1", "PMd")]
    if (physical.get("additional_scientific_trial_count") != 0
            or [(cell.get("animal"), cell.get("session"), cell.get("region")) for cell in cells] != expected
            or any(cell.get("block") != {"bands": [[0., 0.]], "name": "lmp"}
                   or any(not _finite(cell.get(key)) for key, _, _, _ in SERIES) for cell in cells)):
        raise ValueError("the physical score summary must retain all seven paired sessions and the LMP block")
    source_paths.append(physical_path)
    return accuracies, cells, source_paths, done["completed_utc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    _compute_admission()
    if args.out_dir.resolve() != DESTINATION:
        raise ValueError("use the fixed EP06 LMP-control figure directory")
    outputs = tuple(args.out_dir / f"ep06_lmp_controls.{suffix}" for suffix in ("png", "pdf"))
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("refuse to overwrite either completed or partial figure format")
    accuracies, cells, source_paths, completed_utc = _load_completed()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 8.2),
                             gridspec_kw={"width_ratios": (1.03, 1.28, 1.28)})
    fig.subplots_adjust(left=.12, right=.975, bottom=.34, top=.77, wspace=.36)
    fig.suptitle(TITLE, x=.525, y=.967, fontsize=16, fontweight="bold")
    fig.text(.525, .918, "Selected development comparisons  |  Seven paired sessions, two animals  |  Reserved outcomes remain closed",
             ha="center", fontsize=11, color="#4F5661")

    classification = axes[0]
    classification.set_title("A  Region classification", loc="left", fontsize=12, fontweight="bold", pad=14)
    classification.axvline(.5, color="#68717C", linestyle="--", linewidth=1.15)
    classification.axhspan(1.62, 2.38, color="#EFF5FA", zorder=0)
    direction_styles = (("Mihili → Chewie-L", "#0072B2", "o", .12),
                        ("Chewie-L → Mihili", "#8C4E97", "^", -.12))
    for row_index, values in enumerate(accuracies):
        y = 2 - row_index
        for direction, (_, color, marker, offset) in enumerate(direction_styles):
            value = values[direction]
            classification.scatter(value, y + offset, s=68, color=color, marker=marker,
                                   edgecolors="white", linewidths=.7, zorder=3)
            classification.annotate(f"{100 * value:.1f}%", (value, y + offset), xytext=(8, 0),
                                    textcoords="offset points", va="center", fontsize=9.5, color=color)
    classification.set_xlim(0, 1.08)
    classification.set_ylim(-.55, 2.55)
    classification.set_yticks((2, 1, 0), [row[1] for row in COMPARISONS])
    classification.tick_params(axis="y", length=0, pad=9)
    classification.set_xticks((0., .25, .5, .75, 1.))
    classification.xaxis.set_major_formatter(PercentFormatter(xmax=1., decimals=0))
    classification.set_xlabel("Balanced accuracy", labelpad=10)
    classification.grid(axis="x", color="#E7EBEF", linewidth=.7)
    classification.set_axisbelow(True)
    direction_handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7, label=label)
                         for label, color, marker, _ in direction_styles]
    classification.legend(handles=direction_handles, loc="upper left", bbox_to_anchor=(-.02, -.19),
                          frameon=False, fontsize=9.3, handletextpad=.55, borderaxespad=0)

    xs = (0, 1, 2, 4, 5, 6, 7)
    all_scores = [cell[key] for cell in cells for key, _, _, _ in SERIES]
    lower, upper = min(0., min(all_scores)), max(0., max(all_scores))
    padding = (upper - lower) * .14 if upper > lower else .01
    for region_index, region in enumerate(("M1", "PMd")):
        ax = axes[region_index + 1]
        region_cells = cells[region_index::2]
        ax.set_title(f"{'BC'[region_index]}  {region}: physical LMP scores", loc="left",
                     fontsize=12, fontweight="bold", pad=14)
        ax.axhspan(lower - padding, 0., color="#F5F6F7", zorder=0)
        ax.axhline(0., color="#6D747D", linewidth=1.05, zorder=1)
        ax.axvline(3., color="#BFC6CE", linestyle=":", linewidth=1.)
        for key, _, color, marker in SERIES:
            values = [cell[key] for cell in region_cells]
            # Connect dates within the same animal only; the central gap is
            # categorical and does not imply a common continuous time axis.
            for start, stop in ((0, 3), (3, 7)):
                ax.plot(xs[start:stop], values[start:stop], color=color, marker=marker,
                        markersize=6, linewidth=1.25, markeredgecolor="white", markeredgewidth=.5, zorder=3)
        ax.set_xlim(-.55, 7.55)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xticks(xs, [row[2] for row in SESSIONS], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Residual-energy recovery, S")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", color="#E3E7EB", linewidth=.7)
        ax.set_axisbelow(True)
        ax.text(1., -.28, "Mihili · 2014", transform=ax.get_xaxis_transform(), ha="center",
                fontsize=9.5, color="#404751")
        ax.text(5.5, -.28, "Chewie-L · 2016", transform=ax.get_xaxis_transform(), ha="center",
                fontsize=9.5, color="#404751")
    for ax in axes:
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        ax.spines["left"].set_color("#B2B8C0")
        ax.spines["bottom"].set_color("#B2B8C0")
    classification.spines["left"].set_visible(False)
    series_handles = [Line2D([], [], color=color, marker=marker, linewidth=1.25, markersize=6, label=label)
                      for _, label, color, marker in SERIES]
    fig.legend(handles=series_handles, loc="center", bbox_to_anchor=(.678, .835), ncol=3,
               frameon=False, fontsize=10, columnspacing=1.7)

    fig.text(.12, .175,
        "Original LMP passes raw signed-profile agreement, including all seven signed-profile omissions; selected-unit LMP fails.",
        fontsize=10.5, color="#333C46", fontweight="bold")
    caption = (
        "LMP was chosen after 20 development comparisons. This is exploratory reuse, not independent confirmation or evidence for a multi-band fingerprint.\n"
        "Selected units change the population basis. The mismatch is one fixed within-direction whole-trial permutation, not a p-value or a complete mechanism test.\n"
        "Panels B–C use the original population basis for LMP, mismatch and linear time; S retains negative values. Paired regions and days are not additional animals.\n"
        "Panel A's dashed 50% line is a chance reference, not a significance threshold. No new confidence intervals, tests, fits or scientific comparisons were added.")
    fig.text(.12, .134, caption, va="top", fontsize=8.8, linespacing=1.5, color="#505862")
    fig.text(.12, .035, "Source: residual_fingerprint_development/representation001/scores — completed, fixed comparison reports.",
             fontsize=8.3, color="#68717B")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    description = TITLE + "\n" + caption + "\n" + "\n".join(str(path) for path in source_paths)
    description += f"\nRepresentation score stage completed: {completed_utc}"
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("xb")) for path in outputs]
        fig.savefig(handles[0], format="png", dpi=220, facecolor="white",
                    metadata={"Title": TITLE, "Description": description})
        fig.savefig(handles[1], format="pdf", facecolor="white",
                    metadata={"Title": TITLE, "Subject": description, "Creator": "EP06 completed-report plot"})
    plt.close(fig)
    print(json.dumps({"state": "completed_lmp_figure_written", "outputs": [str(path) for path in outputs],
                      "source_reports": [str(path) for path in source_paths], "new_inference": False}), flush=True)


if __name__ == "__main__":
    main()
