"""Plot completed EP06 development comparisons; no new fitting or inference.

Read only the named initial/successor/control JSON reports after completed
score-stage admission. Display existing two-direction accuracies and one
explicitly named session omission. No neural data, model state, calibration
values, reserved dates, resampling or statistical tests are accessed. Run in
the coordinator's existing Slurm allocation. PNG/PDF outputs are exclusive.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import socket


EPISODE = Path(__file__).resolve().parents[2]
RUN = EPISODE / "outputs/runs/residual_fingerprint_development"
ADAPTIVE = RUN / "adaptive001/scores"
CONTROLS = RUN / "controls001/scores"
DESTINATION = EPISODE / "outputs/figures/development_controls"
TITLE = "EP06: the three candidates fail signed-profile stability"
ANIMALS = ("Mihili", "Chewie-L")


def _compute_admission():
    job = os.environ.get("SLURM_JOB_ID", "")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if (not job.isdigit() or "-ln" in socket.gethostname()
            or not cpus.isdigit() or int(cpus) < 1):
        raise SystemExit("render in the actual task-owned Slurm compute shell")
    temporary = Path(os.environ.get("L_SCRATCH_JOB", ""))
    if not temporary.is_absolute() or not temporary.is_dir():
        raise SystemExit("existing job-local scratch is required for the plotting cache")
    os.environ["MPLCONFIGDIR"] = str(temporary / "ep06_controls_matplotlib")
    os.environ["MPLBACKEND"] = "Agg"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"


def _read(path):
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("unexpected size for a fixed small completed JSON report")
    return json.loads(path.read_text())


def _accuracy_pair(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values)):
        raise ValueError("two complete finite reported balanced accuracies are required")
    return tuple(float(value) for value in values)


def _directions(rows):
    if (not isinstance(rows, list) or len(rows) != 2
            or any(row.get("source_animal") != ANIMALS[index]
                   or row.get("target_animal") != ANIMALS[1 - index]
                   for index, row in enumerate(rows))):
        raise ValueError("direction order must be Mihili-to-Chewie then Chewie-to-Mihili")
    return _accuracy_pair([row["balanced_accuracy"] for row in rows])


def _load_completed():
    # Completion markers precede all newly admitted scientific report reads.
    adaptive_done = _read(ADAPTIVE / "complete.json")
    control_done = _read(CONTROLS / "complete.json")
    expected_controls = ("control00_metadata", "control01_leaveout", "control02_band", "control03_mismatch")
    if (adaptive_done.get("fit_cells") != 28 or adaptive_done.get("scored_cells") != 28
            or [row.get("candidate_id") for row in adaptive_done.get("candidates", [])] != ["adaptive00", "adaptive01"]
            or not adaptive_done.get("completed_utc") or not control_done.get("completed_utc")
            or control_done.get("valid_scientific_control_count") != 4
            or control_done.get("available_controls") != dict.fromkeys(expected_controls, True)
            or control_done.get("calibration_values_used") is not False
            or control_done.get("reserved_sessions_opened") is not False):
        raise PermissionError("both successor and all four control score stages must be complete")
    paths = (RUN / "starting001/scores/starting_10.json", ADAPTIVE / "adaptive00.json",
             ADAPTIVE / "adaptive01.json", CONTROLS / "control00_metadata.json",
             CONTROLS / "profile_controls.json", CONTROLS / "control03_mismatch.json")
    original, ridge, regularized, metadata, profiles, mismatch = (_read(path) for path in paths)
    for name, report in (("starting_10", original), ("adaptive00", ridge), ("adaptive01", regularized)):
        signed = report["signed_agreement"]
        if (report["candidate"].get("candidate_id") != name
                or report["joint_label_reference"].get("complete") is not True
                or signed.get("raw_available") is not True or signed.get("raw_all_positive") is not False):
            raise ValueError("this figure describes the three complete candidates with failed signed stability")
    if any(report["comparison_to_starting10"].get("numerical_improvement_pass") is not False
           for report in (ridge, regularized)):
        raise ValueError("the reported successor decisions differ from the figure's interpretation")
    band, leaveout = profiles["band_permutation"], profiles["leave_one_out"]
    if (metadata.get("status") != "complete" or band.get("complete") is not True
            or leaveout.get("complete") is not True or len(leaveout.get("omissions", [])) != 7
            or mismatch.get("control_id") != "control03_mismatch"
            or mismatch["joint_label_reference"].get("complete") is not True):
        raise ValueError("complete fixed metadata, omission and permutation results are required")
    original_ba = _accuracy_pair(original["directional_balanced_accuracy"])
    mismatch_ba = _accuracy_pair(mismatch["directional_balanced_accuracy"])
    if mismatch_ba != original_ba or mismatch.get("directional_accuracy_change") != [0., 0.]:
        raise ValueError("this figure's unchanged-accuracy annotation requires the reported equality")
    named = [row for row in leaveout["omissions"]
             if row.get("omitted_animal") == "Mihili" and row.get("omitted_session") == "Mihili_CO_20140304.mat"]
    if len(named) != 1 or named[0].get("complete") is not True:
        raise ValueError("the complete named March 4 omission is required")
    omission = _directions(named[0]["directions"])
    rows = (
        ("Original low-band CCA", original_ba, "#0072B2", "o"),
        ("Successor: two-coordinate ridge", _accuracy_pair(ridge["directional_balanced_accuracy"]), "#4B91BB", "o"),
        ("Successor: CCA regularization 0.1", _accuracy_pair(regularized["directional_balanced_accuracy"]), "#4B91BB", "o"),
        ("Recording/support metadata", _directions(metadata["directions"]), "#626B75", "s"),
        ("Paired band-label permutation", _directions(band["directions"]), "#D55E00", "D"),
        ("Within-direction trial mismatch", mismatch_ba, "#A85A00", "D"),
    )
    return rows, omission, paths, control_done["completed_utc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    _compute_admission()
    if args.out_dir.resolve() != DESTINATION:
        raise ValueError("use the fixed EP06 development-control figure directory")
    outputs = tuple(args.out_dir / f"ep06_development_controls.{suffix}" for suffix in ("png", "pdf"))
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("refuse to overwrite either completed or partial figure format")
    rows, omission, source_paths, completed_utc = _load_completed()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.305, right=.975, bottom=.31, top=.79, wspace=.16)
    fig.suptitle(TITLE, x=.515, y=.975, fontsize=16, fontweight="bold")
    fig.text(.515, .929, "Exploratory development  |  Seven sessions, two animals  |  Four reserved sessions remain closed",
             ha="center", fontsize=10.5, color="#4F5661")
    fig.text(.515, .878, "Neither successor improves the original; region-classification accuracy survives the trial mismatch.",
             ha="center", fontsize=10.5, color="#333B45")
    ys = tuple(range(5, -1, -1))
    for direction, ax in enumerate(axes):
        ax.set_title(("Mihili → Chewie-L\n4 target sessions", "Chewie-L → Mihili\n3 target sessions")[direction],
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlim(0., 1.045)
        ax.set_ylim(-.6, 5.6)
        for y in (5, 0):
            ax.axhspan(y - .38, y + .38, color="#EEF5FA", zorder=0)
        ax.axvline(.5, color="#68717C", linestyle="--", linewidth=1.2, zorder=1)
        for y, (_, values, color, marker) in zip(ys, rows):
            value = values[direction]
            ax.scatter([value], [y], s=68, color=color, marker=marker,
                       edgecolors="white", linewidths=.8, zorder=3)
            ax.annotate(f"{100 * value:.1f}%", (value, y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=10, color=color, fontweight="bold")
        ax.set_yticks(ys, [row[0] for row in rows])
        ax.tick_params(axis="y", length=0, pad=11, labelsize=10)
        ax.set_xticks((0., .25, .5, .75, 1.))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1., decimals=0))
        ax.set_xlabel("Reported balanced accuracy", labelpad=9)
        ax.grid(axis="x", color="#E5E9ED", linewidth=.7)
        ax.set_axisbelow(True)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color("#A0A6AD")
    fig.text(.305, .223,
        f"Source-session influence: omit Mihili 2014-03-04 → Mihili-to-Chewie accuracy {100 * omission[0]:.1f}%\n"
        f"(original {100 * rows[0][1][0]:.1f}%; the same four Chewie target sessions).",
        va="top", fontsize=10, color="#7A3D00", linespacing=1.5)
    caption = (
        "Dashed line: 50% chance reference, not a significance threshold. Points are existing estimates; no new intervals or tests.\n"
        "One fixed band permutation and one fixed within-direction trial mismatch are sensitivities, not p-values.\n"
        "Metadata scores below the original do not rule out recording confounds; unchanged accuracy does not establish unchanged recovery.\n"
        "Sources: residual_fingerprint_development/{starting001, adaptive001, controls001}/scores (completed development reports).")
    fig.text(.305, .115, caption, va="top", fontsize=8.3, linespacing=1.5, color="#505862")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    description = TITLE + "\n" + caption + "\n" + "\n".join(str(path) for path in source_paths)
    description += f"\nControl score stage completed: {completed_utc}"
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("xb")) for path in outputs]
        fig.savefig(handles[0], format="png", dpi=220, facecolor="white",
                    metadata={"Title": TITLE, "Description": description})
        fig.savefig(handles[1], format="pdf", facecolor="white",
                    metadata={"Title": TITLE, "Subject": description, "Creator": "EP06 completed-report plot"})
    plt.close(fig)
    print(json.dumps({"state": "completed_development_figure_written", "outputs": [str(path) for path in outputs],
                      "source_reports": [str(path) for path in source_paths], "new_inference": False}), flush=True)


if __name__ == "__main__":
    main()
