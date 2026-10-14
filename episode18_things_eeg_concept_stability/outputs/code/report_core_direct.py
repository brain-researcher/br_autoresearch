"""Report the complete registered 30-participant G/T calibration only.

This downstream consumer requires every original development participant and
all six outer folds. It never opens images, raw EEG, concept/false scores or
audit data. Intervals are descriptive two-sided 95% participant t intervals;
they do not replace the policy's confirmatory families or positive-control
thresholds. Execute only inside an allocated compute job.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

import production_baseline_controller as controller
import production_core_reference as core
from production_core_direct import atomic_json


def report(initialized_root, run_root, output_root, figure_root):
    controller.require_compute()
    settings = controller._json(Path(initialized_root)/"settings.json", 32*1024**2)
    people = tuple(settings["participants"])
    core.require(len(people) == 30 and len(set(people)) == 30 and people == tuple(sorted(people)),
                 "exact_frozen_30_development_participants_required")
    settings_id = controller.digest(settings)
    # Refuse an incomplete cohort before opening any participant loss record.
    paths = [Path(run_root)/person/"complete.json" for person in people]
    core.require(all(path.is_file() for path in paths), "all30_complete_records_required")
    values = []
    for person, path in zip(people, paths):
        row = controller._json(path, 16*1024**2)
        core.require(row["participant_id"] == person and row["settings_id"] == settings_id
            and row["status"] == "complete_core_reference_person" and row["inner_splits"] == 30
            and row["outer_splits"] == 6 and row["models"] == list(core.CORE_MODELS)
            and row["audit_access"] is False and row["concept_or_false_scores_computed"] is False
            and row["backend"] == "weighted_gram_cholesky_v1"
            and [fold["outer_index"] for fold in row["outer"]] == list(range(6)),
            "complete_identical_core_reference_required")
        losses = {name: float(np.mean([fold["normalized_losses"][name] for fold in row["outer"]]))
                  for name in core.CORE_MODELS}
        core.require(all(np.isfinite(value) and value >= 0
            and value == row["normalized_participant_losses"][name] for name, value in losses.items()),
            "complete_finite_six_fold_participant_mean_required")
        values.append((losses["nuisance"]-losses["image"],
                       losses["reduced_task_nuisance"]-losses["nuisance"]))
    data = np.asarray(values)
    means = data.mean(axis=0)
    half_width = t.ppf(.975, 29)*data.std(axis=0, ddof=1)/np.sqrt(30)
    record = {"status": "complete_30_person_core_calibration", "participants": people,
        "participant_count": 30, "inner_splits": 900, "outer_splits": 180,
        "participant_G_T": dict(zip(people, data.tolist())), "G": float(means[0]), "T": float(means[1]),
        "interval_method": "descriptive_two_sided_95_percent_participant_t_df29",
        "intervals_G_T": np.column_stack((means-half_width, means+half_width)).tolist(),
        "delta_C": max(.001, .05*means[0]), "delta_P": max(.0005, .025*means[0]),
        "delta_Q": max(.002, .25*means[0]), "delta_T": max(.001, .25*means[1]),
        "image_gain_positive": bool(means[0] > 0), "reference_underidentified": bool(means[0] <= 0),
        "scientific_role": "registered_development_core_calibration",
        "concept_or_false_scores_computed": False, "audit_access": False,
        "evaluation_separation": "procedural_same_account_not_technically_enforced",
        "job_id": os.environ["SLURM_JOB_ID"]}
    output, figures = Path(output_root), Path(figure_root)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    figures.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(output/"core_reference.complete.json", record)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), layout="constrained")
    labels = ("Image prediction gain G", "Target/response nuisance gain T")
    for index, ax in enumerate(axes):
        ax.axvline(0, color=".65", linewidth=.8)
        ax.scatter(data[:, index], np.linspace(-.19, .19, 30), color="#5c789a", s=20, alpha=.75,
                   label="30 independent participants")
        ax.errorbar(means[index], .42, xerr=half_width[index], fmt="o", color="#191919",
                    capsize=5, linewidth=2, label="Mean and 95% participant interval")
        ax.set(title=labels[index], xlabel="Reduction in normalized held-out loss",
               yticks=[], ylim=(-.32, .65))
        ax.spines[["left", "right", "top"]].set_visible(False)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("EP18: registered core calibration on real development EEG", fontsize=11)
    fig.savefig(figures/"core_reference_G_T.png", dpi=180)
    fig.savefig(figures/"core_reference_G_T.pdf")
    plt.close(fig)
    summary = (
        f"Across 30 development participants, image gain G = {means[0]:.6g} "
        f"(95% CI {means[0]-half_width[0]:.6g} to {means[0]+half_width[0]:.6g}); "
        f"task-nuisance gain T = {means[1]:.6g} (95% CI {means[1]-half_width[1]:.6g} to {means[1]+half_width[1]:.6g}).\n"
        "Each participant's six outer folds were averaged before descriptive participant-level t intervals; evaluation separation is procedural within one account.\n"
        "These estimates calibrate the registered reference and margins; they do not establish a concept-specific advantage, complete the matched-control comparison, or evaluate the closed audit.\n")
    (output/"summary.md").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initialized-root", "run-root", "output-root", "figure-root"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    report(args.initialized_root, args.run_root, args.output_root, args.figure_root)
