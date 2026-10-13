"""Plot the already published Task-A aggregates; never open neural inputs."""
import argparse
import json
import os
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinated compute allocation")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.diagnostic.read_text())
    rows = report["condition_summaries"]
    if len(rows) != 29 or report["ranges_are_confidence_intervals"] is not False:
        raise ValueError("Complete descriptive diagnostic required")
    outputs = [args.output_prefix.with_suffix(suffix) for suffix in (".png", ".pdf")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve existing figures")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 10.5), sharey=True)
    for axis, key, color, title in zip(axes, ("S", "N"), ("#147D92", "#B05A37"),
            ("S: between-session difference covariance", "N: within-session difference covariance")):
        for index, row in enumerate(rows):
            lower, upper = row[key + "_delete_one_range"]
            axis.plot([lower, upper], [index, index], color=color, alpha=.55, linewidth=2)
            axis.plot(row[key], index, "o", color=color, markersize=4)
        axis.axvline(0, color="#555555", linewidth=.8, linestyle="--")
        axis.set_title(title, fontsize=11, loc="left", pad=14)
        axis.set_xlabel("Covariance on the source beta scale squared")
        axis.set_xlim(-.28, .51)
        axis.set_yticks(range(len(rows)))
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#E6E6E6", linewidth=.6)
        axis.set_axisbelow(True)
    axes[0].set_yticklabels([row["condition"] for row in rows], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("EP15 | Real Task-A partition diagnostic", x=.03, ha="left", fontsize=17)
    fig.text(.03, .938, "12 development participants · 29 conditions · 208 fixed cerebellar locations", fontsize=11)
    fig.text(.03, .054,
             "Dots: full-cohort estimates. Lines: range after deleting each participant; these are not confidence intervals.\n"
             "P1–P4 contain runs 1–4, 5–8, 9–12, 13–16. S = U(P1−P3, P2−P4); N = U(P1−P2, P3−P4).\n"
             "Session interpretation follows documented run ordering. Physiology and shared nuisance are not separated.\n"
             "Source: task_a_error_support/attempt002/diagnostic.json. Task B and audit data were not opened.",
             fontsize=9, va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=.17, right=.98, top=.885, bottom=.16, wspace=.17)
    for path in outputs:
        fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(json.dumps({"conditions": len(rows), "S_positive": sum(row["S"] > 0 for row in rows),
        "S_positive_for_every_single_person_deletion": sum(row["S_delete_one_range"][0] > 0 for row in rows),
        "N_positive": sum(row["N"] > 0 for row in rows),
        "N_negative": sum(row["N"] < 0 for row in rows),
        "outputs": [str(path) for path in outputs], "ranges_are_confidence_intervals": False}))


if __name__ == "__main__":
    main()
