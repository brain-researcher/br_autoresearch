#!/usr/bin/env python3
"""Plot a completed acquired-only report; no source recordings or model fits."""
import argparse
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output-stem", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use coordinator-owned compute for figure rendering")
    report = json.loads(a.report.read_text())
    if report["status"] != "completed_one_acquired_only_comparison_with_random_reference":
        raise ValueError("The complete acquired-only comparison is required")
    destinations = [a.output_stem.with_suffix("."+extension) for extension in ("png", "pdf")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserve existing figures")
    animals, counts, totals = ("Chewie", "Mihili"), (8, 16, 32, 96), (32, 48, 64, 80, 96)
    branches = ("initial_marginal", "fixed_random")
    colors = {"initial_marginal": "#285a90", "fixed_random": "#ad5b2d"}
    labels = {"initial_marginal": "Select after 32-trial screen", "fixed_random": "Fix random set before trial 1"}
    sessions = report["sessions"]
    if len(sessions) != 8 or any(sum(s["animal"] == animal for s in sessions) != 4 for animal in animals):
        raise ValueError("The two complete four-session animal panels are required")
    index = {}
    for session in sessions:
        for branch in branches:
            cells = session["panels"][branch]["cells"]
            values = {(c["training_trials"], c["electrodes"]): c for c in cells}
            if len(cells) != 20 or set(values) != {(n, e) for n in totals for e in counts}:
                raise ValueError("Preserve the complete20-cell protocol")
            index[session["filename"], branch] = values

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex=True)
    for row, animal in enumerate(animals):
        selected = [s for s in sessions if s["animal"] == animal]
        row_values = []
        for column, count in enumerate(counts):
            ax = axes[row, column]
            for branch in branches:
                values = np.asarray([[index[s["filename"], branch][n, count]["Q"] for n in totals] for s in selected])
                row_values.extend(values.ravel().tolist())
                for trace in values:
                    ax.plot(totals, trace, color=colors[branch], linewidth=.7, alpha=.18)
                ax.plot(totals, values.mean(axis=0), color=colors[branch], marker="o" if branch == "initial_marginal" else "s",
                        markersize=4, linewidth=2, linestyle="-" if branch == "initial_marginal" else "--",
                        label=labels[branch])
            ax.grid(axis="y", alpha=.15)
            ax.set_xticks(totals)
            if row == 0:
                ax.set_title(f"{count} retained electrodes", loc="left", fontsize=11)
            if column == 0:
                ax.set_ylabel(f"{animal}\nQ")
        low, high = min(row_values), max(row_values)
        margin = max(.005, .08*(high-low))
        for ax in axes[row]:
            ax.set_ylim(low-margin, high+margin)

    # Exposure is taken directly from completed policy records, not inferred
    # from feature IO totals (which also contain reuse and duplicate reads).
    exposure_maximum = 0.
    for column, count in enumerate(counts):
        ax = axes[2, column]
        for branch in branches:
            vectors = [[index[s["filename"], branch][n, count]["LFP_numeric_elements"] for n in totals] for s in sessions]
            if any(v != vectors[0] for v in vectors[1:]):
                raise ValueError("Common protocol exposure differs across sessions")
            values = np.asarray(vectors[0], dtype=float)/1000
            exposure_maximum = max(exposure_maximum, float(values.max()))
            ax.plot(totals, values, color=colors[branch], marker="o" if branch == "initial_marginal" else "s",
                    markersize=4, linewidth=2, linestyle="-" if branch == "initial_marginal" else "--")
        ax.set_xlabel("Acquired trials")
        ax.set_xticks(totals)
        ax.grid(axis="y", alpha=.15)
        if column == 0:
            ax.set_ylabel("LFP scalar observations\n(thousands)")
    for ax in axes[2]:
        ax.set_ylim(0, exposure_maximum*1.06)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(.065, .935), frameon=False, ncol=2)
    fig.suptitle("EP08 · Early electrode selection and its observation cost", x=.065, ha="left", fontsize=17)
    fig.text(.065, .944, "Development only · fixed electrode sets after initialization · four held sessions remain closed", fontsize=10)
    fig.text(.065, .077, "Bold: four-session animal means; faint: individual sessions. Random curves average the five scarce-set replicates. No confidence intervals.", fontsize=9)
    fig.text(.065, .050, "Selection observes all 96 electrodes for 32 initial trials; random observes only its set. Each electrode-trial contributes 3 bands × 10 bins.", fontsize=9)
    fig.text(.065, .023, "This is the separate 20-cell acquired-only protocol, not a continuation of the earlier 12-cell ranking. Initial observation costs are unequal.", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.14, hspace=.25, wspace=.22)
    a.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for path in destinations:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"source": str(a.report), "figures": [str(path) for path in destinations],
                      "source_data_or_model_fits": False, "sessions_per_animal": 4}))


if __name__ == "__main__":
    main()
