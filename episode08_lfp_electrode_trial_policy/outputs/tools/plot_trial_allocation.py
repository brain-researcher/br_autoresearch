#!/usr/bin/env python3
"""Render complete allocator report only; no neural source or refits."""
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
        raise RuntimeError("Render inside the coordinator's compute allocation")
    report = json.loads(a.report.read_text())
    if report["status"] != "completed_two_trial_allocators_with_balanced_reference":
        raise ValueError("Complete allocator report required")
    files = [a.output_stem.with_suffix("."+extension) for extension in ("png", "pdf")]
    if any(path.exists() for path in files):
        raise FileExistsError("Preserve prior figures")
    sessions = sorted(report["sessions"], key=lambda row: (row["animal"], row["filename"]))
    if len(sessions) != 8 or any(sum(row["animal"] == animal for row in sessions) != 4 for animal in ("Chewie", "Mihili")):
        raise ValueError("Both complete four-session animal panels required")
    colors = {"mean_value": "#286d91", "mean_error": "#b96836"}
    labels = {"mean_value": "Estimated reducible mean error", "mean_error": "Estimated total mean error"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    for index, row in enumerate(sessions):
        values = [row["panels"][policy]["intermediate_mean_delta_Q_common_from_balanced"] for policy in colors]
        ax.plot(values, [index, index], color=".72", linewidth=1, zorder=1)
        for value, policy in zip(values, colors):
            ax.scatter(value, index, color=colors[policy], s=34, zorder=2)
    ax.set_yticks(range(8), [row["filename"].replace("_CO_", " ").replace(".mat", "") for row in sessions])
    ax.invert_yaxis()
    ax.axvline(0, color=".4", linewidth=.8)
    ax.axhline(3.5, color=".85", linewidth=.8)
    ax.set_title("Paired session effects", loc="left")
    ax.set_xlabel("Intermediate mean ΔQ_common vs balanced")
    ax.grid(axis="x", alpha=.15)
    limits = []
    for ax, animal in zip(axes[1:], ("Chewie", "Mihili")):
        people = [row for row in sessions if row["animal"] == animal]
        for policy in colors:
            values = []
            for row in people:
                cells = row["panels"][policy]["cells"]
                if len(cells) != 20:
                    raise ValueError("All twenty frozen resource cells required")
                values.append([np.mean([cell["delta_Q_common_from_balanced"] for cell in cells
                                        if cell["training_trials"] == total]) for total in (48, 64, 80)])
            values = np.asarray(values)
            for trace in values:
                ax.plot((48, 64, 80), trace, color=colors[policy], linewidth=.7, alpha=.18)
            means = values.mean(axis=0)
            ax.plot((48, 64, 80), means, color=colors[policy], marker="o", linewidth=2, label=labels[policy])
            limits.extend(values.ravel().tolist())
        ax.axhline(0, color=".4", linewidth=.8)
        ax.set_xticks((48, 64, 80))
        ax.set_xlabel("Acquired trials")
        ax.set_title(animal, loc="left")
        ax.grid(axis="y", alpha=.15)
    low, high = min(limits+[0]), max(limits+[0])
    margin = max(.0005, .08*(high-low))
    for ax in axes[1:]:
        ax.set_ylim(low-margin, high+margin)
    axes[1].set_ylabel("ΔQ_common vs balanced")
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(.17, .92), frameon=False, ncol=2)
    fig.suptitle("EP08 · Which next trial is useful?", x=.17, ha="left", fontsize=17)
    fig.text(.17, .928, "Development only · fixed random electrodes · common initializer scoring · equal observations at each E,n", fontsize=10)
    fig.text(.17, .10, "Intermediate = 48/64/80 trials × four electrode counts. Initial32 and final96 models are shared; their contrasts are exactly zero.", fontsize=9)
    fig.text(.17, .062, "Left: each session's paired policy effects. Right: four-session animal means and faint session traces, averaging electrode counts. No CIs.", fontsize=9)
    fig.text(.17, .025, "Utilities estimate direction-mean uncertainty/error, not full-decoder information gain. Four held sessions remain closed.", fontsize=9)
    fig.subplots_adjust(left=.17, right=.99, bottom=.24, top=.81, wspace=.42)
    a.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for file in files:
        fig.savefig(file, dpi=180, facecolor="white")
    plt.close(fig)
    print(json.dumps({"report": str(a.report), "figures": [str(file) for file in files], "new_fits": False}))


if __name__ == "__main__":
    main()
