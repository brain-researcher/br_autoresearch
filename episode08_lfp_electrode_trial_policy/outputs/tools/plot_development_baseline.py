#!/usr/bin/env python3
"""Plot the completed development baseline; never open source recordings."""
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
    raise RuntimeError("Plot in the coordinator's compute allocation")
root = Path(__file__).resolve().parents[1]
report_path = root / "runs/simple_comparisons/development_baseline_attempt001/development_baseline.json"
report = json.loads(report_path.read_text())
assert report["status"] == "completed_one_development_baseline_no_scientific_winner"
destination = root / "figures"
destination.mkdir(exist_ok=True)
outputs = [destination / f"development_baseline_attempt001.{suffix}" for suffix in ("png", "pdf")]
if any(path.exists() for path in outputs):
    raise FileExistsError("Preserve prior figure outputs")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True)
electrodes = [8, 16, 32, 96]
colors = {"Mihili": "#285a90", "Chewie": "#b45b2b"}
for ax, trial_count in zip(axes, (32, 64, 96)):
    for animal, color in colors.items():
        values = []
        for session in report["sessions"]:
            if session["animal"] != animal:
                continue
            by_count = {cell["electrodes"]: cell["mean_Q"] for cell in session["cells"]
                        if cell["training_trials"] == trial_count}
            row = [by_count[count] for count in electrodes]
            values.append(row)
            ax.plot(electrodes, row, color=color, linewidth=.8, alpha=.25)
        assert len(values) == 4
        ax.plot(electrodes, np.mean(values, axis=0), "o-", color=color,
                linewidth=2, markersize=5, label=f"{animal}: 4-session mean")
    ax.axhline(0, color="#999999", linewidth=.7, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(electrodes, [str(count) for count in electrodes])
    ax.set_xlabel("Retained physical electrodes")
    ax.set_title(f"{trial_count} balanced training trials")
    ax.grid(axis="y", alpha=.16)
axes[0].set_ylabel("Predictive Q vs training-mean null")
fig.suptitle("EP08 · First random-subset development baseline", x=.06, ha="left", fontsize=16)
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.055, .11), ncol=2, frameon=False)
fig.text(.06, .075, "Thin lines: individual sessions. Subset replicates averaged within each cell; no uncertainty intervals.", fontsize=9)
fig.text(.06, .035, "8 development sessions · 1 method evaluation · 40 evaluation trials/session · 4 held sessions unopened", fontsize=9)
fig.subplots_adjust(left=.065, right=.985, top=.83, bottom=.30, wspace=.16)
for path in outputs:
    fig.savefig(path, dpi=180, facecolor="white")
plt.close(fig)
print(json.dumps({"source": str(report_path), "figures": [str(path) for path in outputs]}))
