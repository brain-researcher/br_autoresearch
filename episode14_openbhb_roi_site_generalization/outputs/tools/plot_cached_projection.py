#!/usr/bin/env python3
"""Plot the completed descriptive cached-prediction report; no model fitting."""
import json
import os
from pathlib import Path
import socket

if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
    raise RuntimeError("Run inside the coordinator's Slurm compute shell")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
run = root / "runs/cached_projection_diagnostics/attempt001"
report = json.loads((run / "cached_projection_report.json").read_text())
names = ["Desikan", "Destrieux"]
values = [report["atlases"][name.lower()]["pooled"]["external"]["contrast"]
          ["R3_minus_R0_domain_macro"] for name in names]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={"wspace": .35})
x = np.arange(2)
colors = ["#24778c", "#d18432", "#536174"]
panels = [
    (["mae", "centered_mae"], ["Original age error", "After domain centering"],
     "Age prediction error", "Change in MAE (years)"),
    (["mean_residual_squared", "centered_mse", "mse"],
     ["Domain offset squared", "Within-domain variance", "Total"],
     "Exact squared-error decomposition", "Change in MSE component (years²)"),
]
for ax, (keys, labels, title, ylabel) in zip(axes, panels):
    width = .24
    for j, (key, label) in enumerate(zip(keys, labels)):
        offset = (j-(len(keys)-1)/2)*width
        bars = ax.bar(x+offset, [v[key] for v in values], width,
                      label=label, color=colors[j])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axhline(0, color="#5b6570", linewidth=.8)
    ax.set_xticks(x, names)
    ax.set_title(title, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel)
    ax.margins(y=.24)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.12), fontsize=9)
fig.suptitle("Site-direction removal: similar within-domain harm, different offsets",
             x=.08, ha="left", fontsize=14, fontweight="bold")
fig.text(.08, .018,
         "Exposed development data • 13 external domains • R3 minus R0 • Point estimates only\n"
         "Centering uses evaluation ages for diagnosis, not a deployable correction. MAE components are not additive.",
         fontsize=9, color="#50545a")
fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.28)
fig.savefig(run / "cached_projection_diagnostics.png", dpi=180)
fig.savefig(run / "cached_projection_diagnostics.svg")
plt.close(fig)
print(run / "cached_projection_diagnostics.png")
