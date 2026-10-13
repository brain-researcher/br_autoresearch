#!/usr/bin/env python3
"""Plot only the completed scalar and three multichannel summary artifacts."""
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render in the coordinated compute allocation")
    root = Path(__file__).resolve().parents[1]
    scalar = json.loads((root/"runs/adaptive_search/primary_reference_summary_attempt001/primary_reference.json").read_text())
    multi = json.loads((root/"runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json").read_text())
    if multi["status"] != "complete_three_multichannel_comparisons":
        raise ValueError("All sixteen sessions and all three recipes must be complete")
    if scalar["status"] != "complete_first_primary_reference":
        raise ValueError("The complete original scalar comparison is required")
    panels = [scalar]+multi["panels"]
    if any(len(panel["sessions"]) != 16 for panel in panels):
        raise ValueError("Complete fixed session support required")
    sessions = [row["session_id"] for row in panels[0]["sessions"]]
    if any([row["session_id"] for row in panel["sessions"]] != sessions for panel in panels):
        raise ValueError("The same session order is required")
    colors = {"Mihili": "#276f94", "Chewie": "#bb6632"}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    x = np.arange(4)
    specs = [("delta_R2", "primary_increment", "primary_by_animal", .01,
              "Improvement over direction/time baseline"),
             ("identity_increment", "identity_increment", "identity_by_animal", .005,
              "Correct-trial versus wrong-trial pairing")]
    for ax, (session_key, overall_key, animal_key, margin, title) in zip(axes, specs):
        for animal, color in colors.items():
            for index, row in enumerate(panels[0]["sessions"]):
                if row["animal"] == animal:
                    ax.plot(x, [p["sessions"][index][session_key] for p in panels], color=color,
                            alpha=.24, lw=.8, marker="o", ms=3)
            ax.plot(x, [p[animal_key][animal] for p in panels], color=color, lw=2.5,
                    marker="o", ms=6, label=animal+" session mean")
        ax.plot(x, [p[overall_key] for p in panels], color="#202020", marker="D", ms=6,
                lw=1.4, linestyle=":", label="Equal-animal mean")
        ax.axhline(0, color=".45", lw=.8)
        ax.axhline(margin, color="#754c8c", lw=1, linestyle="--", label=f"Operating margin {margin:g}")
        ax.set_xticks(x, ["Spatial mean\nLMP", "Full-channel\nLMP", "Full-channel\n0.5–12 Hz", "Full-channel\n100–400 Hz"])
        ax.set_ylabel("Increment in joint X/Y trajectory R²")
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.14)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("EP05 · Do electrode contrasts preserve useful trial information?", x=.065,
                 y=.975, ha="left", fontsize=17)
    fig.text(.065, .92, "Exposed development only · 16 sessions, 2 animals · held-direction fits and outer-session penalty selection", fontsize=10.5)
    fig.text(.065, .10, "Faint lines: individual sessions. Solid lines: within-animal session means. Diamonds: equal weight to the two animals.", fontsize=9.5)
    fig.text(.065, .065, "Feature widths differ; no matched-capacity spectral claim. Operating margins alone cannot promote a candidate. No confidence intervals.", fontsize=9.5)
    fig.text(.065, .03, "Source: completed primary_reference_summary_attempt001 and primary_multichannel_summary_attempt001; no new fits or source reads.", fontsize=9)
    fig.subplots_adjust(left=.065, right=.98, top=.835, bottom=.23, wspace=.24)
    target = root/"figures/primary_multichannel_attempt001"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    print(target.with_suffix(".png"))


if __name__ == "__main__":
    main()
