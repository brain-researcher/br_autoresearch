#!/usr/bin/env python3
"""Plot the fixed covariance-protection contrasts from completed results only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_development as legacy

PAIRS = (("original", "raw"), ("covariance_protected", "raw"),
         ("covariance_protected", "original"), ("sham_protected", "original"),
         ("covariance_protected", "sham_protected"))
LABELS = ("Original − raw", "Protected − raw", "Protected − original",
          "Sham protected − original", "Protected − sham protected")
COLORS = {"desikan": "#176B87", "destrieux": "#A34F31"}
OUTPUTS = Path(__file__).resolve().parents[1]


def draw_panel(ax, panel, atlas, title, is_site=False):
    for position, (a, b) in enumerate(PAIRS):
        key = f"{a}_minus_{b}" if is_site else f"{a}_alpha1000_minus_{b}_alpha1000"
        item = panel["contrasts"].get(key)
        if item is None:
            if panel.get("status") != "insufficient_support":
                raise ValueError(f"Missing prespecified contrast: {atlas}/{title}/{key}")
            ax.text(.03, position, "Insufficient support", transform=ax.get_yaxis_transform(), fontsize=9)
            continue
        estimate = item["delta_balanced_accuracy" if is_site else "delta_mae_years"]
        interval = item.get("descriptive_percentiles_2_5_97_5")
        if interval is not None:
            # Draw interval endpoints directly: percentile intervals need not
            # contain their point estimate, unlike errorbar's expected offsets.
            ax.plot(interval, [position, position], color=COLORS[atlas], lw=2)
        ax.plot(estimate, position, "o", color=COLORS[atlas], markersize=6)
    ax.axvline(0, color="#737373", lw=1, linestyle="--")
    ax.set_yticks(np.arange(len(PAIRS)), LABELS)
    ax.set_ylim(len(PAIRS) - .5, -.5)
    ax.grid(axis="x", color="#E3E3E3", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Δ balanced accuracy" if is_site else "Δ MAE (years)")
    n_domain = len(panel["domains"] if is_site else panel["included_domains"])
    n_person = sum(d["participants"] for d in panel["domains"].values()) if is_site else panel.get("participants", 0)
    ax.set_title(f"{atlas.capitalize()} · {title}\n{n_domain} domains, {n_person} participants", fontsize=10)


def render(run):
    with (run / "covariance_protected_report.json").open() as handle:
        report = json.load(handle)
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 8), sharey=True)
    for row, atlas in enumerate(("desikan", "destrieux")):
        data = report["atlases"][atlas]
        draw_panel(axes[row, 0], data["pooled"]["external"]["all_rows"]["absolute"],
                   atlas, "External absolute MAE: all rows")
        draw_panel(axes[row, 1], data["pooled"]["external"]["common_support"]["centered"],
                   atlas, "External centered MAE: common support")
        draw_panel(axes[row, 2], data["site_probe"]["pooled_internal"],
                   atlas, "Internal ordinary site prediction", is_site=True)
    figure.suptitle("Does protecting fitting-age covariance preserve prediction utility?", fontsize=15, y=.97)
    figure.text(.5, .035,
                "Protection preserves the fitting marginal covariance vector, not all age information or ridge predictions. "
                "Fixed ridge α = 1000.\n"
                "Points and descriptive 95% paired bootstrap intervals. Negative age contrasts favor the first arm; "
                "negative site contrasts mean less decodable site.\n"
                "One within-site age shuffle; overlapping development views; restricted common support; "
                "no independent confirmation or causal interpretation.",
                ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, .13, 1, .92), w_pad=2.2)
    figures = OUTPUTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(figures / f"covariance_protected_projection.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    legacy.compute_only()
    run = args.run.resolve()
    if not run.is_relative_to(OUTPUTS / "runs/covariance_protected_projection"):
        raise ValueError("Only this diagnostic's episode-local output directory is supported")
    render(run)


if __name__ == "__main__":
    main()
