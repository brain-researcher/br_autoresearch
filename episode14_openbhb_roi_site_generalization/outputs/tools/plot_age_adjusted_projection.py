#!/usr/bin/env python3
"""Render only prespecified contrasts from the completed EP14 diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_development as legacy

PAIRS = (("original", "raw"), ("age_adjusted", "raw"),
         ("age_adjusted", "original"), ("sham_adjusted", "original"),
         ("age_adjusted", "sham_adjusted"))
LABELS = ("Original − raw", "Age adjusted − raw", "Age adjusted − original",
          "Sham adjusted − original", "Age adjusted − sham")
COLORS = {"desikan": "#176B87", "destrieux": "#A34F31"}


def render(run):
    with (run / "age_adjusted_report.json").open() as handle:
        report = json.load(handle)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for row, atlas in enumerate(("desikan", "destrieux")):
        data = report["atlases"][atlas]
        all_absolute = data["pooled"]["external"]["all_rows"]["absolute"]
        common_centered = data["pooled"]["external"]["common_support"]["centered"]
        site = data["site_probe"]["pooled_internal"]
        panels = ((all_absolute, "External absolute MAE: all rows", "Δ MAE (years)", False),
                  (common_centered, "External centered MAE: common support", "Δ MAE (years)", False),
                  (site, "Internal ordinary site prediction", "Δ balanced accuracy", True))
        for col, (panel, title, xlabel, is_site) in enumerate(panels):
            ax = axes[row, col]
            for position, (a, b) in enumerate(PAIRS):
                key = f"{a}_minus_{b}" if is_site else f"{a}_alpha1000_minus_{b}_alpha1000"
                value = panel["contrasts"].get(key)
                if value is None:
                    ax.text(.03, position, "Insufficient support", transform=ax.get_yaxis_transform(), fontsize=9)
                    continue
                estimate = value["delta_balanced_accuracy" if is_site else "delta_mae_years"]
                interval = value.get("descriptive_percentiles_2_5_97_5")
                if interval is not None:
                    # A percentile interval need not contain its point estimate.
                    ax.plot(interval, [position, position], color=COLORS[atlas], lw=2)
                ax.plot(estimate, position, "o", color=COLORS[atlas], markersize=6)
            ax.axvline(0, color="#737373", lw=1, linestyle="--")
            ax.set_yticks(np.arange(len(PAIRS)), LABELS)
            ax.set_ylim(len(PAIRS) - .5, -.5)
            ax.grid(axis="x", color="#E3E3E3", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel(xlabel)
            n_domain = len(panel["domains"] if is_site else panel["included_domains"])
            n_person = sum(d["participants"] for d in panel["domains"].values()) if is_site else panel.get("participants", 0)
            ax.set_title(f"{atlas.capitalize()} · {title}\n{n_domain} domains, {n_person} participants", fontsize=10)
    figure.suptitle("Does linear age adjustment change which directions are removed?", fontsize=15, y=.97)
    figure.text(.5, .035,
                "Fixed ridge α = 1000. Points and descriptive 95% paired bootstrap intervals; overlapping development views.\n"
                "Negative age contrasts favor the first arm; negative site contrasts mean less decodable site. "
                "Common support covers a younger, restricted population.\n"
                "One within-site age shuffle; evaluator-only centering; no independent confirmation or causal interpretation.",
                ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, .13, 1, .92), w_pad=2.2)
    figures = Path(__file__).resolve().parents[1] / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(figures / f"age_adjusted_projection.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    legacy.compute_only()
    render(args.run.resolve())


if __name__ == "__main__":
    main()
