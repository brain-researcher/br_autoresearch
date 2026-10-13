#!/usr/bin/env python3
"""Render the completed fixed EP14 bilateral attempt; no fitting or selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket

EPISODE = Path(__file__).resolve().parents[2]
REPORT = EPISODE / "outputs/runs/bilateral_regularization/attempt001/bilateral_regularization_report.json"
OUTPUT = EPISODE / "outputs/figures/bilateral_regularization_attempt001"
ARMS = ("raw", "homologous", "sham14015", "sham14016", "sham14017")
LABELS = {
    "raw": "Raw ridge", "homologous": "Homologous pairs",
    "sham14015": "Shuffled pairs 14015", "sham14016": "Shuffled pairs 14016",
    "sham14017": "Shuffled pairs 14017",
}
COLORS = ("#697786", "#087F8C", "#D47C36", "#9370A8", "#759147")


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("EP14 rendering requires the allocated compute shell")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(REPORT.read_text())
    if (report.get("analysis_id") != "bilateral_regularization"
            or report.get("complete") is not True
            or report.get("holdout_access") is not False
            or report.get("candidate_promotion") is not False):
        raise ValueError("requires the complete development-only bilateral report")
    results = report["results"]
    if set(results) != set(ARMS):
        raise ValueError("all five prospectively specified arms are required")
    domains = sorted(results["raw"]["external_age"]["domains"])
    if not domains:
        raise ValueError("report has no held-out development domains")
    for arm in ARMS:
        if set(results[arm]["external_age"]["domains"]) != set(domains):
            raise ValueError("arm domain support differs")
        if len(results[arm]["per_view_external"]) != 3:
            raise ValueError("report must contain all three views")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    bottom_height = max(4.5, .28 * len(domains) + 1.6)
    fig = plt.figure(figsize=(16, 5.8 + bottom_height), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(4.3, bottom_height), width_ratios=(1.05, 1))
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    fig.suptitle("EP14 · Bilateral regularization under acquisition shift\n"
                 "Exploratory development; three overlapping RAMP views", fontsize=17)

    y = np.arange(len(ARMS))
    mae = np.array([results[a]["external_age"]["domain_macro"] for a in ARMS])
    ax_a.barh(y, mae, color=COLORS, alpha=.9, height=.62)
    ax_a.set_yticks(y, [LABELS[a] for a in ARMS])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, max(mae) * 1.36)
    for row, arm, value in zip(y, ARMS, mae):
        site = results[arm]["internal_site"]["domain_macro"]
        ax_a.text(value + max(mae) * .015, row, f"{value:.3f} y  |  {site:.3f}",
                  va="center", fontsize=9)
    ax_a.set_xlabel("External domain-macro age MAE (years)\nLabels: MAE | internal site balanced accuracy")
    ax_a.set_title("A  All five fixed arms", loc="left", fontweight="bold")
    ax_a.grid(axis="x", alpha=.2)
    ax_a.set_axisbelow(True)

    raw_view = np.array([r["domain_macro"] for r in results["raw"]["per_view_external"]])
    for index, (arm, color) in enumerate(zip(ARMS, COLORS)):
        changes = np.array([r["domain_macro"] for r in results[arm]["per_view_external"]]) - raw_view
        ax_b.plot(np.arange(3) + (index - 2) * .025, changes, "o-", color=color,
                  linewidth=1.3, markersize=5, label=LABELS[arm])
    ax_b.axhline(0, color="#555555", linewidth=.8)
    ax_b.set_xticks(range(3), ["View 0", "View 1", "View 2"])
    ax_b.set_ylabel("Age MAE change versus raw ridge (years)")
    ax_b.set_title("B  Every overlapping development view", loc="left", fontweight="bold")
    ax_b.legend(fontsize=8, frameon=False, loc="best")
    ax_b.grid(axis="y", alpha=.2)

    matrix = np.array([[results[arm]["external_age"]["domains"][domain]["mean"]
                        - results["raw"]["external_age"]["domains"][domain]["mean"]
                        for arm in ARMS[1:]] for domain in domains])
    scale = max(float(np.max(np.abs(matrix))), .01)
    heatmap = ax_c.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    labels = [f"{domain}  (n={results['raw']['external_age']['domains'][domain]['participants']})"
              for domain in domains]
    ax_c.set_yticks(range(len(domains)), labels, fontsize=8)
    ax_c.set_xticks(range(4), ["Homologous", "Shuffle\n14015", "Shuffle\n14016", "Shuffle\n14017"], fontsize=9)
    for row in range(len(domains)):
        for column in range(4):
            value = matrix[row, column]
            ax_c.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=7,
                      color="white" if abs(value) > .58 * scale else "#222222")
    ax_c.set_title("C  Every held-out development domain\n"
                   "Fixed identifier order; negative values favor the paired arm", loc="left", fontweight="bold")
    fig.colorbar(heatmap, ax=ax_c, shrink=.65, label="Age MAE change versus raw ridge (years)")

    references = ("raw", "sham14015", "sham14016", "sham14017")
    effects, lows, highs = [], [], []
    for reference in references:
        contrast = report["homologous_contrasts"][reference]
        effects.append(contrast["delta_age"])
        low, high = contrast["age_interval"]["descriptive_percentiles_2_5_97_5"]
        lows.append(low)
        highs.append(high)
    positions = np.arange(4)
    for row, effect, low, high in zip(positions, effects, lows, highs):
        ax_d.plot([low, high], [row, row], color="#087F8C", linewidth=2)
        ax_d.scatter([effect], [row], color="#087F8C", s=42, zorder=3)
        ax_d.text(.02, row + .20, f"{effect:+.3f} [{low:+.3f}, {high:+.3f}] years",
                  transform=ax_d.get_yaxis_transform(), va="top", fontsize=9)
    ax_d.axvline(0, color="#555555", linewidth=.8)
    ax_d.set_yticks(positions, ["versus " + LABELS[r] for r in references])
    ax_d.set_ylim(5.5, -.6)
    ax_d.set_xlabel("Homologous minus reference age MAE (years)\nNegative values favor homologous pairing")
    ax_d.set_title("D  Homologous pairing versus every comparator\n"
                   "95% descriptive hierarchical bootstrap intervals", loc="left", fontweight="bold")
    ax_d.grid(axis="x", alpha=.2)
    note = ("10,000 paired domain/participant resamples.\n"
            "Intervals describe observed development domains;\n"
            "they are not adjusted for multiple comparisons.\n\n"
            "Repeated appearances are combined as participant losses.\n"
            "All three shuffled pairings are shown without selection.\n"
            "Seven α settings for raw; 35 α/ρ settings per paired arm.\n"
            "No final-data evaluation or candidate promotion.")
    ax_d.text(.02, .02, note, transform=ax_d.transAxes, fontsize=9, va="bottom",
              bbox={"boxstyle": "round,pad=.5", "facecolor": "#F4F6F8", "edgecolor": "none"})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(OUTPUT.with_suffix(suffix), dpi=170)
    plt.close(fig)
    print(json.dumps({"source": str(REPORT), "figures": [str(OUTPUT.with_suffix(s)) for s in (".png", ".svg")],
                      "arms": len(ARMS), "domains": len(domains), "views": 3, "evidence": "exploratory_development"}))


if __name__ == "__main__":
    main()
