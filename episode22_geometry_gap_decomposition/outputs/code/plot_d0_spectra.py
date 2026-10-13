#!/usr/bin/env python3
"""Render EP22 D0 outcome-blind spectra from d0_spectra_summary.json only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VLMS = ("CLIP", "PEcore", "SigLIP2")
COLORS = {"CLIP": "#2a78d6", "PEcore": "#eb6834", "SigLIP2": "#1baf7a"}  # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STRATA = [("laion", "LAION"), ("things", "THINGS"), ("thingsplus", "THINGSplus"),
          ("ood_photographic", "OOD\nphoto"), ("ood_synthetic", "OOD\nsynth")]


def values(summary, model, metric):
    """Per-participant dots and pooled (mean, q025, q975) for each stratum."""
    strata = summary["strata"]
    n_sub = summary["settings"]["n_sub"]
    participants = sorted({k.split("|")[1] for k in strata if k.startswith("dev|sub-")})
    out = []
    for key, _ in STRATA:
        if key.startswith("ood_"):
            pooled_key, dots = f"ood|{key[4:]}|n{n_sub}", []
        else:
            pooled_key = f"dev|pooled|{key}|n{n_sub}"
            dots = [pick(strata.get(f"dev|{p}|{key}|n{n_sub}"), model, metric) for p in participants]
            dots = [d["mean"] for d in dots if d]
        pooled = pick(strata.get(pooled_key), model, metric)
        out.append((dots, pooled))
    return out


def pick(stratum, model, metric):
    if not stratum or "skipped" in stratum:
        return None
    if metric == "log_ratio":
        return stratum["log_ED_ratio_vs_DINOv2"].get(model)
    return stratum["pairs"].get(f"DINOv2|{model}", {}).get("CKA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads(a.summary.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey="row", constrained_layout=True)
    rows = [("log_ratio", "log(ED$_{model}$ / ED$_{DINOv2}$)"), ("cka", "Linear CKA with DINOv2")]
    for r, (metric, ylabel) in enumerate(rows):
        for c, model in enumerate(VLMS):
            ax = axes[r, c]
            color = COLORS[model]
            for x, (dots, pooled) in enumerate(values(summary, model, metric)):
                if dots:
                    ax.scatter([x - 0.12] * len(dots), dots, s=26, color=color, alpha=0.55,
                               edgecolors="white", linewidths=0.8, zorder=3)
                if pooled:
                    ax.plot([x + 0.12, x + 0.12], [pooled["q025"], pooled["q975"]], color=color, lw=2, zorder=2,
                            solid_capstyle="round")
                    ax.scatter([x + 0.12], [pooled["mean"]], marker="D", s=40, color=color,
                               edgecolors="white", linewidths=1.2, zorder=4)
            if metric == "log_ratio":
                ax.axhline(0, color=INK2, lw=1, zorder=1)
            ax.set_xticks(range(len(STRATA)), [label for _, label in STRATA])
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(model, color=INK, fontsize=11, fontweight="bold", loc="left")
            if c == 0:
                ax.set_ylabel(ylabel)
    s = summary["settings"]
    fig.suptitle("EP22 D0 · outcome-blind embedding spectra (no neural data)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02,
             f"Random subsets of {s['n_sub']} images, {s['repeats']} per stratum. Dots: subject-unique development "
             "images per participant (mean over subsets). Diamonds and bars: pooled mean and 2.5–97.5% of subsets "
             "(subset spread, not a confidence interval). Positive log ratio = flatter VLM spectrum than DINOv2.",
             ha="left", va="top", color=INK2, fontsize=8, wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(a.out_dir / f"d0_outcome_blind_spectra.{ext}", dpi=200, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
