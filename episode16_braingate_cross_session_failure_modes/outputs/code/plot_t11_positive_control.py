#!/usr/bin/env python3
"""Figure: T11 local signal and the known-positive control (development data, calibration04)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ep16_data import assemble, load_feature_dir, passing_component_days, structural_records
from ep16_panel import Panel

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support = "outputs/runs/structural_support/20261016-source_fact_support01/source_fact_support_report.json"
    days = passing_component_days(support, "T11")
    records = structural_records("outputs/runs/structural_support/20261003-fields_all01/structural_sessions.jsonl", "T11", days)
    feats = load_feature_dir(args.feature_dir, "T11", days)
    control = json.loads((args.run_dir / "positive_control.json").read_text())
    lam, J_min = control["values"]["ridge_lambda_sbp"], control["values"]["J_min"]
    lam_frozen = control["values"].get("ridge_lambda_sbp_frozen_scaling", 177.82794100389228)
    J = {}
    for norm, l in (("block", lam), ("none", lam_frozen)):
        sessions, _, _ = assemble(records, feats, normalization=norm)
        J[norm] = Panel(sessions, "sbp", l, J_min).tables["fit"]["J"]
        session_days = np.array([s.day for s in sessions])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), gridspec_kw={"width_ratios": [1.25, 1.2, 1.05]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax = axes[0]
    ax.scatter(session_days, J["none"], s=22, color=ORANGE, edgecolor=SURFACE, linewidth=0.8, zorder=3, label="frozen source-only scaling")
    ax.scatter(session_days, J["block"], s=22, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=4, label="block-normalized (proposed)")
    ax.axhline(J_min, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(session_days.max(), J_min + 0.015, f"J_min {J_min:.2f}", ha="right", va="bottom", color=INK2, fontsize=8)
    fails = J["block"] <= J_min
    ax.set_title("A  T11 local signal per session", loc="left")
    ax.set_xlabel("days since implant")
    ax.set_ylabel("J = local score − null (fold-averaged)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(0.99, 0.02, f"{fails.sum()} of {len(fails)} block-normalized sessions ≤ J_min", transform=ax.transAxes,
            ha="right", va="bottom", color=INK2, fontsize=8)

    ax = axes[1]
    obs = control["runs"]["screened_diagnostic"]["uninjected"]["point"]["adjusted"]
    for name, color, label in (("uninjected", MUTED, "observed T11"), ("extra_loss_injected", BLUE, "+0.20 long-gap injection")):
        draws = np.load(args.run_dir / f"draws_screened_diagnostic_{name}.npz")["adjusted"]
        finite = draws[np.isfinite(draws)]
        ax.hist(finite, bins=60, color=color, alpha=0.55 if name == "uninjected" else 0.75, label=f"{label} ({np.isnan(draws).mean():.0%} draws nonestimable)",
                edgecolor=SURFACE, linewidth=0.3)
    ax.axvline(obs + 0.10, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(obs + 0.10, ax.get_ylim()[1] * 0.55, " observed + 0.10:\n injected lower bound\n must exceed this", color=INK2, fontsize=7.5, va="top")
    ax.set_title("B  Adjusted contrast, screened diagnostic", loc="left")
    ax.set_xlabel("long − near normalized transport loss (finite draws)")
    ax.set_ylabel("bootstrap draws")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 1.0))
    ax.set_xlim(np.nanpercentile(np.load(args.run_dir / "draws_screened_diagnostic_extra_loss_injected.npz")["adjusted"], [0.5, 99.5]) + np.array([-0.4, 0.4]))

    ax = axes[2]
    rows = [(v, r) for v in ("strict_frozen_readiness", "screened_diagnostic") for r in ("uninjected", "extra_loss_injected", "local_signal_injected")]
    def fraction(v, r):  # an unresolved injection is a failed control: drawn as a full hatched bar
        run = control["runs"][v][r]
        if run.get("status") == "unresolved":
            return None
        return run["bounds"]["nonestimable"]["adjusted" if r != "local_signal_injected" else "fractional_local"] / control["draws"]
    fractions = [fraction(v, r) for v, r in rows]
    labels = [("strict" if v.startswith("strict") else "screened") + "\n" + {"uninjected": "observed", "extra_loss_injected": "+0.20 R",
              "local_signal_injected": "−0.30 J"}[r] for v, r in rows]
    colors = [ORANGE if v.startswith("strict") else BLUE for v, _ in rows]
    heights = [1.0 if f is None else f for f in fractions]
    bars = ax.bar(range(len(rows)), heights, color=colors, width=0.62, edgecolor=SURFACE, linewidth=2)
    for bar, f in zip(bars, fractions):
        if f is None:
            bar.set_facecolor(SURFACE)
            bar.set_edgecolor(ORANGE)
            bar.set_hatch("///")
    ax.axhline(0.0125, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("dashed: 1.25%, the most\nnonestimable draws a\nfinite bound allows", xy=(4.5, 0.0125), xytext=(4.5, 0.62), ha="center",
                color=INK2, fontsize=7.5, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    for i, f in enumerate(fractions):
        ax.text(i, (1.0 if f is None else f) + 0.015, "unresolved\ninjection" if f is None else f"{f:.0%}",
                ha="center", va="bottom", color=INK, fontsize=7.5 if f is None else 8)
    ax.set_xticks(range(len(rows)), labels, fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("nonestimable draws (readiness or rank)")
    ax.set_title("C  Why the frozen control is unresolved", loc="left")
    fig.suptitle("EP16 T11 known-positive control: injections are recovered, but readiness failures leave every R bound infinite",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Development data only (T11). 9,999 session-node draws; evaluation trials resampled within block × octant strata. "
             "Screened = sessions with observed J ≤ J_min removed before pairing (diagnostic, not the frozen rule).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(args.output, dpi=170, facecolor=SURFACE)
    print(args.output)


if __name__ == "__main__":
    main()
