#!/usr/bin/env python3
"""Plot EP16 structural support under the 2026-10-16 source-fact rules (support metrics only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ELIGIBLE, SUPPORTED, NEUTRAL = "#2a78d6", "#1baf7a", "#c3c2b7"
ROWS = [("T5", "development"), ("T10", "development"), ("T11", "development"), ("T3", "development, not holdable"),
        ("T2", "held-out candidate"), ("T6", "held-out candidate"), ("T7", "held-out candidate"),
        ("T8", "held-out candidate"), ("T9", "held-out candidate")]
SHORT = {"P0": "P0\nprimary", "S1_exclude_incomplete_blocks": "S1\nincomplete\nblocks out",
         "S2_strict_slot_tolerance": "S2\nstrict\nslots", "S3_narrow_families": "S3\nradial-8 +\ngrid only",
         "S4_scale_locked_matching": "S4\nscale\nlocked", "S5_placeholder_blocks_excluded": "S5\nplaceholder\nblocks out",
         "S6_strict_transitions": "S6\nstrict\nedges", "S7_session_schedule_combinations": "S7\nsession-\nschedule",
         "S8_without_packet_coverage_condition": "S8\nno packet\ncoverage", "S9_strict_group_rule": "S9\nstrict\ngroup",
         "S10_finite_sampling_check_not_applied": "S10\nno sampling\ncheck"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    variants = list(report["variants"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": NEUTRAL,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(12.5, 5.2), facecolor=SURFACE)
    left = fig.add_axes([0.165, 0.17, 0.47, 0.66], facecolor=SURFACE)
    right = fig.add_axes([0.715, 0.17, 0.265, 0.66], facecolor=SURFACE)

    # Panel a: participant support status per rule variant.
    for r, (participant, role) in enumerate(ROWS):
        y = len(ROWS) - 1 - r
        for c, name in enumerate(variants):
            state = report["variants"][name]["participants"][participant]
            if state["held_out_eligible"]:
                left.scatter(c, y, s=120, color=ELIGIBLE, edgecolor=SURFACE, linewidth=2, zorder=3)
            elif state["supported"]:
                left.scatter(c, y, s=120, color=SUPPORTED, edgecolor=SURFACE, linewidth=2, zorder=3)
            else:
                left.plot([c - 0.12, c + 0.12], [y, y], color=NEUTRAL, linewidth=2, solid_capstyle="round")
    designs = [report["variants"][name]["design"]["design_supported"] for name in variants]
    for c, ok in enumerate(designs):
        left.text(c, -1.25, "yes" if ok else "no", ha="center", va="center", color=INK, fontsize=8.5,
                  fontweight="bold")
    left.axhline(-0.62, color=GRID, linewidth=1)
    left.axhline(4.5, color=GRID, linewidth=1, linestyle=(0, (2, 2)))
    left.set_yticks([len(ROWS) - 1 - r for r in range(len(ROWS))] + [-1.25])
    left.set_yticklabels([f"{p}  ({role})" for p, role in ROWS] + ["4 dev + 3 held-out\nsupported?"], fontsize=8)
    left.set_xticks(range(len(variants)))
    left.set_xticklabels([SHORT.get(v, v) for v in variants], fontsize=6.8)
    left.xaxis.tick_top()
    left.set_xlim(-0.6, len(variants) - 0.4)
    left.set_ylim(-1.8, len(ROWS) - 0.4)
    for side in ("left", "right", "bottom", "top"):
        left.spines[side].set_visible(False)
    left.tick_params(length=0)
    left.set_title("a  Support status by rule variant (layout-only rules fixed before application)",
                   loc="left", color=INK, fontsize=9.5, pad=46)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=ELIGIBLE, label="held-out eligible (≥6 near, ≥6 long pairs in passing components)"),
               Line2D([], [], marker="o", linestyle="", markersize=8, color=SUPPORTED, label="supported (≥1 passing component)"),
               Line2D([], [], color=NEUTRAL, linewidth=2, label="no passing component")]
    left.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.30, -0.02), ncol=3, frameon=False,
                fontsize=7.5, labelcolor=INK2)

    # Panel b: primary-rule gate metrics for each participant's best full-rank both-lag component.
    p0 = report["variants"]["P0"]
    best, deficient = {}, []
    for group in p0["groups_with_designs"]:
        for design in group["designs"]:
            if not (design["near_pairs"] and design["long_pairs"]):
                continue
            p = group["participant"]
            if design["design_rank"] < design["design_columns"]:
                deficient.append((p, design["near_pairs"], design["long_pairs"]))
            elif p not in best or design["retained_energy_fraction"] > best[p]["retained_energy_fraction"]:
                best[p] = design
    right.add_patch(Rectangle((0.10, 0.0), 0.25, 0.25, facecolor="#e8f1fc", edgecolor="none", zorder=0))
    right.axvline(0.10, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    right.axhline(0.25, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
    right.text(0.105, 0.24, "passes both gates", ha="left", va="top", color=INK2, fontsize=7.5)
    for p, d in sorted(best.items()):
        ok = d["conditional_numerical_pass"]
        x, y = d["retained_energy_fraction"], d["maximum_contrast_node_share"]
        right.scatter(x, y, s=70, color=ELIGIBLE if ok else MUTED, edgecolor=SURFACE, linewidth=2, zorder=3)
        right.annotate(f"{p}  {d['near_pairs']}/{d['long_pairs']}", (x, y), xytext=(6, 4), textcoords="offset points",
                       color=INK, fontsize=7.8)
    right.set_xlim(0, 0.35)
    right.set_ylim(0, 1.0)
    right.set_xlabel("retained long-gap information (gate: > 0.10)")
    right.set_ylabel("largest single-session share (gate: < 0.25)")
    right.grid(color=GRID, linewidth=0.6)
    right.set_axisbelow(True)
    for side in ("right", "top"):
        right.spines[side].set_visible(False)
    right.set_title("b  P0 gate metrics, best full-rank component", loc="left", color=INK,
                    fontsize=9.5, pad=8)
    text = "rank-deficient: " + ", ".join(f"{p} {n}/{l}" for p, n, l in sorted(deficient, key=lambda x: int(x[0][1:]))) + "\nno both-lag component: " + \
        ", ".join(p for p, _ in ROWS if p not in best and p not in {x[0] for x in deficient})
    right.text(0.97, 0.53, "labels: near/long pairs\n" + text, transform=right.transAxes, ha="right", va="top", color=INK2,
               fontsize=7.3, linespacing=1.4)
    fig.text(0.012, 0.015, "Structural metadata only (one target point per trial, block/trial indices, electrode identity); "
             "no neural, cursor, success or decoder values. Run 20261016-source_fact_support01, allocation "
             f"{report['slurm_job_id']}.", color=MUTED, fontsize=6.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
