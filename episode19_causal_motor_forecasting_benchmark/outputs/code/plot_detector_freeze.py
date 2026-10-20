#!/usr/bin/env python3
"""One figure for the EP19 development detector freeze (calibration evidence)."""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"WAY": "#2a78d6", "self_paced": "#eb6834"}
NAME = {"WAY": "WAY-EEG-GAL (P1–P12, S1–S7)", "self_paced": "Self-paced (15 development people)"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    d = args.run_dir
    exposure = list(csv.DictReader(open(os.path.join(d, "exposure_by_participant.csv"))))
    way = list(csv.DictReader(open(os.path.join(d, "way_cues.csv"))))
    sp = list(csv.DictReader(open(os.path.join(d, "self_paced_cues.csv"))))
    summary = json.load(open(os.path.join(d, "detector_freeze_summary.json")))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.885, bottom=0.12, hspace=0.55, wspace=0.18)
    labels = [r["participant"].replace("sub-", "s") for r in exposure]
    srcs = [r["source"] for r in exposure]
    x = np.arange(len(exposure))
    colors = [COLOR[s] for s in srcs]

    ax = axes[0, 0]
    style(ax, "a  Eligible stillness (armed, at-risk time) per person")
    minutes = [60 * num(r["eligible_stillness_hours"]) for r in exposure]
    ax.bar(x, minutes, width=0.62, color=colors, edgecolor=SURFACE, linewidth=1)
    ax.set_ylabel("minutes", color=INK2, fontsize=8)
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.legend(handles=[Patch(facecolor=COLOR[s], label=NAME[s]) for s in ("WAY", "self_paced")],
              frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    ax = axes[0, 1]
    style(ax, "b  Primary horizon (300–600 ms): scorable at-risk decisions and onsets")
    scorable = [num(r["scorable_300_600"]) for r in exposure]
    positive = [num(r["positive_300_600"]) for r in exposure]
    ax.bar(x, scorable, width=0.62, color=[c + "40" for c in colors], edgecolor=SURFACE, linewidth=1)
    ax.bar(x, positive, width=0.36, color=colors, edgecolor=SURFACE, linewidth=1)
    ax.set_ylabel("decisions (50-ms grid)", color=INK2, fontsize=8)
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.legend(handles=[Patch(facecolor="#89878140", label="scorable (status known through 600 ms; light)"),
                       Patch(facecolor=MUTED, label="onset 300–600 ms ahead (dark)")],
              frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    ax = axes[1, 0]
    style(ax, "c  Detector-crossing delay (backward supra-low run + 20-ms support)")
    bound = summary["timing_bound_ms"]["primary_p95"]
    texts = []
    for src, rows in (("WAY", way), ("self_paced", sp)):
        v = np.sort([num(r.get("departure_ms")) for r in rows if np.isfinite(num(r.get("departure_ms")))])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=COLOR[src], linewidth=2, label=NAME[src],
                solid_capstyle="round")
        p95 = summary["crossing_delay_ms"][src]["p95"]
        ax.plot([p95, p95], [0, 0.95], color=COLOR[src], linewidth=1)
        ax.plot([p95], [0.95], "o", color=COLOR[src], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        texts.append((p95, src))
    for p95, src in texts:
        ax.annotate(f"p95 {p95:.0f} ms", (p95, 0.95), xytext=(5, -22),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlim(0, max(250, summary["crossing_delay_ms"]["self_paced"].get("p99", 0) * 1.2))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("ms from sensor departure to the labelled crossing", color=INK2, fontsize=8)
    ax.set_ylabel("cumulative fraction of onsets", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    ax.text(0.98, 0.30, f"Timing bound with the p95 term:\nWAY {bound['WAY_ms']:.0f} ms, self-paced "
            f"{bound['self_paced_ms']:.0f} ms (limit < 300 ms)", transform=ax.transAxes, ha="right",
            fontsize=8, color=INK)

    ax = axes[1, 1]
    style(ax, "d  Label onset minus provider onset (agreement diagnostic only)")
    bins = np.arange(-800, 801, 20)
    for src, rows, key, prov in (("WAY", way, "onset_minus_handstart_ms", "HandStart"),
                                 ("self_paced", sp, "onset_minus_accstart_ms", "AccStartIndex")):
        v = np.array([num(r.get(key)) for r in rows])
        v = v[np.isfinite(v)]
        h, e = np.histogram(np.clip(v, -800, 800), bins=bins)
        ax.step(e[:-1], h / max(1, len(v)), where="post", color=COLOR[src], linewidth=2,
                label=f"{'WAY' if src == 'WAY' else 'Self-paced'} − {prov} (median {np.median(v):.0f} ms, n={len(v)})")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)  # headroom so the legend clears the peaks
    ax.set_xlabel("ms (negative = EP19 label earlier); clipped at ±800 ms", color=INK2, fontsize=8)
    ax.set_ylabel("fraction of onsets per 20-ms bin", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    eq = summary["equivalence_generic_vs_WayRiskStream"]
    fig.suptitle("EP19 development onset/stillness detector freeze — calibration evidence, not a scientific result",
                 x=0.06, ha="left", fontsize=12, color=INK, y=0.975)
    fig.text(0.06, 0.945,
             "Rules fixed from development stationary noise before any detector output (component_details.md §1.2). "
             f"Job {summary['job_id']}. Generic replay = WayRiskStream on {eq['series']} WAY series "
             f"({eq['grid_mismatch'] + eq['onset_mismatch'] + eq['label_mismatch']} mismatches).\n"
             "No EEG, WAY S8–S9 or held-out person used. Provider onsets use zero-phase smoothing and are agreement checks, "
             "not labels.", fontsize=8, color=INK2, va="top")
    fig.savefig(args.output, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
