#!/usr/bin/env python3
"""Six-panel figure for the October 16 EP19 source-fact checks (development data only).

Usage: plot_source_fact_checks.py RUN_DIR OUTPUT_PNG
"""
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
G = 9.81
G_VOLT = 1.45e6


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def note(ax, text, x=0.98, y=0.95, ha="right"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top", fontsize=7.5, color=INK2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))


def main():
    run, png = sys.argv[1], sys.argv[2]
    way = json.load(open(os.path.join(run, "way_source_facts.json")))
    sp = json.load(open(os.path.join(run, "self_paced_source_facts.json")))
    summ = json.load(open(os.path.join(run, "source_facts_summary.json")))
    trials = [dict(t, P=f["participant"]) for f in way["files"] for t in f["trials"]]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.08, wspace=0.28, hspace=0.52)

    # A: static-hold load force by object mass (participant medians).
    ax = axes[0, 0]
    rng = np.random.default_rng(19)
    for grams in (165, 330, 660):
        vals = [np.median([t["LF_hold_net"] for t in trials if t["P"] == p and t.get("weight_id") == f"{grams}g"
                           and "LF_hold_net" in t]) for p in range(1, 13)
                if any(t["P"] == p and t.get("weight_id") == f"{grams}g" and "LF_hold_net" in t for t in trials)]
        ax.scatter(grams + rng.uniform(-12, 12, len(vals)), vals, s=26, color=S1, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    xs = np.array([120, 700])
    ax.plot(xs, xs / 1000 * G, color=INK2, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(560, 0.56 * G + 0.45, "m·g", color=INK2, fontsize=8, ha="right", va="bottom")
    style(ax, "A  WAY load force at static hold", "object mass (g)", "FX1 + FX2 minus rest (N)")
    ax.set_xticks([165, 330, 660])
    note(ax, "dots: 12 participant medians\nper mass, series 1–7", x=0.03, ha="left")

    # B (post hoc classification): automatic LED onset delay in weight series vs button-route series.
    ax = axes[0, 1]
    post = json.load(open(os.path.join(run, "post_hoc_summary.json")))
    kinds = summ["way"]["W5"]["series_kind_by_file"]
    ax.axvspan(1, 3, color=GRID, zorder=0)
    for kind_name, color, label in (("weight", S1, "weight series (automatic cue)"),
                                    ("friction", S2, "friction + mixed (button cue)")):
        sel = ("weight",) if kind_name == "weight" else ("friction", "mixed")
        vals = [t["delay_since_min_below15_s"] for f in way["files"] for t in f["trials"]
                if kinds[f"P{f['participant']}S{f['series']}"] in sel and "hs_high_s" in t
                and t.get("delay_since_min_below15_s") is not None]
        ax.hist(np.clip(vals, 0, 15), bins=np.arange(0, 15.01, 0.25), histtype="step", color=color, linewidth=1.6,
                label=label, zorder=3)
    ax.text(3.2, ax.get_ylim()[1] * 0.80, "documented 1–3 s", ha="left", va="center", fontsize=7.2, color=INK2)
    w6 = summ["way"]["W6"]
    frac_w = post["W6_cue_rule_by_series_type"]["weight"]["min"]["fraction_1_to_3s"]
    style(ax, "B  WAY 15-cm cue rule holds in centimetres", "LED onset minus last digit–object distance < 15 (s)",
          "trials")
    ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=INK2)
    note(ax, f"weight series:\n{100 * frac_w:.1f}% in 1–3 s\n(post hoc split)\nlift {w6['lift_height']['p50']:.1f} cm",
         x=0.27, y=0.62, ha="left")

    # C: HS LED rise to sustained wrist motion.
    ax = axes[0, 2]
    lat = [t["wrist_motion_latency_s"] for t in trials if t.get("wrist_motion_latency_s") is not None]
    ax.hist(lat, bins=np.arange(0, 2.51, 0.05), color=S1, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    w2 = summ["way"]["W2"]
    n_win = summ["way"]["windows"]
    n_m1000 = w2["n0_minus_500tst"].get("-1000", 0)
    style(ax, "C  WAY clock: LED rise to wrist motion in HS", "wrist speed > 10 cm/s after LED rise (s)", "trials")
    note(ax, f"{n_m1000} of {n_win} WS windows start exactly\n2.000 s before trial_start_time in HS\n"
             f"(kin, EEG, EMG agree; values identical)")

    # D: sub-03 ses-02 recording timeline.
    ax = axes[1, 0]
    runs = sorted((r for r in sp["runs"] if r["stem"].startswith("sub-03_ses-02")), key=lambda r: r["stem"])
    t0 = runs[0]["start_stamp"]
    for i, r in enumerate(runs):
        bad = not r["vmrk_datafile_ok"]
        ax.barh(i, r["duration_s"] / 60, left=(r["start_stamp"] - t0) / 60, height=0.6,
                color=S2 if bad else S1, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        label = r["stem"][-8:-4]
        ax.text((r["start_stamp"] - t0 + r["duration_s"]) / 60 + 1.0, i,
                f"run {label}" + ("  marker names run 0008" if bad else ""), va="center", fontsize=7.2, color=INK2)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, (runs[-1]["start_stamp"] - t0 + runs[-1]["duration_s"]) / 60 + 38)
    start = datetime.fromtimestamp(t0).strftime("%H:%M")
    style(ax, "D  Self-paced sub-03/ses-02 runs do not overlap", f"minutes after {start} (marker NewSegment)", "")
    ax.legend(handles=[Patch(color=S1, label="marker DataFile names own run"),
                       Patch(color=S2, label="marker DataFile names run 0008")],
              loc="upper right", fontsize=7, frameon=False, labelcolor=INK2)

    # E: TRIG cue-interval deviations from their class medians.
    ax = axes[1, 1]
    med = summ["self_paced"]["S4"]["class_medians_s"]
    for kind, color, lab in (("start_end", S1, "start→end"), ("end_start", S2, "end→start")):
        dev = [1000 * (g - med[kind]) for r in sp["runs"] for g, k in zip(r["cue_gaps_s"], r["cue_gap_kinds"]) if k == kind]
        ax.hist(np.clip(dev, -150, 150), bins=np.arange(-150, 151, 5), histtype="step", color=color, linewidth=1.6,
                label=f"{lab} (median {med[kind]:.3f} s)", zorder=3)
    ax.axvline(-60, color=AXIS, linewidth=1, linestyle="--")
    ax.axvline(60, color=AXIS, linewidth=1, linestyle="--")
    style(ax, "E  Self-paced cue intervals (unmarked-gap check)", "deviation from class median (ms; clipped ±150)",
          "intervals")
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK2)
    s4 = summ["self_paced"]["S4"]
    dec = json.load(open(os.path.join(run, "post_hoc_summary.json")))["S4_cycle_decomposition"]
    note(ax, f"{s4['intervals']} intervals, 155 runs\n{len(s4['flagged_intervals'])} beyond ±60 ms\n\n"
             f"uncompensated short\ncycles (post hoc):\n{dec['cold_start_first_cycle']} first cue of run\n"
             f"{len(dec['spurious_burst'])} spurious burst\n{len(dec['ambiguous_mid_run'])} ambiguous, "
             f"{dec['ambiguous_dev_s']['p50']:+.2f} s\n(of {dec['cycles']} cycles)", x=0.98, y=0.74)

    # F: resting accelerometer norm against the manufacturer 1 g scale.
    ax = axes[1, 2]
    ratio = [r["acc_norm_median_uV"] / G_VOLT for r in sp["runs"]]
    ax.axvspan(0.95, 1.05, color=GRID, zorder=0)
    ax.hist(ratio, bins=np.arange(0.85, 1.251, 0.005), color=S1, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.text(0.945, ax.get_ylim()[1] * 0.92, "1.45 V/g ± 5%", ha="right", va="top", fontsize=7.5, color=INK2)
    off = sum(x < 0.85 for x in ratio)
    ax.set_xlim(0.85, 1.25)
    note(ax, f"{off} runs off-scale at 0.37:\nsub-22/ses-01, X axis flat\n\n5 runs at 1.04:\nsub-15/ses-02 (1 kHz)",
         x=0.98, y=0.80)
    style(ax, "F  Self-paced accelerometer at rest", "median |X,Y,Z| ÷ 1.45×10⁶ µV  (stored value × resolution)",
          "runs")

    fig.text(0.06, 0.965, "EP19 source facts checked on development data (October 16, 2026)", fontsize=12,
             color=INK, fontweight="bold")
    fig.text(0.06, 0.93, "WAY-EEG-GAL P1–P12 series 1–7 and 155 self-paced development runs; WAY series 8–9, "
                         "the eight held-out people and all EEG models untouched. Calibration evidence, not a forecasting result.",
             fontsize=8.5, color=INK2)
    fig.savefig(png, dpi=180, facecolor=SURFACE)


if __name__ == "__main__":
    main()
