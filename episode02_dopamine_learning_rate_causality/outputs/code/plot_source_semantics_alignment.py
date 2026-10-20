#!/usr/bin/env python3
"""Plot lickState-window agreement by candidate reward sample (October 16 check).

Input is the candidate-role `semantics.json`; it holds agreement fractions and
onset indices only, so the figure shows no lick probability, gain or other
outcome magnitude. Run inside a Slurm allocation.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [("w75", "750 ms", "#2a78d6"), ("w100", "1 s", "#eb6834"),
          ("w150", "1.5 s", "#1baf7a"), ("w50", "500 ms", "#eda100")]
LABEL_AT = {"w75": (284, 0.99, "right"), "w100": (334, 0.962, "left"),
            "w150": (334, 0.884, "left"), "w50": (266, 0.955, "right")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("semantics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.semantics.read_text())
    onset = result["predicates"]["onset"]
    r_star = result["R_star_primary_onset"]
    e1 = result["E1_uncued_lick_rise"]["cross_10"]
    e2 = result["E2_rewarded_minus_omission_rise"]["cross_10"]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)

    for key, name, color in SERIES:
        profile = onset["scan_profiles"][key]
        x = range(profile["R_first"], profile["R_first"] + len(profile["agreement"]))
        best = onset["best_by_window"][key]
        ax.plot(list(x), profile["agreement"], color=color, linewidth=1.6, solid_joinstyle="round",
                solid_capstyle="round", label=f"{name} window: max {best['agreement']:.3f} at R = {best['R']}")
        lx, ly, align = LABEL_AT[key]
        ax.plot([lx + (3 if align == "right" else -14), lx + (14 if align == "right" else -3)], [ly, ly],
                color=color, linewidth=1.6)
        ax.text(lx, ly, f"{name}", ha=align, va="center", color=INK2, fontsize=8.5)

    ax.axvline(r_star, color=INK2, linewidth=0.8)
    ax.text(r_star - 3, 0.615, f"R = {r_star}: best fit of the 750-ms window\n(cue onset implied at {r_star - 150})",
            color=INK2, fontsize=8, va="bottom", ha="right")
    for x in (e1, e2):
        ax.plot([x], [1.004], marker="v", markersize=6, color=MUTED, clip_on=False)

    ax.set_xlim(150, 450)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Candidate reward sample (zero-based index on the 701-sample, 10-ms axis)")
    ax.set_ylabel("Agreement of lickState with a lick\nonset in the window before reward")
    fig.suptitle("lickState equals 'a lick in the 750 ms before reward' when reward sits at sample 299",
                 x=0.02, ha="left", fontsize=11, color=INK)
    ax.set_title(f"EP02 development controls only (9 mice, {onset['usable_trials']:,} trials). Lick onsets = one-sample rises "
                 f"≥ {result['theta']} Hz in the causal lick-rate trace.\n▾ independent post-reward lick rises (10% points): "
                 f"uncued trials R+{e1 - r_star}, rewarded minus omission R+{e2 - r_star}. No outcome magnitude is plotted.\n"
                 "Source: outputs/runs/source_semantics_checks_job_48180654/semantics.json",
                 loc="left", fontsize=7.5, color=MUTED)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8, labelcolor=INK2,
                       title="Window before reward", title_fontsize=8)
    legend.get_title().set_color(INK2)
    for text in legend.get_texts():
        text.set_color(INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=SURFACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
