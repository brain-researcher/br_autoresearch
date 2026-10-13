#!/usr/bin/env python3
"""Plot the complete fixed LMP temporal-offset report; no arrays or new fits."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANIMALS = ("Mihili", "Chewie")
COLORS = {"Mihili": "#276f94", "Chewie": "#bb6632"}
OFFSETS = tuple(range(1, 15))
METRICS = ("delta_R2", "identity_increment")
SESSION_COUNTS = {"Mihili": 6, "Chewie": 10}


def source_rows(report):
    if (report["status"] != "complete_fixed_temporal_offset_comparison"
            or report["new_fits"] != 0 or report["new_source_values_read"] is not False
            or report["offsets"] != list(OFFSETS)):
        raise ValueError("Complete fixed-model fourteen-offset report required")
    sessions = report["control"]["sessions"]
    ids = [session["session_id"] for session in sessions]
    if (len(ids) != 16 or len(set(ids)) != 16 or set(ids) != set(report["completed_sessions"])
            or any(sum(s["animal"] == animal for s in sessions) != SESSION_COUNTS[animal] for animal in ANIMALS)):
        raise ValueError("All sixteen sessions in their original two-animal groups required")
    plotted = []
    for session in sessions:
        if (session["offsets"] != list(OFFSETS)
                or [row["offset_bins"] for row in session["offset_rows"]] != list(OFFSETS)
                or session["zero_offset_exact_prediction_replay"] is not True):
            raise ValueError("Exact aligned anchor and every original offset required")
        for row in session["offset_rows"]:
            if row["donor_time_indices"] != [(t+row["offset_bins"]) % 15 for t in range(15)]:
                raise ValueError("Recorded cyclic map differs from the frozen offset definition")
        for metric in METRICS:
            values = [session["anchor_replay"][metric], *(row[metric] for row in session["offset_rows"])]
            if not np.isfinite(values).all() or not np.isfinite(session[metric]):
                raise ValueError("Finite source scores required without exclusions")
            if not np.isclose(np.mean(values[1:]), session[metric], rtol=1e-12, atol=1e-12):
                raise ArithmeticError("Stored fourteen-offset mean differs from its complete score rows")
        for offset, row in [(0, session["anchor_replay"]), *[(r["offset_bins"], r) for r in session["offset_rows"]]]:
            plotted.append({"level": "session", "animal": session["animal"], "session_id": session["session_id"],
                "condition": "aligned_anchor" if offset == 0 else "circular_offset", "offset_bins": offset,
                **{metric: row[metric] for metric in METRICS}})
        plotted.append({"level": "session", "animal": session["animal"], "session_id": session["session_id"],
            "condition": "mean_nonzero_offsets", "offset_bins": "1-14_mean",
            **{metric: session[metric] for metric in METRICS}})
    profiles = {}
    for animal in ANIMALS:
        selected = [s for s in sessions if s["animal"] == animal]
        profiles[animal] = {}
        for metric in METRICS:
            profiles[animal][metric] = np.mean([[s["anchor_replay"][metric],
                *(r[metric] for r in s["offset_rows"]), s[metric]] for s in selected], axis=0)
            summary_key = "primary_by_animal" if metric == "delta_R2" else "identity_by_animal"
            if (not np.isclose(profiles[animal][metric][-1], report["control"][summary_key][animal], rtol=1e-12, atol=1e-12)
                    or not np.isclose(profiles[animal][metric][0], report["reference_anchor"][summary_key][animal], rtol=1e-12, atol=1e-12)):
                raise ArithmeticError("Original animal control/anchor summary replay differs")
        for index in range(16):
            plotted.append({"level": "animal_mean", "animal": animal, "session_id": "all_original_sessions",
                "condition": "aligned_anchor" if index == 0 else ("mean_nonzero_offsets" if index == 15 else "circular_offset"),
                "offset_bins": "1-14_mean" if index == 15 else index,
                **{metric: float(profiles[animal][metric][index]) for metric in METRICS}})
    return sessions, profiles, plotted


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
        default=root/"runs/adaptive_search/primary_temporal_offset_attempt001/primary_temporal_offset.json")
    parser.add_argument("--output-stem", type=Path, default=root/"figures/primary_temporal_offset_attempt001")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render inside the coordinator's verified compute allocation")
    targets = {kind: args.output_stem.with_suffix("."+kind) for kind in ("png", "pdf", "csv")}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("Preserve existing temporal-offset figures and plotted values")
    report = json.loads(args.report.read_text())
    sessions, profiles, plotted = source_rows(report)
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8))
    specs = (("delta_R2", "A  Improvement over the direction/time baseline", "Primary increment in joint XY R²"),
             ("identity_increment", "B  Correct-trial versus wrong-trial pairing", "Identity increment in joint XY R²"))
    for ax, (metric, title, ylabel) in zip(axes, specs):
        ax.axvspan(-.35, .35, color=".93", zorder=0)
        ax.axvspan(15.45, 16.55, color=".96", zorder=0)
        for animal in ANIMALS:
            selected = [s for s in sessions if s["animal"] == animal]
            shift = -.075 if animal == "Mihili" else .075
            for index, session in enumerate(selected):
                values = [session["anchor_replay"][metric], *(r[metric] for r in session["offset_rows"])]
                ax.plot(range(15), values, color=COLORS[animal], lw=.75, alpha=.18, zorder=1)
                jitter = (index-(len(selected)-1)/2)*.014
                ax.scatter(16+shift+jitter, session[metric], color=COLORS[animal], s=11, alpha=.25, zorder=2)
            mean = profiles[animal][metric]
            ax.plot(range(15), mean[:15], color=COLORS[animal], marker="o", ms=3.5, lw=2.3,
                    label=f"{animal} · mean of {SESSION_COUNTS[animal]} sessions", zorder=3)
            ax.scatter(0, mean[0], s=64, facecolor="white", edgecolor=COLORS[animal], linewidth=1.8, zorder=4)
            ax.scatter(16+shift, mean[-1], color=COLORS[animal], marker="D", s=55, zorder=4)
        ax.axhline(0, color=".45", lw=.8, zorder=0)
        ax.set_xlim(-.55, 16.6)
        ax.set_xticks((0, 2, 4, 6, 8, 10, 12, 14, 16),
                      ("0\naligned", "2", "4", "6", "8", "10", "12", "14", "Mean\n1–14"))
        ax.set_xlabel("Circular bin offset k; rightmost point is the complete control mean")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", alpha=.14)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(.07, .917), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("EP05 · Temporal alignment of the fixed LMP decoder", x=.07, y=.985, ha="left", fontsize=17)
    fig.text(.07, .933, "Exposed development · 16 sessions, 2 animals · fixed anchor and penalty · qₖ(t) = (t + k) mod 15", fontsize=10.5)
    fig.text(.07, .155, "Faint traces and rightmost dots: all sessions. Solid curves: equal-session animal means. Hollow circles: aligned anchor; diamonds: control mean.", fontsize=9.4)
    fig.text(.07, .112, "The control averages 14 nonzero-offset correct/wrong SSEs, not predictions. Offsets are the entire finite set, not Monte Carlo or biological replicates.", fontsize=9.4)
    fig.text(.07, .071, "Circular wrapping can mix movement phases: k is not one physical lead/lag across every bin. No confidence intervals or promotion-threshold lines.", fontsize=9.4)
    fig.text(.07, .032, "Source: completed primary_temporal_offset_attempt001/primary_temporal_offset.json. No new fits, source arrays, exclusions or offset selection.", fontsize=9.2)
    fig.subplots_adjust(left=.075, right=.988, bottom=.29, top=.815, wspace=.29)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for kind in ("png", "pdf"):
        fig.savefig(targets[kind], dpi=180, facecolor="white")
    plt.close(fig)
    with targets["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    print(json.dumps({"source_report": str(args.report), "figures": [str(targets[k]) for k in ("png", "pdf")],
                      "plotted_csv": str(targets["csv"]), "plotted_rows": len(plotted),
                      "sessions": len(sessions), "new_fits": 0, "new_source_values_read": False}))


if __name__ == "__main__":
    main()
