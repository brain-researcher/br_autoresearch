#!/usr/bin/env python3
"""Plot released PLS2 scalar aggregates only; never open banks or source arrays."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket


SUMMARY_RELATIVE = "runs/adaptive_search/primary_history_pls_summary_attempt001/compact_evidence_attempt001.json"
MAX_SUMMARY_BYTES = 512*1024
POLICIES = ("HW", "blend", "PLS2", "mean8_target_shuffle")
LABELS = ("Weighted history (HW)", "Current/history blend", "Two-component PLS", "Mean 8 target shuffles")
ANIMALS = ("Mihili", "Chewie")
METRICS = (
    ("primary_increment", "primary_by_animal", .01, "Primary: full velocity − baseline"),
    ("identity_increment", "identity_by_animal", .005, "Identity: correct − wrong trial"),
)


def released_aggregates(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SUMMARY_BYTES+1)
    if len(raw) > MAX_SUMMARY_BYTES:
        raise ValueError("Only the compact evidence artifact of at most 512 KiB may be read")
    document = json.loads(raw)
    if (document.get("status") != "complete_history_pls_panel"
            or document.get("promotion_authorized") is not False
            or any(document.get(key) is not True for key in ("all_new_saved_scores_replayed",
                "all_original_anchor_scores_replayed", "original_selectors_independently_replayed"))):
        raise ValueError("Complete replayed compact PLS evidence is required")
    records = {name: document["original_anchors"][name] for name in ("HW", "blend")}
    records.update(PLS2=document["candidate"], mean8_target_shuffle=document["control"])
    rows = []
    for policy in POLICIES:
        for aggregate_key, animal_key, threshold, _ in METRICS:
            mean = float(records[policy][aggregate_key])
            by_animal = records[policy][animal_key]
            if set(by_animal) != set(ANIMALS):
                raise ValueError("Exactly the two original animal summaries are required")
            values = [float(by_animal[animal]) for animal in ANIMALS]
            if not all(math.isfinite(value) for value in (mean, *values)):
                raise ValueError("Nonfinite released aggregate")
            if not math.isclose(mean, sum(values)/2, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Released mean does not preserve equal animal weighting")
            for unit, value in (("animal_balanced_mean", mean), *zip(ANIMALS, values)):
                rows.append({"policy": policy, "metric": aggregate_key, "unit": unit,
                    "value": value, "registered_margin": threshold,
                    "evidence": "exposed_development_two_animals",
                    "source": "outputs/"+SUMMARY_RELATIVE})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        help="Must be outputs/figures/primary_history_pls; writes PNG, SVG and exact plotted CSV")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render in the coordinator's verified Slurm compute allocation")
    outputs = Path(__file__).resolve().parents[1]
    expected_output = outputs/"figures/primary_history_pls"
    output = args.output_dir.resolve() if args.output_dir else expected_output
    if output != expected_output:
        raise ValueError("Only the assigned PLS figure directory is permitted")
    rows = released_aggregates(outputs/SUMMARY_RELATIVE)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False})
    lookup = {(row["policy"], row["metric"], row["unit"]): row["value"] for row in rows}
    # One common axis covers every displayed value and both registered margins.
    # Extend the former HW105 display range if needed; never clip an animal.
    values = [row["value"] for row in rows]
    lower = min(-.002, math.floor(min(values)/.002)*.002)
    upper = max(.012, math.ceil(max(values)/.002)*.002)
    colors = {"Mihili": "#2674A8", "Chewie": "#C56C23"}
    offsets = {"Mihili": .12, "Chewie": -.12}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.24, right=.975, top=.76, bottom=.35, wspace=.13)
    fig.suptitle("EP05 · two-component PLS on weighted history", x=.08, ha="left", y=.96,
                 fontsize=16, fontweight="bold")
    fig.text(.08, .90, "16 sessions · two animals · held-out directions · exposed development evidence",
             ha="left", color="#555555", fontsize=10)
    for ax, (metric, _, threshold, title) in zip(axes, METRICS):
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.axvline(0, color="#808080", linewidth=.9, zorder=0)
        ax.axvline(threshold, color="#6C6C6C", linestyle="--", linewidth=1.2, zorder=0)
        ax.text(threshold-.00015, .99, f"margin {threshold:.3f}",
                transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=9, color="#555555")
        for index, policy in enumerate(POLICIES):
            y = len(POLICIES)-1-index
            for animal in ANIMALS:
                ax.scatter(lookup[policy, metric, animal], y+offsets[animal], s=34,
                    color=colors[animal], edgecolors="white", linewidths=.5, zorder=3)
            ax.scatter(lookup[policy, metric, "animal_balanced_mean"], y, marker="D", s=38,
                color="#252525", edgecolors="white", linewidths=.6, zorder=4)
        ax.set_xlim(lower, upper)
        ax.set_ylim(-.55, len(POLICIES)-.35)
        # Keep the familiar 0.002 step while maintaining legibility if a
        # released aggregate requires a materially wider common axis.
        step = .002 if upper-lower <= .020 else (upper-lower)/7
        ticks = [lower+i*step for i in range(int(round((upper-lower)/step))+1)]
        ax.set_xticks(ticks, [f"{tick:.3f}" for tick in ticks], fontsize=8)
        ax.set_yticks(range(len(POLICIES)), list(reversed(LABELS)))
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", alpha=.16, linewidth=.6)
        ax.set_xlabel(r"Increment in $R^2_{SSE}$", labelpad=9)

    legend = [Line2D([], [], color="#252525", marker="D", linestyle="none", markersize=6,
                     label="Animal-balanced mean")]
    legend += [Line2D([], [], color=colors[animal], marker="o", linestyle="none", markersize=6,
                      label=animal) for animal in ANIMALS]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.07, .21), ncol=3,
               frameon=False, fontsize=10)
    fig.text(.08, .18, "Blend: current/history ridge (W/HW). Reference penalties selected independently; controls use the PLS penalty.",
             fontsize=9, color="#444444")
    fig.text(.08, .13, "Target shuffles refit both the supervised projection and ridge head. Displayed control is the mean of 8 losses.",
             fontsize=9, color="#444444")
    fig.text(.08, .08, "Control flexibility can differ. Dots are animal summaries, not confidence intervals; 8 shuffles are not biological replicates.",
             fontsize=9, color="#444444")
    primary = lookup["PLS2", "primary_increment", "animal_balanced_mean"]
    identity = lookup["PLS2", "identity_increment", "animal_balanced_mean"]
    primary_word = "meets" if primary >= .01 else "is below"
    identity_word = "meets" if identity >= .005 else "is below"
    fig.text(.08, .035, f"PLS primary {primary:.6f} {primary_word} 0.010; identity {identity:.6f} {identity_word} 0.005. No population inference.",
             fontsize=9, color="#444444")
    stem = output/"primary_history_pls_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output), "policies": 4, "animals": 2,
        "tidy_rows": len(rows), "formats": ["png", "svg", "csv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
