#!/usr/bin/env python3
"""Plot only released EP05 summary aggregates; no source arrays or refitting."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket


SUMMARY_RELATIVE = "runs/adaptive_search/primary_history_rbf_summary_attempt001/compact_evidence_attempt001.json"
POLICIES = ("HW", "K", "blend", "HW105", "mean8_control")
LABELS = ("HW", "K", "Blend", "HW105", "Mean 8 controls")
ANIMALS = ("Mihili", "Chewie")
METRICS = (
    ("primary_increment", "primary_by_animal", 0.01, "Primary: full velocity − baseline"),
    ("identity_increment", "identity_by_animal", 0.005, "Identity: correct − wrong trial"),
)
AXIS_LIMITS = (-0.002, 0.012)


def released_aggregates(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Released aggregate summary exceeds its bounded reader")
    document = json.loads(raw)
    if (document.get("status") != "complete_history_rbf_panel"
            or document.get("all_new_saved_scores_replayed") is not True
            or document.get("all_original_anchor_scores_replayed") is not True
            or document.get("original_selectors_independently_replayed") is not True):
        raise ValueError("Complete replayed compact evidence is required")
    # Project only the specifically released aggregate fields. No session
    # arrays, predictions, per-draw records or other report fields are used.
    records = {name: document["original_anchors"][name] for name in ("HW", "K", "blend")}
    records.update(HW105=document["candidate"], mean8_control=document["control"])
    rows = []
    for policy in POLICIES:
        for aggregate_key, animal_key, threshold, _ in METRICS:
            mean = float(records[policy][aggregate_key])
            by_animal = records[policy][animal_key]
            if set(by_animal) != set(ANIMALS):
                raise ValueError("Exactly Mihili and Chewie animal summaries are required")
            values = [float(by_animal[animal]) for animal in ANIMALS]
            if not all(math.isfinite(value) for value in [mean] + values):
                raise ValueError("Nonfinite released aggregate")
            if not math.isclose(mean, sum(values) / 2, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Released mean is inconsistent with equal animal weighting")
            if any(not AXIS_LIMITS[0] <= value <= AXIS_LIMITS[1] for value in [mean] + values):
                raise ValueError("Released aggregate exceeds the fixed common display axis")
            for unit, value in [("animal_balanced_mean", mean)] + list(zip(ANIMALS, values)):
                rows.append({"policy": policy, "metric": aggregate_key, "unit": unit,
                             "value": value, "registered_margin": threshold,
                             "evidence": "exposed_development_two_animals",
                             "source": "outputs/" + SUMMARY_RELATIVE})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Must be outputs/figures/primary_history_rbf; regenerates the three fixed figure exports")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render in the coordinator's verified Slurm compute allocation")
    outputs = Path(__file__).resolve().parents[1]
    expected_output = outputs / "figures/primary_history_rbf"
    output = args.output_dir.resolve() if args.output_dir else expected_output
    if output != expected_output:
        raise ValueError("Only the assigned figure directory is permitted")
    rows = released_aggregates(outputs / SUMMARY_RELATIVE)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    lookup = {(row["policy"], row["metric"], row["unit"]): row["value"] for row in rows}
    colors = {"Mihili": "#2674A8", "Chewie": "#C56C23"}
    offsets = {"Mihili": 0.12, "Chewie": -0.12}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.975, top=0.76, bottom=0.34, wspace=0.12)
    fig.suptitle("EP05 · development evidence in two animals", x=0.15, ha="left", y=0.96,
                 fontsize=16, fontweight="bold")
    fig.text(0.15, 0.90, "Held-out directions · animal-balanced session summaries · no population inference",
             ha="left", color="#555555", fontsize=10)
    for ax, (metric, _, threshold, title) in zip(axes, METRICS):
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.axvline(0, color="#808080", linewidth=0.9, zorder=0)
        ax.axvline(threshold, color="#6C6C6C", linestyle="--", linewidth=1.2, zorder=0)
        ax.text(threshold - 0.00015, 0.99, f"margin {threshold:.3f}",
                transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=9, color="#555555")
        for index, policy in enumerate(POLICIES):
            y = len(POLICIES) - 1 - index
            for animal in ANIMALS:
                ax.scatter(lookup[policy, metric, animal], y + offsets[animal],
                           s=31, color=colors[animal], edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter(lookup[policy, metric, "animal_balanced_mean"], y,
                       marker="D", s=34, color="#252525", edgecolors="white", linewidths=0.6, zorder=4)
        ax.set_xlim(*AXIS_LIMITS)
        ax.set_ylim(-0.55, 4.65)
        ticks = (-0.002, 0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012)
        ax.set_xticks(ticks, [f"{tick:.3f}" for tick in ticks], fontsize=8)
        ax.set_yticks(range(5), list(reversed(LABELS)))
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", alpha=0.16, linewidth=0.6)
        ax.set_xlabel(r"Increment in $R^2_{SSE}$", labelpad=9)

    legend = [Line2D([], [], color="#252525", marker="D", linestyle="none", markersize=6,
                     label="Animal-balanced mean")]
    legend += [Line2D([], [], color=colors[animal], marker="o", linestyle="none", markersize=6,
                      label=animal) for animal in ANIMALS]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.14, 0.15), ncol=3,
               frameon=False, fontsize=10)
    fig.text(0.15, 0.12, "HW: weighted history   K: current-bin RBF   Blend: current/history ridge   HW105: history RBF",
             fontsize=9, color="#444444")
    fig.text(0.15, 0.075, "Controls: mean of 8 training-pairing surrogates. Dots are animal summaries, not confidence intervals.",
             fontsize=9, color="#444444")
    fig.text(0.15, 0.035, "All five primary means are below 0.01. HW105 is below both margins; HW exceeds the identity margin only.",
             fontsize=9, color="#444444")

    stem = output / "primary_history_rbf_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output), "policies": 5, "animals": 2,
                      "tidy_rows": len(rows), "formats": ["png", "svg", "csv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
