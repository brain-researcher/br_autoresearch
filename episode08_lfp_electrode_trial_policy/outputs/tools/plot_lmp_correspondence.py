#!/usr/bin/env python3
"""Plot only the completed native-LMP correspondence report; no source access."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def short_name(filename):
    animal, _, date = filename.removesuffix(".mat").split("_")
    return ("Mi" if animal == "Mihili" else "Ch")+" "+date


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if (report["status"] != "completed_two_LMP_correspondence_comparisons"
            or report["valid_evaluations_total"] != 28 or report["held_session_accessed"]
            or len(report["sessions"]) != 8 or len(report["fixed_prediction_session_omission"]) != 8
            or len(report["seed_summaries"]) != 5):
        raise ValueError("Complete eight-session/five-seed development result required")
    for suffix in (".png", ".pdf", ".csv"):
        if args.output_prefix.with_suffix(suffix).exists():
            raise ValueError("Preserve existing plot artifacts; select a fresh output prefix")
    policy, control = report["policy"], report["control"]
    candidate = next(panel for panel in report["panels"] if panel["policy"] == policy)
    panels = (
        ("A", "LMP true selection − its label control", "mean_delta_Q_from_"+control),
        ("B", "LMP true selection − direction means", "mean_delta_Q_from_direction_only"),
        ("C", "LMP true selection − three-band true selection", "mean_delta_Q_from_initial_marginal"))
    colors = {"Mihili": "#2077a8", "Chewie": "#b96b20"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    plotted = []
    for ax, (letter, title, metric) in zip(axes.flat, panels):
        labels = []
        for index, session in enumerate(report["sessions"]):
            value = session["panels"][policy][metric]
            labels.append(short_name(session["filename"]))
            ax.plot(value, index, "o", color=colors[session["animal"]], markersize=5)
            plotted.append({"panel": letter, "kind": "session", "identity": session["filename"],
                            "animal": session["animal"], "contrast": metric, "value": value})
        for offset, animal in enumerate(("Mihili", "Chewie")):
            recorded = next(item[metric] for item in candidate["animals"] if item["animal"] == animal)
            replay = float(np.mean([s["panels"][policy][metric] for s in report["sessions"] if s["animal"] == animal]))
            if not np.isclose(recorded, replay, atol=1e-12, rtol=1e-12):
                raise ArithmeticError("Plotted animal mean does not replay its sessions")
            labels.append(animal+" mean")
            ax.plot(recorded, 8+offset, "D", color=colors[animal], markersize=7)
            plotted.append({"panel": letter, "kind": "animal_mean", "identity": animal,
                            "animal": animal, "contrast": metric, "value": recorded})
        ax.axvline(0, color="0.45", linewidth=.8)
        ax.axhline(7.5, color="0.85", linewidth=.7)
        ax.set_yticks(range(10), labels)
        ax.invert_yaxis()
        ax.set_title(letter+"  "+title, loc="left", fontsize=11)
        ax.set_xlabel("Difference in Q (right favors the LMP rule)")
        ax.grid(axis="x", alpha=.2)
    ax = axes[1, 1]
    metric = "mean_delta_Q_from_"+control
    full = candidate[metric]
    labels, values, kinds = [], [], []
    for omission in report["fixed_prediction_session_omission"]:
        value = next(panel[metric] for panel in omission["panels"] if panel["policy"] == policy)
        labels.append("omit "+short_name(omission["excluded_session"]))
        values.append(value)
        kinds.append("omission")
        plotted.append({"panel": "D", "kind": "omission", "identity": omission["excluded_session"],
                        "animal": "equal_animal", "contrast": metric, "value": value})
    for seed in report["seed_summaries"]:
        labels.append("control seed "+str(seed["mc_seed"]))
        values.append(seed["true_minus_seed_Q"])
        kinds.append("seed")
        plotted.append({"panel": "D", "kind": "seed", "identity": str(seed["mc_seed"]),
                        "animal": "equal_animal", "contrast": "true_minus_seed_Q", "value": seed["true_minus_seed_Q"]})
    for index, (value, kind) in enumerate(zip(values, kinds)):
        ax.plot(value, index, "o" if kind == "omission" else "s",
                color="#41626f" if kind == "omission" else "#855797", markersize=5)
    ax.axvline(0, color="0.45", linewidth=.8)
    ax.axvline(full, color="#b96b20", linestyle="--", linewidth=1.1, label="full mean-five control contrast")
    ax.axhline(7.5, color="0.85", linewidth=.7)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title("D  Correspondence sensitivity: overall contrasts", loc="left", fontsize=11)
    ax.set_xlabel("True selection − control Q; equal animal weighting")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    plotted.append({"panel": "D", "kind": "full", "identity": "all_sessions_mean_five_control",
                    "animal": "equal_animal", "contrast": metric, "value": full})
    fig.suptitle("Native LMP supports residual prediction; its electrode-ranking advantage is fragile", fontsize=15)
    fig.text(.5, .02,
             "EP08 development only: 8 sessions, 2 previously exposed animals; 4 held sessions unopened.\n"
             "Points are observed session/animal means, fixed-prediction omissions or ordinary control seeds; no confidence intervals.\n"
             "Removing high-frequency power changes input dimension and capacity; native LMP is not certified spike-free.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .96))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("panel", "kind", "identity", "animal", "contrast", "value"))
        writer.writeheader()
        writer.writerows(plotted)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=160)
    fig.savefig(args.output_prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({"plotted_rows": len(plotted), "report": str(args.report),
                      "output_prefix": str(args.output_prefix), "new_fits": 0, "held_session_accessed": False}))


if __name__ == "__main__":
    main()
