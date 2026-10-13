#!/usr/bin/env python3
"""Render the completed top16 panel; no source reads or new fits."""
import csv
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render in the coordinated compute allocation")
    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_relevance_summary_attempt001/primary_relevance.json"
    report = json.loads(source.read_text())
    if report["status"] != "complete_relevance_control_panel":
        raise ValueError("Complete sixteen-session panel required")
    panels = {p["recipe"]: p["sessions"] for p in report["panels"]}
    groups = [report["matched_lambda_anchor_sessions"],
              panels["lmp_inner_lodo_top16_ridge_v1"],
              panels["lmp_inner_lodo_top16_rank_labels8_v1"]]
    names = ("Full LMP", "Top16 relevance", "Rank-label control\nmean of 8 models")
    keys = ("delta_R2", "identity_increment")
    animals = ("Mihili", "Chewie")
    colors = {"Mihili": "#276f94", "Chewie": "#bb6632"}
    ids = {r["session_id"] for r in groups[0]}
    if len(ids) != 16 or any(len(g) != 16 or {r["session_id"] for r in g} != ids for g in groups):
        raise ValueError("All original sessions must match")
    stem = root / "figures/primary_relevance_attempt001"
    if any(stem.with_suffix("." + ext).exists() for ext in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing figures")
    rows = []
    for group, name in zip(groups, names):
        for row in group:
            rows.append({"level": "session", "animal": row["animal"],
                         "session": row["session_id"], "condition": name.replace("\n", " "),
                         **{k: row[k] for k in keys}})
        for animal in animals:
            rows.append({"level": "animal_mean", "animal": animal, "session": "all",
                         "condition": name.replace("\n", " "),
                         **{k: float(np.mean([r[k] for r in group if r["animal"] == animal])) for k in keys}})
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6))
    for ax, key, title, margin in zip(axes, keys,
            ("A  Improvement over direction/time baseline", "B  Correct versus wrong trial pairing"), (.01, .005)):
        for animal in animals:
            selected = [r for r in groups[0] if r["animal"] == animal]
            values = np.array([[next(r[key] for r in g if r["session_id"] == s["session_id"])
                                for g in groups] for s in selected])
            if not np.isfinite(values).all():
                raise ValueError("Finite complete scores required")
            for vector in values:
                ax.plot(range(3), vector, color=colors[animal], alpha=.23, lw=.8)
            ax.plot(range(3), values.mean(axis=0), color=colors[animal], lw=2.5,
                    marker="o", ms=7, label=f"{animal}: mean of {len(selected)} sessions")
        ax.axhline(0, color=".5", lw=.8)
        ax.axhline(margin, color=".3", ls="--", lw=1)
        ax.text(.98, margin, f" Practical margin {margin:g}", va="bottom", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=9, color=".3")
        ax.set(title=title, ylabel="Increment in joint XY R²", xticks=range(3),
               xticklabels=names, xlim=(-.15, 2.15))
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle("Top16 relevance loses to full-channel LMP in both animals", fontsize=16,
                 x=.065, ha="left", y=.98)
    fig.text(.065, .025,
             "Exposed development evidence · 2 animals · thin lines: all 16 sessions · no confidence intervals\n"
             "All three use candidate-selected λ (1 in 15 sessions; 0.1 in 1). Top16 versus control fails in Mihili.",
             fontsize=10, color=".3")
    fig.tight_layout(rect=(.03, .11, .995, .925))
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + ext), dpi=180)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "rows": len(rows), "source": str(source)}))


if __name__ == "__main__":
    main()
