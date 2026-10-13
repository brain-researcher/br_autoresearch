#!/usr/bin/env python3
"""Plot complete-report initializer sensitivity; no source or partial-report reads."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_initializer_robustness import ANIMALS, CONDITIONS, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render only in verified coordinator compute")
    paths = {kind: args.output_prefix.with_suffix("."+kind) for kind in ("png", "pdf", "csv")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Preserve prior figures and choose a fresh output prefix")
    summary = extract(json.loads(args.report.read_text()))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    means = {(condition["condition"], animal["animal"]): animal
             for condition in summary["conditions"] for animal in condition["animals"]}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True)
    colors = plt.get_cmap("tab10")
    for ax, animal in zip(axes, ANIMALS):
        rows = [r for r in summary["session_rows"] if r["animal"] == animal]
        sessions = sorted({r["filename"] for r in rows})
        ax.axhline(0, color="#555555", linewidth=1)
        ax.axvspan(-.42, .42, color="#eeeeee", zorder=0)
        ax.axvspan(3.58, 4.42, color="#eaf2fa", zorder=0)
        for index, filename in enumerate(sessions):
            values = {r["condition"]: r["contrast"] for r in rows if r["filename"] == filename}
            offset = (index-1.5)*.055
            ax.scatter([x+offset for x in range(5)], [values[c] for c in CONDITIONS],
                       s=31, color=colors(index), alpha=.85, zorder=3,
                       label=filename.rsplit("_", 1)[-1].replace(".mat", ""))
        for x, condition in enumerate(CONDITIONS):
            mean = means[condition, animal]
            low, high = mean["omission_ranges"]["contrast"]
            ax.vlines(x, low, high, color="black", linewidth=1.5, zorder=4)
            ax.hlines([low, high], x-.055, x+.055, color="black", linewidth=1.2, zorder=4)
            ax.scatter([x], [mean["contrast"]], marker="D", s=36, color="black", zorder=5)
        ax.set_title(animal, fontsize=13, fontweight="bold")
        ax.set_xticks(range(5), ["Original\ncontext", "New order 0", "New order 1", "New order 2", "Mean of\n3 new orders"])
        ax.set_ylabel("Candidate − matched label control (Q)")
        ax.set_xlim(-.55, 4.55)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Development session", loc="best", fontsize=8, title_fontsize=8, frameon=False)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle("Does residual marginal selection survive different initializers?", fontsize=15, y=.97)
    fig.text(.5, .895, "Points: four sessions per animal. Black diamond: animal mean. Bars: fixed session-omission range.",
             ha="center", fontsize=9)
    screens = summary["prospective_directional_screens"]
    average = "PASS" if screens["mean_three_order_correspondence_positive_in_both_animals"] else "FAIL"
    stable = "PASS" if screens["correspondence_positive_in_both_animals_in_every_order"] else "FAIL"
    fig.text(.5, .105, f"Mean-order both-animal screen: {average}     Each-order both-animal sensitivity screen: {stable}",
             ha="center", fontsize=10)
    fig.text(.5, .045,
             "Original is separate context: training prefixes, nulls and CV change. Mean-five SSE within order, then mean-three Q.\n"
             "Orders overlap; two animals and exposed development sessions. Bars are not confidence intervals. Held sessions remain closed.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(top=.82, bottom=.24, left=.075, right=.975, wspace=.28)
    fig.savefig(paths["png"], dpi=180, facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    plt.close(fig)
    fields = ["row_kind", "condition", "animal", "filename", "candidate_Q", "control_Q", "contrast",
              "direction_Q", "candidate_minus_direction", "fixed_session_omission_low", "fixed_session_omission_high"]
    rows = [{"row_kind": "session", **row} for row in summary["session_rows"]]
    for condition in summary["conditions"]:
        for animal in condition["animals"]:
            rows.append({"row_kind": "animal_mean", "condition": condition["condition"], "animal": animal["animal"],
                **{key: animal[key] for key in ("candidate_Q", "control_Q", "contrast", "direction_Q", "candidate_minus_direction")},
                "fixed_session_omission_low": animal["omission_ranges"]["contrast"][0],
                "fixed_session_omission_high": animal["omission_ranges"]["contrast"][1]})
    with paths["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"outputs": {kind: str(path) for kind, path in paths.items()},
                      "csv_rows": len(rows), "source_report": str(args.report),
                      "within_order_full96_replay": summary["within_order_full96_replay"]}))


if __name__ == "__main__":
    main()
