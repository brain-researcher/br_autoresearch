"""Render only the completed EP06 LMP support report; no fit or inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_lmp_support_followup as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run.initial._compute_only()
    if args.report.resolve().parent.parent != run.ANALYSIS:
        raise PermissionError("read the named completed EP06 follow-up only")
    if args.output.parent.resolve() != run.initial.EPISODE / "outputs/figures/lmp_support_followup":
        raise PermissionError("write only a fresh EP06 figure attempt")
    report = json.loads(args.report.read_text())
    if report["program"] != run.program():
        raise ValueError("report does not match the fixed follow-up")
    args.output.mkdir()
    keys = [f"{support}/{method}" for support in ("all_units", "on_electrodes", "disjoint_electrodes")
            for method in run.METHODS]
    labels = [f"{s}\n{m}" for s in ("All units", "Matched same electrode", "Matched disjoint electrode")
              for m in ("LMP", "100–400 Hz", "LMP after HF projection")]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    for ax, metric, title in zip(axes, ("score", "minus_mismatch"),
            ("Residual fraction recovered", "Gain over 199 whole-trial mismatches")):
        values = np.full((7, 9), np.nan)
        for column, key in enumerate(keys):
            for row, entry in enumerate(report["summary"][key]["dates"]):
                if entry["values"] is not None:
                    values[row, column] = entry["values"][metric]
        maximum = max(.01, float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else .01
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#dddddd")
        shown = ax.imshow(values, cmap=cmap, vmin=-maximum, vmax=maximum, aspect="auto")
        ax.set_xticks(range(9), labels, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(7), [f"{s.animal} {s.filename.split('_')[-1][:-4]}" for s in run.SESSIONS], fontsize=9)
        ax.set_title(title)
        ax.axhline(2.5, color="black", linewidth=1)
        for row in range(7):
            for column in range(9):
                value = values[row, column]
                ax.text(column, row, "NA" if not np.isfinite(value) else f"{value:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(shown, ax=ax, shrink=.65)
    fig.suptitle("EP06 exploratory M1 LMP controls — seven exposed dates, two animals\n"
                 "Unit draws averaged within date; changed targets and HF removal do not identify contamination")
    fig.savefig(args.output / "lmp_support.png", dpi=180)
    fig.savefig(args.output / "lmp_support.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
