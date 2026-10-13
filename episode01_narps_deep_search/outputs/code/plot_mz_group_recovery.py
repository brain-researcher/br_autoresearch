#!/usr/bin/env python3
"""Display a completed200-replicate MZ point-bias summary, never source data.

The distributions describe variation across recovery simulations, not confidence
intervals. Only the explicitly supplied summary JSON is read; its report paths
are not followed. No fitting, interval computation or scheduler action occurs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from qualify_assets import require_compute


def completed_values(summary):
    if (summary.get("phase") != "MZ_SESOI_group_point_bias_summary"
            or summary.get("generating_model") != "MZ"
            or summary.get("status") != "complete_fixed_cell_point_bias"
            or summary.get("group_point_bias_complete") is not True
            or summary.get("statistics_provisional") is not False
            or summary.get("planned_cell_replicates") != 200
            or summary.get("valid_replicates") != 200
            or summary.get("remaining_unique_valid_replicates") != 0
            or summary.get("true_rejection_displacement") != .05
            or summary.get("relative_bias_limit") != .10
            or summary.get("interval_coverage_evaluated") is not False
            or summary.get("full_recovery_gate") != "not_evaluated"):
        raise ValueError("Only a completed fixed200-replicate MZ point-bias summary is supported")
    rows = summary.get("valid_reports")
    if (not isinstance(rows, list) or len(rows) != 200
            or any(type(row.get("replicate_index")) is not int for row in rows)
            or {row["replicate_index"] for row in rows} != set(range(200))):
        raise ValueError("All200 distinct original replicate indices are required")
    rows = sorted(rows, key=lambda row: row["replicate_index"])
    values = {}
    for version in ("EI", "ER"):
        values[version] = [row["rejection_displacement"][version] for row in rows]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values[version]):
            raise ValueError("Every displayed replicate effect must be finite")
        mean = math.fsum(values[version]) / 200
        statistics = summary["version_statistics"][version]
        if (statistics["mean_rejection_displacement"] != mean
                or statistics["absolute_relative_bias"] != abs(mean - .05) / .05):
            raise ValueError("The saved mean/bias does not match the displayed replicate distribution")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Select a PNG, PDF or SVG output")
    if args.output.exists():
        raise FileExistsError("Preserve existing figures; select a fresh output path")
    summary = json.loads(args.summary.read_text())
    values = completed_values(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low = min(.045, min(values["EI"]), min(values["ER"]))
    high = max(.055, max(values["EI"]), max(values["ER"]))
    padding = .08 * (high - low)
    lower, upper = low - padding, high + padding
    bins = [lower + (upper - lower) * index / 20 for index in range(21)]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
        figure.subplots_adjust(left=.08, right=.98, top=.76, bottom=.24, wspace=.12)
        for ax, version, color in zip(axes, ("EI", "ER"), ("#2563eb", "#b45309")):
            statistics = summary["version_statistics"][version]
            mean = statistics["mean_rejection_displacement"]
            ax.hist(values[version], bins=bins, color=color, alpha=.62, edgecolor="white", linewidth=.5)
            ax.axvline(.05, color="#111827", linestyle="--", linewidth=1.7, label="True effect: 0.050")
            ax.axvline(mean, color=color, linewidth=2.2, label=f"Mean: {mean:.5f}")
            ax.set_title(f"{version} · 200 replicates", loc="left", fontweight="bold")
            ax.set_xlabel("Estimated rejection displacement (0.5 − mean z)")
            ax.set_xlim(lower, upper)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, loc="upper left", fontsize=9)
            ax.text(.98, .98, f"Mean relative bias\n{statistics['absolute_relative_bias']:.2%}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
        axes[0].set_ylabel("Recovery replicates")
        figure.suptitle("MZ group-effect recovery at the fixed 0.05 effect", y=.96, fontsize=15, fontweight="bold")
        figure.text(.5, .875, "54 participants per task version · both primary training halves · fixed relative-bias limit: 10%",
                    ha="center", fontsize=10)
        figure.text(.08, .105, "Histograms show variability across simulations; they are not confidence intervals.", fontsize=10)
        figure.text(.08, .06, "95% interval coverage and full behavioral recovery are not evaluated.", fontsize=10, color="#4b5563")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180, facecolor="white")
        plt.close(figure)
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
