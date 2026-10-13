#!/usr/bin/env python3
"""Plot the completed MZ0 primary, alternate-partition and LOO summaries."""
import argparse
import json
import math
from pathlib import Path

from qualify_assets import require_compute

EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs/behavioral_mechanism_kill_test"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    outputs = [args.output_dir / f"mz0_run_sensitivity.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve existing figures")
    primary = json.loads((RUNS / "20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json").read_text())
    if primary.get("status") != "necessary_primary_comparison_completed" or primary.get("replicate_index") != 0:
        raise ValueError("Require the completed original MZ0 primary comparison")
    rows = [("Primary 14 / 23", primary["comparison"], "#2563eb")]
    for partition in ("12_34", "13_24"):
        report = json.loads((RUNS / "20261005-MZ-other-partitions-46604164" / partition / "mz_other_partitions.json").read_text())
        if report.get("status") != "partition_diagnostic_completed" or report.get("replicate_index") != 0:
            raise ValueError("Require both completed MZ0 alternate partitions")
        rows.append(("Alternate " + partition.replace("_", " / "), report["nested_diagnostic"], "#2563eb"))
    pooled = json.loads((RUNS / "20261005-MZ-LOO-46605191/pooled_loo.json").read_text())
    if (pooled.get("pooled_diagnostic_complete") is not True or pooled.get("complete_held_out_runs") != [1, 2, 3, 4]
            or pooled.get("participants") != 108 or pooled.get("LOO_runs_are_independent_replications") is not False):
        raise ValueError("Require the complete four-run,108-participant LOO summary")
    rows.extend((f"Omitted run {row['held_out_run']}", row["nested_diagnostic"], "#64748b") for row in pooled["per_run"])
    rows.append(("LOO pooled within person", pooled["pooled_nested_diagnostic"], "#047857"))
    for _, row, _ in rows:
        if any(not math.isfinite(row[key]) for key in ("mean_gain_nats_per_trial", "participant_SE")) or row["participant_SE"] < 0:
            raise ValueError("Only finite saved means and participant SEs may be plotted")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(10, 6.3))
        fig.subplots_adjust(left=.29, right=.96, top=.79, bottom=.24)
        for y, (_, row, color) in enumerate(rows):
            ax.errorbar(row["mean_gain_nats_per_trial"], y, xerr=2 * row["participant_SE"],
                        fmt="o", color=color, capsize=4, markersize=6, linewidth=1.7)
        ax.axvline(0, color="#cbd5e1", linewidth=1)
        ax.axvline(.01, color="#b45309", linestyle="--", linewidth=1.5, label="Fixed 0.01 nat/trial margin")
        ax.axhline(2.5, color="#e2e8f0", linewidth=1)
        ax.set_yticks(range(len(rows)), [label for label, _, _ in rows])
        ax.invert_yaxis()
        ax.set_xlim(-.0005, .022)
        ax.set_xlabel("Held-out MZ − M0 log predictive gain (nat/trial)")
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        fig.suptitle("MZ0: predictive gain across run partitions", y=.97, fontsize=16, fontweight="bold")
        fig.text(.5, .90, "One synthetic cohort · 108 participants · saved completed comparisons", ha="center")
        fig.text(.29, .14, "Points: mean gain. Whiskers: ±2 participant standard errors.", fontsize=10)
        fig.text(.29, .095, "LOO averages four held-out run scores within each participant.", fontsize=10)
        fig.text(.29, .05, "Overlapping folds are sensitivities; full recovery and interval coverage remain unresolved.", fontsize=9, color="#475569")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
    for path in outputs:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
