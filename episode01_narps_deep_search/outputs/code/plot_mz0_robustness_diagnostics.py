#!/usr/bin/env python3
"""Plot three completed, saved MZ0 predictive contrasts without recomputation."""
import argparse
import json
import math
from pathlib import Path

from qualify_assets import require_compute

EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs/behavioral_mechanism_kill_test"
SOURCE = RUNS / "20261005-MZ-SESOI-replicate000-attempt001"
COMMON = RUNS / "20261005-MZ-common-gain-MZ-reverse-repair-46617879/common_gain_mz_reverse_repaired_summary.json"
NONLINEAR = RUNS / "20261005-MZ-nonlinear-M0-directions-repair-46618614/nonlinear_m0_directions_repaired_summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    outputs = [args.output_dir / f"mz0_robustness_diagnostics.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve existing figures")

    primary = json.loads((SOURCE / "single_process_recovery.json").read_text())
    common = json.loads(COMMON.read_text())
    nonlinear = json.loads(NONLINEAR.read_text())
    identities = (
        (primary, "MZ_SESOI_fitted_recovery_component", "necessary_primary_comparison_completed"),
        (common, "MZ0_common_gain_reverse_MZ_repaired_nested_diagnostic", "common_gain_nested_diagnostic_completed"),
        (nonlinear, "MZ0_nonlinear_M0_independent_directions_and_preserved_MZ_nested_diagnostic", "nonlinear_pair_diagnostic_completed"),
    )
    for report, phase, status in identities:
        if (report.get("phase") != phase or report.get("status") != status
                or report.get("generating_model") != "MZ" or report.get("replicate_index") != 0
                or report.get("participants") != 108):
            raise ValueError("Require the three selected completed MZ0 contrasts")
    for report in (common, nonlinear):
        if report.get("pooled_diagnostic_complete") is not True or report.get("source_dir") != str(SOURCE):
            raise ValueError("Require completed sensitivity summaries from the same original cohort")
    if (primary.get("trials") != 27648 or common.get("gain_range_dollars") != [10, 20]
            or common.get("filter_boundary_inclusive") is not True
            or nonlinear.get("combined_heldout_trials_per_participant") != 256):
        raise ValueError("Require the selected full-session and common-gain supports")

    rows = [
        ("Primary fixed bound\nFull 256 trials per participant", primary["comparison"], "#2563eb"),
        ("Common gain $10–$20\nEI: 96 / ER: 176 retained trials", common["common_gain_nested_diagnostic"], "#047857"),
        ("Nonlinear value · full 256 trials\nCurvature-adjusted M0 reference", nonlinear["nonlinear_nested_diagnostic"], "#7c3aed"),
    ]
    for _, row, _ in rows:
        if (any(not isinstance(row[key], (int, float)) or isinstance(row[key], bool)
                or not math.isfinite(row[key]) for key in ("mean_gain_nats_per_trial", "participant_SE"))
                or row["participant_SE"] < 0):
            raise ValueError("Only finite saved means and participant SEs may be plotted")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.subplots_adjust(left=.36, right=.96, top=.77, bottom=.34)
        for y, (_, row, color) in enumerate(rows):
            ax.errorbar(row["mean_gain_nats_per_trial"], y, xerr=2 * row["participant_SE"],
                        fmt="o", color=color, capsize=4, markersize=7, linewidth=1.7)
        ax.axvline(0, color="#cbd5e1", linewidth=1)
        ax.axvline(.01, color="#b45309", linestyle="--", linewidth=1.5,
                   label="Fixed 0.01 nat/trial margin")
        ax.set_yticks(range(len(rows)), [label for label, _, _ in rows])
        ax.set_ylim(2.5, -.5)
        ax.set_xlim(-.0005, .021)
        ax.set_xlabel("Held-out MZ − M0 log predictive gain (nat/trial)")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        fig.suptitle("MZ0: completed predictive sensitivity diagnostics", y=.96,
                     fontsize=16, fontweight="bold")
        fig.text(.5, .88, "One saved synthetic cohort · same 108 participants · primary run partition",
                 ha="center")
        fig.text(.09, .22, "Points: saved mean gain. Bars: ±2 participant SE; not formal confidence intervals.", fontsize=10)
        fig.text(.09, .165, "Fixed nested rule: mean ≥0.01 and mean >2SE; bars need not lie entirely above 0.01.", fontsize=10)
        fig.text(.09, .11, "These overlapping analyses are not independent replicates; denominators differ as labeled.", fontsize=10)
        fig.text(.09, .055, "No interval qualification or full robustness verdict. Full gate remains pending.",
                 fontsize=10, color="#475569")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
    for path in outputs:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
