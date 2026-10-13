#!/usr/bin/env python3
"""Uniform half/twice-SESOI MZ baseline components using the frozen four-fit loop.

Only prospective focal truth, seed and report/diagnostic scope differ. Runner
capability for fixed indices0–199 does not select their execution or a wave.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_single_process_recovery as core


LEVELS = {"half": {"multiplier": .5, "z": .475, "base_seed": 1700047},
          "twice": {"multiplier": 2., "z": .40, "base_seed": 1900057}}
SEED_STRIDE, FIXED_COHORTS, MAX_SECONDS = 10007, 200, 5400
REPORT_NAME = "mz_effect_level_recovery.json"
CLAIM_FIELDS = ("family_selection", "family_recovery_rate", "confusion_rate", "group_recovery",
                "interval_coverage", "heterogeneity_qualification", "full_gate", "early_kill")


def level_seed(level, index):
    if level not in LEVELS or type(index) is not int or not 0 <= index < FIXED_COHORTS:
        raise ValueError("Only half/twice levels and fixed indices0 through199 are defined")
    return LEVELS[level]["base_seed"] + SEED_STRIDE * index


def focal_truth(level):
    level_seed(level, 0)
    z = LEVELS[level]["z"]
    return {"eta": 0., "rho": math.log(z / (1 - z)), "criterion": 0.}


def criterion_scope(level):
    applies = LEVELS[level]["multiplier"] >= 1
    return {"recovery_90pct_criterion_applies": applies,
            "required_recovery_fraction": .90 if applies else None,
            "recovery_criterion_scope": "at_or_above_SESOI_necessary_component_only" if applies
                else "below_SESOI_diagnostic_only_no_90pct_pass_failure_or_curtailment"}


def run(args):
    require_compute()
    level_seed(args.level, args.replicate_index)
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Preserve the original1 through5400-second process cap")
    if args.design_events.resolve() != core.DESIGN_SOURCE.resolve():
        raise ValueError("Only the existing seven-column MC attempt003 design source is selected")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Each worker requires one allocated CPU and numerical thread limits of one")
    level = LEVELS[args.level]
    scope = criterion_scope(args.level)
    original_summary, original_writer = core.summarize_paired_scores, core.write_report

    def truth(model):
        if model != "MZ":
            raise ValueError("These two cells generate only MZ")
        return focal_truth(args.level)

    def comparison(scores, model, seed):
        result = original_summary(scores, model, seed)
        necessary = result.pop("core_family_selection_pass")
        result.update(necessary_nested_MZ_vs_M0_pass=necessary, **scope,
            counts_as_curtailment_failure=scope["recovery_90pct_criterion_applies"] and not necessary,
            curtailment_rule="only_valid_at_or_above_SESOI_mean_0p01_and_strict_2SE_failures; no_automatic_curtailment"
                if scope["recovery_90pct_criterion_applies"] else "not_applicable_below_SESOI",
            family_selection="not_evaluated", family_recovery_rate="not_evaluated")
        return result

    def write_checkpoint(directory, report):
        report.update(phase=f"MZ_{args.level}_SESOI_baseline_component", effect_level=args.level,
            SESOI_level=level["multiplier"], focal_generating_truth=focal_truth(args.level),
            generating_z=level["z"], **scope, fixed_cohort_count=FIXED_COHORTS,
            selected_replicates=[args.replicate_index], fitted_models=["M0", "MZ"],
            wave_selected=False, full_menu_wave_selected=False, retry_selected=False, automatic_retry=False,
            baseline_diagnostic_complete=report["valid_recovery_component"],
            **{field: "not_evaluated" for field in CLAIM_FIELDS}, episode_terminal=False,
            interpretation="One uniform-effect MZ baseline component with four original500 M0/MZ fits and unchanged held-out diagnostics. Half-SESOI never enters the90% at-or-above-SESOI pass/failure criterion. No full-menu family selection, recovery rate, interval, heterogeneity, individual-rank or gate verdict is supplied; no wave is selected.")
        if not scope["recovery_90pct_criterion_applies"] or not report["valid_recovery_component"]:
            report["counts_as_curtailment_failure"] = False
        original_writer(directory, report)

    with ExitStack() as stack:
        for name, value in {"BASE_SEEDS": {"MZ": level["base_seed"]}, "focal_truth": truth,
                            "summarize_paired_scores": comparison, "write_report": write_checkpoint,
                            "REPORT_NAME": REPORT_NAME}.items():
            stack.enter_context(patch.object(core, name, value))
        return core.run(argparse.Namespace(model="MZ", replicate_index=args.replicate_index,
            design_events=args.design_events, implementation=args.implementation,
            output_dir=args.output_dir, max_seconds=args.max_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=tuple(LEVELS), required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--design-events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "effect_level", "replicate_index", "seed",
        "baseline_diagnostic_complete", "recovery_90pct_criterion_applies", "counts_as_curtailment_failure",
        "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["baseline_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
