#!/usr/bin/env python3
"""One externally selected half-SESOI baseline index using the frozen runner.

Indices1 through199 are selected externally. This below-SESOI cell supplies
diagnostics only and never contributes90% criterion or curtailment failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import run_mz_effect_level_recovery as levels


_run_effect_level = levels.run
DISPATCH_SCOPE = "externally_selected_half_SESOI_baseline_indices1_through199"


def run(args):
    if type(args.replicate_index) is not int or not 1 <= args.replicate_index < levels.FIXED_COHORTS:
        raise ValueError("The selected half-SESOI wave contains only indices1 through199")
    if getattr(args, "level", "half") != "half":
        raise ValueError("This selection contains only half-SESOI components")
    selected = argparse.Namespace(level="half", replicate_index=args.replicate_index,
        design_events=args.design_events, implementation=args.implementation,
        output_dir=args.output_dir, max_seconds=args.max_seconds)
    original_writer = levels.core.write_report

    def write_checkpoint(directory, report):
        # Called after the frozen level writer has set below-SESOI criterion
        # fields, including false curtailment counting for every checkpoint.
        report.update(wave_selected=True, dispatch_scope=DISPATCH_SCOPE,
            externally_selected_index_range=[1, 199], subsequent_wave_selected=False,
            interpretation="One externally selected half-SESOI MZ baseline component from fixed indices1 through199. Four original500 M0/MZ fits retain the unchanged seed, uniform z=.475 truth and held-out diagnostics. Below-SESOI results never enter the90% at-or-above-SESOI criterion and never count as curtailment failures. No full-menu family selection, recovery rate, interval, heterogeneity, individual-rank or gate verdict is supplied. This process dispatches no subsequent wave or retry.")
        return original_writer(directory, report)

    with patch.object(levels.core, "write_report", write_checkpoint):
        return _run_effect_level(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--design-events", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=levels.MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "effect_level", "replicate_index", "seed",
        "wave_selected", "dispatch_scope", "baseline_diagnostic_complete", "recovery_90pct_criterion_applies",
        "counts_as_curtailment_failure", "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["baseline_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
