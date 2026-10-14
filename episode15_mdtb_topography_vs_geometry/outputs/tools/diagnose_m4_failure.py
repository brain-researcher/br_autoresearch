#!/usr/bin/env python3
"""Capture the first failed canonical Task-A fit without changing its refusal.

The unchanged complete observed-map panel runs with publication withheld. A
failed canonical fit writes only its Task-A objective to a protected
cache and aggregate numerical diagnostics to the selected report. No result
summary is opened, no fit is bypassed, and the original panel exit is retained.
"""
import argparse
import os
from pathlib import Path

import run_development_point_panel as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-json", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--publication-mode", required=True,
                        choices=("withheld_cost_measurement",))
    args, remaining = parser.parse_known_args(argv)
    runner._compute()
    scope = runner.OUTPUTS / "runs/observed_beta_development"
    output, selected = runner._path(args.diagnostic_json), runner._path(args.cache_dir)
    for path in (output, selected):
        if path.exists() or not path.parent.is_dir() or scope not in path.parents:
            parser.error("fresh diagnostic and cache paths inside the observed-beta route are required")

    import proposed_development_point_panel as adapter
    from diagnose_m4_global import _save_cache
    from paired_stretch_variants import M4CertificateUnverified, stretch_specifications

    original_fit = adapter._fit_with_certificate
    canonical_names = {specification.name for specification in stretch_specifications()}
    captured, attempts = {}, 0

    def inspect(objective, specification, X, Y, B, O, settings, *, check):
        nonlocal attempts
        attempts += 1
        try:
            return original_fit(objective, specification, X, Y, B, O, settings, check=check)
        except M4CertificateUnverified as error:
            if specification.name in canonical_names and not captured:
                check()
                path = _save_cache(selected, (X, Y, B, O), settings, specification)
                captured.update(status="canonical_certificate_unresolved", variant=specification.name,
                    numerical_fit_attempt=attempts, reason=str(error),
                    diagnostics=error.diagnostics,
                    private_Task_A_cache=str(path), nonuniqueness_established=False)
            raise

    adapter._fit_with_certificate = inspect
    try:
        panel_exit = runner.main([*remaining, "--publication-mode", args.publication_mode])
    finally:
        adapter._fit_with_certificate = original_fit
    report = dict(status="failed_Task_A_fit_diagnostic" if captured else "no_canonical_failure_captured",
        original_panel_exit=panel_exit, job_id=os.environ["SLURM_JOB_ID"],
        numerical_fit_attempts=attempts, production_acceptance_changed=False,
        scientific_summaries_opened=False, scientific_summaries_published=False,
        audit_values_read=False, cache_contains_only_Task_A_objective_inputs=bool(captured),
        diagnostic=captured)
    runner._write_new(output, runner._encoded(report, runner.REPORT_BYTES))
    return panel_exit


if __name__ == "__main__":
    raise SystemExit(main())
