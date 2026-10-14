#!/usr/bin/env python3
"""Check the new diagonal certificate at the original first Task-A refusal.

Compute only. The normal panel still raises its original curvature refusal.
Only aggregate numerical diagnostics are published; no Task-B prediction,
score, participant identity or fitted coefficient is written by this module.
"""
import argparse
from collections import Counter
from pathlib import Path

import run_development_point_panel as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-json", type=Path, required=True)
    parser.add_argument("--fit-method", choices=("original_slsqp", "scalar_dual"), default="original_slsqp")
    args, remaining = parser.parse_known_args(argv)
    runner._compute()
    path = runner._path(args.diagnostic_json)
    if path.exists() or not path.parent.is_dir() or runner.OUTPUTS / "runs" not in path.parents:
        parser.error("diagnostic output must be fresh in an existing episode outputs/runs directory")
    import proposed_development_point_panel as adapter
    from diagonal_m4_dual import diagonal_solution_certificate, fit_diagonal_via_dual
    from paired_stretch_variants import (
        MAX_ITERATIONS, OPTIMIZER_FTOL, M4CertificateUnverified, fit_m4_task_a,
    )

    original_bound, original_objective = adapter.proposed_curvature_certificate, adapter.PairedM4TaskAObjective
    captured = {}

    def objective_with_original_arguments(*positional, **keywords):
        objective = original_objective(*positional, **keywords)
        objective._diagnostic_original_arguments = (positional, keywords)
        return objective

    def inspect(objective, specification, *, check):
        try:
            return original_bound(objective, specification, check=check)
        except M4CertificateUnverified:
            if captured:
                raise
            captured.update(status="started_first_Task_A_refusal_only",
                            full_domain_positive_curvature_passed=False,
                            fit_method=args.fit_method,
                            operator_locality_check_exercised=False)
            check()
            positional, keywords = objective._diagnostic_original_arguments
            try:
                if args.fit_method == "scalar_dual":
                    numerical, certificate = fit_diagonal_via_dual(objective, specification, check=check)
                else:
                    numerical = fit_m4_task_a(*positional, **keywords,
                        spectral_norm_bound=specification.spectral_norm_bound,
                        optimizer_ftol=OPTIMIZER_FTOL, max_iterations=MAX_ITERATIONS)
                    check()
                    _, certificate = diagonal_solution_certificate(objective, specification, numerical, check=check)
                energy = certificate["objective_scale"]
                captured.update(status="dual_global_uniqueness_and_gap_passed",
                    first_order_gap_per_energy=certificate["first_order_gap"]/energy,
                    global_gap_upper_bound_per_energy=certificate["global_gap_upper_bound"]/energy,
                    absolute_objective_difference_upper_bound_per_energy=certificate["absolute_objective_difference_upper_bound"]/energy,
                    required_gap_upper_bound_per_energy=certificate["required_gap_upper_bound"]/energy,
                    coefficient_growth_lower_bound_per_energy=certificate["global_coefficient_growth_lower_bound"]/energy,
                    coefficient_distance_upper_bound=certificate["coefficient_distance_upper_bound"],
                    coordinate_branch_counts=dict(Counter(certificate["coordinate_branches"])),
                    dual_bisection_iterations=certificate["dual_bisection_iterations"],
                    growth_cover_cells_total=sum(certificate["growth_cover_cells"]),
                    optimizer_iterations=numerical["iterations"],
                    exact_global_optimizer=False, interval_arithmetic_proof=False)
            except M4CertificateUnverified as error:
                captured.update(status="new_sufficient_certificate_unresolved",
                                reason=str(error), diagnostics=error.diagnostics,
                                nonuniqueness_established=False)
            except (ArithmeticError, ValueError) as error:
                captured.update(status="optimizer_or_numerical_error", reason=str(error))
            raise

    adapter.PairedM4TaskAObjective = objective_with_original_arguments
    adapter.proposed_curvature_certificate = inspect
    try:
        panel_exit = runner.main(remaining)
    finally:
        adapter.proposed_curvature_certificate = original_bound
        adapter.PairedM4TaskAObjective = original_objective
    report = dict(status="diagnosed_original_refusal" if captured else "no_refusal_captured",
                  original_panel_exit=panel_exit, production_acceptance_changed=False,
                  Task_B_scores_or_predictions_published=False, audit_values_read=False,
                  diagnostic=captured)
    runner._write_new(path, runner._encoded(report, runner.REPORT_BYTES))
    return 0 if captured else 2


if __name__ == "__main__":
    raise SystemExit(main())
