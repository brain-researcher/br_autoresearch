#!/usr/bin/env python3
"""Focused orchestration tests for the original-backend M1 replay."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnose_v3_m1_real_repro_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ep10_diagnose_v3_m1_real_repro_v2_tested", DIAGNOSTIC_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import repro diagnostic: {DIAGNOSTIC_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


class ReproContractTests(unittest.TestCase):
    @staticmethod
    def _optimizer(theta_value: float) -> SimpleNamespace:
        np = diagnostic.runner.np
        return SimpleNamespace(
            theta=np.full(diagnostic.m1.PARAMETER_DIMENSION, theta_value),
            objective=1.0,
            gradient_infinity_norm=0.0,
            iterations=1,
            accepted_steps=1,
            relative_objective_change=0.0,
            telemetry={
                "status": "pass",
                "convergence_witness": "gradient_and_relative_objective_change",
                "candidate_visits": 1,
                "evaluation_seconds": 0.1,
            },
        )

    def test_repeat_count_and_tolerances_match_frozen_gate(self) -> None:
        self.assertEqual(diagnostic.REPEAT_COUNT, 2)
        self.assertEqual(diagnostic.OBJECTIVE_TOLERANCE, 1.0e-12)
        self.assertEqual(diagnostic.COEFFICIENT_TOLERANCE, 1.0e-10)
        self.assertEqual(diagnostic.PROBABILITY_TOLERANCE, 1.0e-10)

    def test_repeat_comparison_is_scalar_and_fail_closed(self) -> None:
        np = diagnostic.runner.np
        first = SimpleNamespace(objective=1.0, theta=np.zeros(4, dtype=float))
        second = SimpleNamespace(
            objective=1.0 + 5.0e-13,
            theta=np.full(4, 5.0e-11, dtype=float),
        )
        original = diagnostic.m1.compare_conditional_distributions
        try:
            diagnostic.m1.compare_conditional_distributions = lambda *args: {
                "maximum_candidate_probability_absolute_difference": 5.0e-11
            }
            passed = diagnostic._repeat_comparison(
                1.0, first, second, object(), object(), object()
            )
            self.assertEqual(passed["status"], "pass")
            self.assertNotIn("theta", passed)
            diagnostic.m1.compare_conditional_distributions = lambda *args: {
                "maximum_candidate_probability_absolute_difference": 2.0e-10
            }
            failed = diagnostic._repeat_comparison(
                1.0, first, second, object(), object(), object()
            )
            self.assertEqual(failed["status"], "fail")
        finally:
            diagnostic.m1.compare_conditional_distributions = original

    def test_cli_has_no_m0_swc_final_or_unassigned_route(self) -> None:
        destinations = {
            action.dest
            for action in diagnostic.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(
            destinations,
            {
                "contract",
                "interface_root",
                "geometry_root",
                "actual_k_root",
                "target_root",
                "prior_fit_root",
                "output_root",
            },
        )

    def test_output_root_is_exact_job_scoped_and_nonoverwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            episode = (
                scratch
                / "br_autoresearch"
                / "episode10_single_cell_coprojection"
            )
            episode.mkdir(parents=True)
            expected = episode / "v3_m1_real_warm_repro_v2_job_7001"
            with patch.dict(
                os.environ,
                {"SCRATCH": str(scratch), "SLURM_JOB_ID": "7001"},
                clear=False,
            ):
                self.assertEqual(diagnostic._prepare_output_root(expected), expected)
                with self.assertRaises(diagnostic.ReproDiagnosticError):
                    diagnostic._prepare_output_root(expected)
                with self.assertRaises(diagnostic.ReproDiagnosticError):
                    diagnostic._prepare_output_root(episode / "wrong")

    def test_repeat_two_runs_after_repeat_one_expected_failure(self) -> None:
        np = diagnostic.runner.np
        calls: list[tuple[float, object]] = []

        def fake_fit(
            prepared: object,
            library: object,
            geometry: object,
            ridge_lambda: float,
            initial: object = None,
        ) -> SimpleNamespace:
            del prepared, library, geometry
            calls.append(
                (
                    ridge_lambda,
                    None if initial is None else np.asarray(initial).copy(),
                )
            )
            if len(calls) == 1:
                raise diagnostic.m1.UncertifiedLineSearchFailure(
                    "synthetic expected failure", {"optimizer_iteration": 1}
                )
            return self._optimizer(float(len(calls)))

        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            episode = (
                scratch
                / "br_autoresearch"
                / "episode10_single_cell_coprojection"
            )
            episode.mkdir(parents=True)
            output = episode / "v3_m1_real_warm_repro_v2_job_7002"
            args = SimpleNamespace(
                contract=Path("contract"),
                interface_root=Path("interface"),
                geometry_root=Path("geometry"),
                actual_k_root=Path("actual_k"),
                target_root=Path("target"),
                prior_fit_root=Path("prior"),
                output_root=output,
            )
            with (
                patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": "7002"},
                    clear=False,
                ),
                patch.object(diagnostic.runner, "validate_real_contract"),
                patch.object(diagnostic.runner, "require_actual_k_prerequisite"),
                patch.object(diagnostic.v1, "require_failed_attempt"),
                patch.object(diagnostic, "_numeric_backend", return_value={}),
                patch.object(
                    diagnostic,
                    "_prepare_failed_training",
                    return_value=(object(), object(), object(), 175, 10),
                ),
                patch.object(
                    diagnostic.m1, "LAMBDAS_DESCENDING", (1000.0, 100.0)
                ),
                patch.object(diagnostic.m1, "fit_newton", side_effect=fake_fit),
            ):
                result = diagnostic.run_diagnostic(args)

        self.assertEqual(len(calls), 3)
        self.assertIsNone(calls[0][1])
        self.assertIsNone(calls[1][1])
        self.assertTrue(np.all(calls[2][1] == 2.0))
        self.assertEqual(result["attempted_repeat_count"], 2)
        self.assertEqual(result["failure_recurrence_count"], 1)
        self.assertEqual(result["decision"], "trigger_single_thread_arm")

    def test_two_successful_paths_reset_and_take_bounded_branch(self) -> None:
        np = diagnostic.runner.np
        calls: list[object] = []

        def fake_fit(
            prepared: object,
            library: object,
            geometry: object,
            ridge_lambda: float,
            initial: object = None,
        ) -> SimpleNamespace:
            del prepared, library, geometry, ridge_lambda
            calls.append(None if initial is None else np.asarray(initial).copy())
            return self._optimizer(float(len(calls)))

        comparison = {
            "lambda": 1000.0,
            "status": "pass",
            "objective_absolute_difference": 0.0,
            "objective_absolute_tolerance": 1.0e-12,
            "coefficient_maximum_absolute_difference": 0.0,
            "coefficient_maximum_absolute_tolerance": 1.0e-10,
            "probability_maximum_absolute_difference": 0.0,
            "probability_maximum_absolute_tolerance": 1.0e-10,
        }
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            episode = (
                scratch
                / "br_autoresearch"
                / "episode10_single_cell_coprojection"
            )
            episode.mkdir(parents=True)
            output = episode / "v3_m1_real_warm_repro_v2_job_7003"
            args = SimpleNamespace(
                contract=Path("contract"),
                interface_root=Path("interface"),
                geometry_root=Path("geometry"),
                actual_k_root=Path("actual_k"),
                target_root=Path("target"),
                prior_fit_root=Path("prior"),
                output_root=output,
            )
            with (
                patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": "7003"},
                    clear=False,
                ),
                patch.object(diagnostic.runner, "validate_real_contract"),
                patch.object(diagnostic.runner, "require_actual_k_prerequisite"),
                patch.object(diagnostic.v1, "require_failed_attempt"),
                patch.object(diagnostic, "_numeric_backend", return_value={}),
                patch.object(
                    diagnostic,
                    "_prepare_failed_training",
                    return_value=(object(), object(), object(), 175, 10),
                ),
                patch.object(
                    diagnostic.m1, "LAMBDAS_DESCENDING", (1000.0, 100.0)
                ),
                patch.object(diagnostic.m1, "fit_newton", side_effect=fake_fit),
                patch.object(
                    diagnostic, "_repeat_comparison", return_value=comparison
                ),
            ):
                result = diagnostic.run_diagnostic(args)

        self.assertEqual(len(calls), 4)
        self.assertIsNone(calls[0])
        self.assertTrue(np.all(calls[1] == 1.0))
        self.assertIsNone(calls[2])
        self.assertTrue(np.all(calls[3] == 3.0))
        self.assertEqual(result["completed_repeat_count"], 2)
        self.assertEqual(
            result["decision"], "bounded_nonreproduction_at_four_threads"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
