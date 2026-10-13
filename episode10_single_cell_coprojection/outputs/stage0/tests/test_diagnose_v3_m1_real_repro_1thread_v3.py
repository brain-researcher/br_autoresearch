#!/usr/bin/env python3
"""Focused contract tests for the single-thread failed-path diagnostic."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


DIAGNOSTIC_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnose_v3_m1_real_repro_1thread_v3.py"
)
LAUNCHER_PATH = DIAGNOSTIC_PATH.with_suffix(".sbatch")
SPEC = importlib.util.spec_from_file_location(
    "ep10_diagnose_v3_m1_real_repro_1thread_v3_tested", DIAGNOSTIC_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import single-thread diagnostic: {DIAGNOSTIC_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


class SingleThreadDiagnosticTests(unittest.TestCase):
    def test_backend_and_decision_constants_are_single_thread(self) -> None:
        self.assertEqual(diagnostic.EXPECTED_OPENBLAS_THREADS, 1)
        self.assertEqual(diagnostic.v2.REPEAT_COUNT, 2)
        self.assertEqual(
            diagnostic.v2.AGREEMENT_DECISION, "single_thread_backend_pass"
        )
        self.assertEqual(
            diagnostic.v2.EXPECTED_FAILURE_DECISION,
            "single_thread_arm_failed_requires_solver_amendment",
        )
        self.assertEqual(
            diagnostic.v2.DISAGREEMENT_DECISION,
            "single_thread_arm_failed_requires_solver_amendment",
        )

    def test_cli_has_only_authorized_development_routes(self) -> None:
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

    def test_main_delegates_without_mutating_policy(self) -> None:
        with patch.object(diagnostic.v2, "main", return_value=0) as delegated:
            self.assertEqual(diagnostic.main(["--help"]), 0)
        delegated.assert_called_once_with(["--help"])
        self.assertEqual(diagnostic.v2.EXPECTED_OPENBLAS_THREADS, 1)

    def test_launcher_pins_numeric_threads_independent_of_slurm_rounding(self) -> None:
        text = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=1", text)
        self.assertIn("export OMP_NUM_THREADS=1", text)
        self.assertIn("export OPENBLAS_NUM_THREADS=1", text)
        self.assertIn("export MKL_NUM_THREADS=1", text)
        self.assertNotIn(
            'export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"', text
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
