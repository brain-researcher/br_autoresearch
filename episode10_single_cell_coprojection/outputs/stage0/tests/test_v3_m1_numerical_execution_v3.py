#!/usr/bin/env python3
"""Focused tests for the frozen EP10 M1 one-thread execution gate."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


STAGE0 = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = STAGE0 / "tools" / "validate_v3_m1_numerical_execution_v3.py"
WRAPPER_PATH = (
    STAGE0 / "tools" / "run_v3_m1_synthetic_warm_zero_single_thread_v3.py"
)
LAUNCHER_PATH = (
    STAGE0 / "tools" / "probe_v3_m1_synthetic_warm_zero_single_thread_v3.sbatch"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import test dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = load_module("ep10_v3_m1_execution_validator_tested", VALIDATOR_PATH)
wrapper = load_module("ep10_v3_m1_single_thread_wrapper_tested", WRAPPER_PATH)


class NumericalExecutionPolicyTests(unittest.TestCase):
    def test_frozen_addendum_is_exact_and_changes_only_execution(self) -> None:
        payload = validator.validate_addendum()
        execution = payload["sole_execution_change"]
        self.assertEqual(execution["scheduler_constraint"], "CPU_SKU:7502")
        self.assertEqual(execution["dispatched_OpenBLAS_core_required"], "Zen")
        self.assertEqual(execution["requested_cpus_per_task"], 1)
        self.assertTrue(
            all(
                value is False
                for key, value in execution.items()
                if key.startswith("changes_") or key.startswith("adds_")
            )
        )
        self.assertFalse(
            payload["checkpoint_policy"][
                "four_thread_job_45609750_checkpoints_reusable"
            ]
        )
        self.assertEqual(
            payload["required_qualification"]["pass_authorizes_only"],
            "clean_single_thread_development_M1_production_fit",
        )
        self.assertEqual(
            payload["resource_reconciliation"][
                "conservative_path_equivalents"
            ],
            157,
        )

    def test_validator_and_wrapper_cli_expose_no_real_input_route(self) -> None:
        validator_destinations = {
            action.dest
            for action in validator.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(validator_destinations, {"addendum", "runtime"})
        wrapper_destinations = {
            action.dest
            for action in wrapper.build_parser()._actions
            if action.dest != "help"
        }
        self.assertEqual(
            wrapper_destinations,
            {
                "contract",
                "synthetic_design_addendum",
                "synthetic_recovery_addendum",
                "solver_addendum",
                "execution_addendum",
                "output_root",
            },
        )

    def test_runtime_gate_ignores_rounded_slurm_allocation(self) -> None:
        environment = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "2",
            "SLURM_TRES_PER_TASK": "cpu=1",
            "SLURM_JOB_CPUS_PER_NODE": "2",
        }
        openblas = {
            "library": "libopenblas-test.so",
            "config": "OpenBLAS test",
            "core": "Zen",
            "threads": 1,
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(validator, "_openblas_runtime", return_value=openblas),
            patch.object(
                validator,
                "_cpu_model",
                return_value="AMD EPYC 7502 32-Core Processor",
            ),
        ):
            observed = validator.numeric_backend()
        self.assertEqual(observed["openblas_threads"], 1)
        self.assertEqual(observed["SLURM_cpus_per_task"], "2")

    def test_runtime_gate_rejects_effective_thread_mismatch(self) -> None:
        environment = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        openblas = {
            "library": "libopenblas-test.so",
            "config": "OpenBLAS test",
            "core": "Zen",
            "threads": 2,
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(validator, "_openblas_runtime", return_value=openblas),
        ):
            with self.assertRaises(validator.ExecutionPolicyError):
                validator.numeric_backend()

    def test_launcher_is_clean_single_thread_and_target_blind(self) -> None:
        text = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --constraint=CPU_SKU:7502", text)
        self.assertIn("#SBATCH --cpus-per-task=1", text)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIn(f"export {name}=1", text)
        self.assertNotIn("$SLURM_CPUS_PER_TASK", text)
        self.assertNotIn("tee", text)
        self.assertNotIn(">(", text)
        self.assertIn('exec >>"$runtime_log" 2>&1', text)
        self.assertIn("--execution-addendum", text)
        self.assertLess(
            text.index('"$python_bin" "$validator"'),
            text.index('"$python_bin" "$engine_test"'),
        )
        self.assertIn(
            "v3_m1_synthetic_warm_zero_diagnostic_v2_job_${SLURM_JOB_ID}",
            text,
        )
        for forbidden in (
            "interface_root",
            "geometry_root",
            "target_root",
            "actual_k_root",
        ):
            self.assertNotIn(forbidden, text)

    def test_wrapper_attests_backend_before_unchanged_engine(self) -> None:
        text = WRAPPER_PATH.read_text(encoding="utf-8")
        self.assertLess(
            text.index("backend = execution_policy.numeric_backend()"),
            text.index("engine.run_synthetic_warm_zero_diagnostic(engine_args)"),
        )
        self.assertNotIn("engine.fit_newton =", text)

    def test_wrapper_fails_closed_on_combined_real_access_seal(self) -> None:
        payload = {"real_target_calls_or_real_K_or_membership_opened": True}
        with self.assertRaises(wrapper.execution_policy.ExecutionPolicyError):
            wrapper._attach_execution_witness(
                payload,
                {"schema_version": "test"},
                {"openblas_threads": 1},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
