from __future__ import annotations

import os
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.null_trial_runner import (
    NullTrialRunnerError,
    ProductionFullSearchCallback,
    _fit_counts,
    callback_code_identity,
)
from ep12_executor.policy import EpisodePolicy, digest_object


EPISODE_ROOT = Path(__file__).resolve().parents[3]


class RecordingCallback(ProductionFullSearchCallback):
    def __init__(self) -> None:
        self.policy = SimpleNamespace(cpu_cores_per_trial_maximum=32)
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _fit_direct(self, **arguments: Any) -> dict[str, Any]:  # type: ignore[override]
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.01 * (1 + int(arguments["delay"])))
            return {
                "branch_id": str(arguments["branch_id"]),
                "shared_counts_id": id(arguments["counts"]),
                "provider_fit_workers": self._provider_worker_count(),
            }
        finally:
            with self._lock:
                self._active -= 1


class RecordingBinaryCallback(ProductionFullSearchCallback):
    def __init__(self, policy: EpisodePolicy) -> None:
        self.policy = policy

    def _fit_raw(self, **arguments: Any) -> dict[str, Any]:  # type: ignore[override]
        branch_id = str(arguments["branch_id"])
        configuration = dict(arguments["configuration"])
        return {
            "branch_id": branch_id,
            "configuration": configuration,
            "result_sha256": digest_object(
                {"branch_id": branch_id, "configuration": configuration}
            ),
        }


class AccountingCallback(ProductionFullSearchCallback):
    """Exercise the production fit_score adapter without scientific fitting."""

    def __init__(self, policy: EpisodePolicy, artifact_root: Path) -> None:
        self.policy = policy
        self.inputs = SimpleNamespace(
            artifact_root=artifact_root,
            callback_data_sha256="ab" * 32,
        )
        self._objectives: dict[tuple[int, str], dict[str, Any]] = {}
        self._handlers = {
            "binary_weighted_vocabulary_and_rank_sensitivities": self._control
        }

    @staticmethod
    def _arrays(*_: Any, **__: Any) -> tuple[np.ndarray, ...]:
        return tuple(np.zeros(2, dtype=object) for _ in range(4))

    @staticmethod
    def _control(*_: Any, **__: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        objective = {
            "left_to_right": 0.2,
            "right_to_left": 0.1,
            "bilateral": 0.15,
            "bilateral_floor": 0.1,
            "direction_disagreement": 0.1,
            "types_both_above_margin": 2,
        }
        return (
            {"objective": objective, "result_sha256": "cd" * 32},
            {
                "falsifier_id": (
                    "binary_weighted_vocabulary_and_rank_sensitivities"
                ),
                "control_passed": True,
                "fit_branch_count": 7,
                "analytic_branch_count": 0,
                "max_concurrent_fit_workers": 4,
            },
        )


class NullTrialRunnerParallelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prior_allocation = os.environ.get("SLURM_CPUS_PER_TASK")
        self.prior_numeric_threads = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
        for name in self.prior_numeric_threads:
            os.environ[name] = "1"

    def tearDown(self) -> None:
        if self.prior_allocation is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = self.prior_allocation
        for name, value in self.prior_numeric_threads.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_branches_are_serial_and_each_gets_full_provider_allocation(self) -> None:
        os.environ["SLURM_CPUS_PER_TASK"] = "4"
        callback = RecordingCallback()
        shared_counts = np.arange(12, dtype=np.int64).reshape(4, 3)
        jobs = [
            {
                "kind": "direct",
                "branch_id": f"branch_{index}",
                "arguments": {
                    "branch_id": f"branch_{index}",
                    "counts": shared_counts,
                    "delay": (6 - index) % 3,
                },
            }
            for index in range(7)
        ]

        results, workers = callback._run_fit_jobs(jobs)

        self.assertEqual(workers, 1)
        self.assertEqual(callback.max_active, 1)
        self.assertEqual(
            [result["branch_id"] for result in results],
            [job["branch_id"] for job in jobs],
        )
        self.assertEqual(
            {result["shared_counts_id"] for result in results},
            {id(shared_counts)},
        )
        self.assertEqual(
            {result["provider_fit_workers"] for result in results},
            {4},
        )

    def test_executable_identity_binds_sampler_loader_and_callback(self) -> None:
        identity = callback_code_identity()
        self.assertTrue(
            {
                "fitted_null.py",
                "full_search_null.py",
                "full_search_runtime.py",
                "model_artifacts.py",
                "null_trial_runner.py",
            }
            <= set(identity)
        )
        self.assertTrue(all(len(value) == 64 for value in identity.values()))

    def test_provider_worker_count_uses_allocation_and_rejects_bad_values(self) -> None:
        callback = RecordingCallback()
        os.environ["SLURM_CPUS_PER_TASK"] = "16"
        self.assertEqual(callback._provider_worker_count(), 16)

        for invalid in ("0", "33", "not-an-integer"):
            with self.subTest(invalid=invalid):
                os.environ["SLURM_CPUS_PER_TASK"] = invalid
                with self.assertRaises(NullTrialRunnerError):
                    callback._provider_worker_count()

        os.environ["SLURM_CPUS_PER_TASK"] = "2"
        os.environ["OMP_NUM_THREADS"] = "2"
        with self.assertRaises(NullTrialRunnerError):
            callback._provider_worker_count()

    def test_duplicate_branch_ids_fail_before_any_fit(self) -> None:
        os.environ["SLURM_CPUS_PER_TASK"] = "2"
        callback = RecordingCallback()
        jobs = [
            {
                "kind": "direct",
                "branch_id": "duplicate",
                "arguments": {
                    "branch_id": "duplicate",
                    "counts": np.ones((2, 3), dtype=np.int64),
                    "delay": 0,
                },
            }
            for _ in range(2)
        ]
        with self.assertRaises(NullTrialRunnerError):
            callback._run_fit_jobs(jobs)
        self.assertEqual(callback.max_active, 0)

    def test_binary_branch_order_and_hashes_match_serial_execution(self) -> None:
        policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
        configuration = dict(coverage_configurations(policy)[0])
        trial = {
            "configuration": configuration,
            "falsifier_id": "binary_weighted_vocabulary_and_rank_sensitivities",
        }
        sample = SimpleNamespace(raw_counts=np.asarray([[2, 0, 1], [0, 3, 1]]))
        focal_types = np.asarray(["A", "A"], dtype=object)
        sides = np.asarray(["L", "R"], dtype=object)

        os.environ["SLURM_CPUS_PER_TASK"] = "1"
        serial_primary, serial_receipt = RecordingBinaryCallback(
            policy
        )._binary_control(sample, trial, focal_types, sides)
        os.environ["SLURM_CPUS_PER_TASK"] = "8"
        parallel_primary, parallel_receipt = RecordingBinaryCallback(
            policy
        )._binary_control(sample, trial, focal_types, sides)

        expected_count = (
            2
            + len(policy.grammar_choices["partner_vocabulary"])
            * len(policy.grammar_choices["rank"])
        )
        self.assertEqual(expected_count, 14)
        self.assertEqual(serial_primary, parallel_primary)
        self.assertEqual(
            serial_receipt["expected_branch_ids"],
            parallel_receipt["expected_branch_ids"],
        )
        self.assertEqual(
            serial_receipt["branch_manifests"],
            parallel_receipt["branch_manifests"],
        )
        self.assertEqual(parallel_receipt["fit_branch_count"], 14)
        self.assertEqual(parallel_receipt["analytic_branch_count"], 0)
        self.assertEqual(parallel_receipt["max_concurrent_fit_workers"], 1)
        self.assertEqual(parallel_receipt["max_provider_fit_workers"], 8)

    def test_provider_pool_is_bounded_ordered_and_worker_invariant(self) -> None:
        provider_types = np.repeat(np.asarray(["A", "B", "C"], dtype=object), 4)
        sides = np.tile(np.asarray(["L", "L", "R", "R"], dtype=object), 3)
        counts = np.column_stack(
            [np.arange(1, 13, dtype=np.int64), np.arange(13, 25, dtype=np.int64)]
        )
        covariates = np.zeros((12, 1), dtype=float)
        neuron_ids = np.asarray([f"n{index}" for index in range(12)], dtype=object)
        configuration = {
            "component_count": 2,
            "rank": 1,
            "regularization": 0.1,
            "composition_pseudocount": 0.5,
            "representation": "count_aware_log_ratio",
            "covariance": "diagonal_shrinkage",
            "covariance_rank": 1,
        }
        settings = {"integration_draws": 4, "source_cv_folds": 1}
        active = 0
        maximum_active = 0
        guard = threading.Lock()

        class FakeFit:
            U = SimpleNamespace(model_id="U.fake")
            M = SimpleNamespace(
                model_id="M.fake",
                basis=np.eye(1),
                prior_means=np.zeros((2, 1)),
                prior_weights=np.asarray([0.5, 0.5]),
            )
            reference_model_id = "U.fake"
            null_generator_model_id = "U.fake"

            @staticmethod
            def score_target(target_counts: np.ndarray, _: np.ndarray) -> dict[str, Any]:
                rows = len(target_counts)
                return {
                    "T": np.full(rows, -2.0),
                    "U": np.full(rows, -1.5),
                    "M": np.full(rows, -1.0),
                    "reference_model_id": "U.fake",
                    "null_generator_model_id": "U.fake",
                }

        def fake_fit(*_: Any, **__: Any) -> FakeFit:
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.01)
                return FakeFit()
            finally:
                with guard:
                    active -= 1

        arguments = {
            "counts": counts,
            "covariates": covariates,
            "focal_types": provider_types,
            "sides": sides,
            "neuron_ids": neuron_ids,
            "configuration": configuration,
            "settings": settings,
            "seed_material": ("test",),
            "meaningful_margin": 0.02,
        }
        with mock.patch(
            "ep12_executor.null_trial_runner.fit_triplet", side_effect=fake_fit
        ), mock.patch(
            "ep12_executor.null_trial_runner.component_partner_contrasts",
            return_value=[],
        ):
            serial = _fit_counts(**arguments, fit_workers=1)
            maximum_active = 0
            parallel = _fit_counts(**arguments, fit_workers=3)

        self.assertEqual(serial, parallel)
        self.assertGreater(maximum_active, 1)
        self.assertLessEqual(maximum_active, 3)
        self.assertEqual(
            [record["provider_type"] for record in parallel["records"]],
            ["A", "B", "C"],
        )

    def test_fit_worker_validation_rejects_invalid_counts_and_threading(self) -> None:
        common = {
            "counts": np.ones((4, 2), dtype=np.int64),
            "covariates": np.zeros((4, 1)),
            "focal_types": np.asarray(["A"] * 4, dtype=object),
            "sides": np.asarray(["L", "L", "R", "R"], dtype=object),
            "neuron_ids": np.asarray(["a", "b", "c", "d"], dtype=object),
            "configuration": {
                "component_count": 2,
                "rank": 1,
                "regularization": 0.1,
                "composition_pseudocount": 0.5,
                "representation": "count_aware_log_ratio",
                "covariance": "diagonal_shrinkage",
                "covariance_rank": 1,
            },
            "settings": {"integration_draws": 4, "source_cv_folds": 1},
            "seed_material": ("test",),
            "meaningful_margin": 0.02,
        }
        for invalid in (0, 33, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(NullTrialRunnerError):
                _fit_counts(**common, fit_workers=invalid)
        os.environ["MKL_NUM_THREADS"] = "2"
        with self.assertRaises(NullTrialRunnerError):
            _fit_counts(**common, fit_workers=2)

    def test_production_adapter_authenticates_actual_control_fit_count(self) -> None:
        policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
        configuration = dict(coverage_configurations(policy)[0])
        trial = {
            "valid_trial_ordinal": 1,
            "trial_id": str(configuration["trial_id"]),
            "configuration": configuration,
            "configuration_sha256": digest_object(configuration),
            "falsifier_id": (
                "binary_weighted_vocabulary_and_rank_sensitivities"
            ),
            "promotion_eligible": False,
        }
        sample = SimpleNamespace(
            replicate_index=0,
            replicate_id="null_001",
            replicate_seed=123,
        )
        context = {
            "callback_protocol": "ep12.full_search_fit_score_callback.v3",
            "procedure_lock_sha256": "12" * 32,
            "controller_program_sha256": "34" * 32,
            "replicate_index": 0,
            "replicate_id": "null_001",
            "replicate_seed": 123,
            "sample_receipt_sha256": "56" * 32,
            "initial_ledger_empty": True,
            "nested_null_searches": 0,
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = AccountingCallback(policy, root).fit_score(
                sample=sample, trial=trial, context=context
            )
            artifact_path = root / str(result["fit_score_artifact_path"])
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(result["executed_complete_T_U_M_fit_count"], 7)
        self.assertEqual(result["executed_model_family_fit_count"], 21)
        self.assertEqual(artifact["executed_complete_T_U_M_fit_count"], 7)
        self.assertEqual(artifact["executed_model_family_fit_count"], 21)
        self.assertEqual(
            artifact_path.name,
            f"001_{trial['configuration_sha256']}.json",
        )


if __name__ == "__main__":
    unittest.main()
