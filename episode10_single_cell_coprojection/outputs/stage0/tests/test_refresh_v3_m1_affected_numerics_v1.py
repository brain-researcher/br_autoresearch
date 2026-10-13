#!/usr/bin/env python3
"""Mocked orchestration regressions; no scientific optimizer is executed."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import refresh_v3_m1_affected_numerics_v1 as refresh


def reservation():
    return {
        "qualification_job_id": "46328672", "reservation_included": True,
        "request_wall_hours": 8, "requested_cpus": 1, "conservative_allocated_cpus": 4,
        "memory_GiB": 16, "remaining_unreserved_wall_hours": 60,
        "remaining_unreserved_allocated_core_hours": 900,
    }


class RefreshTests(unittest.TestCase):
    def test_exact_new_qualification_binding_and_comparison_threshold(self):
        # Use the actual aggregate target-free result; never open predictions.
        source = refresh.STAGE0 / "v3_m1_synthetic_warm_zero_diagnostic_v2_job_46328672" / "v3_m1_synthetic_warm_zero_diagnostic.json"
        summary = dict(refresh._read_json(source))
        root = Path(summary["output_root"])
        backend = summary["numeric_backend"]
        refresh.validate_qualification(summary, root, backend)
        stale = copy.deepcopy(summary)
        stale["SLURM_job_id"] = "45734258"
        with self.assertRaises(refresh.RefreshError):
            refresh.validate_qualification(stale, root, backend)
        failed = copy.deepcopy(summary)
        failed["warm_zero_comparisons"][0]["zero_start_probability_max_absolute_difference"] = 1.1e-5
        with self.assertRaises(refresh.RefreshError):
            refresh.validate_qualification(failed, root, backend)
        different_backend = dict(backend, openblas_threads=2)
        with self.assertRaises(refresh.RefreshError):
            refresh.validate_qualification(summary, root, different_backend)

    def test_accounting_requires_existing_reset_and_reserved_request(self):
        payload = {
            "accounting_epoch": "wall_reset_20261001", "no_new_wall_reset": True,
            "historical_strict_wall_hours_preserved": 129.83266666666668,
            "historical_allocated_core_hours_floor_preserved": 302.02566666666667,
            "m1_affected_numerics_refresh": reservation(),
        }
        for actual_cpus in (1, 2, 3, 4):
            self.assertEqual(refresh.validate_resources(payload, actual_cpus), reservation())
        for key, value in (("accounting_epoch", "new_epoch"), ("m1_affected_numerics_refresh", {})):
            with self.assertRaises(refresh.RefreshError):
                refresh.validate_resources(dict(payload, **{key: value}), 2)
        with self.assertRaises(refresh.RefreshError):
            refresh.validate_resources(payload, 5)
        insufficient = dict(payload, m1_affected_numerics_refresh=dict(reservation(), conservative_allocated_cpus=2))
        with self.assertRaises(refresh.RefreshError):
            refresh.validate_resources(insufficient, 3)

    def test_population_subset_calls_only_existing_exact_recovery(self):
        fixture = tuple(object() for _ in range(6))
        with mock.patch.object(refresh.m1, "generate_synthetic_dataset", return_value=fixture) as generate, \
             mock.patch.object(refresh.m1, "_synthetic_population_recovery_gate", return_value={"status": "pass"}) as recovery, \
             mock.patch.object(refresh.m1, "rank_condition_gate", side_effect=AssertionError("static rank rerun")), \
             mock.patch.object(refresh.m1, "fit_newton", side_effect=AssertionError("unmocked optimizer")):
            self.assertEqual(refresh.recovery_subset(), {"status": "pass"})
        generate.assert_called_once_with(groups=16, rows_per_group=34, seed=20260927)
        recovery.assert_called_once_with(fixture[0], fixture[2], fixture[3], fixture[4], fixture[5])

    def test_all_k8_only_runs_frozen_numerical_operations(self):
        cells = tuple(SimpleNamespace(k=8, group=i % 12) for i in range(194))
        library = mock.Mock()
        library.basis.return_value.count = 12870
        geometry = mock.Mock()
        prepared = object()
        evaluation = SimpleNamespace(objective=1.0, gradient=np.ones(2), hessian=np.eye(2), candidate_visits=4993560)
        path = {lam: SimpleNamespace(telemetry={"status": "pass"}, iterations=3)
                for lam in refresh.m1.LAMBDAS_DESCENDING}
        with mock.patch.object(refresh.m1, "load_target_free_real_shape", return_value=(cells, ())) as load, \
             mock.patch.object(refresh.m1, "CandidateLibrary", return_value=library), \
             mock.patch.object(refresh.m1, "GeometryEngine", return_value=geometry), \
             mock.patch.object(refresh.m1, "clone_with_k", return_value=cells) as clone, \
             mock.patch.object(refresh.m1, "prepare_data", return_value=prepared), \
             mock.patch.object(refresh.m1, "evaluate_model", return_value=evaluation) as evaluate, \
             mock.patch.object(refresh.m1, "_newton_direction", return_value=(np.ones(2), {"pass": True})), \
             mock.patch.object(refresh.m1, "fit_path", return_value=path) as fit, \
             mock.patch.object(refresh.m1, "rank_condition_gate", side_effect=AssertionError("static rank rerun")), \
             mock.patch.object(refresh.m1, "run_synthetic_gates", side_effect=AssertionError("36-fit rerun")):
            result = refresh.all_k8_subset(Path("interface"), Path("geometry"))
        load.assert_called_once_with(Path("interface"), Path("geometry"))
        clone.assert_called_once_with(cells, 8)
        self.assertEqual(evaluate.call_count, 2)
        for call in evaluate.call_args_list:
            self.assertEqual(call.args[4], 1e-4)
            self.assertEqual(call.kwargs, {"need_hessian": True})
        fit.assert_called_once_with(prepared, library, geometry)
        self.assertEqual(result["all_K8_unique_candidate_states"], 2496780)
        self.assertEqual(len(result["complete_path_iterations"]), 12)

    def test_recovery_failure_is_retained_and_stops_throughput(self):
        payload = {}
        checkpoints = []
        with mock.patch.object(refresh, "recovery_subset", return_value={"status": "fail", "failure_class": "NumericalFailure"}), \
             mock.patch.object(refresh, "all_k8_subset", side_effect=AssertionError("must stop")):
            with self.assertRaises(refresh.RefreshError):
                refresh.run_numerics(payload, Path("interface"), Path("geometry"), reservation(), lambda: checkpoints.append(copy.deepcopy(payload)))
        self.assertEqual(checkpoints[-1]["recovery"]["failure_class"], "NumericalFailure")

    def test_projection_retains_postflight_gate_and_conservative_157_paths(self):
        result = refresh.production_projection({"complete_12_lambda_warm_path_seconds": 100}, reservation(), 120, 300)
        self.assertIn("pending", result["status"])
        self.assertIsNone(result["Slurm_MaxRSS"])
        self.assertFalse(result["production_authorized"])
        self.assertEqual(result["projected_production_wall_hours"], 15700 / 3600)
        self.assertEqual(result["projected_allocated_core_hours"], 62800 / 3600)
        self.assertEqual(result["conservative_production_allocated_cpus"], 4)
        self.assertEqual(refresh.production_projection({"complete_12_lambda_warm_path_seconds": 150}, reservation(), 120, 300)["status"], "fail")
        scarce = dict(reservation(), remaining_unreserved_wall_hours=1)
        self.assertEqual(refresh.production_projection({"complete_12_lambda_warm_path_seconds": 100}, scarce, 120, 300)["status"], "fail")
        self.assertEqual(refresh.production_projection({"complete_12_lambda_warm_path_seconds": 100}, reservation(), 120, 24000)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
