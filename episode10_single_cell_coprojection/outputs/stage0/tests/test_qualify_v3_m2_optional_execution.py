#!/usr/bin/env python3
"""Lightweight regressions for independent optional recovery case execution."""

from concurrent.futures import Future
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import qualify_v3_m2_optional as qualification


def passing_gate(family, rank, *, context=False):
    """Small complete evidence records; numerical solvers are always mocked."""
    case = {
        "status": "pass", "deterministic_repeat_bit_exact": True,
        "lambda_reference": 0.02,
        "mean_conditional_KL_bits": 4e-8,
        "maximum_candidate_probability_absolute_error": 2e-5,
        "mean_inclusion_marginal_absolute_error": 6e-6,
        "population_correction_planted_gradient_maximum_absolute_error": 4e-16,
        "fit_relative_objective_change": 1e-12,
    }
    gate = {
        "status": "pass", "ranks": [rank], "cases": {str(rank): case},
        "mean_conditional_KL_maximum_bits": qualification.RECOVERY_KL_MAXIMUM_BITS,
    }
    if family == qualification.RECOVERY_FAMILIES[0]:
        case.update(rho=0.001, fit_iterations=10, fit_gradient_infinity=5e-7)
        gate["every_authorized_rank_qualified"] = True
    else:
        case.update(
            rank=rank, named_sparse_edges=16, rho_low=0.001, rho_sparse=0.001,
            selected_anchor_rho=0.1, core_lasso_lambda_maximum=0.02,
            fit_sweeps=10, fit_joint_stationarity_infinity=5e-7,
        )
        gate.update(every_possible_activated_rank_qualified=True, named_sparse_edges=16)
    if context:
        gate["qualification_context"] = qualification._case_context(family, rank)
        gate["producer_job_id"] = "123"
    return gate


class RecoveryExecutionTests(unittest.TestCase):
    def _pool(self, failed_case=None):
        submissions = []

        class Pool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, function, family, rank):
                submissions.append((function, family, rank))
                future = Future()
                if (family, rank) == failed_case:
                    future.set_exception(RuntimeError("injected case failure"))
                else:
                    future.set_result((family, rank, passing_gate(family, rank)))
                return future

        return Pool, submissions

    def test_all_twelve_cases_are_saved_and_merged_in_frozen_order(self):
        pool, submissions = self._pool()
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory)
            with mock.patch.object(qualification, "ProcessPoolExecutor", pool):
                with redirect_stdout(io.StringIO()):
                    gates = qualification._parallel_recovery_gates(12, progress)
            self.assertEqual(len(submissions), 12)
            self.assertEqual(len(list(progress.glob("*.json"))), 12)
            for family, gate in gates.items():
                self.assertEqual(gate["ranks"], [1, 2, 3, 4, 6, 8])
                self.assertEqual(list(gate["cases"]), ["1", "2", "3", "4", "6", "8"])
                saved = json.loads((progress / f"{family}_rank_8.json").read_text())
                self.assertEqual(saved["ranks"], [8])
                self.assertEqual(saved["qualification_context"], qualification._case_context(family, 8))
            self.assertTrue(gates["low_rank_population_recovery"]["every_authorized_rank_qualified"])
            self.assertTrue(gates["sparse_plus_low_rank_population_recovery"]["every_possible_activated_rank_qualified"])

    def test_failed_case_cannot_emit_pass_and_other_completions_survive(self):
        family = "sparse_plus_low_rank_population_recovery"
        pool, submissions = self._pool((family, 2))
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory)
            with mock.patch.object(qualification, "ProcessPoolExecutor", pool):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(qualification.OptionalQualificationError, "injected case failure"):
                        qualification._parallel_recovery_gates(12, progress)
            self.assertEqual(len(submissions), 12)
            self.assertEqual(len(list(progress.glob("*.json"))), 11)
            self.assertFalse((progress / f"{family}_rank_2.json").exists())

    def _three_saved_cases(self, directory, *, context=True):
        directory.mkdir()
        family = qualification.RECOVERY_FAMILIES[0]
        for rank in (1, 2, 3):
            (directory / f"{family}_rank_{rank}.json").write_text(
                json.dumps(passing_gate(family, rank, context=context)), encoding="utf-8",
            )

    def test_resume_submits_only_nine_and_merges_all_twelve(self):
        pool, submissions = self._pool()
        with tempfile.TemporaryDirectory() as directory:
            source, progress = Path(directory) / "source", Path(directory) / "new"
            self._three_saved_cases(source)
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            saved = qualification._load_completed_cases(source)
            with mock.patch.object(qualification, "ProcessPoolExecutor", pool), \
                 redirect_stdout(io.StringIO()):
                gates = qualification._parallel_recovery_gates(9, progress, saved)
            self.assertEqual(len(submissions), 9)
            self.assertFalse(any(family == qualification.RECOVERY_FAMILIES[0] and rank in (1, 2, 3)
                                 for _, family, rank in submissions))
            self.assertEqual(len(list(progress.glob("*.json"))), 12)
            for family, gate in gates.items():
                self.assertEqual(list(gate["cases"]), ["1", "2", "3", "4", "6", "8"])
                for rank in qualification.LOW_RANK_RANKS:
                    self.assertEqual(gate["cases"][str(rank)], passing_gate(family, rank)["cases"][str(rank)])
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})

    def test_failed_resumed_run_preserves_old_and_new_completed_cases(self):
        family = qualification.RECOVERY_FAMILIES[1]
        pool, submissions = self._pool((family, 2))
        with tempfile.TemporaryDirectory() as directory:
            source, progress = Path(directory) / "source", Path(directory) / "new"
            self._three_saved_cases(source)
            saved = qualification._load_completed_cases(source)
            with mock.patch.object(qualification, "ProcessPoolExecutor", pool), \
                 redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(qualification.OptionalQualificationError, "injected case failure"):
                qualification._parallel_recovery_gates(9, progress, saved)
            self.assertEqual(len(submissions), 9)
            self.assertEqual(len(list(progress.glob("*.json"))), 11)
            self.assertFalse((progress / f"{family}_rank_2.json").exists())

    def test_legacy_context_only_allowed_at_known_source_and_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            self._three_saved_cases(source, context=False)
            with self.assertRaisesRegex(qualification.OptionalQualificationError, "context"):
                qualification._load_completed_cases(source)
            with mock.patch.object(qualification, "LEGACY_COMPLETED_CASES", source):
                loaded = qualification._load_completed_cases(source)
            for saved in loaded.values():
                self.assertNotIn("qualification_context", saved["gate"])
                self.assertTrue(saved["provenance"]["legacy_context_from_recorded_producer"])
                self.assertEqual(saved["provenance"]["original_producer_job_id"], "46115632")

    def test_copied_legacy_cases_resume_only_if_unchanged_from_original(self):
        pool, _ = self._pool()
        with tempfile.TemporaryDirectory() as directory:
            original, copied = Path(directory) / "original", Path(directory) / "copied"
            self._three_saved_cases(original, context=False)
            with mock.patch.object(qualification, "LEGACY_COMPLETED_CASES", original):
                saved = qualification._load_completed_cases(original)
                with mock.patch.object(qualification, "ProcessPoolExecutor", pool), \
                     redirect_stdout(io.StringIO()):
                    qualification._parallel_recovery_gates(9, copied, saved)
                reloaded = qualification._load_completed_cases(copied)
                self.assertEqual(len(reloaded), 12)
                for rank in (1, 2, 3):
                    family = qualification.RECOVERY_FAMILIES[0]
                    name = f"{family}_rank_{rank}.json"
                    record = reloaded[(family, rank)]
                    self.assertNotIn("qualification_context", record["gate"])
                    self.assertEqual(record["provenance"]["artifact"], str(copied / name))
                    self.assertEqual(record["provenance"]["original_artifact"], str(original / name))
                    self.assertEqual(record["provenance"]["original_producer_job_id"], "46115632")
                path = copied / f"{qualification.RECOVERY_FAMILIES[0]}_rank_1.json"
                changed = json.loads(path.read_text(encoding="utf-8"))
                changed["cases"]["1"]["fit_gradient_infinity"] = 4e-7
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(qualification.OptionalQualificationError, "context"):
                    qualification._load_completed_cases(copied)

    def test_rejects_wrong_context_threshold_nonpass_and_incomplete_evidence(self):
        family, rank = qualification.RECOVERY_FAMILIES[0], 1
        baseline = passing_gate(family, rank, context=True)
        mutations = (
            ("seed", lambda gate: gate["qualification_context"].update(seed=7)),
            ("family", lambda gate: gate["qualification_context"].update(family=qualification.RECOVERY_FAMILIES[1])),
            ("rank", lambda gate: gate.update(ranks=[2])),
            ("threshold", lambda gate: gate.update(mean_conditional_KL_maximum_bits=1e-3)),
            ("status", lambda gate: gate.update(status="incomplete")),
            ("repeat", lambda gate: gate["cases"]["1"].update(deterministic_repeat_bit_exact=False)),
            ("nonfinite", lambda gate: gate["cases"]["1"].update(fit_gradient_infinity=float("nan"))),
            ("gradient", lambda gate: gate["cases"]["1"].update(fit_gradient_infinity=1e-3)),
            ("missing", lambda gate: gate["cases"]["1"].pop("lambda_reference")),
            ("parameters", lambda gate: gate["cases"]["1"].update(rho=0.1)),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            path = source / f"{family}_rank_{rank}.json"
            for label, mutate in mutations:
                with self.subTest(label=label):
                    gate = copy.deepcopy(baseline)
                    mutate(gate)
                    path.write_text(json.dumps(gate), encoding="utf-8")
                    with mock.patch.object(qualification, "ProcessPoolExecutor") as pool, \
                         self.assertRaises(qualification.OptionalQualificationError):
                        qualification._load_completed_cases(source)
                    pool.assert_not_called()
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(qualification.OptionalQualificationError, "malformed"):
                qualification._load_completed_cases(source)

    def test_report_records_reuse_without_counting_old_runtime(self):
        pool, submissions = self._pool()
        fixture = SimpleNamespace(cells=(), moments=(), candidates=(), geometry=(), pairs=())
        transform = SimpleNamespace(diagnostics={"rank_stable": True}, rank=3)
        with tempfile.TemporaryDirectory() as directory:
            source, progress = Path(directory) / "source", Path(directory) / "new"
            self._three_saved_cases(source)
            with mock.patch.object(qualification, "_validate_runtime"), \
                 mock.patch.object(qualification.core_qualification, "_frozen_fixture", return_value=fixture), \
                 mock.patch.object(qualification.m2, "fit_residual_transform", return_value=(transform, object())), \
                 mock.patch.object(qualification, "qualify_optional_trigger_diagnostic", return_value={"status": "pass"}), \
                 mock.patch.object(qualification, "ProcessPoolExecutor", pool), \
                 mock.patch.object(qualification.time, "monotonic", side_effect=(100.0, 105.0)), \
                 mock.patch.dict(os.environ, {"SLURM_JOB_ID": "456"}), \
                 redirect_stdout(io.StringIO()):
                report = qualification.run_qualification(workers=9, progress_dir=progress,
                                                        resume_completed_cases=source)
            self.assertEqual(len(submissions), 9)
            self.assertEqual(report["elapsed_wall_seconds"], 5.0)
            self.assertEqual(report["execution"]["new_recovery_case_count"], 9)
            self.assertEqual(report["execution"]["reused_recovery_case_count"], 3)
            self.assertIn("excludes_reused_case_runtime", report["execution"]["elapsed_wall_seconds_scope"])
            self.assertEqual(len(report["execution"]["reused_recovery_cases"]), 3)
            self.assertEqual(report["gate_inventory"]["required_gate_count"], 3)

    def test_worker_keeps_one_hybrid_case_inside_one_call(self):
        fixture = SimpleNamespace(cells=(), moments=(), candidates=(), geometry=(), pairs=())
        transform = SimpleNamespace(diagnostics={"rank_stable": True})
        prepared = object()
        gate = {"status": "pass", "ranks": [6], "cases": {"6": {"status": "pass"}}}
        with mock.patch.object(qualification, "_validate_runtime"), \
             mock.patch.object(qualification.core_qualification, "_frozen_fixture", return_value=fixture), \
             mock.patch.object(qualification.m2, "fit_residual_transform", return_value=(transform, prepared)), \
             mock.patch.object(qualification, "qualify_hybrid_population_recovery", return_value=gate) as hybrid:
            result = qualification._run_recovery_case("sparse_plus_low_rank_population_recovery", 6)
        hybrid.assert_called_once_with(fixture, prepared, ranks=(6,))
        self.assertEqual(result, ("sparse_plus_low_rank_population_recovery", 6, gate))

    def test_worker_count_obeys_allocated_cpus(self):
        environment = {
            "SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "6",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        }
        with mock.patch.dict(os.environ, environment):
            qualification._validate_runtime(6)
            with self.assertRaisesRegex(qualification.OptionalQualificationError, "exceed requested Slurm CPUs"):
                qualification._validate_runtime(12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
