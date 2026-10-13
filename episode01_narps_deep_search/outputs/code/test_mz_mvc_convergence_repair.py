"""Focused two-mode MVC repair tests; execute only in coordinated compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import run_mz_mvc_convergence_repair as repair
import test_mz_mvc_repaired_menu as fixtures
from test_mz_null_representative import THREADS


class ConvergenceRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RepairedMenuTests.setUpClass()
        cls.menu_fixture = fixtures.RepairedMenuTests("test_existing_nine_contrasts_and_original_validation_remain_with_explicit_runtime_mvc")
        cls.fixture = cls.menu_fixture.fixture
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def source(self, index):
        base = self.menu_fixture.reports()[0]
        base.update(replicate_index=index, seed=repair.saved.cell_seed("MZ", index))
        return base

    def failed(self, index=2, second=False):
        fits = [self.fixture.fit("MVC", train, held) for train, held in repair.saved.HALVES[:2 if second else 1]]
        for fit in fits:
            fit["elapsed_seconds"] = 600.
        fits[-1].update(converged=False, objective_history=[float(value) for value in range(500, 0, -1)],
                        elapsed_seconds=800.)
        return {
            "analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_saved_replicate_rival_diagnostics",
            "generating_model": "MZ", "fitted_model": "MVC", "replicate_index": index,
            "seed": repair.saved.cell_seed("MZ", index), "source_dir": str(repair.saved.source_directory(index)),
            "source_finished_utc": self.source(index)["finished_utc"], "implementation": deepcopy(self.spec),
            "started_utc": "2026-10-05T05:00:00+00:00", "finished_utc": "2026-10-05T06:00:00+00:00",
            "participants": 108, "runs": 432, "trials": 27648,
            "status": "technical_rival_failure", "rival_diagnostics_complete": False,
            "error": "RuntimeError: Nonconverged fit is technical; it is not a recovery failure",
            "saved_scores_replayed": True, "saved_score_replay_max_absolute_discrepancy": 0.,
            "saved_score_replay_absolute_tolerance": 1e-12,
            "scored_primary_directions": [{"training_runs": [1, 4], "scoring_runs": [2, 3]}] if second else [],
            "completed_rivals": [], "diagnostics": {}, "elapsed_seconds": 1500. if second else 900.,
            "fits": fits, "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
            "observed_choices_or_RTs_used": False, "full_recovery_pass": None, "episode_terminal": False,
            **{field: "not_evaluated" for field in repair.CLAIM_FIELDS[:5]},
        }

    def args(self, output, index=2, **changes):
        values = dict(replicate_index=index, implementation=repair.saved.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=repair.REUSE_MAX_SECONDS if index == 3 else repair.MAX_SECONDS)
        values.update(changes)
        return argparse.Namespace(**values)

    def references(self):
        saved = repair.saved
        return (saved.validate_source, saved.fit_partially_pooled, saved.validate_fit,
                saved.shared.score_fit, saved.write_report, saved.REPORT_NAME)

    def exercise(self, output, *, index=2, second=False, failed=None, failure=None, max_seconds=None):
        saved = repair.saved
        base = self.source(index)
        failed = self.failed(index, second) if failed is None else failed
        truth = deepcopy(self.fixture.truth)
        for params in truth.values():
            params.update(saved.focal_truth("MZ"))
        preserved = deepcopy((failed, base, truth, self.spec))
        allowed = {saved.IMPLEMENTATION: json.dumps(self.spec), repair.failed_path(index): json.dumps(failed),
                   saved.source_directory(index) / "single_process_recovery.json": json.dumps(base),
                   saved.source_directory(index) / "generating_parameters.json": json.dumps(truth)}
        calls, validations, checkpoints, scored, alarms = [], [], [], [], []
        validator, writer = saved.validate_fit, repair.write_report
        references = self.references()
        handlers = tuple(signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM))
        clock = [100.]
        args = self.args(output, index)
        if max_seconds is not None:
            args.max_seconds = max_seconds

        def reader(path, *unused, **kwargs):
            self.assertEqual(alarms[-1], args.max_seconds)
            if failure == "source_timeout" and path == repair.failed_path(index):
                signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
            return allowed[path]

        def fitter(training, model, rival, spec):
            self.assertEqual(alarms[-1], args.max_seconds)
            calls.append((training, model, rival, deepcopy(spec)))
            if failure == "fit_timeout":
                signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
            converged = failure != "fit"
            fit = self.fixture.fit(model, converged=converged)
            count = 1067 if converged else 1500
            fit["objective_history"] = [float(value) for value in range(count, 1, -1)] + [2. if converged else 1.]
            return fit

        def validate(fit, model, train, held, spec, people):
            validations.append((model, tuple(train), spec["optimizer"]["maximum_sweeps"]))
            return validator(fit, model, train, held, spec, people)

        def score(fit, scoring, spec, check):
            check()
            scored.append((fit["model"], tuple(fit["training_runs"]), scoring))
            self.assertEqual(spec, self.spec)
            if failure == "rescore" and fit.get("repair_execution") == "reused_validated_original_fit":
                raise RuntimeError("Synthetic preserved-first-fit scoring failure")
            value = {"M0": -1., "MZ": -.99, "MVC": -.98}[fit["model"]]
            if failure == "replay" and fit["model"] == "M0":
                value += .001
            return {person: value for person in self.people}

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            result = writer(directory, report)
            if report["rival_diagnostics_complete"]:
                self.assertEqual(alarms[-1], args.max_seconds)
                if failure == "finalization_timeout":
                    clock[0] = 100. + args.max_seconds
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(saved, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(repair.time, "monotonic", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(repair.signal, "alarm", side_effect=alarms.append))
            stack.enter_context(patch.object(saved.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(saved, "load_split", side_effect=lambda events, train, held: (tuple(train), tuple(held))))
            stack.enter_context(patch.object(saved, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(saved, "validate_fit", side_effect=validate))
            stack.enter_context(patch.object(saved.shared, "score_fit", side_effect=score))
            stack.enter_context(patch.object(repair, "write_report", side_effect=checkpoint))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.references()
            report = repair.run(args)
            self.assertEqual(self.references(), installed)
        self.assertEqual(self.references(), references)
        self.assertEqual((failed, base, truth, self.spec), preserved)
        self.assertEqual(handlers, tuple(signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)))
        self.assertEqual(alarms, [args.max_seconds, 0])
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        return report, calls, validations, checkpoints, base, scored

    def assert_boundary(self, report, checkpoints, prior):
        self.assertGreater(len(checkpoints), 1)
        for checkpoint in checkpoints:
            self.assertEqual(checkpoint["phase"], repair.PHASE)
            if "implementation" in checkpoint:
                self.assertEqual(checkpoint["implementation"], self.spec)
                runtime = deepcopy(checkpoint["runtime_implementation"])
                self.assertEqual(runtime["optimizer"]["maximum_sweeps"], 1500)
                runtime["optimizer"]["maximum_sweeps"] = 500
                self.assertEqual(runtime, self.spec)
            else:
                self.assertFalse(checkpoint["repair_context_validated"])
                self.assertEqual(checkpoint["fits"], [])
            self.assertEqual(checkpoint["numerical_repair"]["original_maximum_sweeps"], 500)
            self.assertEqual(checkpoint["numerical_repair"]["runtime_maximum_sweeps"], 1500)
            for field in repair.CLAIM_FIELDS:
                self.assertEqual(checkpoint[field], "not_evaluated")
            for field in ("automatic_retry", "retry_wave_selected", "full_menu_wave_selected", "episode_terminal", "wave_selected"):
                self.assertFalse(checkpoint[field])
            self.assertIsNone(checkpoint["full_recovery_pass"])
            self.assertFalse(checkpoint["numerical_repair"]["nonconverged_checkpoint_state_reused"])
        self.assertEqual(report["cumulative_process_elapsed_seconds"], prior + report["elapsed_seconds"])

    def test_first_half_exhaustion_freshly_fits_both_halves_with_original_baseline_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, validations, checkpoints, base, scored = self.exercise(Path(directory) / "first")
        self.assertTrue(report["rival_diagnostics_complete"])
        self.assertEqual([(train, model, rival) for train, model, rival, spec in calls],
                         [((1, 4), "MVC", "fixed"), ((2, 3), "MVC", "fixed")])
        for train, model, rival, spec in calls:
            expected = deepcopy(self.spec)
            expected["optimizer"]["maximum_sweeps"] = 1500
            self.assertEqual(spec, expected)
        self.assertEqual([ceiling for model, train, ceiling in validations], [500] * 4 + [1500] * 2)
        self.assertEqual([fit["repair_execution"] for fit in report["fits"]], ["fresh_fit"] * 2)
        self.assertEqual(report["numerical_repair"]["classification"], repair.FIRST)
        self.assertEqual(report["internal_seconds"], 12600)
        self.assertEqual(report["numerical_repair"]["qualifying_MZ42_report"], str(repair.QUALIFYING_REPORT))
        for person, row in base["participant_scores"].items():
            self.assertEqual({key: report["participant_scores"][person][key] for key in row}, row)
        self.assert_boundary(report, checkpoints, 900.)

    def test_second_half_reuses_and_rescores_original_first_fit_without_double_counting_cost(self):
        old = self.failed(3, True)
        with tempfile.TemporaryDirectory() as directory:
            report, calls, validations, checkpoints, base, scored = self.exercise(Path(directory) / "second", index=3, second=True)
        self.assertTrue(report["rival_diagnostics_complete"])
        self.assertEqual([(train, model) for train, model, rival, spec in calls], [((2, 3), "MVC")])
        self.assertEqual([ceiling for model, train, ceiling in validations], [500] * 6 + [1500])
        for key, value in old["fits"][0].items():
            if key != "elapsed_seconds":
                self.assertEqual(report["fits"][0][key], value)
        self.assertEqual(report["fits"][0]["original_fit_elapsed_seconds"], 600.)
        metadata = report["numerical_repair"]
        self.assertEqual([row["fit_maximum_sweeps"] for row in metadata["directions"]], [500, 1500])
        self.assertEqual(report["internal_seconds"], 6600)
        self.assertEqual(metadata["previous_fit_elapsed_seconds"], [600., 800.])
        self.assertEqual(metadata["reused_first_fit_materialization_elapsed_seconds"], report["fits"][0]["elapsed_seconds"])
        self.assertGreaterEqual(metadata["reused_first_fit_rescore_elapsed_seconds"], 0.)
        self.assertTrue(metadata["reused_first_fit_rescored"])
        self.assertEqual(metadata["reused_MVC_score_replay"], "not_available_original_participant_scores_were_discarded")
        self.assertEqual([(model, train, scoring) for model, train, scoring in scored if model == "MVC"],
                         [("MVC", (1, 4), (2, 3)), ("MVC", (2, 3), (1, 4))])
        self.assert_boundary(report, checkpoints, 1500.)

    def test_bad_identity_claim_or_unresolved_cause_is_rejected_without_mutation(self):
        changes = (("replicate_index", 3), ("seed", 1), ("phase", repair.PHASE), ("generating_model", "M0"),
                   ("source_dir", "/other/source"), ("family_selection", "evaluated"),
                   ("error", "RuntimeError: Internal whole-process deadline exceeded"),
                   ("saved_score_replay_max_absolute_discrepancy", 2e-12), ("scored_primary_directions", [{}]))
        for field, value in changes:
            failed = self.failed()
            failed[field] = value
            preserved = deepcopy(failed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.repair_context(2, self.spec, failed)
            self.assertEqual(failed, preserved)
        for field, value in (("converged", True), ("subject_optimizer_failures", ["failed"]),
                             ("objective_history", [1.] * 500)):
            failed = self.failed()
            failed["fits"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                repair.repair_context(2, self.spec, failed)

    def test_actual_source_timestamp_and_participants_are_required_before_fitting(self):
        for mismatch in ("timestamp", "participants"):
            failed = self.failed(3, True)
            if mismatch == "timestamp":
                failed["source_finished_utc"] = "2026-10-05T00:00:00+00:00"
            else:
                for fit in failed["fits"]:
                    fit["participants"] = [f"wrong{index:03d}" for index in range(108)]
            # This wrong shared identity can pass inventory-only prior-half
            # validation, but must fail against the actual loaded cohort.
            repair.repair_context(3, self.spec, failed)
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, base, scored = self.exercise(
                    Path(directory) / "mismatch", index=3, second=True, failed=failed)
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertEqual(calls, [])
            self.assertEqual(report["fits"], [])
            self.assertFalse(report["numerical_repair"]["source_and_preserved_fit_validated_against_actual_cohort"])
            self.assertNotIn("participant_scores", report)

    def test_replay_or_preserved_fit_rescoring_failure_stops_before_new_optimization(self):
        for failure in ("replay", "rescore"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, base, scored = self.exercise(
                    Path(directory) / failure, index=3, second=True, failure=failure)
            self.assertEqual(calls, [])
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertEqual(len(report["fits"]), 0 if failure == "replay" else 1)
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertFalse(report["numerical_repair"]["reused_first_fit_rescored"])
            self.assert_boundary(report, checkpoints, 1500.)

    def test_fresh_fit_failure_preserves_checkpoints_and_stops_without_final_scores(self):
        for second in (False, True):
            with self.subTest(second=second), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, base, scored = self.exercise(
                    Path(directory) / "failure", index=3 if second else 2, second=second, failure="fit")
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertEqual(len(report["fits"]), 2 if second else 1)
            self.assertFalse(report["fits"][-1]["converged"])
            self.assertEqual(len(report["fits"][-1]["objective_history"]), 1500)
            self.assertEqual(len(report["scored_primary_directions"]), 1 if second else 0)
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["diagnostics"], {})
            self.assertEqual(report["completed_rivals"], [])
            self.assert_boundary(report, checkpoints, 1500. if second else 900.)

    def test_compute_scope_cpu_time_and_protected_output_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No metadata read")):
            with patch.object(repair, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                repair.run(self.args(Path(directory) / "fresh"))
            with patch.object(repair, "require_compute"):
                for changes in ([{"replicate_index": index} for index in (0, 1, 42, 92, 105, 200, True)] +
                                [{"max_seconds": 0}, {"max_seconds": 12601},
                                 {"implementation": Path(directory) / "other.json"}]):
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        repair.run(self.args(Path(directory) / "fresh", **changes))
                for name in ("SLURM_CPUS_PER_TASK", *THREADS):
                    with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                        repair.run(self.args(Path(directory) / "fresh"))
                for output in (Path(directory), repair.FAILED_ROOT / "new", repair.saved.source_directory(2) / "new",
                               repair.QUALIFYING_REPORT.parent / "new"):
                    with self.subTest(output=output), self.assertRaises(FileExistsError):
                        repair.run(self.args(output))

    def test_whole_process_caps_cover_source_reads_fits_finalization_and_restore_signal_handlers(self):
        for failure in ("source_timeout", "fit_timeout", "finalization_timeout"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                report, calls, validations, checkpoints, base, scored = self.exercise(
                    Path(directory) / failure, index=3, second=True, failure=failure)
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            if failure == "source_timeout":
                self.assertEqual(calls, [])
                self.assertEqual(report["fits"], [])
                self.assertFalse(report["repair_context_validated"])
                self.assertNotIn("cumulative_process_elapsed_seconds", report)
            else:
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(report["fits"]), 1 if failure == "fit_timeout" else 2)
                self.assertTrue(report["numerical_repair"]["reused_first_fit_rescored"])
                self.assert_boundary(report, checkpoints, 1500.)
        with tempfile.TemporaryDirectory() as directory:
            report, calls, validations, checkpoints, base, scored = self.exercise(
                Path(directory) / "wrong_mode_cap", index=3, second=True, max_seconds=6601)
        self.assertEqual(calls, [])
        self.assertFalse(report["source_baseline_validated"])
        self.assertIn("bounded at6600", report["error"])
        self.assert_boundary(report, checkpoints, 1500.)


if __name__ == "__main__":
    unittest.main()
