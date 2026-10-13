"""Focused numerical-repair checks; coordinator-owned compute only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_twice_mz_mvc_repair_pilot as repair
import summarize_twice_mz_menu_wave as collector
import test_twice_mz_menu_pilot as prior_tests


class TwiceMVCRepairPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior_tests.TwiceMenuPilotTests.setUpClass()
        cls.prior = prior_tests.TwiceMenuPilotTests(
            "test_eight_fresh_fits_preserve_source_baseline_and_all_nine_nested_edges")
        cls.helpers, cls.spec = cls.prior.helpers, deepcopy(cls.prior.spec)
        cls.source = deepcopy(cls.prior.source)
        cls.source.update(started_utc="2026-10-05T10:00:00+00:00", finished_utc=repair.SOURCE_FINISHED)
        cls.failed = cls.prior.exercise(nonconverged_at=7)[0]
        cls.failed.update(job_id=repair.FAILED_JOB, started_utc="2026-10-05T16:45:45.659386+00:00",
            finished_utc=repair.FAILED_FINISHED, source_finished_utc=repair.SOURCE_FINISHED,
            source_report=str(repair.SOURCE_REPORT), source_dir=str(repair.SOURCE_REPORT.parent),
            elapsed_seconds=882.4487084900029, error=repair.NONCONVERGENCE)
        last = cls.failed["fits"][-1]
        last.update(subject_optimizer_failures=[], elapsed_seconds=319.1398453810252,
            objective_history=[float(10500 - i) for i in range(498)] + [9574.702666643396, 9574.700473306923],
            support_contacts=[{"level": "version_log_standard_deviation", "version": version,
                "parameter": name, "value": -3.9999998503354774, "bounds": [-4, 1]}
                for version, name in (("EI", "eta"), ("EI", "criterion"), ("ER", "kappa"),
                                      ("ER", "eta"), ("ER", "criterion"))])

    def exercise(self, *, source_change=None, failed_change=None, replay_error=False,
                 nonconverged_at=None, over_ceiling=False, worker_error_at=None, expiry=None):
        source, failed, spec = deepcopy((self.source, self.failed, self.spec))
        if source_change:
            source_change(source)
        if failed_change:
            failed_change(failed)
        calls, likelihoods, checkpoints, validations, now = [], [], [], [], [100.]
        runtime = repair.runtime_specification(spec)
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        original_writer, original_reader = repair.write_report, Path.read_text
        original_validator = self.helpers.core.validate_fit

        def validate(fit, model, train, held, supplied, people):
            validations.append((model, list(train), deepcopy(supplied)))
            return original_validator(fit, model, train, held, supplied, people)

        def likelihood(theta, names, data, supplied, rival):
            self.assertEqual((supplied, rival), (spec, "fixed"))
            self.assertEqual({len(value) for value in data.values()}, {128})
            likelihoods.append(tuple(names))
            focal = tuple(names[3:])
            value = {(): -1., ("rho",): -.98, ("eta", "criterion"): -.975}[focal]
            if focal != ("eta", "criterion"):
                value += 1e-4 if replay_error else 3e-13
            return np.full(128, value)

        def fitter(training, model, rival, supplied, *, workers, max_seconds, progress):
            self.assertEqual((model, rival, supplied, workers), ("MVC", "fixed", runtime, 4))
            self.assertGreater(max_seconds, 0.)
            self.assertLessEqual(max_seconds, 5395.)
            self.assertEqual(sorted(training), sorted(self.prior.people))
            self.assertTrue(all(len(record["rows"]) == 128 for record in training.values()))
            runs = sorted({int(row["run"]) for record in training.values() for row in record["rows"]})
            calls.append((runs, deepcopy(supplied)))
            self.assertEqual(runs, [[1, 4], [2, 3]][len(calls) - 1])
            if worker_error_at == len(calls):
                error = TimeoutError("mocked bounded worker failure")
                error.parallel_progress = {"workers_cleaned_up": True, "stage": "conditional", "sweep": 2,
                    "completed_results": {}, "pending_participants": sorted(training), "worker_pids": [1, 2, 3, 4]}
                raise error
            converged = nonconverged_at != len(calls)
            fit = self.prior.fixture.fixture.fit("MVC", converged=converged)
            fit["subject_optimizer_failures"] = []
            fit["objective_history"] = ([float(600 - i) for i in range(500)] + [101.]) if converged else [float(1501 - i) for i in range(1500)]
            if over_ceiling:
                fit["objective_history"] = [101.] * 1501
            progress({"stage": "sweep", "sweep": len(fit["objective_history"]),
                "objective": fit["objective_history"][-1], "converged": converged, "subject_optimizer_failures": []})
            return fit

        def writer(output, report):
            checkpoints.append(deepcopy(report))
            original_writer(output, report)
            if expiry == "staged" and report["active_fit"] is not None and report["fresh_fit_calls"] == 0:
                now[0] = 5501.
            if expiry == "returned_fit" and len(report["fits"]) == 1:
                now[0] = 5501.
            if expiry == "publication" and report["mvc_repair_complete"]:
                now[0] = 5501.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 5501.

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            episode = Path(directory)
            source_path = episode / "outputs/runs/preserved/twice/replicate0/mz_effect_level_recovery.json"
            failed_path = episode / "outputs/runs/failed/twice_mz_menu_pilot.json"
            source_path.parent.mkdir(parents=True)
            failed_path.parent.mkdir(parents=True)
            failed.update(source_report=str(source_path), source_dir=str(source_path.parent))
            source_path.write_text(json.dumps(source, allow_nan=False))
            (source_path.parent / "generating_parameters.json").write_text(json.dumps(self.prior.truth))
            (source_path.parent / "simulated_events.csv.gz").write_bytes(self.prior.events)
            failed_path.write_text(json.dumps(failed, allow_nan=False))
            implementation = episode / "original.json"
            implementation.write_text(json.dumps(spec))
            output = episode / "outputs/runs/new"
            inputs = [source_path, failed_path, implementation,
                source_path.parent / "generating_parameters.json", source_path.parent / "simulated_events.csv.gz"]
            frozen = {path: path.read_bytes() for path in inputs}

            def reader(path, *args, **kwargs):
                if path not in inputs:
                    raise AssertionError(f"No index9, wave, or other source may be read: {path}")
                value = original_reader(path, *args, **kwargs)
                if expiry == "failed_read" and path == failed_path:
                    now[0] = 5501.
                return value

            for context in (patch.object(repair, "EPISODE", episode),
                    patch.object(repair, "SOURCE_REPORT", source_path), patch.object(repair, "FAILED_REPORT", failed_path),
                    patch.object(repair, "IMPLEMENTATION", implementation), patch.object(repair, "require_compute"),
                    patch.object(collector, "baseline_path", side_effect=lambda index: source_path if index == 0 else self.fail("Only0")),
                    patch.dict(os.environ, {**{name: "1" for name in repair.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}),
                    patch.object(repair.pilot, "load_helpers", return_value=self.helpers),
                    patch.object(repair.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(repair.signal, "alarm", side_effect=alarm),
                    patch.object(repair, "write_report", side_effect=writer), patch.object(Path, "read_text", reader),
                    patch.object(self.helpers.parallel, "fit_partially_pooled_parallel", side_effect=fitter),
                    patch.object(self.helpers.shared, "trial_likelihood", side_effect=likelihood),
                    patch.object(self.helpers.core, "validate_fit", side_effect=validate)):
                stack.enter_context(context)
            report = repair.run(argparse.Namespace(implementation=implementation, output_dir=output, max_seconds=5400))
            self.assertEqual(report, json.loads(original_reader(output / repair.REPORT_NAME)))
            self.assertEqual(frozen, {path: path.read_bytes() for path in inputs})
        self.assertEqual({sig: signal.getsignal(sig) for sig in old_handlers}, old_handlers)
        self.assertEqual(spec, self.spec)
        return report, calls, likelihoods, checkpoints, validations

    def test_two_fresh_runtime1500_halves_preserve_original500_and_completed_prefix_exactly(self):
        report, calls, likelihoods, checkpoints, validations = self.exercise()
        self.assertTrue(report["mvc_repair_complete"])
        self.assertEqual(report["status"], "mvc_repair_completed")
        self.assertEqual(len(calls), 2)
        self.assertEqual([len(fit["objective_history"]) for fit in report["fits"]], [501, 501])
        self.assertEqual(report["implementation"], self.spec)
        changed = deepcopy(report["runtime_implementation"])
        changed["optimizer"]["maximum_sweeps"] = 500
        self.assertEqual(changed, self.spec)
        self.assertEqual([row[2]["optimizer"]["maximum_sweeps"] for row in validations], [1500, 1500])
        old = report["numerical_repair"]
        for field, value in (("previous_completed_fits", self.failed["fits"][:6]),
                ("previous_scored_components", self.failed["scored_components"]),
                ("previous_participant_scores", self.failed["participant_scores"]),
                ("previous_MVC_fit", self.failed["fits"][-1])):
            self.assertEqual(old[field], value)
        self.assertEqual(old["previous_process_elapsed_seconds"], self.failed["elapsed_seconds"])
        self.assertNotIn("cumulative_process_elapsed_seconds", report)
        self.assertEqual(len(likelihoods), 6 * 108)
        self.assertTrue(report["parallel_workers_cleaned_up"])
        for person, values in report["participant_scores"].items():
            for key in ("M0", "MZ", "version"):
                self.assertEqual(values[key], self.source["participant_scores"][person][key])
            self.assertEqual(values["MVC"], sum(row["participant_contributions"][person] for row in report["scored_components"]))
        self.assertEqual(report["completed_rivals"], ["MVC"])
        for checkpoint in checkpoints:
            self.assertEqual((checkpoint["replicate_index"], checkpoint["seed"]), (0, 1900057))
            self.assertTrue(all(checkpoint[field] == "not_evaluated" for field in repair.CLAIMS))
            self.assertFalse(checkpoint["automatic_retry"] or checkpoint["retry_wave_selected"] or checkpoint["full_menu_wave_selected"])
            self.assertNotIn("nested_comparisons", checkpoint)

    def test_failed_identity_cause_and_completed_prefix_are_required_before_new_fits(self):
        changes = (lambda r: r.update(job_id="1"), lambda r: r.update(finished_utc="2026-10-05T17:00:29+00:00"),
            lambda r: r.update(error="Timeout"), lambda r: r.update(parallel_workers_cleaned_up=False),
            lambda r: r["fits"][-1].update(subject_optimizer_failures=[{"participant": "p000"}]),
            lambda r: r["fits"][-1].update(objective_history=[100.] * 500),
            lambda r: r["fits"][-1]["objective_history"].pop(),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r["participant_scores"]["p000"].update(MC=-.1))
        for change in changes:
            with self.subTest(change=change):
                report, calls, _, _, _ = self.exercise(failed_change=change)
                self.assertFalse(report["mvc_repair_complete"])
                self.assertFalse(report["repair_context_validated"])
                self.assertEqual(calls, [])

    def test_wrong_original_source_or_failed_replay_stops_before_fitter(self):
        for options in ({"source_change": lambda r: r.update(seed=910031)}, {"replay_error": True}):
            with self.subTest(options=options):
                report, calls, _, _, _ = self.exercise(**options)
                self.assertEqual(calls, [])
                self.assertFalse(report["saved_scores_replayed"])
                self.assertEqual(report["status"], "technical_mvc_repair_failure")

    def test_first_nonconvergence_stops_reverse_and_second_failure_preserves_scored_prefix(self):
        for number in (1, 2):
            with self.subTest(number=number):
                report, calls, _, _, _ = self.exercise(nonconverged_at=number)
                self.assertEqual(len(calls), number)
                self.assertEqual(len(report["fits"]), number)
                self.assertEqual(len(report["scored_components"]), number - 1)
                self.assertFalse(report["fits"][-1]["converged"])
                self.assertFalse(report["mvc_repair_complete"])
                self.assertEqual(report["completed_rivals"], [])
                self.assertEqual(report["numerical_repair"]["previous_completed_rivals"], list(repair.PRESERVED))
        report, calls, _, _, _ = self.exercise(over_ceiling=True)
        self.assertEqual(len(calls), 1)
        self.assertFalse(report["mvc_repair_complete"])
        self.assertIn("convergence evidence", report["error"])

    def test_worker_failure_preserves_cleanup_and_first_scored_half(self):
        report, calls, _, _, _ = self.exercise(worker_error_at=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(report["fits"]), 1)
        self.assertEqual(len(report["scored_components"]), 1)
        self.assertTrue(report["parallel_workers_cleaned_up"])
        self.assertEqual(report["parallel_failure"]["stage"], "conditional")
        self.assertFalse(report["mvc_repair_complete"])

    def test_timed_reads_staging_return_cleanup_and_publication_retain_technical_evidence(self):
        for stage, count in (("failed_read", 0), ("staged", 0), ("returned_fit", 1), ("cleanup", 2), ("publication", 2)):
            with self.subTest(stage=stage):
                report, calls, _, checkpoints, _ = self.exercise(expiry=stage)
                self.assertEqual(len(calls), count)
                self.assertFalse(report["mvc_repair_complete"])
                self.assertEqual(report["status"], "technical_mvc_repair_failure")
                self.assertIn("deadline", report["error"].lower())
                self.assertFalse(checkpoints[-1]["mvc_repair_complete"])
                if stage == "returned_fit":
                    self.assertEqual(len(report["fits"]), 1)
                    self.assertEqual(report["scored_components"], [])

    def test_four_cpu_guard_precedes_legacy_import_and_historical_paths_are_protected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.dict(os.environ, {**{name: "1" for name in repair.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}), \
                patch.object(repair.pilot, "load_helpers", side_effect=AssertionError("No legacy import")) as loader:
            args = argparse.Namespace(implementation=repair.IMPLEMENTATION,
                                      output_dir=Path(directory) / "new", max_seconds=5400)
            for name, value in (("SLURM_CPUS_PER_TASK", "1"), ("SLURM_CPUS_PER_TASK", "8"), ("BLIS_NUM_THREADS", "2")):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    repair.run(args)
            for field, value in (("max_seconds", 5401), ("max_seconds", True), ("implementation", Path("other.json"))):
                changed = argparse.Namespace(**vars(args))
                setattr(changed, field, value)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    repair.run(changed)
            for output in (args.output_dir, repair.SOURCE_REPORT.parent / "new", repair.FAILED_REPORT.parent / "new"):
                args.output_dir = output
                with self.assertRaises(FileExistsError):
                    repair.run(args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
