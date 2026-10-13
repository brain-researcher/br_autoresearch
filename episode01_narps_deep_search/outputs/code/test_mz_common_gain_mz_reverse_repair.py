"""Exact dir1/MZ repair and four-worker orchestration checks; compute only."""
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

import numpy as np

import run_mz_common_gain_mz_reverse_repair as repair
import test_mz_common_gain_range_repair as fixtures


class CommonGainMZReverseRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CommonGainRangeRepairTests.setUpClass()
        cls.prior = fixtures.CommonGainRangeRepairTests()
        cls.spec, cls.people = cls.prior.spec, cls.prior.people
        cls.old_failures = cls.prior.failed
        cls.old_components = {}
        for direction in (0, 1):
            for model in ("M0", "MZ"):
                with tempfile.TemporaryDirectory() as directory:
                    report, *_ = cls.prior.exercise(Path(directory) / "old", direction, model)
                report.update(job_id=repair.FAILED_JOB, started_utc="2026-10-05T12:40:00+00:00",
                              finished_utc="2026-10-05T12:50:00+00:00")
                cls.old_components[direction, model] = report
        failed = cls.old_components[1, "MZ"]
        fit = failed["fits"][0]
        contacts = []
        for version_index, version in enumerate(("EI", "ER")):
            for name in ("kappa", "rho"):
                bounds = cls.spec["parameters"][name]["individual_log_scale_bounds"]
                value = bounds[0] + 1e-7
                contacts.append({"level": "version_log_standard_deviation", "version": version,
                    "parameter": name, "value": value, "bounds": bounds[:]})
                fit["version_standard_deviations"][version_index][fit["parameter_names"].index(name)] = float(np.exp(value))
        fit.update(converged=False, elapsed_seconds=repair.FAILED_FIT_SECONDS,
            objective_history=[repair.FAILED_TAIL[0] + 1] * 498 + repair.FAILED_TAIL,
            support_contacts=contacts)
        fit["parameters"][0][0] = 1.5
        failed.update(finished_utc=repair.FAILED_FINISHED, elapsed_seconds=repair.FAILED_PROCESS_SECONDS,
            status="technical_common_gain_failure", error=repair.previous.NONCONVERGENCE,
            common_gain_component_complete=False, scored_fits=[], completed_model_scores={})
        failed.pop("participant_scores")
        cls.failed = failed

    def source(self):
        return self.prior.source()

    def args(self, output, **changes):
        values = dict(source_dir=repair.SOURCE, implementation=repair.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=3600)
        values.update(changes)
        return argparse.Namespace(**values)

    def context(self, failed=None):
        source, _ = self.source()
        with patch.object(Path, "read_text", return_value=json.dumps(self.old_failures[1])):
            return repair.repair_context(self.failed if failed is None else failed,
                                         source, source["participant_scores"], self.spec)

    def exercise(self, output, *, bad_source=False, failed_change=None, replay_error=False,
                 nonconverged=False, over_ceiling=False, score_error=False, parallel_error=False,
                 deadline_stage=None, checkpoint_delay=0.):
        source, truth = self.source()
        failed, original_failed = deepcopy(self.failed), deepcopy(self.old_failures[1])
        if bad_source:
            source["replicate_index"] = 1
        if failed_change:
            failed_change(failed)
        before = deepcopy((source, truth, failed, original_failed, self.spec, self.people))
        bindings = self.prior.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        read_text, writer, now = Path.read_text, repair.write_report, [100.]
        original_alarm = signal.alarm
        fits, likelihoods, checkpoints, reads = [], [], [], []
        train, held = repair.common.selected_runs(1)
        delayed = [False]

        def reader(path, *args, **kwargs):
            if path == repair.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == repair.SOURCE / repair.common.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 3601
                return json.dumps(source)
            if path == repair.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            if path == repair.FAILED_REPORT:
                reads.append(path)
                if deadline_stage == "failed_read":
                    now[0] += 3601
                return json.dumps(failed)
            if path == repair.previous.failed_path(1):
                reads.append(path)
                return json.dumps(original_failed)
            if path == repair.previous.failed_path(0) or repair.FAILED_REPORT.parents[2] in path.parents:
                raise AssertionError("The repair must not reread or replace the other completed components")
            return read_text(path, *args, **kwargs)

        def split(path, a, b):
            self.assertEqual(path, repair.SOURCE / "simulated_events.csv.gz")
            self.assertIn((tuple(a), tuple(b)), repair.common.HALVES)
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(a), panel(b)

        def fitter(training, model, rival, runtime, *, workers, max_seconds, progress):
            self.assertEqual((model, rival, workers, os.environ["SLURM_CPUS_PER_TASK"]), ("MZ", "fixed", 4, "4"))
            self.assertEqual(runtime, repair.runtime_specification(self.spec))
            self.assertEqual(max_seconds, 3600 - repair.parallel.CLEANUP_SECONDS - 1 - checkpoint_delay)
            self.assertEqual(len(likelihoods), 432)
            self.assertEqual(set(training), set(self.people))
            for person, record in training.items():
                expected = [row for row in self.people[person] if int(row["run"]) in train and 10 <= float(row["gain"]) <= 20]
                self.assertEqual(record["rows"], expected)
                self.assertTrue(all(left is right for left, right in zip(record["rows"], expected)))
                self.assertTrue(all(int(row["run"]) not in held for row in record["rows"]))
            fits.append((model, deepcopy(runtime), workers, max_seconds))
            progress({"stage": "conditional", "sweep": 1, "participant": "p000", "completed_participants": 1,
                "total_participants": 108, "result": {"participant": "p000", "success": True}})
            if parallel_error:
                error = RuntimeError("bounded participant worker failed")
                error.parallel_progress = {"stage": "conditional", "sweep": 1,
                    "completed_results": {"p000": {"participant": "p000", "success": True}},
                    "pending_participants": sorted(self.people)[1:], "objective_history": [],
                    "worker_pids": [1001, 1002, 1003, 1004], "workers_cleaned_up": True, "elapsed_seconds": .2}
                raise error
            progress({"stage": "sweep", "sweep": 1, "objective": 100., "converged": False,
                      "subject_optimizer_failures": []})
            fit = self.prior.fixture.fixture.fit("MZ", train, held, converged=not nonconverged)
            fit["objective_history"] = [100.] * (1501 if over_ceiling else 1001)
            if deadline_stage == "fit":
                now[0] += 3601
            return fit

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual((spec, rival), (self.spec, "fixed"))
            likelihoods.append(len(data["gain"]))
            if not fits:
                self.assertEqual(len(data["gain"]), 128)
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            self.assertTrue(np.all((data["gain"] >= 10) & (data["gain"] <= 20)))
            if score_error:
                return np.zeros(1)
            if deadline_stage == "scoring":
                now[0] += 3601
            return np.full(len(data["gain"]), -2.94)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if checkpoint_delay and "retained_support" in report and not report["fresh_fit_calls"] and not delayed[0]:
                now[0] += checkpoint_delay
                delayed[0] = True
            if deadline_stage == "finalization" and report["common_gain_component_complete"]:
                now[0] += 3601

        def alarm(seconds):
            result = original_alarm(seconds)
            if seconds == 0 and deadline_stage == "cleanup":
                now[0] += 3601
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**{key: "1" for key in repair.parallel.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(repair.common.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(repair.common, "load_split", side_effect=split))
            stack.enter_context(patch.object(repair.parallel, "fit_partially_pooled_parallel", side_effect=fitter))
            stack.enter_context(patch.object(repair.common.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(repair, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(repair.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(patch.object(repair.signal, "alarm", side_effect=alarm))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = repair.run(self.args(output))
            self.assertEqual(os.environ["SLURM_CPUS_PER_TASK"], "4")
        self.assertEqual(self.prior.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual((source, truth, failed, original_failed, self.spec, self.people), before)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        for record in checkpoints:
            self.assert_scope(record)
        return report, fits, likelihoods, checkpoints, reads

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["direction"], report["fitted_model"]), (repair.PHASE, 1, "MZ"))
        self.assertEqual(report["fitted_models"], ["MZ"])
        self.assertEqual((report["workers"], report["allocated_cpus"], report["parallel_execution"]), (4, 4, True))
        self.assertEqual(report["numerical_override"], repair.numerical_override())
        self.assertEqual((report["original_maximum_sweeps"], report["fitted_maximum_sweeps"]), (500, 1500))
        for field in repair.common.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_common_gain",
                      "observed_choices_or_RTs_used", "wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        self.assertNotIn("cumulative_process_elapsed_seconds", report)
        self.assertNotIn("nested_diagnostic", report)

    def test_exact_context_rejects_wrong_identity_support_and_numerical_cause(self):
        context = self.context()
        self.assertEqual(context["previous_process_elapsed_seconds"], repair.FAILED_PROCESS_SECONDS)
        self.assertEqual(context["previous_support_contacts"], self.failed["fits"][0]["support_contacts"])
        self.assertEqual(context["retained_support"], self.failed["retained_support"])
        for change in (lambda f: f.update(job_id="other"), lambda f: f.update(finished_utc="wrong"),
                       lambda f: f.update(error="RuntimeError: timeout"),
                       lambda f: f.update(source_finished_utc="different"),
                       lambda f: f["retained_support"]["p000"]["heldout_slots"].reverse(),
                       lambda f: f["fits"][0].update(subject_optimizer_failures=[{}]),
                       lambda f: f["fits"][0].update(objective_history=[100.] * 500),
                       lambda f: f["fits"][0]["support_contacts"].pop()):
            changed = deepcopy(self.failed)
            change(changed)
            with self.assertRaises((ValueError, RuntimeError)):
                self.context(changed)

    def test_one_fresh_four_worker_fit_has_original_replay_scoring_and_cleanup_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fits, calls, checkpoints, reads = self.exercise(Path(directory) / "fresh", checkpoint_delay=7.)
        self.assertTrue(report["common_gain_component_complete"])
        self.assertEqual(len(fits), 1)
        self.assertEqual((report["fresh_fit_calls"], len(calls)), (1, 540))
        self.assertEqual(reads, [repair.FAILED_REPORT, repair.previous.failed_path(1)])
        self.assertEqual(report["original_implementation"], self.spec)
        self.assertEqual(report["implementation"], self.spec)
        self.assertEqual(report["runtime_implementation"], repair.runtime_specification(self.spec))
        self.assertEqual(report["prior_MZ_failed_context"], self.context())
        self.assertEqual(report["original_M0_failed_context"], self.failed["prior_failed_context"])
        self.assertIs(report["parallel_workers_cleaned_up"], True)
        self.assertIsNone(report["parallel_failure"])
        self.assertEqual(len(report["parallel_completed_sweeps"]), 1)
        self.assertTrue(any(record["parallel_completed_sweeps"] and not record["fits"] for record in checkpoints))
        self.assertNotEqual(report["fits"][0]["parameters"][0][0], self.failed["fits"][0]["parameters"][0][0])
        for row in report["participant_scores"].values():
            self.assertEqual(set(row), {"version", "heldout_trials", "MZ_log_sum"})
            self.assertAlmostEqual(row["MZ_log_sum"], -2.94 * row["heldout_trials"], places=10)

    def test_bad_source_replay_or_failure_context_never_calls_new_parallel_fit(self):
        for option in ({"bad_source": True}, {"replay_error": True},
                       {"failed_change": lambda f: f.update(error="wrong cause")}):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "fresh", **option)
            self.assertEqual(fits, [])
            self.assertEqual(report["fresh_fit_calls"], 0)
            self.assertFalse(report["common_gain_component_complete"])
            self.assertNotIn("participant_scores", report)

    def test_parallel_failure_retains_partial_progress_and_confirmed_cleanup_without_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fits, *_ = self.exercise(Path(directory) / "fresh", parallel_error=True)
        self.assertEqual(len(fits), 1)
        self.assertEqual(report["fits"], [])
        self.assertEqual(report["parallel_progress"]["participant"], "p000")
        self.assertEqual(list(report["parallel_failure"]["completed_results"]), ["p000"])
        self.assertIs(report["parallel_workers_cleaned_up"], True)
        self.assertEqual(report["completed_model_scores"], {})
        self.assertNotIn("participant_scores", report)
        self.assertEqual(report["status"], "technical_common_gain_failure")

    def test_returned_nonconvergence_ceiling_or_score_error_preserves_fit_without_score_claim(self):
        for option in ({"nonconverged": True}, {"over_ceiling": True}, {"score_error": True}):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "fresh", **option)
            self.assertEqual((len(fits), len(report["fits"])), (1, 1))
            self.assertIs(report["parallel_workers_cleaned_up"], True)
            self.assertFalse(report["common_gain_component_complete"])
            self.assertEqual(report["completed_model_scores"], {})
            self.assertNotIn("participant_scores", report)

    def test_whole_deadline_covers_source_lineage_fit_score_and_final_publication(self):
        for stage, count in (("source", 0), ("failed_read", 0), ("fit", 1), ("scoring", 1), ("finalization", 1), ("cleanup", 1)):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "fresh", deadline_stage=stage)
            self.assertEqual(len(fits), count)
            self.assertIn("whole-process deadline", report["error"])
            self.assertFalse(report["common_gain_component_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertEqual(list(report["completed_model_scores"]), ["MZ"] if stage in ("finalization", "cleanup") else [])

    def test_real_resource_fixed_input_fresh_output_guards_and_cli(self):
        env = {**{key: "1" for key in repair.parallel.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.dict(os.environ, env), patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for changes in ({"source_dir": Path("/other")}, {"implementation": Path("/other")},
                            {"max_seconds": 0}, {"max_seconds": 3601}, {"max_seconds": True}):
                with self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "fresh", **changes))
            for changes in ({"SLURM_CPUS_PER_TASK": "1"}, {"SLURM_CPUS_PER_TASK": "8"}, {"OMP_NUM_THREADS": "2"}):
                with patch.dict(os.environ, changes), self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "fresh"))
            for output in (repair.SOURCE / "new", repair.previous.FAILED_ROOT / "new", repair.FAILED_REPORT.parents[2] / "new", Path(directory)):
                with self.assertRaises(FileExistsError):
                    repair.run(self.args(output))
        with patch.object(repair, "run", return_value={"common_gain_component_complete": True}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(repair.main(["--source-dir", str(repair.SOURCE), "--implementation", str(repair.IMPLEMENTATION),
                "--output-dir", "/fresh", "--max-seconds", "3600"]), 0)
        self.assertEqual(run.call_args.args[0].max_seconds, 3600)
        self.assertFalse(hasattr(run.call_args.args[0], "direction"))


if __name__ == "__main__":
    unittest.main()
