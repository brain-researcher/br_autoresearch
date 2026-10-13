"""Focused selected nonlinear M0 direction checks; task-owned compute only."""
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

import run_mz_nonlinear_m0_direction_repair as repair
import test_mz_nonlinear_m0_repair as fixtures


class NonlinearM0DirectionRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NonlinearM0RepairTests.setUpClass()
        cls.prior = fixtures.NonlinearM0RepairTests()
        cls.spec, cls.people = cls.prior.spec, cls.prior.people
        cls.original_failed = cls.prior.failed
        with tempfile.TemporaryDirectory() as directory:
            failed, *_ = cls.prior.exercise(Path(directory) / "old1500", fail_fit=1)
        failed.update(job_id=repair.FAILED_JOB, started_utc="2026-10-05T11:35:37.630267+00:00",
            finished_utc=repair.FAILED_FINISHED, elapsed_seconds=repair.FAILED_PROCESS_SECONDS,
            cumulative_process_elapsed_seconds=cls.original_failed["elapsed_seconds"] + repair.FAILED_PROCESS_SECONDS)
        fit = failed["fits"][0]
        fit.update(elapsed_seconds=repair.FAILED_FIT_SECONDS,
            objective_history=[repair.FAILED_TAIL[0] + 1] * 1497 + repair.FAILED_TAIL,
            support_contacts=[])
        fit["parameters"][0][0] = 1.6
        for version, name in (("EI", "gamma_gain"), ("EI", "gamma_loss"),
                              ("ER", "kappa"), ("ER", "gamma_gain"), ("ER", "gamma_loss")):
            value = -3.9999998503354774 if name == "kappa" else -3.999999830370791
            fit["support_contacts"].append({"level": "version_log_standard_deviation", "version": version,
                "parameter": name, "value": value, "bounds": cls.spec["parameters"][name]["individual_log_scale_bounds"][:]})
            fit["version_standard_deviations"][0 if version == "EI" else 1][fit["parameter_names"].index(name)] = float(np.exp(value))
        cls.failed1500 = cls.failed = failed

    def source(self):
        return self.prior.fixture.source()

    def args(self, output, direction=0, **changes):
        values = dict(direction=direction, source_dir=repair.SOURCE, implementation=repair.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=4800)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return self.prior.bindings() + (repair.nonlinear.PHASE, repair.nonlinear.HALVES, repair.nonlinear.shared.HALVES)

    def context(self, failed=None, original_failed=None):
        source, _ = self.source()
        original = self.original_failed if original_failed is None else original_failed
        with patch.object(Path, "read_text", return_value=json.dumps(original)):
            return repair.repair_context(self.failed1500 if failed is None else failed,
                                         source, source["participant_scores"], self.spec)

    def exercise(self, output, direction=0, *, bad_source=False, failed_change=None, original_change=None,
                 replay_error=False, nonconverged=False, over_ceiling=False, score_error=False,
                 parallel_error=False, deadline_stage=None, checkpoint_delay=0.):
        source, truth = self.source()
        failed, original_failed = deepcopy(self.failed1500), deepcopy(self.original_failed)
        if bad_source:
            source["replicate_index"] = 1
        if failed_change:
            failed_change(failed)
        if original_change:
            original_change(original_failed)
        before = deepcopy((source, truth, failed, original_failed, self.spec, self.people))
        bindings = self.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        read_text, writer, alarm_original = Path.read_text, repair.write_report, signal.alarm
        now, delayed = [100.], [False]
        fits, likelihoods, checkpoints, reads = [], [], [], []
        train, held = repair.selected_runs(direction)

        def reader(path, *args, **kwargs):
            if path == repair.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == repair.SOURCE / repair.nonlinear.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 4801
                return json.dumps(source)
            if path == repair.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            if path == repair.FAILED_REPORT:
                reads.append(path)
                if deadline_stage == "failed_read":
                    now[0] += 4801
                return json.dumps(failed)
            if path == repair.previous.FAILED_REPORT:
                reads.append(path)
                return json.dumps(original_failed)
            if repair.previous.FAILED_REPORT.parents[1] in path.parents:
                raise AssertionError("Never read or refit the sibling nonlinear MZ")
            return read_text(path, *args, **kwargs)

        def split(path, a, b):
            self.assertEqual(path, repair.SOURCE / "simulated_events.csv.gz")
            self.assertIn((tuple(a), tuple(b)), repair.nonlinear.HALVES)
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(a), panel(b)

        def fitter(training, model, rival, runtime, *, workers, max_seconds, progress):
            self.assertEqual((model, rival, workers, os.environ["SLURM_CPUS_PER_TASK"]), ("M0", "nonlinear", 4, "4"))
            self.assertEqual(runtime, repair.runtime_specification(self.spec, direction))
            self.assertEqual(max_seconds, 4800 - repair.parallel.CLEANUP_SECONDS - 1 - checkpoint_delay)
            self.assertEqual(likelihoods, ["fixed"] * 432)
            self.assertEqual(set(training), set(self.people))
            for person, record in training.items():
                expected = [row for row in self.people[person] if int(row["run"]) in train]
                self.assertEqual(record["rows"], expected)
                self.assertEqual(len(record["rows"]), 128)
                self.assertTrue(all(left is right for left, right in zip(record["rows"], expected)))
                self.assertTrue(all(int(row["run"]) not in held for row in record["rows"]))
            fits.append((model, rival, train, deepcopy(runtime), workers, max_seconds))
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
            fit = self.prior.fixture.fit("M0", train, held, converged=not nonconverged)
            sweeps = repair.SWEEPS[direction] + 1 if over_ceiling else (1501 if direction == 0 else 1001)
            fit["objective_history"] = [7000.] * sweeps
            if deadline_stage == "fit":
                now[0] += 4801
            return fit

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual(spec, self.spec)
            self.assertEqual(len(data["gain"]), 128)
            likelihoods.append(rival)
            if rival == "fixed":
                self.assertEqual(fits, [])
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            self.assertEqual(rival, "nonlinear")
            self.assertEqual(names, repair.nonlinear.parameter_names("M0", "nonlinear"))
            if score_error:
                return np.zeros(127)
            if deadline_stage == "scoring":
                now[0] += 4801
            return np.full(128, -2.7 + .02 * direction)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if checkpoint_delay and report["repair_context_validated"] and not report["fresh_fit_calls"] and not delayed[0]:
                now[0] += checkpoint_delay
                delayed[0] = True
            if deadline_stage == "finalization" and report["nonlinear_direction_complete"]:
                now[0] += 4801

        def alarm(seconds):
            result = alarm_original(seconds)
            if seconds == 0 and deadline_stage == "cleanup":
                now[0] += 4801
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**{key: "1" for key in repair.parallel.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(repair.nonlinear.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(repair.nonlinear, "load_split", side_effect=split))
            stack.enter_context(patch.object(repair.parallel, "fit_partially_pooled_parallel", side_effect=fitter))
            stack.enter_context(patch.object(repair.nonlinear.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(repair, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(repair.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(patch.object(repair.signal, "alarm", side_effect=alarm))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = repair.run(self.args(output, direction))
            self.assertEqual(os.environ["SLURM_CPUS_PER_TASK"], "4")
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual((source, truth, failed, original_failed, self.spec, self.people), before)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        for record in checkpoints:
            self.assert_scope(record, direction)
        return report, fits, likelihoods, checkpoints, reads

    def assert_scope(self, report, direction):
        self.assertEqual((report["phase"], report["fitted_model"], report["rival"], report["direction"]),
                         (repair.PHASE, "M0", "nonlinear", direction))
        self.assertEqual(report["numerical_override"], repair.numerical_override(direction))
        self.assertEqual((report["original_maximum_sweeps"], report["fitted_maximum_sweeps"]), (500, repair.SWEEPS[direction]))
        self.assertEqual((report["workers"], report["allocated_cpus"], report["parallel_execution"]), (4, 4, True))
        self.assertIs(report["direction_contribution_only"], True)
        self.assertEqual(report["score_normalization"], repair.NORMALIZATION)
        self.assertNotIn("nonlinear_component_complete", report)
        self.assertNotIn("cumulative_process_elapsed_seconds", report)
        for field in repair.nonlinear.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_nonlinear",
                      "observed_choices_or_RTs_used", "wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])

    def test_exact1500_failure_chains_original500_and_rejects_other_identity_cause_or_cost(self):
        context = self.context()
        self.assertEqual(context["previous_process_elapsed_seconds"], repair.FAILED_PROCESS_SECONDS)
        self.assertEqual(context["previous_cumulative_process_elapsed_seconds"],
                         self.original_failed["elapsed_seconds"] + repair.FAILED_PROCESS_SECONDS)
        self.assertEqual(context["previous_support_contacts"], self.failed1500["fits"][0]["support_contacts"])
        self.assertGreater(context["final_objective_change"], context["unchanged_convergence_threshold"])
        for change in (lambda r: r.update(job_id="wrong"), lambda r: r.update(finished_utc="wrong"),
                       lambda r: r.update(error="RuntimeError: timeout"), lambda r: r.update(source_finished_utc="wrong"),
                       lambda r: r.update(cumulative_process_elapsed_seconds=1.),
                       lambda r: r["fits"][0].update(subject_optimizer_failures=[{}]),
                       lambda r: r["fits"][0].update(objective_history=[7000.] * 1500),
                       lambda r: r["fits"][0]["support_contacts"].pop()):
            failed = deepcopy(self.failed1500)
            change(failed)
            with self.assertRaises((ValueError, RuntimeError)):
                self.context(failed)
        original = deepcopy(self.original_failed)
        original["finished_utc"] = "wrong"
        with self.assertRaises(ValueError):
            self.context(original_failed=original)

    def test_each_direction_has_only_one_fresh_correct_ceiling_fit_and128_over256_contribution(self):
        for direction in (0, 1):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as directory:
                report, fits, calls, checkpoints, reads = self.exercise(Path(directory) / "new", direction, checkpoint_delay=7.)
            self.assertTrue(report["nonlinear_direction_complete"])
            self.assertEqual((len(fits), report["fresh_fit_calls"]), (1, 1))
            self.assertEqual(calls, ["fixed"] * 432 + ["nonlinear"] * 108)
            self.assertEqual(reads, [repair.FAILED_REPORT, repair.previous.FAILED_REPORT])
            runtime = deepcopy(report["runtime_implementation"])
            self.assertEqual(runtime["optimizer"]["maximum_sweeps"], repair.SWEEPS[direction])
            runtime["optimizer"]["maximum_sweeps"] = 500
            self.assertEqual(runtime, self.spec)
            self.assertEqual(report["prior_1500_failed_context"], self.context())
            self.assertEqual(report["original_500_failed_context"], self.failed1500["numerical_repair"])
            self.assertEqual(report["completed_direction_scores"], report["participant_scores"])
            self.assertEqual(len(report["scored_primary_directions"]), 1)
            self.assertEqual(report["scored_primary_directions"][0]["training_runs"], list(repair.selected_runs(direction)[0]))
            self.assertIs(report["parallel_workers_cleaned_up"], True)
            self.assertTrue(any(r["parallel_completed_sweeps"] and not r["fits"] for r in checkpoints))
            self.assertNotEqual(report["fits"][0]["parameters"][0], self.failed1500["fits"][0]["parameters"][0])
            for row in report["participant_scores"].values():
                self.assertAlmostEqual(row["nonlinear_score"], (-2.7 + .02 * direction) * 128 / 256)
                self.assertNotAlmostEqual(row["nonlinear_score"], -2.7 + .02 * direction)

    def test_bad_source_original_failure_current_failure_or_replay_prevents_new_fit(self):
        for options in ({"bad_source": True}, {"replay_error": True},
                        {"failed_change": lambda r: r.update(error="wrong")},
                        {"original_change": lambda r: r.update(job_id="wrong")}):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "new", **options)
            self.assertEqual(fits, [])
            self.assertEqual(report["fresh_fit_calls"], 0)
            self.assertFalse(report["nonlinear_direction_complete"])
            self.assertNotIn("participant_scores", report)

    def test_worker_failure_retains_partial_progress_and_cleanup_with_no_direction_score(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fits, *_ = self.exercise(Path(directory) / "new", 1, parallel_error=True)
        self.assertEqual(len(fits), 1)
        self.assertEqual(report["fits"], [])
        self.assertEqual(report["parallel_progress"]["participant"], "p000")
        self.assertEqual(list(report["parallel_failure"]["completed_results"]), ["p000"])
        self.assertIs(report["parallel_workers_cleaned_up"], True)
        self.assertEqual(report["completed_direction_scores"], {})
        self.assertFalse(report["nonlinear_direction_complete"])
        self.assertNotIn("participant_scores", report)

    def test_returned_nonconvergence_wrong_ceiling_and_non128_scores_remain_technical(self):
        for direction in (0, 1):
            for options in ({"nonconverged": True}, {"over_ceiling": True}, {"score_error": True}):
                with tempfile.TemporaryDirectory() as directory:
                    report, fits, *_ = self.exercise(Path(directory) / "new", direction, **options)
                self.assertEqual((len(fits), len(report["fits"])), (1, 1))
                self.assertFalse(report["nonlinear_direction_complete"])
                self.assertEqual(report["completed_direction_scores"], {})
                self.assertNotIn("participant_scores", report)

    def test_whole_cap_covers_source_lineage_fit_score_publication_and_handler_cleanup(self):
        for stage, count in (("source", 0), ("failed_read", 0), ("fit", 1), ("scoring", 1), ("finalization", 1), ("cleanup", 1)):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertEqual(len(fits), count)
            self.assertIn("whole-process deadline", report["error"])
            self.assertFalse(report["nonlinear_direction_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertEqual(len(report["completed_direction_scores"]), 108 if stage in ("finalization", "cleanup") else 0)

    def test_real_four_cpu_direction_cap_fresh_output_and_explicit_cli_guards(self):
        env = {**{key: "1" for key in repair.parallel.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.dict(os.environ, env), patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for changes in ({"source_dir": Path("/other")}, {"implementation": Path("/other")},
                            {"max_seconds": 0}, {"max_seconds": 4801}, {"max_seconds": True}):
                with self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "new", **changes))
            for direction in (-1, 2, True):
                with self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "new", direction))
            for changes in ({"SLURM_CPUS_PER_TASK": "1"}, {"SLURM_CPUS_PER_TASK": "8"}, {"BLIS_NUM_THREADS": "2"}):
                with patch.dict(os.environ, changes), self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "new"))
            for output in (repair.SOURCE / "new", repair.FAILED_REPORT.parent / "new", repair.previous.FAILED_REPORT.parent / "new", Path(directory)):
                with self.assertRaises(FileExistsError):
                    repair.run(self.args(output))
        with patch.object(repair, "run", return_value={"nonlinear_direction_complete": True}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(repair.main(["--direction", "1", "--source-dir", str(repair.SOURCE),
                "--implementation", str(repair.IMPLEMENTATION), "--output-dir", "/fresh", "--max-seconds", "4800"]), 0)
        self.assertEqual((run.call_args.args[0].direction, run.call_args.args[0].max_seconds), (1, 4800))
        self.assertFalse(hasattr(run.call_args.args[0], "model"))


if __name__ == "__main__":
    unittest.main()
