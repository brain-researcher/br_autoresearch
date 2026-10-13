"""Focused exact-failure nonlinear M0 repair checks; task-owned compute only."""
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

import run_mz_nonlinear_m0_repair as repair
import test_mz_nonlinear as fixtures


class NonlinearM0RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NonlinearTests.setUpClass()
        cls.fixture = fixtures.NonlinearTests("test_each_model_replays_fixed_primary_then_fits_only_its_two_nonlinear_halves")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        with tempfile.TemporaryDirectory() as directory:
            failed, *_ = cls.fixture.exercise(Path(directory) / "original", "M0")
        failed.update(job_id=repair.FAILED_JOB, started_utc="2026-10-05T10:29:25.000000+00:00",
            finished_utc=repair.FAILED_FINISHED_UTC, status="technical_nonlinear_failure",
            nonlinear_component_complete=False, error=repair.NONCONVERGENCE,
            elapsed_seconds=2460.9616029759636, scored_primary_directions=[])
        failed.pop("participant_scores")
        failed["fits"] = failed["fits"][:1]
        fit = failed["fits"][0]
        fit.update(converged=False, elapsed_seconds=2459.7199579349253,
            objective_history=[8000. - .1 * index for index in range(498)] + [7871.745268005732, 7871.742835853569])
        fit["parameters"][0][0] = 1.5  # Distinct failed state must never seed fresh fits.
        fit["support_contacts"] = []
        for column, name in enumerate(fit["parameter_names"]):
            bounds = cls.spec["parameters"][name]["individual_log_scale_bounds"]
            fit["version_standard_deviations"][0][column] = float(np.exp(bounds[0]))
            fit["support_contacts"].append({"level": "version_log_standard_deviation", "version": "EI",
                "parameter": name, "value": bounds[0], "bounds": list(bounds)})
        cls.failed = failed

    def args(self, output, **changes):
        values = dict(source_dir=repair.SOURCE, implementation=repair.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=15600)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        core = repair.nonlinear
        return (core.MAX_SECONDS, core.REPORT_NAME, core.shared.validate_source,
                core.fit_partially_pooled, core.validate_nonlinear_fit, core.write_report)

    def exercise(self, output, *, bad_source=False, failed_change=None, replay_error=False,
                 fail_fit=None, deadline_stage=None):
        core = repair.nonlinear
        source, truth = self.fixture.source()
        failed = deepcopy(self.failed)
        if bad_source:
            source["replicate_index"] = 1
        if failed_change:
            failed_change(failed)
        before = deepcopy((source, truth, failed, self.spec, self.people))
        bindings = self.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        read_text, writer, now = Path.read_text, core.write_report, [100.]
        fits, likelihoods, checkpoints = [], [], []

        def reader(path, *args, **kwargs):
            if path == repair.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == repair.SOURCE / core.SOURCE_REPORT:
                return json.dumps(source)
            if path == repair.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            if path == repair.FAILED_REPORT:
                if deadline_stage == "failure_read":
                    now[0] += 15601
                return json.dumps(failed)
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, repair.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(train), panel(held)

        def fitter(training, model, rival, runtime):
            self.assertEqual((model, rival), ("M0", "nonlinear"))
            self.assertEqual(runtime, repair.runtime_specification(self.spec))
            train, held = core.HALVES[len(fits)]
            self.assertEqual(set(training), set(self.people))
            self.assertTrue(all(len(record["rows"]) == 128 and {int(row["run"]) for row in record["rows"]} == set(train)
                                for record in training.values()))
            self.assertEqual(len(likelihoods), 432 + 108 * len(fits))
            fits.append((model, rival, train, deepcopy(runtime)))
            fit = self.fixture.fit(model, train, held, converged=len(fits) != fail_fit)
            fit["objective_history"] = [7000.] * 1001 if fit["converged"] else [8000. - .1 * i for i in range(1500)]
            self.assertNotEqual(fit["parameters"][0], failed["fits"][0]["parameters"][0])
            if deadline_stage == "fit" and len(fits) == 1:
                now[0] += 15601
            return fit

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual(spec, self.spec)  # Likelihood receives the original unchanged scientific specification.
            self.assertEqual(len(data["gain"]), 128)
            likelihoods.append(rival)
            if rival == "fixed":
                self.assertEqual(fits, [])
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            self.assertEqual(rival, "nonlinear")
            self.assertEqual(names, core.parameter_names("M0", "nonlinear"))
            return np.full(128, -2.7)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["nonlinear_component_complete"]:
                now[0] += 15601

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(core, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(core.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(core, "load_split", side_effect=split))
            stack.enter_context(patch.object(core, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(core.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(core, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(core.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = repair.run(self.args(output))
        self.assertEqual((source, truth, failed, self.spec, self.people), before)
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assert_scope(record)
        return report, fits, likelihoods, checkpoints

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["fitted_model"], report["rival"], report["internal_seconds"]),
                         (repair.PHASE, "M0", "nonlinear", 15600))
        self.assertEqual(report["numerical_override"], repair.numerical_override())
        self.assertEqual(report["failed_report"], str(repair.FAILED_REPORT))
        for field in repair.nonlinear.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_nonlinear",
                      "observed_choices_or_RTs_used", "wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        if report.get("implementation") is not None:
            self.assertEqual(report["implementation"], self.spec)
            self.assertEqual(report["original_implementation"], self.spec)
            self.assertEqual(report["runtime_implementation"], repair.runtime_specification(self.spec))
        if report["repair_context_validated"]:
            self.assertEqual(report["numerical_repair"]["previous_support_contacts"], self.failed["fits"][0]["support_contacts"])
            if "elapsed_seconds" in report:
                self.assertEqual(report["cumulative_process_elapsed_seconds"], self.failed["elapsed_seconds"] + report["elapsed_seconds"])

    def test_exact_failure_context_original500_and_only_runtime_sweep_ceiling_changes(self):
        source, truth = self.fixture.source()
        context = repair.repair_context(self.failed, source, self.people, self.spec)
        self.assertEqual(context["previous_support_contacts"], self.failed["fits"][0]["support_contacts"])
        self.assertEqual(context["previous_process_elapsed_seconds"], 2460.9616029759636)
        self.assertGreater(context["final_objective_change"], context["unchanged_convergence_threshold"])
        runtime = repair.runtime_specification(self.spec)
        self.assertEqual(runtime["optimizer"]["maximum_sweeps"], 1500)
        runtime["optimizer"]["maximum_sweeps"] = 500
        self.assertEqual(runtime, self.spec)
        changes = (lambda report: report.update(job_id="46606083"),
                   lambda report: report.update(finished_utc="2026-10-05T11:10:26.656204+00:00"),
                   lambda report: report.update(source_finished_utc="wrong"),
                   lambda report: report.update(error="RuntimeError: interrupted by signal15"),
                   lambda report: report["fits"][0].update(subject_optimizer_failures=[{"participant": "p000"}]),
                   lambda report: report["fits"][0].update(objective_history=[7000.] * 500),
                   lambda report: report["fits"][0].update(participants=report["fits"][0]["participants"][:-1]),
                   lambda report: report["fits"][0]["support_contacts"].pop(),
                   lambda report: report["saved_primary_participant_scores"]["p000"].update(M0=-2.))
        for number, change in enumerate(changes):
            failed = deepcopy(self.failed)
            change(failed)
            with self.subTest(change=number), self.assertRaises(ValueError):
                repair.repair_context(failed, source, self.people, self.spec)

    def test_exactly_two_fresh_runtime1500_fits_follow_original_source_replay_with_cost_once(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fits, likelihoods, checkpoints = self.exercise(Path(directory) / "new")
        self.assertTrue(report["nonlinear_component_complete"])
        self.assertTrue(report["repair_context_validated"])
        self.assertEqual(report["fresh_fit_calls"], 2)
        self.assertEqual([train for model, rival, train, runtime in fits], [train for train, held in repair.nonlinear.HALVES])
        self.assertTrue(all(len(fit["objective_history"]) == 1001 for fit in report["fits"]))
        self.assertEqual(likelihoods, ["fixed"] * 432 + ["nonlinear"] * 216)
        source, _ = self.fixture.source()
        self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
        self.assertEqual(report["cumulative_process_elapsed_seconds"], self.failed["elapsed_seconds"] + report["elapsed_seconds"])
        self.assertNotEqual(report["fits"][0]["support_contacts"], report["numerical_repair"]["previous_support_contacts"])

    def test_source_failure_wrong_prior_or_replay_failure_never_starts_a_repair_fit(self):
        for options in ({"bad_source": True}, {"failed_change": lambda report: report.update(fitted_model="MZ")},
                        {"replay_error": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                report, fits, likelihoods, checkpoints = self.exercise(Path(directory) / "new", **options)
            self.assertFalse(report["nonlinear_component_complete"])
            self.assertEqual(fits, [])
            self.assertNotIn("participant_scores", report)

    def test_first_or_second_nonconvergence_retains_technical_prefix_without_scores(self):
        for failed_at in (1, 2):
            with self.subTest(failed_at=failed_at), tempfile.TemporaryDirectory() as directory:
                report, fits, likelihoods, checkpoints = self.exercise(Path(directory) / "new", fail_fit=failed_at)
            self.assertEqual(report["status"], "technical_nonlinear_failure")
            self.assertEqual(len(report["fits"]), failed_at)
            self.assertEqual(len(report["scored_primary_directions"]), failed_at - 1)
            self.assertNotIn("participant_scores", report)
            self.assert_scope(report)

    def test_whole_cap_covers_failed_report_read_new_fit_and_finalization_and_restores_all_bindings(self):
        for stage, expected in (("failure_read", 0), ("fit", 1), ("finalization", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, fits, likelihoods, checkpoints = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertEqual(report["status"], "technical_nonlinear_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(len(fits), expected)
            self.assertFalse(report["nonlinear_component_complete"])
            self.assertNotIn("participant_scores", report)

    def test_m0_only_cap_and_preserved_failed_output_guards_precede_source_reads(self):
        bindings = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.object(repair.nonlinear, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for change in ({"model": "MZ"}, {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": 15601},
                           {"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "new", **change))
            for output in (Path(directory), repair.SOURCE / "new", repair.FAILED_REPORT.parent / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    repair.run(self.args(output))
        self.assertEqual(self.bindings(), bindings)

    def test_cli_has_no_model_or_initialization_choice_and_uses_explicit_finite_cap(self):
        report = {"nonlinear_component_complete": True}
        with patch.object(repair, "run", return_value=report) as runner, redirect_stdout(io.StringIO()):
            self.assertEqual(repair.main(["--source-dir", str(repair.SOURCE), "--implementation", str(repair.IMPLEMENTATION),
                "--output-dir", "/tmp/nonlinear-m0-repair-fresh", "--max-seconds", "15600"]), 0)
        self.assertEqual(vars(runner.call_args.args[0]), vars(self.args("/tmp/nonlinear-m0-repair-fresh")))


if __name__ == "__main__":
    unittest.main()
