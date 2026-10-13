"""Focused exact-failure/single-model numerical repair checks; compute only."""
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

import run_mz_common_gain_range_repair as repair
import test_mz_common_gain_range as fixtures


class CommonGainRangeRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CommonGainRangeTests.setUpClass()
        cls.fixture = fixtures.CommonGainRangeTests()
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        cls.failed = {}
        for direction in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = cls.fixture.exercise(Path(directory) / "original", direction)
            identity = repair.IDENTITIES[direction]
            fit = report["fits"][0]
            fit.update(converged=False, elapsed_seconds=identity["fit_elapsed_seconds"],
                objective_history=[identity["tail"][0] + 1] * 498 + identity["tail"],
                support_contacts=[{"level": "version_log_standard_deviation", "version": version,
                    "parameter": "kappa", "value": -3.9999998503354774, "bounds": [-4., 1.]} for version in ("EI", "ER")])
            fit["parameters"][0][0] = 1.5  # Failed state must not be used as a new start.
            for values in fit["version_standard_deviations"]:
                values[0] = float(np.exp(-3.9999998503354774))
            report.update(job_id=repair.FAILED_JOB, started_utc="2026-10-05T12:00:00+00:00",
                finished_utc=identity["finished_utc"],
                elapsed_seconds=identity["elapsed_seconds"], status="technical_common_gain_failure",
                error=repair.NONCONVERGENCE, common_gain_component_complete=False,
                fits=[fit], scored_fits=[], completed_model_scores={})
            report.pop("participant_scores")
            cls.failed[direction] = report

    def source(self):
        return self.fixture.source()

    def args(self, output, direction=0, model="M0", **changes):
        values = dict(direction=direction, model=model, source_dir=repair.SOURCE, implementation=repair.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=4200)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        core = repair.common
        return (core.MODELS, core.PHASE, core.REPORT_NAME, core.MAX_SECONDS, core.fit_partially_pooled,
                core.validate_fit, core.write_report, core.shared.validate_source, core.filtered_panels)

    def exercise(self, output, direction=0, model="M0", *, bad_source=False, failed_change=None,
                 replay_error=False, nonconverged=False, over_ceiling=False, score_error=False, deadline_stage=None):
        source, truth = self.source()
        failed = deepcopy(self.failed[direction])
        if bad_source:
            source["replicate_index"] = 1
        if failed_change:
            failed_change(failed)
        before, bindings = deepcopy((source, truth, failed, self.spec, self.people)), self.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        read_text, writer, now = Path.read_text, repair.common.write_report, [100.]
        fits, likelihoods, checkpoints, reads = [], [], [], []
        train, held = repair.common.selected_runs(direction)

        def reader(path, *args, **kwargs):
            if path == repair.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == repair.SOURCE / repair.common.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 4201
                return json.dumps(source)
            if path == repair.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            if path == repair.failed_path(direction):
                reads.append(path)
                if deadline_stage == "failed_read":
                    now[0] += 4201
                return json.dumps(failed)
            if path == repair.failed_path(1 - direction):
                raise AssertionError("Do not read the other original direction")
            return read_text(path, *args, **kwargs)

        def split(path, a, b):
            self.assertEqual((tuple(a), tuple(b)) in repair.common.HALVES, True)
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(a), panel(b)

        def fitter(training, selected, rival, runtime):
            self.assertEqual((selected, rival), (model, "fixed"))
            self.assertEqual(runtime, repair.runtime_specification(self.spec, model))
            self.assertEqual(len(likelihoods), 432)
            self.assertEqual(set(training), set(self.people))
            for person, record in training.items():
                expected = [row for row in self.people[person] if int(row["run"]) in train and 10 <= float(row["gain"]) <= 20]
                self.assertEqual(record["rows"], expected)
            fits.append((selected, deepcopy(runtime)))
            fit = self.fixture.fixture.fit(model, train, held, converged=not nonconverged)
            sweeps = runtime["optimizer"]["maximum_sweeps"] + 1 if over_ceiling else (501 if model == "M0" else 7)
            fit["objective_history"] = [100.] * sweeps
            if deadline_stage == "fit":
                now[0] += 4201
            return fit

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual((spec, rival), (self.spec, "fixed"))
            likelihoods.append(len(data["gain"]))
            if not fits:
                self.assertEqual(len(data["gain"]), 128)
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            if score_error:
                return np.zeros(1)
            if deadline_stage == "scoring":
                now[0] += 4201
            return np.full(len(data["gain"]), -2.95 if model == "MZ" else -3.)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["common_gain_component_complete"]:
                now[0] += 4201

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(repair.common, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(repair.common.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(repair.common, "load_split", side_effect=split))
            stack.enter_context(patch.object(repair.common, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(repair.common.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(repair.common, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(repair.common.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = repair.run(self.args(output, direction, model))
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual((source, truth, failed, self.spec, self.people), before)
        self.assertEqual(json.loads((output / repair.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        for record in checkpoints:
            self.assert_scope(record, direction, model)
        return report, fits, likelihoods, checkpoints, reads

    def assert_scope(self, report, direction, model):
        self.assertEqual((report["phase"], report["direction"], report["fitted_model"]), (repair.PHASE, direction, model))
        self.assertEqual(report["fitted_models"], [model])
        self.assertEqual(report["numerical_override"], repair.numerical_override(model))
        self.assertEqual(report["fitted_maximum_sweeps"], 1500 if model == "M0" else 500)
        self.assertEqual(report["execution_kind"], "M0_numerical_repair" if model == "M0" else "previously_unattempted_MZ_component")
        for field in repair.common.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_common_gain",
                      "observed_choices_or_RTs_used", "wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        self.assertNotIn("cumulative_process_elapsed_seconds", report)
        self.assertNotIn("nested_diagnostic", report)

    def test_exact_original_failure_contexts_preserve_both_contacts_and_reject_other_causes(self):
        source, _ = self.source()
        for direction in (0, 1):
            context = repair.repair_context(self.failed[direction], source, self.people, self.spec, direction)
            self.assertEqual(context["previous_support_contacts"], self.failed[direction]["fits"][0]["support_contacts"])
            self.assertGreater(context["final_objective_change"], context["unchanged_convergence_threshold"])
            for change in (lambda f: f.update(job_id="46613140"), lambda f: f.update(finished_utc="wrong"),
                           lambda f: f.update(error="RuntimeError: timeout"),
                           lambda f: f["fits"][0].update(subject_optimizer_failures=[{"participant": "p000"}]),
                           lambda f: f["fits"][0]["support_contacts"].pop(),
                           lambda f: f["fits"][0].update(objective_history=[100.] * 500),
                           lambda f: f.update(source_finished_utc="different")):
                changed = deepcopy(self.failed[direction])
                change(changed)
                with self.assertRaises((ValueError, RuntimeError)):
                    repair.repair_context(changed, source, self.people, self.spec, direction)

    def test_four_components_use_exactly_one_fresh_model_at_only_its_qualified_ceiling(self):
        for direction in (0, 1):
            for model in ("M0", "MZ"):
                with self.subTest(direction=direction, model=model), tempfile.TemporaryDirectory() as directory:
                    report, fits, calls, records, reads = self.exercise(Path(directory) / "fresh", direction, model)
                self.assertTrue(report["common_gain_component_complete"])
                self.assertEqual(len(fits), 1)
                self.assertEqual(report["fresh_fit_calls"], 1)
                self.assertEqual(len(calls), 540)
                self.assertEqual(reads, [repair.failed_path(direction)])
                self.assertEqual(report["implementation"], self.spec)
                self.assertEqual(report["original_implementation"], self.spec)
                self.assertEqual(report["runtime_implementation"], repair.runtime_specification(self.spec, model))
                self.assertEqual(list(report["completed_model_scores"]), [model])
                self.assertNotEqual(report["fits"][0]["parameters"][0][0], self.failed[direction]["fits"][0]["parameters"][0][0])
                self.assertEqual(report["retained_support"], report["prior_failed_context"]["retained_support"])
                self.assertEqual(report["prior_direction_process_elapsed_seconds"], repair.IDENTITIES[direction]["elapsed_seconds"])
                for row in report["participant_scores"].values():
                    self.assertEqual(set(row), {"version", "heldout_trials", model + "_log_sum"})
                    self.assertAlmostEqual(row[model + "_log_sum"], row["heldout_trials"] * (-3. if model == "M0" else -2.95), places=10)

    def test_invalid_source_failure_context_or_replay_prevents_any_new_fit(self):
        for option in ({"bad_source": True}, {"failed_change": lambda f: f.update(error="other")}, {"replay_error": True}):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "fresh", **option)
            self.assertEqual(report["status"], "technical_common_gain_failure")
            self.assertEqual(fits, [])
            self.assertEqual(report["fresh_fit_calls"], 0)
            self.assertNotIn("participant_scores", report)

    def test_actual_filter_must_match_old_legal_slot_panel_before_fitting(self):
        def change(failed):
            support = failed["retained_support"]["p000"]
            existing = {tuple(slot) for slot in support["heldout_slots"]}
            support["heldout_slots"][0] = next([r, t] for r in (2, 3) for t in range(1, 65) if (r, t) not in existing)
        with tempfile.TemporaryDirectory() as directory:
            report, fits, *_ = self.exercise(Path(directory) / "fresh", failed_change=change)
        self.assertEqual(fits, [])
        self.assertIn("retained slots/counts", report["error"])
        self.assertFalse(report["common_gain_component_complete"])

    def test_new_nonconvergence_wrong_ceiling_or_score_failure_remains_technical_and_retained(self):
        for model in ("M0", "MZ"):
            for option in ({"nonconverged": True}, {"over_ceiling": True}, {"score_error": True}):
                with tempfile.TemporaryDirectory() as directory:
                    report, fits, *_ = self.exercise(Path(directory) / "fresh", model=model, **option)
                self.assertEqual(len(fits), 1)
                self.assertEqual(len(report["fits"]), 1)
                self.assertEqual(report["status"], "technical_common_gain_failure")
                self.assertFalse(report["common_gain_component_complete"])
                self.assertEqual(report["completed_model_scores"], {})
                self.assertNotIn("participant_scores", report)

    def test_whole_cap_includes_failed_read_fit_score_and_finalization_and_restores_bindings(self):
        for stage, count in (("source", 0), ("failed_read", 0), ("fit", 1), ("scoring", 1), ("finalization", 1)):
            with tempfile.TemporaryDirectory() as directory:
                report, fits, *_ = self.exercise(Path(directory) / "fresh", deadline_stage=stage)
            self.assertEqual(len(fits), count)
            self.assertIn("whole-process deadline", report["error"])
            self.assertFalse(report["common_gain_component_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertEqual(list(report["completed_model_scores"]), ["M0"] if stage == "finalization" else [])

    def test_model_direction_cap_and_output_guards_and_cli(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for changes in ({"model": "MVC"}, {"direction": True}, {"direction": 2},
                            {"max_seconds": 0}, {"max_seconds": 4201}, {"max_seconds": True}):
                with self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "fresh", **changes))
            with self.assertRaises(FileExistsError):
                repair.run(self.args(repair.FAILED_ROOT / "new"))
        with patch.object(repair, "run", return_value={"common_gain_component_complete": True}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(repair.main(["--direction", "1", "--model", "MZ", "--source-dir", str(repair.SOURCE),
                "--implementation", str(repair.IMPLEMENTATION), "--output-dir", "/fresh", "--max-seconds", "4200"]), 0)
        self.assertEqual((run.call_args.args[0].direction, run.call_args.args[0].model), (1, "MZ"))


if __name__ == "__main__":
    unittest.main()
