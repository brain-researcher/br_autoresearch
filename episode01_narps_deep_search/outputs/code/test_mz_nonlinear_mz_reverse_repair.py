"""Focused nonlinear MZ preserved-first/fresh-reverse repair checks; compute only."""
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

import run_mz_nonlinear_mz_reverse_repair as repair
import test_mz_nonlinear as fixtures


class NonlinearMZReverseRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NonlinearTests.setUpClass()
        cls.fixture = fixtures.NonlinearTests("test_each_model_replays_fixed_primary_then_fits_only_its_two_nonlinear_halves")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people
        with tempfile.TemporaryDirectory() as directory:
            failed, *_ = cls.fixture.exercise(Path(directory) / "original", "MZ")
        failed.update(job_id=repair.FAILED_JOB, started_utc="2026-10-05T10:29:25.000000+00:00",
            finished_utc=repair.FAILED_FINISHED_UTC, status="technical_nonlinear_failure",
            nonlinear_component_complete=False, error=repair.NONCONVERGENCE, elapsed_seconds=3434.9226916739717,
            scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
        failed.pop("participant_scores")
        failed["fits"][0].update(objective_history=[7600.] * 153, elapsed_seconds=895.3248969219858)
        failed["fits"][1].update(converged=False, elapsed_seconds=2538.2366071930155,
            objective_history=[8000. - .1 * i for i in range(498)] + [7703.8378774324, 7703.836821963588])
        failed["fits"][1]["parameters"][0][0] = 1.5
        for fit in failed["fits"]:
            fit["support_contacts"] = []
            for version, names in (("EI", ("rho", "gamma_gain", "gamma_loss")),
                                   ("ER", ("kappa", "rho", "gamma_gain", "gamma_loss"))):
                for name in names:
                    bounds = cls.spec["parameters"][name]["individual_log_scale_bounds"]
                    fit["version_standard_deviations"][0 if version == "EI" else 1][fit["parameter_names"].index(name)] = float(np.exp(bounds[0]))
                    fit["support_contacts"].append({"level": "version_log_standard_deviation", "version": version,
                        "parameter": name, "value": bounds[0], "bounds": list(bounds)})
        cls.failed = failed

    def args(self, output, **changes):
        values = dict(source_dir=repair.SOURCE, implementation=repair.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=7800)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        core = repair.nonlinear
        return (core.MAX_SECONDS, core.REPORT_NAME, core.shared.validate_source, core.fit_partially_pooled,
                core.validate_nonlinear_fit, core.score_nonlinear_fit, core.write_report)

    def exercise(self, output, *, bad_source=False, failed_change=None, replay_error=False,
                 rescore_error=False, nonconverged=False, deadline_stage=None):
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
        reader_original, writer, validator = Path.read_text, core.write_report, core.validate_nonlinear_fit
        now = [100.]
        fresh_calls, likelihoods, validations, checkpoints = [], [], [], []

        def reader(path, *args, **kwargs):
            if path == repair.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == repair.SOURCE / core.SOURCE_REPORT:
                return json.dumps(source)
            if path == repair.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            if path == repair.FAILED_REPORT:
                if deadline_stage == "failure_read":
                    now[0] += 7801
                return json.dumps(failed)
            return reader_original(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, repair.SOURCE / "simulated_events.csv.gz")
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(train), panel(held)

        def fitter(training, model, rival, runtime):
            self.assertEqual((model, rival), ("MZ", "nonlinear"))
            self.assertEqual(runtime, repair.runtime_specification(self.spec))
            self.assertEqual(set(training), set(self.people))
            self.assertTrue(all(len(record["rows"]) == 128 and {int(row["run"]) for row in record["rows"]} == {2, 3}
                                for record in training.values()))
            self.assertEqual(len(likelihoods), 540)  # Complete baseline replay plus preserved-first rescoring.
            fresh_calls.append(deepcopy(runtime))
            fit = self.fixture.fit("MZ", *core.HALVES[1], converged=not nonconverged)
            fit["objective_history"] = [7000.] * 1001 if not nonconverged else [8000. - .1 * i for i in range(1500)]
            self.assertNotEqual(fit["parameters"][0], failed["fits"][1]["parameters"][0])
            if deadline_stage == "fit":
                now[0] += 7801
            return fit

        def validate_fit(fit, model, train, held, spec, people):
            validations.append((tuple(train), spec["optimizer"]["maximum_sweeps"]))
            return validator(fit, model, train, held, spec, people)

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual(spec, self.spec)
            self.assertEqual(len(data["gain"]), 128)
            likelihoods.append(rival)
            if rival == "fixed":
                self.assertEqual(fresh_calls, [])
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            self.assertEqual(rival, "nonlinear")
            self.assertEqual(names, core.parameter_names("MZ", "nonlinear"))
            if rescore_error and not fresh_calls:
                raise RuntimeError("Preserved-first rescore interrupted")
            return np.full(128, -2.66)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["nonlinear_component_complete"]:
                now[0] += 7801

        with ExitStack() as stack:
            stack.enter_context(patch.object(repair, "require_compute"))
            stack.enter_context(patch.object(core, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(core.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(core, "load_split", side_effect=split))
            stack.enter_context(patch.object(core, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(core, "validate_nonlinear_fit", side_effect=validate_fit))
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
        return report, fresh_calls, likelihoods, validations, checkpoints

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["fitted_model"], report["rival"], report["internal_seconds"]),
                         (repair.PHASE, "MZ", "nonlinear", 7800))
        self.assertEqual(report["numerical_override"], repair.numerical_override())
        self.assertEqual(report["per_half_provenance"], repair.per_half_provenance())
        self.assertEqual([row["validation_maximum_sweeps"] for row in report["per_half_provenance"]], [500, 1500])
        for field in repair.nonlinear.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_nonlinear",
                      "observed_choices_or_RTs_used", "wave_selected", "retry_wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        if report.get("implementation") is not None:
            self.assertEqual(report["original_implementation"], self.spec)
            self.assertEqual(report["runtime_implementation"], repair.runtime_specification(self.spec))
        if report["repair_context_validated"]:
            self.assertEqual(report["numerical_repair"]["previous_support_contacts"], [fit["support_contacts"] for fit in self.failed["fits"]])
            self.assertEqual(report["numerical_repair"]["first_half_original_fit_elapsed_seconds"], 895.3248969219858)
            if "elapsed_seconds" in report:
                self.assertEqual(report["cumulative_process_elapsed_seconds"], self.failed["elapsed_seconds"] + report["elapsed_seconds"])
        if report["fits"]:
            self.assertEqual({key: value for key, value in report["fits"][0].items() if key != "elapsed_seconds"},
                             {key: value for key, value in self.failed["fits"][0].items() if key != "elapsed_seconds"})
            self.assertEqual(report["first_half_copy_elapsed_seconds"], report["fits"][0]["elapsed_seconds"])

    def test_context_validates_original_first_and_exact_reverse_exhaustion_without_mutation(self):
        source, _ = self.fixture.source()
        first, metadata = repair.repair_context(self.failed, source, self.people, self.spec)
        self.assertEqual(first, self.failed["fits"][0])
        self.assertIsNot(first, self.failed["fits"][0])
        self.assertGreater(metadata["final_reverse_objective_change"], metadata["unchanged_convergence_threshold"])
        self.assertEqual(metadata["previous_process_elapsed_seconds"], 3434.9226916739717)
        mutations = (lambda r: r.update(job_id="46606083"), lambda r: r.update(finished_utc="wrong"),
            lambda r: r.update(source_finished_utc="wrong"), lambda r: r.update(error="RuntimeError: signal15"),
            lambda r: r["fits"][0].update(converged=False), lambda r: r["fits"][0].update(objective_history=[7600.] * 501),
            lambda r: r["fits"][1].update(objective_history=[7700.] * 500),
            lambda r: r["fits"][1].update(subject_optimizer_failures=[{}]),
            lambda r: r["fits"][1]["participants"].pop(), lambda r: r["fits"][0]["support_contacts"].pop(),
            lambda r: r["saved_primary_participant_scores"]["p000"].update(M0=-2.))
        for number, mutation in enumerate(mutations):
            failed = deepcopy(self.failed)
            mutation(failed)
            with self.subTest(mutation=number), self.assertRaises((ValueError, RuntimeError)):
                repair.repair_context(failed, source, self.people, self.spec)

    def test_only_reverse_is_fresh_first_is_exactly_preserved_rescored_and_costed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fresh, likelihoods, validations, checkpoints = self.exercise(Path(directory) / "new")
        self.assertTrue(report["nonlinear_component_complete"])
        self.assertEqual((len(fresh), report["fresh_fit_calls"]), (1, 1))
        self.assertEqual(validations, [((1, 4), 500), ((2, 3), 1500)])
        self.assertEqual(likelihoods, ["fixed"] * 432 + ["nonlinear"] * 216)
        self.assertTrue(report["first_half_fit_reused"])
        self.assertTrue(report["first_half_rescored"])
        self.assertTrue(report["fresh_second_half_fit_called"])
        self.assertGreaterEqual(report["first_half_rescore_elapsed_seconds"], 0.)
        self.assertEqual(len(report["fits"][0]["objective_history"]), 153)
        self.assertEqual(len(report["fits"][1]["objective_history"]), 1001)
        source, _ = self.fixture.source()
        self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
        self.assertEqual(report["cumulative_process_elapsed_seconds"], self.failed["elapsed_seconds"] + report["elapsed_seconds"])

    def test_bad_source_prior_identity_or_replay_prevents_preserved_and_fresh_fit_dispatches(self):
        for options in ({"bad_source": True}, {"failed_change": lambda r: r.update(fitted_model="M0")}, {"replay_error": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                report, fresh, likelihoods, validations, checkpoints = self.exercise(Path(directory) / "new", **options)
            self.assertFalse(report["nonlinear_component_complete"])
            self.assertFalse(report["first_half_fit_reused"])
            self.assertEqual(fresh, [])
            self.assertNotIn("participant_scores", report)

    def test_preserved_first_rescore_failure_blocks_fresh_reverse_and_keeps_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fresh, likelihoods, validations, checkpoints = self.exercise(Path(directory) / "new", rescore_error=True)
        self.assertEqual(fresh, [])
        self.assertEqual(len(report["fits"]), 1)
        self.assertTrue(report["first_half_fit_reused"])
        self.assertFalse(report["first_half_rescored"])
        self.assertIsNone(report["first_half_rescore_elapsed_seconds"])
        self.assertEqual(report["scored_primary_directions"], [])
        self.assertNotIn("participant_scores", report)

    def test_reverse_nonconvergence_retains_two_fits_and_only_first_scored_as_technical(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fresh, likelihoods, validations, checkpoints = self.exercise(Path(directory) / "new", nonconverged=True)
        self.assertEqual(report["status"], "technical_nonlinear_failure")
        self.assertEqual((len(report["fits"]), len(report["scored_primary_directions"])), (2, 1))
        self.assertEqual(len(fresh), 1)
        self.assertNotIn("participant_scores", report)

    def test_whole_cap_covers_failure_read_reverse_fit_and_finalization_with_restoration(self):
        for stage, returned in (("failure_read", 0), ("fit", 2), ("finalization", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, fresh, likelihoods, validations, checkpoints = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertEqual(report["status"], "technical_nonlinear_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(len(report["fits"]), returned)
            self.assertFalse(report["nonlinear_component_complete"])
            self.assertNotIn("participant_scores", report)

    def test_mz_only_fixed_paths_cap_fresh_output_guards_and_cli(self):
        bindings = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(repair, "require_compute"), \
                patch.object(repair.nonlinear, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for change in ({"model": "M0"}, {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": 7801},
                           {"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    repair.run(self.args(Path(directory) / "new", **change))
            for output in (Path(directory), repair.SOURCE / "new", repair.FAILED_REPORT.parent / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    repair.run(self.args(output))
        self.assertEqual(self.bindings(), bindings)
        with patch.object(repair, "run", return_value={"nonlinear_component_complete": True}) as runner, redirect_stdout(io.StringIO()):
            self.assertEqual(repair.main(["--source-dir", str(repair.SOURCE), "--implementation", str(repair.IMPLEMENTATION),
                "--output-dir", "/tmp/nonlinear-mz-reverse-fresh", "--max-seconds", "7800"]), 0)
        self.assertEqual(vars(runner.call_args.args[0]), vars(self.args("/tmp/nonlinear-mz-reverse-fresh")))


if __name__ == "__main__":
    unittest.main()
