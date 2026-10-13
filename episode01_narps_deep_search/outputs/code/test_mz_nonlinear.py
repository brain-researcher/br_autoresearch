"""Focused nonlinear component checks; execute only inside task-owned compute."""
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

import run_mz_nonlinear as nonlinear
import test_mz_other_partitions as fixtures


class NonlinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OtherPartitionsTests.setUpClass()
        cls.base = fixtures.OtherPartitionsTests("test_both_partitions_replay_primary_then_fit_four_fresh_original500_models")
        cls.fixture, cls.spec, cls.people = cls.base.fixture, cls.base.spec, cls.base.people

    def source(self):
        return self.base.source()

    def fit(self, model, train=(1, 4), held=(2, 3), converged=True):
        fit = self.fixture.fit(model, train, held, converged)
        names = nonlinear.parameter_names(model, "nonlinear")
        centers = [self.spec["parameters"][name]["population_center"] for name in names]
        scales = [self.spec["parameters"][name]["individual_scale_center"] for name in names]
        fit.update(rival="nonlinear", parameter_names=names, parameters=[list(centers) for _ in self.people],
                   global_mean=centers, version_means=[list(centers), list(centers)],
                   version_standard_deviations=[list(scales), list(scales)])
        return fit

    def args(self, output, model="M0", **changes):
        values = dict(model=model, source_dir=nonlinear.SOURCE, implementation=nonlinear.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def exercise(self, output, model="M0", *, bad_source=False, replay_error=False,
                 fail_second=False, score_error=False, deadline_stage=None):
        source, truth = self.source()
        if bad_source:
            source["replicate_index"] = 1
        before = deepcopy((source, truth, self.spec, self.people))
        original_halves = nonlinear.shared.HALVES
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        splits, fits, likelihoods, checkpoints = [], [], [], []
        read_text, writer, now = Path.read_text, nonlinear.write_report, [100.]

        def reader(path, *args, **kwargs):
            if path == nonlinear.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == nonlinear.SOURCE / nonlinear.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 5401
                return json.dumps(source)
            if path == nonlinear.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, nonlinear.SOURCE / "simulated_events.csv.gz")
            splits.append((tuple(train), tuple(held)))
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(train), panel(held)

        def fitter(training, selected, rival, spec):
            self.assertEqual((selected, rival, spec), (model, "nonlinear", self.spec))
            self.assertEqual(spec["optimizer"]["maximum_sweeps"], 500)
            train, held = nonlinear.HALVES[len(fits)]
            self.assertEqual(len(training), 108)
            for record in training.values():
                self.assertEqual(len(record["rows"]), 128)
                self.assertEqual({int(row["run"]) for row in record["rows"]}, set(train))
                self.assertTrue(all(int(row["run"]) not in held for row in record["rows"]))
            self.assertEqual(len(likelihoods), 432 + 108 * len(fits))
            fits.append((selected, rival, train, deepcopy(spec)))
            return self.fit(selected, train, held, converged=not (fail_second and len(fits) == 2))

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual(spec, self.spec)
            self.assertEqual(len(data["gain"]), 128)
            likelihoods.append((rival, tuple(names)))
            if rival == "fixed":
                self.assertEqual(fits, [])
                self.assertNotIn("gamma_gain", names)
                value = -1.98 if "rho" in names else -2.
                return np.full(128, value + (1e-8 if replay_error else 0.))
            self.assertEqual(rival, "nonlinear")
            self.assertEqual(names, nonlinear.parameter_names(model, "nonlinear"))
            if score_error:
                return np.full(64, -1.)
            return np.full(128, -2.66 if model == "MZ" else -2.7)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["nonlinear_component_complete"]:
                now[0] += 5401

        with ExitStack() as stack:
            stack.enter_context(patch.object(nonlinear, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(nonlinear.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(nonlinear, "load_split", side_effect=split))
            stack.enter_context(patch.object(nonlinear, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(nonlinear.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(nonlinear, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(nonlinear.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = nonlinear.run(self.args(output, model))
        self.assertEqual((source, truth, self.spec, self.people), before)
        self.assertEqual(nonlinear.shared.HALVES, original_halves)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / nonlinear.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        self.assertFalse((output / "generating_parameters.json").exists())
        for record in checkpoints:
            self.assert_scope(record, model)
        return report, splits, fits, likelihoods, source

    def assert_scope(self, report, model):
        self.assertEqual((report["phase"], report["fitted_model"], report["rival"], report["replicate_index"], report["seed"]),
                         (nonlinear.PHASE, model, "nonlinear", 0, 910031))
        for field in nonlinear.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_nonlinear",
                      "observed_choices_or_RTs_used", "wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        for field in ("core_family_selection_pass", "counts_as_curtailment_failure", "valid_recovery_component", "nested_diagnostic"):
            self.assertNotIn(field, report)
        self.assertIn("curvature-adjusted reference, not globally symmetric valuation", report["interpretation"])

    def test_nonlinear_validator_requires_five_or_six_parameters_and_unchanged_fit_evidence(self):
        for model, dimension in (("M0", 5), ("MZ", 6)):
            fit = self.fit(model)
            before = deepcopy((fit, self.spec))
            nonlinear.validate_nonlinear_fit(fit, model, (1, 4), (2, 3), self.spec, self.people)
            self.assertEqual(len(fit["parameter_names"]), dimension)
            self.assertEqual(fit["parameter_names"][-2:], ["gamma_gain", "gamma_loss"])
            self.assertEqual((fit, self.spec), before)
            changes = (lambda item: item.update(rival="fixed"),
                       lambda item: item.update(parameter_names=item["parameter_names"][:-1]),
                       lambda item: item["parameters"][0].__setitem__(-1, 1.50001),
                       lambda item: item.update(converged=False),
                       lambda item: item.update(subject_optimizer_failures=[{"participant": "p000"}]),
                       lambda item: item.update(objective_history=[1.] * 501),
                       lambda item: item.update(objective_history=[2., 1.]),
                       lambda item: item["version_standard_deviations"][0].__setitem__(-1, 0.))
            for number, change in enumerate(changes):
                altered = deepcopy(fit)
                change(altered)
                with self.subTest(model=model, change=number), self.assertRaises((ValueError, RuntimeError)):
                    nonlinear.validate_nonlinear_fit(altered, model, (1, 4), (2, 3), self.spec, self.people)

    def test_explicit_nonlinear_scorer_nests_fixed_at_unit_powers_and_changes_with_unequal_powers(self):
        rows = [dict(self.people["p000"][0], gain="20", loss="20", choice=str(choice),
                     no_response=str(int(choice == -1)), response_time="" if choice == -1 else "1.2",
                     response_category={-1: "no_response", 0: "simulated_reject", 1: "simulated_accept"}[choice])
                for choice in (1, 0, -1, 1)] * 32
        scoring = {"p000": {"rows": rows}}
        for model in nonlinear.MODELS:
            old, new = self.fixture.fit(model), self.fit(model)
            for fit in (old, new):
                fit["participants"], fit["parameters"] = ["p000"], fit["parameters"][:1]
            fixed_score = nonlinear.shared.score_fit(old, scoring, self.spec)
            nonlinear_score = nonlinear.score_nonlinear_fit(new, scoring, self.spec)
            self.assertEqual(nonlinear_score, fixed_score)
            self.assertTrue(np.isfinite(nonlinear_score["p000"]))
            new["parameters"][0][-2:] = [1.25, .75]
            changed = nonlinear.score_nonlinear_fit(new, scoring, self.spec)
            self.assertNotAlmostEqual(changed["p000"], fixed_score["p000"], places=10)
        with patch.object(nonlinear.shared, "trial_likelihood", return_value=np.full(128, -2.)):
            self.assertEqual(nonlinear.score_nonlinear_fit(new, scoring, self.spec), {"p000": -1.})
        for bad in (np.zeros(64), np.full(128, np.nan)):
            with patch.object(nonlinear.shared, "trial_likelihood", return_value=bad), self.assertRaises(ValueError):
                nonlinear.score_nonlinear_fit(new, scoring, self.spec)

    def test_each_model_replays_fixed_primary_then_fits_only_its_two_nonlinear_halves(self):
        for model in nonlinear.MODELS:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", model)
            self.assertTrue(report["nonlinear_component_complete"])
            self.assertEqual(splits, list(nonlinear.HALVES))
            self.assertEqual([(selected, rival, train) for selected, rival, train, spec in fits],
                             [(model, "nonlinear", train) for train, held in nonlinear.HALVES])
            self.assertEqual([rival for rival, names in likelihoods], ["fixed"] * 432 + ["nonlinear"] * 216)
            self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
            self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
            self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
            self.assertEqual(report["scored_primary_directions"], [{"training_runs": list(train), "scoring_runs": list(held)}
                                                                   for train, held in nonlinear.HALVES])
            for row in report["participant_scores"].values():
                self.assertEqual(set(row), {"nonlinear_score", "version"})
                self.assertAlmostEqual(row["nonlinear_score"], -2.66 if model == "MZ" else -2.7)

    def test_source_identity_or_fixed_primary_replay_failure_prevents_nonlinear_fits(self):
        for option in ({"bad_source": True}, {"replay_error": True}):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", **option)
            self.assertEqual(report["status"], "technical_nonlinear_failure")
            self.assertEqual(fits, [])
            self.assertFalse(report["saved_primary_scores_replayed"])
            self.assertNotIn("participant_scores", report)

    def test_failed_fit_or_scoring_retains_returned_prefix_as_unknown(self):
        for option, fit_count, scored_count in (({"fail_second": True}, 2, 1), ({"score_error": True}, 1, 0)):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", "MZ", **option)
            self.assertEqual(report["status"], "technical_nonlinear_failure")
            self.assertFalse(report["nonlinear_component_complete"])
            self.assertEqual(len(report["fits"]), fit_count)
            self.assertEqual(len(report["scored_primary_directions"]), scored_count)
            self.assertNotIn("participant_scores", report)
            self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])

    def test_whole_cap_covers_source_and_finalization_and_restores_handlers(self):
        for stage, count in (("source", 0), ("finalization", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertEqual(report["status"], "technical_nonlinear_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(len(report["fits"]), count)
            self.assertFalse(report["nonlinear_component_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertTrue(report["finished_utc"])

    def test_scope_cpu_time_and_fresh_output_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(nonlinear, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read allowed")):
            for change in ({"model": "MV"}, {"source_dir": Path(directory)},
                           {"implementation": Path(directory) / "other.json"}, {"max_seconds": True},
                           {"max_seconds": 0}, {"max_seconds": 5401}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    nonlinear.run(self.args(Path(directory) / "new", **change))
            for field in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
                with self.subTest(field=field), patch.dict(os.environ, {field: "2"}), self.assertRaises(ValueError):
                    nonlinear.run(self.args(Path(directory) / "new"))
            for output in (Path(directory), nonlinear.SOURCE / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    nonlinear.run(self.args(output))


if __name__ == "__main__":
    unittest.main()
