"""Focused effect-level adapter checks; execute only inside task-owned compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_mz_effect_level_recovery as levels
import test_mz_null_rival as fixtures


class EffectLevelRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.source = cls.fixture.spec, cls.fixture.people

    def arguments(self, output, level="half", index=0, **changes):
        values = dict(level=level, replicate_index=index, design_events=levels.core.DESIGN_SOURCE,
                      implementation=levels.core.IMPLEMENTATION, output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def bindings(self):
        return tuple(getattr(levels.core, name) for name in
                     ("BASE_SEEDS", "focal_truth", "summarize_paired_scores", "write_report", "REPORT_NAME"))

    def exercise(self, output, level, index=0, *, margin=.02, nonconverged=False):
        core = levels.core
        before, bindings = deepcopy((self.source, self.spec)), self.bindings()
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        fits, checkpoints, simulator_z, score_calls = [], [], [], []
        original_write, original_rng = core.write_report, np.random.default_rng

        def simulator(rng, drift, boundary, z, *args, **kwargs):
            simulator_z.append(z)
            return np.ones(len(drift), dtype=int), np.full(len(drift), .5)

        def split(path, train, held):
            return (tuple(train), {person: {"rows": [row for row in rows if int(row["run"]) in held]}
                                  for person, rows in self.source.items()})

        def fitter(training, model, rival, spec):
            fits.append((training, model, rival, deepcopy(spec)))
            return self.fixture.fit(model, converged=not (nonconverged and len(fits) == 2))

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual(spec, self.spec)
            self.assertEqual(rival, "fixed")
            score_calls.append(tuple(names))
            return np.full(128, -1. + (margin if "rho" in names else 0.))

        def writer(directory, report):
            checkpoints.append(deepcopy(report))
            return original_write(directory, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(levels, "require_compute"))
            stack.enter_context(patch.object(core, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(core, "load_design", return_value=self.source))
            stack.enter_context(patch.object(core.deadline_ddm, "simulate", side_effect=simulator))
            stack.enter_context(patch.object(core, "load_split", side_effect=split))
            stack.enter_context(patch.object(core, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(core, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(core, "write_report", side_effect=writer))
            rng = stack.enter_context(patch.object(np.random, "default_rng", wraps=original_rng))
            stack.enter_context(redirect_stdout(io.StringIO()))
            installed = self.bindings()
            report = levels.run(self.arguments(output, level, index))
            self.assertEqual(self.bindings(), installed)
            self.assertEqual([call.args[0] for call in rng.call_args_list],
                             [levels.level_seed(level, index)] + ([] if nonconverged else [levels.level_seed(level, index) + 4099]))
        self.assertEqual(self.bindings(), bindings)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual((self.source, self.spec), before)
        self.assertEqual(json.loads((output / levels.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "single_process_recovery.json").exists())
        self.assertEqual(len(simulator_z), 108)
        np.testing.assert_allclose(simulator_z, levels.LEVELS[level]["z"], rtol=0, atol=1e-15)
        return report, fits, checkpoints, score_calls

    def assert_scope(self, report, level, index):
        self.assertEqual((report["phase"], report["effect_level"], report["seed"]),
                         (f"MZ_{level}_SESOI_baseline_component", level, levels.level_seed(level, index)))
        self.assertEqual(report["SESOI_level"], levels.LEVELS[level]["multiplier"])
        self.assertEqual(report["focal_generating_truth"], levels.focal_truth(level))
        self.assertEqual(report["generating_z"], levels.LEVELS[level]["z"])
        self.assertEqual(report["implementation"], self.spec)
        self.assertEqual(report["implementation"]["optimizer"]["maximum_sweeps"], 500)
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual(report["planned_cell_replicates"], 200)
        self.assertEqual(report["fixed_cohort_count"], 200)
        self.assertEqual(report["fitted_models"], ["M0", "MZ"])
        self.assertEqual(report["baseline_diagnostic_complete"], report["valid_recovery_component"])
        for field in levels.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("wave_selected", "full_menu_wave_selected", "retry_selected", "automatic_retry",
                      "observed_choices_or_RTs_used", "old_synthetic_responses_used", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        self.assertEqual(report["full_behavioral_gate"], "not_evaluated")
        self.assertNotIn("core_family_selection_pass", report)
        for key, expected in levels.criterion_scope(level).items():
            self.assertEqual(report[key], expected)
            if "comparison" in report:
                self.assertEqual(report["comparison"][key], expected)
        if "comparison" in report:
            self.assertNotIn("core_family_selection_pass", report["comparison"])
            self.assertEqual(report["comparison"]["family_selection"], "not_evaluated")
            self.assertEqual(report["counts_as_curtailment_failure"], report["comparison"]["counts_as_curtailment_failure"])
        if level == "half":
            self.assertFalse(report["counts_as_curtailment_failure"])
            if "comparison" in report:
                self.assertEqual(report["comparison"]["curtailment_rule"], "not_applicable_below_SESOI")

    def test_effect_is_scaled_in_z_not_in_logit_and_has_no_eta_or_criterion(self):
        original_rho = levels.core.focal_truth("MZ")["rho"]
        for level, multiplier, z in (("half", .5, .475), ("twice", 2., .40)):
            truth = levels.focal_truth(level)
            self.assertEqual(truth, {"eta": 0., "rho": math.log(z / (1 - z)), "criterion": 0.})
            self.assertAlmostEqual(.5 - z, multiplier * .05)
            self.assertNotAlmostEqual(truth["rho"], multiplier * original_rho, places=6)

    def test_prospective_400_seed_namespace_is_disjoint_from_every_existing_fixed_cell(self):
        seeds = [levels.level_seed(level, index) for level in levels.LEVELS for index in range(200)]
        existing = {base + 10007 * index for base in (710031, 810031, 910031, 1300037) for index in range(200)}
        self.assertEqual(len(set(seeds)), 400)
        self.assertTrue(set(seeds).isdisjoint(existing))
        self.assertEqual((levels.level_seed("half", 0), levels.level_seed("half", 199)), (1700047, 3691440))
        self.assertEqual((levels.level_seed("twice", 0), levels.level_seed("twice", 199)), (1900057, 3891450))
        for level, index in (("sesoi", 0), ("half", -1), ("twice", 200), ("half", True)):
            with self.subTest(level=level, index=index), self.assertRaises(ValueError):
                levels.level_seed(level, index)

    def test_original_simulation_and_four_fits_use_selected_level_seed_and_frozen_spec(self):
        for level, index in (("half", 0), ("twice", 199)):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "new"
                report, fits, checkpoints, scores = self.exercise(output, level, index)
                truth = json.loads((output / "generating_parameters.json").read_text())
                self.assertTrue((output / "simulated_events.csv.gz").exists())
            self.assertTrue(report["baseline_diagnostic_complete"])
            self.assertEqual([(train, model, rival) for train, model, rival, spec in fits],
                             [(train, model, "fixed") for train, held in levels.core.HALVES for model in ("M0", "MZ")])
            self.assertTrue(all(spec == self.spec for train, model, rival, spec in fits))
            self.assertEqual(len(scores), 432)
            self.assertEqual((report["participants"], report["runs"], report["trials"]), (108, 432, 27648))
            self.assertTrue(report["comparison"]["necessary_nested_MZ_vs_M0_pass"])
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, level, index)
            self.assertTrue(all(all(params[key] == value for key, value in levels.focal_truth(level).items())
                                for params in truth.values()))
            rng = np.random.default_rng(levels.level_seed(level, index))
            expected = {name: levels.core.truncated_normal(rng, self.spec["parameters"][name]["population_center"],
                        self.spec["parameters"][name]["individual_scale_center"], self.spec["parameters"][name]["bounds"])
                        for name in ("kappa", "log_a", "nondecision_logit")}
            self.assertEqual({name: truth["p000"][name] for name in expected}, expected)

    def test_same_failed_nested_screen_is_never_a_half_SESOI_90pct_failure(self):
        for level in levels.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                report, fits, checkpoints, scores = self.exercise(Path(directory) / "new", level, margin=0.)
            self.assertTrue(report["baseline_diagnostic_complete"])
            self.assertFalse(report["comparison"]["necessary_nested_MZ_vs_M0_pass"])
            self.assertFalse(report["comparison"]["requirements"]["mean_gain_at_least_0p01"])
            self.assertFalse(report["comparison"]["requirements"]["mean_gain_greater_than_two_participant_SE"])
            self.assertEqual(report["counts_as_curtailment_failure"], level == "twice")
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, level, 0)

    def test_nonconvergence_preserves_returned_prefix_and_is_never_a_scientific_failure(self):
        for level in levels.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                report, fits, checkpoints, scores = self.exercise(Path(directory) / "new", level, nonconverged=True)
            self.assertEqual(report["status"], "technical_recovery_failure")
            self.assertFalse(report["baseline_diagnostic_complete"])
            self.assertFalse(report["counts_as_curtailment_failure"])
            self.assertEqual(len(report["fits"]), 2)
            self.assertFalse(report["fits"][-1]["converged"])
            self.assertNotIn("comparison", report)
            self.assertNotIn("participant_scores", report)
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, level, 0)

    def test_entry_exception_restores_scoped_hooks_and_resource_guards_precede_core(self):
        before = self.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(levels, "require_compute"), \
                patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}):
            with patch.object(levels.core, "run", side_effect=RuntimeError("entry failed")) as runner:
                with self.assertRaisesRegex(RuntimeError, "entry failed"):
                    levels.run(self.arguments(Path(directory) / "new"))
                self.assertEqual(self.bindings(), before)
                runner.reset_mock()
                for change in ({"max_seconds": True}, {"max_seconds": 5401}, {"design_events": Path("other.csv.gz")}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        levels.run(self.arguments(Path(directory) / "new", **change))
                with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), self.assertRaises(ValueError):
                    levels.run(self.arguments(Path(directory) / "new"))
                runner.assert_not_called()
        self.assertEqual(self.bindings(), before)


if __name__ == "__main__":
    unittest.main()
