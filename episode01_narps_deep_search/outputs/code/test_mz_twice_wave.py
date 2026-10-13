"""Focused selected-wave checks; execute only inside task-owned compute."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_mz_twice_wave as wave
import test_mz_effect_level_recovery as fixtures


class TwiceWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.EffectLevelRecoveryTests.setUpClass()
        cls.fixture = fixtures.EffectLevelRecoveryTests(
            "test_original_simulation_and_four_fits_use_selected_level_seed_and_frozen_spec")
        cls.spec = cls.fixture.spec

    def exercise(self, output, index=1, **options):
        original = wave.levels.run
        bindings = self.fixture.bindings()
        with patch.object(wave.levels, "run", side_effect=wave.run):
            result = self.fixture.exercise(output, "twice", index, **options)
        self.assertIs(wave.levels.run, original)
        self.assertIs(wave._run_effect_level, original)
        self.assertEqual(self.fixture.bindings(), bindings)
        return result

    def assert_scope(self, report, index):
        self.assertEqual((report["phase"], report["effect_level"], report["replicate_index"], report["seed"]),
                         ("MZ_twice_SESOI_baseline_component", "twice", index, 1900057 + 10007 * index))
        self.assertEqual(report["implementation"], self.spec)
        self.assertEqual(report["implementation"]["optimizer"]["maximum_sweeps"], 500)
        self.assertEqual(report["focal_generating_truth"], wave.levels.focal_truth("twice"))
        self.assertEqual((report["SESOI_level"], report["generating_z"]), (2., .40))
        self.assertEqual(report["selected_replicates"], [index])
        self.assertEqual((report["planned_cell_replicates"], report["fixed_cohort_count"]), (200, 200))
        self.assertIs(report["wave_selected"], True)
        self.assertEqual(report["dispatch_scope"], wave.DISPATCH_SCOPE)
        self.assertEqual(report["externally_selected_index_range"], [1, 199])
        for field in ("subsequent_wave_selected", "full_menu_wave_selected", "retry_selected", "automatic_retry",
                      "observed_choices_or_RTs_used", "old_synthetic_responses_used", "episode_terminal"):
            self.assertIs(report[field], False)
        for field in wave.levels.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertEqual(report["full_behavioral_gate"], "not_evaluated")
        self.assertEqual(report["baseline_diagnostic_complete"], report["valid_recovery_component"])
        self.assertNotIn("no wave is selected", report["interpretation"])
        self.assertNotIn("core_family_selection_pass", report)
        for field, value in wave.levels.criterion_scope("twice").items():
            self.assertEqual(report[field], value)
            if "comparison" in report:
                self.assertEqual(report["comparison"][field], value)

    def test_both_endpoints_preserve_seed_truth_four_fits_scoring_and_every_checkpoint_scope(self):
        for index in (1, 199):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "new"
                report, fits, checkpoints, scores = self.exercise(output, index)
                truth = json.loads((output / "generating_parameters.json").read_text())
                self.assertTrue((output / "simulated_events.csv.gz").exists())
            self.assertTrue(report["baseline_diagnostic_complete"])
            self.assertEqual([(train, model, rival) for train, model, rival, spec in fits],
                             [(train, model, "fixed") for train, held in wave.levels.core.HALVES for model in ("M0", "MZ")])
            self.assertTrue(all(spec == self.spec for train, model, rival, spec in fits))
            self.assertEqual(len(scores), 432)
            self.assertEqual((report["participants"], report["runs"], report["trials"]), (108, 432, 27648))
            self.assertTrue(report["comparison"]["necessary_nested_MZ_vs_M0_pass"])
            self.assertFalse(report["counts_as_curtailment_failure"])
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint, index)
            self.assertTrue(all(all(params[key] == value for key, value in wave.levels.focal_truth("twice").items())
                                for params in truth.values()))
            rng = np.random.default_rng(1900057 + 10007 * index)
            expected = {name: wave.levels.core.truncated_normal(rng, self.spec["parameters"][name]["population_center"],
                        self.spec["parameters"][name]["individual_scale_center"], self.spec["parameters"][name]["bounds"])
                        for name in ("kappa", "log_a", "nondecision_logit")}
            self.assertEqual({name: truth["p000"][name] for name in expected}, expected)

    def test_valid_failed_nested_screen_keeps_twice_criterion_without_a_gate_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fits, checkpoints, scores = self.exercise(Path(directory) / "new", margin=0.)
        self.assertTrue(report["baseline_diagnostic_complete"])
        self.assertFalse(report["comparison"]["necessary_nested_MZ_vs_M0_pass"])
        self.assertTrue(report["counts_as_curtailment_failure"])
        self.assertTrue(report["comparison"]["counts_as_curtailment_failure"])
        for checkpoint in checkpoints:
            self.assert_scope(checkpoint, 1)

    def test_technical_failure_keeps_prefix_unknown_and_selected_wave_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            report, fits, checkpoints, scores = self.exercise(Path(directory) / "new", nonconverged=True)
        self.assertEqual(report["status"], "technical_recovery_failure")
        self.assertFalse(report["baseline_diagnostic_complete"])
        self.assertFalse(report["counts_as_curtailment_failure"])
        self.assertEqual(len(report["fits"]), 2)
        self.assertFalse(report["fits"][-1]["converged"])
        self.assertNotIn("comparison", report)
        self.assertNotIn("participant_scores", report)
        for checkpoint in checkpoints:
            self.assert_scope(checkpoint, 1)

    def test_half_pilot_or_outside_index_guards_precede_source_and_exception_restores_writer(self):
        bindings = self.fixture.bindings()
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "read_text", side_effect=AssertionError("No source read")), \
                patch.object(wave, "_run_effect_level", side_effect=RuntimeError("entry failed")) as runner:
            for level, index in (("half", 1), ("twice", 0), ("twice", 200), ("twice", -1), ("twice", True)):
                with self.subTest(level=level, index=index), self.assertRaises(ValueError):
                    wave.run(self.fixture.arguments(Path(directory) / "new", level, index))
            runner.assert_not_called()
            self.assertEqual(self.fixture.bindings(), bindings)
            with self.assertRaisesRegex(RuntimeError, "entry failed"):
                wave.run(self.fixture.arguments(Path(directory) / "new", "twice", 1))
            self.assertEqual(self.fixture.bindings(), bindings)
            self.assertEqual(runner.call_args.args[0].level, "twice")

    def test_cli_normalizes_only_twice_and_preserves_paths_and_process_cap(self):
        report = {"baseline_diagnostic_complete": True}
        with patch.object(wave, "_run_effect_level", return_value=report) as runner, redirect_stdout(io.StringIO()):
            code = wave.main(["--replicate-index", "199", "--design-events", str(wave.levels.core.DESIGN_SOURCE),
                "--implementation", str(wave.levels.core.IMPLEMENTATION), "--output-dir", "/tmp/explicit-twice-output",
                "--max-seconds", "5400"])
        args = runner.call_args.args[0]
        self.assertEqual(code, 0)
        self.assertEqual(vars(args), dict(level="twice", replicate_index=199,
            design_events=wave.levels.core.DESIGN_SOURCE, implementation=wave.levels.core.IMPLEMENTATION,
            output_dir=Path("/tmp/explicit-twice-output"), max_seconds=5400))


if __name__ == "__main__":
    unittest.main()
