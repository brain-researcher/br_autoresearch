"""Focused terminal-versus-complete menu pooling checks; run on compute only."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_menu_wave as wave
import test_mz_null_rival as fixtures


class MenuWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")

    def setUp(self):
        self.implementation = deepcopy(self.fixture.spec)

    def technical(self, index=2, *, half=1):
        fits = [self.fixture.fit("MVC", train, held) for train, held in wave.HALVES[:half]]
        for fit in fits:
            fit["elapsed_seconds"] = 10.0
        fits[-1].update(converged=False, objective_history=list(range(500, 0, -1)))
        return {
            "analysis_id": "behavioral_mechanism_kill_test",
            "phase": "MZ_SESOI_saved_replicate_rival_diagnostics",
            "generating_model": "MZ", "fitted_model": "MVC", "replicate_index": index,
            "seed": wave.cell_seed("MZ", index), "implementation": self.implementation,
            "source_dir": str(wave.source_directory(index)), "status": "technical_rival_failure",
            "rival_diagnostics_complete": False,
            "error": "RuntimeError: Nonconverged fit is technical; it is not a recovery failure",
            "elapsed_seconds": 25.0, "fits": fits, "completed_rivals": [],
            "participants": 108, "runs": 432, "trials": 27648,
            "saved_scores_replayed": True, "saved_score_replay_max_absolute_discrepancy": 0.0,
            "saved_score_replay_absolute_tolerance": 1e-12,
            "scored_primary_directions": [{"training_runs": list(train), "scoring_runs": list(held)}
                                          for train, held in wave.HALVES[:half - 1]],
        }

    def summary(self, maximal):
        # Existing builders own numeric/fit validation; these small return
        # fixtures isolate whether the orchestrator counts only their results.
        return {
            "maximal_eligible_models_diagnostic": [maximal],
            "nested_comparisons": {
                f"{model}-{baseline}": {"qualifies_nested_comparison": model == maximal}
                for model, baseline in wave.fixed.EDGES},
            "nested_comparison_eligibility": {
                model: model in ("M0", maximal) for model in wave.fixed.MODELS},
            "multiple_incomparable_eligible_models": False,
        }

    def reports(self, indices):
        values = {wave.IMPLEMENTATION: json.dumps(self.implementation)}
        for index in indices:
            for path in wave.input_paths(index):
                values[path] = json.dumps({"report_path": str(path), "status": "rival_diagnostics_completed"})
        return values

    def invoke(self, arguments, values, *, saved_error=None):
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            return values[path]

        with patch.object(wave, "require_compute"), patch.object(Path, "read_text", reader), \
                patch.object(wave.fixed, "build_summary", return_value=self.summary("MZ")) as fixed_builder, \
                patch.object(wave.saved, "build_summary", return_value=self.summary("M0"),
                             side_effect=saved_error) as saved_builder, redirect_stdout(io.StringIO()):
            result = wave.main(arguments)
        return result, reads, fixed_builder, saved_builder

    def test_terminal_technical_failure_stays_unknown_and_never_enters_counts(self):
        values = self.reports((0, 3))
        mvc_path = wave.input_paths(2)[-1]
        values[mvc_path] = json.dumps(self.technical())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            result, reads, fixed_builder, saved_builder = self.invoke(
                ["--terminal-indices", "3,2,0", "--output", str(output)], values)
            summary = json.loads(output.read_text())
        self.assertEqual(result, 0)
        self.assertEqual(set(reads), set(values))
        self.assertEqual(reads.count(mvc_path), 1)
        fixed_builder.assert_called_once()
        saved_builder.assert_called_once()
        self.assertEqual(saved_builder.call_args.args[-1], 3)
        self.assertEqual(summary["fixed_cohort_count"], 200)
        self.assertEqual(summary["completed_indices"], [0, 3])
        self.assertEqual(summary["selected_terminal_indices"], [0, 2, 3])
        self.assertEqual(summary["selection_mode"], "terminal_with_technical_unknowns")
        self.assertEqual(len(summary["unknown_indices"]), 198)
        self.assertIn(1, summary["unknown_indices"])
        self.assertIn(2, summary["unknown_indices"])
        unknown = summary["cohorts"][2]
        self.assertEqual(unknown["status"], "unknown")
        self.assertEqual(unknown["reason"], "MVC_technical_rival_failure")
        self.assertEqual(unknown["technical_report"], str(mvc_path))
        self.assertEqual(unknown["error"], self.technical()["error"])
        self.assertEqual(unknown["technical_cause"]["classification"], "first_half_500_sweep_exhaustion")
        self.assertNotIn("summary", unknown)
        self.assertEqual(summary["cohorts"][1]["reason"], "MC1_interrupted_retry_held")
        counts = summary["diagnostic_counts"]
        self.assertEqual((counts["completed_menus"], counts["unknown_menus"]), (2, 198))
        self.assertEqual(counts["nested_comparison_qualifications"]["MZ-M0"], 1)
        self.assertEqual(counts["nested_comparison_eligibility"]["M0"], 2)
        self.assertEqual(sum(row["menus"] for row in counts["maximal_eligible_model_sets_diagnostic"]), 2)
        for field in ("family_selection", "confusion_rate", "interval_coverage", "full_gate"):
            self.assertEqual(summary[field], "not_evaluated")
        self.assertIsNone(summary["full_recovery_pass"])

    def test_second_half_exhaustion_validates_prior_fit_and_retains_costs(self):
        report = self.technical(half=2)
        with patch.object(wave, "validate_fit", wraps=wave.validate_fit) as validator:
            cause = wave.technical_cause(report, self.implementation)
        validator.assert_called_once()
        self.assertEqual(validator.call_args.args[4], self.implementation)
        self.assertEqual(cause["classification"], "second_half_500_sweep_exhaustion")
        self.assertEqual(cause["prior_half_validation"], "passed_under_original_implementation")
        self.assertEqual(cause["process_elapsed_seconds"], 25.0)
        self.assertEqual(cause["returned_fit_elapsed_seconds"], [10.0, 10.0])
        self.assertEqual((cause["training_runs"], cause["scoring_runs"]), ([2, 3], [1, 4]))
        self.assertGreater(cause["final_objective_change"], cause["unchanged_convergence_threshold"])
        self.assertNotIn("retry_eligible", cause)

    def test_other_causes_and_incomplete_metadata_are_explicitly_unresolved(self):
        changes = (
            ("fit", "converged", True), ("fit", "objective_history", [1.0] * 499),
            ("fit", "objective_history", [1.0] * 500),
            ("fit", "objective_history", [1.0] * 499 + [math.nan]),
            ("fit", "subject_optimizer_failures", [{"participant": "p000", "message": "failed"}]),
            ("fit", "training_runs", [2, 3]), ("fit", "elapsed_seconds", math.inf),
            ("report", "error", "RuntimeError: Internal whole-process deadline exceeded"),
            ("report", "elapsed_seconds", None), ("report", "saved_scores_replayed", False),
            ("report", "saved_score_replay_max_absolute_discrepancy", 2e-12),
            ("report", "saved_score_replay_absolute_tolerance", 1e-10),
            ("report", "scored_primary_directions", [{"training_runs": [1, 4], "scoring_runs": [2, 3]}]),
            ("report", "participant_scores", {}), ("report", "fits", []),
        )
        for location, field, value in changes:
            report = self.technical()
            target = report["fits"][0] if location == "fit" else report
            target[field] = value
            with self.subTest(location=location, field=field, value=value):
                cause = wave.technical_cause(report, self.implementation)
                self.assertEqual(cause["classification"], "unresolved")
                self.assertTrue(cause["reason"])
                self.assertNotIn("retry_eligible", cause)
        changed = deepcopy(self.implementation)
        changed["optimizer"]["maximum_sweeps"] = 1000
        self.assertEqual(wave.technical_cause(self.technical(), changed)["classification"], "unresolved")

    def test_second_half_requires_original_prior_fit_and_correct_scored_direction(self):
        for field, value in (("converged", False), ("objective_history", [100.0] * 501),
                             ("parameter_names", [])):
            report = self.technical(half=2)
            report["fits"][0][field] = value
            with self.subTest(field=field):
                cause = wave.technical_cause(report, self.implementation)
                self.assertEqual(cause["classification"], "unresolved")
                self.assertEqual(cause["reason"], "prior_half_does_not_pass_original_fit_validation")
        report = self.technical(half=2)
        report["scored_primary_directions"] = []
        self.assertEqual(wave.technical_cause(report, self.implementation)["classification"], "unresolved")

    def test_completed_selection_still_rejects_a_technical_menu(self):
        values = self.reports((2,))
        values[wave.input_paths(2)[-1]] = json.dumps(self.technical())
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(wave, "technical_mvc_unknown", side_effect=AssertionError("Strict mode cannot skip")):
            output = Path(directory) / "summary.json"
            with self.assertRaisesRegex(ValueError, "complete two-direction"):
                self.invoke(["--completed-indices", "2", "--output", str(output)], values,
                            saved_error=ValueError("Only complete two-direction rival reports may enter"))
            self.assertFalse(output.exists())

    def test_technical_identity_mismatch_aborts_instead_of_becoming_unknown(self):
        changes = (
            {"analysis_id": "different_analysis"}, {"phase": "M0_null_saved_representative_rival_diagnostics"},
            {"generating_model": "M0"}, {"fitted_model": "MC"}, {"replicate_index": 3},
            {"seed": wave.cell_seed("MZ", 3)}, {"implementation": {}},
            {"source_dir": str(wave.source_directory(3))}, {"rival_diagnostics_complete": True},
            {"error": ""},
        )
        for change in changes:
            report = deepcopy(self.technical())
            report.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                wave.technical_mvc_unknown(report, 2, self.implementation, wave.input_paths(2)[-1])
        with self.assertRaises(ValueError):
            wave.technical_mvc_unknown(self.technical(), 2, self.implementation, wave.input_paths(3)[-1])

    def test_malformed_or_invalid_completed_terminal_reports_still_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            values = {wave.IMPLEMENTATION: json.dumps(self.implementation), wave.input_paths(2)[-1]: "{"}
            with self.assertRaises(json.JSONDecodeError):
                self.invoke(["--terminal-indices", "2", "--output", str(output)], values)
            self.assertFalse(output.exists())
            for status in ("running", "rival_diagnostics_completed"):
                values = self.reports((2,))
                values[wave.input_paths(2)[-1]] = json.dumps({"status": status})
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "Invalid completed menu"):
                    self.invoke(["--terminal-indices", "2", "--output", str(output)], values,
                                saved_error=ValueError("Invalid completed menu"))
                self.assertFalse(output.exists())

    def test_selection_modes_are_exclusive_and_index_one_remains_closed(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("No payload read")), \
                redirect_stderr(io.StringIO()):
            for arguments in (
                ["--output", "/unused/output.json"],
                ["--completed-indices", "0", "--terminal-indices", "2", "--output", "/unused/output.json"],
                ["--terminal-indices", "0,1,2", "--output", "/unused/output.json"],
                ["--terminal-indices", "2,2", "--output", "/unused/output.json"],
            ):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    wave.main(arguments)


if __name__ == "__main__":
    unittest.main()
