"""Fixed-cell bounds and identity checks, without fits or scientific payloads."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import summarize_single_process_recovery as core


class SingleProcessSummaryTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"recovery_plan": {"replicates_per_qualified_generating_cell": 200},
                     "optimizer": {"maximum_sweeps": 500}}

    def report(self, index, model="MV", core_pass=False, richer_pass=False, valid=True):
        requirements = {name: True for name in core.CORE + core.DIAGNOSTICS}
        requirements[core.CORE[0]] = core_pass
        requirements[core.DIAGNOSTICS[0]] = richer_pass
        truth = {"eta": math.log(1.15) if model == "MV" else 0.,
                 "rho": math.log(.45/.55) if model == "MZ" else 0., "criterion": 0.}
        result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": f"{model}_SESOI_fitted_recovery_component",
            "generating_model": model, "replicate_index": index, "planned_cell_replicates": 200,
            "seed": core.BASE_SEEDS[model] + 10007 * index, "SESOI_level": 1.,
            "focal_generating_truth": truth, "implementation": copy.deepcopy(self.spec),
            "full_behavioral_gate": "not_evaluated", "full_recovery_pass": None,
            "valid_recovery_component": valid, "counts_as_curtailment_failure": valid and not core_pass,
            "status": "necessary_primary_comparison_completed" if valid else "running"}
        if valid:
            result["comparison"] = {"focal_model": model, "requirements": requirements,
                "core_family_selection_pass": core_pass, "counts_as_curtailment_failure": not core_pass,
                "necessary_primary_comparison_pass": core_pass and richer_pass,
                "richer_primary_diagnostic_pass": core_pass and richer_pass, "full_recovery_pass": None}
        return f"replicate{index:03d}.json", result

    def test_twenty_vs_twenty_one_failures_in_both_fixed_cells(self):
        for model in ("MV", "MZ"):
            reports = [self.report(i, model) for i in range(20)]
            first = core.summarize_reports(reports, model, self.spec)
            self.assertEqual(first["best_possible_completed_success_count"], 180)
            self.assertFalse(first["necessary_90pct_cell_criterion_unreachable"])
            next_one = core.summarize_reports(reports + [self.report(20, model)], model, self.spec)
            self.assertEqual(next_one["best_possible_completed_success_count"], 179)
            self.assertTrue(next_one["necessary_90pct_cell_criterion_unreachable"])
            self.assertFalse(next_one["episode_terminal"])
            self.assertEqual(next_one["full_recovery_gate"], "not_evaluated")

    def test_technical_and_richer_only_failures_do_not_spend_the_failure_allowance(self):
        reports = [self.report(i, core_pass=True, richer_pass=False) for i in range(21)]
        reports += [self.report(21, valid=False), self.report(22, valid=False)]
        result = core.summarize_reports(reports, "MV", self.spec)
        self.assertEqual((result["valid_failures"], result["valid_successes"]), (0, 21))
        self.assertEqual(result["valid_richer_primary_failures"], 21)
        self.assertEqual(result["remaining_unique_replicates"], 179)
        self.assertEqual(result["best_possible_completed_success_count"], 200)
        self.assertEqual(len(result["technical_attempts"]), 2)

    def test_technical_then_one_valid_attempt_counts_index_once_but_two_valid_refuse(self):
        first = self.report(0, valid=False)
        valid = self.report(0, core_pass=True, richer_pass=True)
        result = core.summarize_reports([first, valid], "MV", self.spec)
        self.assertEqual(result["valid_replicates"], 1)
        self.assertEqual(len(result["technical_attempts"]), 1)
        for reports in ([valid, valid], [valid, self.report(0)]):
            with self.subTest(reports=reports), self.assertRaisesRegex(ValueError, "Multiple valid"):
                core.summarize_reports(reports, "MV", self.spec)

    def test_model_seed_truth_index_and_implementation_mixes_refuse(self):
        for field, value in (("generating_model", "MZ"), ("phase", "MC_SESOI_fitted_recovery_component"),
                             ("seed", 810032), ("replicate_index", 200), ("replicate_index", True),
                             ("planned_cell_replicates", 199), ("SESOI_level", .5)):
            path, report = self.report(0)
            report[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                core.summarize_reports([(path, report)], "MV", self.spec)
        for technical in (False, True):
            path, report = self.report(0, valid=not technical)
            report["focal_generating_truth"]["criterion"] = .01
            with self.assertRaisesRegex(ValueError, "truth"):
                core.summarize_reports([(path, report)], "MV", self.spec)
            path, report = self.report(0, valid=not technical)
            report["implementation"]["optimizer"]["maximum_sweeps"] = 501
            with self.assertRaisesRegex(ValueError, "implementation"):
                core.summarize_reports([(path, report)], "MV", self.spec)
        with self.assertRaises(ValueError):
            core.summarize_reports([self.report(0, "MV"), self.report(1, "MZ")], "MV", self.spec)

    def test_missing_nonboolean_or_contradictory_core_flags_refuse(self):
        for field, value in (("core_family_selection_pass", True), ("counts_as_curtailment_failure", False),
                             ("richer_primary_diagnostic_pass", True), ("necessary_primary_comparison_pass", True)):
            path, report = self.report(0)
            report["comparison"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                core.summarize_reports([(path, report)], "MV", self.spec)
        for value in (None, 0, "false"):
            path, report = self.report(0)
            report["comparison"]["requirements"][core.CORE[0]] = value
            with self.assertRaises(ValueError):
                core.summarize_reports([(path, report)], "MV", self.spec)
        path, report = self.report(0, valid=False)
        report["counts_as_curtailment_failure"] = True
        with self.assertRaises(ValueError):
            core.summarize_reports([(path, report)], "MV", self.spec)

    def test_two_SE_failure_alone_counts_and_complete_success_never_passes_full_gate(self):
        path, report = self.report(0, core_pass=False, richer_pass=True)
        report["comparison"]["requirements"][core.CORE[0]] = True
        report["comparison"]["requirements"][core.CORE[1]] = False
        result = core.summarize_reports([(path, report)], "MV", self.spec)
        self.assertEqual(result["valid_failures"], 1)
        complete = core.summarize_reports([self.report(i, core_pass=True, richer_pass=True) for i in range(200)], "MV", self.spec)
        self.assertEqual(complete["valid_successes"], 200)
        self.assertEqual(complete["remaining_unique_replicates"], 0)
        self.assertEqual(complete["full_recovery_gate"], "not_evaluated")
        self.assertFalse(complete["episode_terminal"])

    def test_explicit_path_cli_publishes_fresh_output_without_rewriting_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            implementation = root / "fixed.json"
            implementation.write_text(json.dumps(self.spec))
            report_path = root / "one.json"
            original = json.dumps(self.report(0)[1])
            report_path.write_text(original)
            output = root / "summary.json"
            command = ["--model", "MV", "--implementation", str(implementation), "--report", str(report_path), "--output", str(output)]
            with patch.object(core, "IMPLEMENTATION", implementation), patch.object(core, "require_compute"), patch("builtins.print"):
                self.assertEqual(core.main(command), 0)
                with self.assertRaises(FileExistsError):
                    core.main(command)
            self.assertEqual(report_path.read_text(), original)
            self.assertEqual(json.loads(output.read_text())["valid_failures"], 1)


if __name__ == "__main__":
    unittest.main()
