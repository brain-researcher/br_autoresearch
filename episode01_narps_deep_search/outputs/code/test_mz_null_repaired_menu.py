"""Focused null0 repaired-menu admission and claim checks; run on compute."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_null_repaired_menu as summary
import test_mz_null_menu as menu_fixtures
import test_mz_null_mvc_convergence_repair as repair_fixtures


class NullRepairedMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        menu_fixtures.NullMenuTests.setUpClass()
        repair_fixtures.NullMVCRepairTests.setUpClass()
        cls.menu_fixture = menu_fixtures.NullMenuTests("test_complete_six_model_panel_retains_null_and_no_claim_even_when_mz_unique")
        cls.repair_fixture = repair_fixtures.NullMVCRepairTests("test_context_preserves_original_fit_and_changes_only_second_half_ceiling")
        cls.spec, cls.truth, cls.people = cls.menu_fixture.spec, cls.menu_fixture.truth, cls.menu_fixture.people

    def reports(self):
        reports = deepcopy(self.menu_fixture.reports())
        base, mc, mv, mvz, mvc = reports
        mc["job_id"] = mv["job_id"] = summary.SIMPLE_JOB_ID
        failed = self.repair_fixture.failed(base)
        runtime, first, metadata = summary.repair.repair_context(self.spec, failed, base, self.people)
        first["elapsed_seconds"] = .002
        second = self.menu_fixture.fixture.fit("MVC", *summary.menu.HALVES[1])
        second.update(elapsed_seconds=2000., objective_history=[float(i) for i in range(599, 0, -1)] + [1.])
        mvc.update(phase=summary.repair.PHASE, job_id=summary.REPAIR_JOB_ID,
                   runtime_implementation=runtime, numerical_repair=metadata,
                   selected_replicates=[0], selected_models=["MVC"],
                   retry_wave_selected=False, full_menu_wave_selected=False,
                   repair_context_validated=True, first_half_fit_reused=True, fresh_second_half_fit_called=True,
                   elapsed_seconds=2020., cumulative_process_elapsed_seconds=failed["elapsed_seconds"] + 2020.,
                   internal_seconds=5400, fits=[first, second], per_half_provenance=[
                       {"training_runs": [1, 4], "scoring_runs": [2, 3], "fit_source": str(summary.repair.FAILED_REPORT),
                        "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": 500},
                       {"training_runs": [2, 3], "scoring_runs": [1, 4], "fit_source": "current_repair",
                        "action": "fresh_fit", "validation_maximum_sweeps": 1000}])
        return [*reports, failed]

    def summarize(self, reports=None, truth=None):
        return summary.build_summary(*(self.reports() if reports is None else reports), self.spec,
                                     self.truth if truth is None else truth, self.people)

    def test_original_nine_contrasts_and_mixed_fit_validation_preserve_null_claim_boundary(self):
        reports = self.reports()
        before = deepcopy(reports)
        original = summary.menu.validate_rival
        with patch.object(summary.menu.fixed, "nested_contrast", wraps=summary.menu.fixed.nested_contrast) as contrasts, \
                patch.object(summary.menu.fixed, "validate_fit", wraps=summary.menu.fixed.validate_fit) as fits, \
                patch.object(summary.menu.null, "validate_fit", wraps=summary.menu.null.validate_fit) as baselines:
            result = self.summarize(reports)
        self.assertIs(summary.menu.validate_rival, original)
        self.assertEqual(reports, before)
        self.assertEqual(contrasts.call_count, 9)
        self.assertEqual([(call.args[1], call.args[4]["optimizer"]["maximum_sweeps"]) for call in fits.call_args_list],
                         [("MC", 500), ("MC", 500), ("MV", 500), ("MV", 500),
                          ("MVZ", 500), ("MVZ", 500), ("MVC", 500), ("MVC", 1000)])
        self.assertEqual([call.args[4]["optimizer"]["maximum_sweeps"] for call in baselines.call_args_list], [500] * 4)
        self.assertEqual((result["generating_model"], result["seed"], result["replicate_index"]), ("M0", 1300037, 0))
        self.assertEqual(result["phase"], summary.PHASE)
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ"])
        self.assertEqual(result["implementation"], self.spec)
        self.assertEqual(result["per_half_provenance"], reports[4]["per_half_provenance"])
        self.assertEqual(result["numerical_repair"], reports[4]["numerical_repair"])
        self.assertEqual(result["cumulative_mvc_process_elapsed_seconds"], 5570.)
        for field in (*summary.menu.null.CLAIM_FIELDS, "early_kill"):
            self.assertEqual(result[field], "not_evaluated")
        for field in ("wave_selected", "retry_wave_selected", "full_menu_wave_selected", "episode_terminal"):
            self.assertFalse(result[field])
        self.assertIsNone(result["full_recovery_pass"])

    def test_original_menu_still_rejects_repair_and_original_fit_ceiling_is_not_relaxed(self):
        base, mc, mv, mvz, mvc, failed = self.reports()
        with self.assertRaises(ValueError):
            summary.menu.build_summary(base, mc, mv, mvz, mvc, self.spec, self.truth, self.people)
        with self.assertRaises(ValueError):
            summary.menu.fixed.fit_panel(mvc, [("MVC", train, held) for train, held in summary.menu.HALVES],
                                         self.spec, self.people)
        reports = self.reports()
        reports[1]["fits"][0]["objective_history"] = deepcopy(reports[4]["fits"][1]["objective_history"])
        with self.assertRaises(ValueError):
            self.summarize(reports)

    def test_only_exact_repair_original_failure_and_completed_source_components_are_admitted(self):
        changes = ((4, "job_id", "46582237"), (4, "phase", "M0_null_saved_representative_rival_diagnostics"),
                   (4, "seed", 1300038), (4, "replicate_index", 1), (4, "generating_model", "MZ"),
                   (4, "source_dir", "/different/source"), (4, "source_finished_utc", "different"),
                   (5, "job_id", "46576235"), (5, "source_finished_utc", "different"),
                   (1, "job_id", "46577174"), (2, "job_id", "46577174"), (3, "job_id", "46575932"),
                   (0, "job_id", "46574267"))
        for position, field, value in changes:
            reports = self.reports()
            reports[position][field] = value
            with self.subTest(position=position, field=field), self.assertRaises(ValueError):
                self.summarize(reports)
        truth = deepcopy(self.truth)
        truth["p000"]["rho"] = .1
        with self.assertRaises(ValueError):
            self.summarize(truth=truth)
        reports = self.reports()
        reports[5]["fits"][0]["participants"][0] = "different"
        with self.assertRaises(ValueError):
            self.summarize(reports)

    def test_preserved_first_fit_mixed_provenance_and_exact_cost_context_are_required(self):
        mutations = (
            lambda report: report["fits"][0]["parameters"][0].__setitem__(0, report["fits"][0]["parameters"][0][0] + 1e-8),
            lambda report: report["fits"][0]["objective_history"].__setitem__(0, 999.),
            lambda report: report["fits"][0].update(objective_history=[1.] * 501),
            lambda report: report["per_half_provenance"][0].update(validation_maximum_sweeps=1000),
            lambda report: report["numerical_repair"].update(first_half_original_fit_elapsed_seconds=.002),
            lambda report: report["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
            lambda report: report.update(cumulative_process_elapsed_seconds=6970.),
            lambda report: report.update(elapsed_seconds=float("nan")),
        )
        for number, mutate in enumerate(mutations):
            reports = self.reports()
            mutate(reports[4])
            with self.subTest(number=number), self.assertRaises(ValueError):
                self.summarize(reports)

    def test_incomplete_repair_replay_baseline_or_claim_changes_abort_and_restore_adapter(self):
        original = summary.menu.validate_rival
        changes = [("status", "running"), ("rival_diagnostics_complete", False), ("fits", []),
                   ("scored_primary_directions", []), ("repair_context_validated", False),
                   ("first_half_fit_reused", False), ("fresh_second_half_fit_called", False),
                   ("saved_scores_replayed", False), ("saved_score_replay_max_absolute_discrepancy", 2e-12),
                   ("saved_score_replay_absolute_tolerance", 1e-10), ("wave_selected", True),
                   ("retry_wave_selected", True), ("full_menu_wave_selected", True),
                   ("full_recovery_pass", True), ("early_kill", "evaluated")]
        changes += [(field, "evaluated") for field in summary.menu.null.CLAIM_FIELDS]
        for field, value in changes:
            reports = self.reports()
            reports[4][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)
            self.assertIs(summary.menu.validate_rival, original)
        reports = self.reports()
        reports[4]["fits"][1]["converged"] = False
        with self.assertRaises(RuntimeError):
            self.summarize(reports)
        reports = self.reports()
        reports[4]["participant_scores"]["p000"]["M0"] += 1e-13
        with self.assertRaises(ValueError):
            self.summarize(reports)
        self.assertIs(summary.menu.validate_rival, original)

    def test_cli_compute_original_spec_and_fresh_output_guards_precede_any_reads(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            output = Path(directory) / "summary.json"
            args = ["--implementation", str(summary.menu.IMPLEMENTATION), "--output", str(output)]
            with patch.object(summary, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                summary.main(args)
            with patch.object(summary, "require_compute"):
                with self.assertRaises(ValueError):
                    summary.main(["--implementation", str(Path(directory) / "wrong.json"), "--output", str(output)])
                output.write_text("preserved")
                with self.assertRaises(FileExistsError):
                    summary.main(args)

    def test_cli_reads_only_six_selected_reports_and_preserved_null_source(self):
        reports = self.reports()
        truth_path = summary.menu.null.SOURCE / "generating_parameters.json"
        events_path = summary.menu.null.SOURCE / "simulated_events.csv.gz"
        allowed = {path: json.dumps(report) for path, report in zip(summary.INPUT_REPORTS, reports)}
        allowed[summary.menu.IMPLEMENTATION] = json.dumps(self.spec)
        allowed[truth_path] = json.dumps(self.truth)
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            return allowed[path]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            with patch.object(summary, "require_compute"), patch.object(Path, "read_text", reader), \
                    patch.object(summary.menu.null.shared, "load_cohort", return_value=self.people) as loader, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(summary.main(["--implementation", str(summary.menu.IMPLEMENTATION), "--output", str(output)]), 0)
            result = json.loads(output.read_text())
        self.assertEqual(reads, [summary.menu.IMPLEMENTATION, *summary.INPUT_REPORTS, truth_path])
        loader.assert_called_once_with(events_path)
        self.assertEqual(result["input_reports"], [str(path) for path in summary.INPUT_REPORTS])
        self.assertEqual(result["repaired_mvc_report"], str(summary.REPAIR_REPORT))
        self.assertEqual(len(result["nested_comparisons"]), 9)


if __name__ == "__main__":
    unittest.main()
