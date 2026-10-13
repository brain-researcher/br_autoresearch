"""Focused fixed reverse-completion menu admission; execute on compute only."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_mvc_repaired_menu as menu
import run_mz_mvc_convergence_pilot as pilot
import test_mz_mvc_convergence_pilot as fixtures


class RepairedMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This class/fixture API is also used by the prepared convergence and
        # reverse-runner tests. Keep it independent of those test modules.
        fixtures.ConvergencePilotTests.setUpClass()
        cls.pilot_fixture = fixtures.ConvergencePilotTests("test_runtime_copy_changes_only_the_recorded_sweep_ceiling")
        cls.fixture, cls.spec = cls.pilot_fixture.fixture, cls.pilot_fixture.spec

    def reports(self):
        # Synthetic report fixtures only. reports()[0] remains the original500
        # source fixture consumed by the other prepared runner test modules.
        base = deepcopy(self.fixture.source())
        flags = {name: True for name in ("mean_gain_at_least_0p01", "mean_gain_greater_than_two_participant_SE",
                 "positive_in_both_task_versions", "positive_participant_median", "participant_median_95CI_lower_positive")}
        base.update(phase="MZ_SESOI_fitted_recovery_component", generating_model="MZ", replicate_index=42,
                    seed=menu.saved.cell_seed("MZ", 42), planned_cell_replicates=200, SESOI_level=1.,
                    focal_generating_truth=pilot.saved.focal_truth("MZ"),
                    finished_utc=menu.repair.SOURCE_FINISHED_UTC,
                    started_utc="2026-10-05T00:40:00+00:00",
                    full_behavioral_gate="not_evaluated", full_recovery_pass=None,
                    valid_recovery_component=True, counts_as_curtailment_failure=False,
                    status="necessary_primary_comparison_completed",
                    comparison={"requirements": flags, "focal_model": "MZ", "core_family_selection_pass": True,
                                "counts_as_curtailment_failure": False, "necessary_primary_comparison_pass": True,
                                "richer_primary_diagnostic_pass": True, "full_recovery_pass": None})
        rivals = []
        for model in menu.saved.RIVALS:
            rivals.append({
                "analysis_id": "behavioral_mechanism_kill_test", "phase": "MZ_SESOI_saved_replicate_rival_diagnostics",
                "generating_model": "MZ", "fitted_model": model, "replicate_index": 42,
                "seed": menu.saved.cell_seed("MZ", 42), "implementation": deepcopy(self.spec),
                "source_dir": str(pilot.saved.source_directory(42)), "source_finished_utc": base["finished_utc"],
                "started_utc": "2026-10-05T01:00:00+00:00", "finished_utc": "2026-10-05T02:00:00+00:00",
                "participants": 108, "runs": 432, "trials": 27648,
                "status": "rival_diagnostics_completed", "rival_diagnostics_complete": True,
                "saved_scores_replayed": True, "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
                "observed_choices_or_RTs_used": False, "completed_rivals": [model],
                "scored_primary_directions": [{"training_runs": list(train), "scoring_runs": list(held)}
                                              for train, held in menu.saved.HALVES],
                "saved_score_replay_max_absolute_discrepancy": 0., "saved_score_replay_absolute_tolerance": 1e-12,
                **{field: "not_evaluated" for field in menu.CLAIM_FIELDS},
                "full_recovery_pass": None, "episode_terminal": False,
                "participant_scores": {person: {**row, model: -2.05} for person, row in base["participant_scores"].items()},
                "fits": [self.fixture.fit(model, train, held) for train, held in menu.saved.HALVES],
            })
        failed = deepcopy(self.pilot_fixture.failed())
        failed.update(job_id=menu.repair.ORIGINAL_JOB_ID, started_utc="2026-10-05T04:35:00+00:00",
                      finished_utc=menu.repair.ORIGINAL_FINISHED_UTC, source_finished_utc=base["finished_utc"],
                      participants=108, runs=432, trials=27648, error=menu.repair.NONCONVERGENCE,
                      saved_scores_replayed=True, saved_score_replay_absolute_tolerance=1e-12,
                      saved_score_replay_max_absolute_discrepancy=0., simulation_regenerated=False,
                      saved_M0_MZ_refitted=False, observed_choices_or_RTs_used=False,
                      full_recovery_pass=None, episode_terminal=False, diagnostics={},
                      **{field: "not_evaluated" for field in menu.CLAIM_FIELDS})
        failed["fits"][0]["participants"] = sorted(self.fixture.people)
        first_runtime, prior_metadata = pilot.repair_context(self.spec, failed)
        first = self.fixture.fit("MVC", *menu.saved.HALVES[0])
        first.update(objective_history=[float(value) for value in range(666, 1, -1)] + [2.],
                     elapsed_seconds=2156.1116262879223)
        exhausted = self.fixture.fit("MVC", *menu.saved.HALVES[1], converged=False)
        exhausted.update(objective_history=[float(value) for value in range(1000, 0, -1)],
                         elapsed_seconds=3234.350768652046)
        previous = deepcopy(failed)
        previous.update(phase=pilot.PHASE, job_id=menu.repair.PILOT_JOB_ID,
                        started_utc="2026-10-05T05:21:42.783354+00:00", finished_utc=menu.repair.PILOT_FINISHED_UTC,
                        runtime_implementation=first_runtime, numerical_repair=prior_metadata,
                        internal_seconds=5400, selected_replicates=[42], retry_wave_selected=False,
                        fits=[first, exhausted], elapsed_seconds=5392.607597094029,
                        cumulative_process_elapsed_seconds=failed["elapsed_seconds"] + 5392.607597094029,
                        scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
        first_runtime, runtime, preserved_first, metadata = menu.repair.repair_context(
            self.spec, failed, previous, base, self.fixture.people)
        preserved_first["elapsed_seconds"] = .002
        fresh = self.fixture.fit("MVC", *menu.saved.HALVES[1])
        fresh.update(objective_history=[float(value) for value in range(1200, 1, -1)] + [2.], elapsed_seconds=4000.)
        metadata.update(first_half_materialization_elapsed_seconds=.002,
                        first_half_rescore_elapsed_seconds=.01, first_half_rescored=True)
        mvc = rivals[-1]
        mvc.update(phase=menu.repair.PHASE, job_id=menu.COMPLETION_JOB_ID,
                   started_utc="2026-10-05T07:10:00+00:00", finished_utc="2026-10-05T08:20:00+00:00",
                   first_half_runtime_implementation=first_runtime, runtime_implementation=runtime,
                   numerical_repair=metadata, selected_replicates=[42], selected_models=["MVC"],
                   wave_selected=False, retry_wave_selected=False, automatic_retry=False, full_menu_wave_selected=False,
                   repair_context_validated=True, first_half_fit_reused=True, fresh_reverse_fit_called=True,
                   baseline_validation_maximum_sweeps=500, internal_seconds=5400,
                   per_half_provenance=[
                       {"training_runs": [1, 4], "scoring_runs": [2, 3], "fit_source": str(menu.repair.PILOT_REPORT),
                        "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": 1000},
                       {"training_runs": [2, 3], "scoring_runs": [1, 4], "fit_source": "current_completion",
                        "action": "fresh_fit", "validation_maximum_sweeps": 1500}],
                   fits=[preserved_first, fresh], elapsed_seconds=4200.,
                   cumulative_process_elapsed_seconds=previous["cumulative_process_elapsed_seconds"] + 4200.)
        return [base, *rivals, failed, previous]

    def summarize(self, reports=None):
        return menu.build_summary(*(self.reports() if reports is None else reports), self.spec)

    def test_existing_nine_contrasts_and_original_validation_remain_with_explicit_runtime_mvc(self):
        reports = self.reports()
        preserved = deepcopy(reports)
        original_validator = menu.saved.validate_rival
        with patch.object(menu.saved.fixed, "validate_fit", wraps=menu.saved.fixed.validate_fit) as validate, \
                patch.object(menu.saved.fixed, "nested_contrast", wraps=menu.saved.fixed.nested_contrast) as contrast:
            result = self.summarize(reports)
        self.assertIs(menu.saved.validate_rival, original_validator)
        self.assertEqual(reports, preserved)
        self.assertEqual(contrast.call_count, 9)
        self.assertEqual([(call.args[1], call.args[4]["optimizer"]["maximum_sweeps"]) for call in validate.call_args_list],
                         [("M0", 500), ("MZ", 500), ("M0", 500), ("MZ", 500),
                          ("MC", 500), ("MC", 500), ("MV", 500), ("MV", 500),
                          ("MVZ", 500), ("MVZ", 500), ("MVC", 1000), ("MVC", 1500)])
        self.assertEqual(result["phase"], menu.PHASE)
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ"])
        self.assertEqual(result["implementation"], self.spec)
        for field in ("first_half_runtime_implementation", "runtime_implementation", "numerical_repair",
                      "per_half_provenance", "cumulative_process_elapsed_seconds"):
            self.assertEqual(result[field], reports[4][field])
        for field in menu.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        self.assertFalse(result["episode_terminal"])
        self.assertFalse(result["retry_wave_selected"])
        self.assertEqual(result["fixed_cohort_count"], 200)
        self.assertEqual(result["selected_replicates"], [42])

    def test_original_menu_rejects_completion_and_failed_pilot_cannot_supply_the_new_menu(self):
        base, mc, mv, mvz, mvc, failed, previous = self.reports()
        baseline = menu.saved.validate_base(base, self.spec, 42)
        with self.assertRaises(ValueError):
            menu.saved.validate_rival(mvc, "MVC", base, self.spec, baseline, 42)
        with self.assertRaises(ValueError):
            menu.saved.fixed.fit_panel(mvc, [("MVC", train, held) for train, held in menu.saved.HALVES], self.spec, baseline)
        reports = self.reports()
        reports[4] = deepcopy(previous)
        with self.assertRaises(ValueError):
            self.summarize(reports)
        reports = self.reports()
        reports[1]["fits"][0]["objective_history"] = deepcopy(mvc["fits"][0]["objective_history"])
        with self.assertRaises(ValueError):
            self.summarize(reports)

    def test_exact_new_attempt_both_prior_attempts_and_repair_cost_metadata_are_required(self):
        changes = (("job_id", "46587906"), ("replicate_index", 43), ("seed", 1),
                   ("phase", pilot.PHASE), ("source_dir", "/another/source"),
                   ("source_finished_utc", "2026-10-05T01:00:00+00:00"),
                   ("first_half_runtime_implementation", self.spec), ("runtime_implementation", self.spec),
                   ("numerical_repair", {}), ("cumulative_process_elapsed_seconds", 4200.),
                   ("per_half_provenance", []), ("baseline_validation_maximum_sweeps", 1000))
        for field, value in changes:
            reports = self.reports()
            reports[4][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)
        for mutate in (lambda reports: reports[4]["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
                       lambda reports: reports[4]["numerical_repair"].update(nonconverged_checkpoint_state_reused=True),
                       lambda reports: reports[4]["numerical_repair"].update(first_half_rescored=False),
                       lambda reports: reports[4]["numerical_repair"].update(first_half_materialization_elapsed_seconds=1.),
                       lambda reports: reports[4]["numerical_repair"].update(first_half_rescore_elapsed_seconds=-1.),
                       lambda reports: reports[5].update(job_id="46572753"),
                       lambda reports: reports[5].update(source_finished_utc="different"),
                       lambda reports: reports[6].update(job_id="46577370"),
                       lambda reports: reports[6].update(error="RuntimeError: Interrupted")):
            reports = self.reports()
            mutate(reports)
            with self.assertRaises(ValueError):
                self.summarize(reports)

    def test_incomplete_completion_failed_fit_or_changed_preserved_first_is_rejected_and_adapter_restored(self):
        original_validator = menu.saved.validate_rival
        for field, value in (("status", "running"), ("rival_diagnostics_complete", False),
                             ("completed_rivals", []), ("scored_primary_directions", []), ("fits", []),
                             ("repair_context_validated", False), ("first_half_fit_reused", False),
                             ("fresh_reverse_fit_called", False)):
            reports = self.reports()
            reports[4][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)
            self.assertIs(menu.saved.validate_rival, original_validator)
        for field in ("parameters", "objective_history"):
            reports = self.reports()
            if field == "parameters":
                reports[4]["fits"][0][field][0][0] += .001
            else:
                reports[4]["fits"][0][field][0] += .001
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)
        reports = self.reports()
        reports[4]["fits"][1]["converged"] = False
        with self.assertRaises(RuntimeError):
            self.summarize(reports)
        self.assertIs(menu.saved.validate_rival, original_validator)

    def test_replay_last_bits_and_diagnostic_only_claim_boundary(self):
        changes = [("saved_scores_replayed", False), ("saved_score_replay_max_absolute_discrepancy", 2e-12),
                   ("saved_score_replay_absolute_tolerance", 1e-10), ("simulation_regenerated", True),
                   ("saved_M0_MZ_refitted", True), ("retry_wave_selected", True), ("selected_replicates", [42, 43]),
                   ("automatic_retry", True), ("wave_selected", True), ("full_menu_wave_selected", True),
                   ("full_recovery_pass", True)] + [(field, "evaluated") for field in menu.CLAIM_FIELDS]
        for field, value in changes:
            reports = self.reports()
            reports[4][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)
        reports = self.reports()
        reports[4]["participant_scores"]["p000"]["M0"] += 1e-13
        with self.assertRaises(ValueError):
            self.summarize(reports)

    def test_cli_compute_exact_implementation_and_fresh_output_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(Path, "read_text", side_effect=AssertionError("No payload read")):
            output = Path(directory) / "summary.json"
            arguments = ["--implementation", str(menu.IMPLEMENTATION), "--output", str(output)]
            with patch.object(menu, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                menu.main(arguments)
            with patch.object(menu, "require_compute"):
                with self.assertRaises(ValueError):
                    menu.main(["--implementation", str(Path(directory) / "other.json"), "--output", str(output)])
                output.write_text("preserved")
                with self.assertRaises(FileExistsError):
                    menu.main(arguments)

    def test_cli_reads_only_seven_fixed_reports_and_preserves_explicit_mixed_provenance(self):
        reports = self.reports()
        allowed = {path: json.dumps(report) for path, report in zip(menu.INPUT_REPORTS, reports)}
        allowed[menu.IMPLEMENTATION] = json.dumps(self.spec)
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            return allowed[path]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            with patch.object(menu, "require_compute"), patch.object(Path, "read_text", reader), redirect_stdout(io.StringIO()):
                self.assertEqual(menu.main(["--implementation", str(menu.IMPLEMENTATION), "--output", str(output)]), 0)
            result = json.loads(output.read_text())
        self.assertEqual(reads, [menu.IMPLEMENTATION, *menu.INPUT_REPORTS])
        self.assertEqual(result["input_reports"], [str(path) for path in menu.INPUT_REPORTS])
        self.assertEqual(result["mvc_completion_report"], str(menu.COMPLETION_REPORT))
        self.assertEqual(result["failed_mvc_pilot_report"], str(menu.repair.PILOT_REPORT))
        self.assertEqual(result["original_failed_mvc_report"], str(menu.repair.ORIGINAL_REPORT))
        self.assertEqual(len(result["nested_comparisons"]), 9)


if __name__ == "__main__":
    unittest.main()
