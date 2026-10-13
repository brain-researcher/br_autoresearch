"""Five-report adapter boundaries; unchanged contrast thresholds are not retested."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import summarize_mz_saved_replicate_menu as core
from test_mz_group_recovery import PEOPLE, report_fixture


def reports_fixture(spec, index=21):
    base = report_fixture(spec, index=index)
    base.update(started_utc="2026-10-04T23:00:00+00:00", design_source=str(core.DESIGN_SOURCE),
                design_fields=list(core.DESIGN_FIELDS))
    values = {"M0": -1., "MV": -.98, "MZ": -.96, "MC": -.97, "MVZ": -.93, "MVC": -.935}
    base["participant_scores"] = {person: {"M0": values["M0"], "MZ": values["MZ"],
        "version": "EI" if i < 54 else "ER"} for i, person in enumerate(PEOPLE)}
    reports = [base]
    for model in core.RIVALS:
        names = ["kappa", "log_a", "nondecision_logit"] + {"MC": ["criterion"], "MV": ["eta"],
            "MVZ": ["eta", "rho"], "MVC": ["eta", "criterion"]}[model]
        center = [spec["parameters"][name]["population_center"] for name in names]
        fits = []
        for train, held in core.HALVES:
            fits.append({"model": model, "rival": "fixed", "training_runs": list(train), "scoring_runs": list(held),
                "participants": PEOPLE[:], "parameter_names": names[:], "parameters": [center[:] for _ in PEOPLE],
                "global_mean": center[:], "version_means": [center[:], center[:]],
                "version_standard_deviations": [[.3] * len(names), [.3] * len(names)],
                "version_order": ["EI", "ER"], "inference": spec["method"],
                "converged": True, "subject_optimizer_failures": [], "objective_history": [100., 100.],
                "support_contacts": [], "group_interval_coverage_qualified": False,
                "individual_rank_recovery_qualified": False})
        reports.append({"analysis_id": "behavioral_mechanism_kill_test",
            "phase": "MZ_SESOI_saved_replicate_rival_diagnostics", "generating_model": "MZ", "fitted_model": model,
            "replicate_index": index, "seed": core.cell_seed("MZ", index), "implementation": spec,
            "source_dir": str(core.source_directory(index)), "source_finished_utc": base["finished_utc"],
            "started_utc": "2026-10-05T00:01:00+00:00", "finished_utc": "2026-10-05T00:02:00+00:00",
            "participants": 108, "runs": 432, "trials": 27648, "status": "rival_diagnostics_completed",
            "rival_diagnostics_complete": True, "saved_scores_replayed": True,
            "saved_score_replay_max_absolute_discrepancy": 0., "saved_score_replay_absolute_tolerance": 1e-12,
            "simulation_regenerated": False, "saved_M0_MZ_refitted": False, "observed_choices_or_RTs_used": False,
            "completed_rivals": [model], "scored_primary_directions":
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in core.HALVES],
            "fits": fits, "family_selection": "not_evaluated", "confusion_rate": "not_evaluated",
            "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated", "full_gate": "not_evaluated",
            "full_recovery_pass": None, "episode_terminal": False,
            "participant_scores": {person: dict(row, **{model: values[model]}) for person, row in base["participant_scores"].items()}})
    return reports


class SavedReplicateMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((CODE / "behavioral_implementation.json").read_text())

    def summarize(self, reports, index=21):
        return core.build_summary(*reports, self.spec, index)

    def test_valid_five_report_adapter_reuses_exact_nine_edges_without_a_family_verdict(self):
        reports = reports_fixture(self.spec)
        before = deepcopy(reports)
        with patch.object(core.fixed, "nested_contrast", wraps=core.fixed.nested_contrast) as contrasts:
            result = self.summarize(reports)
        self.assertEqual(contrasts.call_count, 9)
        self.assertEqual([call.args[1:] for call in contrasts.call_args_list], list(core.fixed.EDGES))
        self.assertEqual(set(result["nested_comparisons"]), {f"{model}-{baseline}" for model, baseline in core.fixed.EDGES})
        self.assertEqual(result["replicate_index"], 21)
        self.assertEqual(result["seed"], 910031 + 10007 * 21)
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MVZ", "MVC"])
        self.assertTrue(result["multiple_incomparable_eligible_models"])
        self.assertEqual(result["model_menu"], list(core.fixed.MODELS))
        for field in ("family_selection", "confusion_rate", "group_recovery", "interval_coverage", "full_gate"):
            self.assertEqual(result[field], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        self.assertFalse(result["episode_terminal"])
        self.assertNotIn("participant_scores", result)
        self.assertEqual(reports, before)

    def test_actual_fixed_array_indices_retained_without_zero_relabelling(self):
        for index in (1, 20, 21, 199):
            with self.subTest(index=index):
                result = self.summarize(reports_fixture(self.spec, index), index)
                self.assertEqual(result["replicate_index"], index)
                self.assertEqual(result["source_dir"], str(core.source_directory(index)))
        with self.assertRaises(ValueError):
            self.summarize(reports_fixture(self.spec), 0)

    def test_mismatched_index_seed_source_implementation_or_source_attempt_refused(self):
        changes = ({"replicate_index": 20}, {"seed": 910031},
                   {"source_dir": str(core.source_directory(20))},
                   {"implementation": dict(self.spec, policy_version=99)},
                   {"source_finished_utc": "2026-10-05T00:00:01+00:00"})
        for change in changes:
            reports = reports_fixture(self.spec)
            reports[1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.summarize(reports)
        reports = reports_fixture(self.spec)
        reports[0].update(replicate_index=20, seed=core.cell_seed("MZ", 20))
        with self.assertRaises(ValueError):
            self.summarize(reports)

    def test_even_sub_tolerance_baseline_change_is_not_exact_saved_score_reuse(self):
        for model in ("M0", "MZ"):
            reports = reports_fixture(self.spec)
            reports[2]["participant_scores"][PEOPLE[0]][model] += 5e-13
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "exact saved"):
                self.summarize(reports)

    def test_changed_cohort_and_balanced_version_swap_are_refused(self):
        reports = reports_fixture(self.spec)
        scores = reports[3]["participant_scores"]
        scores["different-person"] = scores.pop(PEOPLE[0])
        with self.assertRaisesRegex(ValueError, "identities differ"):
            self.summarize(reports)
        reports = reports_fixture(self.spec)
        scores = reports[3]["participant_scores"]
        scores[PEOPLE[0]]["version"], scores[PEOPLE[54]]["version"] = "ER", "EI"
        with self.assertRaisesRegex(ValueError, "task versions differ"):
            self.summarize(reports)

    def test_technical_live_and_missing_finished_reports_never_enter_edges(self):
        changes = ({"status": "technical_rival_failure", "rival_diagnostics_complete": False},
                   {"status": "running"}, {"finished_utc": ""}, {"rival_diagnostics_complete": False})
        for change in changes:
            reports = reports_fixture(self.spec)
            reports[4].update(change)
            with self.subTest(change=change), patch.object(core.fixed, "nested_contrast") as contrasts, self.assertRaises(ValueError):
                self.summarize(reports)
            contrasts.assert_not_called()

    def test_missing_fit_scored_direction_wrong_model_or_five_report_slot_refused(self):
        for kind in ("fit", "direction", "model", "swapped", "missing"):
            reports = reports_fixture(self.spec)
            if kind == "fit":
                reports[1]["fits"].pop()
            elif kind == "direction":
                reports[1]["scored_primary_directions"].pop()
            elif kind == "model":
                reports[1]["fits"][0]["model"] = "MV"
            elif kind == "swapped":
                reports[1], reports[2] = reports[2], reports[1]
            else:
                reports[1] = {}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.summarize(reports)

    def test_failed_replay_and_invented_claims_are_refused(self):
        changes = ({"saved_scores_replayed": False}, {"saved_score_replay_max_absolute_discrepancy": 2e-12},
                   {"saved_score_replay_absolute_tolerance": 1e-6}, {"simulation_regenerated": True},
                   {"saved_M0_MZ_refitted": True}, {"family_selection": "MZ"}, {"interval_coverage": "passed"})
        for change in changes:
            reports = reports_fixture(self.spec)
            reports[2].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.summarize(reports)

    def test_explicit_cli_requires_five_distinct_paths_before_reading_inputs(self):
        base = core.source_directory(21) / "single_process_recovery.json"
        args = ["--base-report", str(base), "--mc-report", "same.json", "--mv-report", "same.json",
                "--mvz-report", "mvz.json", "--mvc-report", "mvc.json", "--replicate-index", "21",
                "--implementation", str(core.IMPLEMENTATION), "--output", "must-not-be-created.json"]
        with patch.object(core, "require_compute"), patch.object(Path, "read_text") as reader, \
                self.assertRaisesRegex(ValueError, "five distinct"):
            core.main(args)
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
