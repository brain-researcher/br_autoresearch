"""Focused single-null-menu identity, completeness and claim tests; use compute."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_null_menu as menu
import test_mz_null_rival as fixtures


class NullMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing synthetic source/fit fixtures, without running its tests.
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.truth, cls.people = cls.fixture.spec, cls.fixture.truth, cls.fixture.people

    def reports(self):
        base = self.fixture.source()
        rivals = []
        for model in menu.RIVALS:
            rivals.append({
                "analysis_id": "behavioral_mechanism_kill_test", "phase": "M0_null_saved_representative_rival_diagnostics",
                "generating_model": "M0", "fitted_model": model, "replicate_index": 0, "seed": 1300037,
                "implementation": self.spec, "source_dir": str(menu.null.SOURCE),
                "source_report": menu.null.SOURCE_REPORT, "source_job_id": "46574266",
                "source_finished_utc": base["finished_utc"],
                "job_id": menu.SELECTED_JOB_IDS.get(model, "60000000"),
                "started_utc": "2026-10-05T05:00:00+00:00", "finished_utc": "2026-10-05T06:00:00+00:00",
                "participants": 108, "runs": 432, "trials": 27648,
                "status": "rival_diagnostics_completed", "rival_diagnostics_complete": True,
                "saved_scores_replayed": True, "simulation_regenerated": False, "saved_M0_MZ_refitted": False,
                "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
                "completed_rivals": [model], "scored_primary_directions": deepcopy(base["scored_primary_directions"]),
                "saved_score_replay_max_absolute_discrepancy": 0., "saved_score_replay_absolute_tolerance": 1e-12,
                **{field: "not_evaluated" for field in menu.null.CLAIM_FIELDS},
                "full_recovery_pass": None, "episode_terminal": False, "wave_selected": False,
                "participant_scores": {person: {**row, model: -2.05} for person, row in base["participant_scores"].items()},
                "fits": [self.fixture.fit(model, train, held) for train, held in menu.HALVES],
            })
        return [base, *rivals]

    def summarize(self, reports=None, truth=None):
        return menu.build_summary(*(self.reports() if reports is None else reports), self.spec,
                                  self.truth if truth is None else truth, self.people)

    def test_complete_six_model_panel_retains_null_and_no_claim_even_when_mz_unique(self):
        reports = self.reports()
        result = self.summarize(reports)
        self.assertEqual(result["generating_model"], "M0")
        self.assertEqual((result["seed"], result["replicate_index"], result["SESOI_level"]), (1300037, 0, 0.))
        self.assertEqual(result["model_menu"], list(menu.fixed.MODELS))
        self.assertEqual(set(result["nested_comparisons"]), {f"{model}-{baseline}" for model, baseline in menu.fixed.EDGES})
        self.assertEqual(len(result["nested_comparisons"]), 9)
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ"])
        for person, baseline in reports[0]["participant_scores"].items():
            self.assertEqual({key: result["participant_scores"][person][key] for key in baseline}, baseline)
            for model in menu.RIVALS:
                self.assertEqual(result["participant_scores"][person][model], -2.05)
        for field in menu.null.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        self.assertFalse(result["episode_terminal"])
        self.assertFalse(result["wave_selected"])
        self.assertEqual(result["selected_replicates"], [0])

    def test_base_cannot_be_relabelled_from_mz_or_generated_with_nonnull_truth(self):
        reports = self.reports()
        reports[0]["generating_model"] = "MZ"
        with self.assertRaises(ValueError):
            self.summarize(reports)
        truth = deepcopy(self.truth)
        truth["p000"]["rho"] = -.2
        with self.assertRaises(ValueError):
            self.summarize(truth=truth)

    def test_rival_source_model_and_selected_attempt_identity_must_match(self):
        changes = (("generating_model", "MZ"), ("seed", 910031), ("replicate_index", 1),
                   ("source_dir", "/different/cohort"), ("source_report", "single_process_recovery.json"),
                   ("source_job_id", "46574267"), ("source_finished_utc", "2026-10-05T00:00:00+00:00"),
                   ("fitted_model", "MV"), ("phase", "MZ_SESOI_saved_replicate_rival_diagnostics"))
        for field, value in changes:
            reports = self.reports()
            reports[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)
        for position in (3, 4):
            reports = self.reports()
            reports[position]["job_id"] = "60000001"
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.summarize(reports)

    def test_missing_or_incomplete_rival_cannot_produce_a_menu(self):
        for position in range(1, 5):
            reports = self.reports()
            reports[position] = None
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.summarize(reports)
        for field, value in (("status", "running"), ("rival_diagnostics_complete", False),
                             ("completed_rivals", []), ("scored_primary_directions", []),
                             ("fits", []), ("participant_scores", {})):
            reports = self.reports()
            reports[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)

    def test_baseline_last_bits_and_task_versions_must_be_identical(self):
        reports = self.reports()
        reports[2]["participant_scores"]["p000"]["M0"] += 1e-13
        with self.assertRaises(ValueError):
            self.summarize(reports)
        reports = self.reports()
        reports[2]["participant_scores"]["p000"]["version"] = "ER"
        reports[2]["participant_scores"]["p054"]["version"] = "EI"
        with self.assertRaises(ValueError):
            self.summarize(reports)

    def test_replay_evidence_and_diagnostic_only_claims_are_required(self):
        changes = [("saved_scores_replayed", False), ("saved_score_replay_max_absolute_discrepancy", 2e-12),
                   ("saved_score_replay_absolute_tolerance", 1e-10), ("simulation_regenerated", True),
                   ("saved_M0_MZ_refitted", True), ("wave_selected", True), ("full_recovery_pass", True)]
        changes += [(field, "evaluated") for field in menu.null.CLAIM_FIELDS]
        for field, value in changes:
            reports = self.reports()
            reports[4][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize(reports)

    def cli(self, directory):
        paths = [menu.BASE_REPORT, *[Path(directory) / f"{model}.json" for model in menu.RIVALS]]
        args = []
        for flag, path in zip(("--base-report", "--mc-report", "--mv-report", "--mvz-report", "--mvc-report"), paths):
            args.extend((flag, str(path)))
        args.extend(("--implementation", str(menu.IMPLEMENTATION), "--output", str(Path(directory) / "summary.json")))
        return args, paths

    def test_cli_rejects_changed_base_duplicate_inputs_and_existing_output_before_reads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(menu, "require_compute"), \
                patch.object(Path, "read_text", side_effect=AssertionError("No input read is allowed")):
            args, paths = self.cli(directory)
            changed = list(args)
            changed[1] = str(Path(directory) / "other-base.json")
            with self.assertRaises(ValueError):
                menu.main(changed)
            changed = list(args)
            changed[3] = changed[5]
            with self.assertRaises(ValueError):
                menu.main(changed)
            Path(args[-1]).write_text("preserved")
            with self.assertRaises(FileExistsError):
                menu.main(args)

    def test_cli_writes_complete_summary_from_only_explicit_and_fixed_source_paths(self):
        reports = self.reports()
        with tempfile.TemporaryDirectory() as directory:
            args, paths = self.cli(directory)
            allowed = {path: json.dumps(report) for path, report in zip(paths, reports)}
            allowed[menu.IMPLEMENTATION] = json.dumps(self.spec)
            allowed[menu.null.SOURCE / "generating_parameters.json"] = json.dumps(self.truth)
            reads = []

            def reader(path, *args, **kwargs):
                reads.append(path)
                return allowed[path]

            with patch.object(menu, "require_compute"), patch.object(Path, "read_text", reader), \
                    patch.object(menu.null.shared, "load_cohort", return_value=self.people) as loader, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(menu.main(args), 0)
            self.assertEqual(set(reads), set(allowed))
            loader.assert_called_once_with(menu.null.SOURCE / "simulated_events.csv.gz")
            result = json.loads(Path(args[-1]).read_text())
            self.assertEqual(result["input_reports"], [str(path.resolve()) for path in paths])
            self.assertEqual(result["generating_model"], "M0")
            self.assertEqual(len(result["nested_comparisons"]), 9)
            self.assertEqual(result["full_menu_false_selection"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
