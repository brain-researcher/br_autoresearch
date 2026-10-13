"""Focused null-source, replay, claim-boundary and checkpoint tests; use compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hierarchical_behavior import parameter_names
from test_mz_null_representative import full_design, THREADS
import run_mz_null_rival as rival


class NullRivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(rival.IMPLEMENTATION.read_text())
        cls.people = full_design()
        for rows in cls.people.values():
            for row in rows:
                row.update(choice="1", no_response="0", strength="-1", response_time=".5",
                           response_category="simulated_accept")
        cls.truth = {person: {**{name: cls.spec["parameters"][name]["population_center"]
                               for name in ("kappa", "log_a", "nondecision_logit")},
                            "eta": 0., "rho": 0., "criterion": 0.} for person in cls.people}

    def fit(self, model, train=(1, 4), held=(2, 3), converged=True):
        names = parameter_names(model, "fixed")
        centers = [self.spec["parameters"][name]["population_center"] for name in names]
        scales = [self.spec["parameters"][name]["individual_scale_center"] for name in names]
        return {"model": model, "rival": "fixed", "training_runs": list(train), "scoring_runs": list(held),
                "participants": sorted(self.people), "parameter_names": names, "version_order": ["EI", "ER"],
                "inference": self.spec["method"], "converged": converged, "subject_optimizer_failures": [],
                "parameters": [list(centers) for _ in self.people], "global_mean": centers,
                "version_means": [list(centers), list(centers)],
                "version_standard_deviations": [list(scales), list(scales)],
                "objective_history": [100., 100.], "support_contacts": []}

    def source(self):
        return {
            "analysis_id": "behavioral_mechanism_kill_test", "phase": "M0_null_MZ_pairwise_representative",
            "generating_model": "M0", "fitted_models": ["M0", "MZ"], "seed": 1300037,
            "replicate_index": 0, "job_id": "46574266", "SESOI_level": 0.,
            "focal_generating_truth": {"eta": 0., "rho": 0., "criterion": 0.}, "implementation": self.spec,
            "status": "pairwise_diagnostic_completed", "pairwise_diagnostic_complete": True,
            "scored_primary_directions": [{"training_runs": list(train), "scoring_runs": list(held)}
                                          for train, held in rival.HALVES],
            "participants": 108, "runs": 432, "trials": 27648,
            "started_utc": "2026-10-05T04:45:00+00:00", "finished_utc": "2026-10-05T04:55:00+00:00",
            "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
            "design_fields": list(rival.DESIGN_FIELDS), "design_source": str(rival.DESIGN_SOURCE),
            "nuisance_generation": "recorded_center_and_individual_scale_truncated_normal_as_profile01",
            "deadline_missingness": "generated_from_frozen_likelihood_4s_deadline_and_identical_contaminant_channel",
            "focal_effect_between_participant_variance": 0.,
            **{field: "not_evaluated" for field in rival.CLAIM_FIELDS},
            "full_recovery_pass": None, "episode_terminal": False, "selected_replicates": [0], "wave_selected": False,
            "simulated_deadline_misses": 0,
            "participant_scores": {person: {"M0": -2. + 5e-13, "MZ": -1.98 - 5e-13,
                                            "version": rows[0]["task_version"]} for person, rows in self.people.items()},
            "fits": [self.fit(model, train, held) for train, held in rival.HALVES for model in ("M0", "MZ")],
        }

    def arguments(self, output, model="MVZ", **changes):
        values = dict(model=model, source_dir=rival.SOURCE, implementation=rival.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=3300)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_only_completed_fixed_null_source_is_accepted(self):
        source = self.source()
        rival.validate_source(source, self.truth, self.people, self.spec)
        for field, value in (("generating_model", "MZ"), ("seed", 910031), ("replicate_index", 1),
                             ("job_id", "46574267"), ("SESOI_level", 1.),
                             ("phase", "MZ_SESOI_fitted_recovery_component"),
                             ("status", "running"), ("pairwise_diagnostic_complete", False),
                             ("scored_primary_directions", [])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                rival.validate_source({**source, field: value}, self.truth, self.people, self.spec)

    def test_nonnull_truth_or_changed_design_scope_is_rejected(self):
        source = self.source()
        for focal in ("eta", "rho", "criterion"):
            truth = deepcopy(self.truth)
            truth["p000"][focal] = .05
            with self.subTest(focal=focal), self.assertRaises(ValueError):
                rival.validate_source(source, truth, self.people, self.spec)
        for field, value in (("observed_choices_or_RTs_used", True), ("old_synthetic_responses_used", True),
                             ("design_source", "/not-the-fixed-design.csv.gz"), ("design_fields", []),
                             ("participants", 107), ("simulated_deadline_misses", 1),
                             ("nuisance_generation", "different_regime")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                rival.validate_source({**source, field: value}, self.truth, self.people, self.spec)

    def test_source_cannot_supply_a_claim_or_select_a_null_wave(self):
        source = self.source()
        changes = [(field, "evaluated") for field in rival.CLAIM_FIELDS]
        changes += [("full_recovery_pass", True), ("episode_terminal", True),
                    ("wave_selected", True), ("selected_replicates", [0, 1])]
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                rival.validate_source({**source, field: value}, self.truth, self.people, self.spec)

    def test_complete_baseline_panel_and_matching_participants_are_required(self):
        source = self.source()
        with self.assertRaises(ValueError):
            rival.validate_source({**source, "fits": source["fits"][:3]}, self.truth, self.people, self.spec)
        changed = deepcopy(source)
        changed["participant_scores"]["p000"]["version"] = "ER"
        changed["participant_scores"]["p054"]["version"] = "EI"
        with self.assertRaises(ValueError):
            rival.validate_source(changed, self.truth, self.people, self.spec)
        changed = deepcopy(source)
        changed["implementation"]["contaminant"]["fraction"] = .03
        with self.assertRaises(ValueError):
            rival.validate_source(changed, self.truth, self.people, self.spec)

    def test_scope_cpu_and_time_guards_precede_any_source_read(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(rival, "require_compute"), \
                patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read is allowed")):
            for changes in ({"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"},
                            {"model": "M0"}, {"max_seconds": 0}, {"max_seconds": 5401}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "new", **changes))
            for name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
                with self.subTest(name=name), patch.dict(os.environ, {name: "2"}), self.assertRaises(ValueError):
                    rival.run(self.arguments(Path(directory) / "new"))
            with self.assertRaises(FileExistsError):
                rival.run(self.arguments(Path(directory)))
            with self.assertRaises(FileExistsError):
                rival.run(self.arguments(rival.SOURCE / "forbidden-nested-output"))

    def run_mocked(self, output, model="MVZ", *, replay_error=False, fail_second=False):
        source = self.source()
        calls = []
        read_text = Path.read_text

        def reader(path, *args, **kwargs):
            if path == rival.SOURCE / rival.SOURCE_REPORT:
                return json.dumps(source)
            if path == rival.SOURCE / "generating_parameters.json":
                return json.dumps(self.truth)
            if path == rival.IMPLEMENTATION:
                return json.dumps(self.spec)
            return read_text(path, *args, **kwargs)

        def fitter(training, selected, bound, spec):
            calls.append(selected)
            return self.fit(selected, converged=not (fail_second and len(calls) == 2))

        def scorer(fit, scoring, spec, check):
            check()
            value = {"M0": -1., "MZ": -.99}.get(fit["model"], -.98)
            if replay_error and fit["model"] == "M0":
                value += 1e-9
            return {person: value for person in self.people}

        with ExitStack() as stack:
            stack.enter_context(patch.object(rival, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(rival.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(rival, "load_split", return_value=({}, {})))
            stack.enter_context(patch.object(rival, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(rival.shared, "score_fit", side_effect=scorer))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = rival.run(self.arguments(output, model))
        return report, calls, source

    def test_each_rival_replays_exact_saved_values_and_fits_only_itself_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in rival.MODELS:
                with self.subTest(model=model):
                    output = Path(directory) / model
                    report, calls, source = self.run_mocked(output, model)
                    self.assertEqual(calls, [model, model])
                    self.assertEqual(report["status"], "rival_diagnostics_completed")
                    self.assertEqual(report["generating_model"], "M0")
                    self.assertEqual((report["seed"], report["replicate_index"]), (1300037, 0))
                    self.assertTrue(report["saved_scores_replayed"])
                    self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
                    self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
                    self.assertEqual(report["saved_score_replay_absolute_tolerance"], 1e-12)
                    for person, saved in source["participant_scores"].items():
                        self.assertEqual({key: report["participant_scores"][person][key] for key in saved}, saved)
                    self.assertEqual([(fit["training_runs"], fit["scoring_runs"]) for fit in report["fits"]],
                                     [(list(train), list(held)) for train, held in rival.HALVES])
                    self.assertEqual(report["completed_rivals"], [model])
                    self.assertEqual(set(report["diagnostics"]), {model})
                    self.assertFalse(report["simulation_regenerated"])
                    self.assertFalse(report["saved_M0_MZ_refitted"])
                    self.assertFalse(report["wave_selected"])
                    self.assertIsNone(report["full_recovery_pass"])
                    for field in rival.CLAIM_FIELDS:
                        self.assertEqual(report[field], "not_evaluated")
                    self.assertNotIn("counts_as_curtailment_failure", report)
                    self.assertNotIn("core_family_selection_pass", report)
                    self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text()), report)
                    self.assertFalse((output / "simulated_events.csv.gz").exists())

    def test_replay_mismatch_stops_before_any_new_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, source = self.run_mocked(Path(directory) / "bad-replay", replay_error=True)
            self.assertEqual(calls, [])
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertFalse(report["saved_scores_replayed"])
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertEqual(report["fits"], [])
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertNotIn("participant_scores", report)
            self.assertIn("replay tolerance", report["error"])

    def test_failed_second_fit_retains_checkpoints_without_completed_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "technical"
            report, calls, source = self.run_mocked(output, fail_second=True)
            self.assertEqual(calls, ["MVZ", "MVZ"])
            self.assertEqual(report["status"], "technical_rival_failure")
            self.assertTrue(report["saved_scores_replayed"])
            self.assertFalse(report["rival_diagnostics_complete"])
            self.assertEqual(len(report["fits"]), 2)
            self.assertTrue(report["fits"][0]["converged"])
            self.assertFalse(report["fits"][1]["converged"])
            self.assertEqual(report["completed_rivals"], [])
            self.assertEqual(report["diagnostics"], {})
            self.assertNotIn("participant_scores", report)
            self.assertEqual(json.loads((output / rival.REPORT_NAME).read_text())["fits"], report["fits"])


if __name__ == "__main__":
    unittest.main()
