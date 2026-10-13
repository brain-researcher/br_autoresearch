"""Focused twice0 rival-menu checks; execute only in coordinated compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_twice_mz_menu_pilot as pilot
import test_mz_effect_level_recovery as fixtures


class TwiceMenuPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.EffectLevelRecoveryTests.setUpClass()
        cls.fixture = fixtures.EffectLevelRecoveryTests(
            "test_original_simulation_and_four_fits_use_selected_level_seed_and_frozen_spec")
        cls.helpers = pilot.load_helpers()
        cls.spec = deepcopy(cls.fixture.spec)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mocked_original"
            cls.source, *_ = cls.fixture.exercise(output, "twice", 0)
            cls.source["job_id"] = "46603942"
            cls.truth = json.loads((output / "generating_parameters.json").read_text())
            cls.events = (output / "simulated_events.csv.gz").read_bytes()
            cls.people = cls.helpers.shared.load_cohort(output / "simulated_events.csv.gz")

    def exercise(self, *, source_change=None, truth_change=None, replay_error=False,
                 nonconverged_at=None, worker_error_at=None, expiry=None):
        source, truth, spec = deepcopy((self.source, self.truth, self.spec))
        if source_change:
            source_change(source)
        if truth_change:
            truth_change(truth)
        calls, likelihood_calls, checkpoints, now = [], [], [], [100.]
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        frozen = deepcopy((source, truth, spec))
        original_write = pilot.write_report

        def likelihood(theta, names, data, supplied, rival):
            self.assertEqual((supplied, rival), (self.spec, "fixed"))
            self.assertEqual({len(value) for value in data.values()}, {128})
            self.assertTrue(np.all(data["choice"] == 1))
            likelihood_calls.append(tuple(names))
            focal = tuple(names[3:])
            value = {(): -1., ("rho",): -.98, ("criterion",): -.985,
                     ("eta",): -.995, ("eta", "rho"): -.975, ("eta", "criterion"): -.980}[focal]
            # Baseline last bits differ deliberately within replay tolerance.
            return np.full(128, value + ((1e-4 if replay_error else 3e-13) if focal in ((), ("rho",)) else 0.))

        def fitter(training, model, rival, supplied, *, workers, max_seconds, progress):
            self.assertEqual((model in pilot.RIVALS, rival, supplied, workers), (True, "fixed", self.spec, 4))
            self.assertGreater(max_seconds, 0.)
            self.assertLessEqual(max_seconds, 5395.)
            self.assertEqual(sorted(training), sorted(self.people))
            runs = sorted({int(row["run"]) for record in training.values() for row in record["rows"]})
            self.assertIn(runs, [[1, 4], [2, 3]])
            self.assertTrue(all(len(record["rows"]) == 128 for record in training.values()))
            calls.append((model, runs, deepcopy(supplied), max_seconds))
            if worker_error_at == len(calls):
                error = TimeoutError("mocked active worker deadline")
                error.parallel_progress = {"workers_cleaned_up": True, "stage": "conditional", "sweep": 2,
                    "completed_results": {}, "pending_participants": sorted(training), "worker_pids": [1, 2, 3, 4]}
                raise error
            progress({"stage": "sweep", "sweep": 2, "objective": 100., "converged": True,
                      "subject_optimizer_failures": []})
            fit = self.fixture.fixture.fit(model, converged=nonconverged_at != len(calls))
            fit["support_contacts"] = [{"scope": "fixture", "parameter": "kappa", "bound": "lower"}]
            if nonconverged_at == len(calls):
                fit["subject_optimizer_failures"] = [{"participant": "p000", "message": "mocked unsuccessful return"}]
            return fit

        def writer(output, report):
            checkpoints.append(deepcopy(report))
            original_write(output, report)
            if expiry == "staged" and report["active_fit"] is not None and report["fresh_fit_calls"] == 0:
                now[0] = 5501.
            if expiry == "publication" and report["menu_pilot_complete"]:
                now[0] = 5501.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 5501.

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            episode = Path(directory)
            source_path = episode / "outputs/runs/preserved/twice/replicate0/mz_effect_level_recovery.json"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(json.dumps(source, allow_nan=False))
            (source_path.parent / "generating_parameters.json").write_text(json.dumps(truth, allow_nan=False))
            (source_path.parent / "simulated_events.csv.gz").write_bytes(self.events)
            implementation = episode / "original_implementation.json"
            implementation.write_text(json.dumps(spec))
            output = episode / "outputs/runs/new_pilot"
            source_bytes = {path.name: path.read_bytes() for path in source_path.parent.iterdir()}
            stack.enter_context(patch.object(pilot, "EPISODE", episode))
            stack.enter_context(patch.object(pilot, "SOURCE_REPORT", source_path))
            stack.enter_context(patch.object(pilot, "IMPLEMENTATION", implementation))
            stack.enter_context(patch.object(pilot, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**{name: "1" for name in pilot.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}))
            stack.enter_context(patch.object(pilot, "load_helpers", return_value=self.helpers))
            stack.enter_context(patch.object(pilot.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(patch.object(pilot.signal, "alarm", side_effect=alarm))
            stack.enter_context(patch.object(pilot, "write_report", side_effect=writer))
            stack.enter_context(patch.object(self.helpers.parallel, "fit_partially_pooled_parallel", side_effect=fitter))
            stack.enter_context(patch.object(self.helpers.shared, "trial_likelihood", side_effect=likelihood))
            report = pilot.run(argparse.Namespace(source_report=source_path, implementation=implementation,
                                                  output_dir=output, max_seconds=5400))
            self.assertEqual(report, json.loads((output / pilot.REPORT_NAME).read_text()))
            self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in source_path.parent.iterdir()})
            self.assertEqual(json.loads(implementation.read_text()), spec)
        self.assertEqual((source, truth, spec), frozen)
        self.assertEqual({sig: signal.getsignal(sig) for sig in old_handlers}, old_handlers)
        return report, calls, likelihood_calls, checkpoints

    def test_eight_fresh_fits_preserve_source_baseline_and_all_nine_nested_edges(self):
        report, calls, likelihood_calls, checkpoints = self.exercise()
        self.assertTrue(report["menu_pilot_complete"])
        self.assertEqual([(model, runs) for model, runs, _, _ in calls],
                         [(model, list(train)) for model in pilot.RIVALS for train, held in self.helpers.core.HALVES])
        self.assertEqual(report["fresh_fit_calls"], 8)
        self.assertEqual(report["completed_rivals"], list(pilot.RIVALS))
        self.assertEqual(len(likelihood_calls), (4 + 8) * 108)
        self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
        self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
        for person in self.source["participant_scores"]:
            for key in ("M0", "MZ", "version"):
                self.assertEqual(report["participant_scores"][person][key], self.source["participant_scores"][person][key])
            for model in pilot.RIVALS:
                expected = sum(component["participant_contributions"][person] for component in report["scored_components"]
                               if component["model"] == model)
                self.assertEqual(report["participant_scores"][person][model], expected)
        self.assertEqual(len(report["nested_comparisons"]), 9)
        self.assertEqual(report["nested_comparisons"], {f"{m}-{b}": self.helpers.menu.nested_contrast(
            report["participant_scores"], m, b) for m, b in self.helpers.menu.EDGES})
        self.assertTrue(all(row["heldout_trials_per_participant"] == 128 and row["per_participant_denominator"] == 256
                            for row in report["scored_components"]))
        for field in ("family_selection", "family_recovery_rate", "confusion_rate", "group_recovery", "interval_coverage", "full_gate"):
            self.assertEqual(report[field], "not_evaluated")
        self.assertFalse(report["simulation_regenerated"] or report["saved_M0_MZ_refitted"] or report["automatic_retry"])
        self.assertTrue(report["parallel_workers_cleaned_up"])
        self.assertTrue(all(fit["support_contacts"] for fit in report["fits"]))

    def test_wrong_level_truth_seed_source_fit_or_replay_stops_before_any_new_fit(self):
        for change in (lambda row: row.update(seed=910031), lambda row: row.update(effect_level="half"),
                       lambda row: row.update(baseline_diagnostic_complete=False),
                       lambda row: row["fits"][0].update(converged=False)):
            changed = deepcopy(self.source)
            change(changed)
            with self.subTest(change=change), self.assertRaises((ValueError, RuntimeError)):
                pilot.validate_source(changed, self.truth, self.people, self.spec, self.helpers)
        truth = deepcopy(self.truth)
        truth["p000"]["rho"] = -.2
        with self.assertRaises(ValueError):
            pilot.validate_source(self.source, truth, self.people, self.spec, self.helpers)
        for options in ({"source_change": lambda row: row.update(seed=910031)}, {"replay_error": True}):
            with self.subTest(options=options):
                report, calls, _, _ = self.exercise(**options)
                self.assertFalse(report["menu_pilot_complete"])
                self.assertEqual(calls, [])
                self.assertNotIn("nested_comparisons", report)

    def test_nonconvergence_keeps_returned_fit_and_complete_rival_prefix_without_menu(self):
        report, calls, _, _ = self.exercise(nonconverged_at=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(report["fits"]), 3)
        self.assertFalse(report["fits"][-1]["converged"])
        self.assertTrue(report["fits"][-1]["subject_optimizer_failures"])
        self.assertEqual(report["completed_rivals"], ["MC"])
        self.assertEqual(len(report["scored_components"]), 2)
        self.assertEqual(set(report["participant_scores"]["p000"]), {"M0", "MZ", "MC", "version"})
        self.assertEqual(report["status"], "technical_menu_pilot_failure")
        self.assertNotIn("nested_comparisons", report)
        self.assertIsNone(report["full_recovery_pass"])

    def test_worker_failure_preserves_cleanup_metadata_and_scored_prefix(self):
        report, calls, _, _ = self.exercise(worker_error_at=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(report["fits"]), 2)
        self.assertEqual(report["completed_rivals"], ["MC"])
        self.assertTrue(report["parallel_workers_cleaned_up"])
        self.assertEqual(report["parallel_failure"]["stage"], "conditional")
        self.assertEqual(len(report["parallel_failure"]["pending_participants"]), 108)
        self.assertFalse(report["menu_pilot_complete"])
        self.assertNotIn("nested_comparison_eligibility", report)

    def test_deadline_at_staged_fit_handler_cleanup_or_final_write_never_leaves_success(self):
        for stage, expected_fits in (("staged", 0), ("cleanup", 8), ("publication", 8)):
            with self.subTest(stage=stage):
                report, calls, _, checkpoints = self.exercise(expiry=stage)
                self.assertEqual(len(calls), expected_fits)
                self.assertEqual(len(report["fits"]), expected_fits)
                self.assertFalse(report["menu_pilot_complete"])
                self.assertEqual(report["status"], "technical_menu_pilot_failure")
                self.assertNotIn("nested_comparisons", report)
                self.assertFalse(checkpoints[-1]["menu_pilot_complete"])
                if stage == "publication":
                    self.assertTrue(checkpoints[-2]["menu_pilot_complete"])

    def test_real_capacity_and_thread_guards_run_before_any_legacy_import_or_source_read(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "require_compute"), \
                patch.dict(os.environ, {**{name: "1" for name in pilot.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}), \
                patch.object(pilot, "load_helpers", side_effect=AssertionError("Do not import legacy code")) as loader:
            args = argparse.Namespace(source_report=pilot.SOURCE_REPORT, implementation=pilot.IMPLEMENTATION,
                                      output_dir=Path(directory) / "new", max_seconds=5400)
            for name, value in (("SLURM_CPUS_PER_TASK", "1"), ("SLURM_CPUS_PER_TASK", "8"),
                                ("BLIS_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "8")):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    pilot.run(args)
            for key, value in (("source_report", Path("other.json")), ("implementation", Path("other.json")),
                               ("max_seconds", 5401), ("max_seconds", True)):
                changed = argparse.Namespace(**vars(args))
                setattr(changed, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    pilot.run(changed)
            with self.assertRaises(FileExistsError):
                pilot.run(args)  # Outside the episode's selected output tree.
            loader.assert_not_called()

    def test_two_eligible_bias_families_remain_incomparable_without_winner(self):
        scores = {person: {"version": rows[0]["task_version"], "M0": -1., "MV": -1.,
                          "MZ": -.96, "MC": -.96, "MVZ": -.96, "MVC": -.96}
                  for person, rows in self.people.items()}
        result = pilot.menu_diagnostics(scores, self.helpers)
        self.assertEqual(result["eligible_models_diagnostic"], ["M0", "MZ", "MC"])
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ", "MC"])
        self.assertTrue(result["multiple_incomparable_eligible_models"])
        self.assertEqual(len(result["nested_comparisons"]), 9)
        self.assertNotIn("selected_family", result)


if __name__ == "__main__":
    unittest.main()
