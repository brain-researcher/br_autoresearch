"""Focused saved-MZ0 alternate-split checks; execute only inside compute."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_mz_other_partitions as other
import test_mz_null_rival as fixtures


class OtherPartitionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def source(self):
        source = self.fixture.source()
        source.update(phase="MZ_SESOI_fitted_recovery_component", generating_model="MZ", seed=910031,
            SESOI_level=1., planned_cell_replicates=200, focal_generating_truth=other.shared.focal_truth("MZ"),
            status="necessary_primary_comparison_completed", valid_recovery_component=True,
            counts_as_curtailment_failure=False, full_behavioral_gate="not_evaluated",
            comparison={"focal_model": "MZ", "core_family_selection_pass": True,
                "counts_as_curtailment_failure": False, "necessary_primary_comparison_pass": True,
                "richer_primary_diagnostic_pass": True, "full_recovery_pass": None,
                "requirements": {name: True for name in ("mean_gain_at_least_0p01",
                    "mean_gain_greater_than_two_participant_SE", "positive_in_both_task_versions",
                    "positive_participant_median", "participant_median_95CI_lower_positive")}})
        truth = {person: {**params, **other.shared.focal_truth("MZ")} for person, params in self.fixture.truth.items()}
        return source, truth

    def args(self, output, partition="12_34", **changes):
        values = dict(partition=partition, source_dir=other.SOURCE, implementation=other.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def exercise(self, output, partition="12_34", *, source_change=None, replay_error=False,
                 fail_at=None, deadline_stage=None, score_error_at=None):
        source, truth = self.source()
        if source_change:
            source_change(source, truth)
        frozen = deepcopy((source, truth, self.people, self.spec))
        original_halves = other.shared.HALVES
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        reads, splits, fits, score_calls, checkpoints = [], [], [], [], []
        read_text, writer, real_score = Path.read_text, other.write_report, other.shared.score_fit
        now = [100.]

        def reader(path, *args, **kwargs):
            if path == other.IMPLEMENTATION:
                reads.append(path)
                return json.dumps(self.spec)
            if path == other.SOURCE / other.SOURCE_REPORT:
                reads.append(path)
                if deadline_stage == "source":
                    now[0] += 5401.
                return json.dumps(source)
            if path == other.SOURCE / "generating_parameters.json":
                reads.append(path)
                return json.dumps(truth)
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, other.SOURCE / "simulated_events.csv.gz")
            splits.append((tuple(train), tuple(held)))
            self.assertTrue(set(train).isdisjoint(held))
            self.assertEqual(set(train) | set(held), {1, 2, 3, 4})
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(train), panel(held)

        def fitter(training, model, rival, spec):
            self.assertEqual(spec, self.spec)
            self.assertEqual(rival, "fixed")
            self.assertEqual(spec["optimizer"]["maximum_sweeps"], 500)
            self.assertEqual(other.shared.HALVES, original_halves)
            train = tuple(sorted({int(row["run"]) for item in training.values() for row in item["rows"]}))
            self.assertEqual(len(training), 108)
            self.assertTrue(all(len(item["rows"]) == 128 for item in training.values()))
            self.assertEqual(len(score_calls), 4 + len(fits))  # Primary replay precedes any new fitting.
            fits.append((model, train, deepcopy(spec)))
            return self.fixture.fit(model, converged=len(fits) != fail_at)

        def scorer(fit, scoring, spec, check):
            self.assertEqual(spec, self.spec)
            score_calls.append((fit["model"], tuple(fit["training_runs"]), tuple(fit["scoring_runs"])))
            self.assertEqual(len(scoring), 108)
            self.assertTrue(all(len(item["rows"]) == 128 for item in scoring.values()))
            self.assertTrue(all({int(row["run"]) for row in item["rows"]} == set(fit["scoring_runs"])
                                for item in scoring.values()))
            if score_error_at == len(score_calls):
                raise RuntimeError("scoring interrupted")
            primary = tuple(fit["training_runs"]) in ((1, 4), (2, 3))
            value = ({"M0": -2., "MZ": -1.98} if primary else {"M0": -2.4, "MZ": -2.36})[fit["model"]]
            if replay_error and primary:
                value += 1e-8
            # Exercise the actual128-trial scorer and sum/256 normalization.
            with patch.object(other.shared, "trial_likelihood", return_value=np.full(128, value)):
                return real_score(fit, scoring, spec, check)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["partition_diagnostic_complete"]:
                now[0] += 5401.

        with ExitStack() as stack:
            stack.enter_context(patch.object(other, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(other.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(other, "load_split", side_effect=split))
            stack.enter_context(patch.object(other, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(other.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(other, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(other.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = other.run(self.args(output, partition))
        self.assertEqual(other.shared.HALVES, original_halves)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual((source, truth, self.people, self.spec), frozen)
        self.assertEqual(json.loads((output / other.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        self.assertFalse((output / "generating_parameters.json").exists())
        for checkpoint in checkpoints:
            self.assert_scope(checkpoint)
        return report, splits, fits, score_calls, source

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["generating_model"], report["replicate_index"], report["seed"]),
                         (other.PHASE, "MZ", 0, 910031))
        self.assertEqual(report["selected_replicates"], [0])
        self.assertIsNone(report["full_recovery_pass"])
        for key in other.CLAIM_FIELDS:
            self.assertEqual(report[key], "not_evaluated")
        for key in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_alternate_partition",
                    "observed_choices_or_RTs_used", "wave_selected", "automatic_retry",
                    "partitions_are_independent_replications", "episode_terminal"):
            self.assertFalse(report[key])
        for key in ("core_family_selection_pass", "counts_as_curtailment_failure", "valid_recovery_component"):
            self.assertNotIn(key, report)
            self.assertNotIn(key, report.get("nested_diagnostic", {}))

    def test_both_partitions_replay_primary_then_fit_four_fresh_original500_models(self):
        for partition, directions in other.PARTITIONS.items():
            with self.subTest(partition=partition), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, scores, source = self.exercise(Path(directory) / "new", partition)
            self.assertTrue(report["partition_diagnostic_complete"])
            self.assertEqual(splits, list(other.shared.HALVES) + list(directions))
            self.assertEqual([(model, train) for model, train, spec in fits],
                             [(model, train) for train, held in directions for model in ("M0", "MZ")])
            self.assertEqual(scores, [(model, train, held) for train, held in (*other.shared.HALVES, *directions)
                                      for model in ("M0", "MZ")])
            self.assertEqual(report["scored_partition_directions"], report["partition_directions"])
            self.assertEqual(len(report["scored_fits"]), 4)
            self.assertEqual((report["participants"], report["runs"], report["trials"]), (108, 432, 27648))
            self.assertEqual(report["heldout_trials_per_direction_per_participant"], 128)
            self.assertEqual(report["combined_scored_trials_per_participant"], 256)

    def test_alternate_scores_start_at_zero_and_saved_primary_last_bits_stay_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            report, splits, fits, scores, source = self.exercise(Path(directory) / "new")
        self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
        self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
        self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
        for row in report["participant_scores"].values():
            self.assertAlmostEqual(row["M0"], -2.4)
            self.assertAlmostEqual(row["MZ"], -2.36)
        self.assertEqual(report["nested_diagnostic"], other.fixed.nested_contrast(report["participant_scores"], "MZ", "M0"))
        self.assertAlmostEqual(report["nested_diagnostic"]["mean_gain_nats_per_trial"], .04)

    def test_wrong_source_identity_truth_or_incomplete_primary_panel_stops_before_new_fits(self):
        changes = (lambda source, truth: source.update(replicate_index=1),
                   lambda source, truth: source.update(phase="alternate_already"),
                   lambda source, truth: source.update(fits=source["fits"][:3]),
                   lambda source, truth: truth["p000"].update(rho=0.))
        for number, change in enumerate(changes):
            with self.subTest(case=number), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, scores, source = self.exercise(Path(directory) / "new", source_change=change)
            self.assertEqual(report["status"], "technical_partition_failure")
            self.assertFalse(report["source_baseline_validated"])
            self.assertEqual((splits, fits, scores), ([], [], []))

    def test_primary_score_replay_failure_is_technical_before_alternate_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            report, splits, fits, scores, source = self.exercise(Path(directory) / "new", replay_error=True)
        self.assertEqual(report["status"], "technical_partition_failure")
        self.assertTrue(report["source_baseline_validated"])
        self.assertFalse(report["saved_primary_scores_replayed"])
        self.assertIn("numerical replay tolerance", report["error"])
        self.assertEqual(splits, list(other.shared.HALVES))
        self.assertEqual(fits, [])
        self.assertNotIn("participant_scores", report)

    def test_returned_nonconverged_fit_and_scoring_failure_preserve_prefix_without_scientific_result(self):
        for options, fit_count, scored_count in (({"fail_at": 3}, 3, 2), ({"score_error_at": 6}, 2, 1)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, scores, source = self.exercise(Path(directory) / "new", **options)
            self.assertEqual(report["status"], "technical_partition_failure")
            self.assertEqual(len(report["fits"]), fit_count)
            self.assertEqual(len(report["scored_fits"]), scored_count)
            self.assertFalse(report["partition_diagnostic_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_diagnostic", report)
            self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
            if "fail_at" in options:
                self.assertFalse(report["fits"][-1]["converged"])

    def test_deadline_covers_source_reads_and_finalization_with_failure_checkpoint_and_handler_restoration(self):
        for stage, fit_count in (("source", 0), ("finalization", 4)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, scores, source = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertEqual(report["status"], "technical_partition_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(len(report["fits"]), fit_count)
            self.assertFalse(report["partition_diagnostic_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_diagnostic", report)
            self.assertGreaterEqual(report["elapsed_seconds"], 0.)
            self.assertTrue(report["finished_utc"])

    def test_source_partition_cpu_fresh_output_and_time_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(other, "require_compute"), \
                patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source reads allowed")):
            for change in ({"partition": "14_23"}, {"source_dir": Path(directory)},
                           {"implementation": Path(directory) / "different.json"}, {"max_seconds": True},
                           {"max_seconds": 0}, {"max_seconds": 5401}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    other.run(self.args(Path(directory) / "new", **change))
            for field in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
                with self.subTest(field=field), patch.dict(os.environ, {field: "2"}), self.assertRaises(ValueError):
                    other.run(self.args(Path(directory) / "new"))
            for output in (Path(directory), other.SOURCE / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    other.run(self.args(output))


if __name__ == "__main__":
    unittest.main()
