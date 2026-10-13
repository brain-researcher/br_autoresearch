"""Focused64-trial LOO execution checks; run only inside task-owned compute."""
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

import run_mz_leave_one_run_out as loo
import test_mz_other_partitions as fixtures


class LeaveOneRunOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OtherPartitionsTests.setUpClass()
        cls.base = fixtures.OtherPartitionsTests("test_both_partitions_replay_primary_then_fit_four_fresh_original500_models")
        cls.fixture, cls.spec, cls.people = cls.base.fixture, cls.base.spec, cls.base.people

    def source(self):
        return self.base.source()

    def args(self, output, held=1, **changes):
        values = dict(held_out_run=held, source_dir=loo.SOURCE, implementation=loo.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def exercise(self, output, held=1, *, bad_source=False, replay_error=False,
                 split_error=None, fail_second=False, deadline_stage=None, score_error=False):
        source, truth = self.source()
        if bad_source:
            source["replicate_index"] = 1
        before = deepcopy((source, truth, self.spec, self.people))
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        original_halves = loo.shared.HALVES
        reads, splits, fits, likelihoods, checkpoints = [], [], [], [], []
        read_text, writer, now = Path.read_text, loo.write_report, [100.]

        def reader(path, *args, **kwargs):
            if path == loo.IMPLEMENTATION:
                reads.append(path)
                return json.dumps(self.spec)
            if path == loo.SOURCE / loo.SOURCE_REPORT:
                reads.append(path)
                if deadline_stage == "source":
                    now[0] += 5401
                return json.dumps(source)
            if path == loo.SOURCE / "generating_parameters.json":
                reads.append(path)
                return json.dumps(truth)
            return read_text(path, *args, **kwargs)

        def split(path, train, held_runs):
            self.assertEqual(path, loo.SOURCE / "simulated_events.csv.gz")
            splits.append((tuple(train), tuple(held_runs)))
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            training, scoring = panel(train), panel(held_runs)
            if len(held_runs) == 1:
                if split_error == "training_leak":
                    training["p000"]["rows"][0] = scoring["p000"]["rows"][0]
                elif split_error == "heldout_count":
                    scoring["p000"]["rows"].pop()
                elif split_error == "participant":
                    training.pop("p000")
            return training, scoring

        def fitter(training, model, rival, spec):
            self.assertEqual((rival, spec), ("fixed", self.spec))
            self.assertEqual(spec["optimizer"]["maximum_sweeps"], 500)
            self.assertEqual(len(training), 108)
            train, held_runs = loo.selected_runs(held)
            for record in training.values():
                self.assertEqual(len(record["rows"]), 192)
                self.assertEqual({int(row["run"]) for row in record["rows"]}, set(train))
                self.assertFalse(any(int(row["run"]) == held for row in record["rows"]))
            self.assertEqual(len(likelihoods), 432 + 108 * len(fits))
            fits.append((model, train, deepcopy(spec)))
            return self.fixture.fit(model, train, held_runs, converged=not (fail_second and len(fits) == 2))

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual((spec, rival), (self.spec, "fixed"))
            count = len(data["gain"])
            likelihoods.append((count, tuple(names)))
            if count == 128:
                self.assertEqual(fits, [])
                value = -1.98 if "rho" in names else -2.
                return np.full(128, value + (1e-8 if replay_error else 0.))
            self.assertEqual(count, 64)
            if score_error:
                return np.full(128, -1.)
            return np.arange(64, dtype=float) / 1000 - 3 + (.04 if "rho" in names else 0.)

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["loo_diagnostic_complete"]:
                now[0] += 5401

        with ExitStack() as stack:
            stack.enter_context(patch.object(loo, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(loo.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(loo, "load_split", side_effect=split))
            stack.enter_context(patch.object(loo, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(loo.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(loo, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(loo.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = loo.run(self.args(output, held))
        self.assertEqual((source, truth, self.spec, self.people), before)
        self.assertEqual(loo.shared.HALVES, original_halves)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / loo.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        self.assertFalse((output / "generating_parameters.json").exists())
        for record in checkpoints:
            self.assert_scope(record, held)
        return report, splits, fits, likelihoods, source

    def assert_scope(self, report, held):
        self.assertEqual((report["phase"], report["held_out_run"], report["replicate_index"], report["seed"]),
                         (loo.PHASE, held, 0, 910031))
        train, held_runs = loo.selected_runs(held)
        self.assertEqual((report["training_runs"], report["scoring_runs"]), (list(train), list(held_runs)))
        for field in loo.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_LOO",
                      "observed_choices_or_RTs_used", "wave_selected", "automatic_retry",
                      "LOO_runs_are_independent_replications", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        for field in ("core_family_selection_pass", "counts_as_curtailment_failure", "valid_recovery_component"):
            self.assertNotIn(field, report)
            self.assertNotIn(field, report.get("nested_diagnostic", {}))

    def test_all_four_complements_use_only192_training_rows_and64_heldout_rows_after_primary_replay(self):
        for held in (1, 2, 3, 4):
            with self.subTest(held=held), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", held)
            train, held_runs = loo.selected_runs(held)
            self.assertTrue(report["loo_diagnostic_complete"])
            self.assertEqual(splits, list(loo.shared.HALVES) + [(train, held_runs)])
            self.assertEqual([(model, runs) for model, runs, spec in fits], [("M0", train), ("MZ", train)])
            self.assertEqual([count for count, names in likelihoods], [128] * 432 + [64] * 216)
            self.assertEqual(report["scored_fits"], [{"model": model, "training_runs": list(train),
                "scoring_runs": list(held_runs)} for model in ("M0", "MZ")])
            self.assertEqual((report["participants"], report["runs"], report["trials"]), (108, 432, 27648))
            self.assertEqual((report["training_trials_per_participant"], report["heldout_trials_per_participant"]), (192, 64))
            self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
            self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
            self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
            for row in report["participant_scores"].values():
                self.assertAlmostEqual(row["M0"], -2.9685)
                self.assertAlmostEqual(row["MZ"], -2.9285)
            self.assertAlmostEqual(report["nested_diagnostic"]["mean_gain_nats_per_trial"], .04)

    def test_local_scorer_requires64_finite_likelihoods_and_divides_by64_not_primary256(self):
        fit = self.fixture.fit("M0", (2, 3, 4), (1,))
        scoring = {person: {"rows": [row for row in rows if int(row["run"]) == 1]} for person, rows in self.people.items()}
        with patch.object(loo.shared, "trial_likelihood", return_value=np.arange(64, dtype=float)):
            values = loo.score_run(fit, scoring, self.spec)
        self.assertEqual(set(values.values()), {31.5})
        for bad in (np.zeros(128), np.full(64, np.nan), np.full(64, np.inf)):
            with self.subTest(shape=bad.shape), patch.object(loo.shared, "trial_likelihood", return_value=bad), self.assertRaises(ValueError):
                loo.score_run(fit, scoring, self.spec)
        scoring["p000"]["rows"] *= 2
        with patch.object(loo.shared, "trial_likelihood") as likelihood, self.assertRaises(ValueError):
            loo.score_run(fit, scoring, self.spec)
        likelihood.assert_not_called()

    def test_changed_source_or_primary_replay_failure_stops_before_new_training(self):
        for option in ({"bad_source": True}, {"replay_error": True}):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", **option)
            self.assertEqual(report["status"], "technical_loo_failure")
            self.assertEqual(fits, [])
            self.assertFalse(report["saved_primary_scores_replayed"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_diagnostic", report)

    def test_heldout_training_leak_or_wrong_support_is_rejected_before_fitting(self):
        for error in ("training_leak", "heldout_count", "participant"):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", split_error=error)
            self.assertEqual(report["status"], "technical_loo_failure")
            self.assertTrue(report["saved_primary_scores_replayed"])
            self.assertEqual(fits, [])

    def test_nonconvergence_or_scoring_failure_retains_fit_prefix_as_technical_unknown(self):
        for option, fit_count, scored in (({"fail_second": True}, 2, 1), ({"score_error": True}, 1, 0)):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", **option)
            self.assertEqual(report["status"], "technical_loo_failure")
            self.assertEqual(len(report["fits"]), fit_count)
            self.assertEqual(len(report["scored_fits"]), scored)
            self.assertFalse(report["loo_diagnostic_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertNotIn("nested_diagnostic", report)
            if "fail_second" in option:
                self.assertFalse(report["fits"][-1]["converged"])

    def test_whole_cap_includes_source_reads_and_final_write_and_restores_handlers(self):
        for stage, count in (("source", 0), ("finalization", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertEqual(report["status"], "technical_loo_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(len(report["fits"]), count)
            self.assertFalse(report["loo_diagnostic_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertTrue(report["finished_utc"])

    def test_scope_cpu_time_and_fresh_output_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(loo, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read allowed")):
            for change in ({"held_out_run": True}, {"held_out_run": 0}, {"held_out_run": 5},
                           {"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"},
                           {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": 5401}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    loo.run(self.args(Path(directory) / "new", **change))
            for field in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
                with self.subTest(field=field), patch.dict(os.environ, {field: "2"}), self.assertRaises(ValueError):
                    loo.run(self.args(Path(directory) / "new"))
            for output in (Path(directory), loo.SOURCE / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    loo.run(self.args(output))


if __name__ == "__main__":
    unittest.main()
