"""Focused common-gain execution checks; run only inside task-owned compute."""
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

import run_mz_common_gain_range as common
import test_mz_other_partitions as fixtures


class CommonGainRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OtherPartitionsTests.setUpClass()
        cls.base = fixtures.OtherPartitionsTests("test_both_partitions_replay_primary_then_fit_four_fresh_original500_models")
        cls.fixture, cls.spec = cls.base.fixture, cls.base.spec
        cls.people = deepcopy(cls.base.people)
        for person in ("p000", "p054"):
            row = next(row for row in cls.people[person] if float(row["gain"]) == 10)
            row.update(choice="-1", no_response="1", response_time="", response_category="no_response")

    def source(self):
        source, truth = self.base.source()
        source["simulated_deadline_misses"] = 2
        return source, truth

    def args(self, output, direction=0, **changes):
        values = dict(direction=direction, source_dir=common.SOURCE, implementation=common.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=5400)
        values.update(changes)
        return argparse.Namespace(**values)

    def exercise(self, output, direction=0, *, bad_source=False, replay_error=False,
                 fail_second=False, score_error=False, deadline_stage=None):
        source, truth = self.source()
        if bad_source:
            source["replicate_index"] = 1
        before = deepcopy((source, truth, self.spec, self.people))
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        original_halves = common.shared.HALVES
        splits, fits, likelihoods, checkpoints = [], [], [], []
        read_text, writer, now = Path.read_text, common.write_report, [100.]
        train, held = common.selected_runs(direction)

        def reader(path, *args, **kwargs):
            if path == common.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == common.SOURCE / common.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 5401
                return json.dumps(source)
            if path == common.SOURCE / "generating_parameters.json":
                return json.dumps(truth)
            return read_text(path, *args, **kwargs)

        def split(path, training_runs, scoring_runs):
            self.assertEqual(path, common.SOURCE / "simulated_events.csv.gz")
            splits.append((tuple(training_runs), tuple(scoring_runs)))
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(training_runs), panel(scoring_runs)

        def fitter(training, model, rival, spec):
            self.assertEqual((rival, spec), ("fixed", self.spec))
            self.assertEqual(spec["optimizer"]["maximum_sweeps"], 500)
            self.assertEqual(model, common.MODELS[len(fits)])
            self.assertEqual(set(training), set(self.people))
            self.assertEqual(len(likelihoods), 432 + 108 * len(fits))
            for person, record in training.items():
                expected = [row for row in self.people[person] if int(row["run"]) in train
                            and 10 <= float(row["gain"]) <= 20]
                self.assertEqual(record["rows"], expected)
                self.assertTrue(all(left is right for left, right in zip(record["rows"], expected)))
                self.assertTrue(all(int(row["run"]) not in held for row in record["rows"]))
            fits.append((model, train, deepcopy(spec)))
            if deadline_stage == "fit":
                now[0] += 5401
            return self.fixture.fit(model, train, held, converged=not (fail_second and len(fits) == 2))

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual((spec, rival), (self.spec, "fixed"))
            likelihoods.append((len(data["gain"]), tuple(names)))
            if not fits:
                self.assertEqual(len(data["gain"]), 128)
                return np.full(128, (-1.98 if "rho" in names else -2.) + (1e-8 if replay_error else 0.))
            self.assertTrue(np.all((data["gain"] >= 10) & (data["gain"] <= 20)))
            if score_error:
                return np.zeros(128)
            if deadline_stage == "scoring":
                now[0] += 5401
            return (-2.95 if "rho" in names else -3.) + .001 * data["gain"]

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["common_gain_component_complete"]:
                now[0] += 5401

        with ExitStack() as stack:
            stack.enter_context(patch.object(common, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(common.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(common, "load_split", side_effect=split))
            stack.enter_context(patch.object(common, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(common.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(common, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(common.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = common.run(self.args(output, direction))
        self.assertEqual((source, truth, self.spec, self.people), before)
        self.assertEqual(common.shared.HALVES, original_halves)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / common.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        self.assertFalse((output / "generating_parameters.json").exists())
        for record in checkpoints:
            self.assert_scope(record, direction)
        return report, splits, fits, likelihoods, source

    def assert_scope(self, report, direction):
        self.assertEqual((report["phase"], report["direction"], report["replicate_index"], report["seed"]),
                         (common.PHASE, direction, 0, 910031))
        self.assertEqual(report["gain_range_dollars"], [10, 20])
        self.assertTrue(report["filter_boundary_inclusive"])
        for field in common.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "saved_primary_fits_reused_for_common_gain",
                      "observed_choices_or_RTs_used", "wave_selected", "automatic_retry",
                      "task_versions_are_independent_replications", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        self.assertEqual(report["score_normalization"], common.NORMALIZATION)
        for field in ("core_family_selection_pass", "counts_as_curtailment_failure", "nested_diagnostic"):
            self.assertNotIn(field, report)

    def test_both_directions_replay_original_then_fit_only_unchanged_filtered_rows_and_keep_sums(self):
        reports = []
        for direction in (0, 1):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory() as directory:
                report, splits, fits, likelihoods, source = self.exercise(Path(directory) / "fresh", direction)
            reports.append(report)
            train, held = common.selected_runs(direction)
            self.assertTrue(report["common_gain_component_complete"])
            self.assertEqual(splits, list(common.HALVES))
            self.assertEqual([(model, runs) for model, runs, _ in fits], [("M0", train), ("MZ", train)])
            self.assertEqual(likelihoods[:432], [(128, tuple(source["fits"][i // 108]["parameter_names"])) for i in range(432)])
            self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
            self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
            self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
            self.assertEqual(list(report["completed_model_scores"]), ["M0", "MZ"])
            self.assertEqual(report["scored_fits"], [{"model": model, "training_runs": list(train),
                "scoring_runs": list(held)} for model in common.MODELS])
            for person, row in report["participant_scores"].items():
                selected = [r for r in self.people[person] if int(r["run"]) in held and 10 <= float(r["gain"]) <= 20]
                n = len(selected)
                self.assertEqual(row["heldout_trials"], n)
                for model, intercept in (("M0", -3.), ("MZ", -2.95)):
                    expected = float(sum(intercept + .001 * float(r["gain"]) for r in selected))
                    self.assertAlmostEqual(row[model + "_log_sum"], expected, places=10)
                    self.assertEqual(report["completed_model_scores"][model][person]["trials"], n)
                support = report["retained_support"][person]
                self.assertEqual(support["training_trials"] + n, 96 if row["version"] == "EI" else 176)
                self.assertEqual(support["heldout_slots"], [[int(r["run"]), int(r["trial_order"])] for r in selected])
        for person in self.people:
            left, right = [report["retained_support"][person] for report in reports]
            self.assertEqual(left["training_slots"], right["heldout_slots"])
            self.assertEqual(left["heldout_slots"], right["training_slots"])
        self.assertNotEqual(reports[0]["retained_support"]["p000"]["training_trials"],
                            reports[0]["retained_support"]["p000"]["heldout_trials"])

    def test_filter_is_inclusive_preserves_misses_and_rejects_missing_duplicate_or_empty_support(self):
        training, scoring, support = common.filtered_panels(self.people, 0)
        for person in self.people:
            retained = training[person]["rows"] + scoring[person]["rows"]
            self.assertEqual({float(row["gain"]) for row in retained},
                             set(range(10, 21, 2) if support[person]["version"] == "EI" else range(10, 21)))
        self.assertEqual(sum(int(r["no_response"]) for p in (training, scoring) for item in p.values() for r in item["rows"]), 2)
        for change in (lambda p: p.pop("p000"),
                       lambda p: p["p000"].pop(0),
                       lambda p: p["p000"].__setitem__(1, p["p000"][0]),
                       lambda p: [r.update(run="1") for r in p["p000"] if 10 <= float(r["gain"]) <= 20]):
            people = deepcopy(self.people)
            change(people)
            with self.assertRaises(ValueError):
                common.filtered_panels(people, 0)

    def test_local_scorer_uses_actual_count_without128_or256_normalization(self):
        _, scoring, _ = common.filtered_panels(self.people, 0)
        fit = self.fixture.fit("M0")
        with patch.object(common.shared, "trial_likelihood", side_effect=lambda t, n, d, s, r: np.arange(len(d["gain"]), dtype=float)):
            result = common.score_filtered(fit, scoring, self.spec)
        for person, value in result.items():
            n = len(scoring[person]["rows"])
            self.assertEqual(value, {"log_sum": n * (n - 1) / 2, "trials": n})
        for bad in (np.zeros(128), np.array([np.nan]), np.array([np.inf])):
            with patch.object(common.shared, "trial_likelihood", return_value=bad), self.assertRaises(ValueError):
                common.score_filtered(fit, scoring, self.spec)

    def test_source_or_replay_failure_stops_before_any_new_fit(self):
        for option in ({"bad_source": True}, {"replay_error": True}):
            with tempfile.TemporaryDirectory() as directory:
                report, splits, fits, calls, source = self.exercise(Path(directory) / "fresh", **option)
            self.assertEqual(report["status"], "technical_common_gain_failure")
            self.assertEqual(fits, [])
            self.assertFalse(report["saved_primary_scores_replayed"])
            self.assertEqual(report["completed_model_scores"], {})
            self.assertNotIn("participant_scores", report)

    def test_nonconvergence_or_scoring_failure_preserves_fits_and_completed_score_prefix_as_unknown(self):
        for option, fit_count, models in (({"fail_second": True}, 2, ["M0"]), ({"score_error": True}, 1, [])):
            with tempfile.TemporaryDirectory() as directory:
                report, *_ = self.exercise(Path(directory) / "fresh", **option)
            self.assertEqual(report["status"], "technical_common_gain_failure")
            self.assertEqual(len(report["fits"]), fit_count)
            self.assertEqual(list(report["completed_model_scores"]), models)
            self.assertEqual([row["model"] for row in report["scored_fits"]], models)
            self.assertFalse(report["common_gain_component_complete"])
            self.assertNotIn("participant_scores", report)

    def test_whole_cap_covers_source_fit_scoring_and_final_publication_with_prefixes_retained(self):
        for stage, fit_count, models in (("source", 0, []), ("fit", 1, []),
                                         ("scoring", 1, []), ("finalization", 2, ["M0", "MZ"])):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, *_ = self.exercise(Path(directory) / "fresh", deadline_stage=stage)
            self.assertEqual(report["status"], "technical_common_gain_failure")
            self.assertIn("whole-process deadline", report["error"])
            self.assertEqual(len(report["fits"]), fit_count)
            self.assertEqual(list(report["completed_model_scores"]), models)
            self.assertFalse(report["common_gain_component_complete"])
            self.assertNotIn("participant_scores", report)

    def test_direction_source_cpu_cap_and_fresh_output_guards_precede_reads(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(common, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for change in ({"direction": -1}, {"direction": 2}, {"direction": True},
                           {"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"},
                           {"max_seconds": 0}, {"max_seconds": 5401}, {"max_seconds": True}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    common.run(self.args(Path(directory) / "fresh", **change))
            for field in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
                with patch.dict(os.environ, {field: "2"}), self.assertRaises(ValueError):
                    common.run(self.args(Path(directory) / "fresh"))
            for output in (Path(directory), common.SOURCE / "new"):
                with self.assertRaises(FileExistsError):
                    common.run(self.args(output))
        with patch.object(common, "run", return_value={"common_gain_component_complete": True}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(common.main(["--direction", "1", "--source-dir", str(common.SOURCE),
                "--implementation", str(common.IMPLEMENTATION), "--output-dir", "/fresh", "--max-seconds", "5400"]), 0)
        self.assertEqual(run.call_args.args[0].direction, 1)


if __name__ == "__main__":
    unittest.main()
