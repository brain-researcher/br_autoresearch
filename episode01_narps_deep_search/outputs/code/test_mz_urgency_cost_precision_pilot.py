"""Focused urgency numerical-pilot checks; execute only in task-owned compute."""
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

import run_mz_urgency_cost_precision_pilot as pilot
import test_mz_other_partitions as fixtures


class UrgencyPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OtherPartitionsTests.setUpClass()
        cls.fixture = fixtures.OtherPartitionsTests("test_both_partitions_replay_primary_then_fit_four_fresh_original500_models")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def args(self, output, **changes):
        values = dict(source_dir=pilot.SOURCE, implementation=pilot.IMPLEMENTATION,
                      output_dir=Path(output), max_seconds=1800)
        values.update(changes)
        return argparse.Namespace(**values)

    def exercise(self, output, *, bad_source=False, replay_error=False, fail_call=None,
                 bad_scores=False, deadline_stage=None):
        source, truth = self.fixture.source()
        if bad_source:
            source["replicate_index"] = 1
        before = deepcopy((source, truth, self.people, self.spec))
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        halves = pilot.shared.HALVES
        read_text, writer, now = Path.read_text, pilot.write_report, [100.]
        reads, splits, fixed_calls, urgency_calls, checkpoints = [], [], [], [], []

        def reader(path, *args, **kwargs):
            if path == pilot.IMPLEMENTATION:
                reads.append(path)
                return json.dumps(self.spec)
            if path == pilot.SOURCE / pilot.SOURCE_REPORT:
                reads.append(path)
                if deadline_stage == "source":
                    now[0] += 1801
                return json.dumps(source)
            if path == pilot.SOURCE / "generating_parameters.json":
                reads.append(path)
                return json.dumps(truth)
            return read_text(path, *args, **kwargs)

        def split(path, train, held):
            self.assertEqual(path, pilot.SOURCE / "simulated_events.csv.gz")
            splits.append((tuple(train), tuple(held)))
            def panel(runs):
                return {person: {"version": rows[0]["task_version"],
                    "rows": [row for row in rows if int(row["run"]) in runs]} for person, rows in self.people.items()}
            return panel(train), panel(held)

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual(len(data["gain"]), 128)
            if rival == "fixed":
                self.assertEqual(spec, self.spec)
                self.assertEqual(urgency_calls, [])
                fixed_calls.append(tuple(names))
                value = -1.98 if "rho" in names else -2.
                return np.full(128, value + (1e-8 if replay_error else 0.))
            self.assertEqual(rival, "urgency")
            self.assertEqual(len(fixed_calls), 432)
            self.assertEqual(len(names), 5 if "rho" in names else 4)
            expected = deepcopy(self.spec)
            refined = len(urgency_calls) % 2 == 1
            if refined:
                expected["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)
            self.assertEqual(spec, expected)
            urgency_calls.append((theta.copy(), list(names), deepcopy(data), deepcopy(spec)))
            if len(urgency_calls) == fail_call:
                if bad_scores:
                    return np.full(128, np.nan)
                raise RuntimeError("Selected likelihood call interrupted")
            values = np.full(128, -1.9 if "rho" in names else -2.)
            if refined:
                values += .0001 * (np.arange(128) - 64)
            if deadline_stage == "likelihood" and len(urgency_calls) == 1:
                now[0] += 1801
            return values

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)
            if deadline_stage == "finalization" and report["pilot_complete"]:
                now[0] += 1801

        with ExitStack() as stack:
            stack.enter_context(patch.object(pilot, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(pilot.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(pilot.shared, "load_split", side_effect=split))
            stack.enter_context(patch.object(pilot.shared, "trial_likelihood", side_effect=likelihood))
            stack.enter_context(patch.object(pilot.shared, "fit_partially_pooled", side_effect=AssertionError("No model fitting")))
            stack.enter_context(patch.object(pilot.behavior, "fit_partially_pooled", side_effect=AssertionError("No hierarchical fit")))
            stack.enter_context(patch.object(pilot.behavior, "minimize", side_effect=AssertionError("No optimizer")))
            stack.enter_context(patch.object(pilot, "write_report", side_effect=checkpoint))
            stack.enter_context(patch.object(pilot.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = pilot.run(self.args(output))
        self.assertEqual((source, truth, self.people, self.spec), before)
        self.assertEqual(pilot.shared.HALVES, halves)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / pilot.REPORT_NAME).read_text()), report)
        self.assertFalse((output / "simulated_events.csv.gz").exists())
        self.assertFalse((output / "generating_parameters.json").exists())
        for record in checkpoints:
            self.assert_scope(record)
        return report, splits, fixed_calls, urgency_calls, checkpoints, source

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["replicate_index"], report["seed"], report["rival"]),
                         (pilot.PHASE, 0, 910031, "urgency"))
        for field in pilot.CLAIM_FIELDS:
            self.assertEqual(report[field], "not_evaluated")
        for field in ("simulation_regenerated", "simulation_copied", "observed_choices_or_RTs_used",
                      "optimizer_called", "model_fitted", "actual_fitted_support_evaluated",
                      "wave_selected", "automatic_retry", "episode_terminal"):
            self.assertIs(report[field], False)
        self.assertIsNone(report["full_recovery_pass"])
        self.assertEqual((report["expected_probe_pairs"], report["expected_likelihood_calls"]), (24, 48))
        for field in ("participant_scores", "comparison", "nested_diagnostic", "numerical_precision_passed",
                      "valid_recovery_component", "counts_as_curtailment_failure"):
            self.assertNotIn(field, report)

    def test_exact_existing_initializations_and_actual_drift_formula_are_preserved(self):
        data = pilot.shared.make_data(self.people["p000"][:128])
        before = deepcopy(self.spec)
        for model, dimension in (("M0", 4), ("MZ", 5)):
            names, starts = pilot.initialization_vectors(model, self.spec)
            self.assertEqual(names, pilot.behavior.parameter_names(model, "urgency"))
            self.assertEqual(len(names), dimension)
            centers = np.array([self.spec["parameters"][name]["population_center"] for name in names])
            scales = np.array([self.spec["parameters"][name]["individual_scale_center"] for name in names])
            bounds = np.array([self.spec["parameters"][name]["bounds"] for name in names])
            for theta, offset in zip(starts, (0., -.5, .5)):
                np.testing.assert_array_equal(theta, np.clip(centers + offset * scales, bounds[:, 0], bounds[:, 1]))
                drift, horizon = pilot.drift_and_horizon(theta, names, data)
                params = dict(zip(names, theta))
                expected = pilot.behavior.deadline_ddm.drift_rate(data["gain"], data["loss"], params["kappa"])
                np.testing.assert_array_equal(drift, expected)
                self.assertEqual(horizon, 4 - 4 * pilot.behavior.expit(params["nondecision_logit"]))
            np.testing.assert_allclose([theta[-1] for theta in starts], [.1, 0., .2], rtol=0, atol=0)
            if model == "MZ":
                np.testing.assert_allclose([theta[names.index("rho")] for theta in starts], [0., -.3, .3], rtol=0, atol=0)
        self.assertEqual(self.spec, before)

    def test_full_source_replay_precedes_exact24_training_probe_pairs_without_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            report, splits, fixed_calls, calls, checkpoints, source = self.exercise(Path(directory) / "new")
        self.assertTrue(report["pilot_complete"])
        self.assertEqual(splits, list(pilot.HALVES))
        self.assertEqual((len(fixed_calls), len(calls), len(report["probes"])), (432, 48, 24))
        self.assertEqual((report["completed_probe_pairs"], report["completed_likelihood_calls"]), (24, 48))
        self.assertEqual((report["source_participants"], report["source_runs"], report["source_trials"]), (108, 432, 27648))
        self.assertEqual(report["saved_primary_participant_scores"], source["participant_scores"])
        self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
        selected = {version: min(person for person, rows in self.people.items() if rows[0]["task_version"] == version)
                    for version in ("EI", "ER")}
        self.assertEqual(report["selected_probe_participants"], selected)
        identities = set()
        for index, probe in enumerate(report["probes"]):
            theta, names, data, spec = calls[2 * index]
            self.assertEqual(probe["participant"], selected[probe["task_version"]])
            identities.add((probe["participant"], tuple(probe["training_runs"]), probe["model"], probe["start_index"]))
            self.assertEqual(len(probe["trial_slots"]), 128)
            self.assertEqual({run for run, trial in probe["trial_slots"]}, set(probe["training_runs"]))
            self.assertTrue(set(probe["training_runs"]).isdisjoint(probe["excluded_scoring_runs"]))
            drift, horizon = pilot.drift_and_horizon(theta, names, data)
            self.assertEqual(probe["actual_unique_drift_count"], len(np.unique(drift)))
            self.assertEqual(probe["decision_horizon_seconds"], horizon)
            self.assertTrue(probe["pair_complete"])
            for mesh in ("coarse", "refined"):
                self.assertEqual(len(probe["meshes"][mesh]["trial_log_scores"]), 128)
                self.assertGreaterEqual(probe["meshes"][mesh]["elapsed_seconds"], 0.)
        self.assertEqual(len(identities), 24)
        self.assertNotIn("active_call", report)
        changed = deepcopy(report["refined_numerical_implementation"])
        changed["urgency_solver"].update(dt_seconds=.002, spatial_intervals=200)
        self.assertEqual(changed, self.spec)

    def test_mesh_differences_are_per_trial_and_mean_uses128_not256_or108(self):
        with tempfile.TemporaryDirectory() as directory:
            report, *_ = self.exercise(Path(directory) / "new")
        expected = .0001 * (np.arange(128) - 64)
        for probe in report["probes"]:
            difference = probe["mesh_difference"]
            self.assertEqual(difference["direction"], "refined_minus_coarse")
            np.testing.assert_allclose(difference["per_trial_log_score_difference"], expected, rtol=0, atol=3e-16)
            self.assertAlmostEqual(difference["mean_log_score_difference"], expected.mean(), places=15)
            self.assertAlmostEqual(difference["maximum_absolute_log_score_difference"], np.max(abs(expected)), places=15)
            self.assertNotIn("passes", difference)

    def test_bad_source_or_primary_replay_prevents_all_urgency_probes(self):
        for option in ({"bad_source": True}, {"replay_error": True}):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                report, splits, fixed_calls, calls, checkpoints, source = self.exercise(Path(directory) / "new", **option)
            self.assertFalse(report["pilot_complete"])
            self.assertFalse(report["saved_primary_scores_replayed"])
            self.assertEqual((report["completed_probe_pairs"], report["completed_likelihood_calls"]), (0, 0))
            self.assertEqual(calls, [])
            self.assertTrue(report["incomplete_reason"])

    def test_refined_exception_or_nonfinite_preserves_first_coarse_vector_and_active_call(self):
        for bad_scores in (False, True):
            with self.subTest(bad_scores=bad_scores), tempfile.TemporaryDirectory() as directory:
                report, splits, fixed_calls, calls, checkpoints, source = self.exercise(
                    Path(directory) / "new", fail_call=2, bad_scores=bad_scores)
            self.assertEqual(report["status"], "urgency_cost_precision_pilot_incomplete")
            self.assertEqual((report["completed_probe_pairs"], report["completed_likelihood_calls"]), (0, 1))
            self.assertEqual(report["active_call"], {"probe_index": 0, "mesh": "refined"})
            self.assertEqual(set(report["probes"][0]["meshes"]), {"coarse"})
            self.assertFalse(report["probes"][0]["pair_complete"])
            self.assertTrue(any(record["completed_likelihood_calls"] == 1 for record in checkpoints))

    def test_whole_cap_covers_source_likelihood_and_finalization_and_keeps_finished_work(self):
        for stage, pairs, count in (("source", 0, 0), ("likelihood", 0, 1), ("finalization", 24, 48)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, *_ = self.exercise(Path(directory) / "new", deadline_stage=stage)
            self.assertFalse(report["pilot_complete"])
            self.assertIn("whole-process deadline", report["incomplete_reason"])
            self.assertEqual((report["completed_probe_pairs"], report["completed_likelihood_calls"]), (pairs, count))
            self.assertTrue(report["finished_utc"])
            if count:
                self.assertEqual(len(report["probes"][0]["meshes"]["coarse"]["trial_log_scores"]), 128)

    def test_fixed_source_original_spec_cpu_cap_fresh_output_and_cli(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "require_compute"), \
                patch.dict(os.environ, {**fixtures.fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
            for change in ({"source_dir": Path(directory)}, {"implementation": Path(directory) / "other.json"},
                           {"max_seconds": True}, {"max_seconds": 0}, {"max_seconds": 1801}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    pilot.run(self.args(Path(directory) / "new", **change))
            for field in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                with self.subTest(field=field), patch.dict(os.environ, {field: "2"}), self.assertRaises(ValueError):
                    pilot.run(self.args(Path(directory) / "new"))
            for output in (Path(directory), pilot.SOURCE / "new"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    pilot.run(self.args(output))
        with patch.object(pilot, "run", return_value={"pilot_complete": True}) as runner, redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main(["--source-dir", str(pilot.SOURCE), "--implementation", str(pilot.IMPLEMENTATION),
                "--output-dir", "/tmp/urgency-pilot-fresh", "--max-seconds", "1800"]), 0)
        self.assertEqual(vars(runner.call_args.args[0]), vars(self.args("/tmp/urgency-pilot-fresh")))


if __name__ == "__main__":
    unittest.main()
