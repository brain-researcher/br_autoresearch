"""Source-free checks for the observed-data orchestration; run on compute only."""
from copy import deepcopy
import csv
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import hierarchical_behavior as behavior
from run_single_process_recovery import validate_fit
import run_observed_behavior as observed


class ObservedBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(observed.IMPLEMENTATION.read_text())
        cls.rows = []
        categories = list(observed.CATEGORIES)
        for person_index in range(108):
            version = "EI" if person_index < 54 else "ER"
            gains = list(range(10, 41, 2) if version == "EI" else range(5, 21))
            for index in range(256):
                category = categories[index % 5]
                choice, strength = observed.CATEGORIES[category]
                cls.rows.append(dict(zip(observed.FIELDS, (
                    f"sub-{person_index:03d}", version, str(index//64+1), str(index%64+1),
                    str(5*(index%64)), str(gains[index//16]), str(5+index%16), category,
                    str(choice), str(strength), "" if choice == -1 else str(.2+.01*(index%64)),
                    str(int(choice == -1)), "fixture-template"))))

    def fit_fixture(self, model, people, nonconverged=False):
        names = behavior.parameter_names(model, "fixed")
        centers = [self.spec["parameters"][name]["population_center"] for name in names]
        return {"model": model, "rival": "fixed", "parameter_names": names,
                "participants": sorted(people), "parameters": [list(centers) for _ in people],
                "global_mean": centers, "version_means": [list(centers), list(centers)],
                "version_standard_deviations": [[.3]*len(names), [.3]*len(names)],
                "version_order": ["EI", "ER"], "inference": self.spec["method"],
                "objective_history": [100., 100.], "converged": not nonconverged,
                "subject_optimizer_failures": [], "support_contacts": [],
                "group_interval_coverage_qualified": False, "individual_rank_recovery_qualified": False}

    def exercise(self, command="fit", model="MZ", direction=0, failure=None):
        calls, snapshots, clock = [], [], [0.]
        rows = deepcopy(self.rows)
        if failure == "source":
            rows[0]["choice"] = "0"
        expected_people = observed.validate_rows(deepcopy(self.rows))
        expected_training, expected_held = observed.split_people(expected_people, direction)
        def fit(training, selected_model, rival, spec, **kwargs):
            self.assertEqual(training, expected_training)
            self.assertEqual((selected_model, rival, spec), (model, "fixed", self.spec))
            self.assertEqual(kwargs["workers"], 4)
            self.assertGreater(kwargs["max_seconds"], 0)
            self.assertLessEqual(kwargs["max_seconds"], 1795)
            calls.append("fit")
            kwargs["progress"]({"stage": "sweep", "sweep": 1, "objective": 100., "converged": False,
                                "subject_optimizer_failures": []})
            if failure == "worker":
                error = RuntimeError("fixture worker failure")
                error.parallel_progress = {"workers_cleaned_up": True, "objective_history": [100.]}
                raise error
            return self.fit_fixture(model, training, failure == "nonconverged")
        def likelihood(theta, names, data, spec, rival):
            index = sum(call == "score" for call in calls)
            person = sorted(expected_held)[index]
            expected = behavior.make_data(expected_held[person]["rows"])
            for key in expected:
                np.testing.assert_equal(data[key], expected[key])
            self.assertEqual(rival, "fixed")
            calls.append("score")
            if failure == "score":
                return np.full(128, np.nan)
            return np.full(128, -2.)
        scientific = SimpleNamespace(make_data=behavior.make_data, trial_likelihood=likelihood)
        parallel = SimpleNamespace(CLEANUP_SECONDS=4., fit_partially_pooled_parallel=fit)
        environment = {"hostname": "fixture-compute", "job_id": "12345", "allocated_cpus": 4,
                       "thread_environment": {name: "1" for name in observed.THREAD_ENV}}
        actual_write = observed.atomic_report
        def writer(output, name, report):
            snapshots.append(deepcopy(report))
            actual_write(output, name, report)
            if failure == "publication" and report["status"] == "direction_completed":
                clock[0] = 1801.
        def alarm(seconds):
            if seconds == 0 and failure == "restoration":
                clock[0] = 1801.
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            source = episode / "qualified/source.csv.gz"
            source.parent.mkdir()
            with gzip.open(source, "wt", newline="") as stream:
                writer_csv = csv.DictWriter(stream, fieldnames=observed.FIELDS)
                writer_csv.writeheader()
                writer_csv.writerows(rows)
            source_before = source.read_bytes()
            output = episode / "outputs/runs/attempt"
            args = SimpleNamespace(command=command, model=model, direction=direction,
                                   implementation=observed.IMPLEMENTATION, output_dir=output,
                                   max_seconds=120 if command == "describe" else 1800)
            with patch.object(observed, "EPISODE", episode), patch.object(observed, "SOURCE", source), \
                 patch.object(observed, "require_environment", return_value=environment), \
                 patch.object(observed, "load_helpers", return_value=(np, scientific, parallel, validate_fit)), \
                 patch.object(observed, "atomic_report", side_effect=writer), \
                 patch.object(observed.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(observed.signal, "alarm", side_effect=alarm):
                handlers = {sig: observed.signal.getsignal(sig) for sig in (observed.signal.SIGALRM, observed.signal.SIGTERM)}
                code = observed.run(args)
                self.assertEqual(handlers, {sig: observed.signal.getsignal(sig) for sig in handlers})
            report = json.loads((output / observed.REPORTS[command]).read_text())
            self.assertEqual(source.read_bytes(), source_before)
            score_rows = []
            if (output / "heldout_trial_scores.csv.gz").exists():
                with gzip.open(output / "heldout_trial_scores.csv.gz", "rt", newline="") as stream:
                    score_rows = list(csv.DictReader(stream))
        return code, report, calls, snapshots, score_rows

    def test_five_categories_misses_quantiles_and_full_descriptions(self):
        code, report, calls, _, _ = self.exercise(command="describe")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        summary = report["descriptions"]
        self.assertEqual(summary["overall"]["trials"], 27648)
        self.assertEqual(len(summary["by_participant_run"]), 108)
        self.assertEqual(len(summary["by_version_gain_loss"]), 512)
        self.assertTrue(all(row["trials"] == 54 for row in summary["by_version_gain_loss"]))
        sample = deepcopy(self.rows[:5])
        for row, rt in zip(sample, (1., 3., 2., 4., None)):
            row["response_time"] = "" if rt is None else str(rt)
        stats = observed.describe_rows(sample, np)
        self.assertEqual((stats["accept_count"], stats["reject_count"], stats["miss_count"]), (2, 2, 1))
        self.assertEqual(stats["accept_fraction_responded"], .5)
        np.testing.assert_allclose(stats["responded_rt_by_choice"]["accept"]["seconds_q10_q50_q90"], [1.2, 2., 2.8])
        self.assertFalse(report["scores_available"])

    def test_invalid_mapping_missing_duplicate_grid_and_version_refused(self):
        mutations = [lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, deepcopy(rows[0])),
                     lambda rows: rows[0].update(choice="0"), lambda rows: rows[4].update(response_time="1"),
                     lambda rows: rows[0].update(gain="11"), lambda rows: rows[0].update(task_version="ER"),
                     lambda rows: rows[0].update(response_time="4.1")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = deepcopy(self.rows)
                mutation(rows)
                with self.assertRaises(ValueError):
                    observed.validate_rows(rows)
        code, report, calls, _, _ = self.exercise(failure="source")
        self.assertEqual((code, calls, report["source_validated"]), (1, [], False))

    def test_both_primary_splits_exclude_heldout_responses_from_training(self):
        for direction in (0, 1):
            people = observed.validate_rows(deepcopy(self.rows))
            training, held = observed.split_people(people, direction)
            original_training = deepcopy(training)
            for person in held:
                self.assertEqual((len(training[person]["rows"]), len(held[person]["rows"])), (128, 128))
                train_keys = {(r["run"], r["trial_order"]) for r in training[person]["rows"]}
                held_keys = {(r["run"], r["trial_order"]) for r in held[person]["rows"]}
                self.assertFalse(train_keys & held_keys)
                held[person]["rows"][0].update(choice="-1", response_time="")
            self.assertEqual(training, original_training)

    def test_six_models_fresh_original_fits_and_128_over_256_scoring(self):
        for index, model in enumerate(observed.MODELS):
            with self.subTest(model=model):
                code, report, calls, _, trial_rows = self.exercise(model=model, direction=index % 2)
                self.assertEqual(code, 0)
                self.assertEqual((calls.count("fit"), calls.count("score")), (1, 108))
                self.assertTrue(report["direction_complete"] and report["workers_cleaned_up"])
                self.assertEqual(len(trial_rows), 13824)
                self.assertEqual(report["implementation"], self.spec)
                self.assertTrue(all(r["log_sum"] == -256. and r["contribution_sum_over_256"] == -1.
                                    for r in report["participant_scores"].values()))
                self.assertEqual({int(row["run"]) for row in trial_rows}, set(observed.HALVES[index % 2][1]))
                self.assertEqual(report["model_comparison"], "not_evaluated_requires_both_directions_all_108")

    def test_failures_preserve_fit_or_child_progress_without_scores(self):
        for failure in ("nonconverged", "worker", "score"):
            with self.subTest(failure=failure):
                code, report, calls, _, _ = self.exercise(failure=failure)
                self.assertEqual(code, 1)
                self.assertFalse(report["scores_available"] or report["direction_complete"])
                self.assertNotIn("participant_scores", report)
                self.assertTrue(report["workers_cleaned_up"])
                self.assertEqual(calls.count("fit"), 1)
                if failure == "worker":
                    self.assertEqual(report["parallel_failure_progress"]["objective_history"], [100.])
                else:
                    self.assertIsNotNone(report["fit"])
                if failure != "score":
                    self.assertNotIn("score", calls)

    def test_expiry_in_handler_restoration_or_final_publication_retracts_scores(self):
        for failure in ("restoration", "publication"):
            code, report, calls, snapshots, trial_rows = self.exercise(failure=failure)
            self.assertEqual(code, 1)
            self.assertEqual(calls.count("score"), 108)
            self.assertFalse(report["scores_available"] or report["direction_complete"])
            self.assertNotIn("participant_scores", report)
            self.assertIsNotNone(report["fit"])
            self.assertTrue(report["workers_cleaned_up"])
            self.assertEqual(len(trial_rows), 13824)  # Evidence retained, explicitly unavailable.
            self.assertEqual(snapshots[-1]["status"], "technical_failure")

    def test_environment_and_cap_refusal_before_numerical_imports(self):
        good = {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "4", **{k: "1" for k in observed.THREAD_ENV}}
        for changes, host in (({}, "sh04-ln04"), ({"SLURM_JOB_ID": ""}, "compute"),
                              ({"SLURM_CPUS_PER_TASK": "1"}, "compute"), ({"OMP_NUM_THREADS": "4"}, "compute")):
            with patch.dict(os.environ, {**good, **changes}, clear=True), patch.object(observed.socket, "gethostname", return_value=host):
                with self.assertRaises(RuntimeError):
                    observed.require_environment("fit")
        args = SimpleNamespace(command="fit", max_seconds=1801)
        with patch.object(observed, "require_environment", return_value={}), patch.object(observed, "load_helpers") as loader:
            with self.assertRaises(ValueError):
                observed.run(args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
