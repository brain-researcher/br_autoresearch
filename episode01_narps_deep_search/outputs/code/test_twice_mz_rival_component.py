"""Focused indexed component checks; no scientific fits/solves, compute only."""
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

import run_twice_mz_rival_component as component
import test_mz_twice_wave as fixtures


class TwiceRivalComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TwiceWaveTests.setUpClass()
        cls.fixture = fixtures.TwiceWaveTests(
            "test_both_endpoints_preserve_seed_truth_four_fits_scoring_and_every_checkpoint_scope")
        cls.helpers = component.pilot.load_helpers()
        cls.spec = deepcopy(cls.fixture.spec)
        cls.templates = {}
        for index in (1, 2, 199):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "original"
                source, *_ = cls.fixture.exercise(output, index)
                source["job_id"] = str(46606307 if index == 1 else 46606413 + index - 2)
                cls.templates[index] = {
                    "source": source,
                    "truth": json.loads((output / "generating_parameters.json").read_text()),
                    "events": (output / "simulated_events.csv.gz").read_bytes(),
                    "people": cls.helpers.shared.load_cohort(output / "simulated_events.csv.gz")}

    def template(self, index=1):
        return deepcopy(self.templates[index])

    def exercise(self, model="MC", index=1, *, source_change=None, replay_error=False,
                 nonconverged_second=False, worker_failure_second=False, expiry=None):
        item = self.template(index)
        source, truth, people, events = (item[key] for key in ("source", "truth", "people", "events"))
        if source_change:
            source_change(source)
        calls, checkpoints, now = [], [], [100.]
        original_writer = component.write_report
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual((spec, rival), (self.spec, "fixed"))
            self.assertEqual({len(values) for values in data.values()}, {128})
            self.assertTrue(np.all(data["choice"] == 1))
            focal = tuple(names[3:])
            score = {(): -1., ("rho",): -.98, ("criterion",): -.985,
                     ("eta",): -.995, ("eta", "rho"): -.975, ("eta", "criterion"): -.980}[focal]
            return np.full(128, score + ((1e-4 if replay_error else 3e-13) if focal in ((), ("rho",)) else 0.))

        def fitter(training, fitted, rival, spec, *, workers, max_seconds, progress):
            self.assertEqual((fitted, rival, spec, workers), (model, "fixed", self.spec, 4))
            self.assertGreater(max_seconds, 0.)
            self.assertLessEqual(max_seconds, 5395.)
            self.assertEqual(sorted(training), sorted(people))
            self.assertTrue(all(len(record["rows"]) == 128 for record in training.values()))
            runs = sorted({int(row["run"]) for record in training.values() for row in record["rows"]})
            calls.append((fitted, runs, deepcopy(spec), max_seconds))
            if worker_failure_second and len(calls) == 2:
                error = TimeoutError("mocked active participant failure")
                error.parallel_progress = {"workers_cleaned_up": True, "stage": "conditional", "sweep": 2,
                    "completed_results": {}, "pending_participants": sorted(training), "worker_pids": [1, 2, 3, 4]}
                raise error
            progress({"stage": "sweep", "sweep": 2, "objective": 100., "converged": True,
                      "subject_optimizer_failures": []})
            fit = self.fixture.fixture.fixture.fit(model, converged=not (nonconverged_second and len(calls) == 2))
            fit["support_contacts"] = [{"scope": "fixture", "parameter": "kappa", "bound": "lower"}]
            if not fit["converged"]:
                fit["subject_optimizer_failures"] = [{"participant": "p000", "message": "mocked optimizer failure"}]
            return fit

        def writer(output, report):
            checkpoints.append(deepcopy(report))
            original_writer(output, report)
            if expiry == "staged" and report["active_fit"] is not None and report["fresh_fit_calls"] == 0:
                now[0] = 5501.
            if expiry == "publication" and report["rival_component_complete"]:
                now[0] = 5501.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 5501.

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            episode = Path(directory)
            stack.enter_context(patch.object(component, "EPISODE", episode))
            source_path = component.source_report(index)
            source_path.parent.mkdir(parents=True)
            source_path.write_text(json.dumps(source, allow_nan=False))
            (source_path.parent / "generating_parameters.json").write_text(json.dumps(truth, allow_nan=False))
            (source_path.parent / "simulated_events.csv.gz").write_bytes(events)
            implementation = episode / "original_implementation.json"
            implementation.write_text(json.dumps(self.spec))
            output = episode / "outputs/runs/new_component"
            source_bytes = {path.name: path.read_bytes() for path in source_path.parent.iterdir()}
            stack.enter_context(patch.object(component, "IMPLEMENTATION", implementation))
            stack.enter_context(patch.object(component, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**{name: "1" for name in component.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}))
            stack.enter_context(patch.object(component.pilot, "load_helpers", return_value=self.helpers))
            stack.enter_context(patch.object(component.time, "monotonic", side_effect=lambda: now[0]))
            stack.enter_context(patch.object(component.signal, "alarm", side_effect=alarm))
            stack.enter_context(patch.object(component, "write_report", side_effect=writer))
            stack.enter_context(patch.object(self.helpers.parallel, "fit_partially_pooled_parallel", side_effect=fitter))
            stack.enter_context(patch.object(self.helpers.shared, "trial_likelihood", side_effect=likelihood))
            report = component.run(argparse.Namespace(model=model, replicate_index=index,
                implementation=implementation, output_dir=output, max_seconds=5400))
            self.assertEqual(report, json.loads((output / component.REPORT_NAME).read_text()))
            self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in source_path.parent.iterdir()})
            self.assertEqual(json.loads(implementation.read_text()), self.spec)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        return report, calls, checkpoints

    def test_exact_routes_and_true_index_identity_not_relabelled_representative(self):
        for index, array in ((1, 46606307), (2, 46606413), (199, 46606413)):
            self.assertEqual(component.source_report(index).relative_to(component.EPISODE).as_posix(),
                f"outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-SESOI-wave-{array}/replicate{index}/mz_effect_level_recovery.json")
            item = self.template(index)
            component.validate_source(item["source"], item["truth"], item["people"], self.spec, index, self.helpers)
            for change in (lambda row: row.update(replicate_index=0), lambda row: row.update(seed=910031),
                           lambda row: row.update(effect_level="half")):
                changed = deepcopy(item["source"])
                change(changed)
                with self.assertRaises(ValueError):
                    component.validate_source(changed, item["truth"], item["people"], self.spec, index, self.helpers)
        for index in (0, 200, True, -1):
            with self.assertRaises(ValueError):
                component.source_report(index)

    def test_all_four_models_get_two_fresh_halves_original500_and_exact_saved_scores(self):
        for model, index in (("MC", 1), ("MV", 2), ("MVZ", 199), ("MVC", 199)):
            with self.subTest(model=model, index=index):
                report, calls, _ = self.exercise(model, index)
                baseline = self.templates[index]["source"]
                self.assertTrue(report["rival_component_complete"])
                self.assertEqual(report["status"], "rival_component_completed")
                self.assertEqual([(fitted, runs) for fitted, runs, _, _ in calls], [(model, [1, 4]), (model, [2, 3])])
                self.assertEqual((report["replicate_index"], report["seed"]), (index, 1900057 + 10007 * index))
                self.assertEqual(report["source_job_id"], baseline["job_id"])
                self.assertEqual(report["source_finished_utc"], baseline["finished_utc"])
                self.assertEqual(report["completed_rivals"], [model])
                self.assertEqual(report["fresh_fit_calls"], 2)
                self.assertGreater(report["saved_score_replay_max_absolute_discrepancy"], 0.)
                self.assertLessEqual(report["saved_score_replay_max_absolute_discrepancy"], 1e-12)
                for person, saved in baseline["participant_scores"].items():
                    for key in ("M0", "MZ", "version"):
                        self.assertEqual(report["participant_scores"][person][key], saved[key])
                    self.assertEqual(report["participant_scores"][person][model],
                        sum(row["participant_contributions"][person] for row in report["scored_components"]))
                self.assertTrue(all(row["heldout_trials_per_participant"] == 128 and row["per_participant_denominator"] == 256
                                    for row in report["scored_components"]))
                self.assertEqual(report["scored_primary_directions"], [{"training_runs": list(train), "scoring_runs": list(held)}
                    for train, held in self.helpers.core.HALVES])
                self.assertTrue(report["parallel_workers_cleaned_up"])
                self.assertTrue(all(fit["support_contacts"] for fit in report["fits"]))
                self.assertFalse(report["simulation_regenerated"] or report["saved_M0_MZ_refitted"] or report["automatic_retry"])
                self.assertNotIn("nested_comparisons", report)
                self.assertEqual(report["family_selection"], "not_evaluated")

    def test_second_fit_nonconvergence_retains_first_score_and_failed_fit_without_complete_model(self):
        report, calls, _ = self.exercise(nonconverged_second=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(report["fits"]), 2)
        self.assertFalse(report["fits"][1]["converged"])
        self.assertTrue(report["fits"][1]["subject_optimizer_failures"])
        self.assertEqual(len(report["scored_components"]), 1)
        self.assertEqual(report["completed_rivals"], [])
        self.assertEqual(set(report["participant_scores"]["p000"]), {"M0", "MZ", "version"})
        self.assertFalse(report["rival_component_complete"])
        self.assertEqual(report["status"], "technical_rival_component_failure")

    def test_worker_error_keeps_cleanup_and_complete_first_half(self):
        report, _, _ = self.exercise(worker_failure_second=True)
        self.assertEqual(len(report["fits"]), 1)
        self.assertEqual(len(report["scored_components"]), 1)
        self.assertTrue(report["parallel_failure"]["workers_cleaned_up"])
        self.assertTrue(report["parallel_workers_cleaned_up"])
        self.assertFalse(report["rival_component_complete"])
        self.assertEqual(len(report["parallel_failure"]["pending_participants"]), 108)

    def test_source_or_replay_failure_never_enters_new_fitter(self):
        for options in ({"source_change": lambda row: row.update(seed=1900057)}, {"replay_error": True}):
            with self.subTest(options=options):
                report, calls, _ = self.exercise(**options)
                self.assertEqual(calls, [])
                self.assertFalse(report["rival_component_complete"])
                self.assertEqual(report["fits"], [])

    def test_staged_cleanup_and_final_publication_deadlines_preserve_truthful_prefix(self):
        for stage, fits in (("staged", 0), ("cleanup", 2), ("publication", 2)):
            with self.subTest(stage=stage):
                report, calls, checkpoints = self.exercise(expiry=stage)
                self.assertEqual(len(calls), fits)
                self.assertEqual(len(report["fits"]), fits)
                self.assertFalse(report["rival_component_complete"])
                self.assertFalse(checkpoints[-1]["rival_component_complete"])
                if stage == "publication":
                    self.assertTrue(checkpoints[-2]["rival_component_complete"])

    def test_resource_index_model_and_path_guards_precede_legacy_import(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(component, "require_compute"), \
                patch.dict(os.environ, {**{name: "1" for name in component.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}), \
                patch.object(component.pilot, "load_helpers", side_effect=AssertionError("No legacy import")) as loader:
            args = argparse.Namespace(model="MC", replicate_index=1, implementation=component.IMPLEMENTATION,
                                      output_dir=Path(directory) / "new", max_seconds=5400)
            for name, value in (("SLURM_CPUS_PER_TASK", "1"), ("BLIS_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "8")):
                with self.subTest(name=name), patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    component.run(args)
            for key, value in (("replicate_index", 0), ("replicate_index", 200), ("model", "MZ"),
                               ("max_seconds", 5401), ("max_seconds", True), ("implementation", Path("other.json"))):
                changed = argparse.Namespace(**vars(args))
                setattr(changed, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    component.run(changed)
            with self.assertRaises(FileExistsError):
                component.run(args)
            args.output_dir = component.source_report(199).parent / "forbidden_new_child"
            with self.assertRaises(FileExistsError):
                component.run(args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
