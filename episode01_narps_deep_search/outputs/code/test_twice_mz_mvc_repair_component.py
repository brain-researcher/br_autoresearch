"""Focused indexed numerical-repair checks; execute only on coordinated compute."""
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

import run_twice_mz_mvc_repair_component as repair
import summarize_twice_mz_menu_wave as collector
import test_twice_mz_rival_component as source_tests


class TwiceMVCRepairComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_tests.TwiceRivalComponentTests.setUpClass()
        cls.prior = source_tests.TwiceRivalComponentTests(
            "test_all_four_models_get_two_fresh_halves_original500_and_exact_saved_scores")
        cls.helpers, cls.spec = cls.prior.helpers, deepcopy(cls.prior.spec)
        cls.failures = {}
        for index in (1, 2, 199):
            report = cls.prior.exercise("MVC", index)[0]
            report.update(job_id=str(123400 + index), status="technical_rival_component_failure",
                rival_component_complete=False, error=repair.engine.NONCONVERGENCE, fresh_fit_calls=1,
                completed_rivals=[], scored_components=[], scored_primary_directions=[],
                fits=report["fits"][:1])
            report.pop("total_fresh_fit_seconds", None)
            report["participant_scores"] = {person: {key: row[key] for key in ("version", "M0", "MZ")}
                                           for person, row in report["participant_scores"].items()}
            report["fits"][0].update(converged=False, subject_optimizer_failures=[],
                objective_history=[float(1501 - i) for i in range(500)])
            cls.failures[index] = report

    def exercise(self, index=1, *, source_change=None, failed_change=None, replay_error=False,
                 nonconverged_at=None, worker_error_at=None, expiry=None):
        item = self.prior.template(index)
        source, truth, people, events = (item[key] for key in ("source", "truth", "people", "events"))
        failed = deepcopy(self.failures[index])
        if source_change:
            source_change(source)
        if failed_change:
            failed_change(failed)
        calls, likelihoods, checkpoints, now = [], [], [], [100.]
        original_writer, original_reader = repair.write_report, Path.read_text
        runtime = repair.engine.runtime_specification(self.spec)
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def likelihood(theta, names, data, spec, rival):
            self.assertEqual((spec, rival), (self.spec, "fixed"))
            self.assertEqual({len(values) for values in data.values()}, {128})
            likelihoods.append(tuple(names))
            focal = tuple(names[3:])
            value = {(): -1., ("rho",): -.98, ("eta", "criterion"): -.975}[focal]
            if focal != ("eta", "criterion"):
                value += 1e-4 if replay_error else 3e-13
            return np.full(128, value)

        def fitter(training, model, rival, spec, *, workers, max_seconds, progress):
            self.assertEqual((model, rival, spec, workers), ("MVC", "fixed", runtime, 4))
            self.assertEqual(sorted(training), sorted(people))
            self.assertTrue(all(len(record["rows"]) == 128 for record in training.values()))
            self.assertGreater(max_seconds, 0.)
            self.assertLessEqual(max_seconds, 5395.)
            runs = sorted({int(row["run"]) for record in training.values() for row in record["rows"]})
            calls.append((runs, deepcopy(spec)))
            self.assertEqual(runs, [[1, 4], [2, 3]][len(calls) - 1])
            if worker_error_at == len(calls):
                error = TimeoutError("mocked indexed worker failure")
                error.parallel_progress = {"workers_cleaned_up": True, "stage": "conditional", "sweep": 2,
                    "completed_results": {}, "pending_participants": sorted(training), "worker_pids": [1, 2, 3, 4]}
                raise error
            converged = nonconverged_at != len(calls)
            fit = self.prior.fixture.fixture.fixture.fit("MVC", converged=converged)
            fit["subject_optimizer_failures"] = []
            fit["objective_history"] = ([float(600 - i) for i in range(500)] + [101.]) if converged else [float(1501 - i) for i in range(1500)]
            progress({"stage": "sweep", "sweep": len(fit["objective_history"]),
                "objective": fit["objective_history"][-1], "converged": converged, "subject_optimizer_failures": []})
            return fit

        def writer(output, report):
            checkpoints.append(deepcopy(report))
            original_writer(output, report)
            if expiry == "staged" and report["active_fit"] is not None and report["fresh_fit_calls"] == 0:
                now[0] = 5501.
            if expiry == "publication" and report["mvc_repair_complete"]:
                now[0] = 5501.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 5501.

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            episode = Path(directory)
            stack.enter_context(patch.object(repair, "EPISODE", episode))
            stack.enter_context(patch.object(repair.original_component, "EPISODE", episode))
            stack.enter_context(patch.object(repair, "FAILED_ROOT", episode / "outputs/runs/failed-array-46638796"))
            source_path, failed_path = repair.source_report(index), repair.failed_path(index)
            source_path.parent.mkdir(parents=True)
            failed_path.parent.mkdir(parents=True)
            failed.update(source_report=str(source_path), source_dir=str(source_path.parent))
            source_path.write_text(json.dumps(source, allow_nan=False))
            failed_path.write_text(json.dumps(failed, allow_nan=False))
            (source_path.parent / "generating_parameters.json").write_text(json.dumps(truth))
            (source_path.parent / "simulated_events.csv.gz").write_bytes(events)
            implementation = episode / "original.json"
            implementation.write_text(json.dumps(self.spec))
            stack.enter_context(patch.object(repair, "IMPLEMENTATION", implementation))
            output = episode / "outputs/runs/new-indexed-repair"
            inputs = [source_path, failed_path, implementation,
                source_path.parent / "generating_parameters.json", source_path.parent / "simulated_events.csv.gz"]
            frozen = {path: path.read_bytes() for path in inputs}

            def reader(path, *args, **kwargs):
                if path not in inputs:
                    raise AssertionError(f"No pilot, other index or alternative attempt may be read: {path}")
                result = original_reader(path, *args, **kwargs)
                if expiry == "failed_read" and path == failed_path:
                    now[0] = 5501.
                return result

            for context in (patch.object(repair.engine, "require_compute"),
                    patch.object(collector, "baseline_path", side_effect=lambda selected: source_path if selected == index else self.fail("Different index")),
                    patch.dict(os.environ, {**{name: "1" for name in repair.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}),
                    patch.object(repair.engine.pilot, "load_helpers", return_value=self.helpers),
                    patch.object(repair.engine.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(repair.engine.signal, "alarm", side_effect=alarm),
                    patch.object(repair, "write_report", side_effect=writer), patch.object(Path, "read_text", reader),
                    patch.object(self.helpers.parallel, "fit_partially_pooled_parallel", side_effect=fitter),
                    patch.object(self.helpers.shared, "trial_likelihood", side_effect=likelihood)):
                stack.enter_context(context)
            names = ("EPISODE", "IMPLEMENTATION", "SOURCE_REPORT", "FAILED_REPORT", "PHASE", "REPORT_NAME", "repair_context", "write_report")
            bindings = {name: getattr(repair.engine, name) for name in names}
            validator = repair.engine.pilot.validate_source
            report = repair.run(argparse.Namespace(replicate_index=index, implementation=implementation,
                output_dir=output, max_seconds=5400))
            self.assertEqual(report, json.loads(original_reader(output / repair.REPORT_NAME)))
            self.assertEqual(frozen, {path: path.read_bytes() for path in inputs})
            self.assertEqual(bindings, {name: getattr(repair.engine, name) for name in names})
            self.assertIs(repair.engine.pilot.validate_source, validator)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        for row in checkpoints:
            self.assertEqual((row["phase"], row["replicate_index"], row["seed"], row["selected_replicates"]),
                             (repair.PHASE, index, 1900057 + 10007 * index, [index]))
            self.assertNotIn("index0", row["interpretation"])
            self.assertNotIn("full_menu_wave_selected", row)
            self.assertFalse(row["automatic_retry"] or row["retry_wave_selected"] or row["subsequent_wave_selected"])
            self.assertTrue(all(row[field] == "not_evaluated" for field in repair.engine.CLAIMS))
        return report, calls, likelihoods, checkpoints

    def test_actual_source_routes_and_all_checkpoint_identities_with_two_fresh1500_halves(self):
        for index, array in ((1, 46606307), (2, 46606413), (199, 46606413)):
            with self.subTest(index=index):
                self.assertIn(f"wave-{array}/replicate{index}/", str(repair.source_report(index)))
                self.assertIn(f"wave-46638796/replicate{index}/", str(repair.failed_path(index)))
                report, calls, likelihoods, _ = self.exercise(index)
                self.assertTrue(report["mvc_repair_complete"])
                self.assertEqual(len(calls), 2)
                self.assertEqual([len(fit["objective_history"]) for fit in report["fits"]], [501, 501])
                self.assertEqual(report["implementation"], self.spec)
                changed = deepcopy(report["runtime_implementation"])
                changed["optimizer"]["maximum_sweeps"] = 500
                self.assertEqual(changed, self.spec)
                self.assertEqual(report["numerical_repair"]["previous_MVC_fit"], self.failures[index]["fits"][0])
                self.assertEqual(report["numerical_repair"]["previous_process_elapsed_seconds"], self.failures[index]["elapsed_seconds"])
                self.assertNotIn("cumulative_process_elapsed_seconds", report)
                self.assertEqual(len(likelihoods), 6 * 108)
                for person, row in report["participant_scores"].items():
                    for key in ("version", "M0", "MZ"):
                        self.assertEqual(row[key], self.prior.templates[index]["source"]["participant_scores"][person][key])
                    self.assertEqual(row["MVC"], sum(part["participant_contributions"][person] for part in report["scored_components"]))

    def test_wrong_source_or_failure_cause_and_any_prior_scored_half_are_rejected(self):
        changes = (lambda r: r.update(replicate_index=0), lambda r: r.update(seed=1900057),
            lambda r: r.update(source_job_id="1"), lambda r: r.update(error="Timeout"),
            lambda r: r.update(parallel_workers_cleaned_up=False),
            lambda r: r["fits"][0].update(subject_optimizer_failures=[{"participant": "p000"}]),
            lambda r: r["fits"][0].update(objective_history=[100.] * 500),
            lambda r: r["fits"][0]["objective_history"].pop(),
            lambda r: r["fits"][0].update(training_runs=[2, 3], scoring_runs=[1, 4]),
            lambda r: r["scored_primary_directions"].append({"training_runs": [1, 4], "scoring_runs": [2, 3]}))
        for change in changes:
            with self.subTest(change=change):
                report, calls, _, _ = self.exercise(failed_change=change)
                self.assertEqual(calls, [])
                self.assertFalse(report["repair_context_validated"])
        report, calls, _, _ = self.exercise(source_change=lambda r: r.update(seed=910031))
        self.assertEqual(calls, [])
        self.assertFalse(report["source_baseline_validated"])

    def test_replay_failure_and_first_or_second_nonconvergence_preserve_only_returned_prefix(self):
        report, calls, _, _ = self.exercise(replay_error=True)
        self.assertEqual(calls, [])
        self.assertFalse(report["saved_scores_replayed"])
        for number in (1, 2):
            with self.subTest(number=number):
                report, calls, _, _ = self.exercise(nonconverged_at=number)
                self.assertEqual(len(calls), number)
                self.assertEqual(len(report["fits"]), number)
                self.assertEqual(len(report["scored_components"]), number - 1)
                self.assertFalse(report["mvc_repair_complete"])
                self.assertEqual(report["completed_rivals"], [])

    def test_worker_failure_retains_cleanup_and_forward_score(self):
        report, calls, _, _ = self.exercise(worker_error_at=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(report["fits"]), 1)
        self.assertEqual(len(report["scored_components"]), 1)
        self.assertTrue(report["parallel_workers_cleaned_up"])
        self.assertEqual(report["parallel_failure"]["stage"], "conditional")
        self.assertFalse(report["mvc_repair_complete"])

    def test_read_staging_cleanup_and_finalization_deadlines_keep_indexed_failure_identity(self):
        for stage, count in (("failed_read", 0), ("staged", 0), ("cleanup", 2), ("publication", 2)):
            with self.subTest(stage=stage):
                report, calls, _, checkpoints = self.exercise(expiry=stage)
                self.assertEqual(len(calls), count)
                self.assertFalse(report["mvc_repair_complete"])
                self.assertEqual(report["status"], "technical_mvc_repair_failure")
                self.assertIn("deadline", report["error"].lower())
                self.assertFalse(checkpoints[-1]["mvc_repair_complete"])

    def test_guards_and_binding_restoration_before_any_legacy_import(self):
        for index in (0, 200, True, -1):
            with self.assertRaises(ValueError):
                repair.failed_path(index)
        names = ("EPISODE", "IMPLEMENTATION", "SOURCE_REPORT", "FAILED_REPORT", "PHASE", "REPORT_NAME", "repair_context", "write_report")
        bindings = {name: getattr(repair.engine, name) for name in names}
        validator = repair.engine.pilot.validate_source
        with tempfile.TemporaryDirectory() as directory, patch.object(repair.engine, "require_compute"), \
                patch.dict(os.environ, {**{name: "1" for name in repair.THREAD_ENV}, "SLURM_CPUS_PER_TASK": "4"}), \
                patch.object(repair.engine.pilot, "load_helpers", side_effect=AssertionError("No legacy import")) as loader:
            args = argparse.Namespace(replicate_index=1, implementation=repair.IMPLEMENTATION,
                                      output_dir=Path(directory) / "new", max_seconds=5400)
            for name, value in (("SLURM_CPUS_PER_TASK", "1"), ("BLIS_NUM_THREADS", "2")):
                with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    repair.run(args)
                self.assertEqual(bindings, {key: getattr(repair.engine, key) for key in names})
                self.assertIs(repair.engine.pilot.validate_source, validator)
            for field, value in (("max_seconds", 5401), ("max_seconds", True), ("implementation", Path("other.json"))):
                changed = argparse.Namespace(**vars(args))
                setattr(changed, field, value)
                with self.assertRaises(ValueError):
                    repair.run(changed)
            for output in (args.output_dir, repair.FAILED_ROOT / "new", repair.source_report(199).parent / "new",
                           repair.source_report(1).parent / "new"):
                args.output_dir = output
                with self.assertRaises(FileExistsError):
                    repair.run(args)
            loader.assert_not_called()
        self.assertEqual(bindings, {key: getattr(repair.engine, key) for key in names})
        self.assertIs(repair.engine.pilot.validate_source, validator)


if __name__ == "__main__":
    unittest.main()
