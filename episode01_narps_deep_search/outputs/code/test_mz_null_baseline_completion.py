"""Focused index35 prefix-reuse checks; execute only in a compute allocation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_mz_null_baseline_completion as completion
import test_mz_null_rival as fixtures


class NullBaselineCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullRivalTests.setUpClass()
        cls.fixture = fixtures.NullRivalTests("test_only_completed_fixed_null_source_is_accepted")
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth

    def source(self):
        source = deepcopy(self.fixture.source())
        source.update(phase=completion.indexed.PHASE, replicate_index=35, seed=1650282,
                      job_id=completion.SOURCE_JOB_ID, finished_utc=completion.SOURCE_FINISHED_UTC,
                      status="technical_representative_failure", pairwise_diagnostic_complete=False,
                      error=completion.SOURCE_ERROR, internal_seconds=2700,
                      elapsed_seconds=2698.6008067280054, simulation_seconds=2.,
                      selected_replicates=[35], wave_selected=True,
                      wave_scope=completion.indexed.WAVE_SCOPE, fixed_cohort_count=200,
                      full_menu_wave_selected=False, retry_selected=False,
                      scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
        source.pop("participant_scores")
        source["fits"] = source["fits"][:3]
        for fit, seconds, sweeps in zip(source["fits"], (579.94387, 940.72698, 562.84932), (292, 319, 286)):
            fit.update(elapsed_seconds=seconds, objective_history=[float(i) for i in range(sweeps - 1, 0, -1)] + [1.])
        return source

    def arguments(self, output, **changes):
        values = dict(implementation=completion.IMPLEMENTATION, output_dir=Path(output), max_seconds=2700)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_context_validates_original_three_fit_prefix_without_changing_history_or_cost(self):
        source = self.source()
        before = deepcopy(source)
        fits, metadata = completion.completion_context(source, self.truth, self.people, self.spec)
        self.assertEqual(source, before)
        self.assertEqual(fits, source["fits"])
        self.assertIsNot(fits, source["fits"])
        self.assertIsNot(fits[0]["parameters"], source["fits"][0]["parameters"])
        self.assertEqual(metadata["previous_process_elapsed_seconds"], source["elapsed_seconds"])
        self.assertEqual([row["original_fit_elapsed_seconds"] for row in metadata["per_fit_provenance"][:3]],
                         [fit["elapsed_seconds"] for fit in source["fits"]])
        self.assertTrue(all(row["validation_maximum_sweeps"] == 500 for row in metadata["per_fit_provenance"]))
        self.assertEqual(metadata["per_fit_provenance"][3]["mode"], "fresh_fit_original_initialization")

    def test_only_exact_diagnosed_worker_and_failed_prefix_are_admitted(self):
        changes = (("job_id", "46580479"), ("phase", "other"), ("replicate_index", 36), ("seed", 1650283),
                   ("generating_model", "MZ"), ("finished_utc", "2026-10-05T06:46:39+00:00"),
                   ("internal_seconds", 5400), ("status", "running"), ("pairwise_diagnostic_complete", True),
                   ("error", "RuntimeError: Nonconverged fit is technical; it is not a recovery failure"),
                   ("scored_primary_directions", []), ("participant_scores", {}), ("nested_comparison", {}),
                   ("implementation", {}), ("wave_scope", "full_menu"), ("full_recovery_pass", True),
                   ("simulated_deadline_misses", 1), ("elapsed_seconds", 100.), ("simulation_seconds", float("nan")))
        for field, value in changes:
            with self.subTest(field=field), self.assertRaises(ValueError):
                completion.completion_context({**self.source(), field: value}, self.truth, self.people, self.spec)
        for count in (2, 4):
            source = self.source()
            source["fits"] = source["fits"][:count] if count == 2 else source["fits"] + [self.fixture.fit("MZ", (2, 3), (1, 4))]
            with self.subTest(count=count), self.assertRaises(ValueError):
                completion.completion_context(source, self.truth, self.people, self.spec)

    def test_all_preserved_fits_use_original500_actual_people_and_centered_truth(self):
        for position in range(3):
            for field, value in (("participants", ["different"] * 108), ("converged", False),
                                 ("subject_optimizer_failures", ["p000"]), ("objective_history", [1.] * 501),
                                 ("elapsed_seconds", float("nan"))):
                source = self.source()
                source["fits"][position][field] = value
                with self.subTest(position=position, field=field), self.assertRaises((ValueError, RuntimeError)):
                    completion.completion_context(source, self.truth, self.people, self.spec)
        for parameter in ("eta", "rho", "criterion", "kappa"):
            truth = deepcopy(self.truth)
            truth["p000"][parameter] = .1 if parameter != "kappa" else 1e20
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                completion.completion_context(self.source(), truth, self.people, self.spec)
        spec = deepcopy(self.spec)
        spec["optimizer"]["maximum_sweeps"] = 1000
        source = self.source()
        source["implementation"] = spec
        with self.assertRaises(ValueError):
            completion.completion_context(source, self.truth, self.people, spec)

    def exercise(self, output, *, source_bad=False, score_bad=False, new_fit_error=False, nonconverged=False):
        source = self.source()
        if source_bad:
            source["job_id"] = "46580479"
        original = deepcopy(source)
        reads, calls, scores, checkpoints = [], [], [], []
        allowed = {completion.IMPLEMENTATION: json.dumps(self.spec),
                   completion.SOURCE / completion.SOURCE_REPORT: json.dumps(source),
                   completion.SOURCE / "generating_parameters.json": json.dumps(self.truth)}
        writer = completion.write_report
        halves = {tuple(train): ({"training": list(train)}, {"scoring": list(held)})
                  for train, held in completion.HALVES}

        def reader(path, *args, **kwargs):
            reads.append(path)
            return allowed[path]

        def split(path, train, held):
            self.assertEqual(path, completion.SOURCE / "simulated_events.csv.gz")
            self.assertEqual(halves[tuple(train)][1], {"scoring": held})
            return halves[tuple(train)]

        def fitter(training, model, rival, spec):
            calls.append((deepcopy(training), model, rival, deepcopy(spec)))
            if new_fit_error:
                raise RuntimeError("test signal interruption inside unfinished new fit")
            return self.fixture.fit(model, converged=not nonconverged)

        def scorer(fit, scoring, spec, check):
            check()
            self.assertEqual(scoring, {"scoring": fit["scoring_runs"]})
            self.assertEqual(spec, self.spec)
            scores.append(deepcopy(fit))
            if score_bad:
                raise ValueError("test invalid preserved-fit score")
            return {person: (-1. if fit["model"] == "M0" else -.99) for person in self.people}

        def checkpoint(directory, report):
            checkpoints.append(deepcopy(report))
            writer(directory, report)

        with ExitStack() as stack:
            stack.enter_context(patch.object(completion, "require_compute"))
            stack.enter_context(patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "999"}))
            stack.enter_context(patch.object(Path, "read_text", reader))
            loader = stack.enter_context(patch.object(completion.null.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(completion, "load_split", side_effect=split))
            stack.enter_context(patch.object(completion, "fit_partially_pooled", side_effect=fitter))
            stack.enter_context(patch.object(completion.null.shared, "score_fit", side_effect=scorer))
            stack.enter_context(patch.object(completion, "write_report", side_effect=checkpoint))
            simulator = stack.enter_context(patch.object(completion.indexed.representative, "simulate_null", side_effect=AssertionError("No resimulation")))
            report = completion.run(self.arguments(output))
            simulator.assert_not_called()
            loader.assert_called_once()
            self.assertEqual(loader.call_args.args[0], completion.SOURCE / "simulated_events.csv.gz")
        self.assertEqual(source, original)
        return report, calls, scores, checkpoints, source, reads

    def assert_scope(self, report):
        self.assertEqual((report["phase"], report["generating_model"], report["replicate_index"], report["seed"]),
                         (completion.PHASE, "M0", 35, 1650282))
        self.assertEqual(report["selected_replicates"], [35])
        for key in ("simulation_regenerated", "simulation_copied", "saved_completed_fits_refitted", "unfinished_fit_state_reused",
                    "wave_selected", "retry_wave_selected", "full_menu_wave_selected", "episode_terminal"):
            self.assertFalse(report[key])
        for key in completion.null.CLAIM_FIELDS:
            self.assertEqual(report[key], "not_evaluated")
        self.assertIsNone(report["full_recovery_pass"])
        self.assertEqual(report["fixed_cohort_count"], 200)
        self.assertEqual(report["saved_score_agreement"], "not_evaluated_original_failure_has_no_saved_scores")

    def test_only_reverse_MZ_is_new_all_four_scores_recomputed_and_old_cost_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "complete"
            report, calls, scores, checkpoints, source, reads = self.exercise(output)
            self.assertTrue(report["pairwise_diagnostic_complete"])
            self.assertEqual(calls, [({"training": [2, 3]}, "MZ", "fixed", self.spec)])
            self.assertEqual(scores[:3], source["fits"])
            self.assertEqual(report["fits"][:3], source["fits"])
            self.assertEqual([(fit["model"], fit["training_runs"], fit["scoring_runs"]) for fit in scores],
                             [(model, list(train), list(held)) for model, train, held in completion.EXPECTED])
            self.assertTrue(report["preserved_scores_recomputed"])
            self.assertEqual(report["implementation"], self.spec)
            self.assertEqual(report["cumulative_process_elapsed_seconds"], source["elapsed_seconds"] + report["elapsed_seconds"])
            self.assertEqual(report["previous_process_elapsed_seconds"], source["elapsed_seconds"])
            self.assertEqual(report["participant_scores"]["p000"]["M0"], -2.)
            self.assertEqual(report["participant_scores"]["p000"]["MZ"], -1.98)
            self.assertEqual(report["nested_comparison"], completion.fixed.nested_contrast(report["participant_scores"], "MZ", "M0"))
            self.assertEqual(reads, [completion.IMPLEMENTATION, completion.SOURCE / completion.SOURCE_REPORT,
                                     completion.SOURCE / "generating_parameters.json"])
            self.assertEqual(sorted(path.name for path in output.iterdir()), [completion.REPORT_NAME])
            self.assertEqual(json.loads((output / completion.REPORT_NAME).read_text()), report)
            self.assertGreater(len(checkpoints), 6)
            for checkpoint in checkpoints:
                self.assert_scope(checkpoint)
            # The old pool must continue to reject the distinct completion phase.
            with self.assertRaises(ValueError):
                completion.baseline.validate_identity(report, 35, self.spec)

    def test_source_or_rescore_failure_never_starts_a_new_fit_and_has_no_scientific_result(self):
        with tempfile.TemporaryDirectory() as directory:
            for failure in ("source_bad", "score_bad"):
                report, calls, scores, checkpoints, source, reads = self.exercise(Path(directory) / failure, **{failure: True})
                self.assertEqual(calls, [])
                self.assertFalse(report["pairwise_diagnostic_complete"])
                self.assertEqual(report["status"], "technical_baseline_completion_failure")
                self.assertNotIn("participant_scores", report)
                self.assertNotIn("nested_comparison", report)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint)

    def test_unfinished_or_nonconverged_new_fit_preserves_prefix_without_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            for failure, expected_fits in (("new_fit_error", 3), ("nonconverged", 4)):
                report, calls, scores, checkpoints, source, reads = self.exercise(Path(directory) / failure, **{failure: True})
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(scores), 3)
                self.assertEqual(report["fits"][:3], source["fits"])
                self.assertEqual(len(report["fits"]), expected_fits)
                self.assertFalse(report["pairwise_diagnostic_complete"])
                self.assertNotIn("participant_scores", report)
                self.assertNotIn("nested_comparison", report)
                self.assertEqual(report["scored_primary_directions"], [{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
                self.assertEqual(report["cumulative_process_elapsed_seconds"], source["elapsed_seconds"] + report["elapsed_seconds"])
                if failure == "new_fit_error":
                    self.assertGreaterEqual(report["unfinished_fresh_fit_elapsed_seconds"], 0.)
                for checkpoint in checkpoints:
                    self.assert_scope(checkpoint)

    def test_path_cpu_cap_and_fresh_output_guards_precede_any_read(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(completion, "require_compute"), \
                patch.dict(os.environ, {**fixtures.THREADS, "SLURM_CPUS_PER_TASK": "1"}), \
                patch.object(Path, "read_text", side_effect=AssertionError("No source read before guards")):
            for changes in ({"implementation": Path(directory) / "wrong.json"}, {"max_seconds": 0},
                            {"max_seconds": 2701}, {"max_seconds": True}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    completion.run(self.arguments(Path(directory) / "fresh", **changes))
            for key in ("SLURM_CPUS_PER_TASK", *fixtures.THREADS):
                with self.subTest(key=key), patch.dict(os.environ, {key: "2"}), self.assertRaises(ValueError):
                    completion.run(self.arguments(Path(directory) / "fresh"))
            for output in (Path(directory), completion.SOURCE / "child", completion.DESIGN_SOURCE.parent / "child",
                           completion.indexed.PILOT_DIRECTORY / "child"):
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    completion.run(self.arguments(output))


if __name__ == "__main__":
    unittest.main()
