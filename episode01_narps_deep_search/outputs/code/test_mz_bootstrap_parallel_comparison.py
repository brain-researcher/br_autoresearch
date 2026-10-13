"""Source-free orchestration checks; run only in the coordinated allocation."""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_mz_bootstrap_parallel_comparison as comparison

# Reuse authored whole-person fixtures; no old unittest class is imported into
# this module's discovery namespace and no original scientific source is read.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_mz_bootstrap_pilot as previous_tests


class BootstrapParallelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_tests.MZBootstrapPilotTests.setUpClass()
        cls.previous = previous_tests.MZBootstrapPilotTests("test_no_other_source_or_extra_draw_index_interface")
        cls.spec, cls.halves, cls.versions = cls.previous.spec, cls.previous.halves, cls.previous.versions
        cls.legacy = previous_tests.core

    def inputs(self):
        source = previous_tests.report_fixture(self.spec)
        draws = self.legacy.paired_draw(self.versions)
        fits = []
        for half in self.halves:
            training = self.legacy.resampled_half(half, draws)
            fit = self.previous.fake_fit(training, "MZ", "fixed", self.spec)
            fit["elapsed_seconds"] = 220. if not fits else 487.
            fits.append(fit)
        reference = {"phase": "MZ_SESOI_participant_bootstrap_resource_pilot",
            "status": "paired_refit_draw_completed", "job_id": "46547708",
            "finished_utc": comparison.SERIAL_FINISHED, "source_report": str(comparison.SOURCE_REPORT),
            "source_events": str(comparison.SOURCE_REPORT.parent / "simulated_events.csv.gz"),
            "source_replicate_index": 0, "source_seed": 910031, "bootstrap_index": 0,
            "seed_sequence_components": [1200037, 910031, 0], "implementation": deepcopy(self.spec),
            "bootstrap_draw_complete": True, "same_draw_in_both_halves": True,
            "bootstrap_participants": 108, "bootstrap_draw_count_selected": 1,
            "interval_bootstrap_count_selected": None, "interval_convention_selected": None,
            "full_recovery_gate": "not_evaluated", "observed_choices_or_RTs_used": False,
            "new_responses_simulated": False, "heldout_scoring_performed": False,
            "source_files_modified": False, "interval_coverage_evaluated": False,
            "group_interval_coverage_qualified": False, "participant_draws": draws,
            "unique_source_participants_by_version": {version: len({row["source_participant"] for row in draws
                if row["task_version"] == version}) for version in ("EI", "ER")},
            "fits": fits, "elapsed_seconds": 708.3480217889883, "total_fit_seconds": 707.,
            "physical_rejection_displacement": self.legacy.physical_group_effects(fits, draws)}
        return source, reference

    def args(self, output, **changes):
        fields = dict(source_report=comparison.SOURCE_REPORT, serial_report=comparison.SERIAL_REPORT,
                      implementation=comparison.IMPLEMENTATION, output_dir=Path(output), max_seconds=900)
        fields.update(changes)
        return argparse.Namespace(**fields)

    def exercise(self, output, *, reference_change=None, source_change=None, fit_change=None,
                 nonconverged_at=None, worker_error_at=None, deadline_stage=None, checkpoint_delay=0.):
        source, reference = self.inputs()
        if reference_change:
            reference_change(reference)
        if source_change:
            source_change(source)
        before = deepcopy((source, reference, self.halves, self.spec))
        calls, checkpoints, now = [], [], [100.]
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        real_reader, real_writer, real_alarm = Path.read_text, comparison.write_report, signal.alarm
        delayed = [False]

        def reader(path, *args, **kwargs):
            if path == comparison.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path == comparison.SOURCE_REPORT:
                if deadline_stage == "source":
                    now[0] += 901.
                return json.dumps(source)
            if path == comparison.SERIAL_REPORT:
                return json.dumps(reference)
            return real_reader(path, *args, **kwargs)

        def fitter(training, model, rival, spec, *, workers, max_seconds, progress):
            direction = len(calls)
            self.assertEqual((model, rival, spec, workers), ("MZ", "fixed", self.spec, 4))
            self.assertEqual(training, self.legacy.resampled_half(self.halves[direction], reference["participant_draws"]))
            self.assertEqual(max_seconds, 900. - (now[0] - 100.) - comparison.parallel.CLEANUP_SECONDS - 1.)
            self.assertEqual(len(training), 108)
            calls.append((direction, sorted(training), max_seconds))
            if worker_error_at == len(calls):
                error = RuntimeError("authored worker interruption")
                error.parallel_progress = {"stage": "conditional", "sweep": 1,
                    "workers_cleaned_up": True, "objective_history": [], "pending_participants": sorted(training)}
                raise error
            progress({"stage": "sweep", "sweep": 1, "objective": 100., "converged": False,
                      "subject_optimizer_failures": []})
            result = deepcopy(reference["fits"][direction])
            if nonconverged_at == len(calls):
                result["converged"] = False
            if fit_change:
                fit_change(result, direction)
            return result

        def writer(directory, report):
            checkpoints.append(deepcopy(report))
            real_writer(directory, report)
            if report["active_fit"] is not None and not calls and not delayed[0]:
                now[0] += 901. if deadline_stage == "prefit" else checkpoint_delay
                delayed[0] = True
            if deadline_stage == "finalization" and report["comparison_complete"]:
                now[0] += 901.

        def alarm(seconds):
            result = real_alarm(seconds)
            if seconds == 0 and deadline_stage == "cleanup":
                now[0] += 901.
            return result

        with ExitStack() as stack:
            for context in (patch.object(comparison, "require_compute"),
                    patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4", **{key: "1" for key in comparison.parallel.THREAD_ENV}}),
                    patch.object(comparison, "EPISODE", output.parent.parent.parent),
                    patch.object(Path, "read_text", reader),
                    patch.object(self.legacy, "load_split", return_value=self.halves),
                    patch.object(comparison.parallel, "fit_partially_pooled_parallel", side_effect=fitter),
                    patch.object(comparison, "write_report", side_effect=writer),
                    patch.object(comparison.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(comparison.signal, "alarm", side_effect=alarm)):
                stack.enter_context(context)
            report = comparison.run(self.args(output))
        self.assertEqual((source, reference, self.halves, self.spec), before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        self.assertEqual(json.loads((output / comparison.REPORT_NAME).read_text()), report)
        for record in checkpoints:
            self.assertEqual(record["new_statistical_draws"], 0)
            self.assertTrue(record["serial_bootstrap0_remains_statistical_result"])
            self.assertIsNone(record["interval_bootstrap_count_selected"])
            self.assertIsNone(record["interval_convention_selected"])
            self.assertEqual(record["full_recovery_gate"], "not_evaluated")
            self.assertFalse(any(record[key] for key in ("scientific_equivalence_qualified", "controlled_speedup_established",
                "automatic_core_hour_savings_claimed", "automatic_retry", "observed_choices_or_RTs_used",
                "new_responses_simulated", "heldout_scoring_performed", "source_files_modified",
                "serial_fit_rerun", "saved_fit_parameters_reused", "interval_coverage_evaluated")))
        return report, calls, checkpoints

    def test_exact_saved_draw_two_fresh_fits_and_physical_effect_preserve_source_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, checkpoints = self.exercise(Path(directory) / "outputs/runs/fresh", checkpoint_delay=7.)
        self.assertTrue(report["comparison_complete"])
        self.assertEqual(report["status"], "parallel_comparison_completed")
        self.assertEqual([call[0] for call in calls], [0, 1])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(report["validated_fit_directions"], [0, 1])
        self.assertTrue(report["parallel_workers_cleaned_up"])
        self.assertTrue(report["comparison"]["all_numeric_values_exactly_equal"])
        self.assertTrue(report["comparison"]["engineering_agreement_at_absolute_1e_minus12"])
        self.assertEqual(report["serial_fit_elapsed_seconds"], [220., 487.])
        self.assertAlmostEqual(report["physical_rejection_displacement"]["EI"], .05)
        self.assertAlmostEqual(report["physical_rejection_displacement"]["ER"], .10)
        self.assertTrue(any(len(row["fits"]) > len(row["validated_fit_directions"]) for row in checkpoints))

    def test_changed_serial_membership_identity_spec_effect_or_claim_refuses_before_fitting(self):
        mutations = (
            lambda r: r.update(job_id="other"),
            lambda r: r.update(finished_utc="2026-10-05T02:00:36+00:00"),
            lambda r: r["participant_draws"].reverse(),
            lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r["physical_rejection_displacement"].update(EI=.051),
            lambda r: r.update(interval_bootstrap_count_selected=199),
            lambda r: r["fits"][0]["participants"].reverse(),
            lambda r: r["fits"][1].update(converged=False),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                report, calls, _ = self.exercise(Path(directory) / "outputs/runs/fresh", reference_change=mutation)
            self.assertFalse(report["comparison_complete"])
            self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as directory:
            report, calls, _ = self.exercise(Path(directory) / "outputs/runs/fresh", source_change=lambda r: r.update(seed=910032))
        self.assertEqual(calls, [])
        self.assertFalse(report["comparison_complete"])

    def test_finite_disagreement_and_changed_history_length_are_completed_comparisons(self):
        def changed(fit, direction):
            fit["parameters"][0][-1] += 1e-8
            if direction == 1:
                fit["objective_history"] = [200., 100., 100.]
        with tempfile.TemporaryDirectory() as directory:
            report, calls, _ = self.exercise(Path(directory) / "outputs/runs/fresh", fit_change=changed)
        self.assertTrue(report["comparison_complete"])
        self.assertEqual(len(calls), 2)
        self.assertFalse(report["comparison"]["engineering_agreement_at_absolute_1e_minus12"])
        reverse = report["comparison"]["fits"][1]
        self.assertEqual((reverse["serial_sweeps"], reverse["parallel_sweeps"]), (2, 3))
        self.assertFalse(reverse["numeric_fields"]["objective_history"]["same_shape"])
        self.assertIsNone(reverse["numeric_fields"]["objective_history"]["max_absolute_difference"])
        self.assertTrue(reverse["numeric_fields"]["final_objective"]["exactly_equal"])
        self.assertNotEqual(report["comparison"]["physical_effects"]["EI"]["parallel_minus_serial"], 0.)

    def test_absolute_only_engineering_tolerance_is_not_scaled_by_objective_magnitude(self):
        within = comparison.numeric_comparison(.5 + 5e-13, .5)
        self.assertFalse(within["exactly_equal"])
        self.assertTrue(within["within_absolute_tolerance"])
        outside = comparison.numeric_comparison(8000. + 1e-8, 8000.)
        self.assertFalse(outside["within_absolute_tolerance"])
        self.assertGreater(outside["parallel_minus_serial"], 0.)

    def test_returned_nonconvergence_and_worker_failure_keep_only_completed_fit_prefix(self):
        for options, expected_fits in (({"nonconverged_at": 1}, 1), ({"nonconverged_at": 2}, 2),
                                       ({"worker_error_at": 2}, 1)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                report, calls, _ = self.exercise(Path(directory) / "outputs/runs/fresh", **options)
            self.assertFalse(report["comparison_complete"])
            self.assertEqual(len(report["fits"]), expected_fits)
            self.assertNotIn("physical_rejection_displacement", report)
            self.assertNotIn("comparison", report)
            self.assertTrue(report["parallel_workers_cleaned_up"])
            if "worker_error_at" in options:
                self.assertTrue(report["parallel_failure"]["workers_cleaned_up"])
                self.assertEqual(report["validated_fit_directions"], [0])

    def test_whole_deadline_covers_source_prefit_checkpoint_final_publication_and_cleanup(self):
        for stage, expected_calls in (("source", 0), ("prefit", 0), ("finalization", 2), ("cleanup", 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                report, calls, _ = self.exercise(Path(directory) / "outputs/runs/fresh", deadline_stage=stage)
            self.assertEqual(len(calls), expected_calls)
            self.assertFalse(report["comparison_complete"])
            self.assertEqual(report["status"], "technical_parallel_comparison_failure")
            self.assertNotIn("comparison", report)
            self.assertNotIn("physical_rejection_displacement", report)

    def test_actual_cpu_threads_sources_cap_and_fresh_output_guards_precede_legacy_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(comparison, "require_compute"), \
                patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "4", **{key: "1" for key in comparison.parallel.THREAD_ENV}}), \
                patch.object(self.legacy, "load_split") as loader:
            output = Path(directory) / "outputs/runs/fresh"
            for changes in ({"max_seconds": 901}, {"max_seconds": 0},
                    {"source_report": Path("other-source")}, {"serial_report": Path("other-serial")},
                    {"implementation": Path("other-spec")}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    comparison.run(self.args(output, **changes))
            for changes in ({"SLURM_CPUS_PER_TASK": "1"}, {"SLURM_CPUS_PER_TASK": "8"}, {"OPENBLAS_NUM_THREADS": "4"}):
                with self.subTest(changes=changes), patch.dict(os.environ, changes), self.assertRaises(ValueError):
                    comparison.run(self.args(output))
            with patch.object(comparison, "EPISODE", Path(directory)):
                output.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    comparison.run(self.args(output))
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
