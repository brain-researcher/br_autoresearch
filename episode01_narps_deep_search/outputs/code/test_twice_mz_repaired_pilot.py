"""Real producer-shaped synthetic fixtures; coordinator-owned compute only."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import summarize_twice_mz_repaired_pilot as pool
import test_twice_mz_mvc_repair_pilot as producer_tests


class TwiceRepairedPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        producer_tests.TwiceMVCRepairPilotTests.setUpClass()
        cls.fixture = producer_tests.TwiceMVCRepairPilotTests(
            "test_two_fresh_runtime1500_halves_preserve_original500_and_completed_prefix_exactly")
        cls.helpers = pool.load_helpers()
        cls.spec, cls.base, cls.failed = deepcopy((cls.fixture.spec, cls.fixture.source, cls.fixture.failed))
        cls.complete = cls.report()

    @classmethod
    def report(cls, **options):
        report = cls.fixture.exercise(**options)[0]
        # Fixture runner uses temporary files. Normalize only path/job metadata
        # back to the exact selected production identities, never scientific data.
        report.update(job_id=pool.REPAIR_JOB, source_report=str(pool.BASE_REPORT),
            source_dir=str(pool.BASE_REPORT.parent), failed_report=str(pool.FAILED_REPORT))
        if report.get("numerical_repair") is not None:
            report["numerical_repair"]["failed_report"] = str(pool.FAILED_REPORT)
        return report

    def build(self, report=None, base=None, failed=None):
        inputs = deepcopy((self.base if base is None else base, self.failed if failed is None else failed,
                           self.complete if report is None else report, self.spec))
        before = deepcopy(inputs)
        result = pool.build_summary(*inputs, self.helpers)
        self.assertEqual(inputs, before)
        return result

    def assert_scope(self, result):
        self.assertEqual((result["replicate_index"], result["seed"], result["generating_z"]), (0, 1900057, .40))
        self.assertEqual(result["selected_replicates"], [0])
        for field in ("family_selection", "family_recovery_rate", "confusion_rate", "group_recovery",
                      "interval_coverage", "full_gate", "full_menu_selection"):
            self.assertEqual(result[field], "not_evaluated")
        for field in ("new_fits", "new_simulation", "source_events_read", "saved_scores_rescored", "wave_selected", "automatic_retry"):
            self.assertFalse(result[field])
        self.assertIsNone(result["full_recovery_pass"])
        self.assertNotIn("cumulative_process_elapsed_seconds", result)

    def invoke(self, *, report=None, missing=None, expiry=None):
        documents = {pool.IMPLEMENTATION: self.spec, pool.BASE_REPORT: self.base,
            pool.FAILED_REPORT: self.failed, pool.REPAIR_REPORT: self.complete if report is None else report}
        frozen, reads, now = deepcopy(documents), [], [100.]
        reader, writer = Path.read_text, pool.write_report
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

        def read(path, *args, **kwargs):
            reads.append(path)
            if path not in documents:
                raise AssertionError(f"No events, truth, unrelated cohort or future wave reads: {path}")
            if path == missing:
                raise FileNotFoundError(str(path))
            if expiry == "repair_read" and path == pool.REPAIR_REPORT:
                now[0] = 341.
            return json.dumps(documents[path], allow_nan=False)

        def write(path, result):
            writer(path, result)
            if expiry == "publication" and result["menu_complete"]:
                now[0] = 341.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 341.

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "summary.json"
            for context in (patch.object(pool, "require_compute"),
                    patch.object(pool, "load_helpers", return_value=self.helpers),
                    patch.dict(os.environ, {name: "1" for name in pool.THREAD_ENV}),
                    patch.object(Path, "read_text", read), patch.object(pool, "write_report", side_effect=write),
                    patch.object(pool.signal, "alarm", side_effect=alarm),
                    patch.object(pool.time, "monotonic", side_effect=lambda: now[0]), redirect_stdout(io.StringIO())):
                stack.enter_context(context)
            code = pool.main(["--repair-report", str(pool.REPAIR_REPORT), "--implementation", str(pool.IMPLEMENTATION),
                              "--output", str(output), "--max-seconds", "240"])
            result = json.loads(reader(output))
        self.assertEqual(documents, frozen)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        return result, code, reads

    def test_complete_private1500_pair_combines_unchanged_original500_prefix_and_nine_edges(self):
        result, code, reads = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(reads, [pool.IMPLEMENTATION, pool.BASE_REPORT, pool.FAILED_REPORT, pool.REPAIR_REPORT])
        self.assertEqual((result["status"], result["menu_complete"]), ("complete_menu", True))
        self.assertEqual((result["original_maximum_sweeps"], result["MVC_runtime_maximum_sweeps"]), (500, 1500))
        expected = deepcopy(self.failed["participant_scores"])
        for person in expected:
            expected[person]["MVC"] = self.complete["participant_scores"][person]["MVC"]
        self.assertEqual(result["participant_scores"], expected)
        diagnostics = self.helpers.repair.pilot.menu_diagnostics(expected, SimpleNamespace(menu=self.helpers.fixed))
        for key, value in diagnostics.items():
            self.assertEqual(result[key], value)
        self.assertEqual(len(result["nested_comparisons"]), 9)
        self.assertEqual([row["sweeps"] for row in result["repair_evidence"]["returned_fits"]], [501, 501])
        self.assertEqual(result["cost_references"]["original_pilot_process_seconds"], self.failed["elapsed_seconds"])
        self.assertEqual(result["cost_references"]["MVC_repair_process_seconds"], self.complete["elapsed_seconds"])
        self.assert_scope(result)

    def test_exact_job_source_failure_lineage_and_only_ceiling_change_are_required(self):
        mutations = (lambda r: r.update(job_id="46642547"), lambda r: r.update(phase="MZ_twice_SESOI_index0_fixed_menu_pilot"),
            lambda r: r.update(seed=1900058), lambda r: r.update(source_finished_utc="2026-10-05T10:18:00+00:00"),
            lambda r: r.update(source_report="different.json"), lambda r: r.update(failed_report="different.json"),
            lambda r: r["numerical_repair"].update(previous_process_elapsed_seconds=0.),
            lambda r: r["runtime_implementation"]["optimizer"].update(maximum_sweeps=2000),
            lambda r: r["runtime_implementation"]["optimizer"].update(joint_relative_tolerance=1e-6),
            lambda r: r.update(workers=1), lambda r: r.update(parallel_workers_cleaned_up=False),
            lambda r: r.update(full_menu_wave_selected=True), lambda r: r.update(family_selection="MVC"))
        for mutation in mutations:
            report = deepcopy(self.complete)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.build(report)
        for change in (lambda r: r.update(job_id="1"), lambda r: r.update(error="SIGTERM"),
                       lambda r: r["fits"][-1].update(subject_optimizer_failures=["p000"])):
            failed = deepcopy(self.failed)
            change(failed)
            with self.assertRaises(ValueError):
                self.build(failed=failed)

    def test_both_converged_fits_replay_baseline_last_bits_and128_over256_contributions_required(self):
        report = deepcopy(self.complete)
        report["fits"][1]["converged"] = False
        with self.assertRaisesRegex(RuntimeError, "^Nonconverged fit is technical; it is not a recovery failure$"):
            self.build(report)
        mutations = (lambda r: r["fits"][0].update(objective_history=[1.] * 1501),
            lambda r: r["scored_components"][0].update(per_participant_denominator=128),
            lambda r: r["scored_components"][1].update(heldout_trials_per_participant=64),
            lambda r: r["scored_components"][0]["participant_contributions"].pop("p000"),
            lambda r: r["scored_components"][0]["participant_contributions"].update(p000=-1.),
            lambda r: r["participant_scores"]["p000"].update(M0=-2.),
            lambda r: r["participant_scores"]["p000"].update(version="ER"),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=1.1e-12),
            lambda r: r.update(scored_primary_directions=[]))
        for mutation in mutations:
            report = deepcopy(self.complete)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.build(report)

    def test_actual_technical_prefixes_never_admit_a_pair_even_after_finalization(self):
        cases = ({"source_change": lambda r: r.update(seed=1)}, {"replay_error": True},
            {"failed_change": lambda r: r.update(job_id="1")}, {"nonconverged_at": 1},
            {"nonconverged_at": 2}, {"worker_error_at": 2}, {"expiry": "cleanup"}, {"expiry": "publication"})
        for options in cases:
            with self.subTest(options=options):
                report = self.report(**options)
                result = self.build(report)
                self.assertEqual((result["status"], result["menu_complete"]), ("unknown", False))
                self.assertEqual(result["reason"], "technical_MVC_repair_unknown")
                self.assertNotIn("participant_scores", result)
                self.assertTrue(all(field not in result for field in pool.DIAGNOSTICS))
                self.assertEqual(result["repair_evidence"]["fresh_fit_calls"], report["fresh_fit_calls"])
                if options.get("expiry") in ("cleanup", "publication"):
                    self.assertEqual(result["repair_evidence"]["completed_rivals"], ["MVC"])
                    self.assertEqual(len(result["repair_evidence"]["scored_components"]), 2)
                self.assert_scope(result)

    def test_original_completed_prefix_cannot_be_changed_or_incompletely_scored(self):
        failed = deepcopy(self.failed)
        failed["fits"][0]["converged"] = False
        with self.assertRaisesRegex(RuntimeError, "^Nonconverged fit is technical; it is not a recovery failure$"):
            self.build(failed=failed)
        for change in (lambda r: r["participant_scores"]["p000"].update(MC=-.1),
                lambda r: r["scored_components"].pop(), lambda r: r.update(completed_rivals=["MC", "MV"]),
                lambda r: r["fits"][0].update(objective_history=[1.] * 501)):
            failed = deepcopy(self.failed)
            change(failed)
            with self.assertRaises(ValueError):
                self.build(failed=failed)

    def test_missing_evidence_and_read_publication_cleanup_deadlines_remain_unknown(self):
        result, code, reads = self.invoke(missing=pool.REPAIR_REPORT)
        self.assertEqual((code, result["status"], result["menu_complete"]), (0, "unknown", False))
        self.assertEqual(result["reason"], "selected_terminal_report_missing")
        for expiry in ("repair_read", "publication", "cleanup"):
            with self.subTest(expiry=expiry):
                result, code, _ = self.invoke(expiry=expiry)
                self.assertEqual((code, result["status"], result["menu_complete"]), (1, "unknown", False))
                self.assertEqual(result["reason"], "technical_summary_failure")
                self.assertNotIn("participant_scores", result)
                self.assertTrue(all(field not in result for field in pool.DIAGNOSTICS))
                self.assert_scope(result)

    def test_path_thread_cap_and_fresh_output_guards_precede_numerical_imports(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pool, "require_compute"), \
                patch.dict(os.environ, {name: "1" for name in pool.THREAD_ENV}), \
                patch.object(pool, "load_helpers", side_effect=AssertionError("No numerical import")) as loader:
            output = Path(directory) / "new.json"
            args = ["--repair-report", str(pool.REPAIR_REPORT), "--implementation", str(pool.IMPLEMENTATION),
                    "--output", str(output), "--max-seconds", "240"]
            for index, value in ((1, "other.json"), (3, "other.json"), (7, "241"), (7, "0")):
                changed = list(args)
                changed[index] = value
                with self.assertRaises(ValueError):
                    pool.main(changed)
            with patch.dict(os.environ, {"BLIS_NUM_THREADS": "2"}), self.assertRaises(ValueError):
                pool.main(args)
            output.write_text("preserve")
            with self.assertRaises(FileExistsError):
                pool.main(args)
            self.assertEqual(output.read_text(), "preserve")
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
