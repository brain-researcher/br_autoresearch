"""Focused explicit-source menu collection tests; coordinated compute only."""
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

import summarize_twice_mz_menu_wave as wave
import test_twice_mz_menu_pilot as pilot_tests
import test_twice_mz_rival_component as component_tests


class TwiceMenuWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pilot_tests.TwiceMenuPilotTests.setUpClass()
        cls.pilot_fixture = pilot_tests.TwiceMenuPilotTests(
            "test_eight_fresh_fits_preserve_source_baseline_and_all_nine_nested_edges")
        component_tests.TwiceRivalComponentTests.setUpClass()
        cls.component_fixture = component_tests.TwiceRivalComponentTests(
            "test_all_four_models_get_two_fresh_halves_original500_and_exact_saved_scores")
        cls.spec = deepcopy(cls.component_fixture.spec)
        cls.pilot_complete = cls.pilot_fixture.exercise()[0]
        cls.components = {model: cls.component_fixture.exercise(model=model)[0] for model in wave.pilot.RIVALS}

    def baseline(self, index):
        return deepcopy(self.pilot_fixture.source if index == 0 else self.component_fixture.template(index)["source"])

    def pilot(self, report=None):
        report = deepcopy(self.pilot_complete if report is None else report)
        report.update(job_id=str(wave.PILOT_JOB), source_report=str(wave.baseline_path(0)),
                      source_dir=str(wave.baseline_path(0).parent))
        return report

    def component(self, model="MC", index=1, report=None):
        if report is None:
            report = self.components[model] if index == 1 else self.component_fixture.exercise(model=model, index=index)[0]
        report = deepcopy(report)
        report.update(job_id="987654", source_report=str(wave.baseline_path(index)),
                      source_dir=str(wave.baseline_path(index).parent))
        return report

    def inputs(self):
        declarations = [(model, 900001 + number, [1]) for number, model in enumerate(wave.pilot.RIVALS)]
        reports = {wave.baseline_path(0): self.baseline(0), wave.PILOT_REPORT: self.pilot(),
                   wave.baseline_path(1): self.baseline(1)}
        reports.update({wave.component_path(1, model, array): self.component(model) for model, array, _ in declarations})
        return reports, declarations

    def invoke(self, reports, indices, declarations, *, output=None, expiry=None):
        frozen, reads, now = deepcopy(reports), [], [100.]
        real_reader, real_writer = Path.read_text, wave.write_report
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        allowed = {wave.baseline_path(index) for index in indices}
        if 0 in indices:
            allowed.add(wave.PILOT_REPORT)
        allowed.update(wave.component_path(index, model, array) for model, array, selected in declarations for index in selected)

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path == wave.IMPLEMENTATION:
                return json.dumps(self.spec)
            if path not in allowed:
                raise AssertionError(f"No events/truth or unselected report may be read: {path}")
            if path not in reports:
                raise FileNotFoundError(str(path))
            return json.dumps(reports[path], allow_nan=False)

        def writer(path, result):
            real_writer(path, result)
            if expiry == "publication" and result.get("status") == "terminal_selection_collected":
                now[0] = 341.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 341.

        with ExitStack() as stack:
            for context in (patch.object(Path, "read_text", reader), patch.object(wave, "require_compute"),
                    patch.dict(os.environ, {name: "1" for name in wave.THREAD_ENV}),
                    patch.object(wave.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(wave.signal, "alarm", side_effect=alarm),
                    patch.object(wave, "write_report", side_effect=writer), redirect_stdout(io.StringIO())):
                stack.enter_context(context)
            if output is None:
                result, code = wave.pool(indices, declarations, self.spec), None
            else:
                argv = ["--terminal-indices", ",".join(map(str, indices)), "--implementation", str(wave.IMPLEMENTATION),
                        "--output", str(output), "--max-seconds", "240"]
                for model, array, selected in declarations:
                    argv += ["--component-array", f"{model}:{array}:" + ",".join(map(str, selected))]
                code = wave.main(argv)
                result = json.loads(real_reader(output))
        self.assertEqual(reports, frozen)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        return result, reads, code

    def assert_scope(self, result):
        self.assertEqual((result["effect_level"], result["SESOI_level"], result["generating_z"]), ("twice", 2., .40))
        self.assertEqual(result["complete_menus"] + result["unknown_menus"], 200)
        self.assertEqual(len(result["cohorts"]), 200)
        self.assertEqual([row["replicate_index"] for row in result["cohorts"]], list(range(200)))
        for field in wave.CLAIMS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        for field in ("new_simulation", "new_model_fits", "source_events_read", "saved_scores_rescored", "automatic_retry", "wave_selected"):
            self.assertFalse(result[field])

    def test_complete_original_pilot_and_indexed_menu_reuse_exact_nine_edges_without_arbitration(self):
        reports, declarations = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            result, reads, code = self.invoke(reports, [0, 1], declarations, output=Path(directory) / "summary.json")
        self.assertEqual(code, 0)
        self.assertEqual((result["complete_menus"], result["unknown_menus"]), (2, 198))
        self.assertEqual(len(reads), 8)  # implementation +2 baselines +pilot +4 components
        self.assertEqual(len(result["cohorts"][0]["attempts"]), 1)
        self.assertEqual(len(result["cohorts"][1]["attempts"]), 4)
        for row in result["cohorts"][:2]:
            expected = wave.pilot.menu_diagnostics(row["participant_scores"], self.pilot_fixture.helpers)
            for field, value in expected.items():
                self.assertEqual(row[field], value)
            self.assertEqual(len(row["nested_comparisons"]), 9)
            self.assertEqual(row["maximal_eligible_models_diagnostic"], ["MZ", "MC"])
        self.assertEqual(result["incomparable_maximal_count"], 2)
        self.assert_scope(result)

    def test_explicit_model_array_routes_fixed_sources_and_no_unselected_reads(self):
        self.assertIn("46603942/twice/replicate0", str(wave.baseline_path(0)))
        self.assertIn("wave-46606307/replicate1", str(wave.baseline_path(1)))
        self.assertIn("wave-46606413/replicate199", str(wave.baseline_path(199)))
        declarations = [("MC", 987001, [2]), ("MC", 987002, [199]), ("MVZ", 987003, [199])]
        reports = {wave.baseline_path(index): self.baseline(index) for index in (1, 2, 199)}
        for model, array, selected in declarations:
            index = selected[0]
            reports[wave.component_path(index, model, array)] = self.component(model, index)
        result, reads, _ = self.invoke(reports, [1, 2, 199], declarations)
        self.assertEqual(len(reads), 6)
        self.assertNotIn(wave.PILOT_REPORT, reads)
        self.assertEqual(result["complete_menus"], 0)
        self.assertEqual(result["cohorts"][1]["components"]["MC"]["reason"], "not_selected_as_terminal_by_coordinator")
        self.assertEqual(result["cohorts"][199]["components"]["MVZ"]["status"], "completed_pair")
        self.assert_scope(result)

    def test_later_pilot_failure_preserves_only_explicitly_completed_validated_model_prefix(self):
        failed = self.pilot(self.pilot_fixture.exercise(nonconverged_at=3)[0])
        reports = {wave.baseline_path(0): self.baseline(0), wave.PILOT_REPORT: failed}
        result, _, _ = self.invoke(reports, [0], [])
        row = result["cohorts"][0]
        self.assertEqual(row["components"]["MC"]["status"], "completed_pair")
        self.assertTrue(row["components"]["MC"]["from_technical_attempt"])
        self.assertEqual(row["components"]["MV"]["status"], "unknown")
        self.assertEqual(len(row["attempts"][0]["returned_fits"]), 3)
        self.assertEqual([fit["sweeps"] for fit in row["attempts"][0]["returned_fits"]],
                         [len(fit["objective_history"]) for fit in failed["fits"]])
        self.assertEqual(len(row["attempts"][0]["scored_components"]), 2)
        self.assertIn("error", row["attempts"][0])
        self.assertNotIn("nested_comparisons", row)
        for mutation in (lambda r: r.update(completed_rivals=["MC", "MV"]),
                         lambda r: r["scored_components"].pop(),
                         lambda r: r["fits"][0].update(converged=False)):
            broken = deepcopy(reports)
            mutation(broken[wave.PILOT_REPORT])
            with self.assertRaises((ValueError, RuntimeError)):
                self.invoke(broken, [0], [])
        self.assert_scope(result)

    def test_indexed_technical_stages_are_unknown_even_when_finalization_retains_both_scores(self):
        options = ({"source_change": lambda r: r.update(seed=1)}, {"replay_error": True},
            {"nonconverged_second": True}, {"worker_failure_second": True},
            {"expiry": "staged"}, {"expiry": "cleanup"}, {"expiry": "publication"})
        path = wave.component_path(1, "MC", 987001)
        for case in options:
            with self.subTest(case=case):
                failed = self.component(report=self.component_fixture.exercise(**case)[0])
                result, _, _ = self.invoke({wave.baseline_path(1): self.baseline(1), path: failed}, [1], [("MC", 987001, [1])])
                row = result["cohorts"][1]
                self.assertEqual(row["components"]["MC"]["status"], "unknown")
                self.assertEqual(row["attempts"][0]["elapsed_seconds"], failed["elapsed_seconds"])
                self.assertEqual(len(row["attempts"][0]["returned_fits"]), len(failed["fits"]))
                self.assertNotIn("participant_scores", row)
                self.assert_scope(result)

    def test_malformed_success_source_runtime_scores_rosters_and_normalization_abort(self):
        path = wave.component_path(1, "MC", 987001)
        changes = (lambda r: r.update(phase=wave.pilot.PHASE), lambda r: r.update(seed=1900057),
            lambda r: r.update(generating_z=.45), lambda r: r.update(source_job_id="1"),
            lambda r: r.update(source_finished_utc="2000-01-01T00:00:00+00:00"),
            lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r["participant_scores"]["p000"].update(M0=-.999999999999),
            lambda r: r["participant_scores"]["p000"].update(version="ER"),
            lambda r: r["scored_components"][0].update(per_participant_denominator=128),
            lambda r: r["scored_components"][0].update(heldout_trials_per_participant=64),
            lambda r: r["scored_components"][1]["participant_contributions"].update(p000=-.1),
            lambda r: r["fits"][0].update(converged=False),
            lambda r: r.update(saved_score_replay_max_absolute_discrepancy=1e-4),
            lambda r: r.update(family_selection="MZ"))
        for mutation in changes:
            report = self.component()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                self.invoke({wave.baseline_path(1): self.baseline(1), path: report}, [1], [("MC", 987001, [1])])

    def test_missing_inputs_and_selection_refusals_keep_fixed200_without_substitutions(self):
        result, reads, _ = self.invoke({wave.baseline_path(1): self.baseline(1)}, [0, 1], [("MC", 987001, [1])])
        self.assertEqual(result["unknown_menus"], 200)
        self.assertEqual(result["cohorts"][0]["reason"], "selected_terminal_baseline_missing")
        self.assertEqual(result["cohorts"][1]["components"]["MC"]["reason"], "selected_terminal_report_missing")
        self.assertNotIn(wave.PILOT_REPORT, reads)
        with patch.object(Path, "read_text", side_effect=AssertionError("Reject selection before any report read")):
            for indices, declarations in (([1, 1], []), ([1], [("MC", 1, [2])]),
                    ([0], [("MC", 1, [0])]), ([1], [("MC", 1, [1]), ("MC", 2, [1])])):
                with self.assertRaises(ValueError):
                    wave.pool(indices, declarations, self.spec)
        self.assertEqual(wave.component_array("MVC:123:1,199"), ("MVC", 123, [1, 199]))
        for value in ("MZ:123:1", "MC:123:0", "MC:0:1", "MC:123:1,1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                wave.component_array(value)
        self.assert_scope(result)

    def test_final_publication_or_cleanup_expiry_retains_evidence_without_claiming_completed_collection(self):
        reports, declarations = self.inputs()
        for stage in ("publication", "cleanup"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                result, _, code = self.invoke(reports, [0, 1], declarations,
                    output=Path(directory) / "summary.json", expiry=stage)
            self.assertEqual(code, 1)
            self.assertEqual(result["status"], "technical_menu_collection_failure")
            self.assertIn("deadline", result["error"])
            self.assertEqual(result["complete_menus"], 2)
            self.assert_scope(result)


if __name__ == "__main__":
    unittest.main()
