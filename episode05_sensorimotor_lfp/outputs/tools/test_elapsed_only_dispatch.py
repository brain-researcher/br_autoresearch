"""Mocked elapsed-only wrapper checks; unchanged fixed-wave tests are not rerun."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_elapsed_only_panel as panel
import run_elastic_net_panel as qualified


class ElapsedOnlyDispatchTest(unittest.TestCase):
    def run_wrapper(self, root, *, cpus=4, numerical_threads="1", fail_child=False,
                    fire_deadline=False, invalid_null_count=False):
        paths = {name: root/name for name in (
            "source", "trial-table", "scalar-bank-root", "hw-bank-root", "hw-summary",
            "bank-root", "summary-out", "out")}
        argv = ["run_elapsed_only_panel.py", "--workers", "4", "--max-seconds", "720"]
        for name, path in paths.items():
            argv.extend(["--"+name, str(path)])
        calls, dispatch, handlers, alarms = [], [], {}, []
        packet = {"elapsed_seconds": 1., "source_read_seconds": .2,
                  "single_recipe_fit_and_score_seconds": .7, "maximum_RSS_KiB": 1024}
        summary = {"status": "complete_elapsed_only_HW_panel", "scientific_trial_count_increment": 1,
                   "alternative_explanation_stress_count_increment": 1,
                   "null_control_trial_count_increment": int(invalid_null_count),
                   "updates_primary_incumbent": False, "sensitivity_only": True,
                   "promotion_authorized": False}

        def read(path, *args, **kwargs):
            if path.name == "score_banks.json":
                self.assertIn(path.parent.name, panel.SESSIONS)
                return json.dumps(packet)
            self.assertEqual(path, paths["summary-out"]/"primary_elapsed_only.json")
            return json.dumps(summary)

        def child(command, **kwargs):
            calls.append((command, kwargs))
            if command[2] == "session" and fail_child:
                raise subprocess.CalledProcessError(1, command)

        def fake_dispatch(sessions, workers, run_one, **kwargs):
            dispatch.append((list(sessions), workers, kwargs))
            self.assertEqual(kwargs["max_seconds"], 720)
            self.assertFalse(kwargs["stop_event"].is_set())
            if fire_deadline:
                # The signal callback must only request stop. The qualified
                # helper owns settling its active wave and raising afterward.
                self.assertIsNone(handlers[panel.signal.SIGALRM](panel.signal.SIGALRM, None))
                self.assertTrue(kwargs["stop_event"].is_set())
                raise TimeoutError("mock settled deadline")
            completed = []
            for session in reversed(sessions):
                item = run_one(session)
                kwargs["on_complete"](session, item)
                completed.append(item)
            return sorted(completed, key=lambda row: sessions.index(row["session_id"]))

        env = {"SLURM_CPUS_PER_TASK": str(cpus), "SLURM_JOB_ID": "fixture",
               **{key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        env["NUMEXPR_NUM_THREADS"] = numerical_threads
        with patch.dict(os.environ, env), patch.object(sys, "argv", argv), \
                patch.object(panel, "compute_only"), patch.object(panel, "validate_session"), \
                patch.object(Path, "read_text", read), patch.object(panel.subprocess, "run", side_effect=child), \
                patch.object(panel, "run_sessions", side_effect=fake_dispatch), \
                patch.object(panel.signal, "signal", side_effect=lambda signum, handler: handlers.update({signum: handler})), \
                patch.object(panel.signal, "alarm", side_effect=alarms.append), patch("builtins.print"):
            expected = (RuntimeError if numerical_threads != "1" else ValueError if cpus < 4
                        else subprocess.CalledProcessError if fail_child else TimeoutError if fire_deadline
                        else ValueError if invalid_null_count else None)
            if expected:
                with self.assertRaises(expected):
                    panel.main()
            else:
                panel.main()
        report_path = paths["out"]/"remaining_panel.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        return paths, calls, dispatch, alarms, report

    def test_unchanged_qualified_helper_and_actual_CPU_thread_guards(self):
        self.assertIs(panel.run_sessions, qualified.run_sessions)
        self.assertIs(panel.validate_workers, qualified.validate_workers)
        self.assertEqual(panel.CHILD_EXTERNAL_SECONDS, qualified.CHILD_EXTERNAL_SECONDS)
        for kwargs in ({"cpus": 1}, {"numerical_threads": "2"}):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as directory:
                paths, calls, dispatch, alarms, report = self.run_wrapper(Path(directory), **kwargs)
                self.assertEqual((calls, dispatch, alarms), ([], [], []))
                self.assertIsNone(report)
                self.assertFalse(paths["out"].exists())

    def test_fifteen_children_exact_caps_HW_routing_and_complete_stress_only_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, calls, dispatch, alarms, report = self.run_wrapper(Path(directory))
        sessions = [session for session in panel.SESSIONS if session != panel.PILOT]
        self.assertEqual(len(dispatch), 1)
        self.assertEqual(dispatch[0][:2], (sessions, 4))
        self.assertEqual(len(calls), 16)
        for command, kwargs in calls[:-1]:
            self.assertEqual(command[2], "session")
            self.assertEqual(kwargs, {"check": True, "timeout": 300})
            self.assertEqual(command[command.index("--max-seconds")+1], "270")
            self.assertNotEqual(command[command.index("--session")+1], panel.PILOT)
            for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root"):
                self.assertEqual(command[command.index("--"+name)+1], str(paths[name]))
        command, kwargs = calls[-1]
        self.assertEqual(command[2], "summarize")
        self.assertTrue(kwargs["check"])
        self.assertTrue(0 < kwargs["timeout"] <= 720)
        for name in ("bank-root", "hw-bank-root", "hw-summary"):
            self.assertEqual(command[command.index("--"+name)+1], str(paths[name]))
        self.assertEqual(report["status"], "complete_elapsed_only_HW_remaining_panel")
        self.assertEqual(report["selected_workers"], 4)
        self.assertEqual(report["allocated_CPUs"], 4)
        self.assertEqual([row["session_id"] for row in report["completed_remaining_sessions"]], sessions)
        self.assertEqual(report["scientific_trial_count_increment"], 1)
        self.assertEqual(report["alternative_explanation_stress_count_increment"], 1)
        self.assertEqual(report["null_control_trial_count_increment"], 0)
        self.assertEqual(alarms, [720, 0])

    def test_failed_child_cannot_publish_or_charge_partial_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls, _, alarms, report = self.run_wrapper(Path(directory), fail_child=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][2], "session")
        self.assertEqual(report["status"], "technical_failure")
        self.assertEqual(report["scientific_trial_count_increment"], 0)
        self.assertEqual(report["alternative_explanation_stress_count_increment"], 0)
        self.assertEqual(report["completed_remaining_sessions"], [])
        self.assertEqual(alarms, [720, 0])

    def test_SIGALRM_requests_stop_and_no_summary_runs_after_settled_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls, dispatch, alarms, report = self.run_wrapper(Path(directory), fire_deadline=True)
        self.assertEqual(calls, [])
        self.assertTrue(dispatch[0][2]["stop_event"].is_set())
        self.assertEqual(report["status"], "technical_failure")
        self.assertIn("mock settled deadline", report["error"])
        self.assertEqual(report["scientific_trial_count_increment"], 0)
        self.assertEqual(alarms, [720, 0])

    def test_summary_cannot_turn_elapsed_stress_into_new_null_control(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls, _, _, report = self.run_wrapper(Path(directory), invalid_null_count=True)
        self.assertEqual(len(calls), 16)
        self.assertEqual(report["status"], "technical_failure")
        self.assertEqual(report["scientific_trial_count_increment"], 0)
        self.assertEqual(report["alternative_explanation_stress_count_increment"], 0)
        self.assertEqual(report["null_control_trial_count_increment"], 0)


if __name__ == "__main__":
    unittest.main()
