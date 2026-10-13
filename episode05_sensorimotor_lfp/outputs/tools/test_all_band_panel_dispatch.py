"""Mocked all-band orchestration plus the unchanged fixed-wave helper fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_all_band_panel as panel
import run_elastic_net_panel as previous
# These six tests replace the shared helper's executor/wait with deterministic
# fakes: no real workers, processes or source reads are launched by the suite.
from test_elastic_net_panel_dispatch import DispatchFixtures


class AllBandWiringFixtures(unittest.TestCase):
    def run_wrapper(self, root, *, cpus=4, fail_child=False):
        paths = {name: root/name for name in (
            "source", "trial-table", "scalar-bank-root", "reference-bank-root", "reference-summary",
            "joint-bank-root", "joint-summary", "bank-root", "summary-out", "out")}
        argv = ["run_all_band_panel.py", "--workers", "4", "--max-seconds", "1260"]
        for name, path in paths.items():
            argv.extend(["--"+name, str(path)])
        calls, dispatch = [], []
        packet = {"elapsed_seconds": 1., "source_read_seconds": .2,
                  "two_recipe_fit_and_score_seconds": .7, "maximum_RSS_KiB": 1024}
        summary = {"status": "complete_all_band_control_panel", "scientific_trial_count_increment": 2,
                   "total_valid_primary_trials_including_prior": 27}

        def read(path, *args, **kwargs):
            if path.name == "score_banks.json":
                self.assertIn(path.parent.name, panel.SESSIONS)
                return json.dumps(packet)
            self.assertEqual(path, paths["summary-out"]/"primary_all_band.json")
            return json.dumps(summary)

        def child(command, **kwargs):
            calls.append((command, kwargs))
            if command[2] == "session" and fail_child:
                raise subprocess.CalledProcessError(1, command)

        def fake_dispatch(sessions, workers, run_one, **kwargs):
            dispatch.append((list(sessions), workers, kwargs))
            self.assertFalse(kwargs["stop_event"].is_set())
            self.assertEqual(kwargs["max_seconds"], 1260)
            completed = []
            # Reverse completion order exposes accidental completion-order
            # reporting. The real shared helper is tested separately above.
            for session in reversed(sessions):
                item = run_one(session)
                kwargs["on_complete"](session, item)
                completed.append(item)
            return sorted(completed, key=lambda row: sessions.index(row["session_id"]))

        env = {"SLURM_CPUS_PER_TASK": str(cpus), "SLURM_JOB_ID": "fixture",
               **{key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
        with patch.dict(os.environ, env), patch.object(sys, "argv", argv), \
                patch.object(panel, "compute_only"), patch.object(panel, "validate_session"), \
                patch.object(Path, "read_text", read), patch.object(panel.subprocess, "run", side_effect=child), \
                patch.object(panel, "run_sessions", side_effect=fake_dispatch), \
                patch.object(panel.signal, "signal"), patch.object(panel.signal, "alarm"), patch("builtins.print"):
            expected = ValueError if cpus < 4 else subprocess.CalledProcessError if fail_child else None
            if expected:
                with self.assertRaises(expected):
                    panel.main()
            else:
                panel.main()
        report_path = paths["out"]/"remaining_panel.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        return paths, calls, dispatch, report

    def test_exact_reviewed_helper_and_cpu_guard_before_dispatch(self):
        self.assertIs(panel.run_sessions, previous.run_sessions)
        self.assertIs(panel.validate_workers, previous.validate_workers)
        self.assertEqual(panel.CHILD_EXTERNAL_SECONDS, previous.CHILD_EXTERNAL_SECONDS)
        with tempfile.TemporaryDirectory() as directory:
            paths, calls, dispatch, report = self.run_wrapper(Path(directory), cpus=1)
            self.assertEqual(calls, [])
            self.assertEqual(dispatch, [])
            self.assertIsNone(report)
            self.assertFalse(paths["out"].exists())

    def test_fifteen_fresh_children_exact_bounds_anchor_routing_and_complete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, calls, dispatch, report = self.run_wrapper(Path(directory))
        sessions = [session for session in panel.SESSIONS if session != panel.PILOT]
        self.assertEqual(len(dispatch), 1)
        self.assertEqual(dispatch[0][:2], (sessions, 4))
        self.assertEqual(len(calls), 16)
        for command, kwargs in calls[:-1]:
            self.assertEqual(command[2], "session")
            self.assertEqual(kwargs, {"check": True, "timeout": 300})
            self.assertEqual(command[command.index("--max-seconds")+1], "270")
            self.assertNotEqual(command[command.index("--session")+1], panel.PILOT)
            for name in ("source", "trial-table", "scalar-bank-root", "reference-bank-root", "joint-bank-root"):
                self.assertEqual(command[command.index("--"+name)+1], str(paths[name]))
        command, kwargs = calls[-1]
        self.assertEqual(command[2], "summarize")
        self.assertTrue(kwargs["check"])
        self.assertTrue(0 < kwargs["timeout"] <= 1260)
        for name in ("bank-root", "reference-bank-root", "reference-summary", "joint-bank-root", "joint-summary"):
            self.assertEqual(command[command.index("--"+name)+1], str(paths[name]))
        self.assertEqual(report["status"], "complete_all_band_remaining_panel")
        self.assertEqual(report["selected_workers"], 4)
        self.assertEqual(report["allocated_CPUs"], 4)
        self.assertEqual([row["session_id"] for row in report["completed_remaining_sessions"]], sessions)
        self.assertEqual(report["scientific_trial_count_increment"], 2)

    def test_failed_child_does_not_launch_summary_or_charge_partial_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls, _, report = self.run_wrapper(Path(directory), fail_child=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][2], "session")
        self.assertEqual(report["status"], "technical_failure")
        self.assertEqual(report["scientific_trial_count_increment"], 0)
        self.assertEqual(report["completed_remaining_sessions"], [])


if __name__ == "__main__":
    unittest.main()
