"""Two standard-library admission/composition checks; coordinator compute only."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import run_reconstruction_qualification as entry


def profile():
    return {**entry.FIXED, "resources": dict(entry.RESOURCES), "fixture": dict(entry.FIXTURE),
            "input_scope": dict(entry.INPUT_SCOPE), "authorized": True, "approval_source": "fixture"}


class ReconstructionAdmissionTests(unittest.TestCase):
    def test_exact_CPU_resources_absolute_deadline_and_scope_refusal(self):
        epoch = 1780000000
        stamp = lambda seconds: datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        record = ("JobId=123 JobState=RUNNING NodeList=sh03-06n05 NumCPUs=1 NumTasks=1 NumNodes=1 "
                  "Partition=russpold TimeLimit=02:00:00 MinMemoryNode=7G AllocTRES=cpu=1,mem=7G,node=1 "
                  f"StartTime={stamp(epoch)} EndTime={stamp(epoch + 7200)}")
        env = {**{key: "1" for key in entry.THREADS}, "SLURM_JOB_ID": "123",
               "SLURM_CPUS_PER_TASK": "1", "SLURM_MEM_PER_NODE": "7168", "CUDA_VISIBLE_DEVICES": ""}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(entry.socket, "gethostname", return_value="sh03-06n05.stanford.edu"), \
                mock.patch.object(entry.time, "time", return_value=epoch + 10), \
                mock.patch.object(entry, "_read_text", return_value=record) as read:
            allocation = entry._allocation("reconstruction")
            self.assertEqual(allocation["deadline_epoch"], epoch + 6900)
            with mock.patch.object(entry.time, "time", return_value=epoch + 6450):
                self.assertEqual(entry._allocation("brain_prefix")["deadline_epoch"], epoch + 6900)
                with self.assertRaises(ValueError):
                    entry._allocation("reconstruction")
            for wrong in (record.replace("NumCPUs=1", "NumCPUs=2"),
                          record.replace("mem=7G", "mem=8G"),
                          record.replace("node=1 ", "node=1,gres/gpu=1 "),
                          record.replace("TimeLimit=02:00:00", "TimeLimit=01:00:00"),
                          record.replace("NodeList=sh03-06n05", "NodeList=sh04-ln04")):
                with self.subTest(record=wrong), mock.patch.object(entry, "_read_text", return_value=wrong):
                    with self.assertRaises(ValueError):
                        entry._allocation("reconstruction")
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "456"}):
                with self.assertRaises(ValueError):
                    entry._allocation("brain_prefix")
        for changed in ({**profile(), "authorized": False},
                        {**profile(), "fixture": {**entry.FIXTURE, "updates": 199}},
                        {**profile(), "input_scope": {**entry.INPUT_SCOPE, "tau": True}},
                        {**profile(), "resources": {**entry.RESOURCES, "cpus": True}}):
            with self.subTest(profile=changed), \
                    mock.patch.object(entry, "_read_text", return_value=json.dumps(changed)), \
                    mock.patch.object(entry, "_allocation") as allocate:
                with self.assertRaises(ValueError):
                    entry.admit()
                allocate.assert_not_called()

    def test_exclusive_claim_and_categorical_failure_publication(self):
        sentinel = "UNPUBLISHED_FIXTURE_VALUES"
        for error in (None, RuntimeError(sentinel)):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as temporary, ExitStack() as stack:
                run = Path(temporary)
                grant = profile()
                context = SimpleNamespace(profile=grant, scratch=run / "scratch")
                budget = SimpleNamespace(context=context,
                    allocation={"job_id": "123", "hostname": "sh03-06n05"},
                    check=mock.Mock(), check_storage=mock.Mock())
                calls = []
                def qualify(**kwargs):
                    self.assertTrue((run / "claim.json").exists())
                    self.assertIs(kwargs["budget"], budget)
                    self.assertEqual(kwargs["retained_archive_dir"], run / "assets")
                    calls.append(True)
                    if error is not None:
                        raise error
                    return {"status": "completed_public_runtime_qualification", "fixture_count": 1,
                            "reconstruction": {"completed_updates": 200}}
                stack.enter_context(mock.patch.object(entry, "RUN_DIR", run))
                stack.enter_context(mock.patch.object(entry, "_configure_runtime"))
                stack.enter_context(mock.patch.object(entry, "admit", return_value=(context, budget)))
                stack.enter_context(mock.patch.object(entry, "_profile", return_value=grant))
                stack.enter_context(mock.patch.dict("sys.modules", {
                    "qualify_reconstruction": SimpleNamespace(qualify_reconstruction=qualify)}))
                self.assertEqual(entry.run_once(), 0 if error is None else 1)
                saved = (run / "report.json").read_text()
                report = json.loads(saved)
                self.assertNotIn(sentinel, saved)
                self.assertEqual(report["status"], "complete_public_runtime_qualification" if error is None else "incomplete")
                self.assertFalse(report["empirical_execution"])
                if error is not None:
                    self.assertEqual(report["error_type"], "RuntimeError")
                with self.assertRaises(ValueError):
                    entry.run_once()
                self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
