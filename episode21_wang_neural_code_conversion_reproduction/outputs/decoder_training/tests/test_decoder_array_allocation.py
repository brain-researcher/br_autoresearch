"""Mocked scheduler identity regression; no scheduler or payload access."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_execution_contract as contract


class ArrayAllocationTests(unittest.TestCase):
    def setUp(self):
        self.resources = {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
                          "durable_bytes": 8_000_000_000, "scratch_bytes": 2_000_000_000}
        self.environment = {"SLURM_JOB_ID": "46839670", "SLURM_ARRAY_JOB_ID": "46839670",
                            "SLURM_ARRAY_TASK_ID": "7", "SLURM_CPUS_PER_TASK": "1",
                            "SLURM_MEM_PER_NODE": "7168",
                            **{name: "1" for name in contract.prefix.THREADS}}

    def line(self, job="46839670", parent="46839670", task="7"):
        array = "" if parent is None else f" ArrayJobId={parent} ArrayTaskId={task}"
        end = datetime.fromtimestamp(38800).isoformat()
        return (f"JobId={job}{array} JobState=RUNNING NumCPUs=1 NumNodes=1 NodeList=compute "
                f"AllocTRES=cpu=1,mem=7G TimeLimit=08:00:00 EndTime={end}")

    def verify(self, text, environment=None):
        with mock.patch.dict(contract.os.environ, self.environment if environment is None else environment, clear=True), \
                mock.patch.object(contract.socket, "gethostname", return_value="compute.domain"), \
                mock.patch.object(contract.time, "time", return_value=10000), \
                mock.patch.object(contract.subprocess, "run", return_value=SimpleNamespace(stdout=text)) as command:
            result = contract.verify_allocation(self.resources)
            return result, command.call_args

    def test_array_master_numeric_job_uses_exact_task_selector(self):
        result, called = self.verify(self.line())
        self.assertEqual(called.args[0], ["scontrol", "show", "job", "46839670_7", "-o"])
        self.assertEqual(called.kwargs, dict(check=True, capture_output=True, text=True, timeout=30))
        self.assertEqual(result["job_id"], "46839670")
        self.assertEqual(result["time_limit_seconds"], 28800)
        self.assertEqual(result["job_end_epoch"], 38800)

    def test_other_array_task_preserves_its_distinct_numeric_job(self):
        environment = dict(self.environment, SLURM_JOB_ID="46839671", SLURM_ARRAY_TASK_ID="0")
        result, called = self.verify(self.line(job="46839671", task="0"), environment)
        self.assertEqual(called.args[0][3], "46839670_0")
        self.assertEqual(result["job_id"], "46839671")

    def test_nonarray_numeric_query_and_end_resource_checks_are_preserved(self):
        environment = {key: value for key, value in self.environment.items() if not key.startswith("SLURM_ARRAY_")}
        line = self.line(parent=None)
        result, called = self.verify(line, environment)
        self.assertEqual(called.args[0][3], "46839670")
        self.assertEqual(result["allocated_cpus"], 1)
        for bad, error in ((line.replace("08:00:00", "09:00:00"), "authorized_walltime"),
                           (line.replace("NumCPUs=1", "NumCPUs=2"), "allocation_mismatch"),
                           (line.replace("mem=7G", "mem=7G,gres/gpu=1"), "allocation_mismatch")):
            with self.subTest(error=error), self.assertRaisesRegex(contract.Stop, error):
                self.verify(bad, environment)

    def test_mismatched_numeric_parent_or_task_identity_refuses(self):
        for line in (self.line(job="46839671"), self.line(parent="46839671"), self.line(task="0")):
            with self.subTest(line=line), self.assertRaisesRegex(contract.Stop, "allocation_mismatch"):
                self.verify(line)

    def test_multiple_or_duplicated_scheduler_records_cannot_be_flattened(self):
        for text in (self.line()+"\n"+self.line(job="46839671", task="0"), ""):
            with self.subTest(text=text), self.assertRaisesRegex(contract.Stop, "single_job_record"):
                self.verify(text)
        for text in (self.line()+" "+self.line(job="46839671", task="0"),
                     self.line()+" ArrayTaskId=7", self.line(parent=None)):
            with self.subTest(text=text), self.assertRaisesRegex(contract.Stop, "unambiguous_job_identity"):
                self.verify(text)

    def test_partial_or_noncanonical_array_environment_refuses_before_query(self):
        environments = []
        for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
            missing = dict(self.environment)
            del missing[key]
            environments.append(missing)
            environments.append(dict(self.environment, **{key: ""}))
        environments += [dict(self.environment, SLURM_ARRAY_TASK_ID=value) for value in ("07", "7-8", "-1")]
        environments.append(dict(self.environment, SLURM_ARRAY_JOB_ID="046839670"))
        for environment in environments:
            with self.subTest(environment=environment), \
                    mock.patch.dict(contract.os.environ, environment, clear=True), \
                    mock.patch.object(contract.socket, "gethostname", return_value="compute"), \
                    mock.patch.object(contract.subprocess, "run") as command:
                with self.assertRaisesRegex(contract.Stop, "explicit_array_allocation_identity"):
                    contract.verify_allocation(self.resources)
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
