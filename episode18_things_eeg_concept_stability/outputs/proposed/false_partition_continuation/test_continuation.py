"""Focused lifecycle fixtures; no original producer import or scientific arrays.

Run this file in a coordinator-owned compute allocation. All episode metadata,
source hashes and producer functions below are temporary, explicitly fake
fixtures, never an authority for the actual episode.
"""
from contextlib import contextmanager, redirect_stderr
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import continue_search as adapter


class FixtureBudgetStop(RuntimeError):
    pass


class ContinuationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = int(time.time())
        self.deadline = self.now + 3600
        self.source = self.root / adapter.SOURCE
        self.source.parent.mkdir(parents=True)
        self.source.write_text("# TEST FIXTURE ONLY: no numerical implementation\n")
        self.config_path = self.root / adapter.CONFIG
        self.config_path.parent.mkdir(parents=True)
        self.original_config = {"not_after_unix": adapter.ORIGINAL_DEADLINE,
            "participant_id": "sub-07", "segment_byte_cap": 64*1024**2,
            "fixture_only": True}
        self.config_path.write_text(json.dumps(self.original_config))
        self.bound_config = dict(self.original_config,
            code_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.predecessor = self.root / adapter.PREDECESSOR
        self.predecessor.parent.mkdir(parents=True)
        self.predecessor.write_bytes(b"NOT AN NPZ: source-free fixture; never opened")
        self.prior = {"status": "resource_incomplete", "checkpoint_available": True,
            "EEG_read": False, "participant_id": "sub-07", "member": 0,
            "start": 11, "swaps": 64556, "total_legal_proposals": 1164556}
        (self.predecessor.parent / "report.json").write_text(json.dumps(self.prior))
        self.authority = {"status": "adopted", "episode": "EP18",
            "scope": "bounded_generator_continuation", "decision_record": "TEST FIXTURE ONLY",
            "original_source_sha256": self.bound_config["code_sha256"],
            "original_config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "original_config_id": hashlib.sha256(adapter.encoded(self.bound_config)).hexdigest(),
            "predecessor_job_id": "47679999", "predecessor_checkpoint": adapter.PREDECESSOR.as_posix(),
            "original_deadline_unix": adapter.ORIGINAL_DEADLINE,
            "new_absolute_deadline_unix": self.now+86400, "resources": dict(adapter.RESOURCES)}
        self.authority_path = self.root / "outputs/config/TEST_ONLY_authority.json"
        self.authority_path.write_text(json.dumps(self.authority))
        self.output = self.root / "outputs/runs/false_partition_preparation/fixture/search"
        self.output.parent.mkdir(parents=True)
        self.environment = dict.fromkeys(adapter.THREAD_ENV, "1")
        self.environment.update(SLURM_JOB_ID="TEST_ONLY", SLURM_JOB_NUM_NODES="1", SLURM_NTASKS="1",
            SLURM_JOB_CPUS_PER_NODE="13", SLURM_CPUS_PER_TASK="13", SLURM_MEM_PER_NODE="98304",
            SLURM_JOB_START_TIME=str(self.now-10), SLURM_JOB_END_TIME=str(self.now-10+36000),
            SLURM_RESTART_COUNT="0", SLURM_JOB_GPUS="", SLURM_ARRAY_TASK_ID="")
        self.producer = self.fake_producer()

    def fake_producer(self):
        def write(path, value, cap=8*1024**2):
            encoded = adapter.encoded(value)
            self.assertLessEqual(len(encoded), cap)
            path.write_bytes(encoded)

        @contextmanager
        def budget(deadline, rss):
            self.assertEqual((deadline, rss), (self.deadline, adapter.RESOURCES["rss_byte_cap"]))
            yield lambda: None

        def complete(config, output, predecessor, array_cap, check, report):
            self.assertEqual(config, self.bound_config)
            self.assertEqual(predecessor, self.predecessor)
            self.assertEqual(array_cap, adapter.RESOURCES["array_byte_cap"])
            check()
            report.update(status="complete_image_context_candidate_bank", primary_members=8)

        return SimpleNamespace(THREAD_ENV=adapter.THREAD_ENV, BudgetStop=FixtureBudgetStop,
            require_compute=Mock(), load_config=Mock(return_value=copy.deepcopy(self.bound_config)),
            digest=lambda value: hashlib.sha256(adapter.encoded(value)).hexdigest(),
            atomic_json=write, budget=budget, run_search=Mock(side_effect=complete))

    def run_main(self, env=None):
        with patch.dict(os.environ, env or self.environment, clear=True), \
                patch.object(adapter.socket, "gethostname", return_value="TEST_COMPUTE_NODE"), \
                patch.object(adapter, "episode_root", return_value=self.root), \
                patch.object(adapter, "load_producer", return_value=self.producer) as loader, \
                redirect_stderr(io.StringIO()):
            status = adapter.main(["--authority", str(self.authority_path), "--output", str(self.output),
                                   "--deadline", str(self.deadline)])
        return status, loader

    def test_guards_refuse_before_producer_import_or_output_creation(self):
        cases = (
            ("unadopted", {"status": "proposed"}, {}),
            ("expired", {"new_absolute_deadline_unix": self.now}, {}),
            ("source_changed", {"original_source_sha256": "0"*64}, {}),
            ("binding_changed", {"original_config_id": "0"*64}, {}),
            ("wrong_predecessor", {"predecessor_job_id": "47672861"}, {}),
            ("too_many_CPUs", {}, {"SLURM_JOB_CPUS_PER_NODE": "17"}),
            ("wrong_memory", {}, {"SLURM_MEM_PER_NODE": "8192"}),
            ("larger_time_envelope", {}, {"SLURM_JOB_END_TIME": str(self.now-10+36061)}),
            ("extra_thread", {}, {"OMP_NUM_THREADS": "2"}),
            ("no_compute_allocation", {}, {"SLURM_JOB_ID": ""}),
        )
        for name, changes, environment in cases:
            with self.subTest(name=name):
                self.authority_path.write_text(json.dumps(dict(self.authority, **changes)))
                status, loader = self.run_main(dict(self.environment, **environment))
                self.assertEqual(status, 1)
                loader.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_new_execution_deadline_preserves_original_config_and_identity(self):
        before = self.source.read_bytes(), self.config_path.read_bytes(), self.predecessor.read_bytes()
        # Real Slurm records show +2/+3sec Start/End skew. Accept that metadata
        # while retaining the same Start+9h work deadline; memory may round CPUs.
        status, loader = self.run_main(dict(self.environment, SLURM_CPUS_PER_TASK="16",
            SLURM_JOB_CPUS_PER_NODE="16", SLURM_JOB_END_TIME=str(self.now-10+36003)))
        self.assertEqual(status, 0)
        loader.assert_called_once_with(self.root)
        self.producer.load_config.assert_called_once_with(self.config_path)
        self.producer.run_search.assert_called_once()
        settings = json.loads((self.output / "settings.json").read_text())
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(settings["config"], self.bound_config)
        self.assertEqual(settings["config"]["not_after_unix"], adapter.ORIGINAL_DEADLINE)
        self.assertGreater(settings["deadline"], settings["config"]["not_after_unix"])
        self.assertEqual(settings["adopted_authority"]["original_config_id"], self.authority["original_config_id"])
        self.assertFalse(report["full_false_bank_qualified"])
        self.assertFalse(report["scientific_release_authorized"])
        self.assertEqual(report["maximum_reserved_CPU_seconds"], 16*36000)
        self.assertEqual(before, (self.source.read_bytes(), self.config_path.read_bytes(), self.predecessor.read_bytes()))

    def test_stop_finalization_distinguishes_preserved_exhausted_and_error(self):
        for kind, code, status, checkpoint in (
            ("saved_resource_stop", 75, "resource_incomplete", True),
            ("early_resource_stop", 75, "resource_incomplete", False),
            ("exhausted", 2, "exhausted_unavailable", True),
            ("unexpected_error", 1, "stopped_error", False),
        ):
            with self.subTest(kind=kind):
                self.output = self.output.parent / kind

                def stop(config, output, predecessor, array_cap, check, report):
                    if kind == "early_resource_stop":
                        raise FixtureBudgetStop("time_or_signal")
                    if kind == "unexpected_error":
                        raise RuntimeError("fixture implementation failure")
                    report.update(status=status, checkpoint_available=True, member=0,
                        total_legal_proposals=1164556, total_accepted_proposals=1151838)
                    if kind == "saved_resource_stop":
                        report["budget_stop_reason"] = "time_or_signal"

                self.producer.run_search = Mock(side_effect=stop)
                actual, _ = self.run_main()
                self.assertEqual(actual, code)
                report = json.loads((self.output / "report.json").read_text())
                self.assertEqual(report["status"], status)
                self.assertEqual(report["checkpoint_available"], checkpoint)
                self.assertIn("finished_utc", report)
                self.assertIn("process_peak_rss_bytes", report)
                self.assertFalse(report["automatic_successor"])
                self.assertFalse(report["logical_counters_reset"])
                if checkpoint:
                    self.assertEqual(report["total_legal_proposals"], 1164556)


if __name__ == "__main__":
    if "-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Lifecycle checks require an actual coordinator-owned compute allocation")
    unittest.main()
