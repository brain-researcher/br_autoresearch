"""Focused orchestration regressions; no models, images or neural arrays.

Test reports are explicitly lightweight stand-ins in temporary directories.
They never publish a passing public-runtime report into the episode.
"""
import copy
import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_runtime_harness as harness


class PublicPreludeTests(unittest.TestCase):
    def setUp(self):
        self.parent = {"episode_id": "ep21", "approved": True,
                       "purpose": "regular_vgg19_derivative_export",
                       "includes_public_runtime_conformance": True,
                       "resources": {"jobs": 1, "cpus": 4, "memory_gib": 16,
                                     "minutes": 480, "gpus": 1,
                                     "durable_bytes": 1600000000000,
                                     "scratch_bytes": 340000000000}}
        self.env = {"SLURM_JOB_ID": "TEST_ONLY", "SLURM_CPUS_PER_TASK": "4",
                    "SLURM_MEM_PER_NODE": "16384", "SLURM_JOB_START_TIME": "1",
                    "SLURM_JOB_END_TIME": "2000", "SCRATCH": "/TEST_ONLY_SCRATCH"}
        self.report = {"schema_version": "ep21.public_runtime_conformance.v3",
                       "outcome": harness.OUTCOME, "scope": harness.SCOPE,
                       "resources": harness.RESOURCES,
                       "job_id": "TEST_ONLY_ORIGINAL_JOB", "strict_load": True,
                       "pre_relu_non_inplace": True,
                       "decoder_penalized_bias_structured_prediction": True,
                       "historical_Caffe_feature_parity": "unverified",
                       **{name: False for name in harness.BOUNDARY_FIELDS}}

    def test_larger_authorized_parent_is_accepted(self):
        harness.validate_allocation_authorization(self.parent, self.env)
        self.assertEqual(self.env["SLURM_MEM_PER_NODE"], "16384")

    def test_source_only_approval_cannot_authorize_parent(self):
        source_only = {"approved": True, "purpose": "public_runtime_conformance",
                       "resources": harness.RESOURCES}
        with self.assertRaisesRegex(RuntimeError, "substantive_EP21_allocation"):
            harness.validate_allocation_authorization(source_only, self.env)

    def test_unapproved_parent_and_resource_mismatch_stop(self):
        for field, value in (("approved", False), ("episode_id", "other_episode")):
            parent = copy.deepcopy(self.parent); parent[field] = value
            with self.assertRaises(RuntimeError):
                harness.validate_allocation_authorization(parent, self.env)
        with self.assertRaisesRegex(RuntimeError, "parent_allocation_approval"):
            harness.validate_allocation_authorization(self.parent, {**self.env, "SLURM_MEM_PER_NODE": "8192"})

    def test_substep_does_not_spoof_parent_memory(self):
        env = {**self.env, "SLURM_STEP_ID": "0", "SLURM_CPUS_PER_TASK": "1"}
        harness.validate_allocation_authorization(self.parent, env, public_step=True)
        self.assertEqual(env["SLURM_MEM_PER_NODE"], "16384")
        with self.assertRaisesRegex(RuntimeError, "bounded_public_step"):
            harness.validate_allocation_authorization(self.parent,
                {**env, "SLURM_STEP_ID": "batch"}, public_step=True)

    def test_step_requests_one_cpu_four_gib_no_gpus_in_existing_job(self):
        with mock.patch.dict(os.environ, self.env, clear=True):
            command = harness.step_command(Path("TEST_ONLY_authorization.json"), job_id="TEST_ONLY", timeout=120)
        for argument in ("--jobid=TEST_ONLY", "--exact", "--ntasks=1", "--cpus-per-task=1",
                         "--mem=4G", "--gres=none", "--time=00:30:00", "CUDA_VISIBLE_DEVICES=",
                         "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1"):
            self.assertIn(argument, command)
        self.assertEqual(command.count("srun"), 1)
        self.assertNotIn("sbatch", command)
        self.assertFalse(any(x.startswith("SLURM_MEM") for x in command))
        cache = next(x for x in command if x.startswith("XDG_CACHE_HOME="))
        self.assertIn("public-runtime-TEST_ONLY/cache", cache)
        self.assertNotIn("vgg19-export", cache)

    def test_public_cache_and_temporary_files_are_in_scratch_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cache").mkdir(); (root / "tmp").mkdir()
            (root / "TEST_ONLY_model_metadata").write_bytes(b"12345")
            (root / "cache/TEST_ONLY_cache").write_bytes(b"123456789")
            (root / "tmp/TEST_ONLY_temporary").write_bytes(b"12")
            self.assertEqual(harness.scratch_usage(root), 16)

    def test_deadline_is_capped_by_source_and_parent_limits(self):
        self.assertEqual(harness.stage_timeout(self.env, now=1900), 70)
        self.assertEqual(harness.stage_timeout(self.env, now=0), 1800)
        with self.assertRaisesRegex(RuntimeError, "allocation_deadline_reached"):
            harness.stage_timeout(self.env, now=1990)

    def test_parent_allocation_cannot_extend_approved_time(self):
        with self.assertRaisesRegex(RuntimeError, "parent_allocation_time_approval"):
            harness.validate_allocation_authorization(self.parent,
                {**self.env, "SLURM_JOB_END_TIME": "28802"})

    def test_unchanged_pass_reused_without_rewriting_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.report))
            original = path.read_bytes()
            result = harness.reuse_report(path)
            self.assertEqual(result["job_id"], "TEST_ONLY_ORIGINAL_JOB")
            self.assertEqual(path.read_bytes(), original)

    def test_completed_v2_report_keeps_its_legacy_provenance(self):
        report = {**self.report, "schema_version": "ep21.public_runtime_conformance.v2",
                  "binding": {"test_metadata_only": True, "runtime": "TEST_ONLY_legacy_runtime"}}
        self.assertIs(harness.validate_passed_report(report), report)

    def test_known_relevant_change_stops_cached_reuse(self):
        for use_environment in (False, True):
            with self.subTest(use_environment=use_environment), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(self.report))
                reason = "TEST_ONLY_changed_runtime"
                env = {**self.env, "EP21_PUBLIC_CONFORMANCE_RELEVANT_CHANGE": reason} if use_environment else self.env
                kwargs = {} if use_environment else {"relevant_change": reason}
                with mock.patch.dict(os.environ, env, clear=True), \
                     mock.patch.object(harness, "REPORT", path), \
                     mock.patch.object(harness, "read_allocation_authorization", return_value=self.parent), \
                     mock.patch.object(harness.subprocess, "run") as launch, \
                     mock.patch.object(Path, "read_text", side_effect=AssertionError("cached report must not be consumed after a known change")):
                    with self.assertRaisesRegex(RuntimeError, "known_public_conformance_change_requires_diagnosis"):
                        harness.run_in_allocation("TEST_ONLY_authorization.json", **kwargs)
                    launch.assert_not_called()

    def test_partial_or_boundary_violating_report_never_passes(self):
        partial = copy.deepcopy(self.report); partial["strict_load"] = False
        with self.assertRaisesRegex(RuntimeError, "public_report_not_passed"):
            harness.validate_passed_report(partial)
        for field in harness.BOUNDARY_FIELDS:
            report = copy.deepcopy(self.report); report[field] = True
            with self.assertRaisesRegex(RuntimeError, "public_report_boundary"):
                harness.validate_passed_report(report)

    def test_incomplete_attempt_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "incomplete_public_attempt"):
                harness.reuse_report(Path(directory) / "absent.json")

    def test_report_cap_is_checked_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.report))
            with mock.patch.dict(harness.RESOURCES, durable_bytes=1):
                with self.assertRaisesRegex(RuntimeError, "public_report_cap"):
                    harness.reuse_report(path)

    def test_cached_pass_avoids_binding_source_reads_and_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(self.report))
            with mock.patch.dict(os.environ, self.env, clear=True), \
                 mock.patch.object(harness, "REPORT", path), \
                 mock.patch.object(harness, "SOURCE_AUTH") as source, \
                 mock.patch.object(harness, "read_allocation_authorization", return_value=self.parent), \
                 mock.patch.object(harness, "current_binding", create=True) as binding, \
                 mock.patch.object(Path, "read_bytes", side_effect=AssertionError("public source or asset read")), \
                 mock.patch.object(harness.subprocess, "run") as launch:
                source.read_text.side_effect = AssertionError("source authority only needed for actual public work")
                result = harness.run_in_allocation("TEST_ONLY_authorization.json")
                self.assertEqual(result, self.report)
                binding.assert_not_called()
                source.read_text.assert_not_called()
                launch.assert_not_called()

    def test_timeout_stops_without_retry_or_continuation(self):
        with mock.patch.dict(os.environ, self.env, clear=True), \
             mock.patch.object(harness, "read_allocation_authorization", return_value=self.parent), \
             mock.patch.object(harness, "reuse_report", return_value=None), \
             mock.patch.object(harness, "stage_timeout", return_value=120), \
             mock.patch.object(harness.subprocess, "run", side_effect=subprocess.TimeoutExpired("TEST_ONLY", 120)) as launch:
            with self.assertRaisesRegex(RuntimeError, "public_stage_timeout_no_continuation"):
                harness.run_in_allocation("TEST_ONLY_authorization.json")
            launch.assert_called_once()


class FeaturePreludeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.root = root
        (root / "outputs/pipeline_conformance/tools").mkdir(parents=True)
        shutil.copyfile(Path(__file__).with_name("execution_adapter.py"),
                        root / "outputs/pipeline_conformance/tools/execution_adapter.py")
        path = harness.BASE / "outputs/steward_provisioning/tools/export_regular_vgg19.py"
        spec = importlib.util.spec_from_file_location("TEST_ONLY_feature_draft", path)
        self.feature = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.feature)
        # This local orchestration fixture does not establish the exporter's
        # real cluster runtime; simulate its prerequisite to reach the guard.
        runtime = mock.patch.object(self.feature, "sys",
                                   SimpleNamespace(version_info=(3, 12, 1), modules=sys.modules))
        runtime.start(); self.addCleanup(runtime.stop)
        self.feature.BASE = root
        self.feature.PROV = root / "outputs/steward_provisioning"
        self.feature.PROV.mkdir()
        auth = {"approved": True, "purpose": "regular_vgg19_derivative_export",
                "resources": self.feature.RESOURCES,
                "allow_ood": False, "allow_neural_values": False,
                "allow_fitting": False, "allow_scoring": False}
        self.auth = root / "outputs/regular_vgg19_export_authorization_2026-09-30.json"
        self.auth.write_text(json.dumps(auth))
        self.environment = mock.patch.dict(os.environ, {"SLURM_JOB_ID": "TEST_ONLY",
            "SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "16384"})
        self.environment.start(); self.addCleanup(self.environment.stop)

    def test_public_failure_stops_before_empirical_guard(self):
        with mock.patch.object(self.feature, "run_in_allocation", side_effect=RuntimeError("TEST_ONLY_public_failure")), \
             mock.patch.object(self.feature.importlib.util, "spec_from_file_location") as guard:
            with self.assertRaisesRegex(RuntimeError, "TEST_ONLY_public_failure"):
                self.feature.main()
            guard.assert_not_called()

    def test_public_pass_still_requires_real_lock_and_context_before_payload_imports(self):
        original_import = builtins.__import__
        def forbid_payload_import(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch", "h5py", "PIL"}:
                raise AssertionError("payload package reached before real context")
            return original_import(name, *args, **kwargs)
        with mock.patch.object(self.feature, "run_in_allocation", return_value={"TEST_ONLY_public_report": True}) as public, \
             mock.patch("builtins.__import__", side_effect=forbid_payload_import):
            with self.assertRaisesRegex(ValueError, "E_EMPIRICAL_CONTEXT_UNAVAILABLE"):
                self.feature.main()
            public.assert_called_once_with(self.auth)
        self.assertFalse((self.feature.PROV / "regular_vgg19_export_job_claim.json").exists())

    def test_missing_feature_authority_stops_before_prelude(self):
        self.auth.unlink()
        with mock.patch.object(self.feature, "run_in_allocation") as public:
            with self.assertRaises(FileNotFoundError):
                self.feature.main()
            public.assert_not_called()


if __name__ == "__main__":
    unittest.main()
