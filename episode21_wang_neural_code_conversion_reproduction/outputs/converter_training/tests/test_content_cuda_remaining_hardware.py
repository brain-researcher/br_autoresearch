"""Two mocked hardware-admission checks; no CUDA execution or empirical data."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_content_cuda_remaining as runner
from test_content_cuda_remaining_admission import allocation, fixture


def amendment():
    return {**copy.deepcopy(runner.HARDWARE_AMENDMENT_FIXED),
            "approval_source": "fixture_only_not_empirical_authority"}


def device(name, gib):
    return SimpleNamespace(name=name, total_memory=gib * 1024**3)


class ContentCudaRemainingHardwareTests(unittest.TestCase):
    def test_only_two_qualified_devices_for_exact_pending_jobs_preserve_original_limits(self):
        jobs = {"remaining-000": "47377474",
                **{f"remaining-{stage:03d}": str(job) for stage, job in
                   zip((1, 2, 3, *range(8, 20)), range(47382245, 47382260))}}
        self.assertEqual(amendment()["pending_stage_jobs"], jobs)
        self.assertEqual(set(jobs), set(runner.STAGES))
        self.assertEqual(amendment()["allowed_device_names"],
                         ["Tesla V100-SXM2-32GB", "NVIDIA H100 80GB HBM3"])
        original_fixed = copy.deepcopy(runner.FIXED)
        original_resources = copy.deepcopy(runner.RESOURCES)
        for name, job in jobs.items():
            with self.subTest(stage=name), fixture(name) as f, \
                    mock.patch.object(runner, "_load_operations") as load:
                original_grant = f.grant.read_bytes()
                context, _ = runner.admit(name)
                self.assertNotIn("hardware_eligibility_amendment", context.profile)
                with mock.patch.object(f.torch.cuda, "get_device_properties",
                        return_value=device("NVIDIA H100 80GB HBM3", 80)), \
                        self.assertRaises(runner.AdmissionError):
                    runner.admit(name)  # No amendment retains original V100-only admission.
                path = f.root / runner.HARDWARE_AMENDMENT_RELATIVE
                path.write_text(json.dumps(amendment()))
                (f.run / "allocation.txt").write_text(allocation({"JobId": job}))
                with mock.patch.dict(os.environ, {"SLURM_JOB_ID": job}):
                    for gpu_name, gib in (("Tesla V100-SXM2-32GB", 32),
                                          ("NVIDIA H100 80GB HBM3", 80)):
                        with self.subTest(device=gpu_name), mock.patch.object(
                                f.torch.cuda, "get_device_properties", return_value=device(gpu_name, gib)):
                            context, budget = runner.admit(name)
                            self.assertEqual(context.profile["hardware_eligibility_amendment"], amendment())
                            self.assertEqual(context.allocation["job_id"], job)
                            self.assertEqual(context.profile["resources"], original_resources)
                            self.assertEqual(budget.remaining_seconds(), 169080)
                            self.assertEqual(f.torch.cuda.fractions[-1], (20 / gib, 0))
                            self.assertFalse(f.torch.backends.cuda.matmul.allow_tf32)
                            self.assertFalse(f.torch.backends.cudnn.allow_tf32)
                            self.assertIn(("threads", 1), f.torch.calls)
                            self.assertIn(("interop", 1), f.torch.calls)
                            self.assertIn(("precision", "highest"), f.torch.calls)
                            runner.resolver(context, operation="bind_content_inputs", source=f.stage.source,
                                            target=f.stage.target, budget=budget, execution_options=None)
                    with mock.patch.object(f.torch.cuda, "get_device_properties",
                            return_value=device("NVIDIA A100-SXM4-80GB", 80)), \
                            self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                    with mock.patch.object(f.torch.cuda, "get_device_properties",
                            return_value=device("NVIDIA H100 80GB HBM3", 21)), \
                            self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                self.assertEqual(f.grant.read_bytes(), original_grant)
                self.assertFalse((f.run / "claim.json").exists())
                load.assert_not_called()
        self.assertEqual(runner.FIXED, original_fixed)
        self.assertEqual(runner.RESOURCES, original_resources)

    def test_malformed_unselected_and_changed_amendments_refuse_without_scientific_import(self):
        name = "remaining-000"
        with fixture(name) as f, mock.patch.object(runner, "_load_operations") as load:
            path = f.root / runner.HARDWARE_AMENDMENT_RELATIVE
            valid = amendment()
            bad = [
                {**valid, "authorized": False},
                {**valid, "allowed_device_names": valid["allowed_device_names"] + ["NVIDIA A100-SXM4-80GB"]},
                {**valid, "pending_stage_jobs": {**valid["pending_stage_jobs"], name: "999"}},
                {**valid, "pending_stage_jobs": {**valid["pending_stage_jobs"], "roster-005": "47375558"}},
                {**valid, "pending_stage_jobs": {k: v for k, v in valid["pending_stage_jobs"].items() if k != name}},
                {**valid, "scheduler_constraint": "GPU_MEM:80GB"},
                {**valid, "original_grants_resources_science_unchanged": False},
                {**valid, "approval_source": ""},
                {**valid, "tf32": True},
                {**valid, "resources": {**runner.RESOURCES, "minutes": 5760}},
            ]
            with mock.patch.object(runner, "_allocation") as actual, \
                    mock.patch.object(runner, "_initialize_cuda") as cuda:
                for record in bad:
                    path.write_text(json.dumps(record))
                    with self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                path.write_text("{")
                with self.assertRaises(json.JSONDecodeError):
                    runner.admit(name)
                actual.assert_not_called()
                cuda.assert_not_called()
            path.write_text(json.dumps(valid))
            with mock.patch.object(runner, "_initialize_cuda") as cuda, \
                    self.assertRaises(runner.AdmissionError):
                runner.admit(name)  # Existing fixture job123 is not one of the selected jobs.
            cuda.assert_not_called()
            job = valid["pending_stage_jobs"][name]
            (f.run / "allocation.txt").write_text(allocation({"JobId": job}))
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": job}):
                context, budget = runner.admit(name)
                options = dict(operation="bind_content_inputs", source=f.stage.source,
                               target=f.stage.target, budget=budget, execution_options=None)
                runner.resolver(context, **options)
                path.write_text(json.dumps({**valid, "approval_source": "changed_after_admission"}))
                with self.assertRaises(runner.AdmissionError):
                    runner.resolver(context, **options)
                path.unlink()
                with self.assertRaises(runner.AdmissionError):
                    runner.resolver(context, **options)
            self.assertFalse((f.run / "claim.json").exists())
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
