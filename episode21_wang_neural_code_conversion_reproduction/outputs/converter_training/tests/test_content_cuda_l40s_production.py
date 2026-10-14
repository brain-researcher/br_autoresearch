"""Two focused production-admission methods; mocked devices, no data or CUDA."""
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
import content_cuda_l40s_hardware as hardware
import test_content_cuda_remaining_admission as remaining
import test_content_cuda_roster_admission as roster


def amendment():
    return {**copy.deepcopy(hardware.AMENDMENT_FIXED),
            "approval_source": "fixture_only_not_empirical_authority",
            "selected_at": "fixture_before_first_execution"}


def device(name, gib):
    return SimpleNamespace(name=name, total_memory=gib * 1024**3)


class L40SProductionHardwareTests(unittest.TestCase):
    def test_seventeen_jobs_admit_measured_L40S_and_prior_devices_under_original_limits(self):
        expected = {"roster-006": "47382243", "roster-007": "47382244",
                    **{f"remaining-{i:03d}": str(job) for i, job in
                       zip((1, 2, 3, *range(8, 20)), range(47382245, 47382260))}}
        self.assertEqual(hardware.PENDING_STAGE_JOBS, expected)
        for name, job in expected.items():
            old = roster if name.startswith("roster") else remaining
            runner = old.runner
            with self.subTest(stage=name), old.fixture(name) as f, \
                    mock.patch.object(runner, "_load_operations") as load:
                original_grant = f.grant.read_bytes()
                resources, fixed = copy.deepcopy(runner.RESOURCES), copy.deepcopy(runner.FIXED)
                with mock.patch.object(f.torch.cuda, "get_device_properties",
                        return_value=device("NVIDIA L40S", 48)), self.assertRaises(runner.AdmissionError):
                    runner.admit(name)  # Evidence alone does not activate hardware.
                path = f.root / hardware.AMENDMENT_RELATIVE
                path.write_text(json.dumps(amendment()))
                (f.run / "allocation.txt").write_text(old.allocation({"JobId": job}))
                with mock.patch.dict(os.environ, {"SLURM_JOB_ID": job}):
                    for gpu_name, gib in (("Tesla V100-SXM2-32GB", 32),
                                          ("NVIDIA H100 80GB HBM3", 80), ("NVIDIA L40S", 48)):
                        with self.subTest(device=gpu_name), mock.patch.object(
                                f.torch.cuda, "get_device_properties", return_value=device(gpu_name, gib)):
                            context, budget = runner.admit(name)
                            self.assertEqual(context.profile["hardware_eligibility_amendment"], amendment())
                            self.assertEqual(context.allocation["job_id"], job)
                            self.assertEqual(context.profile["resources"], resources)
                            self.assertEqual(tuple(context.profile[k] for k in
                                             ("training_images", "layers", "iterations")), (4712, 19, 1024))
                            self.assertEqual(budget.remaining_seconds(), 169080)
                            self.assertEqual(f.torch.cuda.fractions[-1], (20 / gib, 0))
                            self.assertFalse(f.torch.backends.cuda.matmul.allow_tf32)
                            self.assertFalse(f.torch.backends.cudnn.allow_tf32)
                            source = context.profile["source"]
                            runner.resolver(context, operation="bind_content_inputs", source=source,
                                target=f.stage.target, budget=budget, execution_options=None)
                            runner.resolver(context, operation=runner.OPERATION, source=source,
                                target=f.stage.target, budget=budget,
                                execution_options=runner._execution_options(f.stage))
                    for gpu_name, gib in (("NVIDIA L40", 48), ("NVIDIA A100-SXM4-80GB", 80),
                                          ("NVIDIA L40S", 21)):
                        with mock.patch.object(f.torch.cuda, "get_device_properties",
                                return_value=device(gpu_name, gib)), self.assertRaises(runner.AdmissionError):
                            runner.admit(name)
                self.assertEqual(f.grant.read_bytes(), original_grant)
                self.assertEqual(runner.RESOURCES, resources)
                self.assertEqual(runner.FIXED, fixed)
                self.assertFalse((f.run / "claim.json").exists())
                load.assert_not_called()

    def test_changed_authority_wrong_job_and_excluded_stages_never_gain_L40S_access(self):
        valid = amendment()
        for old, name in ((remaining, "remaining-001"), (roster, "roster-006")):
            runner = old.runner
            with self.subTest(stage=name), old.fixture(name) as f, \
                    mock.patch.object(runner, "_load_operations") as load:
                path = f.root / hardware.AMENDMENT_RELATIVE
                bad = [{**valid, "authorized": False}, {**valid, "approval_source": ""},
                       {**valid, "selected_at": ""}, {**valid, "measurement_job": "999"},
                       {**valid, "allowed_device_names": valid["allowed_device_names"] + ["NVIDIA L40"]},
                       {**valid, "pending_stage_jobs": {**valid["pending_stage_jobs"], name: "999"}},
                       {**valid, "pending_stage_jobs": {**valid["pending_stage_jobs"], "remaining-000": "47377474"}},
                       {**valid, "resources": {**runner.RESOURCES, "minutes": 5760}}]
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
                    runner.admit(name)  # Fixture job123 is not a selected original job.
                cuda.assert_not_called()
                job = valid["pending_stage_jobs"][name]
                (f.run / "allocation.txt").write_text(old.allocation({"JobId": job}))
                with mock.patch.dict(os.environ, {"SLURM_JOB_ID": job}):
                    context, budget = runner.admit(name)
                    path.write_text(json.dumps({**valid, "approval_source": "changed_after_admission"}))
                    with self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, operation="bind_content_inputs",
                            source=context.profile["source"], target=f.stage.target,
                            budget=budget, execution_options=None)
                self.assertFalse((f.run / "claim.json").exists())
                load.assert_not_called()
        for old, name in ((remaining, "remaining-000"), (roster, "roster-005")):
            runner = old.runner
            with self.subTest(excluded=name), old.fixture(name) as f:
                path = f.root / hardware.AMENDMENT_RELATIVE
                for text in (json.dumps(valid), "{"):
                    path.write_text(text)
                    context, _ = runner.admit(name)
                    self.assertNotIn("hardware_eligibility_amendment", context.profile)
                    with mock.patch.object(f.torch.cuda, "get_device_properties",
                            return_value=device("NVIDIA L40S", 48)), self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                self.assertFalse((f.run / "claim.json").exists())


if __name__ == "__main__":
    unittest.main()
