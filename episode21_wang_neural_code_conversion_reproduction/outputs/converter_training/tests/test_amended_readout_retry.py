"""Stdlib-only future retry admission/control fixtures; no empirical inputs.

These tests neither activate a retry nor establish current-job closure or full
scientific integration. Only the coordinator may run them in verified compute.
"""
import builtins
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_amended_readout_retry as retry


def profiles():
    resources = copy.deepcopy(retry.amended.original.RESOURCES)
    science = {**retry.amended.FIXED, "authorized": True, "approval_source": "fixture source",
        "resources": resources, "input_scope": retry.amended.INPUT_SCOPE,
        "maximum_concurrent_draw_tasks": 8}
    selection = {"schema_version": "ep21.amended_readout_retry.v1", "authorized": True,
        "approval_source": "fixture terminal diagnosis", "attempt": "amended_full002",
        "science_profile": retry.SCIENCE_PROFILE, "resources": resources,
        "cleanup_reserve_seconds": {phase: 60 for phase in retry.PHASES},
        "maximum_concurrent_draw_tasks": 8,
        "prior_decode_terminal": {"job_id": "500", "state": "TIMEOUT", "exit_code": "0:15",
            "observed_at": "2020-01-01T00:00:00+00:00", "diagnosis": "fixture timeout"},
        "retained_attempt_bytes": 1024,
        "combined_storage_cap_bytes": 1024 + resources["decode"]["aggregate_bytes"]}
    return selection, science


def dependencies(first):
    return {"decode": {"job_id": str(first), "afterok_job_ids": [],
                       "completed_model_job_ids": [str(n) for n in range(1000, 1040)]},
            "draw": {"array_job_id": str(first + 1), "afterok_job_id": str(first)},
            "finalize": {"job_id": str(first + 2), "afterok_job_id": str(first + 1)}}


class RetryAdmissionTests(unittest.TestCase):
    def test_missing_or_inactive_selection_refuses_before_allocation_and_scientific_import(self):
        original_import = builtins.__import__
        scientific = []
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch", "h5py", "amended_distributed_readout",
                                      "amended_checkpointed_decode", "feature_protocol"}:
                scientific.append(name)
                raise AssertionError("scientific import before admission")
            return original_import(name, *args, **kwargs)
        for value in (None, {"authorized": False}):
            with self.subTest(value=value), mock.patch.object(retry, "_text",
                    side_effect=FileNotFoundError if value is None else None,
                    return_value=json.dumps(value)), \
                    mock.patch.object(retry.amended.core, "_allocation") as allocation, \
                    mock.patch.object(builtins, "__import__", side_effect=guarded_import), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(retry.main(["--phase", "decode"]), 2)
                allocation.assert_not_called()
        self.assertEqual(scientific, [])

    def read_profiles(self, selection, science):
        def read(path, **unused):
            if Path(path) == retry.PROFILE:
                return json.dumps(selection)
            if Path(path) == retry.BASE / retry.SCIENCE_PROFILE:
                return json.dumps(science)
            raise AssertionError("unexpected fixture metadata read")
        return read

    def test_fixed_science_drift_and_premature_terminal_claim_are_rejected(self):
        selection, science = profiles()
        read = self.read_profiles(selection, science)
        with mock.patch.object(retry, "_text", side_effect=read), \
                mock.patch.object(retry.amended, "_text", side_effect=read):
            accepted = retry._profile(retry._stage("decode"))
            self.assertEqual(accepted["science"]["decoded_slots"], 855)
            self.assertEqual(accepted["science"]["image_bootstrap_draws"], 1000)
            science["image_bootstrap_draws"] = 10
            with self.assertRaises(retry.amended.AdmissionError):
                retry._profile(retry._stage("decode"))
            science["image_bootstrap_draws"] = 1000
            selection["prior_decode_terminal"]["state"] = "RUNNING"
            with self.assertRaises(retry.amended.AdmissionError):
                retry._profile(retry._stage("decode"))
            selection["prior_decode_terminal"]["state"] = "TIMEOUT"
            selection["cleanup_reserve_seconds"]["decode"] = 10
            with self.assertRaises(retry.amended.AdmissionError):
                retry._profile(retry._stage("decode"))
            selection["cleanup_reserve_seconds"]["decode"] = 60
            selection["combined_storage_cap_bytes"] -= 1
            with self.assertRaises(retry.amended.AdmissionError):
                retry._profile(retry._stage("decode"))

    def test_same_producers_and_fresh_all_three_phase_bindings(self):
        profile, _ = profiles()
        stage = retry._stage("decode", resources=profile["resources"]["decode"])
        fresh, old = dependencies(700), dependencies(500)
        def read(path, **unused):
            return json.dumps(old if Path(path).parent == retry.PRIOR_ROOT else fresh)
        with mock.patch.object(retry, "_text", side_effect=read), \
                mock.patch.object(retry.amended, "_text", side_effect=read), \
                mock.patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "", "SLURM_ARRAY_TASK_ID": ""}):
            self.assertEqual(retry._bindings(stage, {"job_id": "700"}, profile), {"decode": fresh["decode"]})
            with self.assertRaises(retry.amended.AdmissionError):
                retry._bindings(stage, {"job_id": "999"}, profile)
            fresh["decode"]["completed_model_job_ids"][0] = "2000"
            with self.assertRaises(retry.amended.AdmissionError):
                retry._bindings(stage, {"job_id": "700"}, profile)
            fresh["decode"]["completed_model_job_ids"][0] = "1000"
            fresh["finalize"]["job_id"] = old["finalize"]["job_id"]
            with self.assertRaises(retry.amended.AdmissionError):
                retry._bindings(stage, {"job_id": "700"}, profile)
            fresh["finalize"]["job_id"] = "702"
            fresh["draw"]["afterok_job_id"] = "500"
            with self.assertRaises(retry.amended.AdmissionError):
                retry._bindings(stage, {"job_id": "700"}, profile)

    def test_work_stop_keeps_hard_cleanup_interval_and_cli_uses_hard_clock(self):
        selection, _ = profiles()
        selection["cleanup_reserve_seconds"]["decode"] = 11
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "decode").mkdir()
            stage = retry.amended.AmendedStage("decode", None, root / "authorization.json", root,
                root / "decode", "fixture", True, tuple(sorted(selection["resources"]["decode"].items())))
            allocation = {"job_id": "700", "hostname": "compute-fixture", "deadline_epoch": 120.}
            context = retry.RetryContext(selection, allocation, root, stage, {"decode": {}}, 0)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "700"}), \
                    mock.patch.object(retry.amended.core.socket, "gethostname", return_value="compute-fixture"), \
                    mock.patch.object(retry.time, "time", return_value=100.) as wall, \
                    mock.patch.object(retry.time, "monotonic", return_value=100.) as monotonic:
                # lifecycle's default clock was captured at import; inject the
                # same controlled clock without importing scientific dependencies.
                real_lifecycle = retry.DecodeLifecycle
                with mock.patch.object(retry, "DecodeLifecycle",
                        side_effect=lambda **kwargs: real_lifecycle(**kwargs, clock=lambda: wall())):
                    budget = retry.RetryBudget(context)
                self.assertEqual(budget.remaining_seconds(), 9)
                self.assertEqual(budget.hard_remaining_seconds(), 20)
                wall.return_value = monotonic.return_value = 110.
                with self.assertRaises(TimeoutError):
                    budget.check()
                self.assertEqual(budget.hard_remaining_seconds(), 10)
                output = io.StringIO()
                with mock.patch.object(retry, "admit", return_value=(context, budget)), redirect_stdout(output):
                    self.assertEqual(retry.main(["--phase", "decode", "--hard-remaining-seconds"]), 0)
                self.assertEqual(output.getvalue().strip(), "10")
                epoch_output = io.StringIO()
                with mock.patch.object(retry, "admit", return_value=(context, budget)), redirect_stdout(epoch_output):
                    self.assertEqual(retry.main(["--phase", "decode", "--hard-stop-epoch"]), 0)
                self.assertEqual(epoch_output.getvalue().strip(), "120")
                wall.return_value = monotonic.return_value = 120.
                with self.assertRaises(TimeoutError):
                    budget.hard_remaining_seconds()

    def test_exact_phase_paths_and_decode_options_keep_retry_separate(self):
        selection, _ = profiles()
        for phase, index in (("decode", None), ("draw", 62), ("finalize", None)):
            stage = retry._stage(phase, index, selection["resources"][phase])
            self.assertEqual(stage.root, retry.ROOT)
            self.assertNotEqual(stage.root, retry.PRIOR_ROOT)
            self.assertEqual(stage.profile_path, retry.PROFILE)
            self.assertIn("ep21_amended_full002_", stage.scratch_name)
        stage = retry._stage("decode", resources=selection["resources"]["decode"])
        lifecycle = mock.Mock(work_stop_epoch=110., hard_stop_epoch=120., record_path=stage.run_dir / "closure.json")
        budget = mock.Mock(lifecycle=lifecycle)
        self.assertEqual(retry._options(stage, budget), {
            "collection_path": stage.stage_h5, "maximum_bytes": stage.resources["collection_bytes"],
            "model_policy": retry.MODEL_POLICY, "work_stop_epoch": 110., "hard_stop_epoch": 120.,
            "closure_record_path": stage.run_dir / "closure.json"})
        with self.assertRaises(retry.amended.AdmissionError):
            retry._stage("draw", 63)


if __name__ == "__main__":
    unittest.main()
