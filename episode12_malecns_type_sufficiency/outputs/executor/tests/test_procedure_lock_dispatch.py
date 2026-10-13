from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ep12_executor.procedure_lock_dispatch as dispatch_module
from ep12_executor.contract_roles import ACTIVE_EPOCH, PREFIX_EPOCH, RANK_RULE
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.procedure_lock_dispatch import (
    ProcedureDispatchError,
    SubmissionRejectedError,
    dispatch_procedure_lock,
)
from ep12_executor.runtime import EPISODE_ROOT


class ProcedureLockDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def _lock(self) -> dict:
        core = {
            "policy_id": self.policy.policy_id,
            "policy_sha256": self.policy.policy_hash,
            "prefix_contract_id": "ep12_outgoing_tum_v1",
            "active_contract_id": "ep12_outgoing_tum_post36_v2",
            "prefix_contract_epoch": PREFIX_EPOCH,
            "active_contract_epoch": ACTIVE_EPOCH,
            "prevalence_parameter_rank_definition": RANK_RULE,
            "pre_null_input_binding": {
                "bundle_sha256": "1" * 64,
                "runtime_source_binding_sha256": "2" * 64,
                "callback_data_sha256": "3" * 64,
                "model_state_sha256": "4" * 64,
                "snapshot_binding_sha256": "5" * 64,
            },
            "procedure_locked": True,
            "null_eligible": True,
            "full_search_null_contract": {
                "required_replicates": self.policy.null_replicates,
                "pass_threshold": self.policy.null_pass_threshold,
                "rerun_complete_frozen_controller": True,
                "observed_realized_schedule_may_be_replayed": False,
            },
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        return {**core, "procedure_lock_sha256": digest_object(core)}

    def _paths(self, root: Path, lock: dict) -> dict:
        outputs = root / "outputs"
        output_dir = outputs / "development_run002" / "pre_null_dispatch"
        output_dir.mkdir(parents=True)
        lock_path = root / "procedure_lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        null_script = root / "null.sbatch"
        aggregate_script = root / "aggregate.sbatch"
        null_script.write_text("#!/bin/bash\n", encoding="utf-8")
        aggregate_script.write_text("#!/bin/bash\n", encoding="utf-8")
        runtime = root / "runtime"
        runtime.mkdir()
        (runtime / "runtime_bundle.json").write_text(
            json.dumps({"bundle_sha256": "a" * 64}), encoding="utf-8"
        )
        authorization = root / "authorization.json"
        authorization.write_text("{}", encoding="utf-8")
        return {
            "outputs": outputs,
            "output_dir": output_dir,
            "procedure_lock_path": lock_path,
            "full_search_null_script": null_script,
            "full_search_null_aggregate_script": aggregate_script,
            "full_null_work_root": root / "scratch-work",
            "full_null_durable_root": outputs / "development_run002" / "full-null",
            "resource_authorization_path": authorization,
            "runtime_bundle": runtime,
        }

    def _dispatch(self, lock: dict, paths: dict, submitter, scheduler_lookup=None) -> dict:
        with (
            patch(
                "ep12_executor.procedure_lock_dispatch.OUTPUTS_ROOT",
                paths["outputs"],
            ),
            patch(
                "ep12_executor.procedure_lock_dispatch.verify_resource_authorization",
                return_value={
                    "scope": "minimum_full_search",
                    "new_hard_cpu_ceiling": None,
                    "final_connectivity_access_authorized": False,
                },
            ),
        ):
            arguments = {
                "lock": lock,
                "policy": self.policy,
                "submitter": submitter,
                **{key: value for key, value in paths.items() if key != "outputs"},
            }
            if scheduler_lookup is not None:
                arguments["scheduler_lookup"] = scheduler_lookup
            return dispatch_procedure_lock(
                **arguments,
            )

    def test_scheduler_reconciliation_requires_one_exact_identity(self) -> None:
        name = "ep12-null-a-token"
        comment = "ep12-dispatch:array:token"
        with patch.object(
            dispatch_module,
            "_run_scheduler_query",
            side_effect=(
                f"801|{name}|{comment}\n801|{name}|{comment}\n",
                f"801_0|{name}|{comment}\n801_1|{name}|{comment}\n",
            ),
        ):
            self.assertEqual(
                dispatch_module._default_scheduler_lookup(
                    name, comment, "2026-09-26"
                ),
                "801",
            )
        with patch.object(
            dispatch_module,
            "_run_scheduler_query",
            side_effect=(
                f"801|{name}|{comment}\n",
                f"802|{name}|{comment}\n",
            ),
        ):
            with self.assertRaisesRegex(ProcedureDispatchError, "multiple Slurm jobs"):
                dispatch_module._default_scheduler_lookup(
                    name, comment, "2026-09-26"
                )

    def test_signal_interrupted_sbatch_is_ambiguous_not_rejected(self) -> None:
        interrupted = subprocess.CalledProcessError(
            -9, ["sbatch"], stderr="terminated"
        )
        with patch.object(dispatch_module.subprocess, "run", side_effect=interrupted):
            with self.assertRaisesRegex(ProcedureDispatchError, "outcome is unresolved"):
                dispatch_module._default_submitter(["sbatch", "job.sbatch"])
        rejected = subprocess.CalledProcessError(
            1, ["sbatch"], stderr="invalid request"
        )
        with patch.object(dispatch_module.subprocess, "run", side_effect=rejected):
            with self.assertRaisesRegex(SubmissionRejectedError, "invalid request"):
                dispatch_module._default_submitter(["sbatch", "job.sbatch"])

    def test_retry_after_aggregate_submit_failure_does_not_duplicate_array(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-dispatch-retry-") as name:
            root = Path(name)
            lock = self._lock()
            paths = self._paths(root, lock)
            first_calls: list[list[str]] = []

            def fail_second(command):
                first_calls.append(list(command))
                if len(first_calls) == 1:
                    return "111"
                raise SubmissionRejectedError("injected aggregate submit failure")

            with self.assertRaisesRegex(ProcedureDispatchError, "injected"):
                self._dispatch(lock, paths, fail_second)
            state = json.loads(
                (paths["output_dir"] / "dispatch_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["phase"], "array_submitted")
            self.assertEqual(state["array_job_id"], "111")
            self.assertIsNone(state["aggregate_job_id"])

            retry_calls: list[list[str]] = []

            def finish(command):
                retry_calls.append(list(command))
                return "222"

            launch = self._dispatch(lock, paths, finish)
            self.assertEqual(len(retry_calls), 1)
            self.assertIn("--dependency=afterok:111", retry_calls[0])
            self.assertIn("--kill-on-invalid-dep=yes", retry_calls[0])
            self.assertEqual(launch["jobs_submitted"][0]["job_id"], "111")
            self.assertEqual(launch["jobs_submitted"][1]["job_id"], "222")

            def unexpected(_command):
                raise AssertionError("completed dispatch must not submit again")

            self.assertEqual(self._dispatch(lock, paths, unexpected), launch)

    def test_array_submit_crash_reconciles_by_unique_scheduler_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-dispatch-array-crash-") as name:
            root = Path(name)
            lock = self._lock()
            paths = self._paths(root, lock)
            calls: list[list[str]] = []
            real_write = dispatch_module._write_state
            crashed = False

            def crash_after_acceptance(path, value):
                nonlocal crashed
                if value.get("phase") == "array_submitted" and not crashed:
                    crashed = True
                    raise RuntimeError("injected post-submit crash")
                real_write(path, value)

            def submit_array(command):
                calls.append(list(command))
                return "501"

            with (
                patch.object(
                    dispatch_module, "_write_state", side_effect=crash_after_acceptance
                ),
                self.assertRaisesRegex(RuntimeError, "post-submit"),
            ):
                self._dispatch(lock, paths, submit_array)
            self.assertEqual(len(calls), 1)
            self.assertTrue(any(value.startswith("--job-name=ep12-null-a-") for value in calls[0]))
            self.assertTrue(any(value.startswith("--comment=ep12-dispatch:array:") for value in calls[0]))
            state_path = paths["output_dir"] / "dispatch_state.json"
            self.assertEqual(json.loads(state_path.read_text())["phase"], "array_submitting")

            def no_duplicate(_command):
                raise AssertionError("uncertain array submission must not be repeated")

            with self.assertRaisesRegex(ProcedureDispatchError, "refusing a duplicate"):
                self._dispatch(
                    lock,
                    paths,
                    no_duplicate,
                    scheduler_lookup=lambda _name, _comment, _date: None,
                )

            aggregate_calls: list[list[str]] = []

            def submit_aggregate(command):
                aggregate_calls.append(list(command))
                return "502"

            launch = self._dispatch(
                lock,
                paths,
                submit_aggregate,
                scheduler_lookup=lambda _name, comment, _date: (
                    "501" if ":array:" in comment else None
                ),
            )
            self.assertEqual(len(aggregate_calls), 1)
            self.assertIn("--dependency=afterok:501", aggregate_calls[0])
            self.assertEqual(launch["jobs_submitted"][0]["job_id"], "501")
            self.assertEqual(launch["jobs_submitted"][1]["job_id"], "502")

    def test_aggregate_submit_crash_reconciles_without_resubmission(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-dispatch-aggregate-crash-") as name:
            root = Path(name)
            lock = self._lock()
            paths = self._paths(root, lock)
            responses = iter(("601", "602"))
            real_write = dispatch_module._write_state
            crashed = False

            def crash_after_acceptance(path, value):
                nonlocal crashed
                if value.get("phase") == "aggregate_submitted" and not crashed:
                    crashed = True
                    raise RuntimeError("injected aggregate ledger crash")
                real_write(path, value)

            with (
                patch.object(
                    dispatch_module, "_write_state", side_effect=crash_after_acceptance
                ),
                self.assertRaisesRegex(RuntimeError, "aggregate ledger"),
            ):
                self._dispatch(lock, paths, lambda _command: next(responses))
            state = json.loads(
                (paths["output_dir"] / "dispatch_state.json").read_text()
            )
            self.assertEqual(state["phase"], "aggregate_submitting")
            self.assertEqual(state["array_job_id"], "601")

            def no_submit(_command):
                raise AssertionError("accepted aggregate must be reconciled, not resubmitted")

            launch = self._dispatch(
                lock,
                paths,
                no_submit,
                scheduler_lookup=lambda _name, comment, _date: (
                    "602" if ":aggregate:" in comment else None
                ),
            )
            self.assertEqual(launch["jobs_submitted"][0]["job_id"], "601")
            self.assertEqual(launch["jobs_submitted"][1]["job_id"], "602")

    def test_inconsistent_dispatch_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-dispatch-invalid-state-") as name:
            root = Path(name)
            lock = self._lock()
            paths = self._paths(root, lock)
            responses = iter(("701", "702"))
            self._dispatch(lock, paths, lambda _command: next(responses))
            state_path = paths["output_dir"] / "dispatch_state.json"
            state = json.loads(state_path.read_text())
            state["phase"] = "authorized"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ProcedureDispatchError, "pre-array"):
                self._dispatch(
                    lock,
                    paths,
                    lambda _command: (_ for _ in ()).throw(
                        AssertionError("invalid state must not submit")
                    ),
                )

    def test_existing_launch_from_another_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-dispatch-stale-") as name:
            root = Path(name)
            lock = self._lock()
            paths = self._paths(root, lock)
            responses = iter(("333", "444"))
            self._dispatch(lock, paths, lambda _command: next(responses))
            (paths["runtime_bundle"] / "runtime_bundle.json").write_text(
                json.dumps({"bundle_sha256": "b" * 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ProcedureDispatchError, "another lock, runtime"):
                self._dispatch(lock, paths, lambda _command: "555")


if __name__ == "__main__":
    unittest.main()
