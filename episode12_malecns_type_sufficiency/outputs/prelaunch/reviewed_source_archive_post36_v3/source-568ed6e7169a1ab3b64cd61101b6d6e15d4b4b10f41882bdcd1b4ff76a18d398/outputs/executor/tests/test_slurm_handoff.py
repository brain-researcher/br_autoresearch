from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MERGED = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency/post36_staging_merged"
)
if str(MERGED) not in sys.path:
    sys.path.insert(0, str(MERGED))

from ep12_executor.journal import HashChainedJournal
from ep12_executor.policy import canonical_json
from ep12_executor.source_freeze import (
    SOURCE_FREEZE_RELATIVE,
    create_source_freeze_manifest,
    source_freeze_file_sha256,
    stage_source_freeze,
)

_CANDIDATE = (
    Path(__file__).resolve().parents[1] / "ep12_executor/slurm_handoff.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "ep12_executor.slurm_handoff_candidate", _CANDIDATE
)
assert _SPEC is not None and _SPEC.loader is not None
slurm_handoff = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = slurm_handoff
_SPEC.loader.exec_module(slurm_handoff)
SubmissionNotInvokedError = slurm_handoff.SubmissionNotInvokedError
SlurmHandoffError = slurm_handoff.SlurmHandoffError
_resolved_job_id = slurm_handoff._resolved_job_id
authenticate_handoff_context = slurm_handoff.authenticate_handoff_context
default_job_resolver = slurm_handoff.default_job_resolver
default_submitter = slurm_handoff.default_submitter
submit_handoff = slurm_handoff.submit_handoff


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SlurmHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "development_run002"
        self.transaction_root = self.run_root / "slurm_handoffs"
        self.child_script = self.root / "child.sbatch"
        self.child_script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        self.canonical_patch = mock.patch.object(
            slurm_handoff,
            "CANONICAL_CHILD_SCRIPTS",
            {
                "readiness_to_post36_tail": self.child_script.resolve(),
                "post36_tail_to_next_tail": self.child_script.resolve(),
                "post36_tail_to_procedure_lock": self.child_script.resolve(),
            },
        )
        self.canonical_patch.start()

    def tearDown(self) -> None:
        self.canonical_patch.stop()
        self.temporary.cleanup()

    def readiness_context(self) -> Path:
        context = self.run_root / "production_readiness"
        transition_identity = {
            "observed_prefix_contract_sha256": "b" * 64,
            "active_contract_sha256": "c" * 64,
            "transition_manifest_file_sha256": "d" * 64,
            "transition_receipt_file_sha256": "e" * 64,
            "transition_receipt_sha256": "f" * 64,
        }
        report = {
            "schema_version": "ep12.production_readiness_gate.v2",
            "policy_sha256": "a" * 64,
            "checks": [{"check_id": "synthetic", "passed": True}],
            "failed_foundational_checks": [],
            "production_readiness_passed": True,
            "claim_guards": {
                "observed_post36_tail_launch": {"permitted": True, "missing": []}
            },
            **transition_identity,
            "observed_outcome_values_read": False,
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        report_path = context / "production_readiness.json"
        _write_json(report_path, report)
        _write_json(
            context / "manifest.json",
            {
                "schema_version": "ep12.production_readiness_gate_manifest.v2",
                "files": {"production_readiness.json": _sha256(report_path)},
                "production_readiness_passed": True,
                **transition_identity,
                "procedure_locked": False,
                "full_search_null_completed": False,
                "final_connectivity_accessed": False,
                "final_connectivity_access_authorized": False,
            },
        )
        return context

    def test_readiness_v2_binds_exact_transition_identity(self) -> None:
        context = self.readiness_context()
        authenticated = authenticate_handoff_context(
            kind="readiness_to_post36_tail",
            context_root=context,
            run_root=self.run_root,
        )
        self.assertEqual(
            authenticated["readiness_schema_version"],
            "ep12.production_readiness_gate.v2",
        )
        self.assertEqual(
            authenticated["readiness_manifest_schema_version"],
            "ep12.production_readiness_gate_manifest.v2",
        )
        self.assertEqual(
            authenticated["scientific_transition"]["active_contract_sha256"],
            "c" * 64,
        )

        manifest_path = context / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["active_contract_sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(SlurmHandoffError, "disagrees"):
            authenticate_handoff_context(
                kind="readiness_to_post36_tail",
                context_root=context,
                run_root=self.run_root,
            )

    def tail_context(self, *, stopped: bool) -> Path:
        context = self.run_root / "post36_tail"
        journal = HashChainedJournal(context / "ledger.jsonl")
        journal.append(
            event_id="controller:initialize",
            timestamp_utc="2026-09-26T00:00:00Z",
            payload={
                "record_kind": "controller_initialization",
                "schema_version": "ep12.post36_tail_controller.v2",
                "final_connectivity_accessed": False,
            },
        )
        ordinal = 36
        if stopped:
            reason = "patience_exhausted"
            journal.append(
                event_id="controller:stop",
                timestamp_utc="2026-09-26T00:01:00Z",
                payload={
                    "record_kind": "controller_stop",
                    "schema_version": "ep12.post36_tail_controller.v2",
                    "valid_trial_ordinal": ordinal,
                    "reason": reason,
                    "final_connectivity_accessed": False,
                },
            )
            action = "stopped"
            next_slot = None
        else:
            ordinal = 37
            reason = None
            journal.append(
                event_id="slot:post36_037:disposition",
                timestamp_utc="2026-09-26T00:01:00Z",
                payload={
                    "record_kind": "slot_disposition",
                    "schema_version": "ep12.post36_tail_controller.v2",
                    "trial_id": "post36_037",
                    "valid_trial_ordinal": ordinal,
                    "valid_ordinal_advanced": True,
                    "final_connectivity_accessed": False,
                },
            )
            action = "continue"
            next_slot = "post36_038"
        ledger = journal.verify()
        _write_json(
            context / "controller_status.json",
            {
                "schema_version": "ep12.post36_tail_status.v2",
                "action": action,
                "reason": reason,
                "current_valid_ordinal": ordinal,
                "next_slot_id": next_slot,
                "ledger": ledger,
                "procedure_locked": False,
                "full_search_null_completed": False,
                "final_connectivity_accessed": False,
                "final_connectivity_access_authorized": False,
            },
        )
        if stopped:
            schedule_path = context / "realized_schedule.json"
            _write_json(
                schedule_path,
                {
                    "schema_version": "ep12.post36_realized_schedule.v2",
                    "stop_valid_trial_ordinal": ordinal,
                    "stop_reason": reason,
                    "completed_promotable_search": True,
                    "ledger_record_count": ledger["record_count"],
                    "ledger_head_hash": ledger["head_hash"],
                    "ledger_hash": ledger["journal_hash"],
                    "procedure_locked": False,
                    "full_search_null_completed": False,
                    "final_connectivity_accessed": False,
                    "final_connectivity_access_authorized": False,
                },
            )
            _write_json(
                context / "realized_schedule_manifest.json",
                {
                    "schema_version": "ep12.post36_realized_schedule_manifest.v2",
                    "files": {
                        "realized_schedule.json": _sha256(schedule_path),
                        "ledger.jsonl": _sha256(context / "ledger.jsonl"),
                    },
                    "ledger_head_hash": ledger["head_hash"],
                    "procedure_locked": False,
                    "full_search_null_completed": False,
                    "final_connectivity_accessed": False,
                },
            )
        return context

    def arguments(self, context: Path) -> dict[str, object]:
        source_freeze = {
            "manifest_path": "/tmp/ep12-source-freeze.json",
            "manifest_file_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "file_count": 42,
        }
        return {
            "kind": "readiness_to_post36_tail",
            "parent_job_id": "70001",
            "context_root": context,
            "transaction_root": self.transaction_root,
            "child_script": self.child_script,
            "exported": (
                "EP12_SOURCE_FREEZE_MANIFEST=/tmp/ep12-source-freeze.json,"
                f"EP12_SOURCE_FREEZE_MANIFEST_SHA256={'a' * 64},"
                f"EP12_SOURCE_STAGE_ROOT={slurm_handoff.SOURCE_STAGE_PARENT}/"
                f"source-{'a' * 64}"
            ),
            "run_root": self.run_root,
            "source_freeze_authenticator": lambda _exports, _script: dict(source_freeze),
        }

    def test_submission_is_exactly_bound_and_replay_never_resubmits(self) -> None:
        commands: list[list[str]] = []
        resolver_calls: list[str] = []

        def resolve(name: str) -> str | None:
            resolver_calls.append(name)
            return None

        def submit(command: object) -> str:
            commands.append(list(command))
            return "71001;sherlock"

        arguments = self.arguments(self.readiness_context())
        receipt = submit_handoff(
            **arguments, submitter=submit, job_resolver=resolve  # type: ignore[arg-type]
        )
        replay = submit_handoff(
            **arguments,
            submitter=lambda _: self.fail("receipt replay submitted another child"),
            job_resolver=lambda _: self.fail("receipt replay queried Slurm"),
        )
        self.assertEqual(receipt, replay)
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[0:2], ["sbatch", "--parsable"])
        self.assertIn("--dependency=afterany:70001", command)
        self.assertNotIn("--dependency=afterok:70001", command)
        self.assertIn("--kill-on-invalid-dep=yes", command)
        self.assertIn(
            "--export=EP12_SOURCE_FREEZE_MANIFEST=/tmp/ep12-source-freeze.json,"
            f"EP12_SOURCE_FREEZE_MANIFEST_SHA256={'a' * 64},"
            f"EP12_SOURCE_STAGE_ROOT={slurm_handoff.SOURCE_STAGE_PARENT}/"
            f"source-{'a' * 64}",
            command,
        )
        self.assertFalse(receipt["ambient_environment_inherited"])
        self.assertEqual(receipt["source_freeze"]["file_count"], 42)
        self.assertEqual(command[-1], str(self.child_script.resolve()))
        name_option = next(item for item in command if item.startswith("--job-name="))
        self.assertEqual(name_option.split("=", 1)[1], receipt["child_job_name"])
        self.assertEqual(resolver_calls, [receipt["child_job_name"]])
        directory = (
            self.transaction_root
            / "readiness_to_post36_tail"
            / "parent-70001"
        )
        self.assertTrue((directory / "intent.json").is_file())
        self.assertTrue((directory / "progress.json").is_file())
        self.assertTrue((directory / "handoff_receipt.json").is_file())
        intent = json.loads((directory / "intent.json").read_text(encoding="utf-8"))
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(intent["schema_version"], "ep12.slurm_child_handoff_intent.v3")
        self.assertEqual(progress["schema_version"], "ep12.slurm_child_handoff_progress.v3")
        self.assertEqual(receipt["schema_version"], "ep12.slurm_child_handoff_receipt.v3")
        self.assertEqual(receipt["dependency"], "afterany:70001")
        self.assertFalse(receipt["parent_success_required"])
        self.assertTrue(receipt["authenticated_context_committed_before_submission"])
        self.assertTrue(receipt["child_independently_authenticates_context"])

    def test_accepted_submission_is_recovered_after_client_interruption(self) -> None:
        accepted: dict[str, str] = {}
        recovery_visible = False
        submission_count = 0
        commands: list[list[str]] = []

        def resolve(name: str) -> str | None:
            return accepted.get(name) if recovery_visible else None

        def submit(command: object) -> str:
            nonlocal submission_count
            submission_count += 1
            commands.append(list(command))
            name = next(
                item.split("=", 1)[1]
                for item in list(command)
                if item.startswith("--job-name=")
            )
            accepted[name] = "72001"
            raise RuntimeError("synthetic client interruption after acceptance")

        arguments = self.arguments(self.readiness_context())
        with self.assertRaisesRegex(SlurmHandoffError, "acceptance is uncertain"):
            submit_handoff(
                **arguments, submitter=submit, job_resolver=resolve  # type: ignore[arg-type]
            )
        progress_path = (
            self.transaction_root
            / "readiness_to_post36_tail"
            / "parent-70001"
            / "progress.json"
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["state"], "submission_uncertain")
        self.assertEqual(len(commands), 1)
        self.assertIn("--dependency=afterany:70001", commands[0])
        self.assertNotIn("--dependency=afterok:70001", commands[0])

        # The parent may now fail due to the lost acknowledgement.  Its
        # durable context is unchanged and remains sufficient for the child to
        # authenticate independently when afterany releases it.
        committed_context = authenticate_handoff_context(
            kind="readiness_to_post36_tail",
            context_root=arguments["context_root"],  # type: ignore[arg-type]
            run_root=self.run_root,
        )
        intent_path = progress_path.with_name("intent.json")
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        self.assertEqual(intent["launch"]["context"], committed_context)
        self.assertFalse(committed_context["final_connectivity_accessed"])

        # Scheduler/accounting propagation still has not exposed the accepted
        # job.  A later transaction retry must fail closed without sbatch.
        with self.assertRaisesRegex(SlurmHandoffError, "refusing duplicate sbatch"):
            submit_handoff(
                **arguments,
                submitter=lambda _: self.fail("uncertain retry submitted a duplicate"),
                job_resolver=resolve,
            )
        self.assertEqual(submission_count, 1)

        recovery_visible = True
        receipt = submit_handoff(
            **arguments,
            submitter=lambda _: self.fail("recovery submitted a duplicate"),
            job_resolver=resolve,
        )
        self.assertEqual(receipt["child_job_id"], "72001")
        self.assertEqual(receipt["dependency"], "afterany:70001")
        self.assertFalse(receipt["parent_success_required"])
        self.assertEqual(submission_count, 1)
        self.assertFalse(receipt["duplicate_submission_permitted"])

    def test_known_pre_invocation_failure_remains_safely_retryable(self) -> None:
        arguments = self.arguments(self.readiness_context())
        submission_count = 0

        def not_invoked(_: object) -> str:
            nonlocal submission_count
            submission_count += 1
            raise SubmissionNotInvokedError("synthetic exec failure")

        with self.assertRaisesRegex(SubmissionNotInvokedError, "exec failure"):
            submit_handoff(
                **arguments, submitter=not_invoked, job_resolver=lambda _: None
            )
        progress_path = (
            self.transaction_root
            / "readiness_to_post36_tail"
            / "parent-70001"
            / "progress.json"
        )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["state"], "planned")
        receipt = submit_handoff(
            **arguments, submitter=lambda _: "72501", job_resolver=lambda _: None
        )
        self.assertEqual(receipt["child_job_id"], "72501")
        self.assertEqual(submission_count, 1)

    def test_mutated_receipt_and_mutated_request_fail_closed(self) -> None:
        arguments = self.arguments(self.readiness_context())
        submit_handoff(
            **arguments, submitter=lambda _: "73001", job_resolver=lambda _: None
        )
        receipt_path = (
            self.transaction_root
            / "readiness_to_post36_tail"
            / "parent-70001"
            / "handoff_receipt.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["child_job_id"] = "99999"
        receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SlurmHandoffError, "digest mismatch"):
            submit_handoff(
                **arguments,
                submitter=lambda _: self.fail("mutated receipt resubmitted"),
                job_resolver=lambda _: self.fail("mutated receipt queried Slurm"),
            )

        other_root = self.root / "other"
        other_run = other_root / "development_run002"
        self.run_root = other_run
        self.transaction_root = other_run / "slurm_handoffs"
        fresh = self.arguments(self.readiness_context())
        submit_handoff(**fresh, submitter=lambda _: "73002", job_resolver=lambda _: None)
        fresh["exported"] = (
            "EP12_SOURCE_FREEZE_MANIFEST=/tmp/ep12-source-freeze.json,"
            f"EP12_SOURCE_FREEZE_MANIFEST_SHA256={'c' * 64},"
            f"EP12_SOURCE_STAGE_ROOT={slurm_handoff.SOURCE_STAGE_PARENT}/"
            f"source-{'c' * 64}"
        )
        with self.assertRaisesRegex(SlurmHandoffError, "intent differs"):
            submit_handoff(
                **fresh,
                submitter=lambda _: self.fail("changed request resubmitted"),
                job_resolver=lambda _: self.fail("changed request queried Slurm"),
            )

    def test_tail_continue_and_terminal_contexts_are_ledger_bound(self) -> None:
        continuation = authenticate_handoff_context(
            kind="post36_tail_to_next_tail",
            context_root=self.tail_context(stopped=False),
            run_root=self.run_root,
        )
        self.assertEqual(continuation["context_kind"], "tail_continue")
        continuation_status = json.loads(
            (self.run_root / "post36_tail/controller_status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            continuation_status["schema_version"],
            "ep12.post36_tail_status.v2",
        )
        self.assertEqual(continuation["current_valid_ordinal"], 37)

        terminal_root = self.root / "terminal" / "development_run002"
        self.run_root = terminal_root
        terminal = authenticate_handoff_context(
            kind="post36_tail_to_procedure_lock",
            context_root=self.tail_context(stopped=True),
            run_root=self.run_root,
        )
        self.assertEqual(terminal["context_kind"], "tail_stopped")
        self.assertIn("realized_schedule_sha256", terminal)
        terminal_context = self.run_root / "post36_tail"
        self.assertEqual(
            json.loads(
                (terminal_context / "realized_schedule.json").read_text(
                    encoding="utf-8"
                )
            )["schema_version"],
            "ep12.post36_realized_schedule.v2",
        )
        self.assertEqual(
            json.loads(
                (terminal_context / "realized_schedule_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["schema_version"],
            "ep12.post36_realized_schedule_manifest.v2",
        )
        status_path = self.run_root / "post36_tail" / "controller_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["ledger"]["head_hash"] = "0" * 64
        _write_json(status_path, status)
        with self.assertRaisesRegex(SlurmHandoffError, "exact ledger"):
            authenticate_handoff_context(
                kind="post36_tail_to_procedure_lock",
                context_root=self.run_root / "post36_tail",
                run_root=self.run_root,
            )

    def test_scheduler_name_recovery_uses_squeue_then_sacct(self) -> None:
        responses = [
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                [], 0, stdout="74001|ep12-r2t-70001-token|zijiao\n", stderr=""
            ),
        ]
        with mock.patch.object(
            slurm_handoff.subprocess, "run", side_effect=responses
        ) as run, mock.patch.object(
            slurm_handoff, "_effective_user", return_value="zijiao"
        ):
            recovered = default_job_resolver("ep12-r2t-70001-token")
        self.assertEqual(recovered, "74001")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][0], "squeue")
        self.assertEqual(run.call_args_list[1].args[0][0], "sacct")
        with self.assertRaisesRegex(SlurmHandoffError, "multiple scheduler jobs"):
            _resolved_job_id(
                "74001|ep12-r2t-70001-token\n74002|ep12-r2t-70001-token\n",
                "ep12-r2t-70001-token",
            )

    def test_scheduler_recovery_rejects_squeue_sacct_disagreement(self) -> None:
        name = "ep12-r2t-70001-token"
        responses = [
            subprocess.CompletedProcess([], 0, stdout=f"74001|{name}|zijiao\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=f"74002|{name}|zijiao\n", stderr=""),
        ]
        with mock.patch.object(
            slurm_handoff.subprocess, "run", side_effect=responses
        ), mock.patch.object(
            slurm_handoff, "_effective_user", return_value="zijiao"
        ), self.assertRaisesRegex(SlurmHandoffError, "multiple scheduler jobs"):
            default_job_resolver(name)

    def test_ambient_export_is_rejected_before_submission(self) -> None:
        arguments = self.arguments(self.readiness_context())
        arguments["exported"] = (
            "ALL,EP12_SOURCE_FREEZE_MANIFEST=/tmp/ep12-source-freeze.json,"
            f"EP12_SOURCE_FREEZE_MANIFEST_SHA256={'a' * 64}"
        )
        with self.assertRaisesRegex(SlurmHandoffError, "ambient environment"):
            submit_handoff(
                **arguments,
                submitter=lambda _: self.fail("ambient launch reached sbatch"),
                job_resolver=lambda _: None,
            )

    def test_noncanonical_child_is_rejected_before_authentication(self) -> None:
        arguments = self.arguments(self.readiness_context())
        alternate = self.root / "alternate.sbatch"
        alternate.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        arguments["child_script"] = alternate
        arguments["source_freeze_authenticator"] = lambda *_: self.fail(
            "noncanonical child reached source-freeze authentication"
        )
        with self.assertRaisesRegex(SlurmHandoffError, "canonical wrapper"):
            submit_handoff(
                **arguments,
                submitter=lambda _: self.fail("noncanonical child reached sbatch"),
                job_resolver=lambda _: None,
            )

    def test_symlinked_transaction_kind_is_rejected_before_submission(self) -> None:
        arguments = self.arguments(self.readiness_context())
        self.transaction_root.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        (self.transaction_root / "readiness_to_post36_tail").symlink_to(
            outside, target_is_directory=True
        )
        with self.assertRaisesRegex(SlurmHandoffError, "safely open transaction"):
            submit_handoff(
                **arguments,
                submitter=lambda _: self.fail("symlinked transaction reached sbatch"),
                job_resolver=lambda _: None,
            )

    def test_default_submitter_ignores_adversarial_sbatch_environment(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="77001\n", stderr="")
        poison = {
            "PATH": "/tmp/hostile-client-bin",
            "SBATCH_EXPORT": "ALL",
            "SBATCH_PARTITION": "attacker",
            "SBATCH_MEM": "1M",
            "SBATCH_WRAP": "touch /tmp/should-not-run",
            "SLURM_EXPORT_ENV": "ALL",
        }
        with mock.patch.dict(os.environ, poison, clear=False), mock.patch.object(
            slurm_handoff.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(default_submitter(["sbatch", "--parsable", "x"]), "77001")
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertTrue(set(environment) <= {"PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE"})
        self.assertFalse(any(name.startswith(("SBATCH_", "SLURM_")) for name in environment))


class ImmutableStagedChildAuthenticationTests(unittest.TestCase):
    def test_staged_wrapper_remains_authenticated_after_live_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            boundary = Path(temporary).resolve()
            episode = boundary / "episode"
            scratch = boundary / "scratch"
            stage_parent = scratch / "source_stages"
            stage_parent.mkdir(parents=True)
            (episode / "SEARCH_POLICY.yaml").parent.mkdir(parents=True)
            (episode / "SEARCH_POLICY.yaml").write_text(
                "schema: staged-test\n", encoding="utf-8"
            )
            relative = Path("outputs/executor/slurm/run_post36_tail.sbatch")
            live_wrapper = episode / relative
            live_wrapper.parent.mkdir(parents=True)
            live_wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            create_source_freeze_manifest(episode_root=episode)
            manifest_path = episode / SOURCE_FREEZE_RELATIVE
            manifest_file_sha256 = source_freeze_file_sha256(
                manifest_path, episode_root=episode
            )
            stage = stage_source_freeze(
                manifest_path,
                expected_file_sha256=manifest_file_sha256,
                stage_parent=stage_parent,
                episode_root=episode,
                scratch_root=scratch,
            )
            staged_wrapper = stage / relative
            exports = {
                "EP12_SOURCE_FREEZE_MANIFEST": str(manifest_path),
                "EP12_SOURCE_FREEZE_MANIFEST_SHA256": manifest_file_sha256,
            }
            canonical = {
                key: live_wrapper
                for key in slurm_handoff.CANONICAL_CHILD_SCRIPTS
            }
            with mock.patch.multiple(
                slurm_handoff,
                EPISODE_ROOT=episode,
                SCRATCH_ROOT=scratch,
                SOURCE_STAGE_PARENT=stage_parent,
                CANONICAL_CHILD_SCRIPTS=canonical,
            ):
                selected = slurm_handoff._canonical_child_script(
                    "readiness_to_post36_tail",
                    staged_wrapper,
                    source_freeze_file_sha256=manifest_file_sha256,
                )
                first = slurm_handoff._authenticate_source_freeze(exports, selected)
                self.assertEqual(first["child_source_origin"], "immutable_source_stage")

                live_wrapper.write_text("#!/bin/bash\nexit 99\n", encoding="utf-8")
                second = slurm_handoff._authenticate_source_freeze(exports, selected)
                self.assertEqual(second, first)
                with self.assertRaisesRegex(
                    SlurmHandoffError, "source-freeze verification failed"
                ):
                    slurm_handoff._authenticate_source_freeze(exports, live_wrapper)


if __name__ == "__main__":
    unittest.main()
