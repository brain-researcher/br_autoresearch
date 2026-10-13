"""Crash-safe exact cancellation of the superseded held attestation job.

This control-plane transaction runs only after the archive-bound replacement
submission receipt is durable.  It persists cancellation intent before sending
the first scancel byte, rechecks the exact live hold and same-user descendant
set, never repeats an uncertain cancellation, and requires unique terminal
accounting evidence before invoking the post36 continuation.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from .policy import digest_object
from .post36_failed_attempt_recovery import (
    verify_failed_post36_attempt_recovery,
)
from .post36_orchestration import (
    OBSERVED_ATTESTATION_JOB_ID,
    OBSERVED_ATTESTATION_JOB_NAME,
    Post36OrchestrationError,
    _atomic_json,
    _effective_user,
    _read_json,
    _run_scheduler,
    _sha256_file,
    _write_or_verify,
)
from .replacement_attestation_launch import (
    HOLD_FILE_SHA256,
    HOLD_RELATIVE,
    TRANSACTION_RELATIVE,
    ReplacementAttestationLaunchError,
    default_obsolete_state_resolver,
    verify_replacement_attestation_receipt,
)
from .secure_transaction import open_regular_lock_at, open_secure_directory


INTENT_SCHEMA = "ep12.obsolete_attestation_cancellation_intent.v2"
PROGRESS_SCHEMA = "ep12.obsolete_attestation_cancellation_progress.v3"
RECEIPT_SCHEMA = "ep12.obsolete_attestation_cancellation_receipt.v3"
CANCEL_ROOT_RELATIVE = TRANSACTION_RELATIVE / "obsolete_cancellation"
INTENT_RELATIVE = CANCEL_ROOT_RELATIVE / "intent.json"
PROGRESS_RELATIVE = CANCEL_ROOT_RELATIVE / "progress.json"
RECEIPT_RELATIVE = CANCEL_ROOT_RELATIVE / "receipt.json"
_DEPENDENCY_JOB_RE = re.compile(
    rf"(?<![0-9]){OBSERVED_ATTESTATION_JOB_ID}(?![0-9])"
)


class ObsoleteAttestationCancellationError(RuntimeError):
    """The obsolete held job cannot be cancelled uniquely and durably."""


JobStateResolver = Callable[[str, str, str], Mapping[str, str]]
DescendantResolver = Callable[[str, str], Sequence[Mapping[str, str]]]
Canceller = Callable[[Sequence[str]], str]
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ObsoleteAttestationCancellationError(message)


def default_descendant_resolver(
    obsolete_job_id: str, submission_user: str
) -> Sequence[Mapping[str, str]]:
    _require(submission_user == _effective_user(), "descendant query user mismatch")
    output = _run_scheduler(
        [
            "/usr/bin/squeue", "--noheader", "--user", submission_user,
            "--format=%i|%j|%u|%E",
        ]
    )
    descendants: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        _require(len(fields) == 4, "malformed descendant squeue row")
        job_id, job_name, observed_user, dependency = fields
        _require(observed_user == submission_user, "descendant query returned another user")
        if job_id == obsolete_job_id:
            continue
        if _DEPENDENCY_JOB_RE.search(dependency):
            _require(job_id.isdigit(), "descendant query returned invalid job ID")
            descendants.append(
                {
                    "job_id": job_id,
                    "job_name": job_name,
                    "user": observed_user,
                    "dependency": dependency,
                }
            )
    return sorted(descendants, key=lambda value: value["job_id"])


def default_canceller(command: Sequence[str]) -> str:
    return _run_scheduler(command).strip()


def _command(submission_user: str) -> list[str]:
    return [
        "/usr/bin/scancel",
        f"--user={submission_user}",
        f"--name={OBSERVED_ATTESTATION_JOB_NAME}",
        "--state=PENDING",
        OBSERVED_ATTESTATION_JOB_ID,
    ]


def _held(value: Mapping[str, str], submission_user: str) -> bool:
    return (
        value.get("job_id") == OBSERVED_ATTESTATION_JOB_ID
        and value.get("job_name") == OBSERVED_ATTESTATION_JOB_NAME
        and value.get("user") == submission_user
        and value.get("state") == "PENDING"
        and value.get("reason") in {"JobHeldUser", "(JobHeldUser)"}
        and value.get("priority") == "0"
    )


def _cancelled(value: Mapping[str, str], submission_user: str) -> bool:
    return (
        value.get("job_id") == OBSERVED_ATTESTATION_JOB_ID
        and value.get("job_name") == OBSERVED_ATTESTATION_JOB_NAME
        and value.get("user") == submission_user
        and str(value.get("state", "")).startswith("CANCELLED")
    )


def _intent(
    *,
    episode_root: Path,
    replacement: Mapping[str, Any],
    submission_user: str,
) -> dict[str, Any]:
    recorded_recovery = replacement.get("failed_attempt_recovery")
    carry_forward = replacement.get(
        "obsolete_job_already_terminally_cancelled"
    ) is True
    _require(
        carry_forward == isinstance(recorded_recovery, Mapping),
        "replacement cancellation mode lacks failed-attempt recovery evidence",
    )
    failed_attempt_recovery = (
        dict(recorded_recovery) if isinstance(recorded_recovery, Mapping) else None
    )
    core = {
        "schema_version": INTENT_SCHEMA,
        "submission_user": submission_user,
        "episode_root": str(episode_root),
        "obsolete_job_id": OBSERVED_ATTESTATION_JOB_ID,
        "obsolete_job_name": OBSERVED_ATTESTATION_JOB_NAME,
        "hold_record_path": str(episode_root / HOLD_RELATIVE),
        "hold_record_file_sha256": HOLD_FILE_SHA256,
        "replacement_job_id": replacement["slurm_job_id"],
        "replacement_receipt_path": replacement["launch_receipt_path"],
        "replacement_receipt_file_sha256": replacement["launch_receipt_file_sha256"],
        "replacement_receipt_sha256": replacement["launch_receipt_sha256"],
        "cancellation_mode": (
            "carry_forward_prior_terminal_cancellation"
            if carry_forward
            else "issue_exact_scancel"
        ),
        "cancellation_command": [] if carry_forward else _command(submission_user),
        "failed_attempt_recovery": failed_attempt_recovery,
        "descendants_immediately_before_cancel": [],
        "intent_persisted_before_first_scancel_byte": not carry_forward,
        "duplicate_scancel_after_uncertain_ack_allowed": False,
        "obsolete_result_authoritative": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def _progress(
    *,
    intent: Mapping[str, Any],
    state: str,
    terminal_scheduler: Mapping[str, str] | None,
) -> dict[str, Any]:
    _require(
        state in {"planned", "cancellation_uncertain", "cancel_acknowledged", "terminal_confirmed"},
        "invalid cancellation progress state",
    )
    _require(
        (state == "terminal_confirmed") == (terminal_scheduler is not None),
        "cancellation terminal-state binding is invalid",
    )
    if terminal_scheduler is not None:
        _require(
            set(terminal_scheduler)
            == {"job_id", "job_name", "user", "state", "reason", "priority"}
            and _cancelled(terminal_scheduler, str(intent["submission_user"])),
            "terminal cancellation scheduler row is invalid",
        )
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "state": state,
        "terminal_scheduler": (
            dict(terminal_scheduler) if terminal_scheduler is not None else None
        ),
        "obsolete_result_authoritative": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _verify_progress(
    value: Mapping[str, Any], *, intent: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _progress(
        intent=intent,
        state=str(value.get("state")),
        terminal_scheduler=(
            {str(key): str(item) for key, item in value["terminal_scheduler"].items()}
            if isinstance(value.get("terminal_scheduler"), Mapping)
            else None
        ),
    )
    _require(dict(value) == expected, "cancellation progress mutated")
    return expected


def _observe_terminal(
    *, submission_user: str, resolver: JobStateResolver
) -> dict[str, str]:
    """Perform one live-then-accounting observation transaction, never poll."""

    try:
        value = dict(
            resolver(
                OBSERVED_ATTESTATION_JOB_ID,
                OBSERVED_ATTESTATION_JOB_NAME,
                submission_user,
            )
        )
    except (ReplacementAttestationLaunchError, Post36OrchestrationError) as exc:
        raise ObsoleteAttestationCancellationError(
            f"terminal cancellation evidence is not yet available: {exc}"
        ) from exc
    _require(
        _cancelled(value, submission_user),
        "obsolete job is not yet terminally cancelled",
    )
    return value


def _receipt(
    *,
    intent: Mapping[str, Any],
    progress: Mapping[str, Any],
    scheduler: Mapping[str, str],
) -> dict[str, Any]:
    carry_forward = (
        intent.get("cancellation_mode")
        == "carry_forward_prior_terminal_cancellation"
    )
    recovery = intent.get("failed_attempt_recovery")
    _require(
        (not carry_forward and recovery is None)
        or (carry_forward and isinstance(recovery, Mapping)),
        "cancellation receipt recovery mode changed",
    )
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "progress_sha256": progress["progress_sha256"],
        "obsolete_job_id": OBSERVED_ATTESTATION_JOB_ID,
        "obsolete_job_name": OBSERVED_ATTESTATION_JOB_NAME,
        "submission_user": intent["submission_user"],
        "terminal_scheduler_state": scheduler["state"],
        "terminal_scheduler_reason": scheduler["reason"],
        "terminal_scheduler_priority": scheduler["priority"],
        "replacement_job_id": intent["replacement_job_id"],
        "replacement_receipt_file_sha256": intent[
            "replacement_receipt_file_sha256"
        ],
        "replacement_receipt_sha256": intent["replacement_receipt_sha256"],
        "descendants_immediately_before_cancel": [],
        "cancellation_evidence_mode": intent["cancellation_mode"],
        "scancel_was_exactly_scoped": True,
        "scancel_issued_for_this_replacement": not carry_forward,
        "prior_scancel_was_exactly_scoped": (
            recovery["prior_scancel_was_exactly_scoped"]
            if isinstance(recovery, Mapping)
            else None
        ),
        "prior_obsolete_cancellation_receipt_path": (
            recovery["prior_obsolete_cancellation_receipt_path"]
            if isinstance(recovery, Mapping)
            else None
        ),
        "prior_obsolete_cancellation_receipt_file_sha256": (
            recovery["prior_obsolete_cancellation_receipt_file_sha256"]
            if isinstance(recovery, Mapping)
            else None
        ),
        "prior_obsolete_cancellation_receipt_sha256": (
            recovery["prior_obsolete_cancellation_receipt_sha256"]
            if isinstance(recovery, Mapping)
            else None
        ),
        "lost_ack_recovery": (
            "prior_terminal_cancellation_receipt_plus_current_sacct_no_scancel"
            if carry_forward
            else "live_absence_plus_unique_terminal_sacct_no_duplicate_scancel"
        ),
        "obsolete_result_authoritative": False,
        "authoritative_attestation_job_identity": True,
        "replacement_attestation_result_authoritative": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def cancel_obsolete_attestation(
    *,
    episode_root: Path,
    state_resolver: JobStateResolver = default_obsolete_state_resolver,
    descendant_resolver: DescendantResolver = default_descendant_resolver,
    canceller: Canceller = default_canceller,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    try:
        _read_json(
            root / RECEIPT_RELATIVE,
            root=root,
            label="obsolete cancellation receipt",
            canonical=True,
        )
    except FileNotFoundError:
        require_released: bool | None = False
    else:
        # Recovery after the controller released R must stay scheduler-inert.
        # The full receipt check below authenticates the already-completed
        # cancellation while allowing either the held or released R state.
        require_released = None
    replacement = verify_replacement_attestation_receipt(
        episode_root=root, require_released=require_released
    )
    recorded_recovery = replacement.get("failed_attempt_recovery")
    verified_recovery: Mapping[str, Any] | None = None
    if recorded_recovery is not None:
        _require(
            isinstance(recorded_recovery, Mapping),
            "replacement failed-attempt recovery evidence is malformed",
        )
        verified_recovery = verify_failed_post36_attempt_recovery(
            episode_root=root, require_outputs_absent=False
        )
        _require(
            dict(recorded_recovery) == verified_recovery,
            "replacement failed-attempt recovery evidence changed",
        )
    submission_user = _effective_user()
    transaction, transaction_fd = open_secure_directory(
        root / CANCEL_ROOT_RELATIVE, root=root, create=True
    )
    lock_fd = open_regular_lock_at(transaction_fd, ".lock")
    os.close(transaction_fd)
    with os.fdopen(lock_fd, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            intent = _write_or_verify(
                root / INTENT_RELATIVE,
                _intent(
                    episode_root=root,
                    replacement=replacement,
                    submission_user=submission_user,
                ),
                root=root,
                label="obsolete cancellation intent",
            )
            try:
                progress = _verify_progress(
                    _read_json(
                        root / PROGRESS_RELATIVE,
                        root=root,
                        label="obsolete cancellation progress",
                        canonical=True,
                    ),
                    intent=intent,
                )
            except FileNotFoundError:
                progress = _progress(
                    intent=intent, state="planned", terminal_scheduler=None
                )
                _atomic_json(root / PROGRESS_RELATIVE, progress, root=root)
            try:
                existing = _read_json(
                    root / RECEIPT_RELATIVE,
                    root=root,
                    label="obsolete cancellation receipt",
                    canonical=True,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None:
                scheduler = {
                    "state": str(existing.get("terminal_scheduler_state")),
                    "reason": str(existing.get("terminal_scheduler_reason")),
                    "priority": str(existing.get("terminal_scheduler_priority")),
                }
                _require(
                    _cancelled(
                        {
                            **scheduler,
                            "job_id": OBSERVED_ATTESTATION_JOB_ID,
                            "job_name": OBSERVED_ATTESTATION_JOB_NAME,
                            "user": submission_user,
                        },
                        submission_user,
                    )
                    and progress["state"] == "terminal_confirmed"
                    and existing
                    == _receipt(
                        intent=intent,
                        progress=progress,
                        scheduler=scheduler,
                    ),
                    "obsolete cancellation receipt mutated",
                )
                return existing

            # Close the launch-verification/lock race: R must still have no
            # durable release transaction while the first cancellation is
            # authorized.
            verify_replacement_attestation_receipt(
                episode_root=root, require_released=False
            )

            state = str(progress["state"])
            scheduler: dict[str, str] | None = None
            if state == "planned":
                if (
                    intent["cancellation_mode"]
                    == "carry_forward_prior_terminal_cancellation"
                ):
                    current = {
                        str(key): str(value)
                        for key, value in replacement[
                            "obsolete_scheduler_observation_immediately_before_submit"
                        ].items()
                    }
                    _require(
                        _cancelled(current, submission_user),
                        "carried-forward obsolete cancellation is not terminal",
                    )
                    _require(
                        isinstance(verified_recovery, Mapping)
                        and str(
                            verified_recovery.get(
                                "prior_terminal_scheduler_state", ""
                            )
                        ).startswith("CANCELLED"),
                        "carried-forward cancellation lacks prior terminal evidence",
                    )
                    scheduler = current
                else:
                    current = dict(
                        state_resolver(
                            OBSERVED_ATTESTATION_JOB_ID,
                            OBSERVED_ATTESTATION_JOB_NAME,
                            submission_user,
                        )
                    )
                    _require(
                        _held(current, submission_user),
                        "obsolete job lost exact hold before cancellation",
                    )
                    descendants = list(
                        descendant_resolver(
                            OBSERVED_ATTESTATION_JOB_ID, submission_user
                        )
                    )
                    _require(
                        descendants == [],
                        "obsolete job gained a same-user queued descendant",
                    )
                    current = dict(
                        state_resolver(
                            OBSERVED_ATTESTATION_JOB_ID,
                            OBSERVED_ATTESTATION_JOB_NAME,
                            submission_user,
                        )
                    )
                    _require(
                        _held(current, submission_user),
                        "obsolete job lost exact hold after descendant check",
                    )
                    progress = _progress(
                        intent=intent,
                        state="cancellation_uncertain",
                        terminal_scheduler=None,
                    )
                    _atomic_json(root / PROGRESS_RELATIVE, progress, root=root)
                    canceller(list(intent["cancellation_command"]))
                    progress = _progress(
                        intent=intent,
                        state="cancel_acknowledged",
                        terminal_scheduler=None,
                    )
                    _atomic_json(root / PROGRESS_RELATIVE, progress, root=root)
                    scheduler = _observe_terminal(
                        submission_user=submission_user,
                        resolver=state_resolver,
                    )
            elif state in {"cancellation_uncertain", "cancel_acknowledged"}:
                scheduler = _observe_terminal(
                    submission_user=submission_user,
                    resolver=state_resolver,
                )
            elif state == "terminal_confirmed":
                scheduler = dict(progress["terminal_scheduler"])
            _require(scheduler is not None, "cancellation lost terminal scheduler evidence")
            progress = _progress(
                intent=intent,
                state="terminal_confirmed",
                terminal_scheduler=scheduler,
            )
            _atomic_json(root / PROGRESS_RELATIVE, progress, root=root)
            receipt = _receipt(
                intent=intent, progress=progress, scheduler=scheduler
            )
            return _write_or_verify(
                root / RECEIPT_RELATIVE,
                receipt,
                root=root,
                label="obsolete cancellation receipt",
            )
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def verify_obsolete_attestation_cancellation_receipt(
    *, episode_root: Path, receipt_path: Path | None = None
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    replacement = verify_replacement_attestation_receipt(
        episode_root=root, require_released=None
    )
    path = receipt_path or (root / RECEIPT_RELATIVE)
    _require(
        Path(os.path.abspath(os.fspath(path))) == root / RECEIPT_RELATIVE,
        "obsolete cancellation receipt path is not canonical",
    )
    intent = _read_json(
        root / INTENT_RELATIVE, root=root,
        label="obsolete cancellation intent", canonical=True,
    )
    _require(
        intent == _intent(
            episode_root=root,
            replacement=replacement,
            submission_user=_effective_user(),
        ),
        "obsolete cancellation intent mutated",
    )
    progress = _verify_progress(
        _read_json(
            root / PROGRESS_RELATIVE, root=root,
            label="obsolete cancellation progress", canonical=True,
        ),
        intent=intent,
    )
    _require(
        progress["state"] == "terminal_confirmed",
        "obsolete cancellation is not terminally confirmed",
    )
    receipt = _read_json(
        path, root=root, label="obsolete cancellation receipt", canonical=True
    )
    scheduler = {
        "state": str(receipt.get("terminal_scheduler_state")),
        "reason": str(receipt.get("terminal_scheduler_reason")),
        "priority": str(receipt.get("terminal_scheduler_priority")),
    }
    _require(
        _cancelled(
            {
                **scheduler,
                "job_id": OBSERVED_ATTESTATION_JOB_ID,
                "job_name": OBSERVED_ATTESTATION_JOB_NAME,
                "user": _effective_user(),
            },
            _effective_user(),
        ),
        "obsolete cancellation receipt is not terminal cancellation",
    )
    _require(
        receipt == _receipt(intent=intent, progress=progress, scheduler=scheduler),
        "obsolete cancellation receipt mutated",
    )
    return {
        "schema_version": "ep12.obsolete_attestation_cancellation_evidence.v1",
        "cancellation_intent_path": str(root / INTENT_RELATIVE),
        "cancellation_intent_file_sha256": _sha256_file(
            root / INTENT_RELATIVE, root=root,
            label="obsolete cancellation intent",
        ),
        "cancellation_intent_sha256": intent["intent_sha256"],
        "cancellation_receipt_path": str(path),
        "cancellation_receipt_file_sha256": _sha256_file(
            path, root=root, label="obsolete cancellation receipt",
        ),
        "cancellation_receipt_sha256": receipt["receipt_sha256"],
        "obsolete_job_id": OBSERVED_ATTESTATION_JOB_ID,
        "obsolete_job_name": OBSERVED_ATTESTATION_JOB_NAME,
        "terminal_scheduler_state": receipt["terminal_scheduler_state"],
        "replacement_job_id": replacement["slurm_job_id"],
        "replacement_receipt_file_sha256": replacement[
            "launch_receipt_file_sha256"
        ],
        "replacement_receipt_sha256": replacement["launch_receipt_sha256"],
        "superseded_result_authoritative": False,
        "authoritative_attestation_job_identity": True,
        "replacement_attestation_result_authoritative": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }
