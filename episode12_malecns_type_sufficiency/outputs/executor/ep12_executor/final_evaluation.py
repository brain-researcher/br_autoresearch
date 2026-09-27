"""Library-only one-shot final-opening primitive for EP12.

This module deliberately contains no MaleCNS outcome reader, scientific result
schema, command-line entry point, or Slurm launcher.  A future final evaluator
must supply the frozen interval calculation, bounded output validator, and
terminal-decision mapping before it can use this transaction core.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .policy import canonical_json
from .runtime import publish_once


DIRECTIONS = ("left_to_right", "right_to_left")


class FinalAccessError(RuntimeError):
    """The one-shot final opening cannot safely proceed or reconcile."""


class FinalTransactionHooks(Protocol):
    """Future trusted evaluator hooks; no implementation is shipped here."""

    def prepare(self, final_lock: Mapping[str, Any]) -> None:
        """Prepare without reading final outcome values."""

    def evaluate_and_commit(
        self, *, final_lock: Mapping[str, Any], result_path: Path
    ) -> Mapping[str, Any]:
        """Read final outcomes once and durably publish the bounded result."""

    def reconcile_committed(
        self, *, final_lock: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Retrieve an already committed result without reopening final data."""


ResultValidator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalAccessError(message)


def _read_regular_json(path: Path, label: str) -> dict[str, Any]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        metadata = path.lstat()
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        descriptor = os.open(path, os.O_RDONLY | nofollow)
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
                value = json.load(handle)
        finally:
            os.close(descriptor)
    except FinalAccessError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalAccessError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _publish_json(path: Path, value: Mapping[str, Any]) -> bool:
    encoded = (canonical_json(dict(value)) + "\n").encode("utf-8")
    try:
        return publish_once(path, encoded)
    except (FileExistsError, OSError) as error:
        raise FinalAccessError(str(error)) from error


def validate_final_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the immutable opening identity, never scientific outcomes."""

    lock = dict(value)
    expected = {
        "operation_id",
        "procedure_lock_id",
        "role_rows_id",
        "selected_trial_id",
        "final_types",
        "directions",
        "maximum_open_count",
    }
    _require(set(lock) == expected, "final lock fields are incomplete or unexpected")
    for field in ("operation_id", "procedure_lock_id", "role_rows_id", "selected_trial_id"):
        _require(
            isinstance(lock[field], str) and bool(lock[field].strip()),
            f"final lock requires {field}",
        )
    final_types = lock["final_types"]
    _require(
        isinstance(final_types, list)
        and final_types
        and all(isinstance(value, str) and value for value in final_types)
        and final_types == sorted(set(final_types)),
        "final lock requires one sorted unique nonempty final-type list",
    )
    _require(
        lock["directions"] == list(DIRECTIONS),
        "final lock requires both reciprocal directions",
    )
    _require(lock["maximum_open_count"] == 1, "final lock permits exactly one opening")
    return lock


def _opening(final_lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "operation_id": final_lock["operation_id"],
        "procedure_lock_id": final_lock["procedure_lock_id"],
        "role_rows_id": final_lock["role_rows_id"],
        "state": "opened",
        "final_open_count": 1,
    }


def _validate_opening(value: Mapping[str, Any], final_lock: Mapping[str, Any]) -> None:
    _require(dict(value) == _opening(final_lock), "final opening belongs to another operation")


def _failure(final_lock: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "operation_id": final_lock["operation_id"],
        "procedure_lock_id": final_lock["procedure_lock_id"],
        "state": "technical_failure",
        "reason": reason,
        "final_open_count": 1,
        "final_evaluation_completed": False,
        "search_updated_from_final": False,
    }


def publish_final_result(path: Path, result: Mapping[str, Any]) -> bool:
    """Publish a validator-bounded future result at its sole commit point."""

    return _publish_json(path, result)


def _open_mutex(output_dir: Path) -> int:
    """Open the transaction mutex without following a pre-existing link."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            output_dir / ".final-operation.lock",
            os.O_RDWR | os.O_CREAT | nofollow,
            0o600,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise FinalAccessError("final transaction mutex is not a regular file")
        return descriptor
    except FinalAccessError:
        raise
    except OSError as error:
        raise FinalAccessError(f"cannot open final transaction mutex: {error}") from error


def _validated_result(
    value: Mapping[str, Any],
    lock: Mapping[str, Any],
    validate_result: ResultValidator,
) -> dict[str, Any]:
    original = dict(value)
    normalized = dict(validate_result(original, lock))
    _require(
        normalized == original,
        "committed final result differs from the validator-bounded result",
    )
    return normalized


def _run_locked(
    *,
    lock: Mapping[str, Any],
    output_dir: Path,
    hooks: FinalTransactionHooks,
    validate_result: ResultValidator,
) -> dict[str, Any]:
    lock_path = output_dir / "final_lock.json"
    open_path = output_dir / "final_open.json"
    result_path = output_dir / "final_result.json"
    failure_path = output_dir / "final_failure.json"

    _publish_json(lock_path, lock)
    if failure_path.exists():
        failure = _read_regular_json(failure_path, "final failure")
        _require(
            failure == _failure(lock, str(failure.get("reason", ""))),
            "final failure record is malformed",
        )
        return failure
    if result_path.exists():
        _require(open_path.exists(), "final result exists without an opening")
        _validate_opening(_read_regular_json(open_path, "final opening"), lock)
        return _validated_result(
            _read_regular_json(result_path, "final result"),
            lock,
            validate_result,
        )

    if open_path.exists():
        _validate_opening(_read_regular_json(open_path, "final opening"), lock)
        candidate = hooks.reconcile_committed(final_lock=lock)
        if candidate is not None:
            normalized = dict(validate_result(candidate, lock))
            _publish_json(result_path, normalized)
            return normalized
        failure = _failure(lock, "opening_consumed_without_reconcilable_result")
        _publish_json(failure_path, failure)
        return failure

    # Preparation is repeatable and cannot read final outcome values.
    hooks.prepare(lock)
    _require(
        _publish_json(open_path, _opening(lock)),
        "final opening appeared while the transaction mutex was held",
    )
    try:
        candidate = hooks.evaluate_and_commit(final_lock=lock, result_path=result_path)
    except Exception as error:
        if result_path.exists():
            return _validated_result(
                _read_regular_json(result_path, "final result"),
                lock,
                validate_result,
            )
        failure = _failure(
            lock,
            f"final_evaluator_failed_after_open:{type(error).__name__}",
        )
        _publish_json(failure_path, failure)
        return failure

    normalized = dict(validate_result(candidate, lock))
    _publish_json(result_path, normalized)
    return normalized


def run_final_transaction(
    *,
    final_lock: Mapping[str, Any],
    output_dir: Path,
    hooks: FinalTransactionHooks,
    validate_result: ResultValidator,
) -> dict[str, Any]:
    """Run or reconcile one operation; this has no built-in data-access path.

    Only the process that atomically creates ``final_open.json`` may call the
    outcome-reading hook.  A retry may retrieve a separately committed result
    for the same operation, but it may never call that hook a second time.
    """

    lock = validate_final_lock(final_lock)
    _require(not output_dir.is_symlink(), "final transaction directory may not be a link")
    output_dir.mkdir(parents=True, exist_ok=True)
    mutex = _open_mutex(output_dir)
    try:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        return _run_locked(
            lock=lock,
            output_dir=output_dir,
            hooks=hooks,
            validate_result=validate_result,
        )
    finally:
        try:
            fcntl.flock(mutex, fcntl.LOCK_UN)
        finally:
            os.close(mutex)


__all__ = [
    "DIRECTIONS",
    "FinalAccessError",
    "FinalTransactionHooks",
    "publish_final_result",
    "run_final_transaction",
    "validate_final_lock",
]
