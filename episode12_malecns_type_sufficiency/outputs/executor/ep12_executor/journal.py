"""Crash-safe, append-only event journal for the EP12 scientific tail."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .policy import canonical_json


class JournalError(RuntimeError):
    """Base journal error."""


class JournalIntegrityError(JournalError):
    """The journal is truncated, malformed, or repeats an event ID."""


class IdempotencyConflict(JournalError):
    """An event ID was reused for different content."""


@dataclass(frozen=True)
class AppendResult:
    record: Mapping[str, Any]
    appended: bool


_LEGACY_IDENTITY_FIELDS = frozenset(
    {"previous_record_hash", "record_hash", "request_hash"}
)


def _event_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"event_id", "timestamp_utc", "sequence"} | _LEGACY_IDENTITY_FIELDS
    return {key: value for key, value in record.items() if key not in excluded}


class EventJournal:
    """An atomic JSONL journal with event-ID idempotency.

    Old hash fields are accepted as inert historical fields so an existing
    scientific run remains readable, but new records do not create or require
    a hash chain, receipt, or environment identity.
    """

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise JournalIntegrityError("journal does not end at a record boundary")
        records: list[dict[str, Any]] = []
        event_ids: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JournalIntegrityError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise JournalIntegrityError(f"line {line_number} is not an object")
            event_id = record.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise JournalIntegrityError(f"missing event id on line {line_number}")
            if event_id in event_ids:
                raise JournalIntegrityError(f"duplicate event id on line {line_number}")
            event_ids.add(event_id)
            records.append(record)
        return records

    def read(self) -> list[dict[str, Any]]:
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def status(self) -> dict[str, Any]:
        records = self.read()
        return {
            "record_count": len(records),
            "last_event_id": records[-1]["event_id"] if records else None,
        }

    def position(self) -> dict[str, Any]:
        """Return the visible append position used by deterministic proposals."""

        return self.status()

    def append(
        self,
        *,
        event_id: str,
        timestamp_utc: str,
        payload: Mapping[str, Any],
    ) -> AppendResult:
        if not event_id:
            raise ValueError("event_id must be nonempty")
        requested_payload = dict(payload)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_unlocked()
                for existing in records:
                    if existing["event_id"] == event_id:
                        if _event_payload(existing) != requested_payload:
                            raise IdempotencyConflict(
                                f"event id {event_id!r} was reused with different content"
                            )
                        return AppendResult(existing, appended=False)

                record = {
                    **requested_payload,
                    "event_id": event_id,
                    "timestamp_utc": timestamp_utc,
                    "sequence": len(records) + 1,
                }
                updated = records + [record]
                serialized = "".join(canonical_json(item) + "\n" for item in updated)
                temporary = self.path.with_name(
                    f".{self.path.name}.tmp.{os.getpid()}.{len(updated)}"
                )
                try:
                    with temporary.open("x", encoding="utf-8") as handle:
                        handle.write(serialized)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.path)
                    directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                return AppendResult(record, appended=True)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
