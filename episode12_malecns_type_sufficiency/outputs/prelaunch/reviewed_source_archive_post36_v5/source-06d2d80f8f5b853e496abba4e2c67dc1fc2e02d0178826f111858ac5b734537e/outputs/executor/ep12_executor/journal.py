"""Crash-safe, idempotent, hash-chained JSONL journal for EP12."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .policy import canonical_json, digest_object


class JournalError(RuntimeError):
    """Base journal error."""


class JournalIntegrityError(JournalError):
    """The authoritative journal was malformed or its hash chain failed."""


class IdempotencyConflict(JournalError):
    """An idempotency key was reused for a different command."""


@dataclass(frozen=True)
class AppendResult:
    record: Mapping[str, Any]
    appended: bool


def record_hash(record_without_hash: Mapping[str, Any]) -> str:
    return digest_object(record_without_hash)


class HashChainedJournal:
    """A small-ledger implementation favoring correctness over throughput.

    Each append rewrites the complete JSONL file to a same-directory temporary
    file and atomically replaces the old file.  Qualification ledgers are
    small, so this gives simple crash semantics while retaining JSONL as the
    authoritative source required by the common controller protocol.
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
        previous: str | None = None
        event_ids: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JournalIntegrityError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise JournalIntegrityError(f"line {line_number} is not an object")
            if record.get("previous_record_hash") != previous:
                raise JournalIntegrityError(f"broken previous hash on line {line_number}")
            claimed = record.get("record_hash")
            if not isinstance(claimed, str):
                raise JournalIntegrityError(f"missing record hash on line {line_number}")
            unhashed = dict(record)
            del unhashed["record_hash"]
            if record_hash(unhashed) != claimed:
                raise JournalIntegrityError(f"record hash mismatch on line {line_number}")
            event_id = record.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise JournalIntegrityError(f"missing event id on line {line_number}")
            if event_id in event_ids:
                raise JournalIntegrityError(f"duplicate event id on line {line_number}")
            event_ids.add(event_id)
            records.append(record)
            previous = claimed
        return records

    def read(self) -> list[dict[str, Any]]:
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_unlocked()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def verify(self) -> dict[str, Any]:
        records = self.read()
        return {
            "record_count": len(records),
            "head_hash": records[-1]["record_hash"] if records else None,
            "journal_hash": digest_object(records),
        }

    def prefix_hash(self) -> str:
        records = self.read()
        return digest_object([record["record_hash"] for record in records])

    def append(
        self,
        *,
        event_id: str,
        timestamp_utc: str,
        payload: Mapping[str, Any],
    ) -> AppendResult:
        if not event_id:
            raise ValueError("event_id must be nonempty")
        request_hash = digest_object(payload)
        with self.lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                records = self._read_unlocked()
                for existing in records:
                    if existing["event_id"] == event_id:
                        if existing.get("request_hash") != request_hash:
                            raise IdempotencyConflict(
                                f"event id {event_id!r} was reused with different content"
                            )
                        return AppendResult(existing, appended=False)

                record = dict(payload)
                record.update(
                    {
                        "event_id": event_id,
                        "timestamp_utc": timestamp_utc,
                        "previous_record_hash": (
                            records[-1]["record_hash"] if records else None
                        ),
                        "request_hash": request_hash,
                    }
                )
                record["record_hash"] = record_hash(record)
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
