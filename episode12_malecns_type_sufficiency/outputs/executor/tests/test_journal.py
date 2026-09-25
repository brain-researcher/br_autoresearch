from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ep12_executor.journal import (
    HashChainedJournal,
    IdempotencyConflict,
    JournalIntegrityError,
)
from ep12_executor.synthetic import SCRATCH_ROOT


class JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH_ROOT)
        self.path = Path(self.temporary.name) / "ledger.jsonl"
        self.journal = HashChainedJournal(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_identical_event_is_idempotent_but_changed_payload_conflicts(self) -> None:
        first = self.journal.append(
            event_id="event-1",
            timestamp_utc="2026-09-24T00:00:00Z",
            payload={"value": 1},
        )
        second = self.journal.append(
            event_id="event-1",
            timestamp_utc="2026-09-24T00:01:00Z",
            payload={"value": 1},
        )
        self.assertTrue(first.appended)
        self.assertFalse(second.appended)
        self.assertEqual(first.record["record_hash"], second.record["record_hash"])
        self.assertEqual(self.journal.verify()["record_count"], 1)
        with self.assertRaises(IdempotencyConflict):
            self.journal.append(
                event_id="event-1",
                timestamp_utc="2026-09-24T00:02:00Z",
                payload={"value": 2},
            )

    def test_tampering_is_detected(self) -> None:
        self.journal.append(
            event_id="event-1",
            timestamp_utc="2026-09-24T00:00:00Z",
            payload={"value": 1},
        )
        record = json.loads(self.path.read_text(encoding="utf-8"))
        record["value"] = 99
        self.path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(JournalIntegrityError):
            self.journal.verify()

    def test_partial_record_is_not_silently_truncated(self) -> None:
        self.path.write_text('{"partial":true}', encoding="utf-8")
        with self.assertRaises(JournalIntegrityError):
            self.journal.read()


if __name__ == "__main__":
    unittest.main()
