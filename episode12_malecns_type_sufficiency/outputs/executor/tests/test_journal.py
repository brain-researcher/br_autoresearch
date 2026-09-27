from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ep12_executor.journal import EventJournal, IdempotencyConflict, JournalIntegrityError


class JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "events.jsonl"
        self.journal = EventJournal(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_event_id_is_idempotent_without_a_hash_chain(self) -> None:
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
        self.assertEqual(self.journal.status()["record_count"], 1)
        self.assertNotIn("record_hash", first.record)
        with self.assertRaises(IdempotencyConflict):
            self.journal.append(
                event_id="event-1",
                timestamp_utc="2026-09-24T00:02:00Z",
                payload={"value": 2},
            )

    def test_duplicate_and_partial_records_are_rejected(self) -> None:
        record = {"event_id": "duplicate", "timestamp_utc": "now", "sequence": 1}
        encoded = json.dumps(record) + "\n"
        self.path.write_text(encoded + encoded, encoding="utf-8")
        with self.assertRaises(JournalIntegrityError):
            self.journal.read()
        self.path.write_text('{"partial":true}', encoding="utf-8")
        with self.assertRaises(JournalIntegrityError):
            self.journal.read()


if __name__ == "__main__":
    unittest.main()
