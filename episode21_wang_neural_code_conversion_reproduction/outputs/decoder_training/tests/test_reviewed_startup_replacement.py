"""Exact work0013 startup repair fixtures; no real scheduler or result access."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import submit_full_decoder_jobs as submit
import test_reviewed_deadline_partials as fixtures


class ReviewedStartupReplacementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReviewedDeadlinePartialTests()
        self.fixture.setUp()
        self.authority, self.manifest = self.fixture.authority, self.fixture.manifest
        prior = self.fixture.batch(("work_0013",))
        failed = self.fixture.fixture.terminal("46880582_12", elapsed=12, state="FAILED", allocation_job_id="46880582")
        failed["exit_code"] = "1:0"
        self.rows = self.fixture.rows + [{"event": "submission_intent", "batch_id": "batch_0003", **prior},
            self.fixture.fixture.accepted("46880582", "batch_0003"), failed]
        self.review = dict(schema_version="ep21.reviewed_startup_replacement.v1", reviewer="coordinating_agent",
            reviewed_utc="2026-10-07T09:04:00Z", predecessor_job_id="46880582_12",
            predecessor_allocation_job_id="46880582", work_item_id="work_0013", array_index=12,
            failure_reason="required_deadline_partial_fixture_precondition", shard_directory_absent=True,
            request_or_status_created=False, scientific_payload_admitted=False, optional_children_started=False,
            automatic_retry=False, fix_evidence="fixture: preserve completed worker in negative test",
            terminal_evidence="fixture: FAILED1:0 after12s", absence_evidence="fixture: shard absent",
            user_authorization_context="fixture: same authorized training, corrected startup check")

    def account(self, extra=()):
        return accounting.replay(self.rows+list(extra), self.authority, self.manifest)

    def batch(self):
        return submit.build_startup_replacement(self.authority, self.manifest, self.review,
            self.fixture.review, "normal", constraint="CPU_GEN:SIE")

    def intent(self):
        return {"event": "submission_intent", "batch_id": "batch_0004", **self.batch()}

    def test_exact_second_attempt_preserves_failed_charge_assignment_and_partial_results(self):
        before = self.account()
        batch = self.batch()
        submit.admit_batch(before, self.manifest, batch, results=self.fixture.results)
        self.assertIn("--array=12%1", batch["command"])
        self.assertEqual(batch["command"][-1], str(submit.REFILL_WRAPPER))
        pending = self.account([self.intent()])
        self.assertNotIn("46880582_12", pending["superseded_pre_admission_jobs"])
        accepted = self.fixture.fixture.accepted("500", "batch_0004")
        after = self.account([self.intent(), accepted])
        self.assertEqual(after["actual_CPU_seconds"], before["actual_CPU_seconds"])
        self.assertEqual(after["assigned"], before["assigned"])
        self.assertEqual(after["jobs"]["46880582_12"]["terminal"], self.rows[-1])
        self.assertEqual(after["jobs"]["500_12"]["logical_id"], "work_0013")
        self.assertIn("46880582_12", after["superseded_pre_admission_jobs"])
        self.assertEqual(submit.summary(after)["reviewed_extra_engineering_attempts"], 1)
        self.assertEqual(after["outstanding_reserved_CPU_seconds"], 5*28800)
        ordinary = self.fixture.batch(("work_0013",))
        with self.assertRaisesRegex(submit.Stop, "submitted_twice"):
            submit.admit_batch(before, self.manifest, ordinary, results=self.fixture.results)

    def test_other_failures_or_repeated_chained_replacements_remain_prohibited(self):
        for key, value in (("predecessor_job_id", "46880582_11"), ("array_index", 11),
            ("scientific_payload_admitted", True), ("optional_children_started", True),
            ("shard_directory_absent", False), ("failure_reason", "different")):
            with self.subTest(key=key), self.assertRaises(accounting.AccountingError):
                accounting.validate_startup_replacement_review({**self.review, key: value})
        for key, value in (("elapsed_seconds", 13), ("exit_code", "2:0"), ("allocation_job_id", "999")):
            rows = copy.deepcopy(self.rows)
            rows[-1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(accounting.AccountingError, "12_second"):
                accounting.replay(rows+[self.intent()], self.authority, self.manifest)
        next_intent = {**self.intent(), "batch_id": "batch_0005"}
        with self.assertRaisesRegex(accounting.AccountingError, "one_exact_startup"):
            self.account([self.intent(), self.fixture.fixture.accepted("500", "batch_0004"), next_intent])
        changed = self.intent()
        del changed[accounting.MANUAL_REFILL_FIELD]
        with self.assertRaises(accounting.AccountingError):
            self.account([changed])

    def test_current_known_queue_and_budget_rules_apply_without_legacy_repair_relaxation(self):
        account, batch = self.account(), self.batch()
        queue = "".join(f"200_{index}|ep21-full-workers-001|RUNNING\n" for index in range(4, 8))
        runner = mock.Mock(return_value=SimpleNamespace(stdout=queue))
        submit.require_manual_refill_queue(account, self.authority, batch, runner)
        self.assertEqual(runner.call_count, 1)
        with self.assertRaisesRegex(submit.Stop, "missing_reserved_worker"):
            submit.require_manual_refill_queue(account, self.authority, batch,
                mock.Mock(return_value=SimpleNamespace(stdout=queue.splitlines()[0]+"\n")))
        account["actual_CPU_seconds"] = 1800000-account["outstanding_reserved_CPU_seconds"]-28800
        submit.admit_batch(account, self.manifest, batch, results=self.fixture.results)
        account["actual_CPU_seconds"] += 1
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            submit.admit_batch(account, self.manifest, batch, results=self.fixture.results)
        with self.assertRaises(accounting.AccountingError):
            accounting.validate_replacement_review(self.review)

    def test_full_review_is_durable_before_single_submission_and_future_refill_reuses_it(self):
        originals = copy.deepcopy((self.authority, self.manifest))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            with accounting.locked_events(path, create=True) as (rows, append):
                for row in self.rows:
                    append(row)
                def scheduler(command, **kwargs):
                    event = json.loads(path.read_text().splitlines()[-1])
                    self.assertEqual(event[accounting.STARTUP_REPLACEMENT_FIELD], self.review)
                    self.assertEqual(event[submit.deadline_partials.FIELD], self.fixture.review)
                    return SimpleNamespace(returncode=0, stdout="500\n", stderr="")
                runner = mock.Mock(side_effect=scheduler)
                submit.submit_one(self.batch(), rows, append, self.authority, self.manifest,
                                  runner=runner, results=self.fixture.results)
                self.assertEqual(runner.call_count, 1)
                after = accounting.replay(rows, self.authority, self.manifest)
                fresh = self.fixture.batch(("work_0009",))
                submit.admit_batch(after, self.manifest, fresh, results=self.fixture.results)
                self.assertIn("46880582_12", after["superseded_pre_admission_jobs"])
        self.assertEqual((self.authority, self.manifest), originals)


if __name__ == "__main__":
    unittest.main()
