"""Diagnosed pre-admission repair checks; mocked records only, run in Slurm.

No actual authority, manifest, scheduler, scientific payload or result is read.
"""
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import submit_full_decoder_jobs as submit
import test_submit_full_decoder_jobs as fixtures


class ReviewedReplacementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FullDecoderSubmissionTests()
        self.authority, self.manifest = self.fixture.plans(count=75, layer=True)
        self.rows = self.fixture.rows(self.authority, self.manifest) + [
            self.fixture.terminal("99", elapsed=19669),
            self.fixture.intent(self.authority, self.manifest), self.fixture.accepted(),
            self.fixture.terminal(elapsed=1000),
            self.fixture.intent(self.authority, self.manifest, "workers",
                                [f"work_{index:04d}" for index in range(1, 9)], number=2),
            self.fixture.accepted("46839670", "batch_0002"),
            self.fixture.terminal("46839670_7", 15, "FAILED", "46839670")]
        self.review = {"schema_version": "ep21.reviewed_pre_admission_replacement.v1",
            "reviewed_utc": "2026-10-06T23:00:00Z", "reviewer": "coordinating_agent",
            "predecessor_job_id": "46839670_7", "predecessor_allocation_job_id": "46839670",
            "work_item_id": "work_0008", "array_index": 7,
            "failure_reason": "actual_work_item_allocation_mismatch", "shard_directory_absent": True,
            "request_or_status_created": False, "scientific_payload_admitted": False,
            "automatic_retry": False, "fix_evidence": "fixture: exact array-task selector corrected",
            "terminal_evidence": "fixture: FAILED 2:0 after 15 seconds",
            "absence_evidence": "fixture: work_0008 directory absent before admission",
            "user_authorization_context": "fixture: resume/fix within unchanged resource grant"}

    def account(self, extra=()):
        return accounting.replay([*self.rows, *extra], self.authority, self.manifest)

    def batch(self):
        return submit.build_reviewed_replacement(self.authority, self.manifest, self.review, "normal")

    def intent(self):
        return {"event": "submission_intent", "batch_id": "batch_0003", **self.batch()}

    def accepted(self):
        return self.fixture.accepted("50000000", "batch_0003")

    def test_same_assignment_extra_attempt_preserves_failure_and_full_reservations(self):
        original = copy.deepcopy(self.rows)
        batch = self.batch()
        self.assertIn("--array=7%1", batch["command"])
        self.assertEqual(batch["command"][-1], str(submit.REPLACEMENT_WRAPPER))
        self.assertIn("--no-requeue", batch["command"])
        submit.admit_batch(self.account(), self.manifest, batch, results=lambda *_: None)
        pending = self.account([self.intent()])
        self.assertFalse(pending["superseded_pre_admission_jobs"])
        self.assertEqual(pending["actual_CPU_seconds"] + pending["outstanding_reserved_CPU_seconds"], 251084)
        account = self.account([self.intent(), self.accepted()])
        self.assertEqual(account["actual_CPU_seconds"], 20684)
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 8 * 28800)
        self.assertEqual(account["assigned"], self.account()["assigned"])
        self.assertEqual(len(account["basis"]["manifest"]["work_items"]), 75)
        self.assertEqual(account["jobs"]["46839670_7"]["terminal"], self.rows[-1])
        self.assertEqual(account["jobs"]["50000000_7"]["logical_id"], "work_0008")
        self.assertEqual(account["superseded_pre_admission_jobs"], {"46839670_7"})
        self.assertEqual(submit.summary(account)["reviewed_extra_engineering_attempts"], 1)
        self.assertEqual(self.rows, original)

    def test_review_is_limited_to_exact_diagnosed_unstarted_case(self):
        changes = {"predecessor_job_id": "46839670_6", "array_index": 6,
                   "work_item_id": "work_0007", "failure_reason": "another_failure",
                   "shard_directory_absent": False, "request_or_status_created": True,
                   "scientific_payload_admitted": True, "automatic_retry": True,
                   "fix_evidence": "", "reviewed_utc": "2026-10-06T23:00:00"}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(accounting.AccountingError):
                accounting.validate_replacement_review({**self.review, key: value})
        for key, value in (("elapsed_seconds", 16), ("state", "COMPLETED"),
                           ("allocation_job_id", "999"), ("exit_code", "1:0")):
            rows = copy.deepcopy(self.rows)
            rows[-1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(accounting.AccountingError, "15_second_failure"):
                accounting.replay([*rows, self.intent()], self.authority, self.manifest)
        with self.assertRaisesRegex(accounting.AccountingError, "15_second_failure"):
            accounting.replay([*self.rows[:-1], self.intent()], self.authority, self.manifest)

    def test_manual_refill_counts_replacement_and_only_exempts_superseded_pre_admission_failure(self):
        extra = [self.intent(), self.accepted(),
                 self.fixture.terminal("46839670_0", 100, allocation_job_id="46840000")]
        account = self.account(extra)
        refill = submit.build_batch(self.authority, self.manifest, "workers", ["work_0009"],
                                    "normal", manual_refill=True)
        reads = []
        submit.admit_batch(account, self.manifest, refill,
                results=lambda a, m: self.fixture.check_closed_fixture_results(a, m, read_jobs=reads))
        self.assertEqual(reads, ["101", "46839670_0"])
        expected = {f"46839670_{i}" for i in range(1, 7)} | {"50000000_7"}
        queue = "".join(f"{job}|ep21-full-workers-001|RUNNING\n" for job in sorted(expected))
        submit.require_manual_refill_queue(account, self.authority, refill,
                                          mock.Mock(return_value=SimpleNamespace(stdout=queue)))
        events = [*extra, {"event": "submission_intent", "batch_id": "batch_0004", **refill},
                  self.fixture.accepted("60000000", "batch_0004")]
        updated = self.account(events)
        self.assertEqual(updated["superseded_pre_admission_jobs"], {"46839670_7"})
        self.assertEqual(updated["actual_CPU_seconds"], 20784)
        self.assertEqual(updated["outstanding_reserved_CPU_seconds"], 8 * 28800)
        self.assertEqual(updated["jobs"]["60000000_8"]["logical_id"], "work_0009")
        mixed = {**self.batch(), accounting.MANUAL_REFILL_FIELD: True}
        with self.assertRaisesRegex(accounting.AccountingError, "cannot_use_manual_refill"):
            submit.admit_batch(self.account(), self.manifest, mixed, results=lambda *_: None)

    def test_duplicate_repeat_chaining_and_assignment_changes_are_refused(self):
        ordinary = self.fixture.intent(self.authority, self.manifest, "workers", ["work_0008"], number=3)
        with self.assertRaisesRegex(accounting.AccountingError, "submitted_twice"):
            self.account([ordinary])
        repeated = {**self.intent(), "batch_id": "batch_0004"}
        with self.assertRaisesRegex(accounting.AccountingError, "one_manual_replacement"):
            self.account([self.intent(), self.accepted(), repeated])
        changed = self.intent()
        changed["allocations"] = accounting.allocations_for(self.authority, self.manifest, "workers", ["work_0007"])
        with self.assertRaisesRegex(accounting.AccountingError, "single_work_0008"):
            self.account([changed])
        mutated = copy.deepcopy(self.manifest)
        mutated["work_items"][7]["units_requested"] += 1
        with self.assertRaisesRegex(accounting.AccountingError, "manifest_is_immutable"):
            accounting.replay(self.rows, self.authority, mutated)

    def test_new_attempt_must_fit_budget_and_only_known_workers_can_remain(self):
        account = self.account()
        account["actual_CPU_seconds"] = 1800000 - account["outstanding_reserved_CPU_seconds"] - 28800
        submit.admit_batch(account, self.manifest, self.batch(), results=lambda *_: None)
        account["actual_CPU_seconds"] += 1
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            submit.admit_batch(account, self.manifest, self.batch(), results=lambda *_: None)
        account = self.account()
        account["jobs"]["101"]["terminal"] = None
        with self.assertRaisesRegex(submit.Stop, "only_original_other_workers"):
            submit.admit_batch(account, self.manifest, self.batch(), results=lambda *_: None)

    def test_queue_allows_only_original_seven_and_never_queries_twice(self):
        good = "".join(f"46839670_{i}|ep21-full-workers-001|RUNNING\n" for i in range(7))
        runner = mock.Mock(return_value=SimpleNamespace(stdout=good + "42|unrelated|RUNNING\n"))
        submit.require_reviewed_replacement_queue(self.account(), self.authority, runner)
        self.assertEqual(runner.call_count, 1)
        self.assertIn("--array", runner.call_args.args[0])
        for extra in ("46839670_7|ep21-full-workers-001|RUNNING\n",
                      "99|ep21-layer-001|RUNNING\n", "50000000_7|ep21-full-workers-001|PENDING\n",
                      "46839670_0|ep21-full-workers-001|RUNNING\n"):
            with self.subTest(extra=extra), self.assertRaisesRegex(submit.Stop, "unreviewed_or_duplicate"):
                submit.require_reviewed_replacement_queue(self.account(), self.authority,
                    mock.Mock(return_value=SimpleNamespace(stdout=good + extra)))
        smaller = copy.deepcopy(self.authority)
        smaller["resources"]["maximum_concurrent_jobs"] = 7
        with self.assertRaisesRegex(submit.Stop, "writer_envelope"):
            submit.require_reviewed_replacement_queue(self.account(), smaller, runner)
        self.assertEqual(runner.call_count, 1)

    def test_uncertain_replacement_keeps_full_reservation_and_never_retries(self):
        rows = copy.deepcopy(self.rows)
        runner = mock.Mock(side_effect=subprocess.TimeoutExpired("sbatch", 30))
        with self.assertRaisesRegex(submit.Stop, "uncertain"):
            submit.submit_one(self.batch(), rows, rows.append, self.authority, self.manifest,
                              runner=runner, results=lambda *_: None)
        self.assertEqual(runner.call_count, 1)
        account = accounting.replay(rows, self.authority, self.manifest)
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 8 * 28800)
        self.assertFalse(account["superseded_pre_admission_jobs"])
        with self.assertRaisesRegex(accounting.AccountingError, "uncertain_submission"):
            submit.admit_batch(account, self.manifest, self.batch(), results=lambda *_: None)

    def test_successful_submission_records_review_before_single_scheduler_call(self):
        rows = copy.deepcopy(self.rows)
        def scheduler(command, **kwargs):
            self.assertEqual(rows[-1]["event"], "submission_intent")
            self.assertEqual(rows[-1][accounting.REPLACEMENT_FIELD], self.review)
            self.assertIn("--array=7%1", command)
            return SimpleNamespace(returncode=0, stdout="50000000\n", stderr="")
        runner = mock.Mock(side_effect=scheduler)
        submitted = submit.submit_one(self.batch(), rows, rows.append, self.authority, self.manifest,
                                      runner=runner, results=lambda *_: None)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(submitted["job_id"], "50000000")
        self.assertFalse(submitted["automatic_successor"])
        self.assertEqual(rows[-1]["event"], "submission_accepted")

    def test_existing_shard_stops_manual_action_before_queue_or_submission(self):
        args = SimpleNamespace(action="replace-unstarted-worker", authorization="fixture",
                               manifest=str(submit.contract.MANIFEST), review=str(submit.REPLACEMENT_REVIEW),
                               partitions="normal")
        locked = mock.MagicMock()
        locked.__enter__.return_value = (self.rows, mock.Mock())
        occupied = submit.contract.ROOT / "shards/work_0008"
        with mock.patch.object(Path, "cwd", return_value=submit.BASE), \
                mock.patch.object(submit.socket, "gethostname", return_value="fixture-ln01"), \
                mock.patch.dict(submit.os.environ, {}, clear=True), \
                mock.patch.object(submit.state, "authorize_full", return_value=self.authority), \
                mock.patch.object(submit.contract, "validate_work_manifest"), \
                mock.patch.object(submit.contract.prefix, "read_json", side_effect=[self.manifest, self.review]), \
                mock.patch.object(accounting, "locked_events", return_value=locked), \
                mock.patch.object(Path, "is_dir", return_value=True), \
                mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "is_symlink", return_value=False), \
                mock.patch.object(Path, "exists", autospec=True, side_effect=lambda path: path == occupied), \
                mock.patch.object(submit, "admit_batch"), \
                mock.patch.object(submit.subprocess, "run") as scheduler, \
                mock.patch.object(submit, "submit_one") as dispatch:
            with self.assertRaisesRegex(submit.Stop, "attempt_outputs_must_be_fresh"):
                submit.run(args)
            scheduler.assert_not_called()
            dispatch.assert_not_called()
            locked.__enter__.return_value[1].assert_not_called()

    def test_future_readiness_requires_exact_successful_replacement_result(self):
        completed = [self.fixture.terminal(f"46839670_{i}", 100, allocation_job_id=str(46840000 + i))
                     for i in range(7)]
        extra = [self.intent(), self.accepted(), *completed]
        account = self.account(extra)
        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "is_symlink", return_value=False):
            def reader(path):
                if path.name == "state_parent_status.json":
                    return {"status": "complete_four_participant_training_states_only",
                            "allocation": {"job_id": "101"},
                            "participants_completed": list(submit.state.NEW_PARTICIPANTS), "participants_remaining": []}
                work = path.parent.name
                index = int(work.split("_")[1]) - 1
                item = self.manifest["work_items"][index]
                return {"status": "complete_work_item_only", "work_item_id": work,
                        "allocation": {"job_id": "50000000" if index == 7 else str(46840000 + index)},
                        "participant": item["participant"], "resources": item["resources"],
                        "requested_spans": item["spans"], "committed_spans": item["spans"],
                        "units_requested": 1, "completed_units": 1, "remaining_spans": [],
                        "completed_units_is_lower_bound": False, "closeout_errors": []}
            with self.assertRaisesRegex(submit.Stop, "terminal_accounting_missing"):
                submit.require_previous_results(account, self.manifest, reader=reader)
            complete = self.fixture.terminal("50000000_7", 100, allocation_job_id="50000000")
            account = self.account([*extra, complete])
            submit.require_previous_results(account, self.manifest, reader=reader)
            calls = []
            def wrong_reader(path):
                record = reader(path)
                calls.append(path)
                if path.parent.name == "work_0008":
                    record["allocation"]["job_id"] = "46839670"
                return record
            with self.assertRaisesRegex(submit.Stop, "allocation_mismatch"):
                submit.require_previous_results(account, self.manifest, reader=wrong_reader)
            self.assertEqual(sum(path.parent.name == "work_0008" for path in calls), 1)
            failed = {**complete, "state": "FAILED", "exit_code": "2:0"}
            with self.assertRaisesRegex(submit.Stop, "failed_parent"):
                submit.require_previous_results(self.account([*extra, failed]), self.manifest, reader=reader)


if __name__ == "__main__":
    unittest.main()
