"""Finite source-free fixtures for reviewed partials, never real job/data I/O."""
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
import reviewed_deadline_partials as partials
import submit_full_decoder_jobs as submit
import test_submit_full_decoder_jobs as fixtures


class ReviewedDeadlinePartialTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FullDecoderSubmissionTests()
        self.authority, self.manifest = self.fixture.plans(count=16)
        for index, item in enumerate(self.manifest["work_items"]):
            item["spans"] = [dict(participant="sub-01", layer="conv1_1", channel=2*index+j,
                                  spatial_Fortran_interval=[0, 64]) for j in range(2)]
            item["units_requested"] = 128
        self.rows = self.fixture.refill_history(self.authority, self.manifest, closed=4)
        for row in self.rows:
            if row.get("event") == "terminal_charge" and row["job_id"] in {"200_1", "200_2", "200_3"}:
                row.update(state="FAILED", exit_code="2:0", elapsed_seconds=28503)
        self.account = accounting.replay(self.rows, self.authority, self.manifest)
        self.review = {
            "schema_version": "ep21.reviewed_deadline_partials.v1", "reviewer": "coordinating_agent",
            "reviewed_utc": "2026-10-07T08:51:10Z", "cause": "ordinary_work_item_deadline",
            "automatic_retry": False, "allows_only": "fresh_unassigned_manual_workers", "partials": [],
        }
        self.records = {}
        for job in self.account["jobs"].values():
            if job["terminal"] is None:
                continue
            work = job["logical_id"]
            if work == "states":
                self.records[work] = dict(status="complete_four_participant_training_states_only",
                    allocation={"job_id": "101"}, participants_completed=list(submit.state.NEW_PARTICIPANTS),
                    participants_remaining=[])
                continue
            item = next(item for item in self.manifest["work_items"] if item["work_item_id"] == work)
            record = dict(schema_version="ep21.decoder_work_item.v1", status="complete_work_item_only",
                work_item_id=work, participant="sub-01", resources=copy.deepcopy(item["resources"]),
                allocation=dict(job_id=job["terminal"]["allocation_job_id"], hostname="sh04-04n23",
                    allocated_cpus=1, memory_gib=7, gpus=0, time_limit_seconds=28800, job_end_epoch=1791363222),
                authorization=str(submit.state.FULL_AUTH), work_manifest=str(submit.contract.MANIFEST),
                shared_state_reference=str(submit.state.state_directory("sub-01") / "shared_training_state.npz"),
                fragment="channels.h5", training_role="subject_unique", training_images=4712,
                original_neural_values_read=False, retained_neural_state_read=True,
                original_budget_consumption_preserved=True, predictions_or_scores_computed=False,
                tau_train_values_read=False, tau_test_values_read=False, OOD_access=False,
                complete_decoder=False, successor_authorized=False, automatic_retry=False,
                requested_spans=copy.deepcopy(item["spans"]), committed_spans=copy.deepcopy(item["spans"]),
                remaining_spans=[], units_requested=128, completed_units=128,
                remaining_interval_status="exact_uncommitted_coverage", completed_units_is_lower_bound=False,
                closeout_errors=[])
            if job["terminal"]["state"] == "FAILED":
                record.update(status="partial_work_item_stopped", error_type="Stop",
                              error="work_item_deadline_reached", completed_units=96)
                record["committed_spans"][1]["spatial_Fortran_interval"] = [0, 32]
                remainder = copy.deepcopy(item["spans"][1])
                remainder["spatial_Fortran_interval"] = [32, 64]
                record["remaining_spans"] = [remainder]
                self.review["partials"].append(dict(work_item_id=work, job_id=job["job_id"],
                    allocation_job_id=job["terminal"]["allocation_job_id"], elapsed_seconds=28503,
                    completed_units=96, status_path=f"{partials.ROOT}/shards/{work}/work_item_status.json"))
            self.records[work] = record

    def batch(self, selected=("work_0009",)):
        return submit.build_batch(self.authority, self.manifest, "workers", selected, "normal",
            manual_refill=True, constraint="CPU_GEN:SIE", deadline_partial_review=copy.deepcopy(self.review))

    def results(self, account, manifest, **kwargs):
        def reader(path):
            work = "states" if path.name == "state_parent_status.json" else path.parent.name
            return copy.deepcopy(self.records[work])
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(Path, "is_symlink", return_value=False):
            return submit.require_previous_results(account, manifest, reader=reader, **kwargs)

    def validate_record(self, record):
        entry = self.review["partials"][0]
        item = self.manifest["work_items"][1]
        return partials.validate_closed_record(entry, record, item,
            authorization_path=str(submit.state.FULL_AUTH), manifest_path=str(submit.contract.MANIFEST),
            shared_state_path=str(submit.state.state_directory("sub-01") / "shared_training_state.npz"))

    def test_review_allows_only_fresh_manual_workers_preserving_incomplete_assignments(self):
        before = copy.deepcopy(self.account)
        submit.admit_batch(self.account, self.manifest, self.batch(), results=self.results)
        self.assertEqual(self.account, before)
        self.assertIn("work_0002", self.account["assigned"])
        self.assertEqual(self.records["work_0002"]["status"], "partial_work_item_stopped")
        for manual in (False, True):
            default = submit.build_batch(self.authority, self.manifest, "workers", ["work_0009"],
                                         "normal", manual_refill=manual)
            with self.assertRaisesRegex(submit.Stop, "failed_parent"):
                submit.admit_batch(self.account, self.manifest, default, results=self.results)
        with self.assertRaisesRegex(partials.ReviewError, "unselected"):
            submit.admit_batch(self.account, self.manifest, self.batch(("work_0002",)), results=self.results)
        assembly = submit.build_batch(self.authority, self.manifest, "assembly", (), "normal")
        with self.assertRaises(submit.Stop):
            submit.admit_batch(self.account, self.manifest, assembly, results=self.results)
        assembly[partials.FIELD] = self.review
        with self.assertRaisesRegex(partials.ReviewError, "fresh_manual"):
            submit.admit_batch(self.account, self.manifest, assembly, results=self.results)

    def test_exact_cover_rejects_gap_overlap_foreign_duplicate_or_reordered_spans(self):
        valid = self.records["work_0002"]
        self.validate_record(valid)
        changes = []
        for start in (31, 33):
            record = copy.deepcopy(valid)
            record["remaining_spans"][0]["spatial_Fortran_interval"][0] = start
            changes.append(record)
        record = copy.deepcopy(valid)
        record["remaining_spans"][0]["channel"] = 500
        changes.append(record)
        record = copy.deepcopy(valid)
        record["committed_spans"].append(copy.deepcopy(record["committed_spans"][0]))
        changes.append(record)
        record = copy.deepcopy(valid)
        record["committed_spans"].reverse()
        changes.append(record)
        record = copy.deepcopy(valid)
        record["committed_spans"][0]["spatial_Fortran_interval"][0] = 1
        changes.append(record)
        for record in changes:
            with self.subTest(record=record["committed_spans"]), self.assertRaises(partials.ReviewError):
                self.validate_record(record)

    def test_failure_uncertainty_scope_and_resource_mismatches_still_stop(self):
        valid = self.records["work_0002"]
        changes = ({"error": "nonfinite_solve"}, {"error_type": "RuntimeError"},
            {"status": "complete_work_item_only"}, {"completed_units": 95},
            {"completed_units_is_lower_bound": True}, {"uncertain_publication": {}},
            {"remaining_interval_status": "unknown"}, {"closeout_errors": [{"affects_fragment": True}]},
            {"closeout_errors": [{}]}, {"training_role": "tau_train"}, {"tau_train_values_read": True},
            {"complete_decoder": True}, {"authorization": "different"},
            {"shared_state_reference": "different"}, {"resources": {**valid["resources"], "cpus": 2}},
            {"allocation": {**valid["allocation"], "job_id": "999"}},
            {"allocation": {**valid["allocation"], "time_limit_seconds": 86400}})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(partials.ReviewError):
                self.validate_record({**copy.deepcopy(valid), **change})
        # An extra failed parent outside the explicit review remains blocking.
        changed = accounting.replay(self.rows + [self.fixture.terminal(
            "200_4", state="FAILED", allocation_job_id="304")], self.authority, self.manifest)
        with self.assertRaisesRegex(submit.Stop, "failed_parent"):
            submit.admit_batch(changed, self.manifest, self.batch(), results=self.results)

    def test_exact_terminal_review_and_closed_files_are_required(self):
        batch = self.batch()
        for field, value in (("allocation_job_id", "999"), ("elapsed_seconds", 1),
                             ("job_id", "999_1"), ("status_path", "elsewhere")):
            changed = copy.deepcopy(batch)
            changed[partials.FIELD]["partials"][0][field] = value
            with self.subTest(field=field), self.assertRaises(partials.ReviewError):
                submit.admit_batch(self.account, self.manifest, changed, results=self.results)
        reviewed = partials.validate_intent(batch, self.account["jobs"], self.manifest)
        closed = {key: job for key, job in self.account["jobs"].items() if job["terminal"] is not None}
        for missing_fragment in (False, True):
            with mock.patch.object(Path, "is_file", autospec=True, side_effect=lambda path: path.name != (
                    "channels.h5" if missing_fragment else "work_item_status.json")), \
                    mock.patch.object(Path, "is_symlink", return_value=False), \
                    self.assertRaisesRegex(submit.Stop, "closed_status_and_fragment"):
                submit.require_previous_results({**self.account, "jobs": {"200_1": closed["200_1"]}},
                    self.manifest, reviewed_partials=reviewed)

    def test_review_is_durable_before_sbatch_and_budget_and_global_slots_unchanged(self):
        originals = copy.deepcopy((self.authority, self.manifest))
        actual = self.account["actual_CPU_seconds"]
        batch = self.batch()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            with accounting.locked_events(path, create=True) as (rows, append):
                for row in self.rows:
                    append(row)
                def scheduler(command, **kwargs):
                    event = json.loads(path.read_text().splitlines()[-1])
                    self.assertEqual(event[partials.FIELD], self.review)
                    self.assertIn("--array=8%1", command)
                    self.assertEqual(accounting.replay(rows, self.authority, self.manifest)
                                     ["outstanding_reserved_CPU_seconds"], 5*28800)
                    return SimpleNamespace(returncode=0, stdout="500\n", stderr="")
                submit.submit_one(batch, rows, append, self.authority, self.manifest,
                                  runner=scheduler, results=self.results)
                account = accounting.replay(rows, self.authority, self.manifest)
                self.assertEqual(account["actual_CPU_seconds"], actual)
                self.assertEqual(account["jobs"]["200_1"]["terminal"]["state"], "FAILED")
                self.assertEqual(account["outstanding_reserved_CPU_seconds"], 5*28800)
                too_many = self.batch(("work_0010", "work_0011", "work_0012", "work_0013"))
                with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
                    submit.admit_batch(account, self.manifest, too_many, results=self.results)
        self.assertEqual((self.authority, self.manifest), originals)

    def test_replay_rejects_mode_bypass_or_rewriting_published_review(self):
        batch = self.batch()
        event = {"event": "submission_intent", "batch_id": "batch_0003", **batch}
        changed = copy.deepcopy(event)
        del changed[accounting.MANUAL_REFILL_FIELD]
        with self.assertRaisesRegex(partials.ReviewError, "fresh_manual"):
            accounting.replay(self.rows+[changed], self.authority, self.manifest)
        history = self.rows+[event, self.fixture.accepted("500", "batch_0003")]
        second = self.batch(("work_0010",))
        second[partials.FIELD]["reviewed_utc"] = "2026-10-07T08:52:00Z"
        with self.assertRaisesRegex(accounting.AccountingError, "review_is_immutable"):
            accounting.replay(history+[{"event": "submission_intent", "batch_id": "batch_0004", **second}],
                              self.authority, self.manifest)

    def appended_review_history(self):
        first = {"event": "submission_intent", "batch_id": "batch_0003", **self.batch()}
        history = self.rows + [first, self.fixture.accepted("500", "batch_0003"),
            self.fixture.terminal("200_4", state="FAILED", elapsed=28502, allocation_job_id="304")]
        second = {"event": "submission_intent", "batch_id": "batch_0004",
                  **self.batch(("work_0010",))}
        second[partials.FIELD]["reviewed_utc"] = "2026-10-07T09:00:00Z"
        second[partials.FIELD]["partials"].append(dict(work_item_id="work_0005", job_id="200_4",
            allocation_job_id="304", elapsed_seconds=28502, completed_units=96,
            status_path=f"{partials.ROOT}/shards/work_0005/work_item_status.json"))
        return history, second

    def test_replay_accepts_newer_append_preserving_original_entries_and_charges(self):
        history, second = self.appended_review_history()
        before = accounting.replay(history, self.authority, self.manifest)
        frozen = copy.deepcopy(history)
        after = accounting.replay(history + [second], self.authority, self.manifest)
        self.assertEqual(history, frozen)
        self.assertEqual(after["jobs"], before["jobs"])
        self.assertEqual(after["actual_CPU_seconds"], before["actual_CPU_seconds"])
        self.assertEqual(after["batches"]["batch_0003"][partials.FIELD], self.review)
        self.assertEqual(after["batches"]["batch_0004"][partials.FIELD], second[partials.FIELD])
        self.assertIn("work_0005", after["assigned"])

    def test_replay_rejects_partial_mutation_removal_reorder_and_nonnewer_append(self):
        history, second = self.appended_review_history()
        changes = []
        changed = copy.deepcopy(second)
        changed[partials.FIELD]["partials"][0]["completed_units"] -= 1
        changes.append(changed)
        changed = copy.deepcopy(second)
        del changed[partials.FIELD]["partials"][1]
        changes.append(changed)
        changed = copy.deepcopy(second)
        changed[partials.FIELD]["partials"][:2] = reversed(changed[partials.FIELD]["partials"][:2])
        changes.append(changed)
        for stamp in (self.review["reviewed_utc"], "2026-10-07T08:00:00Z"):
            changed = copy.deepcopy(second)
            changed[partials.FIELD]["reviewed_utc"] = stamp
            changes.append(changed)
        for changed in changes:
            with self.subTest(review=changed[partials.FIELD]), \
                    self.assertRaisesRegex(accounting.AccountingError, "review_is_immutable"):
                accounting.replay(history + [changed], self.authority, self.manifest)

    def test_replay_accepts_unchanged_latest_review_but_rejects_rollback(self):
        history, second = self.appended_review_history()
        # The original review can be repeated unchanged before any extension.
        duplicate = copy.deepcopy(second)
        duplicate[partials.FIELD] = copy.deepcopy(self.review)
        accounting.replay(history + [duplicate], self.authority, self.manifest)
        extended = history + [second, self.fixture.accepted("501", "batch_0004")]
        third = {"event": "submission_intent", "batch_id": "batch_0005",
                 **self.batch(("work_0011",))}
        third[partials.FIELD] = copy.deepcopy(second[partials.FIELD])
        accounting.replay(extended + [third], self.authority, self.manifest)
        third[partials.FIELD] = copy.deepcopy(self.review)
        with self.assertRaisesRegex(accounting.AccountingError, "review_is_immutable"):
            accounting.replay(extended + [third], self.authority, self.manifest)


if __name__ == "__main__":
    unittest.main()
