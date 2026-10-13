"""Incident admission fixtures; run only in the coordinator's Slurm allocation.

No real journal, status file, scheduler or scientific payload is opened.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_remainder_bundle as bundle
import reviewed_deadline_partials as partials
import reviewed_io_quarantine as quarantine
import submit_full_decoder_jobs as submit
import test_decoder_remainder_supplement as fixtures


class ReviewedIOQuarantineTests(unittest.TestCase):
    def setUp(self):
        self.source = fixtures.DecoderRemainderSupplementTests()
        self.source.setUp()
        self.authority, self.manifest = self.source.authority, self.source.manifest
        self.base, self.fixture = self.source.base, self.source.fixture
        self.rows = copy.deepcopy(self.source.rows)
        for number, parent, selected in (
                (6, "46918155", ["work_0061"]),
                (7, "46918887", ["work_0032", "work_0033", "work_0047", "work_0062"])):
            batch = submit.build_batch(self.authority, self.manifest, "workers", selected, "normal")
            self.rows.extend([self.source.event(batch, number),
                              self.base.accepted(parent, f"batch_{number:04d}")])
        for work, (job, raw, elapsed) in quarantine.KNOWN.items():
            self.rows.append(self.base.terminal(job, elapsed=elapsed, state="FAILED", allocation_job_id=raw))
        self.review = {**quarantine.FIXED, "reviewed_utc": "2026-10-07T23:54:35Z",
            "evidence": "Artificial fixture: reviewed Oak I/O incident", "failures": []}
        for work, (job, raw, elapsed) in quarantine.KNOWN.items():
            entry = {"work_item_id": work, "job_id": job, "allocation_job_id": raw,
                "elapsed_seconds": elapsed,
                "status_path": f"{quarantine.ROOT}/shards/{work}/work_item_status.json"}
            self.review["failures"].append(entry)
            item = self.manifest["work_items"][int(work[-4:])-1]
            record = copy.deepcopy(self.source.records["work_0013"])
            record.update(work_item_id=work, participant=item["participant"], resources=item["resources"],
                requested_spans=item["spans"], units_requested=item["units_requested"],
                error_type="OSError", error="[Errno 5] Input/output error",
                shared_state_reference=str(submit.state.state_directory(item["participant"])
                                           / "shared_training_state.npz"),
                closeout_errors=[] if work == "work_0062" else [
                    {"stage": stage, "affects_fragment": True, "error_type": "OSError",
                     "error": "[Errno 5] Input/output error"}
                    for stage in ("fragment_fsync", "fragment_close", "fragment_accounting")])
            record["allocation"].update(job_id=raw, hostname="sh04-04n21")
            if work in quarantine.UNCERTAIN:
                channel, start, stop = quarantine.UNCERTAIN[work]
                record["uncertain_publication"] = dict(participant=item["participant"], layer="conv1_1",
                    channel=channel, spatial_Fortran_interval=[start, stop],
                    phase="completion_marker_fsync", error_type="OSError")
            self.fixture.documents[submit.BASE / entry["status_path"]] = record
        self.account = self.replay(self.rows)

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    def batch(self, selected=("work_0015",)):
        batch = submit.build_batch(self.authority, self.manifest, "workers", selected, "normal",
            manual_refill=True, deadline_partial_review=copy.deepcopy(self.source.review2))
        batch[quarantine.FIELD] = copy.deepcopy(self.review)
        return batch

    def admit(self, batch, account=None):
        with self.fixture.metadata():
            return submit.admit_batch(account or self.account, self.manifest, batch, results=self.fixture.results)

    def validate_record(self, entry, record):
        item = self.manifest["work_items"][int(entry["work_item_id"][-4:])-1]
        quarantine.validate_closed_record(entry, record, item,
            authorization_path=str(submit.state.FULL_AUTH), manifest_path=str(submit.contract.MANIFEST),
            shared_state_path=str(submit.state.state_directory(item["participant"]) / "shared_training_state.npz"))

    def test_fresh_work_keeps_all_failed_charges_assignments_storage_and_never_probes_fragments(self):
        before = copy.deepcopy(self.account)
        def is_file(path):
            if path.name == "channels.h5" and path.parent.name in quarantine.KNOWN:
                self.fail("Quarantine must not inspect any failed fragment")
            return True
        # metadata() already mocks is_file. A plain replacement function keeps
        # Path method binding without attempting to autospec that outer mock.
        with self.fixture.metadata(), mock.patch.object(Path, "is_file", new=is_file):
            submit.admit_batch(self.account, self.manifest, self.batch(), results=self.fixture.results)
        self.assertEqual(self.account, before)
        updated = self.replay(self.rows + [self.source.event(self.batch(), 8),
                                           self.base.accepted("950", "batch_0008")])
        for work, (job, _, _) in quarantine.KNOWN.items():
            self.assertEqual(updated["jobs"][job], self.account["jobs"][job])
            self.assertIn(work, updated["assigned"])
            self.assertNotIn(job, updated["superseded_pre_admission_jobs"])
        self.assertEqual(updated["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        self.assertEqual(updated["outstanding_reserved_CPU_seconds"],
                         self.account["outstanding_reserved_CPU_seconds"] + 28800)
        self.assertEqual(updated["basis"], self.account["basis"])
        self.assertEqual(updated["additional_remainder_durable_bytes"],
                         self.account["additional_remainder_durable_bytes"])

    def test_existing_bundle_still_validates_its_own_clean_predecessors_and_one_reservation(self):
        batch = self.source.grouped()
        batch[quarantine.FIELD] = copy.deepcopy(self.review)
        self.admit(batch)
        updated = self.replay(self.rows + [self.source.event(batch, 8), self.base.accepted("950", "batch_0008")])
        self.assertEqual(updated["outstanding_reserved_CPU_seconds"],
                         self.account["outstanding_reserved_CPU_seconds"] + 7200)
        self.assertEqual(updated["additional_remainder_durable_bytes"], 2546718336)
        self.assertEqual(bundle.work_ids_for_job(updated["jobs"]["950_5"], updated["batches"]), bundle.MEMBERS)
        path = submit.BASE / partials.ROOT / "shards/work_0006/work_item_status.json"
        self.fixture.documents[path]["closeout_errors"] = [{"affects_fragment": True}]
        with self.assertRaisesRegex(partials.ReviewError, "uncertain_or_unfinalized"):
            self.admit(batch)

    def test_exact_five_identities_terminal_charges_and_fixed_permissions_are_required(self):
        for field, value in (("allocation_job_id", "999"), ("elapsed_seconds", 12690),
                             ("status_path", "elsewhere"), ("job_id", "999_31")):
            review = copy.deepcopy(self.review)
            review["failures"][0][field] = value
            with self.subTest(field=field), self.assertRaises(quarantine.QuarantineError):
                quarantine.validate_review(review, self.account["jobs"], self.manifest)
        for field, value in (("allow_fragment_reuse", True), ("validated_coverage_credit", 1),
                             ("release_durable_reservation", True), ("allow_failed_work_retry", True)):
            with self.subTest(field=field), self.assertRaises(quarantine.QuarantineError):
                quarantine.validate_review({**self.review, field: value}, self.account["jobs"], self.manifest)
        for terminal in (None, {**self.account["jobs"]["46918887_31"]["terminal"], "state": "COMPLETED"}):
            jobs = copy.deepcopy(self.account["jobs"])
            jobs["46918887_31"]["terminal"] = terminal
            with self.assertRaisesRegex(quarantine.QuarantineError, "exact_existing_failed_terminal"):
                quarantine.validate_review(self.review, jobs, self.manifest)

    def test_closed_status_must_preserve_io_diagnosis_identity_and_data_scope(self):
        for entry in self.review["failures"]:
            record = self.fixture.documents[submit.BASE / entry["status_path"]]
            self.validate_record(entry, record)
            for change in ({"error": "nonfinite_solve"}, {"error_type": "Stop"},
                           {"status": "complete_work_item_only"}, {"tau_test_values_read": True},
                           {"authorization": "different"}, {"requested_spans": ["different"]},
                           {"allocation": {**record["allocation"], "memory_gib": 8}}):
                with self.subTest(work=entry["work_item_id"], change=change), self.assertRaises(quarantine.QuarantineError):
                    self.validate_record(entry, {**record, **change})
        missing = submit.BASE / self.review["failures"][0]["status_path"]
        with self.fixture.metadata(), mock.patch.object(Path, "is_file",
                new=lambda path: path != missing), self.assertRaisesRegex(submit.Stop, "quarantine_closed_status"):
            submit.admit_batch(self.account, self.manifest, self.batch(), results=self.fixture.results)

    def test_default_unknown_failures_and_quarantined_retries_still_stop(self):
        batch = self.batch()
        del batch[quarantine.FIELD]
        with self.assertRaisesRegex(submit.Stop, "failed_parent"):
            self.admit(batch)
        for work in quarantine.KNOWN:
            with self.subTest(work=work), self.assertRaises(quarantine.QuarantineError):
                self.admit(self.batch((work,)))
        account = self.replay(self.rows + [self.base.terminal("800_8", elapsed=12,
                                        state="FAILED", allocation_job_id="810")])
        with self.assertRaisesRegex(submit.Stop, "failed_parent"):
            self.admit(self.batch(), account)

    def test_published_quarantine_cannot_change_narrow_or_disappear(self):
        history = self.rows + [self.source.event(self.batch(), 8), self.base.accepted("950", "batch_0008")]
        next_batch = self.batch(("work_0019",))
        self.replay(history + [self.source.event(next_batch, 9)])
        cases = []
        changed = copy.deepcopy(next_batch)
        del changed[quarantine.FIELD]
        cases.append(changed)
        changed = copy.deepcopy(next_batch)
        changed[quarantine.FIELD]["failures"].pop()
        cases.append(changed)
        changed = copy.deepcopy(next_batch)
        changed[quarantine.FIELD]["reviewed_utc"] = "2026-10-08T01:00:00Z"
        cases.append(changed)
        for changed in cases:
            with self.assertRaisesRegex(quarantine.QuarantineError, "cannot_change_narrow_or_disappear"):
                self.replay(history + [self.source.event(changed, 9)])

    def test_no_assembly_replacement_or_arbitrary_remainder_exception(self):
        cases = []
        for phase in ("states", "assembly"):
            batch = submit.build_batch(self.authority, self.manifest, phase, (), "normal")
            batch[quarantine.FIELD] = self.review
            cases.append(batch)
        for field in (accounting.REPLACEMENT_FIELD, accounting.STARTUP_REPLACEMENT_FIELD):
            cases.append({**self.batch(), field: {}})
        ordinary_remainder = self.source.old_batch()
        ordinary_remainder[quarantine.FIELD] = self.review
        cases.append(ordinary_remainder)
        for batch in cases:
            with self.subTest(fields=tuple(batch)), self.assertRaises((quarantine.QuarantineError, bundle.BundleError)):
                self.admit(batch)

    def test_cpu_global_slots_and_unresolved_submission_gates_are_unchanged(self):
        tight = copy.deepcopy(self.account)
        tight["budget"]["cpu_seconds"] = tight["actual_CPU_seconds"] + tight["outstanding_reserved_CPU_seconds"] + 28799
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            self.admit(self.batch(), tight)
        batch = self.batch(tuple(f"work_{i:04d}" for i in (15, 19, 20, 21, 22, 23, 24)))
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            self.admit(batch)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            self.replay(self.rows + [self.source.event(batch, 8)])
        for field, value in (("uncertain_batches", [{"batch_id": "unknown"}]), ("rejected_batch", True)):
            with self.subTest(field=field), self.assertRaises(accounting.AccountingError):
                self.admit(self.batch(), {**self.account, field: value})


if __name__ == "__main__":
    unittest.main()
