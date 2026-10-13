"""Nine exact before-fit replacement fixtures; coordinated compute only.

All histories and closed records are fabricated metadata. No actual scheduler,
journal, startup log, scientific output path or empirical payload is opened.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_deadline_remainders as remainder
import decoder_remainder_bundle as bundle
import reviewed_bundle_startup_replacement as startup
import reviewed_io_quarantine as quarantine
import submit_full_decoder_jobs as submit
import test_reviewed_io_quarantine as fixtures


class ReviewedBundleStartupReplacementTests(unittest.TestCase):
    def setUp(self):
        self.source = fixtures.ReviewedIOQuarantineTests()
        self.source.setUp()
        self.authority, self.manifest = self.source.authority, self.source.manifest
        self.fixture, self.base = self.source.fixture, self.source.base
        self.plan = self.source.source.plan2
        previous = self.source.source.grouped()
        previous[quarantine.FIELD] = copy.deepcopy(self.source.review)
        failed = self.base.terminal(startup.PREDECESSOR, elapsed=25,
                                   state="FAILED", allocation_job_id="46942785")
        failed["exit_code"] = "1:0"
        self.rows = self.source.rows + [self.source.source.event(previous, 8),
            self.base.accepted("46942785", "batch_0008"), failed]
        self.review = {**copy.deepcopy(startup.FIXED),
            "reviewed_utc": "2026-10-08T00:11:20Z",
            "fix_evidence": "Fixture: replace nested autospec with plain callable",
            "terminal_evidence": "Fixture: FAILED1:0 after25seconds at1CPU",
            "absence_evidence": "Fixture: both child paths absent before empirical access",
            "user_authorization_context": "Fixture: one manual technical replacement in existing grant"}

    def replay(self, rows=None):
        return accounting.replay(self.rows if rows is None else rows, self.authority, self.manifest)

    def batch(self):
        # The production builder retains its immutable plan; counterexamples must not.
        batch = copy.deepcopy(self.source.source.grouped())
        batch[quarantine.FIELD] = copy.deepcopy(self.source.review)
        batch[startup.FIELD] = copy.deepcopy(self.review)
        return batch

    def event(self, batch=None, number=9):
        return self.source.source.event(self.batch() if batch is None else batch, number)

    def admit(self, account=None, batch=None):
        with self.fixture.metadata():
            return submit.admit_batch(self.replay() if account is None else account,
                                      self.manifest, self.batch() if batch is None else batch,
                                      results=self.fixture.results)

    def accepted_rows(self):
        return self.rows + [self.event(), self.base.accepted("950", "batch_0009")]

    def test_exact_replacement_preserves_charge_storage_and_one_physical_reservation(self):
        before = self.replay()
        original = copy.deepcopy(before)
        self.admit(before)
        self.assertEqual(before, original)
        pending = self.replay(self.rows + [self.event()])
        self.assertNotIn(startup.PREDECESSOR, pending["superseded_pre_admission_jobs"])
        with self.assertRaisesRegex(remainder.partials.ReviewError, "must_be_outstanding"):
            remainder.registered_remainder_job(pending, self.plan, "work_0006")
        after = self.replay(self.accepted_rows())
        self.assertEqual(after["actual_CPU_seconds"], before["actual_CPU_seconds"])
        self.assertEqual(after["jobs"][startup.PREDECESSOR], before["jobs"][startup.PREDECESSOR])
        self.assertEqual(after["jobs"][startup.PREDECESSOR]["terminal"]["elapsed_seconds"], 25)
        self.assertEqual(after["assigned"], before["assigned"])
        self.assertEqual(after["additional_remainder_durable_bytes"], before["additional_remainder_durable_bytes"])
        self.assertEqual(after["outstanding_reserved_CPU_seconds"], before["outstanding_reserved_CPU_seconds"] + 7200)
        self.assertEqual(set(after["jobs"])-set(before["jobs"]), {"950_5"})
        self.assertIn(startup.PREDECESSOR, after["superseded_pre_admission_jobs"])
        self.assertEqual(submit.summary(after)["reviewed_extra_engineering_attempts"],
                         submit.summary(before)["reviewed_extra_engineering_attempts"] + 1)
        for work in bundle.MEMBERS:
            self.assertEqual(remainder.registered_remainder_job(after, self.plan, work)["job_id"], "950_5")
        self.assertEqual(after["batches"]["batch_0009"][quarantine.FIELD],
                         before["batches"]["batch_0008"][quarantine.FIELD])

    def test_only_exact_nonempirical_failure_and_unchanged_bundle_can_be_replaced(self):
        for field, value in (("predecessor_job_id", "46942785_6"), ("array_index", 12),
                             ("logical_work_items", ["work_0006"]),
                             ("scientific_payload_admitted", True),
                             ("scientific_output_directories_absent", False),
                             ("io_fragment_inspection_started", True),
                             ("failure_reason", "empirical_write_failure")):
            with self.subTest(field=field), self.assertRaises(startup.BundleStartupReplacementError):
                startup.validate_review({**self.review, field: value})
        for field, value in (("elapsed_seconds", 26), ("allocated_cpus", 0),
                             ("exit_code", "2:0"), ("allocation_job_id", "999"),
                             ("state", "COMPLETED")):
            account = copy.deepcopy(self.replay())
            account["jobs"][startup.PREDECESSOR]["terminal"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                    startup.BundleStartupReplacementError, "25_second_failure"):
                startup.validate_intent(self.batch(), account["jobs"], account["batches"])
        changed = self.batch()
        changed[bundle.PLAN_FIELD]["work_items"][-1]["output_directory"] += "_different"
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "preserve_plan_items"):
            account = self.replay()
            startup.validate_intent(changed, account["jobs"], account["batches"])
        changed = self.batch()
        changed[quarantine.FIELD]["evidence"] += " changed"
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "preserve_plan_items"):
            account = self.replay()
            startup.validate_intent(changed, account["jobs"], account["batches"])

    def test_missing_review_arbitrary_modes_and_repeated_or_chained_replacements_refuse(self):
        ordinary = self.batch()
        del ordinary[startup.FIELD]
        with self.assertRaises(remainder.partials.ReviewError):
            self.admit(batch=ordinary)
        fresh = self.source.batch(("work_0015",))
        fresh[startup.FIELD] = self.review
        with self.assertRaises(bundle.BundleError):
            self.admit(batch=fresh)
        changed = self.batch()
        changed["phase"] = "assembly"
        with self.assertRaises((quarantine.QuarantineError, bundle.BundleError)):
            self.admit(batch=changed)
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "only_no_chaining"):
            self.replay(self.accepted_rows() + [self.event(number=10)])
        pending = self.replay(self.rows + [self.event()])
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "only_no_chaining"):
            startup.validate_intent(self.batch(), pending["jobs"], pending["batches"])

    def test_completed_replacement_lookup_and_future_fresh_admission_require_both_real_children(self):
        closed = self.accepted_rows() + [self.base.terminal("950_5", elapsed=1900, allocation_job_id="951")]
        account = self.replay(closed)
        self.source.source.close_bundle_records(raw="951")
        fresh = self.source.batch(("work_0015",))
        self.admit(account, fresh)
        for work in bundle.MEMBERS:
            item = next(item for item in self.plan["work_items"] if item["work_item_id"] == work)
            registered = remainder.registered_remainder_job(account, self.plan, work, require_terminal=True)
            self.assertEqual(registered["job_id"], "950_5")
            record = self.fixture.documents[submit.BASE / item["output_directory"] / "work_item_status.json"]
            remainder.validate_completed_remainder(item, record, registered, base_directory=str(submit.BASE),
                remainder_plan_relative=remainder.PLAN_RELATIVE_002, registered_work_items=bundle.MEMBERS)
        self.assertEqual(account["actual_CPU_seconds"]-self.replay()["actual_CPU_seconds"], 1900)
        second = self.fixture.documents[submit.BASE / self.plan["work_items"][-1]["output_directory"] / "work_item_status.json"]
        second["allocation"]["job_id"] = "46942785"
        with self.assertRaises(remainder.partials.ReviewError):
            self.admit(account, fresh)

    def test_original_clean_predecessors_and_budget_unresolved_history_gates_remain(self):
        account = self.replay()
        tight = copy.deepcopy(account)
        tight["budget"]["cpu_seconds"] = tight["actual_CPU_seconds"] + tight["outstanding_reserved_CPU_seconds"] + 7199
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            self.admit(tight)
        for field, value in (("uncertain_batches", [{"batch_id": "unknown"}]), ("rejected_batch", True)):
            with self.subTest(field=field), self.assertRaises(accounting.AccountingError):
                self.admit({**account, field: value})
        changed = copy.deepcopy(account)
        changed["batches"]["batch_0008"]["accepted_id"] = None
        with self.assertRaises(startup.BundleStartupReplacementError):
            startup.validate_intent(self.batch(), changed["jobs"], changed["batches"])
        original = submit.BASE / remainder.partials.ROOT / "shards/work_0006/work_item_status.json"
        self.fixture.documents[original]["closeout_errors"] = [{"affects_fragment": True}]
        with self.assertRaisesRegex(remainder.partials.ReviewError, "uncertain_or_unfinalized"):
            self.admit()


    def second_failure_rows(self):
        terminal = self.base.terminal(startup.PREDECESSOR_002, elapsed=21,
                                     state="FAILED", allocation_job_id="46943996")
        terminal["exit_code"] = "1:0"
        return copy.deepcopy(self.rows + [self.event(),
            self.base.accepted("46943996", "batch_0009"), terminal])

    def second_batch(self):
        batch = self.batch()
        batch[startup.FIELD] = {**copy.deepcopy(startup.FIXED_002),
            **{key: self.review[key] for key in startup.TEXT_FIELDS},
            "reviewed_utc": "2026-10-08T00:32:00Z",
            "fix_evidence": "Fixture: deep-copy replay result before terminal mutation",
            "terminal_evidence": "Fixture: second FAILED1:0 after21seconds at1CPU"}
        return batch

    def test_second_review_preserves_both_charges_and_only_accepted_replacement_lookup(self):
        rows, batch = self.second_failure_rows(), self.second_batch()
        pristine = copy.deepcopy(rows)
        before = self.replay(rows)
        self.admit(before, batch)
        self.assertEqual(rows, pristine)
        pending = self.replay(rows + [self.event(batch, number=10)])
        self.assertIn(startup.PREDECESSOR, pending["superseded_pre_admission_jobs"])
        self.assertNotIn(startup.PREDECESSOR_002, pending["superseded_pre_admission_jobs"])
        accepted = rows + [self.event(batch, number=10), self.base.accepted("960", "batch_0010")]
        after = self.replay(accepted)
        for job, seconds in ((startup.PREDECESSOR, 25), (startup.PREDECESSOR_002, 21)):
            self.assertEqual(after["jobs"][job], before["jobs"][job])
            self.assertEqual(after["jobs"][job]["terminal"]["elapsed_seconds"], seconds)
            self.assertIn(job, after["superseded_pre_admission_jobs"])
        self.assertEqual(after["actual_CPU_seconds"], before["actual_CPU_seconds"])
        self.assertEqual(after["actual_CPU_seconds"] - self.replay()["actual_CPU_seconds"], 21)
        self.assertEqual(after["outstanding_reserved_CPU_seconds"],
                         before["outstanding_reserved_CPU_seconds"] + 7200)
        self.assertEqual(after["assigned"], before["assigned"])
        self.assertEqual(after["additional_remainder_durable_bytes"],
                         before["additional_remainder_durable_bytes"])
        self.assertEqual(after["batches"]["batch_0009"], before["batches"]["batch_0009"])
        for work in bundle.MEMBERS:
            self.assertEqual(remainder.registered_remainder_job(after, self.plan, work)["job_id"], "960_5")
        closed = self.replay(accepted + [self.base.terminal("960_5", elapsed=1900, allocation_job_id="961")])
        self.source.source.close_bundle_records(raw="961")
        self.admit(closed, self.source.batch(("work_0015",)))
        for work in bundle.MEMBERS:
            self.assertEqual(remainder.registered_remainder_job(
                closed, self.plan, work, require_terminal=True)["job_id"], "960_5")

    def test_second_review_requires_exact_closed_history_and_cannot_extend_itself(self):
        rows, batch = self.second_failure_rows(), self.second_batch()
        before = self.replay(rows)
        original_rows = copy.deepcopy(rows)
        self.assertEqual(startup.review_relative(self.review), startup.REVIEW_RELATIVE)
        self.assertEqual(startup.review_relative(batch[startup.FIELD]), startup.REVIEW_RELATIVE_002)
        for field, value in (("predecessor_job_id", "960_5"), ("scientific_payload_admitted", True),
                             ("previous_startup_review", "elsewhere"), ("automatic_retry", True)):
            changed = copy.deepcopy(batch[startup.FIELD])
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(startup.BundleStartupReplacementError):
                startup.validate_review(changed)
        for field, value in (("elapsed_seconds", 20), ("allocated_cpus", 0),
                             ("state", "COMPLETED"), ("exit_code", "2:0"),
                             ("allocation_job_id", "999")):
            changed = copy.deepcopy(before)
            changed["jobs"][startup.PREDECESSOR_002]["terminal"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                    startup.BundleStartupReplacementError, "21_second_failure"):
                startup.validate_intent(batch, changed["jobs"], changed["batches"])
        self.assertEqual(rows, original_rows)
        changed = copy.deepcopy(before)
        changed["jobs"][startup.PREDECESSOR]["terminal"]["elapsed_seconds"] = 24
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "original_25_second_charge"):
            startup.validate_intent(batch, changed["jobs"], changed["batches"])
        changed = copy.deepcopy(before)
        del changed["batches"]["batch_0009"][startup.FIELD]
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "no_further_chaining"):
            startup.validate_intent(batch, changed["jobs"], changed["batches"])
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "only_no_chaining"):
            startup.validate_intent(self.batch(), before["jobs"], before["batches"])
        for acceptance in ([], [self.base.accepted("960", "batch_0010")]):
            with self.subTest(accepted=bool(acceptance)), self.assertRaisesRegex(
                    startup.BundleStartupReplacementError, "no_further_chaining"):
                self.replay(rows + [self.event(batch, number=10)] + acceptance
                            + [self.event(batch, number=11)])

    def third_failure_rows(self):
        terminal = self.base.terminal(startup.PREDECESSOR_003, elapsed=625,
                                     state="FAILED", allocation_job_id="46945868")
        terminal["exit_code"] = "1:0"
        return copy.deepcopy(self.second_failure_rows() + [self.event(self.second_batch(), number=10),
            self.base.accepted("46945868", "batch_0010"), terminal])

    def third_batch(self):
        batch = self.batch()
        batch[startup.FIELD] = {**copy.deepcopy(startup.FIXED_003),
            **{key: self.review[key] for key in startup.TEXT_FIELDS},
            "reviewed_utc": "2026-10-08T00:49:00Z",
            "fix_evidence": "Fixture: independent read-only inspection no longer gates clean fits",
            "terminal_evidence": "Fixture: FAILED1:0 after625seconds; inspection timed out, fits absent"}
        batch["command"][-1] = str(submit.REMAINDER_BUNDLE_WRAPPER_003)
        return batch

    def test_third_review_preserves_three_charges_and_admits_only_unchanged_unstarted_fits(self):
        rows, batch = self.third_failure_rows(), self.third_batch()
        pristine = copy.deepcopy(rows)
        before = self.replay(rows)
        self.admit(before, batch)
        self.assertEqual(rows, pristine)
        pending = self.replay(rows + [self.event(batch, number=11)])
        self.assertNotIn(startup.PREDECESSOR_003, pending["superseded_pre_admission_jobs"])
        accepted = rows + [self.event(batch, number=11), self.base.accepted("970", "batch_0011")]
        after = self.replay(accepted)
        for job, seconds in ((startup.PREDECESSOR, 25), (startup.PREDECESSOR_002, 21),
                             (startup.PREDECESSOR_003, 625)):
            self.assertEqual(after["jobs"][job], before["jobs"][job])
            self.assertEqual(after["jobs"][job]["terminal"]["elapsed_seconds"], seconds)
            self.assertIn(job, after["superseded_pre_admission_jobs"])
        self.assertEqual(after["actual_CPU_seconds"], before["actual_CPU_seconds"])
        self.assertEqual(after["actual_CPU_seconds"] - self.replay()["actual_CPU_seconds"], 646)
        self.assertEqual(after["outstanding_reserved_CPU_seconds"],
                         before["outstanding_reserved_CPU_seconds"] + 7200)
        self.assertEqual(after["assigned"], before["assigned"])
        self.assertEqual(after["additional_remainder_durable_bytes"],
                         before["additional_remainder_durable_bytes"])
        for number in (9, 10):
            self.assertEqual(after["batches"][f"batch_{number:04d}"], before["batches"][f"batch_{number:04d}"])
        for work in bundle.MEMBERS:
            self.assertEqual(remainder.registered_remainder_job(after, self.plan, work)["job_id"], "970_5")
        # The charged inspection timeout does not prevent independent fresh work
        # once the replacement is accepted; both actual remainder fits remain pending.
        self.admit(after, self.source.batch(("work_0015",)))

    def test_third_review_acknowledges_payload_reads_and_cannot_erase_or_extend_history(self):
        rows, batch = self.third_failure_rows(), self.third_batch()
        before = self.replay(rows)
        pristine = copy.deepcopy(rows)
        self.assertEqual(startup.review_relative(batch[startup.FIELD]), startup.REVIEW_RELATIVE_003)
        for field, value in (("read_only_quarantine_payload_inspected", False),
                             ("io_fragment_inspection_started", False),
                             ("io_fragment_inspection_outcome", "complete"),
                             ("scientific_fit_children_absent", False),
                             ("decoder_fit_request_or_status_created", True),
                             ("predecessor_job_id", "970_5"),
                             ("scientific_payload_admitted", False)):
            changed = copy.deepcopy(batch[startup.FIELD])
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(startup.BundleStartupReplacementError):
                startup.validate_review(changed)
        for job, field, value, message in (
                (startup.PREDECESSOR_003, "elapsed_seconds", 624, "625_second_failure"),
                (startup.PREDECESSOR_003, "allocated_cpus", 0, "625_second_failure"),
                (startup.PREDECESSOR_003, "exit_code", "0:0", "625_second_failure"),
                (startup.PREDECESSOR, "elapsed_seconds", 24, "known_25_and_21"),
                (startup.PREDECESSOR_002, "elapsed_seconds", 20, "known_25_and_21")):
            changed = copy.deepcopy(before)
            changed["jobs"][job]["terminal"][field] = value
            with self.subTest(job=job, field=field), self.assertRaisesRegex(
                    startup.BundleStartupReplacementError, message):
                startup.validate_intent(batch, changed["jobs"], changed["batches"])
        self.assertEqual(rows, pristine)
        changed = copy.deepcopy(before)
        del changed["batches"]["batch_0010"][startup.FIELD]
        with self.assertRaisesRegex(startup.BundleStartupReplacementError, "no_further_chaining"):
            startup.validate_intent(batch, changed["jobs"], changed["batches"])
        for acceptance in ([], [self.base.accepted("970", "batch_0011")]):
            with self.subTest(accepted=bool(acceptance)), self.assertRaisesRegex(
                    startup.BundleStartupReplacementError, "no_further_chaining"):
                self.replay(rows + [self.event(batch, number=11)] + acceptance
                            + [self.event(batch, number=12)])


if __name__ == "__main__":
    unittest.main()
