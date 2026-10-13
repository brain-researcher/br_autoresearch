"""Four exact clean-work5 admission checks; coordinated Slurm only.

All histories and statuses are fabricated metadata. No real job, journal,
fragment, participant payload or source fit is opened by these checks.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_deadline_remainders as remainder
import decoder_parallelism_amendment as parallelism
import decoder_remainder_bundle as bundle
import reviewed_io_quarantine as quarantine
import submit_full_decoder_jobs as submit
import train_decoder_remainder as worker
import test_reviewed_io_quarantine as fixtures


class Work5QuarantineRemainderTests(unittest.TestCase):
    def setUp(self):
        self.source = fixtures.ReviewedIOQuarantineTests()
        self.source.setUp()
        self.authority, self.manifest = self.source.authority, self.source.manifest
        self.base, self.fixture = self.source.base, self.source.fixture
        self.plan = copy.deepcopy(self.source.source.plan2)
        grouped = copy.deepcopy(self.source.source.grouped())
        grouped[quarantine.FIELD] = copy.deepcopy(self.source.review)
        # Close the two artificial generic workers before applying channel-bank
        # accounting; their unrelated fixture spans are deliberately empty.
        self.rows = copy.deepcopy(self.source.rows + [
            self.base.terminal("800_8", allocation_job_id="801"),
            self.base.terminal("802_9", allocation_job_id="803"),
            self.source.source.event(grouped, 8), self.base.accepted("950", "batch_0008"),
            parallelism.make_event(observed_utc="2026-10-08T01:25:00Z",
                user_instruction="Fixture: maximize authorized workers within existing total caps")])
        self.account = self.replay(self.rows)

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    def batch(self):
        batch = submit.build_deadline_remainders(self.authority, self.manifest,
            copy.deepcopy(self.plan), ["work_0005"], "normal", clean_work5=True)
        batch[quarantine.FIELD] = copy.deepcopy(self.source.review)
        return batch

    def event(self, batch=None, number=9):
        return self.source.source.event(self.batch() if batch is None else batch, number)

    def admit(self, batch=None, account=None):
        with self.fixture.metadata():
            return submit.admit_batch(self.account if account is None else account, self.manifest,
                self.batch() if batch is None else batch, results=self.fixture.results)

    def test_exact_work5_admits_with_unchanged_quarantine_storage_charges_and_original_fragment(self):
        before = copy.deepcopy((self.rows, self.account, self.plan, self.source.review))
        batch = self.batch()
        self.assertIs(batch[quarantine.CLEAN_WORK5_FIELD], True)
        self.assertEqual(batch["allocations"], [{"logical_id": "work_0005", "array_index": 4,
                                               "reserved_CPU_seconds": 28800}])
        self.assertEqual(Path(batch["command"][-1]).name, "full_remainder_work5_001.sbatch")
        self.admit(batch)
        after = self.replay(self.rows + [self.event(batch), self.base.accepted("960", "batch_0009")])
        self.assertEqual(after["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        self.assertEqual(after["assigned"], self.account["assigned"])
        self.assertEqual(after["additional_remainder_durable_bytes"],
                         self.account["additional_remainder_durable_bytes"])
        self.assertEqual(after["outstanding_reserved_CPU_seconds"],
                         self.account["outstanding_reserved_CPU_seconds"] + 28800)
        self.assertEqual(after["jobs"]["46839670_4"], self.account["jobs"]["46839670_4"])
        for work, (job, _, _) in quarantine.KNOWN.items():
            self.assertEqual(after["jobs"][job], self.account["jobs"][job])
            self.assertNotIn(job, after["superseded_pre_admission_jobs"])
        self.assertEqual(after["batches"]["batch_0009"][quarantine.FIELD], self.source.review)
        self.assertEqual(after["batches"]["batch_0009"]["effective_parallelism"], parallelism.FIXED_POLICY)
        self.assertEqual(remainder.registered_remainder_job(after, self.plan, "work_0005")["job_id"], "960_4")
        self.assertEqual((self.rows, self.account, self.plan, self.source.review), before)

    def test_exact_marker_plan_work_identity_resources_and_no_quarantine_permission_changes(self):
        good = self.batch()
        cases = []
        for key, value in ((quarantine.CLEAN_WORK5_FIELD, False), (quarantine.CLEAN_WORK5_FIELD, 1),
                           (bundle.FIELD, list(bundle.MEMBERS)), ("phase", "assembly")):
            changed = copy.deepcopy(good)
            changed[key] = value
            cases.append(changed)
        for key, value in (("logical_id", "work_0007"), ("array_index", 5), ("reserved_CPU_seconds", 7200)):
            changed = copy.deepcopy(good)
            changed["allocations"][0][key] = value
            cases.append(changed)
        for key, value in (("units_requested", 246655), ("predecessor_job_id", "46918887_31"),
                           ("output_directory", "elsewhere")):
            changed = copy.deepcopy(good)
            changed[bundle.PLAN_FIELD]["work_items"][0][key] = value
            cases.append(changed)
        changed = copy.deepcopy(good)
        changed[bundle.PLAN_FIELD]["reviewed_partials"] = remainder.REVIEW_RELATIVE
        cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed["allocations"]), self.assertRaises(quarantine.QuarantineError):
                quarantine.validate_clean_work5_remainder(changed)
        missing = copy.deepcopy(good)
        del missing[quarantine.CLEAN_WORK5_FIELD]
        with self.assertRaises(bundle.BundleError):
            self.admit(missing)
        changed = copy.deepcopy(good)
        changed[quarantine.FIELD]["allow_fragment_reuse"] = True
        with self.assertRaises(quarantine.QuarantineError):
            self.admit(changed)
        with self.assertRaises(submit.Stop):
            submit.build_deadline_remainders(self.authority, self.manifest,
                self.plan, ["work_0013"], "normal", clean_work5=True)
        self.assertEqual(good[bundle.PLAN_FIELD], self.plan)

    def test_original_closeout_remains_required_and_started_work5_cannot_retry(self):
        item = self.plan["work_items"][0]
        original = submit.BASE / item["predecessor_status"]
        saved = self.fixture.documents[original]
        changed = copy.deepcopy(saved)
        changed["closeout_errors"] = [{"affects_fragment": True}]
        self.fixture.documents[original] = changed
        try:
            with self.assertRaisesRegex(remainder.partials.ReviewError, "uncertain_or_unfinalized"):
                self.admit()
        finally:
            self.fixture.documents[original] = saved
        rows = self.rows + [self.event(), self.base.accepted("960", "batch_0009")]
        with self.assertRaises(remainder.partials.ReviewError):
            self.replay(rows + [self.event(number=10)])
        self.assertEqual(self.fixture.documents[original], saved)

    def test_runtime_binds_derived_work5_scratch_and_keeps_legacy_resources_and_records(self):
        item = copy.deepcopy(self.plan["work_items"][0])
        resources = copy.deepcopy(item["resources"])
        expected = 4712*224*224*4 + 64*1024**2
        admitted = {"effective_scratch_bytes": expected,
                    "parallelism_amendment_id": parallelism.EVENT}
        effective = worker.remainder_runtime_resources(self.manifest, "work_0005", resources, admitted)
        self.assertEqual(effective, {**resources, "scratch_bytes": expected})
        self.assertEqual(resources, item["resources"])
        legacy = worker.remainder_runtime_resources(self.manifest, "work_0005", resources, None)
        self.assertEqual(legacy, resources)
        self.assertIsNot(legacy, resources)
        for changed in ({**admitted, "effective_scratch_bytes": expected-1},
                        {**admitted, "parallelism_amendment_id": "other"}):
            with self.assertRaises(worker.contract.Stop):
                worker.remainder_runtime_resources(self.manifest, "work_0005", resources, changed)


if __name__ == "__main__":
    unittest.main()
