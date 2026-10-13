"""Exact recovery journal admission; coordinator-owned Slurm execution only.

Only fabricated metadata is used. No real journal, payload or scheduler opens.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_deadline_remainders as deadline
import decoder_parallelism_amendment as parallelism
import reviewed_io_quarantine as quarantine
import reviewed_io_recovery as recovery
import test_decoder_remainder_supplement as deadline_fixtures
import test_reviewed_io_recovery as recovery_fixtures


class IORecoveryAccountingTests(unittest.TestCase):
    def setUp(self):
        source = deadline_fixtures.DecoderRemainderSupplementTests()
        source.setUp()
        self.base = source.base
        self.authority, self.manifest = source.authority, source.manifest
        self.plan, self.partial_review = source.plan2, source.review2
        self.review, originals, _, _, _, self.quarantine = recovery_fixtures.fixture()
        for work in recovery.WORK_IDS:
            index = int(work[-4:]) - 1
            self.manifest["work_items"][index] = deepcopy(originals["work_items"][index])
        for item in self.manifest["work_items"]:
            if not item["spans"]:
                item["spans"] = [{"participant": item["participant"], "layer": "conv3_1",
                    "channel": 0, "spatial_Fortran_interval": [0, 1]}]
        # Match the independently recorded complete original storage basis,
        # retaining all five genuine immutable original caps (6,087,397,376).
        self.authority["resources"].update(state_build_jobs=1, assembly_jobs=1)
        target_original = 489756203706 - 2546718336
        original = (sum(item["resources"]["durable_bytes"] for item in self.manifest["work_items"])
            + self.authority["state_build_resources"]["durable_bytes"]
            + self.authority["assembly_resources"]["durable_bytes"] + 64*1024**2
            + self.authority["closed_included_layer_storage"]["charged_bytes"])
        self.manifest["work_items"][-1]["resources"]["durable_bytes"] += target_original-original
        self.rows = deepcopy(source.rows)
        self.rows[1]["authority"] = deepcopy(self.authority)
        self.rows[1]["manifest"] = deepcopy(self.manifest)
        self.rows += [self.base.terminal("800_8", allocation_job_id="810"),
                      self.base.terminal("802_9", allocation_job_id="812")]
        # Every clean deadline remainder is complete, including the unchanged
        # physical two-member bundle, before this independent I/O route.
        fresh_deadline = {"phase": "workers", "manual_worker_refill": True,
            deadline.FIELD: deepcopy(self.plan), "allocations": accounting.allocations_for(
                self.authority, self.manifest, "workers", ("work_0005", "work_0007"))}
        self.add_accepted(self.rows, fresh_deadline, "900")
        self.rows += [self.base.terminal("900_4", allocation_job_id="910"),
                      self.base.terminal("900_6", allocation_job_id="911")]
        self.add_accepted(self.rows, source.grouped(), "901")
        self.rows.append(self.base.terminal("901_5", allocation_job_id="901"))
        for parent, selected in (("46918155", ("work_0061",)),
                ("46918887", ("work_0032", "work_0033", "work_0047", "work_0062"))):
            batch = {"phase": "workers", "allocations": accounting.allocations_for(
                self.authority, self.manifest, "workers", selected)}
            self.add_accepted(self.rows, batch, parent)
        for job, raw, elapsed in quarantine.KNOWN.values():
            self.rows.append(self.base.terminal(job, elapsed=elapsed, state="FAILED", allocation_job_id=raw))
        self.rows.append(parallelism.make_event(observed_utc="2026-10-08T01:40:00Z",
            user_instruction="Fixture: fixed42 amendment within unchanged overall caps"))
        self.before = self.replay(self.rows)

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    def event(self, batch, rows=None):
        rows = self.rows if rows is None else rows
        number = 1 + sum(row.get("event") == "submission_intent" for row in rows)
        return {"event": "submission_intent", "batch_id": f"batch_{number:04d}", **deepcopy(batch)}

    def add_accepted(self, rows, batch, parent):
        event = self.event(batch, rows)
        rows.extend([event, self.base.accepted(parent, event["batch_id"])])
        return event

    def worker(self, works=("work_0032",)):
        batch = recovery_fixtures.batch(self.review, self.quarantine, works)
        batch[accounting.deadline_partials.FIELD] = deepcopy(self.partial_review)
        return batch

    def ordinary(self, works):
        return {"phase": "workers", "manual_worker_refill": True,
            quarantine.FIELD: deepcopy(self.quarantine),
            accounting.deadline_partials.FIELD: deepcopy(self.partial_review),
            "allocations": accounting.allocations_for(self.authority, self.manifest, "workers", works)}

    def complete_recovery(self):
        rows = deepcopy(self.rows)
        self.add_accepted(rows, self.worker(recovery.WORK_IDS), "999")
        for work in recovery.WORK_IDS:
            index = int(work[-4:]) - 1
            rows.append(self.base.terminal(f"999_{index}", elapsed=1500, allocation_job_id=str(2000+index)))
        return rows

    def assembly(self):
        return {"phase": "assembly", deadline.FIELD: deepcopy(self.plan),
            recovery.FIELD: deepcopy(self.review), quarantine.FIELD: deepcopy(self.quarantine),
            "allocations": accounting.allocations_for(self.authority, self.manifest, "assembly")}

    def test_unaccepted_full_reserve_then_ordinary_refill_preserves_charges_and_history(self):
        pristine = deepcopy((self.rows, self.manifest, self.authority))
        proposed = self.event(self.worker())
        uncertain = self.replay(self.rows + [proposed])
        self.assertEqual(len(uncertain["uncertain_batches"]), 1)
        self.assertEqual(uncertain["outstanding_reserved_CPU_seconds"], 28800)
        self.assertEqual(uncertain["io_recovery_gross_durable_bytes"], 10631803136)
        self.assertEqual(uncertain["retained_original_reservation_reduction_bytes"], 2087397376)
        self.assertEqual(uncertain["additional_io_recovery_durable_bytes"], 8544405760)
        self.assertEqual(uncertain["additional_remainder_durable_bytes"], 2546718336)
        self.assertEqual(uncertain["projected_durable_bytes"], 498300609466)
        rows = self.rows + [proposed, self.base.accepted("999", proposed["batch_id"])]
        ordinary = self.event(self.ordinary(("work_0015",)), rows)
        after = self.replay(rows + [ordinary])
        self.assertNotIn(recovery.FIELD, ordinary)
        self.assertEqual(after["projected_durable_bytes"], uncertain["projected_durable_bytes"])
        self.assertEqual(after["actual_CPU_seconds"], self.before["actual_CPU_seconds"])
        self.assertEqual(after["basis"], self.before["basis"])
        for job, _, _ in quarantine.KNOWN.values():
            self.assertEqual(after["jobs"][job], self.before["jobs"][job])
            self.assertNotIn(job, after["superseded_pre_admission_jobs"])
        self.assertEqual((self.rows, self.manifest, self.authority), pristine)

    def test_first_only_exact_attempt_and_immutable_overlays(self):
        rows = deepcopy(self.rows)
        self.add_accepted(rows, self.worker(), "999")
        with self.assertRaisesRegex(recovery.RecoveryError, "one_original_failed_attempt"):
            self.replay(rows + [self.event(self.worker(), rows)])
        unresolved = self.rows + [self.event(self.worker())]
        with self.assertRaisesRegex(recovery.RecoveryError, "one_IO_complement_intent"):
            self.replay(unresolved + [self.event(self.worker(), unresolved)])
        for changed in (self.worker(("work_0033",)), self.ordinary(("work_0015",))):
            changed.pop(quarantine.FIELD)
            with self.assertRaisesRegex(quarantine.QuarantineError, "cannot_change_narrow_or_disappear"):
                self.replay(rows + [self.event(changed, rows)])
        changed = self.worker(("work_0033",))
        changed[recovery.FIELD]["decision"] += " changed"
        with self.assertRaisesRegex(recovery.RecoveryError, "published_IO_recovery_review_is_immutable"):
            self.replay(rows + [self.event(changed, rows)])
        with self.assertRaisesRegex(quarantine.QuarantineError, "no_retry_or_reassignment"):
            self.replay(self.rows + [self.event(self.ordinary(("work_0032",)))])
        absent_policy = [row for row in self.rows if row.get("event") != parallelism.EVENT]
        with self.assertRaisesRegex(accounting.AccountingError, "published_parallelism"):
            self.replay(absent_policy + [self.event(self.worker(), absent_policy)])

    def test_storage_credit_once_only_and_global_current_authority_cap(self):
        batch = self.worker()
        first = accounting.durable_reservations(self.authority, self.manifest,
            self.before["batches"], proposed=batch)
        batches = {**self.before["batches"], "io_first": batch}
        repeated = accounting.durable_reservations(self.authority, self.manifest, batches,
            proposed=self.worker(("work_0033",)))
        self.assertEqual(first, repeated)
        limited = deepcopy(self.authority)
        limited["resources"]["durable_bytes"] = 498300609465
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_durable_cap"):
            accounting.durable_reservations(limited, self.manifest, batches)
        enlarged = deepcopy(self.manifest)
        enlarged["work_items"][-1]["resources"]["durable_bytes"] += 2_000_000_000
        with self.assertRaisesRegex(accounting.AccountingError, "storage_must_match_full_manifest"):
            accounting.durable_reservations(self.authority, enlarged, batches)
        changed = self.worker()
        changed[recovery.FIELD]["retained_original_reservation"]["retained_reserved_bytes"] = 1
        with self.assertRaises(recovery.RecoveryError):
            accounting.durable_reservations(self.authority, self.manifest, batches, proposed=changed)

    def test_assembly_requires_both_reviews_and_every_successful_fresh_parent(self):
        legacy = self.assembly()
        legacy.pop(recovery.FIELD)
        legacy.pop(quarantine.FIELD)
        with self.assertRaisesRegex(accounting.AccountingError, "explicit_IO_recovery_for_incident_parents"):
            accounting.validate_remainder_assembly_intent(legacy, self.before["jobs"],
                self.before["batches"], self.authority, self.manifest)
        rows = self.complete_recovery()
        assembled = self.replay(rows + [self.event(self.assembly(), rows)])
        self.assertEqual(assembled["outstanding_reserved_CPU_seconds"], 14400)
        self.assertEqual(assembled["projected_durable_bytes"], 498300609466)
        for field in (recovery.FIELD, quarantine.FIELD, deadline.FIELD):
            batch = self.assembly()
            batch.pop(field)
            with self.subTest(field=field), self.assertRaises((ValueError, accounting.AccountingError)):
                self.replay(rows + [self.event(batch, rows)])
        failed = deepcopy(rows)
        failed[-1].update(state="FAILED", exit_code="2:0")
        with self.assertRaisesRegex(recovery.RecoveryError, "complete_charged_IO_complement"):
            self.replay(failed + [self.event(self.assembly(), failed)])
        missing = rows[:-1]
        with self.assertRaisesRegex(recovery.RecoveryError, "closed_accounted_history"):
            self.replay(missing + [self.event(self.assembly(), missing)])

    def test_recovery_cannot_bypass_42_count_or_16GB_scratch(self):
        free = [item["work_item_id"] for item in self.manifest["work_items"]
                if item["work_item_id"] not in self.before["assigned"]]
        rows = deepcopy(self.rows)
        self.add_accepted(rows, self.ordinary(free[:42]), "1001")
        with self.assertRaisesRegex(parallelism.ParallelismError, "42_worker_cap"):
            self.replay(rows + [self.event(self.worker(), rows)])
        # Different fixture: 15 outstanding full conv1 banks fit; a sixteenth
        # would exceed16GB even though there are far fewer than42 parents.
        for work in free[:15]:
            item = self.manifest["work_items"][int(work[-4:])-1]
            item["spans"][0]["layer"] = "conv1_1"
        rows = deepcopy(self.rows)
        rows[1]["manifest"] = deepcopy(self.manifest)
        self.add_accepted(rows, self.ordinary(free[:15]), "1002")
        with self.assertRaisesRegex(parallelism.ParallelismError, "scratch_exceeds_16GB"):
            self.replay(rows + [self.event(self.worker(), rows)])

    def test_full_new_cpu_reservations_are_checked_before_recovery_acceptance(self):
        rows = deepcopy(self.rows)
        free = [item["work_item_id"] for item in self.manifest["work_items"]
                if item["work_item_id"] not in self.before["assigned"]]
        count = (1800000 - self.before["actual_CPU_seconds"] - 5*28800)//28800 + 1
        self.assertLessEqual(count, len(free))
        for index, work in enumerate(free[:count]):
            parent = str(10000+index)
            self.add_accepted(rows, self.ordinary((work,)), parent)
            rows.append(self.base.terminal(parent + "_" + str(int(work[-4:])-1),
                elapsed=28800, allocation_job_id=parent))
        self.assertLess(self.replay(rows)["actual_CPU_seconds"], 1800000)
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            self.replay(rows + [self.event(self.worker(recovery.WORK_IDS), rows)])


if __name__ == "__main__":
    unittest.main()
