"""Artificial bundle metadata only; run in the coordinator's Slurm allocation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_remainder_bundle as bundle


def fixture():
    items = []
    for work, index, units in (("work_0005", 4, 246656), ("work_0006", 5, 57408),
                               ("work_0007", 6, 51776), ("work_0013", 12, 14976)):
        items.append({"work_item_id": work, "array_index": index,
            "participant": "sub-01", "units_requested": units, "spans": ["fixture"],
            "resources": {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
                          "scratch_bytes": 2_000_000_000, "durable_bytes": 123456}})
    plan = {"schema_version": "ep21.decoder_deadline_remainders.v1",
        "original_work_manifest": "outputs/decoder_training/full_continuation_001/work_manifest.json",
        "reviewed_partials": bundle.REVIEW_RELATIVE, "automatic_retry": False,
        "preserve_original_fragments": True,
        "scope": "one_manual_exact_remainder_attempt_per_listed_work_item", "work_items": items}
    batch = {"phase": "workers", "manual_worker_refill": True,
        bundle.FIELD: list(bundle.MEMBERS), bundle.PLAN_FIELD: plan,
        "allocations": [{"logical_id": "work_0006", "array_index": 5, "reserved_CPU_seconds": 7200}],
        "accepted_id": "900", "not_submitted": False}
    job = {"logical_id": "work_0006", "job_id": "900_5", "phase": "workers",
           "batch_id": "batch_0016", "reserved_CPU_seconds": 7200, "terminal": None}
    account = {"jobs": {"900_5": job}, "batches": {"batch_0016": batch},
        "actual_CPU_seconds": 12000, "outstanding_reserved_CPU_seconds": 7200,
        "budget": {"cpu_seconds": 1800000}, "uncertain_batches": [], "rejected_batch": False}
    allocation = {"job_id": "901", "allocated_cpus": 1, "memory_gib": 7,
                  "gpus": 0, "time_limit_seconds": 7200}
    environ = {"SLURM_JOB_ID": "901", "SLURM_ARRAY_JOB_ID": "900", "SLURM_ARRAY_TASK_ID": "5"}
    return batch, job, account, allocation, environ


class RemainderBundleTests(unittest.TestCase):
    def test_exact_two_members_share_one_7200_second_reservation_without_mutation(self):
        batch, job, account, _, _ = fixture()
        before = copy.deepcopy(account)
        self.assertEqual(bundle.validate_bundle(batch), ("work_0006", "work_0013"))
        self.assertEqual(bundle.work_ids_for_job(job, account["batches"]), bundle.MEMBERS)
        self.assertEqual(len(batch["allocations"]), 1)
        self.assertEqual(sum(row["reserved_CPU_seconds"] for row in batch["allocations"]), 7200)
        self.assertEqual(account, before)

    def test_rejects_arbitrary_members_changed_plans_or_two_physical_reservations(self):
        batch, *_ = fixture()
        cases = []
        for field, value in ((bundle.FIELD, ["work_0006"]), (bundle.FIELD, ["work_0013", "work_0006"]),
                             (bundle.FIELD, ["work_0006", "work_0007"]),
                             ("phase", "assembly"), ("manual_worker_refill", False),
                             ("reviewed_deadline_partials", {}), ("reviewed_startup_replacement", {})):
            cases.append({**batch, field: value})
        for field, value in (("array_index", 12), ("array_index", 5.0),
                             ("reserved_CPU_seconds", 28800), ("reserved_CPU_seconds", 7200.0)):
            changed = copy.deepcopy(batch)
            changed["allocations"][0][field] = value
            cases.append(changed)
        changed = copy.deepcopy(batch)
        changed["allocations"].append({"logical_id": "work_0013", "array_index": 12,
                                       "reserved_CPU_seconds": 7200})
        cases.append(changed)
        for field, value in (("reviewed_partials", bundle.REVIEW_RELATIVE.replace("002", "001")),
                             ("automatic_retry", True), ("preserve_original_fragments", False)):
            changed = copy.deepcopy(batch)
            changed[bundle.PLAN_FIELD][field] = value
            cases.append(changed)
        changed = copy.deepcopy(batch)
        changed[bundle.PLAN_FIELD]["work_items"][-1]["resources"]["minutes"] = 120
        cases.append(changed)
        for changed in cases:
            with self.subTest(batch=changed), self.assertRaises(bundle.BundleError):
                bundle.validate_bundle(changed)

    def test_ordinary_job_membership_unchanged_and_bundle_anchor_must_be_authentic(self):
        batch, job, account, _, _ = fixture()
        self.assertEqual(bundle.work_ids_for_job({"logical_id": "work_0020", "batch_id": "old"}, {}),
                         ("work_0020",))
        self.assertEqual(bundle.work_ids_for_job({"logical_id": "assembly", "batch_id": "old"}, {}),
                         ("assembly",))
        for field, value in (("logical_id", "work_0013"), ("job_id", "900_12"),
                             ("reserved_CPU_seconds", 28800), ("phase", "assembly")):
            with self.subTest(field=field), self.assertRaises(bundle.BundleError):
                bundle.work_ids_for_job({**job, field: value}, account["batches"])
        for field, value in (("accepted_id", None), ("accepted_id", "0900"), ("not_submitted", True)):
            with self.subTest(field=field), self.assertRaises(bundle.BundleError):
                bundle.work_ids_for_job(job, {"batch_0016": {**batch, field: value}})

    def test_both_children_admit_same_parent_with_unchanged_real_array_and_logical_indices(self):
        batch, job, account, allocation, environ = fixture()
        before = copy.deepcopy((account, allocation, environ))
        returned = []
        for work in bundle.MEMBERS:
            item = next(value for value in batch[bundle.PLAN_FIELD]["work_items"] if value["work_item_id"] == work)
            returned.append(bundle.admit_member(account, {**item, "committed_spans": []}, allocation, environ))
        self.assertIs(returned[0], job)
        self.assertIs(returned[1], job)
        self.assertEqual(batch[bundle.PLAN_FIELD]["work_items"][-1]["array_index"], 12)
        self.assertEqual(environ["SLURM_ARRAY_TASK_ID"], "5")
        self.assertEqual((account, allocation, environ), before)

    def test_refuses_foreign_or_changed_child_wrong_array_resource_or_closed_parent(self):
        batch, _, account, allocation, environ = fixture()
        item = batch[bundle.PLAN_FIELD]["work_items"][-1]
        for changed in ({**item, "work_item_id": "work_0007"}, {**item, "spans": ["different"]},
                        {**item, "array_index": 5}, {**item, "unregistered": True}):
            with self.subTest(item=changed), self.assertRaises(bundle.BundleError):
                bundle.admit_member(account, changed, allocation, environ)
        for field, value in (("SLURM_ARRAY_JOB_ID", "902"), ("SLURM_ARRAY_JOB_ID", "0900"),
                             ("SLURM_ARRAY_TASK_ID", "12"), ("SLURM_JOB_ID", "900")):
            with self.subTest(field=field), self.assertRaises(bundle.BundleError):
                bundle.admit_member(account, item, allocation, {**environ, field: value})
        for field, value in (("allocated_cpus", 2), ("allocated_cpus", True), ("memory_gib", 8),
                             ("gpus", 1), ("time_limit_seconds", 7201), ("time_limit_seconds", 0)):
            with self.subTest(field=field), self.assertRaises(bundle.BundleError):
                bundle.admit_member(account, item, {**allocation, field: value}, environ)
        closed = copy.deepcopy(account)
        closed["jobs"]["900_5"]["terminal"] = {"state": "COMPLETED"}
        with self.assertRaisesRegex(bundle.BundleError, "outstanding_registered"):
            bundle.admit_member(closed, item, allocation, environ)

    def test_unfunded_uncertain_or_rejected_history_cannot_admit_either_member(self):
        batch, _, account, allocation, environ = fixture()
        cases = [{**account, "budget": {"cpu_seconds": 19199}},
                 {**account, "outstanding_reserved_CPU_seconds": 7199},
                 {**account, "actual_CPU_seconds": -1},
                 {**account, "uncertain_batches": [{"batch_id": "unknown"}]},
                 {**account, "rejected_batch": True}]
        for changed in cases:
            for work in bundle.MEMBERS:
                item = next(row for row in batch[bundle.PLAN_FIELD]["work_items"] if row["work_item_id"] == work)
                with self.subTest(work=work, account=changed), self.assertRaisesRegex(bundle.BundleError, "unfunded"):
                    bundle.admit_member(changed, item, allocation, environ)


if __name__ == "__main__":
    unittest.main()
