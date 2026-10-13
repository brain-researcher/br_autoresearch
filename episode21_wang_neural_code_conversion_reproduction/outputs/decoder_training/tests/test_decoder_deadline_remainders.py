"""Exact remainder metadata fixtures; no real artifacts, payloads or scheduler."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_deadline_remainders as remainder
import reviewed_deadline_partials as partials
import submit_full_decoder_jobs as submit
import test_reviewed_deadline_partials as base_fixtures


def span(layer, channel, start, stop):
    return {"participant": "sub-01", "layer": layer, "channel": channel,
            "spatial_Fortran_interval": [start, stop]}


def fixture():
    """Only known interval/count metadata is copied into these artificial jobs."""
    base = base_fixtures.ReviewedDeadlinePartialTests()
    base.setUp()
    authority, manifest = base.fixture.plans(count=75)
    authority["resources"].update(worker_jobs=75, peak_transient_bytes=16_000_000_000)
    authority["closed_included_layer_storage"] = {"charged_bytes": 4_313_632_594}
    authority["state_build_resources"]["durable_bytes"] = 8_000_000_000
    authority["assembly_resources"]["durable_bytes"] = 1_000_000_000
    for item in manifest["work_items"]:
        item["resources"]["durable_bytes"] = 6_200_000_000
    requests = {
        "work_0005": [span("conv1_2", channel, 0, 50176) for channel in range(30, 50)],
        "work_0006": [span("conv1_2", channel, 0, 50176) for channel in range(50, 64)]
            + [span("conv2_1", channel, 64 if channel == 0 else 0, 12544) for channel in range(24)],
        "work_0007": [span("conv2_1", channel, 0, 12544) for channel in range(24, 104)],
    }
    cursors = {"work_0005": ("conv1_2", 45, 4224),
               "work_0006": ("conv2_1", 19, 5312), "work_0007": ("conv2_1", 99, 10944)}
    review = {**copy.deepcopy(base.review), "partials": []}
    plan = {"schema_version": "ep21.decoder_deadline_remainders.v1",
            "original_work_manifest": remainder.ORIGINAL_MANIFEST_RELATIVE,
            "reviewed_partials": remainder.REVIEW_RELATIVE, "automatic_retry": False,
            "preserve_original_fragments": True,
            "scope": "one_manual_exact_remainder_attempt_per_listed_work_item", "work_items": []}
    records, jobs = {}, {}
    for work in remainder.WORK_IDS:
        index, job_id, allocation, units = remainder.KNOWN[work]
        original = manifest["work_items"][index]
        original.update(spans=requests[work], units_requested=sum(
            row["spatial_Fortran_interval"][1]-row["spatial_Fortran_interval"][0] for row in requests[work]))
        layer, channel, cursor = cursors[work]
        pivot = next(i for i, row in enumerate(requests[work]) if (row["layer"], row["channel"]) == (layer, channel))
        committed, remaining = copy.deepcopy(requests[work][:pivot+1]), copy.deepcopy(requests[work][pivot:])
        committed[-1]["spatial_Fortran_interval"][1] = cursor
        remaining[0]["spatial_Fortran_interval"][0] = cursor
        completed = original["units_requested"]-units
        elapsed = 28505 if work == "work_0005" else 28503
        entry = dict(work_item_id=work, job_id=job_id, allocation_job_id=allocation,
                     elapsed_seconds=elapsed, completed_units=completed,
                     status_path=f"{partials.ROOT}/shards/{work}/work_item_status.json")
        review["partials"].append(entry)
        terminal = base.fixture.terminal(job_id, elapsed=elapsed, state="FAILED", allocation_job_id=allocation)
        jobs[job_id] = dict(job_id=job_id, logical_id=work, phase="workers", batch_id="original",
                            reserved_CPU_seconds=28800, terminal=terminal)
        record = copy.deepcopy(base.records["work_0002"])
        record.update(work_item_id=work, resources=original["resources"], requested_spans=requests[work],
                      committed_spans=committed, remaining_spans=remaining,
                      units_requested=original["units_requested"], completed_units=completed)
        record["allocation"]["job_id"] = allocation
        records[work] = record
        plan["work_items"].append(dict(work_item_id=work, participant="sub-01", array_index=index,
            predecessor_job_id=job_id, predecessor_allocation_job_id=allocation,
            predecessor_status=entry["status_path"],
            output_directory=f"{partials.ROOT}/remainder_shards_001/{work}", spans=remaining,
            units_requested=units, resources=remainder.remainder_resources(units, 5)))
    account = {"jobs": jobs, "batches": {"published_review": {"accepted_id": "800", partials.FIELD: review}}}
    return authority, manifest, plan, review, records, account


class DecoderDeadlineRemainderTests(unittest.TestCase):
    def setUp(self):
        self.authority, self.manifest, self.plan, self.review, self.records, self.account = fixture()

    def validate(self, **changes):
        values = dict(plan=self.plan, authority=self.authority, original_manifest=self.manifest,
            reviewed_partials=self.review, predecessor_records=self.records, account=self.account,
            base_directory=str(submit.BASE))
        values.update(changes)
        return remainder.validate_plan(**values)

    def batch(self, work="work_0005"):
        return {"phase": "workers", "manual_worker_refill": True, remainder.FIELD: self.plan,
                "allocations": [{"logical_id": work, "array_index": remainder.KNOWN[work][0],
                                 "reserved_CPU_seconds": 28800}]}

    def test_three_exact_remainders_keep_originals_and_reserve_all_extra_storage(self):
        before = copy.deepcopy((self.authority, self.manifest, self.plan, self.review, self.records, self.account))
        admitted = self.validate()
        self.assertEqual(tuple(admitted), remainder.WORK_IDS)
        self.assertEqual(sum(item["units_requested"] for item in admitted.values()), 355840)
        self.assertEqual(sum(item["resources"]["durable_bytes"] for item in admitted.values()), 2368108032)
        for work, item in admitted.items():
            self.assertEqual(item["spans"], self.records[work]["remaining_spans"])
            self.assertEqual(item["committed_spans"], self.records[work]["committed_spans"])
            self.assertNotEqual(item["output_directory"], str(Path(item["predecessor_status"]).parent))
        self.assertEqual((self.authority, self.manifest, self.plan, self.review, self.records, self.account), before)

    def test_extra_targets_replay_paths_changed_resources_or_storage_overcommit_refuse(self):
        for field, value in (("array_index", 5), ("predecessor_job_id", "999_4"),
                ("output_directory", f"{partials.ROOT}/shards/work_0005"), ("units_requested", 246657),
                ("spans", self.manifest["work_items"][4]["spans"]),
                ("resources", {**self.plan["work_items"][0]["resources"], "cpus": 2})):
            altered = copy.deepcopy(self.plan)
            altered["work_items"][0][field] = value
            with self.subTest(field=field), self.assertRaises(partials.ReviewError):
                self.validate(plan=altered)
        authority = copy.deepcopy(self.authority)
        # Original reservations fit; the extra2.368GB crosses this ceiling.
        authority["resources"]["durable_bytes"] = 480_000_000_000
        with self.assertRaisesRegex(partials.ReviewError, "storage_or_concurrent_scratch"):
            self.validate(authority=authority)
        authority = copy.deepcopy(self.authority)
        authority["resources"]["peak_transient_bytes"] = 15_999_999_999
        with self.assertRaises(partials.ReviewError):
            self.validate(authority=authority)

    def test_unreviewed_or_uncertain_predecessor_and_changed_exact_remainder_refuse(self):
        for change in ({"uncertain_publication": {}}, {"error": "nonfinite_solve"},
                       {"closeout_errors": [{"affects_fragment": True}]}, {"completed_units": 756863}):
            records = copy.deepcopy(self.records)
            records["work_0005"].update(change)
            with self.subTest(change=change), self.assertRaises(partials.ReviewError):
                self.validate(predecessor_records=records)
        account = copy.deepcopy(self.account)
        account["batches"] = {}
        with self.assertRaisesRegex(partials.ReviewError, "unchanged_published"):
            self.validate(account=account)
        account = copy.deepcopy(self.account)
        account["jobs"]["46839670_4"]["terminal"]["elapsed_seconds"] -= 1
        with self.assertRaisesRegex(partials.ReviewError, "terminal_charge"):
            self.validate(account=account)

    def test_only_one_explicit_physical_attempt_with_full_cpu_reservation_and_registration(self):
        batch = self.batch()
        remainder.validate_remainder_intent(batch, self.account["jobs"], self.account["batches"],
                                            self.authority, self.manifest)
        for change in ({"manual_worker_refill": False}, {partials.FIELD: self.review}):
            with self.subTest(change=tuple(change)), self.assertRaises(partials.ReviewError):
                remainder.validate_remainder_intent({**batch, **change}, self.account["jobs"],
                    self.account["batches"], self.authority, self.manifest)
        short = copy.deepcopy(batch)
        short["allocations"][0]["reserved_CPU_seconds"] = 3600
        with self.assertRaisesRegex(partials.ReviewError, "eight_hour"):
            remainder.validate_remainder_intent(short, self.account["jobs"], self.account["batches"],
                                                self.authority, self.manifest)
        batches = {**self.account["batches"], "new": {**batch, "accepted_id": "900"}}
        with self.assertRaisesRegex(partials.ReviewError, "one_manual_remainder"):
            remainder.validate_remainder_intent(batch, self.account["jobs"], batches, self.authority, self.manifest)
        with self.assertRaisesRegex(partials.ReviewError, "one_exact_accepted"):
            remainder.registered_remainder_job(self.account, self.plan, "work_0005")
        job = dict(job_id="900_4", logical_id="work_0005", phase="workers", batch_id="new",
                   reserved_CPU_seconds=28800, terminal=None)
        account = {"jobs": {**self.account["jobs"], "900_4": job}, "batches": batches}
        self.assertIs(remainder.registered_remainder_job(account, self.plan, "work_0005"), job)
        with self.assertRaisesRegex(partials.ReviewError, "successfully_charged"):
            remainder.registered_remainder_job(account, self.plan, "work_0005", require_terminal=True)

    def test_assembly_requires_complete_fresh_record_and_exact_terminal_allocation(self):
        item = self.validate()["work_0005"]
        terminal = dict(state="COMPLETED", exit_code="0:0", allocated_cpus=1, allocation_job_id="901")
        job = dict(logical_id="work_0005", terminal=terminal)
        record = copy.deepcopy(self.records["work_0005"])
        record.update(status="complete_work_item_only", resources=item["resources"],
            requested_spans=item["spans"], committed_spans=item["spans"], remaining_spans=[],
            units_requested=item["units_requested"], completed_units=item["units_requested"],
            remainder_plan=str(submit.BASE / remainder.PLAN_RELATIVE),
            predecessor_job_id=item["predecessor_job_id"], predecessor_status=item["predecessor_status"])
        record.pop("error")
        record.pop("error_type")
        record["allocation"]["job_id"] = "901"
        remainder.validate_completed_remainder(item, record, job, base_directory=str(submit.BASE))
        for change in ({"status": "partial_work_item_stopped"}, {"completed_units": 1},
                       {"remaining_spans": item["spans"]}, {"uncertain_publication": {}},
                       {"remainder_plan": "other"}, {"tau_test_values_read": True},
                       {"allocation": {**record["allocation"], "job_id": "46840029"}}):
            with self.subTest(change=tuple(change)), self.assertRaises(partials.ReviewError):
                remainder.validate_completed_remainder(item, {**record, **change}, job, base_directory=str(submit.BASE))


if __name__ == "__main__":
    unittest.main()
