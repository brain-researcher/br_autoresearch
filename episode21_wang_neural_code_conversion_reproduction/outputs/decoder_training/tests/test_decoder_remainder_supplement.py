"""Cumulative-plan and serial-bundle integration; compute allocation only.

All histories and records are explicit fixtures. No real journal, scheduler,
closed report or scientific payload is opened.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_assembly as assembly
import decoder_batch_accounting as accounting
import decoder_deadline_remainders as remainder
import decoder_remainder_bundle as bundle
import reviewed_deadline_partials as partials
import submit_full_decoder_jobs as submit
import test_decoder_deadline_remainders as original_fixtures
import test_decoder_remainder_assembly as assembly_fixtures
import test_submit_deadline_remainders as dispatch_fixtures
import test_submit_full_decoder_jobs as base_fixtures


def supplement_fixture():
    authority, manifest, plan1, review1, records, _ = original_fixtures.fixture()
    parent = manifest["work_items"][12]
    request = [original_fixtures.span("conv4_3", channel, 0, 784) for channel in range(108, 128)]
    parent.update(spans=request, units_requested=20*784)
    remaining = copy.deepcopy(request)
    remaining[0]["spatial_Fortran_interval"][0] = 704
    committed = [original_fixtures.span("conv4_3", 108, 0, 704)]
    entry = dict(work_item_id="work_0013", job_id="46881096_12", allocation_job_id="46881096",
        elapsed_seconds=28508, completed_units=704,
        status_path=partials.ROOT + "/shards/work_0013/work_item_status.json")
    review2 = copy.deepcopy(review1)
    review2["reviewed_utc"] = "2026-10-07T18:07:40Z"
    review2["partials"].append(entry)
    record = copy.deepcopy(records["work_0007"])
    record.update(work_item_id="work_0013", resources=parent["resources"], requested_spans=request,
        committed_spans=committed, remaining_spans=remaining, units_requested=20*784, completed_units=704)
    record["allocation"]["job_id"] = "46881096"
    records["work_0013"] = record
    plan2 = copy.deepcopy(plan1)
    plan2["reviewed_partials"] = remainder.REVIEW_RELATIVE_002
    plan2["work_items"].append(dict(work_item_id="work_0013", participant="sub-01", array_index=12,
        predecessor_job_id="46881096_12", predecessor_allocation_job_id="46881096",
        predecessor_status=entry["status_path"],
        output_directory=partials.ROOT + "/remainder_shards_002/work_0013", spans=remaining,
        units_requested=14976, resources=remainder.remainder_resources(14976, 20)))
    return authority, manifest, plan1, plan2, review1, review2, records


class DecoderRemainderSupplementTests(unittest.TestCase):
    def setUp(self):
        (self.authority, self.manifest, self.plan1, self.plan2,
         self.review1, self.review2, self.records) = supplement_fixture()
        self.base = base_fixtures.FullDecoderSubmissionTests()
        self.rows = self.base.rows(self.authority, self.manifest) + [
            self.base.intent(self.authority, self.manifest), self.base.accepted(), self.base.terminal(),
            self.base.intent(self.authority, self.manifest, "workers",
                [f"work_{index+1:04d}" for index in range(8)], number=2),
            self.base.accepted("46839670", "batch_0002")]
        for index in range(8):
            work = f"work_{index+1:04d}"
            if work in remainder.WORK_IDS:
                _, job, raw, _ = remainder.KNOWN[work]
                self.rows.append(self.base.terminal(job, elapsed=28505 if index == 4 else 28503,
                                                    state="FAILED", allocation_job_id=raw))
            else:
                self.rows.append(self.base.terminal(f"46839670_{index}", allocation_job_id=str(1000+index)))
        for number, selected, review, parent in (
                (3, "work_0009", self.review1, "800"),
                (4, "work_0013", self.review1, "46881096"),
                (5, "work_0010", self.review2, "802")):
            batch = submit.build_batch(self.authority, self.manifest, "workers", [selected], "normal",
                                       manual_refill=True, deadline_partial_review=review)
            self.rows.extend([self.event(batch, number), self.base.accepted(parent, f"batch_{number:04d}")])
            if selected == "work_0013":
                self.rows.append(self.base.terminal("46881096_12", elapsed=28508,
                                                    state="FAILED", allocation_job_id="46881096"))
        self.account = self.replay(self.rows)
        self.fixture = dispatch_fixtures.SubmitDeadlineRemainderTests()
        self.fixture.base = self.base
        self.fixture.authority, self.fixture.manifest = self.authority, self.manifest
        self.fixture.plan, self.fixture.review = self.plan1, self.review1
        self.fixture.real_load_context = submit.load_remainder_context
        self.fixture.documents = {submit.DEADLINE_PARTIAL_REVIEW: self.review1,
                                  submit.DEADLINE_PARTIAL_REVIEW_002: self.review2}
        for work, record in self.records.items():
            self.fixture.documents[submit.BASE / partials.ROOT / "shards" / work / "work_item_status.json"] = record

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    def event(self, batch, number=6):
        return {"event": "submission_intent", "batch_id": f"batch_{number:04d}", **batch}

    def grouped(self):
        return submit.build_remainder_bundle(self.authority, self.manifest, self.plan2,
                                              "normal", constraint="CPU_GEN:SIE")

    def old_batch(self, selected=("work_0005",)):
        return submit.build_deadline_remainders(self.authority, self.manifest, self.plan1,
                                                 selected, "normal")

    def close_bundle_records(self, raw="901"):
        for work in bundle.MEMBERS:
            item = next(item for item in self.plan2["work_items"] if item["work_item_id"] == work)
            record = copy.deepcopy(self.records[work])
            record.update(status="complete_work_item_only", resources=item["resources"],
                requested_spans=item["spans"], committed_spans=item["spans"], remaining_spans=[],
                units_requested=item["units_requested"], completed_units=item["units_requested"],
                remainder_plan=str(submit.BASE / remainder.PLAN_RELATIVE_002),
                predecessor_job_id=item["predecessor_job_id"], predecessor_status=item["predecessor_status"])
            record.pop("error")
            record.pop("error_type")
            record[bundle.FIELD] = list(bundle.MEMBERS)
            record["allocation"].update(job_id=raw, time_limit_seconds=7200)
            self.fixture.documents[submit.BASE / item["output_directory"] / "work_item_status.json"] = record

    def test_both_immutable_plans_validate_after_review_extension(self):
        before = copy.deepcopy((self.plan1, self.plan2, self.review1, self.review2, self.records))
        for plan, review, works in ((self.plan1, self.review1, remainder.WORK_IDS),
                                   (self.plan2, self.review2, remainder.WORK_IDS_002)):
            admitted = remainder.validate_plan(plan, self.authority, self.manifest, review,
                {work: self.records[work] for work in works}, self.account, base_directory=str(submit.BASE))
            self.assertEqual(tuple(admitted), works)
        self.assertEqual(self.plan2["work_items"][:3], self.plan1["work_items"])
        self.assertEqual(sum(item["resources"]["durable_bytes"] for item in self.plan2["work_items"]),
                         2546718336)
        self.assertEqual((self.plan1, self.plan2, self.review1, self.review2, self.records), before)

    def test_bundle_admission_reserves_one_two_hour_parent_and_unique_storage(self):
        batch = self.grouped()
        self.assertIn("--time=120", batch["command"])
        self.assertIn("--array=5%1", batch["command"])
        with self.fixture.metadata():
            submit.admit_batch(self.account, self.manifest, batch, results=self.fixture.results)
        updated = self.replay(self.rows + [self.event(batch), self.base.accepted("900", "batch_0006")])
        self.assertEqual(updated["actual_CPU_seconds"], self.account["actual_CPU_seconds"])
        self.assertEqual(updated["outstanding_reserved_CPU_seconds"],
                         self.account["outstanding_reserved_CPU_seconds"] + 7200)
        self.assertEqual(updated["additional_remainder_durable_bytes"], 2546718336)
        self.assertEqual(set(updated["jobs"]) - set(self.account["jobs"]), {"900_5"})
        self.assertEqual(bundle.work_ids_for_job(updated["jobs"]["900_5"], updated["batches"]), bundle.MEMBERS)
        self.assertEqual(submit.summary(updated)["accepted_remainder_attempts"], 2)
        self.assertEqual(submit.summary(updated)["accepted_remainder_parent_allocations"], 1)

    def test_completed_plan001_work7_and_remaining_old_work_survive_supplement(self):
        batch = self.old_batch(("work_0007",))
        history = self.rows + [self.event(batch), self.base.accepted("900", "batch_0006"),
                              self.base.terminal("900_6", elapsed=1338, allocation_job_id="907")]
        account = self.replay(history)
        record = self.fixture.close_remainder_record("work_0007", "907")
        registered = remainder.registered_remainder_job(account, self.plan2, "work_0007", require_terminal=True)
        remainder.validate_completed_remainder(self.plan2["work_items"][2], record, registered,
                                               base_directory=str(submit.BASE))
        with self.fixture.metadata():
            submit.admit_batch(account, self.manifest, self.old_batch(), results=self.fixture.results)
            submit.admit_batch(account, self.manifest, self.grouped(), results=self.fixture.results)
        combined = self.replay(history + [self.event(self.grouped(), 7), self.base.accepted("901", "batch_0007")])
        self.assertEqual(combined["additional_remainder_durable_bytes"], 2546718336)
        self.assertEqual(combined["jobs"]["900_6"], account["jobs"]["900_6"])

    def test_changed_old_item_storage_overcommit_and_repeat_members_refuse(self):
        changed = copy.deepcopy(self.plan2)
        changed["work_items"][0]["resources"]["durable_bytes"] += 1
        self.assertFalse(remainder.plans_compatible(self.plan1, changed))
        with self.assertRaises(partials.ReviewError):
            remainder.validate_plan_structure(changed, self.authority, self.manifest)
        authority = copy.deepcopy(self.authority)
        authority["resources"]["durable_bytes"] = 480_000_000_000
        with self.assertRaisesRegex(partials.ReviewError, "storage_or_concurrent_scratch"):
            remainder.validate_plan_structure(self.plan2, authority, self.manifest)
        history = self.rows + [self.event(self.grouped()), self.base.accepted("900", "batch_0006")]
        account = self.replay(history)
        for attempted in (self.grouped(), self.old_batch(("work_0006",))):
            with self.assertRaisesRegex(partials.ReviewError, "one_original_closed_attempt"):
                remainder.validate_remainder_intent(attempted, account["jobs"], account["batches"],
                                                    self.authority, self.manifest)
        with self.assertRaisesRegex(submit.Stop, "substantive_bundle"):
            submit.build_deadline_remainders(self.authority, self.manifest, self.plan2, ["work_0013"], "normal")

    def test_both_child_reports_required_after_one_parent_charge(self):
        history = self.rows + [self.event(self.grouped()), self.base.accepted("900", "batch_0006"),
                              self.base.terminal("900_5", elapsed=1900, allocation_job_id="901")]
        account = self.replay(history)
        self.close_bundle_records()
        fresh = submit.build_batch(self.authority, self.manifest, "workers", ["work_0011"], "normal",
                                    manual_refill=True, deadline_partial_review=self.review2)
        with self.fixture.metadata():
            submit.admit_batch(account, self.manifest, fresh, results=self.fixture.results)
        path = submit.BASE / self.plan2["work_items"][-1]["output_directory"] / "work_item_status.json"
        record = self.fixture.documents[path]
        sibling = self.fixture.documents[submit.BASE / self.plan2["work_items"][1]["output_directory"]
                                         / "work_item_status.json"]
        record["allocation"]["time_limit_seconds"] = sibling["allocation"]["time_limit_seconds"] = 5400
        with self.fixture.metadata():
            submit.admit_batch(account, self.manifest, fresh, results=self.fixture.results)
        record["allocation"]["time_limit_seconds"] = sibling["allocation"]["time_limit_seconds"] = 7200
        for field, value in (("completed_units", 1), ("remainder_plan", str(submit.REMAINDER_PLAN)),
                             (bundle.FIELD, ["work_0013"]),
                             ("allocation", {**record["allocation"], "time_limit_seconds": 7201})):
            original = record[field]
            record[field] = value
            with self.fixture.metadata(), self.assertRaises(partials.ReviewError):
                submit.admit_batch(account, self.manifest, fresh, results=self.fixture.results)
            record[field] = original
        self.assertEqual(account["actual_CPU_seconds"] - self.account["actual_CPU_seconds"], 1900)

    def test_only_registered_superseded_startup_is_excluded_from_predecessor_count(self):
        account = copy.deepcopy(self.account)
        startup = copy.deepcopy(account["jobs"]["46881096_12"])
        startup.update(job_id="46880582_12", batch_id="fixture_startup")
        startup["terminal"].update(job_id="46880582_12", allocation_job_id="46880582", elapsed_seconds=15)
        account["jobs"]["46880582_12"] = startup
        account["batches"]["fixture_replacement"] = {"accepted_id": "46881096",
            "reviewed_startup_replacement": {"predecessor_job_id": "46880582_12"}}
        remainder.validate_remainder_intent(self.grouped(), account["jobs"], account["batches"],
                                            self.authority, self.manifest)
        account["batches"]["fixture_replacement"]["accepted_id"] = None
        with self.assertRaisesRegex(partials.ReviewError, "one_original_closed_attempt"):
            remainder.validate_remainder_intent(self.grouped(), account["jobs"], account["batches"],
                                                self.authority, self.manifest)

    def test_four_remainder_origins_keep_complete_catalog_without_duplicate_rows(self):
        fixture = assembly_fixtures.DecoderRemainderAssemblyTests
        fixture.setUpClass()
        manifest, replacements = copy.deepcopy(fixture.manifest), copy.deepcopy(fixture.remainders)
        source = next(item for item in manifest["work_items"] if item["work_item_id"] == "fixture_sub-01")
        original = source["spans"].pop()
        start, stop = original["spatial_Fortran_interval"]
        source["units_requested"] -= stop-start
        item = dict(work_item_id="work_0013", participant="sub-01", spans=[original], units_requested=stop-start)
        manifest["work_items"].append(item)
        committed, remaining = copy.deepcopy(original), copy.deepcopy(original)
        midpoint = (start+stop)//2
        committed["spatial_Fortran_interval"][1] = midpoint
        remaining["spatial_Fortran_interval"][0] = midpoint
        replacements["work_0013"] = dict(work_item_id="work_0013", participant="sub-01",
            predecessor_status=partials.ROOT + "/shards/work_0013/work_item_status.json",
            output_directory=partials.ROOT + "/remainder_shards_002/work_0013",
            committed_spans=[committed], spans=[remaining], units_requested=stop-midpoint)
        fragments, states = assembly.planned_catalog(manifest, (), remainder_items=replacements)
        self.assertEqual(assembly.verify_catalog(fragments, states)["units"], 74_306_440)
        self.assertEqual(sum(ref.span.count for ref in fragments if "remainder_shards_002" in ref.path), stop-midpoint)


if __name__ == "__main__":
    unittest.main()
