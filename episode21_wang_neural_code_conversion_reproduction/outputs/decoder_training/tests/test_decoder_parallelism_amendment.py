"""Six finite parallelism checks; execute only in coordinator-owned Slurm.

Fixtures contain fabricated metadata with real layer names and bank dimensions.
No actual authority, journal, scheduler, neural/feature payload or fit is opened.
"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_batch_accounting as accounting
import decoder_parallelism_amendment as parallelism
import decoder_remainder_bundle as bundle
import submit_full_decoder_jobs as submit
import test_submit_full_decoder_jobs as fixtures
import test_decoder_remainder_supplement as remainder_fixtures


class DecoderParallelismAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.FullDecoderSubmissionTests()
        self.authority, self.manifest = self.base.plans(count=75)
        self.authority["resources"].update(worker_jobs=75, state_build_jobs=1,
            assembly_jobs=1, peak_transient_bytes=16_000_000_000)
        for item in self.manifest["work_items"]:
            item["spans"] = [{"participant": "sub-01", "layer": "conv3_1", "channel": 0,
                              "spatial_Fortran_interval": [0, 1]}]
        self.rows = self.base.refill_history(self.authority, self.manifest)
        self.event = parallelism.make_event(observed_utc="2026-10-08T01:12:00Z",
            user_instruction="Fixture: maximize jobs within unchanged total budgets")

    def replay(self, rows):
        return accounting.replay(rows, self.authority, self.manifest)

    def batch(self, selected, number=3, amended=True):
        batch = submit.build_batch(self.authority, self.manifest, "workers", selected, "normal",
            manual_refill=True, effective_parallelism=parallelism.FIXED_POLICY if amended else None)
        return {"event": "submission_intent", "batch_id": f"batch_{number:04d}", **batch}

    def test_prospective_event_keeps_original_history_and_tags_only_later_intents(self):
        pristine = copy.deepcopy((self.rows, self.authority, self.manifest))
        before = self.replay(self.rows)
        amended = self.replay(self.rows + [self.event])
        self.assertIsNone(before["effective_parallelism"])
        self.assertEqual(amended["effective_parallelism"], parallelism.FIXED_POLICY)
        for field in ("basis", "budget", "jobs", "batches", "assigned", "actual_CPU_seconds",
                      "outstanding_reserved_CPU_seconds", "additional_remainder_durable_bytes"):
            self.assertEqual(amended[field], before[field])
        batch = self.batch(["work_0009"])
        after = self.replay(self.rows + [self.event, batch, self.base.accepted("500", "batch_0003")])
        self.assertNotIn("effective_parallelism", after["batches"]["batch_0002"])
        self.assertEqual(after["batches"]["batch_0003"]["effective_parallelism"], parallelism.FIXED_POLICY)
        self.assertNotIn("effective_parallelism", batch)
        self.assertEqual((self.rows, self.authority, self.manifest), pristine)
        for changed_authority, changed_manifest in ((copy.deepcopy(self.authority), self.manifest),
                                                   (self.authority, copy.deepcopy(self.manifest))):
            if changed_authority is not self.authority:
                changed_authority["resources"]["maximum_concurrent_jobs"] = 42
            else:
                changed_manifest["work_items"][0]["resources"]["scratch_bytes"] = 1_000_000_000
            with self.assertRaisesRegex(accounting.AccountingError, "immutable"):
                accounting.replay(self.rows + [self.event], changed_authority, changed_manifest)

    def test_42_low_bank_workers_pass_43_refuses_and_array_throttle_is_prospective(self):
        selected = [f"work_{number:04d}" for number in range(9, 44)]
        legacy = self.batch(selected, amended=False)
        self.assertTrue(any(value.endswith("%8") for value in legacy["command"] if value.startswith("--array=")))
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_worker_cap"):
            self.replay(self.rows + [legacy])
        batch = self.batch(selected)
        self.assertTrue(any(value.endswith("%35") for value in batch["command"] if value.startswith("--array=")))
        self.assertEqual(batch["command"][-1], str(submit.PARALLEL_WRAPPER))
        rows = self.rows + [self.event, batch, self.base.accepted("500", "batch_0003")]
        account = self.replay(rows)
        self.assertEqual(sum(job["terminal"] is None for job in account["jobs"].values()), 42)
        self.assertEqual(account["outstanding_reserved_CPU_seconds"], 42*28800)
        with self.assertRaisesRegex(parallelism.ParallelismError, "42_worker_cap"):
            self.replay(rows + [self.batch(["work_0044"], number=4)])
        spoof = copy.deepcopy(self.batch(["work_0009"]))
        spoof["effective_parallelism"] = copy.deepcopy(parallelism.FIXED_POLICY)
        with self.assertRaisesRegex(accounting.AccountingError, "derived_from_prior_event"):
            self.replay(self.rows + [self.event, spoof])

    def test_full_channel_storage_and_serial_bundle_count_once_and_overflow_before42(self):
        large = copy.deepcopy(self.manifest)
        for item in large["work_items"]:
            item["spans"][0].update(layer="conv1_1", spatial_Fortran_interval=[1, 2])
        bank = 4712*224*224*4
        self.assertEqual(parallelism.work_scratch_bytes(large, ("work_0001",)), bank + 64*1024**2)
        jobs = {f"200_{index}": {"job_id": f"200_{index}", "logical_id": f"work_{index+1:04d}",
                "phase": "workers", "terminal": None, "batch_id": "old"} for index in range(15)}
        batches = {"old": {"accepted_id": "200", "not_submitted": False}}
        request = {"phase": "workers", "manual_worker_refill": True,
                   "allocations": [{"logical_id": "work_0016"}]}
        with self.assertRaisesRegex(parallelism.ParallelismError, "scratch_exceeds_16GB"):
            parallelism.validate_reservation(request, jobs, batches, large, parallelism.FIXED_POLICY)
        jobs.pop("200_14")
        self.assertEqual(parallelism.validate_reservation(request, jobs, batches, large,
            parallelism.FIXED_POLICY), 15*(bank + 64*1024**2))
        fixture = remainder_fixtures.DecoderRemainderSupplementTests()
        fixture.setUp()
        grouped = fixture.grouped()
        job = {"job_id": "900_5", "logical_id": bundle.ANCHOR, "phase": "workers", "terminal": None,
               "batch_id": "grouped", "reserved_CPU_seconds": 7200}
        accepted = {"grouped": {**grouped, "accepted_id": "900", "not_submitted": False}}
        expected = max(parallelism.work_scratch_bytes(fixture.manifest, (work,)) for work in bundle.MEMBERS)
        self.assertEqual(parallelism.outstanding_scratch_bytes({"900_5": job}, accepted,
            fixture.manifest, parallelism.FIXED_POLICY), expected)
        self.assertEqual(parallelism.validate_reservation(grouped, {}, {}, fixture.manifest,
            parallelism.FIXED_POLICY), expected)

    def test_policy_cannot_raise_total_budgets_erase_history_or_amend_unresolved_admission(self):
        for field, value in (("maximum_allocated_CPU_seconds", 1800001),
                ("durable_bytes", 500000000001), ("peak_transient_bytes", 16000000001),
                ("maximum_concurrent_jobs", 43), ("scratch_headroom_bytes", 0),
                ("scientific_fit_roles_and_kernels_unchanged", False)):
            event = copy.deepcopy(self.event)
            event["policy"][field] = value
            with self.subTest(field=field), self.assertRaises(parallelism.ParallelismError):
                self.replay(self.rows + [event])
        with self.assertRaisesRegex(accounting.AccountingError, "one_parallelism_amendment"):
            self.replay(self.rows + [self.event, copy.deepcopy(self.event)])
        with self.assertRaisesRegex(parallelism.ParallelismError, "reconciled_accepted_history"):
            self.replay(self.rows + [self.batch(["work_0009"], amended=False), self.event])
        event = copy.deepcopy(self.event)
        event["observed_utc"] = "2026-10-08T01:12:00"
        with self.assertRaisesRegex(parallelism.ParallelismError, "timezone"):
            self.replay(self.rows + [event])

    def test_full_new_CPU_reservations_remain_inside_unchanged_500_hour_cap(self):
        # Each old 8-hour worker is closed before the next; only one original
        # worker remains active. All historical individual charges fit their limits.
        rows = self.base.refill_history(self.authority, self.manifest, closed=7)
        for number, work in enumerate(range(9, 64), start=3):
            raw = str(1000 + work)
            rows.extend([self.batch([f"work_{work:04d}"], number=number, amended=False),
                self.base.accepted(raw, f"batch_{number:04d}"),
                self.base.terminal(f"{raw}_{work-1}", elapsed=28800, allocation_job_id=raw)])
        before = self.replay(rows + [self.event])
        self.assertEqual(before["actual_CPU_seconds"], 480 + 55*28800)
        self.assertEqual(before["outstanding_reserved_CPU_seconds"], 28800)
        selected = [f"work_{work:04d}" for work in range(64, 71)]
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            self.replay(rows + [self.event, self.batch(selected, number=58)])
        with self.assertRaisesRegex(accounting.AccountingError, "exceeds_CPU_cap"):
            accounting.admit_reservation(before, 7*28800)
        self.assertEqual(before["budget"]["cpu_seconds"], 1800000)
        self.assertEqual(before["budget"]["durable_bytes"], 500000000000)

    def test_amendment_is_appended_once_under_existing_lock_without_rewriting_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.jsonl"
            with accounting.locked_events(path, create=True) as (rows, append):
                for event in self.rows:
                    append(event)
                prefix = path.read_bytes()
                account = accounting.append_parallelism_amendment(rows, append,
                    self.authority, self.manifest, observed_utc=self.event["observed_utc"],
                    user_instruction=self.event["user_instruction"])
                self.assertTrue(path.read_bytes().startswith(prefix))
                self.assertEqual(rows[-1]["event"], parallelism.EVENT)
                self.assertEqual(account["effective_parallelism"], parallelism.FIXED_POLICY)
                length = len(rows)
                with self.assertRaisesRegex(accounting.AccountingError, "one_parallelism_amendment"):
                    accounting.append_parallelism_amendment(rows, append, self.authority,
                        self.manifest, observed_utc=self.event["observed_utc"],
                        user_instruction=self.event["user_instruction"])
                self.assertEqual(len(rows), length)


if __name__ == "__main__":
    unittest.main()
