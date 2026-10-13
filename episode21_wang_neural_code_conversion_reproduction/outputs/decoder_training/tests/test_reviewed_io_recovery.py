"""Seven fixed-incident metadata checks; coordinator Slurm execution only.

All statuses, reports and histories are fabricated in memory. No operational
journal, sidecar, fragment, scientific payload or scheduler is accessed.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import reviewed_io_recovery as recovery
import reviewed_io_quarantine as quarantine

BASE = "/fixture/ep21"


def fixture(base_directory=BASE):
    review = deepcopy(recovery.FIXED)
    review.update(reviewed_utc="2026-10-08T02:00:00Z", decision="Artificial exact-incident fixture", work_items=[])
    directory = recovery.ROOT + "/shards/work_0062/"
    review["retained_original_reservation"] = {
        "work_item_id": "work_0062", "original_reserved_bytes": 6087397376,
        "retained_reserved_bytes": 4000000000, "reservation_reduction_bytes": 2087397376,
        "files_deleted": False, "original_artifacts_modified": False,
        "stat_observed_utc": "2026-10-08T01:36:34Z", "stat_observer": "coordinating_agent",
        "stat_method": "exact_three_file_lstat_no_directory_scan_no_payload_read",
        "files": {directory + name: {"st_size": size, "file_type": "regular", "st_nlink": 1, "is_symlink": False}
                  for name, size in (("channels.h5", 3639836056), ("requested_units.json", 8433), ("work_item_status.json", 12914))},
        "observed_file_bytes": 3639857403, "metadata_allowance_bytes": 67108864,
        "observed_plus_metadata_bytes": 3706966267, "activation_requires_tested_journal_admission": True}
    manifest = {"work_items": [{"work_item_id": f"work_{index:04d}"} for index in range(1, 76)]}
    jobs, records, reports = {}, {}, {}
    quarantine_review = {**deepcopy(quarantine.FIXED), "reviewed_utc": "2026-10-07T23:54:35Z",
                         "evidence": "Artificial incident evidence", "failures": []}
    specifications = ((32, "sub-05", 20, 770560), (33, "sub-05", 40, 632320),
                      (47, "sub-06", 20, 723712), (61, "sub-07", 0, 622080), (62, "sub-07", 20, 571904))
    for number, person, first, saved in specifications:
        work = f"work_{number:04d}"
        job, raw, elapsed = quarantine.KNOWN[work]
        requested, committed, fresh = [], [], []
        for ordinal, channel in enumerate(range(first, first + 20)):
            row = dict(participant=person, layer="conv1_1", channel=channel, spatial_Fortran_interval=[0, 50176])
            requested.append(row)
            stop = min(50176, max(0, saved - ordinal * 50176))
            if stop:
                committed.append({**row, "spatial_Fortran_interval": [0, stop]})
            if stop < 50176:
                fresh.append({**row, "spatial_Fortran_interval": [stop, 50176]})
        # The original cap deliberately differs from the fresh-output formula.
        original_resources = dict(cpus=1, memory_gib=7, gpus=0, minutes=480,
                                  durable_bytes=6087397376, scratch_bytes=2000000000)
        manifest["work_items"][number - 1].update(participant=person, resources=original_resources,
                                                spans=requested, units_requested=1003520)
        original_dir = recovery.ROOT + "/shards/" + work
        errors = [] if number == 62 else [dict(stage=stage, affects_fragment=True, error_type="OSError", error="fixture I/O")
                    for stage in ("fragment_fsync", "fragment_close", "fragment_accounting")]
        uncertain = None
        if number in (32, 61):
            channel, start, stop = (35, 17920, 18176) if number == 32 else (12, 19968, 20224)
            uncertain = dict(participant=person, layer="conv1_1", channel=channel,
                spatial_Fortran_interval=[start, stop], phase="completion_marker_fsync", error_type="OSError")
        old = number in (32, 33)
        report_path = ("outputs/decoder_training/remainder_bundle_checks_job_46945868/closed_io_inspection.json"
                       if old else f"outputs/decoder_training/selected_io_inspection_job_46949821/{work}.json")
        report_time = "2026-10-08T00:36:00Z" if old else "2026-10-08T01:30:00Z"
        report_job = "46945868" if old else "46949821"
        report_status = "stopped_timed_out" if old else "consistent_observations_only"
        item = dict(work_item_id=work, participant=person, array_index=number - 1,
            predecessor_job_id=job, predecessor_allocation_job_id=raw, predecessor_elapsed_seconds=elapsed,
            predecessor_status=original_dir + "/work_item_status.json", predecessor_fragment=original_dir + "/channels.h5",
            predecessor_request=original_dir + "/requested_units.json", original_units_requested=1003520,
            committed_units=saved, committed_spans=deepcopy(committed), spans=deepcopy(fresh), units_requested=1003520-saved,
            output_directory=recovery.ROOT + "/io_remainder_shards_001/" + work,
            resources=dict(cpus=1, memory_gib=7, gpus=0, minutes=480, scratch_bytes=2000000000,
                durable_bytes=(1003520-saved)*6045 + len(fresh)*1048576 + 67108864), effective_scratch_bytes=1012826112,
            inspection=dict(report=report_path, report_observed_utc=report_time, report_status=report_status,
                allocation_job_id=report_job, work_item_id=work, fragment_status="consistent", inspection_complete=True,
                reported_committed_units_checked=saved, observed_marker_2_units=saved, ambiguous_marker_2_units_checked=0),
            historical_closeout_errors=deepcopy(errors), historical_uncertain_publication=deepcopy(uncertain),
            uncertain_rows_in_fresh_remainder=256 if uncertain else 0, uncertain_observed_marker_2_units=0)
        review["work_items"].append(item)
        entry = dict(work_item_id=work, job_id=job, allocation_job_id=raw, elapsed_seconds=elapsed,
                     status_path=item["predecessor_status"])
        quarantine_review["failures"].append(entry)
        jobs[job] = dict(job_id=job, phase="workers", logical_id=work, reserved_CPU_seconds=28800, batch_id="old",
            terminal=dict(state="FAILED", exit_code="2:0", allocated_cpus=1, allocation_job_id=raw, elapsed_seconds=elapsed))
        record = dict(schema_version="ep21.decoder_work_item.v1", status="partial_work_item_stopped",
            work_item_id=work, participant=person, resources=deepcopy(original_resources), error_type="OSError",
            error="[Errno 5] Input/output error", requested_spans=deepcopy(requested), units_requested=1003520,
            committed_spans=deepcopy(committed), completed_units=saved, remaining_spans=deepcopy(fresh),
            authorization=base_directory + "/outputs/decoder_training_full_authorization.json",
            work_manifest=base_directory + "/" + recovery.ROOT + "/work_manifest.json",
            shared_state_reference=base_directory + "/" + recovery.ROOT + "/shared/" + person + "/shared_training_state.npz",
            fragment="channels.h5", training_role="subject_unique", training_images=4712,
            original_neural_values_read=False, retained_neural_state_read=True, original_budget_consumption_preserved=True,
            allocation=dict(job_id=raw, hostname="sh04-04n21", allocated_cpus=1, memory_gib=7, gpus=0, time_limit_seconds=28800),
            closeout_errors=deepcopy(errors), completed_units_is_lower_bound=bool(uncertain),
            remaining_interval_status="not_authoritative_for_repair_due_to_uncertain_marker_publication" if uncertain else "exact_uncommitted_coverage")
        record.update({key: False for key in ("predictions_or_scores_computed", "tau_train_values_read", "tau_test_values_read",
                                            "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry")})
        if uncertain:
            record["uncertain_publication"] = deepcopy(uncertain)
        records[work] = record
        report = reports.setdefault(report_path, dict(schema_version="ep21.closed_io_fragment_inspection.v1",
            observed_utc=report_time, status=report_status, allocation=dict(job_id=report_job), fragments=[],
            original_artifacts_modified=False, repair_coverage_written=False, retry_authorized=False, reuse_authorized=False))
        report["fragments"].append(dict(work_item_id=work, fragment=item["predecessor_fragment"], status="consistent",
            inspection_complete=True, reported_committed_units=saved, reported_committed_units_checked=saved,
            observed_marker_2_units=saved, ambiguous_marker_2_units_checked=0,
            historical_closeout_errors=deepcopy(errors), historical_uncertain_publication=deepcopy(uncertain),
            original_artifacts_modified=False, repair_coverage_written=False, retry_authorized=False, reuse_authorized=False))
    return review, manifest, jobs, records, reports, quarantine_review


def batch(review, quarantine_review, works=("work_0032",)):
    return {"phase": "workers", "manual_worker_refill": True, recovery.FIELD: deepcopy(review),
        quarantine.FIELD: deepcopy(quarantine_review), "allocations": [dict(logical_id=work,
            array_index=int(work[-4:])-1, reserved_CPU_seconds=28800) for work in works]}


def completed_record(review, work, raw="9001", base_directory=BASE):
    _, _, _, originals, _, _ = fixture(base_directory)
    item = next(item for item in review["work_items"] if item["work_item_id"] == work)
    record = deepcopy(originals[work])
    for key in ("error", "error_type", "uncertain_publication"):
        record.pop(key, None)
    record.update(status="complete_work_item_only", resources=deepcopy(item["resources"]),
        requested_spans=deepcopy(item["spans"]), committed_spans=deepcopy(item["spans"]), remaining_spans=[],
        units_requested=item["units_requested"], completed_units=item["units_requested"],
        remaining_interval_status="exact_uncommitted_coverage", completed_units_is_lower_bound=False, closeout_errors=[],
        io_recovery_review=base_directory + "/" + recovery.REVIEW_RELATIVE,
        predecessor_job_id=item["predecessor_job_id"], predecessor_status=item["predecessor_status"],
        effective_scratch_bytes=1012826112, parallelism_amendment_id="parallelism_amendment")
    record["allocation"]["job_id"] = raw
    job = dict(job_id="9000_" + str(item["array_index"]), phase="workers", logical_id=work,
        reserved_CPU_seconds=28800, batch_id="recovery_001", terminal=dict(state="COMPLETED", exit_code="0:0",
            allocated_cpus=1, allocation_job_id=raw, elapsed_seconds=1500))
    return record, job


class ReviewedIORecoveryTests(unittest.TestCase):
    def setUp(self):
        self.review, self.manifest, self.jobs, self.records, self.reports, self.quarantine = fixture()

    def validate(self, *, review=None, jobs=None, records=None, reports=None):
        return recovery.validate_review(self.review if review is None else review, self.manifest,
            self.jobs if jobs is None else jobs, self.records if records is None else records,
            self.reports if reports is None else reports, base_directory=BASE)

    def test_exact_saved_and_fresh_coverage_retains_genuine_original_caps_and_history(self):
        before = deepcopy((self.review, self.manifest, self.jobs, self.records, self.reports))
        admitted = self.validate()
        self.assertEqual(sum(item["committed_units"] for item in admitted.values()), 3320576)
        self.assertEqual(sum(item["units_requested"] for item in admitted.values()), 1697024)
        for work in recovery.WORK_IDS:
            self.assertEqual(self.manifest["work_items"][int(work[-4:])-1]["resources"]["durable_bytes"], 6087397376)
        self.assertEqual((self.review, self.manifest, self.jobs, self.records, self.reports), before)
        self.assertEqual(self.reports[self.review["work_items"][0]["inspection"]["report"]]["status"], "stopped_timed_out")

    def test_incomplete_wrong_report_and_marker_promotion_refused(self):
        path = self.review["work_items"][2]["inspection"]["report"]
        for change in (dict(status="timed_out"), dict(inspection_complete=False),
                       dict(reported_committed_units_checked=723456), dict(observed_marker_2_units=723968),
                       dict(earlier_error="read failed"), dict(reuse_authorized=True)):
            reports = deepcopy(self.reports)
            reports[path]["fragments"][0].update(change)
            with self.subTest(change=change), self.assertRaises(recovery.RecoveryError):
                self.validate(reports=reports)
        reports = deepcopy(self.reports)
        reports[path]["allocation"]["job_id"] = "other"
        with self.assertRaises(recovery.RecoveryError):
            self.validate(reports=reports)

    def test_exact_partition_identity_and_uncertain_rows_cannot_change(self):
        for change in (dict(committed_units=770816), dict(participant="sub-01"), dict(array_index=32),
                       dict(uncertain_rows_in_fresh_remainder=0), dict(output_directory="elsewhere")):
            review = deepcopy(self.review)
            review["work_items"][0].update(change)
            with self.subTest(change=change), self.assertRaises(recovery.RecoveryError):
                self.validate(review=review)
        review = deepcopy(self.review)
        review["work_items"][0]["spans"][0]["spatial_Fortran_interval"][0] -= 256
        with self.assertRaises(recovery.RecoveryError):
            self.validate(review=review)
        review = deepcopy(self.review)
        review["work_items"].pop()
        with self.assertRaises(recovery.RecoveryError):
            self.validate(review=review)

    def test_original_terminal_diagnosis_and_participant_normalization_remain_required(self):
        for field, value in (("state", "COMPLETED"), ("exit_code", "0:0"), ("elapsed_seconds", 12688)):
            jobs = deepcopy(self.jobs)
            jobs["46918887_31"]["terminal"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(jobs=jobs)
        for change in (dict(error_type="Stop"), dict(closeout_errors=[]),
                       dict(shared_state_reference=BASE + "/outputs/decoder_training/retained_prefix_001/shared_training_state.npz"),
                       dict(completed_units_is_lower_bound=False)):
            records = deepcopy(self.records)
            records["work_0032"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(records=records)

    def test_single_attempt_immutable_quarantine_and_separate_assembly_admission(self):
        proposed = batch(self.review, self.quarantine)
        history = {"old": {quarantine.FIELD: deepcopy(self.quarantine)}}
        recovery.validate_intent(proposed, self.jobs, history, self.manifest)
        for key, value in (("manual_worker_refill", False), (quarantine.bundle.FIELD, ["work_0032"]),
                           ("reviewed_deadline_remainder", {})):
            changed = deepcopy(proposed)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(recovery.RecoveryError):
                recovery.validate_intent(changed, self.jobs, history, self.manifest)
        changed = deepcopy(proposed)
        changed[quarantine.FIELD]["evidence"] = "changed history"
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_intent(changed, self.jobs, history, self.manifest)
        history["pending"] = {**deepcopy(proposed), "accepted_id": None}
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_intent(proposed, self.jobs, history, self.manifest)
        changed = deepcopy(proposed)
        changed[recovery.FIELD]["decision"] += " revised"
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_intent(changed, self.jobs, history, self.manifest)
        assembly = {"phase": "assembly", recovery.FIELD: self.review, quarantine.FIELD: self.quarantine}
        with self.assertRaises(recovery.RecoveryError):
            recovery.validate_intent(assembly, self.jobs, {}, self.manifest)

    def test_storage_reconciles_only_work62_reservation_and_reserves_all_complements_once(self):
        self.assertEqual(recovery.reserved_durable_bytes({}), 0)
        two = {"one": {recovery.FIELD: self.review}, "two": {recovery.FIELD: deepcopy(self.review)}}
        self.assertEqual(recovery.reserved_durable_bytes(two), 10631803136)
        self.assertEqual(recovery.net_additional_durable_bytes(self.review), 8544405760)
        self.assertEqual(489756203706 + recovery.net_additional_durable_bytes(self.review), 498300609466)
        for key, value in (("retained_reserved_bytes", 3706966267), ("work_item_id", "work_0061"),
                           ("files_deleted", True), ("observed_file_bytes", 3639836056)):
            review = deepcopy(self.review)
            review["retained_original_reservation"][key] = value
            with self.subTest(key=key), self.assertRaises(recovery.RecoveryError):
                recovery.net_additional_durable_bytes(review)
        review = deepcopy(self.review)
        review["storage_totals"]["fresh_remainder_reserved_bytes"] -= 1
        with self.assertRaises(recovery.RecoveryError):
            recovery.net_additional_durable_bytes(review)

    def test_prefix_exception_and_fresh_completion_keep_their_distinct_records(self):
        original = self.records["work_0032"]
        before = deepcopy(original)
        allowed = recovery.prefix_allowance(self.review, "work_0032", original, base_directory=BASE)
        self.assertEqual(allowed, self.review["work_items"][0]["committed_spans"])
        allowed.clear()
        self.assertEqual(original, before)
        self.assertTrue(original["completed_units_is_lower_bound"])
        self.assertTrue(original["closeout_errors"])
        record, job = completed_record(self.review, "work_0062")
        item = self.review["work_items"][-1]
        recovery.validate_completed_remainder(item, record, job, base_directory=BASE)
        accepted = {**batch(self.review, self.quarantine, ("work_0062",)), "accepted_id": "9000", "not_submitted": False}
        account = {"jobs": {**self.jobs, job["job_id"]: job}, "batches": {"recovery_001": accepted}}
        self.assertEqual(recovery.registered_remainder_job(account, self.review, "work_0062", require_terminal=True), job)
        with self.assertRaises(recovery.RecoveryError):
            recovery.registered_remainder_job(account, self.review, "work_0062")
        for change in (dict(shared_state_reference=BASE + "/wrong_participant.npz"),
                       dict(closeout_errors=[dict(affects_fragment=True)]), dict(effective_scratch_bytes=2000000000)):
            with self.subTest(change=change), self.assertRaises(recovery.RecoveryError):
                recovery.validate_completed_remainder(item, {**record, **change}, job, base_directory=BASE)


if __name__ == "__main__":
    unittest.main()
