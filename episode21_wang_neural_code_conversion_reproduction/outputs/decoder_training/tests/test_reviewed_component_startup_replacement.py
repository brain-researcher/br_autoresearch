"""Focused metadata-only refusal checks; coordinator-owned Slurm execution."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import reviewed_component_startup_replacement as startup


def fixture():
    review = {**startup.FIXED, "reviewed_utc": "2026-10-09T06:30:00Z",
        "fix_evidence": "Fixture: plain tee with unchanged PIPESTATUS checks",
        "terminal_evidence": "Fixture: one CPU,39 seconds,FAILED74:0",
        "absence_evidence": "Fixture: no shard,request,status or HDF5",
        "user_authorization_context": "Fixture: manual existing-grant repair"}
    partial = {"reviewed_utc": "2026-10-07T18:07:40Z", "partials": [
        {"work_item_id": work} for work in ("work_0005", "work_0006", "work_0007", "work_0013")]}
    quarantine = {"fixture": "unchanged authentic review validated by caller"}
    previous = {"phase": "workers", "manual_worker_refill": True,
        "allocations": [deepcopy(startup.ALLOCATION)],
        "command": ["sbatch", "/fixture/" + startup.PREVIOUS_WRAPPER_RELATIVE],
        startup.PARTIAL_FIELD: partial, startup.QUARANTINE_FIELD: quarantine,
        "accepted_id": "47080265", "not_submitted": False}
    jobs = {startup.JOB: {"job_id": startup.JOB, "logical_id": startup.WORK,
        "phase": "workers", "batch_id": "batch_0001", "reserved_CPU_seconds": 28800,
        "terminal": {"allocation_job_id": "47080265", "allocated_cpus": 1,
            "elapsed_seconds": 39, "state": "FAILED", "exit_code": "74:0"}}}
    row = {key: deepcopy(value) for key, value in previous.items()
           if key not in {"accepted_id", "not_submitted"}}
    row["command"][-1] = "/fixture/" + startup.WRAPPER_RELATIVE
    row[startup.FIELD] = deepcopy(review)
    return row, jobs, {"batch_0001": previous}


class ComponentStartupReplacementTests(unittest.TestCase):
    def setUp(self):
        self.row, self.jobs, self.batches = fixture()

    def validate(self):
        return startup.validate_intent(self.row, self.jobs, self.batches)

    def test_exact_review_is_nonmutating_and_unrelated_intents_return_none(self):
        original = deepcopy((self.row, self.jobs, self.batches))
        self.assertEqual(self.validate(), startup.JOB)
        self.assertEqual((self.row, self.jobs, self.batches), original)
        del self.row[startup.FIELD]
        self.assertIsNone(self.validate())

    def test_review_cannot_hide_optional_start_or_admit_science_or_extra_storage(self):
        for key, value in (("optional_pipeline_started", False),
                ("scientific_payload_admitted", True), ("scoring_admitted", True),
                ("fragment_created", True), ("request_or_status_created", True),
                ("shard_directory_absent", False), ("additional_durable_bytes", 1),
                ("required_checks_passed", 73), ("optional_component_exit", 0),
                ("optional_log_write_exit", 0), ("automatic_retry", True),
                ("failure_reason", "scientific_IO_failure"), ("array_index", True)):
            with self.subTest(key=key), self.assertRaises(startup.ComponentStartupReplacementError):
                startup.validate_review({**self.row[startup.FIELD], key: value})
        for date in ("invalid", "2026-10-09T06:30:00"):
            with self.subTest(date=date), self.assertRaises(startup.ComponentStartupReplacementError):
                startup.validate_review({**self.row[startup.FIELD], "reviewed_utc": date})

    def test_exact_failed_parent_charge_cannot_be_forged_or_omitted(self):
        for key, value in (("allocation_job_id", "999"), ("allocated_cpus", True),
                ("elapsed_seconds", 38), ("state", "COMPLETED"), ("exit_code", "2:0")):
            row, jobs, batches = fixture()
            jobs[startup.JOB]["terminal"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                    startup.ComponentStartupReplacementError, "39_second_failure"):
                startup.validate_intent(row, jobs, batches)
        self.jobs[startup.JOB]["terminal"] = None
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "39_second_failure"):
            self.validate()

    def test_only_exact_work_reservation_wrapper_and_manual_mode_are_allowed(self):
        for key, value in (("phase", "assembly"), ("manual_worker_refill", False),
                ("allocations", [{**startup.ALLOCATION, "array_index": 48}]),
                ("allocations", [{**startup.ALLOCATION, "reserved_CPU_seconds": 28801}]),
                ("command", ["sbatch", startup.PREVIOUS_WRAPPER_RELATIVE])):
            with self.subTest(key=key), self.assertRaises(startup.ComponentStartupReplacementError):
                startup.validate_intent({**self.row, key: value}, self.jobs, self.batches)
        for field in startup.FORBIDDEN_FIELDS:
            with self.subTest(field=field), self.assertRaises(startup.ComponentStartupReplacementError):
                startup.validate_intent({**self.row, field: {}}, self.jobs, self.batches)

    def test_current_partial_and_quarantine_must_match_accepted_predecessor(self):
        for field in (startup.PARTIAL_FIELD, startup.QUARANTINE_FIELD):
            row = deepcopy(self.row)
            row[field]["changed"] = True
            with self.subTest(field=field), self.assertRaisesRegex(
                    startup.ComponentStartupReplacementError, "preserves_partial002"):
                startup.validate_intent(row, self.jobs, self.batches)
            row.pop(field)
            with self.subTest(missing=field), self.assertRaises(startup.ComponentStartupReplacementError):
                startup.validate_intent(row, self.jobs, self.batches)
        for target in (self.row, self.batches["batch_0001"]):
            target[startup.PARTIAL_FIELD]["partials"].pop()
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "preserves_partial002"):
            self.validate()

    def test_repeated_replacements_extra_attempts_and_unresolved_history_refuse(self):
        self.batches["batch_0002"] = {**deepcopy(self.row), "accepted_id": None, "not_submitted": False}
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "only_no_chaining"):
            self.validate()
        self.batches.pop("batch_0002")
        self.jobs["999_47"] = {**self.jobs[startup.JOB], "job_id": "999_47"}
        with self.assertRaisesRegex(startup.ComponentStartupReplacementError, "only_no_chaining"):
            self.validate()
        self.jobs.pop("999_47")
        for accepted, rejected in ((None, False), ("999", True)):
            self.batches["batch_0002"] = {"accepted_id": accepted, "not_submitted": rejected}
            with self.subTest(accepted=accepted), self.assertRaisesRegex(
                    startup.ComponentStartupReplacementError, "reconciled_accepted_history"):
                self.validate()


if __name__ == "__main__":
    unittest.main()
