"""Four focused operator-route checks; run only in coordinated Slurm.

All records, stats and scheduler identities below are fixtures. No actual
journal, closed fragment, scientific payload or scheduler is accessed.
"""
from copy import deepcopy
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import submit_full_decoder_jobs as submit
import reviewed_io_recovery as recovery
import reviewed_io_quarantine as quarantine
import decoder_parallelism_amendment as parallelism
import test_reviewed_io_recovery as fixtures
import test_submit_full_decoder_jobs as base_fixtures


class SubmitIOComplementTests(unittest.TestCase):
    def setUp(self):
        self.review, selected_manifest, self.jobs, self.records, self.reports, self.quarantine = \
            fixtures.fixture(str(submit.BASE))
        self.authority, self.manifest = base_fixtures.FullDecoderSubmissionTests().plans(count=75)
        for index, item in enumerate(selected_manifest["work_items"]):
            if "resources" in item:
                self.manifest["work_items"][index] = deepcopy(item)
        self.documents = {submit.BASE / item["predecessor_status"]: self.records[item["work_item_id"]]
                          for item in self.review["work_items"]}
        self.documents.update({submit.BASE / path: report for path, report in self.reports.items()})

    def io_batch(self, works=("work_0032", "work_0062")):
        result = submit.build_io_remainders(self.authority, self.manifest, self.review, works,
            "russpold,normal", constraint="CPU_GEN:SIE", effective_parallelism=parallelism.FIXED_POLICY)
        result[quarantine.FIELD] = deepcopy(self.quarantine)
        return result

    def test_explicit_complements_keep_original_indices_one_cpu_full_reservation_and_fresh_wrapper(self):
        before = deepcopy((self.review, self.manifest, self.authority))
        batch = self.io_batch()
        self.assertEqual(batch["allocations"], [
            dict(logical_id="work_0032", array_index=31, reserved_CPU_seconds=28800),
            dict(logical_id="work_0062", array_index=61, reserved_CPU_seconds=28800)])
        self.assertEqual(batch["command"][-1], str(submit.IO_REMAINDER_WRAPPER))
        for argument in ("--array=31,61%2", "--cpus-per-task=1", "--mem=7G", "--time=480",
                         "--no-requeue", "--constraint=CPU_GEN:SIE"):
            self.assertIn(argument, batch["command"])
        self.assertTrue(batch["manual_worker_refill"])
        self.assertEqual(batch[recovery.FIELD], self.review)
        self.assertEqual((self.review, self.manifest, self.authority), before)
        for works in ((), ("work_0032", "work_0032"), ("work_0048",)):
            with self.subTest(works=works), self.assertRaises(submit.Stop):
                self.io_batch(works)
        with self.assertRaises(parallelism.ParallelismError):
            submit.build_io_remainders(self.authority, self.manifest, self.review, ("work_0032",), "normal")

    def test_context_binds_complete_reports_and_only_three_exact_closed_file_stats(self):
        seen = []
        files = self.review["retained_original_reservation"]["files"]
        def file_stat(path):
            relative = str(path.relative_to(submit.BASE))
            seen.append(relative)
            evidence = files[relative]
            return SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_nlink=1, st_size=evidence["st_size"])
        def reader(path):
            self.assertEqual(path.suffix, ".json", "submission must not open scientific HDF5 payload")
            return deepcopy(self.documents[path])
        account = {"jobs": self.jobs}
        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "is_symlink", return_value=False), \
                mock.patch.object(Path, "lstat", new=file_stat):
            result = submit.load_io_recovery_context(account, self.manifest, self.review, reader=reader)
            self.assertEqual(set(result), set(recovery.WORK_IDS))
            self.assertEqual(seen, list(files))
            report = self.documents[submit.BASE / self.review["work_items"][-1]["inspection"]["report"]]
            report["fragments"][0]["inspection_complete"] = False
            with self.assertRaises(recovery.RecoveryError):
                submit.load_io_recovery_context(account, self.manifest, self.review, reader=reader)
            report["fragments"][0]["inspection_complete"] = True
            with mock.patch.object(Path, "lstat", return_value=SimpleNamespace(
                    st_mode=stat.S_IFLNK | 0o777, st_nlink=1, st_size=1)), \
                    self.assertRaisesRegex(submit.Stop, "file_accounting_changed"):
                submit.load_io_recovery_context(account, self.manifest, self.review, reader=reader)
            with mock.patch.object(Path, "lstat", return_value=SimpleNamespace(
                    st_mode=stat.S_IFREG | 0o600, st_nlink=1, st_size=4000000001)), \
                    self.assertRaisesRegex(submit.Stop, "file_accounting_changed"):
                submit.load_io_recovery_context(account, self.manifest, self.review, reader=reader)

    def test_completed_recovery_reads_fresh_status_and_keeps_participant_binding(self):
        work = "work_0062"
        record, job = fixtures.completed_record(self.review, work, base_directory=str(submit.BASE))
        batch = {**fixtures.batch(self.review, self.quarantine, (work,)), "accepted_id": "9000"}
        account = {"jobs": {job["job_id"]: job}, "batches": {job["batch_id"]: batch}}
        item = next(item for item in self.review["work_items"] if item["work_item_id"] == work)
        expected_path = submit.BASE / item["output_directory"] / "work_item_status.json"
        reader = mock.Mock(return_value=record)
        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "is_symlink", return_value=False):
            submit.require_previous_results(account, self.manifest, reader=reader)
            reader.assert_called_once_with(expected_path)
            bad = deepcopy(record)
            bad["shared_state_reference"] = str(submit.BASE / "outputs/decoder_training/retained_prefix_001/shared_training_state.npz")
            with self.assertRaises(recovery.RecoveryError):
                submit.require_previous_results(account, self.manifest, reader=lambda path: bad)
            bad = deepcopy(record)
            bad["remaining_spans"] = deepcopy(item["spans"])
            with self.assertRaises(recovery.RecoveryError):
                submit.require_previous_results(account, self.manifest, reader=lambda path: bad)

    def test_optional_components_only_select_work48_without_recovery_authority(self):
        batch = submit.build_batch(self.authority, self.manifest, "workers", ("work_0048",), "normal",
            manual_refill=True, effective_parallelism=parallelism.FIXED_POLICY)
        batch[quarantine.FIELD] = deepcopy(self.quarantine)
        original = deepcopy(batch)
        submit.select_component_wrapper(batch, parallelism.FIXED_POLICY)
        self.assertEqual(batch["command"][-1], str(submit.COMPONENT_WRAPPER))
        self.assertEqual(batch["command"][:-1], original["command"][:-1])
        self.assertEqual(batch["allocations"], original["allocations"])
        self.assertNotIn(recovery.FIELD, batch)
        for bad in (self.io_batch(), {**original, "phase": "assembly"},
                    {**original, "manual_worker_refill": False}):
            with self.assertRaisesRegex(submit.Stop, "fresh_amended_work48"):
                submit.select_component_wrapper(bad, parallelism.FIXED_POLICY)


if __name__ == "__main__":
    unittest.main()
