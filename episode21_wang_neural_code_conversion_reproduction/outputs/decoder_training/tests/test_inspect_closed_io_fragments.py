"""Focused closed-I/O inspection fixtures; execute only in root's Slurm job.

All HDF5 and normalization arrays here are small constructed fixtures, never
scientific inputs. They exercise uncertain markers, damaged committed rows,
read-only behavior, bounded reads and failure propagation after the actual I/O
incident; they are not a scientific simulation or model qualification program.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import sys
from contextlib import contextmanager, redirect_stdout

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_closed_io_fragments as inspection


class ClosedIOInspectionTests(unittest.TestCase):
    def fixture(self, base, *, count=8, committed=3, ambiguous=False):
        work = "work_0061" if ambiguous else "work_0033"
        participant, array_id, raw_id = inspection.WORKERS[work]
        span = inspection.assembly.coverage.Span(participant, "conv1_1", 12 if ambiguous else 40, 0, count)
        requested = span.record()
        saved = inspection.assembly.coverage.Span(*span.key, 0, committed).record()
        remaining = inspection.assembly.coverage.Span(*span.key, committed, count).record()
        status = {"schema_version": "ep21.decoder_work_item.v1", "status": "partial_work_item_stopped",
            "work_item_id": work, "participant": participant,
            "allocation": {"job_id": raw_id, "hostname": "sh04-04n21"},
            "authorization": str(Path(base) / "outputs/decoder_training_full_authorization.json"),
            "work_manifest": str(Path(base) / inspection.FULL_ROOT / "work_manifest.json"),
            "training_role": "subject_unique", "training_images": 4712, "fragment": "channels.h5",
            "shared_state_reference": str(Path(base) / inspection.assembly.required_state(participant).path),
            "training_row_binding_reference": str(Path(base) / inspection.assembly.required_state(participant).binding_path),
            "requested_spans": [requested], "committed_spans": [saved], "remaining_spans": [remaining],
            "units_requested": count, "completed_units": committed, "closeout_errors": [],
            "error_type": "OSError", "error": "[Errno 5] Input/output error",
            "completed_units_is_lower_bound": ambiguous,
            "remaining_interval_status": ("not_authoritative_for_repair_due_to_uncertain_marker_publication"
                                          if ambiguous else "exact_uncommitted_coverage")}
        for key in ("predictions_or_scores_computed", "tau_train_values_read", "tau_test_values_read",
                    "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry"):
            status[key] = False
        if ambiguous:
            status["uncertain_publication"] = {**remaining, "spatial_Fortran_interval": [committed, committed + 256],
                                                "phase": "completion_marker_fsync", "error_type": "OSError"}
        path = Path(base) / inspection.FULL_ROOT / "shards" / work / "channels.h5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as handle:
            handle.attrs.update(schema_version="ep21.decoder_work_item_fragment.v1", participant=participant,
                                work_item_id=work, units_requested=count, complete_decoder=False)
            group = handle.create_group(f"{span.layer}/channel_{span.channel:03d}")
            group.attrs.update(participant=participant, layer=span.layer, channel=span.channel,
                spatial_Fortran_start=0, spatial_Fortran_stop=count, complete_decoder=False,
                state_codes="0=incomplete,2=durably_complete", source_fragment_axes=inspection.assembly.AXES)
            markers = np.zeros(count, dtype="u1")
            markers[:committed] = 2
            group.create_dataset("state", data=markers)
            group.create_dataset("selected_native_voxel_ids", data=np.tile(np.arange(1000, 1500, dtype="<i8"), (count, 1)))
            group.create_dataset("weights", data=np.full((count, 500), .25, dtype="<f4"))
            for name, value in (("bias", .5), ("target_mean", 3), ("target_std", 2)):
                group.create_dataset(name, data=np.full(count, value, dtype="<f4"))
            group.create_dataset("solve_seconds", data=np.zeros(count, dtype="<f8"))
            spatial = np.arange(count, dtype="<i8")
            group.create_dataset("spatial_Fortran_index", data=spatial)
            group.create_dataset("full_layer_Fortran_index", data=span.channel + 64 * spatial)
            group.create_dataset("full_layer_C_index", data=span.channel * 224 * 224 + (spatial % 224) * 224 + spatial // 224)
        native = np.arange(1000, 1600, dtype=np.int64)
        mask = np.arange(600) < 550
        normalization = inspection.assembly.Normalization(participant, native, native[mask], mask,
            np.zeros(600, dtype=np.float32), np.ones(600, dtype=np.float32), None, None)
        request = dict(status, status="admitted_pending")
        item = {"work_item_id": work, "participant": participant, "units_requested": count, "spans": [requested]}
        terminal = {"event": "terminal_charge", "job_id": array_id, "allocation_job_id": raw_id,
                    "state": "FAILED", "exit_code": "2:0", "allocated_cpus": 1,
                    "elapsed_seconds": 1000, "observed_utc": "2026-10-07T23:48:00Z", "evidence": "fixture"}
        return path, f"{span.layer}/channel_{span.channel:03d}", status, normalization, request, item, terminal

    def inspect(self, directory, fixture, check=lambda: None):
        return inspection.inspect_fragment(directory, fixture[2], fixture[3], check)

    def test_clean_exact_prefix_read_only_and_no_reuse_authority(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            path, _, status, _, request, item, terminal = fixture
            inspection.validate_record(status["work_item_id"], item, status, request, terminal, temporary)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            original = copy.deepcopy(status)
            result = self.inspect(temporary, fixture)
            self.assertEqual(result["status"], "consistent")
            self.assertTrue(result["inspection_complete"])
            self.assertEqual(result["reported_committed_units_checked"], 3)
            self.assertEqual(result["observed_marker_2_units"], 3)
            self.assertFalse(result["retry_authorized"])
            self.assertFalse(result["reuse_authorized"])
            self.assertFalse(result["repair_coverage_written"])
            self.assertEqual(status, original)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_ambiguous_markers_observed_separately_never_promoted(self):
        for marked in (0, 128, 256):
            with self.subTest(marked=marked), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(temporary, count=300, committed=3, ambiguous=True)
                path, group, status, _, request, item, terminal = fixture
                inspection.validate_record(status["work_item_id"], item, status, request, terminal, temporary)
                with h5py.File(path, "r+") as handle:
                    # Alternating markers test holes within the ambiguous interval.
                    selected = np.zeros(256, dtype="u1")
                    selected[:] = 2 if marked == 256 else 0
                    if marked == 128:
                        selected[::2] = 2
                    handle[group]["state"][3:259] = selected
                result = self.inspect(temporary, fixture)
                self.assertEqual(result["status"], "consistent")
                self.assertEqual(result["reported_committed_units_checked"], 3)
                self.assertEqual(result["ambiguous_marker_2_units_checked"], marked)
                evidence = result["channels"][0]["ambiguous_publication_observation"]
                self.assertEqual(evidence["marker_0_units"], 256 - marked)
                self.assertEqual(evidence["marker_2_units_with_valid_coefficients"], marked)
                self.assertFalse(evidence["historical_durability_reconciled"])
                self.assertNotIn("repaired_committed_spans", result)

    def test_missing_invalid_or_unreported_markers_are_corrupt(self):
        for index, value in ((1, 0), (1, 1), (5, 2)):
            with self.subTest(index=index, value=value), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(temporary)
                with h5py.File(fixture[0], "r+") as handle:
                    handle[fixture[1]]["state"][index] = value
                result = self.inspect(temporary, fixture)
                self.assertEqual(result["status"], "corrupt")
                self.assertFalse(result["inspection_complete"])

    def test_committed_payload_validation_detects_silent_damage(self):
        changes = [("weights", (0, 0), np.nan), ("bias", 0, np.inf), ("target_std", 0, 0),
                   ("solve_seconds", 0, -1), ("selected_native_voxel_ids", (0, 1), 1000),
                   ("selected_native_voxel_ids", (0, 1), 1599), ("full_layer_C_index", 0, -1)]
        for name, index, value in changes:
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(temporary)
                with h5py.File(fixture[0], "r+") as handle:
                    handle[fixture[1]][name][index] = value
                result = self.inspect(temporary, fixture)
                self.assertEqual(result["status"], "corrupt")
                self.assertFalse(result["reuse_authorized"])

    def test_hdf5_external_link_is_rejected_without_following_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            with h5py.File(fixture[0], "r+") as handle:
                group = handle[fixture[1]]
                del group["weights"]
                group["weights"] = h5py.ExternalLink("never-open-this-missing-file.h5", "/weights")
            result = self.inspect(temporary, fixture)
            self.assertEqual(result["status"], "corrupt")
            self.assertIn("link forbidden", result["error"])

    def test_unreadable_and_timeout_are_not_successful_partial_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            with mock.patch.object(inspection.assembly, "read_sparse_block", side_effect=OSError(5, "fixture I/O error")), \
                 mock.patch.object(inspection.signal, "setitimer") as timer:
                result = self.inspect(temporary, fixture)
            timer.assert_called_once_with(inspection.signal.ITIMER_REAL, 0)
            self.assertEqual(result["status"], "unreadable")
            self.assertEqual(result["errno"], 5)
            self.assertFalse(result["inspection_complete"])
            def expired():
                raise inspection.InspectionTimeout("fixture limit")
            result = self.inspect(temporary, fixture, expired)
            self.assertEqual(result["status"], "timed_out")
            self.assertFalse(result["inspection_complete"])
            def resource_limit():
                raise inspection.InspectionLimit("fixture RSS limit")
            with mock.patch.object(inspection.signal, "setitimer") as timer:
                result = self.inspect(temporary, fixture, resource_limit)
            timer.assert_called_once_with(inspection.signal.ITIMER_REAL, 0)
            self.assertEqual(result["status"], "resource_limit")
            self.assertFalse(result["inspection_complete"])

    def test_close_error_invalidates_an_otherwise_complete_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            original = h5py.File
            @contextmanager
            def fail_on_close(path, mode):
                self.assertEqual(mode, "r")
                with original(path, mode) as handle:
                    yield handle
                raise OSError(108, "fixture close failure")
            with mock.patch.object(h5py, "File", side_effect=fail_on_close):
                result = self.inspect(temporary, fixture)
            self.assertEqual(result["reported_committed_units_checked"], 3)
            self.assertEqual(result["status"], "unreadable")
            self.assertEqual(result["errno"], 108)
            self.assertFalse(result["inspection_complete"])

    def test_larger_prefix_streams_only_bounded_coefficient_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary, count=600, committed=515)
            original = inspection.assembly.read_sparse_block
            intervals = []
            def checked(group, origin, ref, normalization, begin, end):
                self.assertLessEqual(end - begin, 256)
                intervals.append((begin, end))
                return original(group, origin, ref, normalization, begin, end)
            with mock.patch.object(inspection.assembly, "read_sparse_block", side_effect=checked):
                result = self.inspect(temporary, fixture)
            self.assertEqual(result["status"], "consistent")
            self.assertEqual(intervals, [(0, 256), (256, 512), (512, 515)])
            self.assertEqual(result["reported_committed_units_checked"], 515)

    def test_admission_rejects_live_parent_and_changed_original_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            _, _, status, _, request, item, terminal = fixture
            for bad in (dict(terminal, state="RUNNING"), dict(terminal, allocation_job_id="12345")):
                with self.assertRaises(ValueError):
                    inspection.validate_record(status["work_item_id"], item, status, request, bad, temporary)
            with self.assertRaises(ValueError):
                inspection.validate_record(status["work_item_id"], dict(item, units_requested=7), status,
                                           request, terminal, temporary)

    def test_budget_is_at_most_600_seconds_and_reserves_actual_allocation_end(self):
        with mock.patch.object(inspection.time, "time", return_value=1000), \
             mock.patch.object(inspection.time, "monotonic", return_value=200):
            budget = inspection.Budget(600, 1400)
            self.assertEqual(budget.seconds, 100)
            self.assertEqual(budget.deadline, 300)
            with self.assertRaises(ValueError):
                inspection.Budget(601, 2000)
            with self.assertRaises(ValueError):
                inspection.Budget(600, 1299)

    def test_selected_admission_does_not_open_unselected_work_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            path, _, status, _, request, item, terminal = fixture
            root = Path(temporary) / inspection.FULL_ROOT
            manifest = root / "work_manifest.json"
            status_path = path.parent / "work_item_status.json"
            request_path = path.parent / "requested_units.json"
            manifest.write_text(json.dumps({"work_items": [item]}))
            (root / "submission.jsonl").write_text(json.dumps(terminal) + "\n")
            status_path.write_text(json.dumps(status))
            request_path.write_text(json.dumps(request))
            # Every unselected work directory is absent, including the first
            # default work item. Only the selected terminal identity is present.
            with mock.patch.object(inspection, "_read_json", wraps=inspection._read_json) as reader:
                records = inspection.admit_inputs(temporary, [status["work_item_id"]])
            self.assertEqual(records, [status])
            self.assertEqual([call.args[0] for call in reader.call_args_list],
                             [manifest, status_path, request_path])

    def test_selection_keeps_default_scope_and_rejects_unknown_empty_or_duplicate_ids(self):
        self.assertEqual(inspection.selected_work_items(), list(inspection.WORKERS))
        self.assertEqual(inspection.selected_work_items(["work_0062", "work_0047"]),
                         ["work_0047", "work_0062"])
        for bad in ([], "work_0047", ["work_0006"], ["../work_0047"],
                    ["work_0047", "work_0047"], [None]):
            with self.subTest(selection=bad), self.assertRaises(ValueError):
                inspection.selected_work_items(bad)

    def run_batch_fixture(self, results, *, work_items=None, normalization_error=None):
        environment = {"SLURM_JOB_ID": "999999", "SLURM_CPUS_PER_TASK": "1", "SLURM_MEM_PER_NODE": "7168"}
        environment.update({key: "1" for key in inspection.training.prefix.THREADS})
        selected = inspection.selected_work_items(work_items)
        records = [{"work_item_id": work, "participant": inspection.WORKERS[work][0]} for work in selected]
        output = io.StringIO()
        args = argparse.Namespace(authorization="fixture", max_seconds=600,
                                  allocation_end_epoch=time.time() + 7200, work_items=work_items)
        with mock.patch.dict(os.environ, environment, clear=True), \
             mock.patch.object(inspection.Path, "cwd", return_value=inspection.BASE), \
             mock.patch.object(inspection.socket, "gethostname", return_value="sh04-01n01"), \
             mock.patch.object(inspection.training, "authorize_full"), \
             mock.patch.object(inspection, "admit_inputs", return_value=records) as admission, \
             mock.patch.object(inspection.assembly, "load_normalization", return_value=object(),
                               side_effect=normalization_error), \
             mock.patch.object(inspection, "inspect_fragment", side_effect=results) as reader, \
             redirect_stdout(output):
            code = inspection.run(args)
            admission.assert_called_once_with(inspection.BASE, selected)
        return code, json.loads(output.getvalue()), reader.call_count

    def test_batch_stops_immediately_on_new_io_error_or_timeout(self):
        for status in ("unreadable", "timed_out", "resource_limit"):
            result = {"work_item_id": "work_0032", "status": status, "inspection_complete": False}
            code, report, calls = self.run_batch_fixture([result])
            self.assertEqual(code, 3 if status == "timed_out" else 2)
            self.assertEqual(calls, 1)
            self.assertEqual(report["status"], "stopped_" + status)
            self.assertFalse(report["all_five_fragments_completely_checked"])
            self.assertEqual(len(report["not_inspected_work_items"]), 4)
        class AlarmAfterAppend(dict):
            armed = True
            def __getitem__(self, key):
                if key == "status" and self.armed:
                    self.armed = False
                    raise inspection.InspectionTimeout("fixture alarm after result append")
                return super().__getitem__(key)
        for status in ("unreadable", "resource_limit"):
            with self.subTest(alarm_after_recorded_failure=status):
                result = AlarmAfterAppend(work_item_id="work_0033", status=status, inspection_complete=False)
                code, report, calls = self.run_batch_fixture([result], work_items=["work_0033"])
                self.assertEqual((code, calls), (2, 1))
                self.assertEqual(report["status"], "stopped_timed_out")
                self.assertEqual(report["fragments"][0]["status"], status)
                self.assertFalse(report["all_requested_fragments_completely_checked"])

    def test_selected_scope_does_not_claim_all_five_checked_or_release_quarantine(self):
        result = {"work_item_id": "work_0047", "status": "consistent", "inspection_complete": True}
        code, report, calls = self.run_batch_fixture([result], work_items=["work_0047"])
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(report["requested_work_items"], ["work_0047"])
        self.assertEqual(report["attempted_work_items"], ["work_0047"])
        self.assertEqual(report["completely_checked_work_items"], ["work_0047"])
        self.assertEqual(report["not_inspected_requested_work_items"], [])
        self.assertEqual(report["unrequested_work_items"],
                         ["work_0032", "work_0033", "work_0061", "work_0062"])
        self.assertEqual(report["not_inspected_work_items"], report["unrequested_work_items"])
        self.assertTrue(report["all_requested_fragments_completely_checked"])
        self.assertFalse(report["all_five_fragments_completely_checked"])
        self.assertEqual(report["quarantined_work_items"], list(inspection.WORKERS))
        self.assertFalse(report["reuse_authorized"])
        self.assertFalse(report["retry_authorized"])
        self.assertFalse(report["repair_coverage_written"])

    def test_internal_timeout_does_not_mask_prior_io_and_preinspection_limits_remain_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(temporary)
            original = h5py.File
            @contextmanager
            def timeout_while_unwinding(path, mode):
                with original(path, mode) as handle:
                    try:
                        yield handle
                    finally:
                        raise inspection.InspectionTimeout("fixture alarm during unwinding")
            with mock.patch.object(h5py, "File", side_effect=timeout_while_unwinding), \
                 mock.patch.object(inspection.assembly, "read_sparse_block",
                                   side_effect=OSError(5, "fixture read failure before alarm")):
                result = self.inspect(temporary, fixture)
            self.assertEqual(result["status"], "timed_out")
            self.assertEqual(result["earlier_error"]["error_type"], "OSError")
            code, report, calls = self.run_batch_fixture([result], work_items=["work_0033"])
            self.assertEqual((code, calls), (2, 1))
            self.assertFalse(report["all_requested_fragments_completely_checked"])
        for error, expected_code in ((inspection.InspectionTimeout("fixture budget"), 3),
                                     (OSError(108, "fixture normalization close"), 2),
                                     (inspection.InspectionLimit("fixture RSS"), 2)):
            with self.subTest(error=type(error).__name__):
                code, report, calls = self.run_batch_fixture([], work_items=["work_0062"],
                                                             normalization_error=error)
                self.assertEqual((code, calls), (expected_code, 0))
                self.assertEqual(report["requested_work_items"], ["work_0062"])
                self.assertEqual(report["attempted_work_items"], [])
                self.assertEqual(report["completely_checked_work_items"], [])
                self.assertEqual(report["not_inspected_requested_work_items"], ["work_0062"])
                self.assertFalse(report["all_requested_fragments_completely_checked"])
                self.assertFalse(report["reuse_authorized"])

    def test_completed_semantic_findings_do_not_authorize_reuse_or_abort_unrelated_work(self):
        results = [{"work_item_id": work, "status": "corrupt", "inspection_complete": False}
                   for work in inspection.WORKERS]
        code, report, calls = self.run_batch_fixture(results)
        self.assertEqual(code, 0)
        self.assertEqual(calls, 5)
        self.assertEqual(report["status"], "inconsistent_observations")
        self.assertFalse(report["all_five_fragments_completely_checked"])
        self.assertFalse(report["reuse_authorized"])
        self.assertFalse(report["retry_authorized"])


if __name__ == "__main__":
    unittest.main()
