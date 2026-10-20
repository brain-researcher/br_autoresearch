"""Five contingency-recovery fixtures; no actual EP21 file or job is read.

Use the existing 224-row fc8 writer fixture with tiny native axes. These tests
must run in a coordinator-owned compute allocation, never on a login node.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import amended_decode_recovery as recovery
import test_finding_a_collection as collection_fixtures
from test_finding_a_grouped_collection import fixture, OWNER, LAYER


original = recovery.original


def closure_fixture():
    jobs = [str(60000000 + index) for index in range(40)]
    claim = {"phase": "decode", "batch_index": None, "operation": recovery.active.DECODE,
        "model_policy": recovery.MODEL_POLICY, "allocation": {"job_id": recovery.SOURCE_JOB},
        "binding": {"decode": {"job_id": recovery.SOURCE_JOB,
            "afterok_job_ids": jobs[:20], "completed_model_job_ids": jobs[20:]}}}
    closure = {"source_job_id": recovery.SOURCE_JOB, "terminal_state": "TIMEOUT",
        "writer_closed": True, "frozen_sources_unchanged": True,
        "hdf_close_completed": True, "hdf_close_evidence": "fixture ordinary HDF close completed",
        "native_file_failure": False, "hdf_corruption_reported": False,
        "original_model_job_ids": sorted(jobs), "recorded_at_utc": "fixture",
        "authority": "fixture only", "terminal_evidence": "fixture terminal record",
        "writer_closure_evidence": "fixture closed writer", "frozen_source_evidence": "fixture original sources"}
    return claim, closure


def nine_stream_fixture():
    arguments, prepared, decoders = fixture()
    template = next(iter(prepared.values()))
    for stream in original.STREAMS:
        if original.metrics._owner(*stream) == OWNER:
            arguments["fitted_object_ids"][stream] = "fixture-original-" + repr(stream)
            if stream not in prepared:
                prepared[stream] = replace(template, method=stream[0], source=stream[1], target=stream[2])
    return arguments, prepared, decoders


def source_fixture(root):
    arguments, prepared, decoders = nine_stream_fixture()
    source_path = root / "source.h5"
    with recovery.active.publication.AmendedStagedCollection(source_path, **arguments) as writer:
        writer.write_owner_layer(owner=OWNER, layer=LAYER, prepared_by_stream=prepared, blocks=iter(decoders))
    return Path(str(source_path) + ".partial"), arguments, prepared


class AmendedDecodeRecoveryTests(unittest.TestCase):
    def setUp(self):
        collection_fixtures.FindingACollectionTests.setUp(self)
        patcher = mock.patch.object(original.assembly, "_require_reader_context",
                                    side_effect=lambda budget: budget.check())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_failed_closed_claim_is_required_and_native_failures_refuse(self):
        claim, closure = closure_fixture()
        self.assertEqual(recovery.validate_source_claim(claim, closure), tuple(closure["original_model_job_ids"]))
        for key, value in (("terminal_state", "RUNNING"), ("terminal_state", "COMPLETED"),
                           ("writer_closed", False), ("frozen_sources_unchanged", False),
                           ("hdf_close_completed", False),
                           ("native_file_failure", True), ("hdf_corruption_reported", True),
                           ("original_model_job_ids", closure["original_model_job_ids"][:-1])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                recovery.validate_source_claim(claim, {**closure, key: value})
        for key in ("native_file_failure", "hdf_corruption_reported", "terminal_evidence",
                    "hdf_close_completed", "hdf_close_evidence"):
            incomplete = dict(closure)
            del incomplete[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                recovery.validate_source_claim(claim, incomplete)
        duplicate = deepcopy(claim)
        duplicate["binding"]["decode"]["completed_model_job_ids"][0] = duplicate["binding"]["decode"]["afterok_job_ids"][0]
        with self.assertRaisesRegex(ValueError, "40_distinct"):
            recovery.validate_source_claim(duplicate, closure)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(recovery.active.original, "_admission", return_value=lambda: None), \
                mock.patch.object(recovery, "_claim", return_value=claim), \
                mock.patch.object(recovery.h5py, "File", side_effect=AssertionError("HDF opened before closure")):
            with self.assertRaisesRegex(ValueError, "actual_failed_closed"):
                recovery.recover_decoded_point(object(), require_admitted_context=lambda *args, **kwargs: None,
                    budget=SimpleNamespace(allocation={"job_id": "fixture"}),
                    stage_h5=Path(directory) / "recovery.h5", maximum_bytes=32 * 1024**2,
                    maximum_source_plus_destination_bytes=96 * 1024**2,
                    authenticated_tau_train_image_ids=(), closure_record={**closure, "writer_closed": False})

    def test_allocated_or_torn_flags_do_not_copy_any_owner_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial, arguments, prepared = source_fixture(root)
            moment = "decoders/" + OWNER + "/feature_moments/" + LAYER
            locations = ["layers/%03d" % original._SLOT_NUMBER[(*stream, LAYER)] for stream in prepared]
            for index, mode in enumerate(("moments_incomplete", "one_stream_incomplete", "all_streams_incomplete")):
                with h5py.File(partial, "r+") as source:
                    source[moment].attrs["complete"] = mode != "moments_incomplete"
                    for position, location in enumerate(locations):
                        source[location].attrs["complete"] = not (mode == "all_streams_incomplete"
                            or mode == "one_stream_incomplete" and position == 0)
                with h5py.File(partial, "r") as source, recovery.active.publication.AmendedStagedCollection(
                        root / ("incomplete%d.h5" % index), **arguments) as writer:
                    recovery.validate_header(source, writer)
                    self.assertFalse(recovery.copy_committed_owner_layer(source, writer, owner=OWNER, layer=LAYER))
                    self.assertFalse(writer._records)
                    self.assertNotIn("layers", writer._h5)
                    self.assertNotIn(moment, writer._h5)

    def test_complete_nine_stream_group_preserves_values_and_rebuilds_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial, arguments, prepared = source_fixture(root)
            before = recovery._stat_identity(partial)
            with h5py.File(partial, "r") as source, recovery.active.publication.AmendedStagedCollection(
                    root / "recovered.h5", **arguments) as writer:
                recovery.validate_header(source, writer)
                self.assertTrue(recovery.copy_committed_owner_layer(source, writer, owner=OWNER, layer=LAYER))
                self.assertEqual(set(writer._records), {(*stream, LAYER) for stream in prepared})
                self.assertEqual(len(writer._records), 9)
                self.assertFalse(writer._scores)
                for stream in prepared:
                    key = (*stream, LAYER)
                    record = writer._records[key]
                    self.assertEqual(record["fitted_object_id"], arguments["fitted_object_ids"][stream])
                    self.assertEqual(record["status"], "decoded")
                    self.assertEqual(record["logical_shape"], [224, 1000])
                    source_group = source[record["hdf5_group"]]
                    copied = writer._h5[record["hdf5_group"]]
                    self.assertTrue(copied.attrs["complete"])
                    for name in ("source_file_features", "valid_mask", "full_layer_C_index", "full_layer_Fortran_index"):
                        self.assertEqual(copied[name].dtype, source_group[name].dtype)
                        np.testing.assert_array_equal(copied[name][...], source_group[name][...])
                moment = "decoders/" + OWNER + "/feature_moments/" + LAYER
                for name in ("mean", "std"):
                    np.testing.assert_array_equal(writer._h5[moment + "/" + name][...], source[moment + "/" + name][...])
                # Recovering one owner/layer must not bypass the original855-slot requirement.
                with self.assertRaisesRegex(ValueError, "all_855"):
                    writer.seal_decoded()
            self.assertEqual(recovery._stat_identity(partial), before)
            self.assertFalse((root / "recovered.h5").exists())

    def test_corrupt_marked_complete_groups_fail_instead_of_becoming_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial, arguments, prepared = source_fixture(root)
            stream = next(iter(prepared))
            location = "layers/%03d" % original._SLOT_NUMBER[(*stream, LAYER)]
            moment = "decoders/" + OWNER + "/feature_moments/" + LAYER
            cases = ((location + "/full_layer_C_index", 0, "duplicate", "coordinates"),
                     (location + "/full_layer_Fortran_index", 0, -1, "coordinates"),
                     (location + "/valid_mask", (0, 0), False, "valid_mask"),
                     (location + "/source_file_features", (0, 0), np.nan, "predictions"),
                     (moment + "/std", 0, 0., "moments"))
            for index, (path, position, replacement, reason) in enumerate(cases):
                with h5py.File(partial, "r+") as source:
                    saved = source[path][position]
                    source[path][position] = source[path][1] if replacement == "duplicate" else replacement
                try:
                    with h5py.File(partial, "r") as source, recovery.active.publication.AmendedStagedCollection(
                            root / ("corrupt%d.h5" % index), **arguments) as writer:
                        recovery.validate_header(source, writer)
                        with self.assertRaisesRegex(ValueError, reason):
                            recovery.copy_committed_owner_layer(source, writer, owner=OWNER, layer=LAYER)
                        self.assertFalse(writer._records)
                        self.assertTrue(writer._closed)
                finally:
                    with h5py.File(partial, "r+") as source:
                        source[path][position] = saved

    def test_header_policy_image_and_native_axis_mismatch_refuse_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial, arguments, _ = source_fixture(root)
            for index, kind in enumerate(("policy", "images", "native")):
                with h5py.File(partial, "r+") as source:
                    if kind == "policy":
                        saved = source.attrs["model_policy"]
                        source.attrs["model_policy"] = "wrong-policy"
                    else:
                        path = "image_ids" if kind == "images" else "decoders/" + OWNER + "/full_native_ids"
                        saved = source[path][0]
                        source[path][0] = "wrong-image" if kind == "images" else 999
                try:
                    with h5py.File(partial, "r") as source, recovery.active.publication.AmendedStagedCollection(
                            root / ("header%d.h5" % index), **arguments) as writer:
                        with self.assertRaises(ValueError):
                            recovery.validate_header(source, writer)
                        self.assertFalse(writer._records)
                        self.assertNotIn("layers", writer._h5)
                finally:
                    with h5py.File(partial, "r+") as source:
                        if kind == "policy":
                            source.attrs["model_policy"] = saved
                        else:
                            source[path][0] = saved


if __name__ == "__main__":
    unittest.main()
