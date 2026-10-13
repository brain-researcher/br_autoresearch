"""Focused source-free geometry, scaling and terminal-state fixtures for EP17."""

import csv
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qualify_anatomical_axis as stage

nib, np = stage.nib, stage.np


def header(shape=(2, 2, 2), dtype=np.int16, sform=None, scode=2, qform=None, qcode=0,
           units="mm", slope=1, intercept=0):
    result = nib.Nifti1Header()
    result.set_data_shape(shape)
    result.set_data_dtype(dtype)
    result.set_sform(np.eye(4) if sform is None else sform, code=scode)
    result.set_qform(np.eye(4) if qform is None else qform, code=qcode)
    result.set_xyzt_units(units)
    result.set_slope_inter(slope, intercept)
    result["vox_offset"] = 352
    return result


def write_volume(path, data, hdr=None, compressed=False):
    hdr = header(data.shape, data.dtype) if hdr is None else hdr
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = hdr.binaryblock + b"\0" * 4 + data.astype(hdr.get_data_dtype()).tobytes(order="F")
    if compressed:
        payload = gzip.compress(payload, mtime=0)
    path.write_bytes(payload)
    return hdr, payload


def old_header(hdr):
    return {"shape": list(hdr.get_data_shape()), "datatype_code": int(hdr["datatype"]),
            "pixdim": hdr["pixdim"].tolist(), "qform_code": int(hdr["qform_code"]),
            "sform_code": int(hdr["sform_code"]),
            "quaternion_and_offset": [float(hdr[x]) for x in
                ("quatern_b", "quatern_c", "quatern_d", "qoffset_x", "qoffset_y", "qoffset_z")],
            "srow_xyz": [hdr[name].tolist() for name in ("srow_x", "srow_y", "srow_z")]}


class VolumeTests(unittest.TestCase):
    def test_file_storage_F_order_endianness_scaling_and_compression(self):
        data = np.arange(12, dtype=np.int16).reshape((2, 3, 2))
        hdr = header(data.shape, data.dtype, slope=2, intercept=1)
        with tempfile.TemporaryDirectory() as directory:
            for endian in ("<", ">"):
                encoded_header = hdr if hdr.endianness == endian else hdr.as_byteswapped()
                for compressed in (False, True):
                    with self.subTest(endian=endian, compressed=compressed):
                        path = Path(directory) / ("fixture.nii.gz" if compressed else "fixture.nii")
                        write_volume(path, data, encoded_header, compressed)
                        got_header, values = stage.read_volume(path, compressed, 100, time.monotonic() + 5)
                        np.testing.assert_array_equal(values, data * 2 + 1)
                        self.assertEqual(got_header.get_data_shape(), (2, 3, 2))

    def test_decode_bound_refuses_before_missing_payload_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.nii"
            path.write_bytes(header((20, 20, 20)).binaryblock)
            with self.assertRaisesRegex(stage.Refusal, "structural_volume_decode_bound"):
                stage.read_volume(path, False, 100, time.monotonic() + 5)

    def test_fractional_nonfinite_masks_labels_and_column_mismatch_refuse(self):
        good = np.ones((2, 2, 2))
        for mask, atlas, count, message in (
            (good / 2, good, 8, "mask_not_binary_finite"),
            (good, good / 2, 8, "atlas_not_finite_integer_labels"),
            (good, good * np.nan, 8, "atlas_not_finite_integer_labels"),
            (good, good, 7, "mask_BCD_column_count_mismatch"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(stage.Refusal, message):
                stage.validate_values(mask, atlas, count)

    def test_existing_union_header_change_is_not_accepted_from_equal_shape(self):
        before = header()
        translated = np.eye(4)
        translated[0, 3] = 2
        after = header(sform=translated)
        with self.assertRaisesRegex(stage.Refusal, "original_union_geometry_changed"):
            stage.verify_mask_header(after, old_header(before))

    def test_atlas_annex_identity_checks_exact_length_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas"
            path.write_bytes(b"fixed source object")
            expected = hashlib.md5(path.read_bytes()).hexdigest()
            deadline = time.monotonic() + 5
            stage.verify_atlas_object(path, 19, expected, deadline)
            with self.assertRaisesRegex(stage.Refusal, "atlas_annex_length_mismatch"):
                stage.verify_atlas_object(path, 18, expected, deadline)
            with self.assertRaisesRegex(stage.Refusal, "atlas_annex_identity_mismatch"):
                stage.verify_atlas_object(path, 19, "0" * 32, deadline)


class AffineTests(unittest.TestCase):
    def test_agreeing_same_frame_prefers_sform(self):
        hdr = header(scode=1, qcode=1)
        affine, info = stage.select_affine(hdr, stage.native_frame_authority(hdr))
        np.testing.assert_array_equal(affine, np.eye(4))
        self.assertEqual(info["selected_form"], "sform")
        self.assertEqual(info["eligible_form_corner_difference_mm"], 0)

    def test_authenticated_qform_wins_over_other_frame_sform(self):
        unrelated = np.eye(4)
        unrelated[0, 3] = 100
        hdr = header(sform=unrelated, scode=4, qcode=1)
        affine, info = stage.select_affine(hdr, {"qform": "T1w", "sform": "MNI"})
        np.testing.assert_array_equal(affine, np.eye(4))
        self.assertEqual(info["selected_form"], "qform")
        # This helper evidence is not fabricated by the production source policy.
        with self.assertRaisesRegex(stage.Refusal, "native_T1w_form_unresolved"):
            stage.native_frame_authority(hdr)

    def test_conflict_unknown_frame_or_invalid_active_affine_never_falls_back(self):
        translated = np.eye(4)
        translated[0, 3] = 0.01
        with self.assertRaisesRegex(stage.Refusal, "authenticated_affines_disagree"):
            stage.select_affine(header(sform=translated, scode=1, qcode=1), {"qform": "T1w", "sform": "T1w"})
        with self.assertRaisesRegex(stage.Refusal, "different_form_frames_need_source_evidence"):
            stage.native_frame_authority(header(scode=2, qcode=1))
        with self.assertRaisesRegex(stage.Refusal, "native_T1w_form_unresolved"):
            stage.native_frame_authority(header(scode=0, qcode=0))
        singular = np.eye(4)
        singular[0, 0] = 0
        with self.assertRaisesRegex(stage.Refusal, "invalid_coded_affine"):
            stage.select_affine(header(sform=singular, scode=1, qcode=1), {"qform": "T1w", "sform": "T1w"})

    def test_units_are_converted_exactly_from_declared_unit_and_unknown_refuses(self):
        affine, info = stage.select_affine(header(units="meter"), {"sform": "T1w"})
        np.testing.assert_array_equal(affine[:3, :3], np.eye(3) * 1000)
        self.assertEqual(info["unit_to_mm"], 1000)
        with self.assertRaisesRegex(stage.Refusal, "spatial_units_unresolved"):
            stage.select_affine(header(units="unknown"), {"sform": "T1w"})


class MappingTests(unittest.TestCase):
    def test_asymmetric_mask_keeps_C_column_order_and_nonROI_columns(self):
        mask = np.zeros((2, 3, 2), dtype=bool)
        coordinates = [(0, 0, 1), (0, 2, 0), (1, 0, 0), (1, 2, 1)]
        labels = [1007, 2007, 42, 3007]
        atlas = np.zeros(mask.shape, dtype=np.int32)
        for coordinate, number in zip(coordinates, labels):
            mask[coordinate], atlas[coordinate] = True, number
        affine = np.diag([-2., 3., 4., 1.])
        affine[:3, 3] = [10, -2, 5]
        mapped = stage.map_columns(mask, atlas, affine, affine)
        np.testing.assert_array_equal(mapped[0], [1, 4, 6, 11])
        np.testing.assert_array_equal(mapped[1], coordinates)
        np.testing.assert_array_equal(mapped[3], labels)
        np.testing.assert_array_equal(mapped[2], np.asarray(coordinates) @ affine[:3, :3].T + affine[:3, 3])
        stream = io.BytesIO()
        table = stage.BoundedTable(stream)
        table.write([], header=True)
        stage.write_columns(table, "sub-01", mapped, time.monotonic() + 5)
        rows = list(csv.DictReader(io.StringIO(stream.getvalue().decode())))
        self.assertEqual([r["model_column_zero_based"] for r in rows], ["0", "1", "2", "3"])
        self.assertEqual([r["nominated_hemisphere"] for r in rows], ["left", "right", "", ""])

    def test_nearest_neighbor_ties_outside_FOV_and_no_fractional_labels(self):
        mask = np.ones((7, 1, 1), dtype=bool)
        atlas = np.array([1007, 2007, 42], dtype=np.int32).reshape((3, 1, 1))
        affine = np.eye(4)
        affine[0, 0], affine[0, 3] = 0.5, -0.5
        mapped = stage.map_columns(mask, atlas, affine, np.eye(4))
        np.testing.assert_array_equal(mapped[3], [0, 1007, 2007, 2007, 42, 42, 0])
        np.testing.assert_array_equal(mapped[4], [True, False, False, False, False, False, True])

    def test_empty_singleton_refusal_and_two_voxels_do_not_claim_adequacy(self):
        atlas = np.array([1007, 2007, 2007], dtype=np.int32)
        result = stage.roi_summary(atlas, atlas, np.zeros((3, 3)))
        self.assertEqual(result["left"]["refusal"], "single_voxel_cannot_change_support")
        self.assertIsNone(result["right"]["refusal"])
        self.assertFalse(result["right"]["scientific_support_adequacy_verified"])
        result = stage.roi_summary(atlas, np.zeros(3), np.zeros((3, 3)))
        self.assertEqual(result["right"]["refusal"], "empty_nominated_ROI")
        self.assertIsNone(result["right"]["world_min_mm"])

    def test_output_byte_cap_does_not_write_the_overflowing_block(self):
        stream = io.BytesIO()
        table = stage.BoundedTable(stream, cap=4)
        table.write([("a",)])
        before = stream.getvalue()
        with self.assertRaisesRegex(stage.Refusal, "uncompressed_table_byte_cap"):
            table.write([("long row",)])
        self.assertEqual(stream.getvalue(), before)
        self.assertEqual(table.rows, 1)


def unit_report(hdr):
    interpreted = hdr.copy()
    interpreted.set_xyzt_units("mm")
    _, geometry = stage.select_affine(interpreted, stage.native_frame_authority(interpreted))
    reconstructed = copy.deepcopy(geometry)
    reconstructed["reconstructed_units_for_comparison"] = reconstructed.pop("original_spatial_units")
    participants, accesses = [], []
    for subject in stage.SUBJECTS:
        size, digest = stage.UNIT_WITNESS_OBJECTS[subject]
        participants.append({"participant": subject, "bytes": size, "md5": digest,
                             "annex_key": f"MD5E-s{size}--{digest}.nii.gz",
                             "source_path": f"{subject}/ses-01/func/{subject}_ses-01_task-things_run-1_part-mag_space-T1w_desc-brain_mask.nii.gz",
                             "witness_geometry": copy.deepcopy(geometry),
                             "recorded_union_numeric_geometry": copy.deepcopy(reconstructed),
                             "inherited_spatial_units": "mm", "union_header_rewritten": False,
                             "union_numeric_geometry_unit_label_is_witness_derived": True,
                             "maximum_corner_difference_mm": 0.0})
        accesses.append({"participant": subject, "verified": True, "compressed_object_bytes_read": size})
    return {"analysis_id": "anatomical_axis", "mode": "unit_witness", "status": "completed",
            "anatomical_spatial_units_witness_complete": True, "errors": [],
            "source_metadata": str(stage.METADATA),
            "fmriprep_commit": "5155ba0d43836a4451141b90b5a70507f41205cf",
            "dataset_id": "81d8c126-b10f-4e8b-9eab-6825964c8552",
            "network_requests": 4, "transport_bytes_read": 28069,
            "automatic_retries": 0, "redirects_followed": 0,
            "decoded_bytes_per_object_maximum": 352, "structural_voxel_arrays_decoded": False,
            "participants": participants, "accesses": accesses}


class UnitWitnessIntegrationTests(unittest.TestCase):
    def test_unknown_units_use_complete_witness_without_mutating_source_header(self):
        hdr = header(units="unknown")
        original = hdr.binaryblock
        evidence = unit_report(hdr)
        cache = {}
        with patch.object(stage, "load_unit_witnesses", return_value={p["participant"]: p for p in evidence["participants"]}) as load:
            for subject in stage.SUBJECTS:
                affine, geometry = stage.select_union_affine(hdr, subject, old_header(hdr), cache)
                np.testing.assert_array_equal(affine, np.eye(4))
                self.assertEqual(geometry["original_spatial_units"], "unknown")
                self.assertEqual(geometry["interpreted_spatial_units"], "mm")
                self.assertTrue(geometry["unit_witness_used"])
                self.assertFalse(geometry["source_header_rewritten"])
        self.assertEqual(load.call_count, 1)
        self.assertEqual(hdr.binaryblock, original)

    def test_explicit_units_are_never_overridden_by_mm_witness(self):
        with patch.object(stage, "load_unit_witnesses") as load:
            for units, factor in (("mm", 1), ("meter", 1000), ("micron", .001)):
                hdr = header(units=units)
                affine, geometry = stage.select_union_affine(hdr, "sub-01", old_header(hdr), {})
                np.testing.assert_array_equal(affine[:3, :3], np.eye(3) * factor)
                self.assertEqual(geometry["original_spatial_units"], units)
                self.assertFalse(geometry["unit_witness_used"])
            load.assert_not_called()

    def test_witness_completion_binding_roster_and_identity_are_required(self):
        good = unit_report(header(units="unknown"))
        cases = []
        for key, value in (("status", "stopped"), ("anatomical_spatial_units_witness_complete", False),
                           ("source_metadata", "other.json"), ("fmriprep_commit", "other"),
                           ("transport_bytes_read", 28068), ("structural_voxel_arrays_decoded", True)):
            changed = copy.deepcopy(good)
            changed[key] = value
            cases.append(changed)
        changed = copy.deepcopy(good)
        changed["participants"] = changed["participants"][:-1]
        cases.append(changed)
        changed = copy.deepcopy(good)
        changed["participants"][0]["md5"] = "0" * 32
        cases.append(changed)
        changed = copy.deepcopy(good)
        changed["accesses"][0]["verified"] = False
        cases.append(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "witness.json"
            with patch.object(stage, "UNIT_WITNESS", path):
                path.write_text(json.dumps(good))
                self.assertEqual(list(stage.load_unit_witnesses()), list(stage.SUBJECTS))
                for index, changed in enumerate(cases):
                    path.write_text(json.dumps(changed))
                    with self.subTest(case=index), self.assertRaises(stage.Refusal):
                        stage.load_unit_witnesses()

    def test_missing_witness_does_not_guess_mm(self):
        hdr = header(units="unknown")
        with patch.object(stage, "load_unit_witnesses", side_effect=FileNotFoundError("fixture")):
            with self.assertRaises(FileNotFoundError):
                stage.select_union_affine(hdr, "sub-01", old_header(hdr), {})
        self.assertEqual(hdr.get_xyzt_units()[0], "unknown")

    def test_current_geometry_and_recorded_witness_affine_must_both_match(self):
        original = header(units="unknown")
        evidence = unit_report(original)
        cache = {p["participant"]: p for p in evidence["participants"]}
        shifted = np.eye(4)
        shifted[0, 3] = 1
        changed = header(units="unknown", sform=shifted)
        with self.assertRaisesRegex(stage.Refusal, "unit_witness_current_union_geometry_changed"):
            stage.select_union_affine(changed, "sub-01", old_header(changed), cache)
        cache["sub-01"]["witness_geometry"]["selected_affine_mm"][0][3] = 1
        with self.assertRaisesRegex(stage.Refusal, "unit_witness_current_affine_mismatch"):
            stage.select_union_affine(original, "sub-01", old_header(original), cache)


class AttemptTests(unittest.TestCase):
    def make_sources(self, directory, empty_right=False):
        source = directory / "source"
        participants, identities = [], {}
        for subject in stage.SUBJECTS:
            mask = np.ones((2, 2, 2), dtype=np.int8)
            atlas = np.array([1007, 1007, 1007, 2007, 2007, 2007, 42, 3007], dtype=np.int16).reshape((2, 2, 2))
            if empty_right:
                atlas[atlas == 2007] = 42
            mask_path = source / f"THINGS/glmsingle/{subject}/glmsingle/input/{subject}_task-things_space-T1w_label-brain_desc-union_mask.nii"
            atlas_path = source / f"anatomical/smriprep/{subject}/anat/{subject}_desc-aparcaseg_dseg.nii.gz"
            hdr, _ = write_volume(mask_path, mask)
            _, payload = write_volume(atlas_path, atlas, compressed=True)
            identities[subject] = (len(payload), hashlib.md5(payload).hexdigest())
            axes = {name: {"stored_shape": [stage.PRESENTATIONS[subject], 1, 1, 8],
                          "file": f"THINGS/glmsingle/{subject}/glmsingle/output/T1w/{filename}",
                          "dataset": "modelmd", "dtype": "float32", "trial_axis_matches_design_count": True}
                    for name, filename in stage.STAGES.items()}
            participants.append({"participant": subject, "presentations": stage.PRESENTATIONS[subject],
                                 "B_C_D_stored_shapes_equal": True, "structural_issues": [],
                                 "stage_axes": axes, "mask_headers": {"union": old_header(hdr)}})
        metadata = directory / "metadata.json"
        metadata.write_text(json.dumps({"analysis_id": "source_metadata", "source": str(source),
                                        "structural_errors": [], "participants": participants}))
        return source, metadata, identities

    def run_fixture(self, directory, empty_right=False, post_completion_failure=False):
        source, metadata, identities = self.make_sources(directory, empty_right)
        out = directory / "attempt"
        out.mkdir()
        original_complete = stage.complete

        def transition(report, deadline):
            original_complete(report, deadline)
            if post_completion_failure:
                raise stage.Refusal("attempt_deadline")

        with patch.object(stage, "SOURCE", source), patch.object(stage, "METADATA", metadata), \
             patch.object(stage, "ATLAS", identities), patch.object(stage, "complete", transition):
            code = stage.run_attempt(out)
        report = json.loads((out / "anatomical_axis_report.json").read_text())
        with gzip.open(out / "voxel_axis.csv.gz", "rt", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return code, report, rows

    def test_all_four_axis_maps_complete_with_neural_files_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, rows = self.run_fixture(Path(directory))
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(report[key] for key in stage.COMPLETION_FLAGS))
        self.assertEqual(len(report["participants"]), 4)
        self.assertEqual([r["model_column_zero_based"] for r in rows[:8]], list(map(str, range(8))))
        self.assertFalse(report["neural_values_read"])
        self.assertFalse(report["finite_response_support_complete"])
        self.assertFalse(report["response_units_verified"])
        self.assertFalse(report["source_roles_complete"])

    def test_empty_ROI_reports_all_four_refusals_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, rows = self.run_fixture(Path(directory), empty_right=True)
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "refused")
        self.assertEqual(len(report["refusals"]), 4)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(not report[key] for key in stage.COMPLETION_FLAGS))

    def test_deadline_after_completion_transition_clears_both_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, rows = self.run_fixture(Path(directory), post_completion_failure=True)
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(not report[key] for key in stage.COMPLETION_FLAGS))
        self.assertEqual(report["errors"], [{"code": "attempt_deadline"}])

    def test_metadata_wrong_modelmd_axes_stop_before_structural_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, identities = self.make_sources(root)
            content = json.loads(metadata.read_text())
            content["participants"][0]["stage_axes"]["C"]["stored_shape"] = [8, 1, 1, 12780]
            metadata.write_text(json.dumps(content))
            out = root / "attempt"
            out.mkdir()
            with patch.object(stage, "SOURCE", source), patch.object(stage, "METADATA", metadata), \
                 patch.object(stage, "ATLAS", identities), patch.object(stage, "read_volume") as reader:
                code = stage.run_attempt(out)
            self.assertEqual(code, 2)
            reader.assert_not_called()
            report = json.loads((out / "anatomical_axis_report.json").read_text())
            self.assertTrue(all(not report[key] for key in stage.COMPLETION_FLAGS))

    def test_full_four_participant_unknown_unit_mapping_consumes_one_saved_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, identities = self.make_sources(root)
            for subject in stage.SUBJECTS:
                path = source / f"THINGS/glmsingle/{subject}/glmsingle/input/{subject}_task-things_space-T1w_label-brain_desc-union_mask.nii"
                body = path.read_bytes()
                hdr = nib.Nifti1Header(binaryblock=body[:348], check=False)
                hdr.set_xyzt_units("unknown")
                path.write_bytes(hdr.binaryblock + body[348:])
            out = root / "attempt"
            out.mkdir()
            witness = root / "witness.json"
            with patch.object(stage, "SOURCE", source), patch.object(stage, "METADATA", metadata), \
                 patch.object(stage, "ATLAS", identities), patch.object(stage, "UNIT_WITNESS", witness):
                witness.write_text(json.dumps(unit_report(hdr)))
                with patch.object(stage, "load_unit_witnesses", wraps=stage.load_unit_witnesses) as load:
                    code = stage.run_attempt(out)
                self.assertEqual(load.call_count, 1)
            report = json.loads((out / "anatomical_axis_report.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(len(report["participants"]), 4)
            self.assertTrue(all(p["mask_geometry"]["unit_witness_used"] for p in report["participants"]))
            self.assertTrue(all(p["mask_geometry"]["original_spatial_units"] == "unknown" for p in report["participants"]))
            self.assertFalse(report["neural_values_read"])
            self.assertEqual(report["network_requests"], 0)


if __name__ == "__main__":
    unittest.main()
