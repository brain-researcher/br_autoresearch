"""Focused Task-A selection, mapping, arithmetic and disclosure fixtures."""
from collections import Counter
import copy
import io
import json
from pathlib import Path
import struct
import sys
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pilot_task_a_mapping as pilot
import join_source_metadata as metadata
from inspect_registration_headers import nifti_header
from test_anatomical_support import MemoryArchive, nifti_fixture
from test_source_metadata import design_fixture, encode_table


def plan_fixture():
    a = metadata.parse_reginfo(encode_table(design_fixture("ses-s1")), "sub-02", "ses-s1")
    b = metadata.parse_reginfo(encode_table(design_fixture("ses-s2")), "sub-02", "ses-s2")
    members = Counter(row["beta_member"] for row in a + b)
    expected = metadata.validate_session(a, members, 736)
    return a, b, members, expected


def float_nifti(values, slope=1., intercept=0.):
    raw = bytearray(nifti_fixture()[:352])
    struct.pack_into("<h", raw, 70, 16)
    struct.pack_into("<h", raw, 72, 32)
    struct.pack_into("<2f", raw, 112, slope, intercept)
    return bytes(raw) + np.asarray(values, dtype="<f4").tobytes(order="F")


class TaskAMappingTests(unittest.TestCase):
    def test_selects_complete_A_only_without_instructions_and_denies_other_streams(self):
        a, b, counts, expected = plan_fixture()
        selected = pilot.select_plan(a + b, "sub-02", expected, counts)
        self.assertEqual(len(selected), 29 * 16)
        self.assertTrue(all(row["task_set"] == "A" and not row["instruction"] for row in selected))
        self.assertEqual({row["run"] for row in selected}, set(range(1, 17)))
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr(selected[0]["beta_member"], b"A")
            archive.writestr(b[0]["beta_member"], b"B_CLOSED")
            archive.writestr(a[0]["beta_member"], b"INSTRUCTION_CLOSED")
        with zipfile.ZipFile(data) as archive:
            checked = pilot.ExactMemberArchive(archive, {row["beta_member"] for row in selected})
            with checked.open(checked.getinfo(selected[0]["beta_member"])) as stream:
                self.assertEqual(stream.read(1), b"A")
            for member in (b[0]["beta_member"], a[0]["beta_member"]):
                with self.assertRaisesRegex(ValueError, "allowlist"):
                    checked.open(archive.getinfo(member))
                with self.assertRaisesRegex(ValueError, "allowlist"):
                    checked.getinfo(member)
            self.assertEqual(checked.bytes_read, 1)
            self.assertEqual(len(checked.opened), 1)

    def test_wrong_member_task_set_missing_run_and_changed_multiplicity_fail(self):
        a, b, counts, expected = plan_fixture()
        for field, value in (("beta_member", b[17]["beta_member"]), ("task_set", "B"),
                             ("half", 2), ("instruction", 1)):
            changed = copy.deepcopy(a)
            changed[17][field] = value
            with self.assertRaisesRegex(ValueError, "identity"):
                pilot.select_plan(changed, "sub-02", expected, counts)
        with self.assertRaisesRegex(ValueError, "row count"):
            pilot.select_plan(a[:-1], "sub-02", expected, counts)
        changed = copy.deepcopy(a)
        changed[-1].update({key: changed[-2][key] for key in
                           ("task_name", "cond_name", "task_code", "cond_code", "common")})
        with self.assertRaisesRegex(ValueError, "condition lacks"):
            pilot.select_plan(changed, "sub-02", expected, counts)

    def test_duplicate_member_or_plan_regressor_cannot_count_as_complete(self):
        a, _, counts, expected = plan_fixture()
        changed = copy.deepcopy(a)
        changed[-1] = dict(changed[17])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pilot.select_plan(changed, "sub-02", expected, counts)
        counts[a[17]["beta_member"]] = 2
        with self.assertRaisesRegex(ValueError, "duplicate beta members"):
            pilot.select_plan(a, "sub-02", expected, counts)

    def test_live_grid_failure_precedes_voxel_payload_and_saved_mask_failure_precedes_open(self):
        payload = nifti_fixture()
        expected = nifti_header(io.BytesIO(payload))
        raw = bytearray(payload)
        struct.pack_into("<f", raw, 292, 11.)  # Live x-origin differs from saved A/mask.
        archive = MemoryArchive(bytes(raw))
        with self.assertRaisesRegex(ValueError, "header differs"):
            pilot.sample_beta(archive, "fixture.nii", expected, expected, [[10, 20, 30]])
        self.assertEqual(archive.read_sizes, [348])
        mask = copy.deepcopy(expected)
        mask["sform_rows"][0][3] = 12.
        archive = MemoryArchive(payload)
        with self.assertRaisesRegex(ValueError, "native cerebellar-mask grid"):
            pilot.sample_beta(archive, "fixture.nii", expected, mask, [[10, 20, 30]])
        self.assertEqual(archive.read_sizes, [])

    def test_real_scaling_and_strict_finite_stencil_are_preserved(self):
        values = np.arange(8).reshape(2, 2, 2).astype(float)
        payload = float_nifti(values, slope=2., intercept=-3.)
        header = nifti_header(io.BytesIO(payload))
        sampled, count = pilot.sample_beta(MemoryArchive(payload), "fixture.nii", header, header,
                                            [[10.5, 20.5, 30.5], [11, 21, 31]])
        np.testing.assert_array_equal(sampled, [4., 11.])
        self.assertEqual(count, 32)
        values[1, 1, 1] = np.nan
        payload = float_nifti(values)
        header = nifti_header(io.BytesIO(payload))
        sampled, _ = pilot.sample_beta(MemoryArchive(payload), "fixture.nii", header, header, [[10, 20, 30]])
        self.assertEqual(sampled[0], 0.)  # Zero-weight NaN corner is ignored.
        for coordinates in ([[10.5, 20.5, 30.5]], [[9.999, 20, 30]]):
            with self.assertRaisesRegex(ValueError, "no support shrink"):
                pilot.sample_beta(MemoryArchive(payload), "fixture.nii", header, header, coordinates)

    def test_equal_regressor_then_equal_run_averaging_not_pooled_regressors(self):
        plan, values = [], {}
        for run in range(1, 17):
            for condition in ("first", "second"):
                # Unequal multiplicities across runs expose incorrect pooled-beta weighting.
                for reg in range(2 if run == 1 else 1):
                    key = f"{run}-{condition}-{reg}"
                    plan.append({"run": run, "cond_name": condition, "beta_member": key})
                    values[key] = np.array([run, 2 * run, 3 * run], dtype=float) + (100 if condition == "second" else 0)
                    if run == 1:
                        values[key] += -10 if reg == 0 else 10
        halves = pilot.condition_halves(plan, ["first", "second"], lambda row: values[row["beta_member"]], points=3)
        np.testing.assert_array_equal(halves[0, 0], [4.5, 9., 13.5])
        np.testing.assert_array_equal(halves[1, 0], [12.5, 25., 37.5])
        np.testing.assert_array_equal(halves[:, 1], halves[:, 0] + 100)
        with self.assertRaisesRegex(ValueError, "lacks"):
            pilot.condition_halves(plan[:-1], ["first", "second"], lambda row: values[row["beta_member"]], points=3)

    def test_QC_exports_only_booleans_and_constant_map_does_not_shrink_support(self):
        halves = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
        report = pilot.half_map_flags(halves)
        self.assertTrue(report["all_half_maps_finite"])
        self.assertTrue(report["all_half_maps_nonconstant"])
        self.assertTrue(all(type(value) is bool for value in report.values()))
        self.assertFalse(report["half_maps_exported"])
        self.assertFalse(report["Task_A_reliability_computed"])
        self.assertEqual(json.loads(json.dumps(report)), report)
        halves[0, 1] = 3.
        self.assertFalse(pilot.half_map_flags(halves)["all_half_maps_nonconstant"])
        self.assertEqual(halves.shape, (2, 3, 4))
        halves[0, 1, 0] = np.nan
        self.assertFalse(pilot.half_map_flags(halves)["all_half_maps_finite"])

    def test_support_reordering_is_rejected(self):
        header = {"shape": [141, 95, 87, 1, 3], "sform_code": 2, "xyzt_units": 2,
                  "sform_rows": [[-1, 0, 0, 70], [0, 1, 0, -100], [0, 0, 1, -75]]}
        ijk, world, shape, affine = pilot.lattice_from_header(header)
        mask = np.zeros(shape, dtype=bool)
        mask.ravel()[:208] = True
        arrays = {"common_mask": mask, "source_transform_voxel_ijk": ijk[:208].copy(),
                  "world_xyz_mm": world[:208].copy(), "lattice_affine_mm": affine}
        np.testing.assert_array_equal(pilot.validate_support(arrays, header), ijk[:208])
        arrays["source_transform_voxel_ijk"][[0, 1]] = arrays["source_transform_voxel_ijk"][[1, 0]]
        with self.assertRaisesRegex(ValueError, "ordering"):
            pilot.validate_support(arrays, header)

    def test_participant_is_lexicographically_first_of_all24_matching_prerequisites(self):
        ids = [f"sub-{i:02d}" for i in range(2, 26)]
        people = [{"participant_id": pid} for pid in reversed(ids)]
        joined = {"participants": people, "metadata_join_complete": True,
                  "all_selected_Task_A_headers_equal_native_cerebellar_mask_grid": True}
        anatomy = {"participants": people, "forward_transform_headers_share_grid": True}
        support = {"participant_ids": ids, "anatomical_intersection_complete": True,
                   "participants_completed": 24, "common_support_points": 208, "Task_B_values_read": False}
        self.assertEqual(pilot.choose_participant(joined, anatomy, support), "sub-02")
        support["participant_ids"] = ids[:-1]
        with self.assertRaisesRegex(ValueError, "all24"):
            pilot.choose_participant(joined, anatomy, support)


if __name__ == "__main__":
    unittest.main()
