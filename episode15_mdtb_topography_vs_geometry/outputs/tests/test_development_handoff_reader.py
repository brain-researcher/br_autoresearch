"""Focused source-free EP15 released-handoff adapter checks; run on compute."""
from binascii import crc32
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import development_handoff_reader as reader
from inspect_registration_headers import nifti_header
from join_source_metadata import beta_member, validate_session
from build_anatomical_support import lattice_from_header
from test_anatomical_support import nifti_fixture
from test_task_a_mapping import float_nifti


def member_fixture():
    shared = set(reader.TASK_A_CONDITIONS[:13]) | {"rest"}
    summaries, rows = {}, []
    for pid in reader.DEVELOPMENT_IDS:
        for session, conditions, nreg in (
                ("ses-s1", reader.TASK_A_CONDITIONS, 46),
                ("ses-s2", tuple(sorted(shared | set(reader.CONDITIONS))), 49)):
            current = []
            for run in range(1, 17):
                labels = list(conditions)
                # Fixed total rows, but two genuine regressors for one A
                # condition in run1; this exposes pooled-regressor weighting.
                if session == "ses-s1" and run == 1:
                    labels.append(conditions[0])
                labels += ["instruction"] * (nreg - len(labels))
                counts = Counter(labels)
                for reg, condition in enumerate(labels):
                    instruction = condition == "instruction"
                    primary = not instruction and session == "ses-s2" and condition in reader.CONDITIONS
                    use = ("instruction_excluded" if instruction else "Task_A_calibration" if session == "ses-s1"
                           else "Task_B_primary" if primary else "Task_B_shared_postdecision_diagnostic")
                    count = 0 if instruction else counts[condition]
                    current.append({"participant_id": pid, "session": session,
                        "task_set": "A" if session == "ses-s1" else "B", "run": run, "reg_id": reg,
                        "task_name": condition, "cond_name": condition, "task_code": "instrct" if instruction else condition,
                        "cond_code": condition, "common": int(condition in shared), "instruction": int(instruction),
                        "half": 1 if run <= 8 else 2, "geometry_partition": (run - 1) // 4 + 1,
                        "beta_member": beta_member(pid, session, run, reg), "participant_role": "development",
                        "planned_handoff": "development", "endpoint_use": use,
                        "condition_map_included": int(not instruction), "regressors_in_condition_run": count,
                        "within_run_weight_numerator": int(not instruction), "within_run_weight_denominator": count or 1})
            summary = validate_session(current, Counter(row["beta_member"] for row in current), nreg * 16)
            if pid == reader.DEVELOPMENT_IDS[0]:
                summaries[session] = summary
            rows.extend(current)
    return rows, summaries


def release_fixture(rows):
    beta_bytes = float_nifti(np.arange(8).reshape(2, 2, 2), slope=2., intercept=-3.)
    beta_header = nifti_header(io.BytesIO(beta_bytes))
    raw = bytearray(nifti_fixture()[:352])
    struct.pack_into("<8h", raw, 40, 5, 141, 95, 87, 1, 3, 1, 1)
    struct.pack_into("<h", raw, 70, 16)
    struct.pack_into("<h", raw, 72, 32)
    struct.pack_into("<2f", raw, 112, 1., 0.)
    forward_bytes = bytes(raw)
    forward_header = nifti_header(io.BytesIO(forward_bytes))
    indices, world, shape, affine = lattice_from_header(forward_header)
    mask = np.zeros(shape, dtype=bool)
    mask.ravel()[:208] = True
    support = {"common_mask": mask, "source_transform_voxel_ijk": indices[:208].copy(),
               "world_xyz_mm": world[:208].copy(), "lattice_affine_mm": affine}
    bindings, archives = [], {}
    for pid in reader.DEVELOPMENT_IDS:
        headers = {suffix: deepcopy(forward_header if suffix == "space-SUIT_xfm.nii" else beta_header)
                   for suffix in reader.ANATOMY}
        bindings.append(reader.DevelopmentReleaseBinding(pid, Path("/released/development") / f"{pid}.zip",
                                                         headers, deepcopy(beta_header)))
        archives[pid] = FakeArchive(pid, [row for row in rows if row["participant_id"] == pid],
                                    beta_bytes, forward_bytes)
    return tuple(bindings), support, archives


class FakeArchive:
    """Real NIfTI beta bytes; forward payload is handled separately in checks.

Closed instruction/shared-B/anatomy members deliberately contain no valid
numeric image. Any attempt to open them raises, including header-only reads.
The forward field is a header-only stub with synthetic CRC metadata; these
checks do not claim to exercise ZIP decompression or CRC verification.
"""
    def __init__(self, pid, rows, beta_bytes, forward_bytes):
        self.pid, self.beta_bytes, self.forward_bytes = pid, beta_bytes, forward_bytes
        self.opened, self.reads, self.override = [], [], {}
        self.selected = {row["beta_member"] for row in rows
                         if row["endpoint_use"] in ("Task_A_calibration", "Task_B_primary")}
        self.forward = f"{pid}/anat/{pid}_space-SUIT_xfm.nii"
        names = [row["beta_member"] for row in rows]
        names += [f"{pid}/anat/{pid}_{suffix}" for suffix in reader.ANATOMY]
        self.infos = {}
        beta_crc = crc32(beta_bytes)
        for name in names:
            info = zipfile.ZipInfo(name)
            info.file_size = (352 + 141 * 95 * 87 * 3 * 4 if name == self.forward else len(beta_bytes))
            info.compress_size = info.file_size
            # ZipFile fills CRC from the central directory; ZipInfo() alone
            # leaves that slot unset. Preserve the real metadata-reader API.
            info.CRC = 0 if name == self.forward else beta_crc
            self.infos[name] = info

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def infolist(self):
        return list(self.infos.values())

    def getinfo(self, member):
        return self.infos[member]

    def open(self, member):
        name = member.filename if hasattr(member, "filename") else member
        if name not in self.selected and name != self.forward:
            raise AssertionError("closed development diagnostic/instruction/anatomy stream opened")
        self.opened.append(name)
        raw = self.override.get(name, self.forward_bytes if name == self.forward else self.beta_bytes)
        owner = self

        class Stream(io.BytesIO):
            def read(self, count=-1):
                owner.reads.append((name, count))
                return super().read(count)

        return Stream(raw)


class DevelopmentHandoffReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.summaries = member_fixture()
        cls.voxels = tuple(range(208))

    def test_complete_cohort_true_halves_quarters_and_equal_run_arithmetic(self):
        calls = []

        def sample(pid, row):
            self.assertIn(row["endpoint_use"], ("Task_A_calibration", "Task_B_primary"))
            self.assertFalse(row["instruction"])
            conditions = reader.TASK_A_CONDITIONS if row["session"] == "ses-s1" else reader.CONDITIONS
            base = 100000 * reader.DEVELOPMENT_IDS.index(pid) + 100 * conditions.index(row["cond_name"])
            offset = 0.
            if row["session"] == "ses-s1" and row["run"] == 1 and row["cond_name"] == conditions[0]:
                offset = -10. if row["reg_id"] == 0 else 10.
            calls.append((pid, row["beta_member"]))
            return base + row["run"] + offset + np.arange(208) / 1024

        result = reader.assemble_development_arrays(self.rows, condition_multiplicity_by_session=self.summaries,
                        voxel_ids=self.voxels, sample=sample, check=lambda: None)
        self.assertEqual(tuple(item.participant_id for item in result.participants), reader.DEVELOPMENT_IDS)
        self.assertEqual(len(calls), 12 * (29 * 16 + 1 + 18 * 16))
        for person, (record, packet, observation) in enumerate(zip(result.participants, result.task_a_packets, result.observations)):
            a_base = 100000 * person + 100 * np.arange(29)[:, None] + np.arange(208) / 1024
            b_base = 100000 * person + 100 * np.arange(18)[:, None] + np.arange(208) / 1024
            np.testing.assert_array_equal(packet.task_a_halves, np.stack((a_base + 4.5, a_base + 12.5)))
            np.testing.assert_array_equal(record.task_a_maps, a_base + 8.5)
            np.testing.assert_array_equal(observation.task_b_partitions,
                np.stack([b_base + midpoint for midpoint in (2.5, 6.5, 10.5, 14.5)]))
            self.assertEqual(record.task_b_condition_ids, reader.CONDITIONS)
            self.assertEqual(packet.half_run_ids, reader.TASK_A_HALF_RUNS)
            self.assertIs(record.task_a_maps, packet.task_a_maps)
            self.assertIs(record.task_b_partitions, observation.task_b_partitions)
            self.assertFalse(record.task_a_maps.flags.writeable)
        self.assertFalse(result.report["reader_isolation_established_by_adapter"])
        self.assertFalse(result.report["independent_run_errors_established"])

    def test_late_metadata_mismatch_or_missing_person_fails_before_sampling(self):
        selected_index = next(index for index in range(len(self.rows) - 1, -1, -1)
                              if self.rows[index]["endpoint_use"] == "Task_B_primary")
        for change in ({"planned_handoff": "sealed_audit"}, {"participant_id": "sub-02"},
                       {"half": 1}, {"cond_name": "rest"}, {"within_run_weight_denominator": 2}):
            rows = list(self.rows)
            rows[selected_index] = {**rows[selected_index], **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.assemble_development_arrays(rows, condition_multiplicity_by_session=self.summaries,
                    voxel_ids=self.voxels, sample=lambda *_: self.fail("sample before metadata complete"), check=lambda: None)
        with self.assertRaisesRegex(ValueError, "all twelve"):
            reader.assemble_development_arrays(self.rows[:-1520], condition_multiplicity_by_session=self.summaries,
                voxel_ids=self.voxels, sample=lambda *_: self.fail("missing participant read"), check=lambda: None)

    def test_actual_scaled_nifti_mapping_and_closed_streams_in_full_reader(self):
        bindings, support, archives = release_fixture(self.rows)
        for binding in bindings:
            for header in binding.anatomy_headers.values():
                header.pop("xyzt_units")
        original_headers = deepcopy([binding.anatomy_headers for binding in bindings])
        native = np.tile([10.5, 20.5, 30.5], (208, 1))
        native[-1] = [11, 21, 31]
        with patch.object(reader.zipfile, "ZipFile", side_effect=lambda path: archives[path.stem]), \
                patch.object(reader, "_native_coordinates", return_value=native) as pull:
            result = reader.read_development_handoffs(release_bindings=bindings, member_rows=self.rows,
                condition_multiplicity_by_session=self.summaries, common_support=support, check=lambda: None)
        self.assertEqual(pull.call_count, 12)
        expected = np.full(208, 4.)
        expected[-1] = 11.
        for record in result.participants:
            np.testing.assert_array_equal(record.task_a_maps, np.broadcast_to(expected, (29, 208)))
            np.testing.assert_array_equal(record.task_b_partitions, np.broadcast_to(expected, (4, 18, 208)))
        expected_ids = tuple(map(int, np.ravel_multi_index(support["source_transform_voxel_ijk"].T,
                                                          (141, 95, 87), order="C")))
        self.assertEqual(result.voxel_ids, expected_ids)
        self.assertEqual(result.qualified_forward_header["xyzt_units"] & 7, 2)
        self.assertEqual(result.qualified_forward_header,
                         nifti_header(io.BytesIO(archives[reader.DEVELOPMENT_IDS[0]].forward_bytes)))
        self.assertEqual(original_headers, [binding.anatomy_headers for binding in bindings])
        for archive in archives.values():
            self.assertEqual(set(archive.opened), archive.selected | {archive.forward})
            self.assertEqual(Counter(archive.opened)[archive.forward], 1)  # header; pull was separately patched
            self.assertTrue(all(Counter(archive.opened)[name] == 2 for name in archive.selected))
        # The real CLI must propagate the authenticated live header rather
        # than reusing the historical record that lacks spatial units.
        import run_development_point_panel as entrypoint
        import development_point_panel as point_panel
        inputs = dict(release_bindings=bindings, common_support=support)
        with patch.object(entrypoint, "_compute"), \
                patch.object(entrypoint, "load_released_inputs", return_value=inputs), \
                patch.object(reader, "read_development_handoffs", return_value=result), \
                patch.object(point_panel, "fit_development_point_panel", return_value=object()) as fit:
            entrypoint.run_released_development(Path("/explicit/fixture.json"),
                                                check=lambda: None, stage=lambda _: None)
        self.assertIs(fit.call_args.kwargs["forward_header"], result.qualified_forward_header)

    def test_every_header_is_checked_before_any_voxel_and_foreign_zip_is_refused(self):
        for problem in ("last_beta_grid", "last_beta_scaling", "last_beta_bitpix",
                        "last_beta_truncation", "last_forward_unknown_units",
                        "last_forward_nonmm_units", "foreign_member"):
            bindings, support, archives = release_fixture(self.rows)
            last = archives[reader.DEVELOPMENT_IDS[-1]]
            if problem in ("last_forward_unknown_units", "last_forward_nonmm_units"):
                raw = bytearray(last.forward_bytes)
                raw[123] = 0 if problem == "last_forward_unknown_units" else 1
                last.override[last.forward] = bytes(raw)
            elif problem != "foreign_member":
                raw = bytearray(last.beta_bytes)
                member = sorted(last.selected)[-1]
                if problem == "last_beta_grid":
                    struct.pack_into("<f", raw, 292, 999.)
                elif problem == "last_beta_scaling":
                    struct.pack_into("<2f", raw, 112, 2., float("inf"))
                elif problem == "last_beta_bitpix":
                    struct.pack_into("<h", raw, 72, 64)
                else:
                    last.infos[member].file_size -= 1
                last.override[member] = bytes(raw)
                last.infos[member].CRC = crc32(last.override[member])
            else:
                info = zipfile.ZipInfo("sub-02/func/ses-s2/closed_beta.nii")
                info.file_size = info.compress_size = 400
                info.CRC = 0  # Opaque foreign entry: no payload is supplied or opened.
                last.infos[info.filename] = info
            with self.subTest(problem=problem), patch.object(reader.zipfile, "ZipFile", side_effect=lambda path: archives[path.stem]), \
                    patch.object(reader, "_native_coordinates", side_effect=AssertionError("voxel decode before complete headers")), \
                    self.assertRaises(ValueError):
                reader.read_development_handoffs(release_bindings=bindings, member_rows=self.rows,
                    condition_multiplicity_by_session=self.summaries, common_support=support, check=lambda: None)
            self.assertTrue(all(count == 348 for archive in archives.values() for _, count in archive.reads))

    def test_fixed_support_release_bindings_and_absolute_scaled_pull_are_preserved(self):
        bindings, support, archives = release_fixture(self.rows)
        bad_support = {name: array.copy() for name, array in support.items()}
        bad_support["source_transform_voxel_ijk"][[0, 1]] = bad_support["source_transform_voxel_ijk"][[1, 0]]
        for supplied, support_value in ((bindings[:-1], support),
                ((replace(bindings[0], participant_id="sub-02"), *bindings[1:]), support)):
            with patch.object(reader.zipfile, "ZipFile", side_effect=AssertionError("archive opened before identity check")), \
                    self.assertRaises(ValueError):
                reader.read_development_handoffs(release_bindings=supplied, member_rows=self.rows,
                    condition_multiplicity_by_session=self.summaries, common_support=support_value, check=lambda: None)
        # Support geometry requires qualified live units, so it is checked
        # after header-only access but must still precede every voxel read.
        with patch.object(reader.zipfile, "ZipFile", side_effect=lambda path: archives[path.stem]), \
                patch.object(reader, "_native_coordinates", side_effect=AssertionError("support failure decoded voxels")), \
                self.assertRaises(ValueError):
            reader.read_development_handoffs(release_bindings=bindings, member_rows=self.rows,
                condition_multiplicity_by_session=self.summaries, common_support=bad_support, check=lambda: None)
        self.assertTrue(all(count == 348 for archive in archives.values() for _, count in archive.reads))
        indices = support["source_transform_voxel_ijk"]

        class ForwardValues:
            def __getitem__(inner, key):
                np.testing.assert_array_equal(np.stack(key[:3], axis=1), indices)
                self.assertEqual(key[3], 0)
                return np.tile([1., 2., 3.], (208, 1))

        with patch.object(reader, "read_image", return_value={"values": ForwardValues(), "slope": 2., "intercept": -3.}):
            world = reader._native_coordinates(object(), "explicit-forward", {}, indices)
        np.testing.assert_array_equal(world, np.tile([-1., 1., 3.], (208, 1)))

    def test_late_nonfinite_or_deadline_returns_no_partial_cohort(self):
        calls = []

        def sample(pid, row):
            calls.append(pid)
            result = np.arange(208, dtype=float)
            if pid == reader.DEVELOPMENT_IDS[-1] and row["session"] == "ses-s2":
                result[-1] = np.nan
            return result

        with self.assertRaisesRegex(ValueError, "finite fixed-support"):
            reader.assemble_development_arrays(self.rows, condition_multiplicity_by_session=self.summaries,
                        voxel_ids=self.voxels, sample=sample, check=lambda: None)
        self.assertEqual(set(calls), set(reader.DEVELOPMENT_IDS))
        calls.clear()

        def check():
            if calls:
                raise TimeoutError("explicit caller deadline")

        with self.assertRaisesRegex(TimeoutError, "caller deadline"):
            reader.assemble_development_arrays(self.rows, condition_multiplicity_by_session=self.summaries,
                voxel_ids=self.voxels, sample=lambda pid, row: (calls.append(pid) or np.zeros(208)), check=check)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
