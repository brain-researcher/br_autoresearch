"""Source-free fixtures; execute only in the coordinator's Slurm allocation."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import materialize_neural_roles as method


def metadata_fixture():
    concepts, images, original, chronological, assigned = [], [], [], [], []
    index = 0
    for cid, role in enumerate(method.ROLES, 1):
        labels = {"role": role, "outer_fold": "fold01" if role == "development" else "",
                  "development_block": "block01" if role == "development" else "",
                  "audit_block": "block01" if role == "sealed_audit" else ""}
        concepts.append({"things_category_nr": str(cid), "uniqueID": f"concept{cid}", "provider_category_nr": str(cid + 10),
                         "original_common_image_count": "5", "qualified_image_count": "4" if cid == 1 else "5", **labels})
        for exemplar in range(5):
            eligible = not (cid == 1 and exemplar == 0)
            image = {field: "" for field in method.stage.IMAGE_FIELDS}
            image.update({"provider_image_path": f"concept{cid}/image{exemplar}.jpg", "image_name": f"image{exemplar}.jpg",
                          "concept_name": f"concept{cid}", "things_category_nr": str(cid), "provider_category_nr": str(cid + 10),
                          "provider_exemplar_nr": str(exemplar), "condition": "exp", "common_to_four": "True",
                          "eligible_image": str(eligible), "failure_reasons": "" if eligible else "fixture_ineligible",
                          "retained_count_sub-01": "2", **labels})
            images.append(image)
            for repeat in range(2):
                row = {field: "" for field in (*method.core.OUTPUT_FIELDS, "concept_name", "condition")}
                row.update({"participant": "sub-01", "glmsingle_trial_index_zero_based": str(index),
                            "expected_provider_image_path": image["provider_image_path"], "source_image_basename": image["image_name"],
                            "things_category_nr": str(cid), "concept_name": image["concept_name"],
                            "provider_category_nr": image["provider_category_nr"], "provider_exemplar_nr": str(exemplar),
                            "condition": "exp", "session": "1", "run": "1", "run_order": "0", "trial_in_run_zero_based": str(index)})
                original.append(row)
                history = {**row, **{field: "" for field in method.chronology.ADDED_FIELDS},
                           "eligible_image": str(eligible), "repeat_half": str(repeat + 1) if eligible else "",
                           "retained_repeat_rank": str(repeat), "ordinal_status": "valid", "retained_acquisition_rank": str(29 - index)}
                chronological.append(history)
                assigned.append({**history, **labels})
                index += 1
    return concepts, images, original, chronological, assigned


def routing(fixture):
    concepts, images, original, history, rows = fixture
    cm = method.check_concepts(concepts, capacities=(1, 1, 1), fold_size=1, development_size=1, audit_size=1)
    qualified = [{field: image[field] for field in method.stage.IMAGE_FIELDS} for image in images]
    return method.check_routing(cm, images, qualified, rows, history, original, {"sub-01": 30})


def copy_rows(n):
    return [{"participant": "sub-01", "role": method.ROLES[i % 3], "glmsingle_trial_index_zero_based": str(i),
             "eligible_image": "False" if i == 0 else "True", "repeat_half": "" if i == 0 else "1"} for i in range(n)]


class NeuralRoleMaterializationTests(unittest.TestCase):
    def test_all_rows_including_ineligible_follow_source_index_not_chronology_or_csv(self):
        fixture = metadata_fixture()
        for rows in fixture[2:]:
            rows.reverse()
        result = routing(fixture)["sub-01"]
        self.assertEqual([int(row["glmsingle_trial_index_zero_based"]) for row in result], list(range(30)))
        self.assertEqual([int(row["retained_acquisition_rank"]) for row in result], list(reversed(range(30))))
        self.assertEqual(sum(row["role"] == "sealed_audit" for row in result), 10)
        self.assertEqual([row["repeat_half"] for row in result[:2]], ["", ""])
        self.assertEqual(result[0]["failure_reasons"], "fixture_ineligible")

    def test_conflicts_missing_and_duplicate_rows_or_changed_halves_refused(self):
        mutations = (
            lambda f: f[4][0].update(role="sealed_audit"),
            lambda f: f[4].append(deepcopy(f[4][0])),
            lambda f: f[4].pop(),
            lambda f: f[4][0].update(repeat_half="1"),
            lambda f: f[1][0].update(concept_name="another_concept"),
            lambda f: f[4][1].update(glmsingle_trial_index_zero_based="100"),
        )
        for mutate in mutations:
            fixture = metadata_fixture()
            mutate(fixture)
            with self.subTest(mutation=mutate), self.assertRaises(method.core.Refusal):
                routing(fixture)

    def test_concept_capacity_and_fold_constraints(self):
        concepts = metadata_fixture()[0]
        method.check_concepts(concepts, (1, 1, 1), 1, 1, 1)
        for field, value in (("outer_fold", "fold02"), ("audit_block", "block01"), ("role", "support_calibration")):
            bad = deepcopy(concepts)
            bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(method.core.Refusal):
                method.check_concepts(bad, (1, 1, 1), 1, 1, 1)

    def test_capacity_needs_explicit_quota_current_job_and_reservation_headroom(self):
        now = datetime(2026, 10, 4, 22, 40, tzinfo=timezone.utc)
        record = {"parent": "/tmp/cache", "slurm_job_id": "123", "observed_at_utc": "2026-10-04T22:39:00Z",
                  "quota_checked": True, "filesystem_available_bytes": 2 * method.BYTE_CAP,
                  "quota_available_bytes": 2 * method.BYTE_CAP, "concurrent_reserved_bytes": method.BYTE_CAP}
        method.check_capacity(record, Path("/tmp/cache"), "123", now)
        for change in ({"quota_available_bytes": None}, {"quota_checked": False}, {"slurm_job_id": "122"},
                       {"quota_available_bytes": 2 * method.BYTE_CAP - 1}, {"filesystem_available_bytes": 0},
                       {"observed_at_utc": "2026-10-04T22:41:00Z"}, {"concurrent_reserved_bytes": True}):
            with self.subTest(change=change), self.assertRaises(method.core.Refusal):
                method.check_capacity({**record, **change}, Path("/tmp/cache"), "123", now)

    def test_chunk_bound_counts_edge_padding_and_conservative_row_extent(self):
        result = method.layout_bound((100, 1, 1, 7), (40, 1, 1, 4), 4)
        self.assertEqual(result["intersecting_chunks"], 4)
        self.assertEqual(result["decoded_work_bound_bytes"], 4 * 40 * 4 * 4)
        self.assertEqual(result["decoded_row_stop_exclusive"], 80)
        contiguous = method.layout_bound((100, 1, 1, 7), None, 4)
        self.assertEqual(contiguous["decoded_work_bound_bytes"], 64 * 7 * 4)
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "huge_chunk.h5", "w") as stream:
                shape = (100, 1, 1, 1000000)
                stream.create_dataset("modelmd", shape=shape, dtype="f4", chunks=shape)
                with self.assertRaises(method.core.Refusal):
                    method.checked_dataset(stream, shape)

    def test_hdf5_soft_external_virtual_and_raw_external_storage_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with h5py.File(path / "source.h5", "w") as stream:
                stream.create_dataset("data", shape=(4, 1, 1, 2), dtype="f4")
            for kind in ("soft", "external_link", "virtual", "external_storage"):
                with self.subTest(kind=kind), h5py.File(path / f"{kind}.h5", "w") as stream:
                    if kind == "soft":
                        stream.create_dataset("other", shape=(4, 1, 1, 2), dtype="f4")
                        stream["modelmd"] = h5py.SoftLink("/other")
                    elif kind == "external_link":
                        stream["modelmd"] = h5py.ExternalLink(str(path / "source.h5"), "/data")
                    elif kind == "virtual":
                        layout = h5py.VirtualLayout(shape=(4, 1, 1, 2), dtype="f4")
                        layout[...] = h5py.VirtualSource(str(path / "source.h5"), "/data", shape=(4, 1, 1, 2))
                        stream.create_virtual_dataset("modelmd", layout)
                    else:
                        stream.create_dataset("modelmd", shape=(4, 1, 1, 2), dtype="f4",
                                              external=[(str(path / "raw.bin"), 0, h5py.h5f.UNLIMITED)])
                    with self.assertRaises(method.core.Refusal):
                        method.checked_dataset(stream, (4, 1, 1, 2), stop=2)

    def test_literal_float_bits_axes_endian_roles_permissions_and_no_value_export(self):
        patterns = [0x00000000, 0x80000000, 0x7FC01234, 0x7F800000, 0xFF800000]
        for endian in ("<", ">"):
            with self.subTest(endian=endian), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                private = parent / "private"
                method.private_directory(private, parent)
                raw = np.array(patterns * 9, dtype=endian + "u4").view(endian + "f4").reshape(9, 1, 1, 5)
                with h5py.File(parent / "source.h5", "w") as source:
                    source.create_dataset("modelmd", data=raw, dtype=raw.dtype, chunks=(4, 1, 1, 3))
                    dataset, layout = method.checked_dataset(source, raw.shape, stop=6)
                    report = method.initial_report()
                    method.copy_block(dataset, copy_rows(9), private, layout, report, lambda: None, stop=6)
                self.assertTrue(report["all_copy_bytes_equal"])
                self.assertEqual(report["logical_source_payload_bytes"], 6 * 5 * 4)
                self.assertEqual(report["decoded_scope_audit_row_count"], 2)
                for role in method.ROLES:
                    filename = private / role / "sub-01/B.h5"
                    self.assertEqual(stat.S_IMODE(filename.stat().st_mode), 0o600)
                    self.assertEqual(stat.S_IMODE(filename.parent.stat().st_mode), 0o700)
                    with h5py.File(filename, "r") as output:
                        data = output["modelmd"][...]
                        indices = [i for i in range(6) if method.ROLES[i % 3] == role]
                        self.assertEqual(data.dtype.str, raw.dtype.str)
                        self.assertTrue(np.array_equal(data.view("u1"), raw[indices].view("u1")))
                        self.assertFalse(output["modelmd"].external)
                        self.assertFalse(output["modelmd"].is_virtual)
                    self.assertFalse(filename.with_suffix(".h5.partial").exists())
                encoded = json.dumps(report)
                for forbidden in ("7fc01234", "finite_count", "nan_count", "sha256", "correlation", "sample_values"):
                    self.assertNotIn(forbidden, encoded.lower())
                self.assertFalse(report["reader_handoffs_activated"])
                self.assertFalse(report["physical_role_isolation_verified"])

    def test_chunk_only_audit_exposure_and_failed_read_are_truthfully_recorded(self):
        class FailingDataset:
            def __getitem__(self, _selection):
                raise OSError("fixture I/O failure")
        rows = copy_rows(9)
        layout = method.layout_bound((9, 1, 1, 5), (4, 1, 1, 5), 4, stop=2)
        report = method.initial_report()
        with self.assertRaises(OSError):
            method.copy_block(FailingDataset(), rows, Path("unused"), layout, report, lambda: None, stop=2)
        self.assertEqual(report["logical_audit_row_count"], 0)
        self.assertEqual(report["decoded_scope_audit_row_count"], 1)
        self.assertTrue(report["neural_values_read"])
        self.assertTrue(report["raw_audit_bytes_accessed"])
        self.assertIsNotNone(report["first_raw_audit_access_utc"])
        self.assertFalse(report["neural_read_completed"])
        self.assertFalse(report["pilot_copy_complete"])
        self.assertEqual(report["scientific_audit_evaluation_count"], 0)

    def test_empty_role_and_exclusive_publication_preserve_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            private = parent / "private"
            method.private_directory(private, parent)
            with h5py.File(parent / "source.h5", "w") as source:
                data = source.create_dataset("modelmd", data=np.zeros((4, 1, 1, 2), dtype="f4"))
                _, layout = method.checked_dataset(source, data.shape, stop=2)
                report = method.initial_report()
                method.copy_block(data, copy_rows(4), private, layout, report, lambda: None, stop=2)
            audit = next(row for row in report["outputs"] if row["role"] == "sealed_audit")
            self.assertTrue(audit["empty_role"])
            self.assertIsNone(audit["array_file"])
            self.assertTrue((private / "sealed_audit/source_index_map.csv").is_file())
            partial, final = parent / "part", parent / "final"
            partial.write_bytes(b"new")
            final.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                method.publish_file(partial, final)
            self.assertEqual(final.read_bytes(), b"old")
            self.assertEqual(partial.read_bytes(), b"new")

    def test_post_copy_deadline_keeps_terminal_report_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            episode = parent / "episode"
            cache = episode / "outputs/cache"
            cache.mkdir(parents=True)
            public = episode / "outputs/runs/pilot"
            public.mkdir(parents=True)
            (parent / ".gitignore").write_text("/*/outputs/cache/\n")
            source = parent / "source"
            source.mkdir()
            with h5py.File(source / "toy.h5", "w") as stream:
                stream.create_dataset("modelmd", shape=(4, 1, 1, 2), dtype="f4")
            def fake_copy(_dataset, _rows, _private, _layout, report, _check):
                report.update(neural_values_read=True, neural_read_completed=True, all_copy_bytes_equal=True)
            def deadline_after_success(report, _check):
                report.update(status="completed", pilot_copy_complete=True)
                raise method.core.Refusal("attempt_deadline")
            with patch.multiple(method, EPISODE=episode, CACHE_PARENT=cache, PRIVATE=cache / "private/pilot", PUBLIC=public,
                                SOURCE_FILE="toy.h5", SHAPES={"sub-01": (4, 1, 1, 2)}), \
                 patch.object(method, "load_plan", return_value=(source, copy_rows(4))), \
                 patch.object(method, "load_json", return_value={}), \
                 patch.object(method, "check_capacity", return_value={}), \
                 patch.object(method, "checked_dataset", side_effect=lambda h, s: (h["modelmd"], {})), \
                 patch.object(method, "copy_block", side_effect=fake_copy), \
                 patch.object(method, "finish", side_effect=deadline_after_success), \
                 patch("sys.stdout", new_callable=io.StringIO):
                code = method.run_pilot(parent / "unused_capacity.json")
            report = json.loads((public / "pilot_report.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "stopped")
            self.assertFalse(report["pilot_copy_complete"])
            self.assertTrue(report["neural_values_read"])
            self.assertEqual(report["errors"], [{"code": "attempt_deadline"}])


if __name__ == "__main__":
    unittest.main()
