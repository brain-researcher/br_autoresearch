"""Six bounded source-free column-copy checks; coordinator-owned compute only."""
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import materialize_neural_roles_columns as method


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        episode = parent / "episode"
        cache = episode / "outputs/cache"
        cache.mkdir(parents=True)
        runs = episode / "outputs/runs/role_materialization"
        runs.mkdir(parents=True)
        source = parent / "source"
        source.mkdir()
        shapes = {"sub-01": (6, 1, 1, 5), "sub-02": (7, 1, 1, 6),
                  "sub-03": (8, 1, 1, 7), "sub-06": (9, 1, 1, 9)}
        routed = {subject: [{"participant": subject, "role": method.original.ROLES[i % 3],
                             "glmsingle_trial_index_zero_based": str(i), "eligible_image": "False" if i == 0 else "True",
                             "repeat_half": "" if i == 0 else "1", "retained_acquisition_rank": str(shape[0] - i - 1)}
                            for i in range(shape[0])] for subject, shape in shapes.items()}
        patterns = np.array([0, 0x80000000, 0x7FC01234, 0x7FA12345, 0x7F800000, 0xFF800000, 0x3F823456], dtype="<u4")
        arrays, headers = {}, []
        for index, (subject, shape) in enumerate(shapes.items()):
            for stage in method.row_copy.STAGES:
                relative = method.row_copy.source_relative(subject, stage)
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                values = np.resize(np.roll(patterns, index + ord(stage)), math.prod(shape)).view("<f4").reshape(shape)
                arrays[(subject, stage)] = values
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("modelmd", data=values, dtype="<f4", chunks=(shape[0], 1, 1, 1), compression="gzip", compression_opts=3)
                headers.append({"participant": subject, "stage": stage, "source_relative": relative,
                                "shape": list(shape), "dtype_str": "<f4", "chunks": [shape[0], 1, 1, 1],
                                "compression": "gzip", "compression_opts": 3})
        header_path = runs / "headers.json"
        header_path.write_text(json.dumps({"scope": "twelve_named_source_headers_only", "neural_values_read": False,
                                           "slurm_job_id": "fixture", "products": headers}))
        for name, error in (("pilot001", "private_directory_permissions_invalid"), ("pilot002", "source_layout_exceeds_decoded_bound")):
            out = runs / name
            out.mkdir()
            (out / "pilot_report.json").write_text(json.dumps({"status": "stopped", "neural_read_attempted": False,
                                                               "neural_values_read": False, "errors": [{"code": error}]}))
        capacity = runs / "capacity.json"
        capacity.write_text(json.dumps({"parent": str(cache), "slurm_job_id": "fixture", "quota_checked": True,
                                        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                                        "filesystem_available_bytes": 2 * method.row_copy.FULL_CAP,
                                        "quota_available_bytes": 2 * method.row_copy.FULL_CAP, "concurrent_reserved_bytes": 0}))
        (parent / ".gitignore").write_text("/*/outputs/cache/\n")
        total_stripes = 3 * sum(math.ceil(shape[3] / 4) for shape in shapes.values())
        with patch.multiple(method, EPISODE=episode, CACHE_PARENT=cache, RUNS=runs,
                            PRIVATE_PARENT=cache / "private/role_materialization", HEADERS=header_path,
                            COLUMNS=4, CHUNK_ROWS=3, FULL_STRIPES=total_stripes), \
             patch.object(method.original, "SHAPES", shapes), \
             patch.object(method.row_copy, "load_full_plan", return_value=(source, routed)), \
             patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}), patch("sys.stdout", new_callable=io.StringIO):
            yield SimpleNamespace(source=source, routed=routed, arrays=arrays, headers=headers, capacity=capacity,
                                  runs=runs, shapes=shapes)


def report(mode):
    _, public = method.paths(mode)
    return json.loads((public / ("pilot_report.json" if mode == "pilot" else "materialization_report.json")).read_text())


class ColumnMaterializationTests(unittest.TestCase):
    def test_pilot_then_full_exact_bits_disjoint_source_stripes_and_all_role_axes(self):
        with fixture() as f:
            reads = []
            original_getitem = h5py.Dataset.__getitem__
            def record_read(dataset, selection):
                path = Path(dataset.file.filename)
                if path.is_relative_to(f.source):
                    self.assertEqual(selection[:3], (slice(None), slice(None), slice(None)))
                    reads.append((path, selection[3].start, selection[3].stop))
                return original_getitem(dataset, selection)
            with patch.object(h5py.Dataset, "__getitem__", record_read):
                self.assertEqual(method.run("pilot", f.capacity), 0)
                pilot = report("pilot")
                self.assertEqual(len(reads), 1)
                self.assertEqual(reads[0][1:], (0, 4))
                self.assertEqual(pilot["logical_source_payload_bytes"], 6 * 4 * 4)
                self.assertEqual(method.run("full", f.capacity, 120), 0)
            result = report("full")
            self.assertTrue(result["full_copy_complete"])
            self.assertEqual(result["pilot_columns_reused"], 4)
            self.assertEqual(len(reads), method.FULL_STRIPES)
            for subject, stage in f.arrays:
                path = f.source / method.row_copy.source_relative(subject, stage)
                intervals = [(start, stop) for seen, start, stop in reads if seen == path]
                self.assertEqual(intervals, [(start, min(start + 4, f.shapes[subject][3]))
                                             for start in range(0, f.shapes[subject][3], 4)])
            for item in result["products"]:
                original = f.arrays[(item["participant"], item["stage"])]
                for entry in item["outputs"]:
                    indices = [i for i, row in enumerate(f.routed[item["participant"]]) if row["role"] == entry["role"]]
                    with h5py.File(entry["array_file"], "r") as handle:
                        copied = handle["modelmd"][...]
                        self.assertEqual(copied.shape, (len(indices), 1, 1, original.shape[3]))
                        self.assertEqual(copied.dtype.str, original.dtype.str)
                        self.assertTrue(np.array_equal(copied.view("u1"), original[indices].view("u1")))
            self.assertEqual(result["scientific_audit_evaluation_count"], 0)
            self.assertTrue(result["raw_audit_bytes_accessed"])
            for key in ("source_roles_complete", "reader_handoffs_activated", "physical_role_isolation_verified",
                        "candidate_scoring_authorized", "full_neural_handoffs_materialized"):
                self.assertFalse(result[key])
            for token in ("7fc01234", "finite_count", "nan_count", "sha256", "sample_values"):
                self.assertNotIn(token, json.dumps(result).lower())

    def test_pilot_reuse_rejects_wrong_column_scope_and_changed_map_before_values(self):
        for change in ("scope", "map"):
            with self.subTest(change=change), fixture() as f:
                self.assertEqual(method.run("pilot", f.capacity), 0)
                private, public = method.paths("pilot")
                if change == "scope":
                    path = public / "pilot_report.json"
                    value = json.loads(path.read_text())
                    value["pilot_scope"]["voxels_start"] = 1
                    path.write_text(json.dumps(value))
                else:
                    path = private / "development/source_index_map.csv"
                    path.write_bytes(path.read_bytes().replace(b"sub-01", b"sub-02", 1))
                with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("values before pilot binding")):
                    self.assertEqual(method.run("full", f.capacity, 120), 2)
                self.assertFalse(report("full")["neural_read_attempted"])

    def test_all_twelve_current_headers_precede_full_values_and_late_change_refuses(self):
        with fixture() as f:
            self.assertEqual(method.run("pilot", f.capacity), 0)
            late = f.source / method.row_copy.source_relative("sub-06", "D")
            with h5py.File(late, "w") as handle:
                handle.create_dataset("modelmd", shape=f.shapes["sub-06"], dtype="<f4", chunks=(1, 1, 1, 1))
            with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("values before complete preflight")):
                self.assertEqual(method.run("full", f.capacity, 120), 2)
            stopped = report("full")
            self.assertFalse(stopped["neural_values_read"])
            self.assertEqual(stopped["errors"], [{"code": "source_header_changed_after_observation"}])

    def test_failed_readback_preserves_private_partial_and_truthful_exposure(self):
        with fixture() as f:
            with patch.object(method, "verify_stripe", side_effect=method.core.Refusal("private_copy_byte_mismatch")):
                self.assertEqual(method.run("pilot", f.capacity), 2)
            stopped = report("pilot")
            self.assertTrue(stopped["neural_values_read"])
            self.assertTrue(stopped["raw_audit_bytes_accessed"])
            self.assertFalse(stopped["pilot_copy_complete"])
            private, _ = method.paths("pilot")
            for role in method.original.ROLES:
                filename = private / role / "sub-01/B.h5"
                self.assertFalse(filename.exists())
                self.assertTrue(filename.with_suffix(".h5.partial").exists())

    def test_padded_bytes_match_allocated_chunks_at_row_and_final_column_edges(self):
        self.assertEqual(method.padded_bytes(0, 513), 0)
        with tempfile.TemporaryDirectory() as directory:
            for rows, columns in ((2, 511), (1024, 512), (1025, 513)):
                with self.subTest(rows=rows, columns=columns):
                    path = Path(directory) / f"r{rows}c{columns}.h5"
                    with h5py.File(path, "w") as handle:
                        dataset = handle.create_dataset("modelmd", shape=(rows, 1, 1, columns), dtype="f4",
                                                        chunks=(min(1024, rows), 1, 1, min(512, columns)))
                        dataset[...] = np.zeros(dataset.shape, dtype="f4")
                        handle.flush()
                        self.assertEqual(dataset.id.get_storage_size(), method.padded_bytes(rows, columns))

    def test_scope_and_storage_admission_keep_original_caps(self):
        observed_shape = (12780, 1, 1, 227214)
        self.assertEqual(method.column_bound(observed_shape, (12780, 1, 1, 1), 0, 512), 26173440)
        with self.assertRaises(method.core.Refusal):
            method.column_bound(observed_shape, (1, 1, 1, 227214), 0, 512)
        with fixture() as f:
            maps = method.map_payloads({"sub-01": f.routed["sub-01"]})
            plan = method.storage_plan("pilot", {"sub-01": f.routed["sub-01"]}, maps, 500)
            self.assertEqual(plan["exact_index_map_bytes"], sum(map(len, maps.values())))
            self.assertEqual(plan["preserved_attempt_bytes"], 500)
            self.assertLess(plan["planned_bytes"], method.original.BYTE_CAP)
            with self.assertRaises(method.core.Refusal):
                method.storage_plan("pilot", {"sub-01": f.routed["sub-01"]}, maps, method.original.BYTE_CAP)
            with self.assertRaises(method.core.Refusal):
                method.run("full", f.capacity)


if __name__ == "__main__":
    unittest.main()
