"""Bounded source-free fixtures; execution belongs in the coordinator's allocation."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import signal
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import materialize_neural_roles_full as method


def rows_for(subject, count):
    return [{"participant": subject, "role": method.pilot.ROLES[i % 3],
             "glmsingle_trial_index_zero_based": str(i), "eligible_image": "False" if i == 0 else "True",
             "repeat_half": "" if i == 0 else "1", "failure_reasons": "fixture" if i == 0 else "",
             "retained_acquisition_rank": str(count - i - 1), "expected_provider_image_path": f"concept/image{i}.jpg"}
            for i in range(count)]


@contextmanager
def full_fixture(endian="<"):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        episode = root / "episode"
        cache = episode / "outputs/cache"
        cache.mkdir(parents=True)
        private_pilot = cache / "private/role_materialization/pilot001"
        public_pilot = episode / "outputs/runs/role_materialization/pilot001"
        public_pilot.mkdir(parents=True)
        full_private = cache / "private/role_materialization/attempt001"
        full_public = episode / "outputs/runs/role_materialization/attempt001"
        full_public.mkdir()
        source = root / "source"
        source.mkdir()
        shapes = {"sub-01": (9, 1, 1, 5), "sub-02": (10, 1, 1, 7),
                  "sub-03": (7, 1, 1, 3), "sub-06": (6, 1, 1, 4)}
        routed = {subject: rows_for(subject, shape[0]) for subject, shape in shapes.items()}
        arrays = {}
        patterns = np.array([0x00000000, 0x80000000, 0x7FC01234, 0x7FA12345,
                             0x7F800000, 0xFF800000, 0x3F812345], dtype=endian + "u4")
        for si, (subject, shape) in enumerate(shapes.items()):
            for ti, stage_name in enumerate(method.STAGES):
                filename = source / method.source_relative(subject, stage_name)
                filename.parent.mkdir(parents=True, exist_ok=True)
                values = np.resize(np.roll(patterns, si + ti), math.prod(shape)).view(endian + "f4").reshape(shape)
                arrays[(subject, stage_name)] = values
                with h5py.File(filename, "w") as handle:
                    handle.create_dataset("modelmd", data=values, dtype=values.dtype, chunks=(3, 1, 1, shape[3]))
        payload = sum(math.prod(shape) for shape in shapes.values()) * 3 * 4
        blocks = sum(math.ceil(shape[0] / 4) for shape in shapes.values()) * 3
        with patch.multiple(method.pilot, EPISODE=episode, CACHE_PARENT=cache, PRIVATE=private_pilot,
                            PUBLIC=public_pilot, SHAPES=shapes, ROWS=4, private_directory=method.private_directory), \
             patch.multiple(method, EPISODE=episode, CACHE_PARENT=cache, PRIVATE=full_private, PUBLIC=full_public,
                            PILOT_PRIVATE=private_pilot, PILOT_PUBLIC=public_pilot,
                            PILOT_REPORT=public_pilot / "pilot_report.json", EXPECTED_PAYLOAD_BYTES=payload,
                            EXPECTED_BLOCKS=blocks):
            method.pilot.private_directory(private_pilot, cache)
            with h5py.File(source / method.source_relative("sub-01", "B"), "r") as handle:
                dataset, layout = method.pilot.checked_dataset(handle, shapes["sub-01"], stop=4)
                report = method.pilot.initial_report()
                report["source_layout"] = layout
                method.pilot.copy_block(dataset, routed["sub-01"], private_pilot, layout, report, lambda: None, stop=4)
            report.update(status="completed", pilot_copy_complete=True, source_open_header_seconds=0.01,
                          elapsed_seconds=0.1, peak_rss_kib=10000)
            method.PILOT_REPORT.write_text(json.dumps(report), encoding="utf-8")
            (root / ".gitignore").write_text("/*/outputs/cache/\n", encoding="utf-8")
            yield SimpleNamespace(root=root, source=source, private=full_private, public=full_public,
                                  routed=routed, shapes=shapes, arrays=arrays, pilot_report=report)


class FullNeuralRoleMaterializationTests(unittest.TestCase):
    def test_all_products_preserve_bits_axes_roles_and_reuse_pilot_without_source_reread(self):
        for endian in ("<", ">"):
            with self.subTest(endian=endian), full_fixture(endian) as fixture:
                preserved = method.validate_pilot(fixture.pilot_report, fixture.routed)
                products = method.preflight_sources(fixture.source, lambda: None)
                budget = method.OutputBudget(preserved)
                method.pilot.private_directory(fixture.private, method.CACHE_PARENT)
                method.write_full_maps(fixture.routed, fixture.private, budget, lambda: None)
                report = method.initial_report(120)
                read_scopes = []
                original_getitem = h5py.Dataset.__getitem__
                def observed_getitem(dataset, selection):
                    filename = Path(dataset.file.filename)
                    if filename.is_relative_to(fixture.source):
                        start, stop = selection[0].start or 0, selection[0].stop
                        read_scopes.append((filename, start, stop))
                        if filename == fixture.source / method.source_relative("sub-01", "B"):
                            self.assertGreaterEqual(start, 4)
                    return original_getitem(dataset, selection)
                with patch.object(h5py.Dataset, "__getitem__", observed_getitem):
                    for product in products:
                        method.copy_product(fixture.source, product, fixture.routed[product["participant"]],
                                            fixture.private, budget, report, lambda: None)
                self.assertEqual(len(report["products"]), 12)
                self.assertTrue(all(item["complete"] for item in report["products"]))
                self.assertEqual(sum(item["pilot_rows_reused"] for item in report["products"]), 4)
                self.assertEqual(len(read_scopes), method.EXPECTED_BLOCKS - 1)
                for product in report["products"]:
                    subject, stage_name = product["participant"], product["stage"]
                    original = fixture.arrays[(subject, stage_name)]
                    for entry in product["outputs"]:
                        indices = [i for i, row in enumerate(fixture.routed[subject]) if row["role"] == entry["role"]]
                        filename = Path(entry["array_file"])
                        with h5py.File(filename, "r") as handle:
                            copied = handle["modelmd"][...]
                            self.assertEqual(copied.dtype.str, original.dtype.str)
                            self.assertTrue(np.array_equal(copied.view("u1"), original[indices].view("u1")))
                            self.assertFalse(handle["modelmd"].is_virtual)
                            self.assertFalse(handle["modelmd"].external)
                        self.assertEqual(stat.S_IMODE(filename.stat().st_mode), 0o600)
                        self.assertEqual(filename.stat().st_nlink, 1)
                        self.assertFalse(filename.with_suffix(".h5.partial").exists())
                for role in method.pilot.ROLES:
                    path = fixture.private / role / "source_index_map.csv"
                    with path.open(newline="") as stream:
                        mapped = list(csv.DictReader(stream))
                    for subject in fixture.shapes:
                        saved = [row for row in mapped if row["participant"] == subject]
                        expected = [row for row in fixture.routed[subject] if row["role"] == role]
                        self.assertEqual([row["local_row_index_zero_based"] for row in saved], [str(i) for i in range(len(saved))])
                        self.assertEqual([row["glmsingle_trial_index_zero_based"] for row in saved],
                                         [row["glmsingle_trial_index_zero_based"] for row in expected])
                        self.assertEqual([row["retained_acquisition_rank"] for row in saved],
                                         [row["retained_acquisition_rank"] for row in expected])
                self.assertEqual(budget.total, preserved + sum(path.stat().st_size for path in budget.sizes))
                self.assertTrue(report["raw_audit_bytes_accessed"])
                self.assertEqual(report["scientific_audit_evaluation_count"], 0)
                for field in ("reader_handoffs_activated", "physical_role_isolation_verified", "source_roles_complete",
                              "candidate_scoring_authorized", "full_neural_handoffs_materialized"):
                    self.assertFalse(report[field])
                encoded = json.dumps(report).lower()
                for forbidden in ("7fc01234", "nan_count", "finite_count", "sha256", "sample_values", "correlation"):
                    self.assertNotIn(forbidden, encoded)

    def test_later_intersecting_chunk_bound_and_last_source_header_fail_before_values(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "chunk.h5", "w") as handle:
                handle.create_dataset("modelmd", shape=(130, 1, 1, 1), dtype="f4", chunks=(80, 1, 1, 1))
                with patch.object(method.pilot, "BYTE_CAP", 500):
                    method.pilot.checked_dataset(handle, (130, 1, 1, 1))
                    with self.assertRaises(method.core.Refusal):
                        method.header_plan(handle, (130, 1, 1, 1))
        with full_fixture() as fixture:
            path = fixture.source / method.source_relative("sub-06", "D")
            with h5py.File(path, "w") as handle:
                handle.create_dataset("modelmd", shape=(5, 1, 1, 4), dtype="f4")
            with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("payload read during preflight")):
                with self.assertRaises(method.core.Refusal):
                    method.preflight_sources(fixture.source, lambda: None)

    def test_pilot_map_must_match_exact_source_indices_before_any_pilot_value(self):
        with full_fixture() as fixture:
            path = method.pilot.PRIVATE / "development/source_index_map.csv"
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["glmsingle_trial_index_zero_based"] = "1"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("payload read during pilot metadata check")):
                with self.assertRaises(method.core.Refusal):
                    method.validate_pilot(fixture.pilot_report, fixture.routed)

    def test_failed_readback_preserves_partial_without_publication_or_readiness(self):
        with full_fixture() as fixture:
            preserved = method.validate_pilot(fixture.pilot_report, fixture.routed)
            products = method.preflight_sources(fixture.source, lambda: None)
            method.pilot.private_directory(fixture.private, method.CACHE_PARENT)
            budget = method.OutputBudget(preserved)
            method.write_full_maps(fixture.routed, fixture.private, budget, lambda: None)
            report = method.initial_report(120)
            with patch.object(method, "verify_segment", side_effect=method.core.Refusal("private_copy_byte_mismatch")):
                with self.assertRaises(method.core.Refusal):
                    method.copy_product(fixture.source, products[0], fixture.routed["sub-01"], fixture.private,
                                        budget, report, lambda: None)
            self.assertFalse(report["products"][0]["complete"])
            self.assertTrue(report["neural_values_read"])
            self.assertTrue(report["raw_audit_bytes_accessed"])
            self.assertFalse(report["full_copy_complete"])
            for role in method.pilot.ROLES:
                filename = fixture.private / role / "sub-01/B.h5"
                self.assertFalse(filename.exists())
                self.assertTrue(filename.with_suffix(".h5.partial").exists())
            self.assertEqual(fixture.pilot_report["private_output_bytes"], preserved - method.PILOT_REPORT.stat().st_size)

    def test_output_ceiling_includes_pilot_and_partial_and_restores_file_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial"
            budget = method.OutputBudget(16, cap=128, report_reserve=8)
            before = resource.getrlimit(resource.RLIMIT_FSIZE)
            old_signal = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            try:
                with self.assertRaises(OSError):
                    with budget.write(path), method.pilot.new_private_file(path) as stream:
                        stream.write(b"x" * 200)
                        stream.flush()
            finally:
                signal.signal(signal.SIGXFSZ, old_signal)
            self.assertEqual(resource.getrlimit(resource.RLIMIT_FSIZE), before)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            self.assertLessEqual(budget.total + budget.report_reserve, budget.cap)

    def test_full_capacity_cannot_use_pilot_only_headroom(self):
        record = {"parent": str(method.CACHE_PARENT), "slurm_job_id": "fixture",
                  "observed_at_utc": datetime.now(timezone.utc).isoformat(), "quota_checked": True,
                  "filesystem_available_bytes": method.FULL_CAP, "quota_available_bytes": method.FULL_CAP,
                  "concurrent_reserved_bytes": 0}
        method.check_full_capacity(record, "fixture")
        for field in ("filesystem_available_bytes", "quota_available_bytes"):
            with self.subTest(field=field), self.assertRaises(method.core.Refusal):
                method.check_full_capacity({**record, field: method.pilot.BYTE_CAP}, "fixture")
        with self.assertRaises(method.core.Refusal):
            method.check_full_capacity({**record, "concurrent_reserved_bytes": 1}, "fixture")


if __name__ == "__main__":
    unittest.main()
