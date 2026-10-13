"""Two structural reader checks; run only in the coordinator's Slurm allocation.

The temporary arrays are file-format fixtures, not observations or a model
recovery study. The no-op admission callback represents a test harness only.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import development_role_reader as reader


def check():
    pass


def fixture(root):
    root.mkdir()
    originals = {"trialID": np.array([[2, 4, 7]], dtype=np.uint16),
        "seshID": np.array([[8.], [1.], [3.]], dtype=np.float64),
        "lickState": np.array([[1], [-1], [0]], dtype=np.int8),
        "lick": np.arange(3*701, dtype=np.float32).reshape(3, 701)}
    originals["lick"][1, 3] = np.nan
    entries = []
    for index in reader.DEVELOPMENT_RECORDS:
        name = f"source_record_{index:02d}"
        (root / name).mkdir()
        fields = {}
        for field, value in originals.items():
            path = root / name / f"{field}.npy"
            np.save(path, value, allow_pickle=False)
            fields[field] = {"relative_path": f"{name}/{field}.npy", "dtype": str(value.dtype),
                             "shape": list(value.shape), "npy_bytes": path.stat().st_size}
        entries.append({"opaque_animal_id": name, "source_index_1_based": index,
                        "cohort_id": "development_control", "fields": fields})
    manifest = {"schema_version": "ep02.dudman.role_materialization.v1",
        "pack_format": "deterministic_npy_tree_v1", "role": "development_full",
        "visibility": "candidate_readable", "source_payload_sha256": reader.SOURCE_SHA256,
        "round_trip_verified": True, "outcome_summaries_emitted": False,
        "record_count": 9, "record_entries": entries}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    return manifest, originals


def opened(root, *, cap=1024*1024, callback=check):
    return reader.open_development_pack(root, admit_development_root=lambda path: None,
                                       maximum_mapped_bytes=cap, check=callback)


class DevelopmentRoleReaderTests(unittest.TestCase):
    def test_external_admission_and_whole_roster_refuse_before_array_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "development"
            manifest, _ = fixture(root)
            admitted = []
            def deny(path):
                admitted.append(path)
                raise PermissionError("external role admission denied")
            with patch.object(reader, "_open_regular", side_effect=AssertionError("file opened before admission")):
                with self.assertRaisesRegex(PermissionError, "external role"):
                    reader.open_development_pack(root, admit_development_root=deny,
                                                 maximum_mapped_bytes=10000, check=check)
            self.assertEqual(admitted, [root])
            real_open_regular = reader._open_regular
            def manifest_only(path):
                if path != root / "MANIFEST.json":
                    raise AssertionError("array file opened before binding")
                return real_open_regular(path)
            with patch.object(reader, "_open_regular", side_effect=manifest_only), \
                 patch.object(reader.np, "memmap", side_effect=AssertionError("array mapped before binding")):
                for changed in (dict(manifest, role="primary_audit_evaluator_only"),
                                dict(manifest, record_count=8)):
                    (root / "MANIFEST.json").write_text(json.dumps(changed))
                    with self.assertRaises((PermissionError, ValueError)):
                        opened(root)
                changed = json.loads(json.dumps(manifest))
                changed["record_entries"][-1]["source_index_1_based"] = 20
                (root / "MANIFEST.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(PermissionError, "identities"):
                    opened(root)
                changed = json.loads(json.dumps(manifest))
                changed["record_entries"][0]["fields"]["lick"]["relative_path"] = "../audit/lick.npy"
                (root / "MANIFEST.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(PermissionError, "field path"):
                    opened(root)
                (root / "MANIFEST.json").write_text(json.dumps(manifest))
                with opened(root) as pack:
                    with self.assertRaises(PermissionError):
                        pack.open_mouse(3)
                with opened(root, cap=1) as pack:
                    with self.assertRaisesRegex(ValueError, "byte bound"):
                        pack.open_mouse(1)

    def test_original_axes_dtype_values_and_bounded_map_lifetime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "development"
            _, originals = fixture(root)
            with opened(root) as pack:
                with pack.open_mouse(1) as mouse:
                    self.assertEqual(tuple(mouse.arrays), reader.FIELDS)
                    self.assertEqual(mouse.source_index_1_based, 1)
                    maps = tuple(mouse.arrays.values())
                    for field, original in originals.items():
                        value = mouse.arrays[field]
                        self.assertEqual((value.dtype, value.shape), (original.dtype, original.shape))
                        self.assertEqual(value.tobytes(), original.tobytes())
                        self.assertFalse(value.flags.writeable)
                        with self.assertRaises(ValueError):
                            value.flat[0] = 9
                    with self.assertRaisesRegex(RuntimeError, "active"):
                        pack.open_mouse(2)
                self.assertTrue(all(value._mmap.closed for value in maps))
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    _ = mouse.arrays
                next_mouse = pack.open_mouse(2)
                next_maps = tuple(next_mouse.arrays.values())
            self.assertTrue(all(value._mmap.closed for value in next_maps))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pack.open_mouse(1)

            # A resource callback abort after the first real map must close
            # that map, even before a DevelopmentMouse can be returned.
            mapped = []
            real_memmap = np.memmap
            def capture(*args, **kwargs):
                result = real_memmap(*args, **kwargs)
                mapped.append(result)
                return result
            def cancel_after_first_map():
                if mapped:
                    raise TimeoutError("caller deadline")
            with opened(root, callback=cancel_after_first_map) as pack:
                with patch.object(reader.np, "memmap", side_effect=capture):
                    with self.assertRaisesRegex(TimeoutError, "caller deadline"):
                        pack.open_mouse(1)
                self.assertEqual(len(mapped), 1)
                self.assertTrue(mapped[0]._mmap.closed)


if __name__ == "__main__":
    unittest.main()
