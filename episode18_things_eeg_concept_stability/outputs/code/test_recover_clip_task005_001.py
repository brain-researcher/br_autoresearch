"""Two focused recovery-binding/publication regressions; coordinator compute only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import recover_clip_task005_001 as recovery
from feature_extractors import DIMENSIONS, require_compute
from run_image_feature_shard import FAMILIES, extract_task, planned_tasks


class Pixels:
    received = []

    def __init__(self, family, assets, device):
        self.family = family

    def encode(self, ids, pixels):
        self.received.extend(ids)
        return np.full((1, DIMENSIONS[self.family]), int(pixels[0][0, 0, 0]), dtype=np.float32), {}


def fixture(root):
    """Small image count, same six-family/26-task plan; no empirical values."""
    run, cache = root/"original", root/"pixels"
    fresh = root/"image_features_full_20261012_recovery001"
    (run/"tasks").mkdir(parents=True)
    cache.mkdir()
    ids = [f"{index:032x}" for index in range(16)]
    config = {"family_shard_counts": dict(zip(FAMILIES, (2, 1, 1, 2, 16, 4))),
        "pixel_pack_images": 16, "pixel_pack_byte_cap": 1024**2,
        "pixel_cache_byte_cap": 1024**2, "metadata_byte_cap": 65536,
        "worker_artifact_byte_cap": {family: 128*1024**2 for family in FAMILIES},
        "assembled_byte_cap": 8*1024**2, "run_dir": str(run), "pixel_cache_dir": str(cache),
        "not_after_unix": 1791927000}
    pack = cache/"pixels_00000.npz"
    with zipfile.ZipFile(pack, "x", compression=zipfile.ZIP_STORED) as archive:
        for index, key in enumerate(ids):
            with archive.open(key+".npy", "w") as destination:
                np.lib.format.write_array(destination, np.full((3, 4, 3), index+1, dtype=np.uint8),
                                          allow_pickle=False)
    manifest = {"status": "complete_pixel_delivery", "image_count": 16, "opaque_ids": ids,
        "tasks": planned_tasks(config, 16), "assets": {},
        "packs": [{"name": pack.name, "start": 0, "stop": 16, "bytes": pack.stat().st_size}]}
    for task in manifest["tasks"]:
        if task["task_index"] != 5:
            extract_task(manifest, config, task["task_index"], lambda: None, Pixels)
    failed = run/"tasks/task_00005"
    failed.mkdir()
    (failed/"features.partial.npy").write_bytes(b"preserved failed bytes, never an assembly input")
    (failed/"failure.json").write_text('{"status":"historical_failure"}')
    grant = {"schema": recovery.SCHEMA, "authorized": True,
        "original_config": str(recovery.ORIGINAL_CONFIG), "fresh_worker_root": str(fresh),
        "original_assembly_target": str(run/"assembled"), "global_deadline_unix": 1791927000,
        "worker": dict(recovery.WORKER), "assembly": dict(recovery.ASSEMBLY),
        "numerical_threads": 1, "replacement_task_index": 5, "automatic_retry": False,
        "reserved_cpu_hours": 9}
    return config, manifest, grant


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_compute()

    def test_only_task5_is_replaced_with_unchanged_extraction_and_complete_original_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, grant = fixture(root)
            original = deepcopy(config)
            failed = Path(config["run_dir"])/"tasks/task_00005/features.partial.npy"
            historical = failed.read_bytes()
            grant_path = root/"grant.json"
            grant_path.write_text(json.dumps(grant))
            # Real grant validation distinguishes the new resources and admits
            # no other replacement task or implicit authorization.
            with patch.object(recovery, "load_config", return_value=config):
                self.assertEqual(recovery.load_grant(grant_path), (grant, config))
                for key, value in (("replacement_task_index", 4), ("authorized", False),
                                   ("reserved_cpu_hours", 91)):
                    changed = dict(grant, **{key: value})
                    grant_path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        recovery.load_grant(grant_path)
            # Fixture alone patches the expected row extent; production main
            # requires all22448 and the literal original11224:22448 range.
            with patch.object(recovery, "EXPECTED_TASK", manifest["tasks"][5]), \
                 patch.object(recovery, "progress_factory", return_value=Pixels):
                Pixels.received = []
                owned = []
                report = recovery.run_worker(grant, config, manifest, lambda: None, owned.append)
                self.assertEqual(Pixels.received, manifest["opaque_ids"][8:])
                self.assertEqual(config, original)
                self.assertEqual(len(owned), 1)
                selected = recovery.completed_sources(manifest, config, grant, 16)
                self.assertEqual(len(selected), 26)
                for record, path in selected:
                    expected_root = Path(grant["fresh_worker_root"] if record["task_index"] == 5 else config["run_dir"])
                    self.assertEqual(path, expected_root/"tasks"/f"task_{record['task_index']:05d}")
                result = recovery.assemble_recovery(manifest, config, grant, lambda: None, image_count=16)
                self.assertEqual(result["complete_tasks"], 26)
                self.assertTrue(result["all_images_each_family_exactly_once"])
                output = Path(grant["original_assembly_target"])
                for family in FAMILIES:
                    values = np.load(output/(family+".npy"), allow_pickle=False)
                    np.testing.assert_array_equal(values[:, 0], np.arange(1, 17))
                sources = json.loads((output/"recovery_sources.json").read_text())
                self.assertEqual(sources["binding"]["partial_rows_reused"], 0)
                self.assertEqual(sources["binding"]["new_reserved_cpu_hours"], 9)
                previous = (output/"report.json").read_bytes()
                recovery_assemble = recovery.assemble_recovery
                with patch.object(recovery, "load_grant", return_value=(grant, config)), \
                     patch.object(recovery, "admit"), \
                     patch.object(recovery, "original_manifest", return_value=manifest), \
                     patch.object(recovery, "assemble_recovery", side_effect=lambda *args, **kw:
                        recovery_assemble(*args, image_count=16, **kw)):
                    # Use the original callable explicitly; duplicate main
                    # must fail without obtaining ownership or mutating history.
                    self.assertEqual(recovery.main(["--grant", "unused", "--phase", "assembly",
                        "--deadline-unix", str(time.time()+30)]), 1)
                self.assertEqual((output/"report.json").read_bytes(), previous)
                self.assertFalse((output/"failure.json").exists())
                self.assertEqual(report["images"], 8)
            self.assertEqual(failed.read_bytes(), historical)

    def test_incomplete_misbound_and_late_detail_failure_never_publish_success_or_replace_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, grant = fixture(root)
            output = Path(grant["original_assembly_target"])
            with patch.object(recovery, "EXPECTED_TASK", manifest["tasks"][5]), \
                 patch.object(recovery, "progress_factory", return_value=Pixels):
                with self.assertRaises(FileNotFoundError):
                    recovery.assemble_recovery(manifest, config, grant, lambda: None, image_count=16)
                self.assertFalse(output.exists())
                recovery.run_worker(grant, config, manifest, lambda: None, lambda _: None)
                replacement = Path(grant["fresh_worker_root"])
                complete_path = replacement/"recovery_complete.json"
                complete = complete_path.read_bytes()
                complete_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    recovery.assemble_recovery(manifest, config, grant, lambda: None, image_count=16)
                self.assertFalse(output.exists())
                complete_path.write_bytes(complete)
                record_path = replacement/"tasks/task_00005/report.json"
                report = json.loads(record_path.read_text())
                report["opaque_ids"][0] = manifest["opaque_ids"][0]
                record_path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    recovery.assemble_recovery(manifest, config, grant, lambda: None, image_count=16)
                self.assertFalse(output.exists())
                report["opaque_ids"][0] = manifest["opaque_ids"][8]
                record_path.write_text(json.dumps(report))
                details_path = replacement/"tasks/task_00005/image_details.jsonl"
                lines = details_path.read_text().splitlines(keepends=True)
                first = json.loads(lines[0])
                first["opaque_image_id"] = manifest["opaque_ids"][0]
                lines[0] = json.dumps(first)+"\n"
                details_path.write_text("".join(lines))
                # Same-length UUID mutation passes byte counts but must fail
                # the original per-row details join after assembly has begun.
                owned = []
                with self.assertRaises(ValueError):
                    recovery.assemble_recovery(manifest, config, grant, lambda: None,
                                               image_count=16, on_create=owned.append)
                self.assertEqual(owned, [output])
                self.assertFalse((output/"report.json").exists())
                self.assertTrue((output/"clip_vit_b32_image.partial.npy").exists())
                self.assertTrue((Path(config["run_dir"])/"tasks/task_00005/features.partial.npy").exists())


if __name__ == "__main__":
    unittest.main()
