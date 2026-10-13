"""Full image delivery/coverage and partial-publication checks on small fixtures."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image, PngImagePlugin

import assemble_image_features as assembly_module
import prepare_image_feature_shards as builder_module
import run_image_feature_shard as worker_module
from assemble_image_features import assemble
from feature_extractors import DIMENSIONS, require_compute
from prepare_image_feature_shards import build_packs
from run_image_feature_shard import FAMILIES, extract_task, planned_tasks, validate_manifest


class FullImageFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_compute()

    def fixture(self, root):
        cache, run = root/"cache", root/"run"
        cache.mkdir()
        (run/"tasks").mkdir(parents=True)
        ids = [f"{index:032x}" for index in range(5)]
        archive = root/"source.zip"
        pairs = []
        with zipfile.ZipFile(archive, "w") as source:
            for index, key in enumerate(ids):
                encoded = io.BytesIO()
                info = PngImagePlugin.PngInfo()
                info.add_text("source_concept", "forbidden-source-name")
                Image.fromarray(np.full((3, 4, 3), index+1, dtype=np.uint8)).save(encoded, format="PNG", pnginfo=info)
                member = f"source/concept_{index}/image.png"
                source.writestr(member, encoded.getvalue())
                pairs.append((key, member))
        config = {"family_shard_counts": dict(zip(FAMILIES, [1, 2, 3, 1, 2, 3])),
                  "pixel_pack_images": 2, "pixel_pack_byte_cap": 1024**2,
                  "pixel_cache_byte_cap": 16*1024**2, "metadata_byte_cap": 65536,
                  "worker_artifact_byte_cap": {family: 2*1024**2 for family in FAMILIES},
                  "assembled_byte_cap": 16*1024**2, "run_dir": str(run), "pixel_cache_dir": str(cache),
                  "rss_bytes": 3*1024**3}
        packs, delivery = build_packs(archive, pairs, cache, config, lambda: None)
        manifest = {"status": "complete_pixel_delivery", "image_count": 5, "opaque_ids": ids,
                    "packs": packs, "tasks": planned_tasks(config, 5), "assets": {}}
        self.assertEqual(delivery["decoded_RGB_bytes"], 5*3*4*3)
        validate_manifest(manifest, config, 5)
        return config, manifest

    def test_every_image_family_exactly_once_across_independent_pack_and_job_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, manifest = self.fixture(Path(temporary))
            received = []

            class PixelAdapter:
                def __init__(self, family, assets, device):
                    self.family = family

                def encode(self, ids, pixels):
                    received.append((self.family, ids[0]))
                    return np.full((1, DIMENSIONS[self.family]), pixels[0].mean(), dtype=np.float32), {
                        "pixel_derived_caption": str(int(pixels[0].mean()))}

            for task in manifest["tasks"]:
                extract_task(manifest, config, task["task_index"], lambda: None, PixelAdapter)
            self.assertEqual(len(received), 30)
            self.assertEqual(set(received), {(family, key) for family in FAMILIES for key in manifest["opaque_ids"]})
            report = assemble(manifest, config, lambda: None, image_count=5)
            self.assertTrue(report["all_images_each_family_exactly_once"])
            assembled = Path(config["run_dir"])/"assembled"
            for family in FAMILIES:
                values = np.load(assembled/(family+".npy"), allow_pickle=False)
                np.testing.assert_array_equal(values, np.repeat(np.arange(1, 6, dtype=np.float32)[:, None], DIMENSIONS[family], axis=1))
                details = [json.loads(line) for line in (assembled/(family+".details.jsonl")).read_text().splitlines()]
                self.assertEqual([value["opaque_image_id"] for value in details], manifest["opaque_ids"])
            for pack in manifest["packs"]:
                payload = (Path(config["pixel_cache_dir"])/pack["name"]).read_bytes()
                self.assertNotIn(b"forbidden-source-name", payload)
                self.assertNotIn(b"concept_", payload)
            # Duplicate CLI invocations must refuse without invalidating any
            # previously successful attempt, even through exception handlers.
            cache, run = Path(config["pixel_cache_dir"]), Path(config["run_dir"])
            (cache/"manifest.json").write_text(json.dumps(manifest))
            prior_task = (run/"tasks/task_00000/report.json").read_bytes()
            prior_assembled = (assembled/"report.json").read_bytes()
            deadline = str(time.time()+30)
            with patch.object(builder_module, "load_config", return_value=config), patch.object(builder_module, "admit"):
                self.assertEqual(builder_module.main(["--config", "unused", "--image-archive", "unused",
                                                       "--deadline-unix", deadline]), 1)
            with patch.object(worker_module, "load_config", return_value=config), patch.object(worker_module, "admit"), \
                 patch.object(worker_module, "validate_manifest", side_effect=lambda value, limits: validate_manifest(value, limits, 5)), \
                 patch.dict(os.environ, {"L_SCRATCH_JOB": temporary}):
                self.assertEqual(worker_module.main(["--config", "unused", "--task-index", "0", "--deadline-unix", deadline,
                                                      "--local-dir", str(Path(temporary)/"worker_local")]), 1)
            original_assemble = assembly_module.assemble
            with patch.object(assembly_module, "load_config", return_value=config), patch.object(assembly_module, "admit"), \
                 patch.object(assembly_module, "assemble", side_effect=lambda *args, **kwargs: original_assemble(*args, image_count=5, **kwargs)):
                self.assertEqual(assembly_module.main(["--config", "unused", "--deadline-unix", deadline]), 1)
            self.assertFalse((run/"pixel_delivery_failure.json").exists())
            self.assertFalse((run/"tasks/task_00000/failure.json").exists())
            self.assertFalse((assembled/"failure.json").exists())
            self.assertEqual((run/"tasks/task_00000/report.json").read_bytes(), prior_task)
            self.assertEqual((assembled/"report.json").read_bytes(), prior_assembled)

    def test_missing_task_or_changed_coverage_cannot_create_assembled_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, manifest = self.fixture(Path(temporary))
            with self.assertRaises(FileNotFoundError):
                assemble(manifest, config, lambda: None, image_count=5)
            self.assertFalse((Path(config["run_dir"])/"assembled").exists())
            for kind in ("missing_range", "duplicate_image", "overlapping_pack"):
                broken = deepcopy(manifest)
                if kind == "missing_range":
                    broken["tasks"][-1]["start"] += 1
                elif kind == "duplicate_image":
                    broken["opaque_ids"][-1] = broken["opaque_ids"][0]
                else:
                    broken["packs"][-1]["start"] -= 1
                with self.assertRaises(ValueError):
                    assemble(broken, config, lambda: None, image_count=5)
                self.assertFalse((Path(config["run_dir"])/"assembled").exists())

    def test_failed_row_preserves_partial_without_retry_or_success_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, manifest = self.fixture(Path(temporary))
            received = []

            class Stops:
                def __init__(self, family, assets, device):
                    self.family = family

                def encode(self, ids, pixels):
                    received.append(ids[0])
                    if len(received) == 2:
                        raise RuntimeError("Fixed fixture failure")
                    return np.ones((1, DIMENSIONS[self.family]), dtype=np.float32), {}

            with self.assertRaises(RuntimeError):
                extract_task(manifest, config, 0, lambda: None, Stops)
            output = Path(config["run_dir"])/"tasks/task_00000"
            self.assertEqual(received, manifest["opaque_ids"][:2])
            self.assertTrue((output/"features.partial.npy").exists())
            self.assertFalse((output/"features.npy").exists())
            self.assertFalse((output/"report.json").exists())


if __name__ == "__main__":
    unittest.main()
