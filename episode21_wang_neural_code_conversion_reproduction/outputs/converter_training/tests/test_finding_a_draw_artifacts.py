"""Three compact persistence checks; source-free, coordinator compute, UNRUN."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_draw_artifacts as artifacts
from test_finding_a_draw_batches import fixture_factory
from test_finding_a_batched_inference import equal
from test_finding_a_inference import TEST_IDS


CAP = 16 * 1024**2


def identity(root):
    return {"collection_path": str(root / "decoded_point.h5"),
        "point_path": str(root / "decoded_point.decoded_point.json") + "#point",
        "fitted_object_ids": [{"method": method, "source": source, "target": target,
            "fitted_object_id": None if method == "brain_loss" and "sub-01" in (source, target)
                else "fixture-original:" + ":".join((method, source, target or "within"))}
            for method, source, target in artifacts.collection.STREAMS],
        "feature_identity": {"role": "tau_test", "dtype": "float32", "extraction_status": "complete",
                             "checkpoint": "fixture-checkpoint", "preprocessing": "fixture-preprocessing"}}


def batch(index):
    schedule = artifacts.batches._schedule()
    make_panel = fixture_factory(schedule)
    start = index * 16
    return artifacts.batches.DrawBatch(index, tuple(make_panel(row) for row in schedule[start:start + 16]))


def options(root, index):
    return dict(expected_stage_identity=identity(root), expected_batch_index=index,
                expected_image_ids=TEST_IDS, maximum_bytes=CAP, check=lambda: None)


class FindingADrawArtifactTests(unittest.TestCase):
    def test_point_and_first_tail_batches_roundtrip_exact_original_values_and_identity(self):
        self.assertEqual((artifacts.batches.BATCH_COUNT, artifacts.batches.DRAW_BATCH_SIZE), (63, 16))
        self.assertEqual((artifacts.original.BOOTSTRAP_DRAWS, artifacts.original.BOOTSTRAP_SEED), (1000, 42))
        make_panel = fixture_factory(artifacts.batches._schedule())
        point = make_panel(np.arange(224), point=True)
        fields = point.pattern
        means = {key: value.copy() for key, value in fields.layer_means.items()}
        means["within_source"][0, 0] = -0.0
        point = replace(point, pattern=artifacts.original._reduce((means,
            fields.layer_defined_counts, fields.layer_undefined_counts, fields.layer_reasons), lambda: None))
        record = artifacts.panel_to_record(point, check=lambda: None)
        self.assertEqual(set(record["pattern"]), set(artifacts._FIELDS))
        self.assertIsNone(record["pattern"]["layer_means"]["within_source"][0][4])
        encoded = json.dumps(record, allow_nan=False)
        restored = artifacts.panel_from_record(json.loads(encoded), expected_image_ids=TEST_IDS, check=lambda: None)
        equal(self, restored, point)
        self.assertTrue(np.signbit(restored.pattern.layer_means["within_source"][0, 0]))
        self.assertIsNone(restored.pattern.group_means["within_source"])
        self.assertFalse(restored.image_indices.flags.writeable)
        # An incorrect derived estimate cannot be discarded unnoticed at write.
        changed = dict(point.pattern.group_means)
        changed["content_loss"] = 123.0
        bad = replace(point, pattern=replace(point.pattern, group_means=MappingProxyType(changed)))
        with self.assertRaisesRegex(ValueError, "reduction"):
            artifacts.panel_to_record(bad, check=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for index, count in ((0, 16), (62, 8)):
                with self.subTest(batch=index):
                    path, expected = root / f"batch{index}.json", batch(index)
                    kwargs = options(root, index)
                    self.assertEqual(artifacts.write_draw_batch(path, expected, **kwargs), path)
                    self.assertLessEqual(path.stat().st_size, CAP)
                    self.assertFalse(path.with_name(path.name + ".partial").exists())
                    actual = artifacts.read_draw_batch(path, **kwargs)
                    self.assertEqual(len(actual.panels), count)
                    self.assertEqual(actual.batch_index, expected.batch_index)
                    for restored_panel, expected_panel in zip(actual.panels, expected.panels):
                        equal(self, restored_panel, expected_panel)
                    saved = path.read_bytes()
                    with self.assertRaises(FileExistsError):
                        artifacts.write_draw_batch(path, expected, **kwargs)
                    self.assertEqual(path.read_bytes(), saved)
            # A callback cannot silently replace the expected published identity.
            supplied = identity(root)
            original_identity = deepcopy(supplied)
            def mutate():
                supplied["feature_identity"]["checkpoint"] = "changed-after-snapshot"
            snap = artifacts.stage_identity_record(supplied, check=mutate)
            self.assertEqual(snap, original_identity)
            self.assertNotEqual(supplied, original_identity)

    def test_mismatched_stage_images_rows_support_and_truncated_records_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "original.json"
            kwargs = options(root, 62)
            artifacts.write_draw_batch(source, batch(62), **kwargs)
            original = json.loads(source.read_text())
            for defect in ("collection", "point", "fitted", "truth", "batch_index", "missing_draw",
                           "reordered_draws", "images", "layers", "rows", "count_type", "count_range",
                           "unfinished_reason", "truncated", "nonfinite_JSON", "duplicate_field"):
                with self.subTest(defect=defect):
                    value = deepcopy(original)
                    if defect == "collection":
                        value["stage_identity"]["collection_path"] = str(root / "other.h5")
                    elif defect == "point":
                        value["stage_identity"]["point_path"] = str(root / "other.json") + "#point"
                    elif defect == "fitted":
                        value["stage_identity"]["fitted_object_ids"][0]["fitted_object_id"] = "different-model"
                    elif defect == "truth":
                        value["stage_identity"]["feature_identity"]["checkpoint"] = "different-truth"
                    elif defect == "batch_index":
                        value["batch_index"] = 61
                    elif defect == "missing_draw":
                        value["panels"].pop()
                    elif defect == "reordered_draws":
                        value["panels"].reverse()
                    elif defect in ("images", "layers"):
                        value["panels"][0]["image_ids" if defect == "images" else "layers"].reverse()
                    elif defect == "rows":
                        value["panels"][0]["image_indices"][0] = True
                    elif defect in ("count_type", "count_range"):
                        value["panels"][0]["pattern"]["layer_defined_counts"]["within_source"][0][0] = (
                            1.5 if defect == "count_type" else -2)
                    elif defect == "unfinished_reason":
                        value["panels"][0]["profile"]["layer_reasons"]["within_source"][0][0] = "not_evaluated"
                    text = json.dumps(value, allow_nan=False)
                    if defect == "truncated":
                        text = text[:-7]
                    elif defect == "nonfinite_JSON":
                        text = text.replace('"batch_index": 62', '"batch_index": NaN', 1)
                    elif defect == "duplicate_field":
                        text = '{"batch_index": 62,' + text[1:]
                    path = root / f"{defect}.json"
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        artifacts.read_draw_batch(path, **kwargs)
            wrong_images = dict(kwargs, expected_image_ids=TEST_IDS[::-1])
            with self.assertRaisesRegex(ValueError, "identity"):
                artifacts.read_draw_batch(source, **wrong_images)
            unopened = mock.Mock(side_effect=AssertionError("invalid task reached filesystem"))
            for invalid in (-1, 63, True, 2.0):
                with self.subTest(index=invalid), mock.patch.object(Path, "open", unopened), \
                        self.assertRaisesRegex(ValueError, "batch_index"):
                    artifacts.read_draw_batch(source, **dict(kwargs, expected_batch_index=invalid))
            unopened.assert_not_called()

    def test_byte_bounds_cancellation_and_exclusive_partials_close_all_streams(self):
        expected = batch(62)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            kwargs, opened = options(root, 62), []
            actual_open = Path.open
            def tracking(path, *args, **options):
                result = actual_open(path, *args, **options)
                opened.append(result)
                return result
            path = root / "capped.json"
            with mock.patch.object(Path, "open", tracking), self.assertRaisesRegex(ValueError, "byte_cap"):
                artifacts.write_draw_batch(path, expected, **dict(kwargs, maximum_bytes=1024))
            partial = path.with_name(path.name + ".partial")
            self.assertTrue(partial.exists())
            self.assertLessEqual(partial.stat().st_size, 1024)
            self.assertFalse(path.exists())
            self.assertTrue(all(stream.closed for stream in opened))
            saved = partial.read_bytes()
            with self.assertRaises(FileExistsError):
                artifacts.write_draw_batch(path, expected, **kwargs)
            self.assertEqual(partial.read_bytes(), saved)

            path = root / "cancelled.json"
            partial = path.with_name(path.name + ".partial")
            def cancel_write():
                if partial.exists() and partial.stat().st_size >= artifacts._CHUNK:
                    raise TimeoutError("selected deadline reached")
            with mock.patch.object(Path, "open", tracking), self.assertRaises(TimeoutError):
                artifacts.write_draw_batch(path, expected, **dict(kwargs, check=cancel_write))
            self.assertTrue(partial.exists())
            self.assertGreater(partial.stat().st_size, 0)
            self.assertLessEqual(partial.stat().st_size, CAP)
            self.assertFalse(path.exists())
            self.assertTrue(all(stream.closed for stream in opened))

            path = root / "complete.json"
            artifacts.write_draw_batch(path, expected, **kwargs)
            with mock.patch.object(Path, "open", tracking), self.assertRaisesRegex(ValueError, "byte_cap"):
                artifacts.read_draw_batch(path, **dict(kwargs, maximum_bytes=path.stat().st_size - 1))
            self.assertTrue(all(stream.closed for stream in opened))
            def cancel_read():
                if any(not stream.closed for stream in opened):
                    raise TimeoutError("read deadline reached")
            with mock.patch.object(Path, "open", tracking), self.assertRaises(TimeoutError):
                artifacts.read_draw_batch(path, **dict(kwargs, check=cancel_read))
            self.assertTrue(all(stream.closed for stream in opened))
            with mock.patch.object(Path, "open") as never_open:
                with self.assertRaises(PermissionError):
                    artifacts.read_draw_batch(path, **dict(kwargs,
                        check=mock.Mock(side_effect=PermissionError("not admitted"))))
                never_open.assert_not_called()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
