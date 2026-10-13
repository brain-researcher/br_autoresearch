"""Source-free I/O fixtures; run only in the coordinator's compute allocation."""
from contextlib import ExitStack
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_empirical_inputs as inputs


IDS = tuple(f"image-{index:03d}" for index in range(224))
SHAPES = tuple((3, 5, 7) if len(shape) == 3 else (11,) for shape in inputs.SHAPES)
IDENTITY = {"role": "tau_test", "dtype": "float32", "extraction_status": "complete",
            "checkpoint": "fixture-checkpoint", "preprocessing": "fixture-preprocessing"}


def table(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def source_fixture(base):
    """Small native axis, full image universe, deliberately non-image-ordered trials."""
    prov = base / "outputs/steward_provisioning"
    root = prov / "regular_neural_sealed/sub-03"
    truth = prov / "regular_vgg19_sealed/tau_test"
    root.mkdir(parents=True)
    truth.mkdir(parents=True)
    order = list(range(223, 0, -1)) + [0, 0, 0]
    values = np.array([[value, value + 1, value + 2] for value in order], dtype=np.float32)
    values[-3:] = np.array([1e8, 1, -1e8], dtype=np.float32)[:, None]
    np.save(root / "native_voxel_indices.npy", np.array([2, 7, 11], dtype=np.int64))
    np.save(root / "tau_test_responses.npy", values)
    np.save(root / "tau_test_finite_packed.npy",
            np.packbits(np.ones(values.shape, dtype=np.uint8), axis=1, bitorder="little"))
    digest = hashlib.sha256((root / "native_voxel_indices.npy").read_bytes()).hexdigest()
    rows = [{"output_row_index": str(index), "participant": "sub-03", "session": "session-a",
             "run": "run-a", "beta_index": str(index), "beta_volume_index": str(index),
             "image_id": IDS[image], "role": "tau_test"} for index, image in enumerate(order)]
    table(root / "tau_test_trials.tsv", inputs.TRIAL_FIELDS, rows)
    groups = tuple(tuple(index for index, image in enumerate(order) if image == number)
                   for number in range(224))
    source = inputs.SourceMetadata(root, len(order), 3, digest, (16,), groups, IDS)
    roles = {"training": [(1, {"image_id": "training-only", "participant": "sub-03", "role": "training"})],
             "tau_test": [(1000 + index, {"image_id": image, "participant": "common", "role": "tau_test"})
                          for index, image in enumerate(IDS)]}
    table(truth / "image_index.tsv", inputs.FEATURE_FIELDS,
          [{"output_row_index": str(index), "regular_HDF5_row": str(1000 + index),
            "image_id": image, "participant": "common", "role": "tau_test"}
           for index, image in enumerate(IDS)])
    recipe = {"layers": list(inputs.LAYERS), "shapes": list(map(list, SHAPES)), "dtype": "float32",
              "finite_mask_bitorder": "little", "checkpoint_members": {"fixture": "fixed"},
              "input_rule": "fixture-preprocessing"}
    alignment = {"fixture": "fixed-image-order"}
    lock = {"state": "valid_prospective_lock", "job_id": "46424955", "neural_execution_authorized": False,
            "OOD_access_authorized": False, "module_states": {"finding_a_feature_decoding": "runnable_locked"},
            "regular_image_binding": alignment, "feature_recipe": recipe}
    neural = {"publication": "complete", "contains_neural_values": False,
              "OOD_arrays_formed_retained_summarized": False, "subjects": [{"participant": "sub-03",
              "roles": {"tau_test": {"rows": len(order), "voxels": 3, "dtype": "float32",
              "repetitions_preserved": True, "finite_mask_bitorder": "little", "nonfinite_elements": 0}},
              "column_order": "ascending_full_grid_C_order", "mask_index_sha256": digest,
              "native_grid": {"shape": [16]}}]}
    masks = {"subjects": [{"participant": "sub-03", "union_voxel_count": 3,
                          "voxel_indices_sha256": digest, "grid": {"shape": [16]}}]}
    features = {"publication": "complete", "OOD_access": False, "feature_recipe": recipe,
                "feature_dtype": "float32", "finite_mask_bitorder": "little", "feature_axes": "NCHW_or_Nvector",
                "HDF5_row_rule": "source_index_CSV_ordinal", "layer_shapes": dict(zip(inputs.LAYERS, map(list, SHAPES))),
                "partitions": [{"partition": role, "images": len(selected), "nonfinite_elements": 0}
                               for role, selected in roles.items()]}
    for path, value in ((base / "outputs/protocol_lock.yaml", lock),
                        (prov / "regular_neural_structural_report.json", neural),
                        (prov / "independent_visual_masks/mask_report.json", masks),
                        (prov / "regular_vgg19_structural_report.json", features)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return source, inputs.TruthMetadata(truth, IDS, IDENTITY), roles, alignment, rows


def truth_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    originals = {}
    with h5py.File(root / "features.h5", "w") as data, h5py.File(root / "finite_packed.h5", "w") as flags:
        for number, (layer, shape) in enumerate(zip(inputs.LAYERS, SHAPES)):
            values = np.arange(224 * int(np.prod(shape)), dtype=np.float32).reshape((224,) + shape)
            values += np.float32(number * 100000)
            if number == 0:
                # C34 shares a packed byte with channel1 but is not requested;
                # C42 is a requested nonfinite unit and must remain present.
                values.reshape(224, -1)[0, [34, 42]] = np.nan
            data.create_dataset(layer, data=values)
            flags.create_dataset(layer, data=np.packbits(np.isfinite(values).reshape(224, -1), axis=1, bitorder="little"))
            originals[layer] = values
    return originals


class FindingAEmpiricalInputsTests(unittest.TestCase):
    def test_admission_metadata_and_complete_source_repetition_axis(self):
        self.assertEqual((inputs.TEST_IMAGES, inputs.NATIVE_VOXELS, inputs.MAX_BLOCK_UNITS,
                          inputs.MAX_TRUTH_BANK_BYTES), (224, 12897, 256, 67108864))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            base = Path(directory).resolve()
            source, truth, roles, alignment, rows = source_fixture(base)
            prov = base / "outputs/steward_provisioning"
            for name, value in (("BASE", base), ("PROV", prov), ("NATIVE_VOXELS", 3), ("SHAPES", SHAPES)):
                stack.enter_context(mock.patch.object(inputs, name, value))
            stack.enter_context(mock.patch.object(inputs, "_image_roles", return_value=(None, roles, alignment, None)))
            stack.enter_context(mock.patch.object(inputs.interface.assembly, "_require_reader_context", side_effect=lambda budget: budget.check()))
            budget = SimpleNamespace(allocation={"job_id": "fixture-job", "hostname": "fixture-node"}, check=lambda: None)
            context, seen = object(), []

            def admitted(actual, **kwargs):
                self.assertIs(actual, context)
                self.assertEqual((kwargs["source"], kwargs["target"]), ("sub-03", "sub-01"))
                self.assertIs(kwargs["budget"], budget)
                self.assertIsNone(kwargs["execution_options"])
                seen.append(kwargs["operation"])

            with mock.patch.object(inputs, "_metadata") as metadata, \
                    mock.patch.object(inputs.np, "load") as load, mock.patch.object(inputs.h5py, "File") as opened:
                with self.assertRaises(PermissionError):
                    inputs.bind_finding_a_inputs(context, require_admitted_context=mock.Mock(side_effect=PermissionError("closed")), budget=budget)
                metadata.assert_not_called(); load.assert_not_called(); opened.assert_not_called()

            # Real metadata parsing refuses wrong roles and feature order before arrays.
            for field, bad in (("role", "training"), ("participant", "sub-01")):
                changed = [dict(row) for row in rows]
                changed[0][field] = bad
                table(source.root / "tau_test_trials.tsv", inputs.TRIAL_FIELDS, changed)
                with mock.patch.object(inputs.np, "load") as load, self.assertRaises(ValueError):
                    inputs.bind_finding_a_inputs(context, require_admitted_context=admitted, budget=budget)
                load.assert_not_called()
            table(source.root / "tau_test_trials.tsv", inputs.TRIAL_FIELDS, rows)
            index = truth.root / "image_index.tsv"
            original_index = index.read_text()
            index.write_text(original_index.replace("image-000", "wrong-image", 1))
            with mock.patch.object(inputs.np, "load") as load, self.assertRaises(ValueError):
                inputs.bind_finding_a_inputs(context, require_admitted_context=admitted, budget=budget)
            load.assert_not_called()
            index.write_text(original_index)
            with mock.patch.object(inputs, "_metadata", return_value=(replace(source, root=source.root.parent / "sub-01"), truth)), \
                    mock.patch.object(inputs.np, "load") as load, self.assertRaises(ValueError):
                inputs.bind_finding_a_inputs(context, require_admitted_context=admitted, budget=budget)
            load.assert_not_called()

            actual_load, maps, paths = np.load, [], []

            def recorded_load(path, **kwargs):
                result = actual_load(path, **kwargs)
                if isinstance(result, np.memmap):
                    maps.append(result); paths.append(Path(path).name)
                    self.assertEqual(kwargs["mmap_mode"], "r")
                return result

            with mock.patch.object(inputs.np, "load", side_effect=recorded_load), \
                    mock.patch.object(inputs.h5py, "File") as unopened:
                bound = inputs.bind_finding_a_inputs(context, require_admitted_context=admitted, budget=budget)
                unopened.assert_not_called()
            try:
                expected = np.array([[value, value + 1, value + 2] for value in range(224)], dtype=np.float32)
                expected[0] = 0  # stored-order float32 mean of [1e8,1,-1e8]
                np.testing.assert_array_equal(bound.source_responses.values, expected)
                self.assertEqual(bound.source_responses.values.dtype, np.float32)
                self.assertFalse(bound.source_responses.values.flags.writeable)
                self.assertEqual(bound.source_responses.native_voxel_ids, (2, 7, 11))
                self.assertEqual(bound.image_ids, IDS)
                self.assertEqual(bound.source_repetition_counts, (3,) + (1,) * 223)
                self.assertEqual(paths, ["tau_test_responses.npy", "tau_test_finite_packed.npy"])
                self.assertTrue(all(array._mmap.closed for array in maps))
                self.assertIn("read_finding_a_tau_test", seen)
                budget.allocation["job_id"] = "moved-job"
                with self.assertRaises(ValueError):
                    bound.check()
            finally:
                bound.close()
            self.assertTrue(bound.truth_reader._closed)

    def test_truth_C_coordinates_packed_offsets_bounded_banks_and_headers(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(inputs, "SHAPES", SHAPES):
            root = Path(directory).resolve()
            original = truth_fixture(root)
            layer = inputs.LAYERS[0]
            # Independent non-square Fortran coordinate association, intentionally unordered.
            coordinates = ((1, 6), (2, 4), (1, 1), (1, 0), (2, 0))
            ids = np.array([channel * 35 + (unit % 5) * 7 + unit // 5 for channel, unit in coordinates])
            expected = np.stack([original[layer][:, channel, unit % 5, unit // 5]
                                 for channel, unit in coordinates], axis=1)
            reads, getitem = [], h5py.Dataset.__getitem__

            def record(dataset, key):
                value = getitem(dataset, key)
                reads.append((dataset.name, key, value.nbytes))
                return value

            reader = inputs.TruthReader(root, image_ids=IDS, feature_identity=IDENTITY, check=lambda: None)
            with mock.patch.object(h5py.Dataset, "__getitem__", record):
                actual = reader(layer=layer, image_ids=IDS, full_layer_C_index=ids, feature_identity=IDENTITY)
                np.testing.assert_array_equal(actual.values, expected)
                np.testing.assert_array_equal(actual.valid_mask, np.isfinite(expected))
                np.testing.assert_array_equal(actual.full_layer_C_index, ids)
                self.assertEqual(actual.values.shape, (224, 5))
                self.assertEqual(actual.values.dtype, np.float32)
                self.assertFalse(actual.values.flags.writeable)
                self.assertFalse(actual.valid_mask.flags.writeable)
                self.assertFalse(actual.full_layer_C_index.flags.writeable)
                self.assertEqual([key for _, key, _ in reads], [
                    (slice(None), 1, slice(None), slice(None)), (slice(None), slice(4, 9)),
                    (slice(None), 2, slice(None), slice(None)), (slice(None), slice(8, 14))])
                self.assertLessEqual(reader._bank.nbytes + reader._packed.nbytes, inputs.MAX_TRUTH_BANK_BYTES)
                reader(layer=layer, image_ids=IDS, full_layer_C_index=np.array([70, 104]), feature_identity=IDENTITY)
                self.assertEqual(len(reads), 4)  # retained last channel, no reread
                fc_ids = np.array([10, 0, 8, 7])
                fc = reader(layer=inputs.LAYERS[-1], image_ids=IDS, full_layer_C_index=fc_ids, feature_identity=IDENTITY)
                np.testing.assert_array_equal(fc.values, original[inputs.LAYERS[-1]][:, fc_ids])
                self.assertEqual(reads[-2][1], (slice(None), slice(None)))
            handles = reader._features, reader._flags
            reader.close()
            self.assertTrue(all(not handle.id.valid for handle in handles))
            self.assertIsNone(reader._bank)

            # A selected NaN with an incorrectly true bit is a binding failure.
            with h5py.File(root / "finite_packed.h5", "r+") as flags:
                flags[layer][0, 42 // 8] = np.uint8(int(flags[layer][0, 42 // 8]) | (1 << (42 % 8)))
            reader = inputs.TruthReader(root, image_ids=IDS, feature_identity=IDENTITY, check=lambda: None)
            with self.assertRaisesRegex(ValueError, "truth_finite_flags"):
                reader(layer=layer, image_ids=IDS, full_layer_C_index=np.array([42]), feature_identity=IDENTITY)
            self.assertTrue(reader._closed)
            with mock.patch.object(inputs, "MAX_TRUTH_BANK_BYTES", 1), \
                    mock.patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("payload before cap")):
                reader = inputs.TruthReader(root, image_ids=IDS, feature_identity=IDENTITY, check=lambda: None)
                with self.assertRaisesRegex(ValueError, "byte_cap"):
                    reader(layer=layer, image_ids=IDS, full_layer_C_index=np.array([35]), feature_identity=IDENTITY)
            with h5py.File(root / "features.h5", "r+") as values:
                del values[inputs.LAYERS[-1]]
                values.create_dataset(inputs.LAYERS[-1], shape=(224, 11), dtype="float64")
            with mock.patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("payload before all headers")):
                reader = inputs.TruthReader(root, image_ids=IDS, feature_identity=IDENTITY, check=lambda: None)
                with self.assertRaisesRegex(ValueError, "headers"):
                    reader(layer=layer, image_ids=IDS, full_layer_C_index=np.array([35]), feature_identity=IDENTITY)
            self.assertTrue(reader._closed)

    def test_cancellation_revocation_and_partial_open_cleanup(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(inputs, "SHAPES", SHAPES):
            source, truth, _, _, _ = source_fixture(Path(directory).resolve())
            truth_fixture(truth.root)
            original_load, maps = np.load, []

            def loading(path, **kwargs):
                result = original_load(path, **kwargs)
                if isinstance(result, np.memmap):
                    maps.append(result)
                return result

            def cancel():
                if len(maps) == 2:
                    raise TimeoutError("fixture cancellation")

            with mock.patch.object(inputs.np, "load", side_effect=loading), self.assertRaises(TimeoutError):
                inputs._read_source(source, cancel)
            self.assertEqual(len(maps), 2)
            self.assertTrue(all(array._mmap.closed for array in maps))
            maps.clear()

            def second_load_failure(path, **kwargs):
                if isinstance(path, Path) and path.name == "tau_test_finite_packed.npy":
                    raise OSError("fixture second map")
                return loading(path, **kwargs)

            with mock.patch.object(inputs.np, "load", side_effect=second_load_failure), self.assertRaises(OSError):
                inputs._read_source(source, lambda: None)
            self.assertEqual(len(maps), 1)
            self.assertTrue(maps[0]._mmap.closed)

            original_file, opened = h5py.File, []

            def second_file_failure(path, *args, **kwargs):
                if Path(path).name == "finite_packed.h5":
                    raise OSError("fixture second HDF")
                handle = original_file(path, *args, **kwargs)
                opened.append(handle)
                return handle

            reader = inputs.TruthReader(truth.root, image_ids=IDS, feature_identity=IDENTITY, check=lambda: None)
            request = dict(layer=inputs.LAYERS[0], image_ids=IDS, full_layer_C_index=np.array([35]), feature_identity=IDENTITY)
            with mock.patch.object(inputs.h5py, "File", side_effect=second_file_failure), self.assertRaises(OSError):
                reader(**request)
            self.assertEqual(len(opened), 1)
            self.assertFalse(opened[0].id.valid)
            self.assertTrue(reader._closed)

            revoked = False

            def still_admitted():
                if revoked:
                    raise PermissionError("fixture revoked")

            reader = inputs.TruthReader(truth.root, image_ids=IDS, feature_identity=IDENTITY, check=still_admitted)
            reader(**request)
            handles = reader._features, reader._flags
            revoked = True
            with mock.patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("read after revocation")), \
                    self.assertRaises(PermissionError):
                reader(**request)
            self.assertTrue(all(not handle.id.valid for handle in handles))
            self.assertIsNone(reader._bank)
            # Binding errors also precede both HDF opens, including cached-read misuse.
            for changed in ({"image_ids": tuple(reversed(IDS))},
                            {"feature_identity": dict(IDENTITY, checkpoint="other")},
                            {"full_layer_C_index": np.array([35, 35])}):
                reader = inputs.TruthReader(truth.root, image_ids=IDS, feature_identity=IDENTITY, check=lambda: None)
                with mock.patch.object(inputs.h5py, "File") as no_open, self.assertRaises(ValueError):
                    reader(**dict(request, **changed))
                no_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
