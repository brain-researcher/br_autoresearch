"""Small assembler fixtures; run only inside the coordinator's Slurm allocation.

No scientific fragment or shared-state payload is opened. Full-scope tests use
compact interval metadata; coefficient fixtures contain at most64x500 weights.
"""
from pathlib import Path
import io
import json
import tempfile
import time
import unittest
from unittest import mock
import sys
import zipfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_assembly as assembly


class DecoderAssemblyTests(unittest.TestCase):
    def arrays(self):
        native = np.arange(1000, 1600, dtype=np.int64)
        mask = np.arange(600) < 550
        std = np.linspace(1, 2, 600, dtype=np.float32)[None, :]
        std[:, ~mask] = 0
        arrays = {"all_native_voxel_ids": native, "eligible_native_voxel_ids": native[mask],
            "eligible_mask": mask, "neural_mean": np.linspace(-1, 1, 600, dtype=np.float32)[None, :],
            "neural_std": std, "training_image_ids": np.arange(4712, dtype=np.int64),
            "image_repetition_counts": np.ones(4712, dtype=np.int64)}
        binding = {"participant": "sub-01", "role": "subject_unique", "image_mean_weighting": "equal_images",
                   "training_image_ids": list(range(4712)), "preserved_trial_rows_per_image": [[i] for i in range(4712)]}
        return arrays, binding

    def normalization(self):
        arrays, binding = self.arrays()
        return assembly.validate_normalization(arrays, binding, "sub-01")

    def fragment(self, path, ref):
        count = 64 if ref.format == "retained_prefix" else ref.span.count
        origin = 0 if ref.format == "retained_prefix" else ref.span.start
        handle = h5py.File(path, "w")
        if ref.format == "grouped_channels":
            handle.attrs.update(schema_version="ep21.decoder_work_item_fragment.v1", participant=ref.span.participant)
            group = handle.create_group(ref.group)
        else:
            group = handle
            group.attrs["schema_version"] = ("ep21.retained_decoder_prefix_layer.v1"
                if ref.format == "retained_prefix" else "ep21.decoder_continuation_fragment.v1")
        group.attrs.update(layer=ref.span.layer, source_fragment_axes=assembly.AXES)
        if ref.format != "retained_prefix":
            group.attrs.update(participant=ref.span.participant, channel=ref.span.channel,
                spatial_Fortran_start=origin, spatial_Fortran_stop=origin + count)
        spatial = np.arange(origin, origin + count, dtype=np.int64)
        coordinates = [assembly.coverage.coordinates(ref.span.shape, ref.span.channel, int(i)) for i in spatial]
        group.create_dataset("state", data=np.full(count, 2, dtype="u1"))
        ids = np.asarray([np.roll(np.arange(1000, 1500), i) for i in range(count)], dtype="<i8")
        group.create_dataset("selected_native_voxel_ids", data=ids)
        group.create_dataset("weights", data=np.arange(count * 500, dtype=np.float32).reshape(count, 500) / 1000)
        for name, value in (("bias", .25), ("target_mean", 4.), ("target_std", 2.)):
            group.create_dataset(name, data=np.full(count, value, dtype="<f4"))
        group.create_dataset("solve_seconds", data=np.zeros(count, dtype="<f8"))
        group.create_dataset("full_layer_Fortran_index", data=np.asarray([row[1] for row in coordinates], dtype="<i8"))
        if ref.format == "retained_prefix":
            group.create_dataset("tensor_coordinate", data=np.asarray([row[0] for row in coordinates], dtype="<i8"))
        else:
            group.create_dataset("spatial_Fortran_index", data=spatial)
            group.create_dataset("full_layer_C_index", data=np.asarray([row[2] for row in coordinates], dtype="<i8"))
        return handle, group

    def ref(self, format="grouped_channels", layer="conv5_1", channel=511, start=193, stop=196):
        span = assembly.coverage.Span("sub-01", layer, channel, start, stop)
        group = f"{layer}/channel_{channel:03d}" if format == "grouped_channels" else ""
        return assembly.FragmentRef("outputs/decoder_training/fixture/channels.h5", format, group,
                                    "outputs/decoder_training/fixture/status.json", span)

    def test_full_scope_metadata_never_claims_physical_completion(self):
        refs = [assembly.FragmentRef("outputs/decoder_training/fixture/channels.h5", "grouped_channels",
                f"{span.layer}/channel_{span.channel:03d}", "outputs/decoder_training/fixture/status.json", span)
                for span in assembly.coverage.complete_spans()]
        states = [assembly.required_state(person) for person in assembly.coverage.PARTICIPANTS]
        result = assembly.verify_catalog(refs, states)
        self.assertEqual(result["units"], 74306440)
        self.assertFalse(result["complete_decoder"])
        self.assertFalse(result["physical_rows_validated"])
        for malformed in (refs[:-1], refs + refs[:1]):
            with self.assertRaises(RuntimeError):
                assembly.verify_catalog(malformed, states)
        with self.assertRaises(ValueError):
            assembly.verify_catalog(refs, states[:-1])

    def test_normalization_preserves_zero_variance_exclusion_and_binding(self):
        arrays, binding = self.arrays()
        result = assembly.validate_normalization(arrays, binding, "sub-01")
        self.assertEqual(len(result.native_ids), 600)
        self.assertEqual(len(result.eligible_ids), 550)
        for bad in (dict(binding, participant="sub-03"),
                    dict(binding, training_image_ids=list(reversed(binding["training_image_ids"]))),
                    dict(binding, preserved_trial_rows_per_image=[[0]] * 4712)):
            with self.assertRaises(ValueError):
                assembly.validate_normalization(arrays, bad, "sub-01")
        modified = dict(arrays, eligible_mask=np.ones(600, dtype=bool))
        with self.assertRaises(ValueError):
            assembly.validate_normalization(modified, binding, "sub-01")

    def test_npz_reader_never_decodes_large_training_X_or_Gram(self):
        arrays, binding = self.arrays()
        state = assembly.required_state("sub-01")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / state.path
            path.parent.mkdir(parents=True)
            with zipfile.ZipFile(path, "w") as archive:
                for key, value in arrays.items():
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    archive.writestr(key + ".npy", buffer.getvalue())
                # Invalid payload deliberately proves these two arrays are never
                # decoded by this reader, rather than requiring huge fixtures.
                archive.writestr("normalized_X.npy", b"must not be read")
                archive.writestr("penalized_Gram.npy", b"must not be read")
            (Path(temporary) / state.binding_path).write_text(json.dumps(binding))
            result = assembly.load_normalization(temporary, state, lambda: None)
            np.testing.assert_array_equal(result.mean, arrays["neural_mean"][0])

    def test_all_three_formats_keep_F_C_axes_and_ranked_voxel_order(self):
        refs = [self.ref(), self.ref("retained_prefix", "conv1_1", 0, 0, 64),
                self.ref("continuation_fragment", "conv1_1", 0, 64, 67),
                self.ref("grouped_channels", "fc8", 0, 997, 1000)]
        normalization = self.normalization()
        with tempfile.TemporaryDirectory() as temporary:
            for index, ref in enumerate(refs):
                with self.subTest(format=ref.format, layer=ref.span.layer):
                    handle, group = self.fragment(Path(temporary) / f"{index}.h5", ref)
                    try:
                        admitted, origin = assembly.fragment_header(handle, ref)
                        block = assembly.read_sparse_block(admitted, origin, ref, normalization, ref.span.start, ref.span.stop)
                        expected = [assembly.coverage.coordinates(ref.span.shape, ref.span.channel, i)
                                    for i in range(ref.span.start, ref.span.stop)]
                        np.testing.assert_array_equal(block.full_layer_Fortran_index, [item[1] for item in expected])
                        np.testing.assert_array_equal(block.full_layer_C_index, [item[2] for item in expected])
                        np.testing.assert_array_equal(block.selected_native_voxel_ids, group["selected_native_voxel_ids"][:])
                        np.testing.assert_array_equal(normalization.native_ids[block.native_columns], block.selected_native_voxel_ids)
                    finally:
                        handle.close()

    def test_sparse_W_bias_and_training_normalization_match_small_dense_reference(self):
        ref, normalization = self.ref(), self.normalization()
        with tempfile.TemporaryDirectory() as temporary:
            handle, group = self.fragment(Path(temporary) / "test.h5", ref)
            try:
                block = assembly.read_sparse_block(group, ref.span.start, ref, normalization, ref.span.start, ref.span.stop)
                # Only this tiny arithmetic fixture constructs a dense600x3 W.
                dense = np.zeros((600, ref.span.count), dtype=np.float32)
                for unit in range(ref.span.count):
                    dense[block.native_columns[unit], unit] = block.weights[unit]
                normalized_input = np.linspace(-.4, .7, 600, dtype=np.float32)
                raw = normalization.mean + normalization.std * normalized_input
                selected_input = (raw[block.native_columns] - block.neural_mean) / block.neural_std
                sparse = ((selected_input * block.weights).sum(axis=1) + block.bias) * block.target_std + block.target_mean
                reference = (normalized_input @ dense + block.bias) * block.target_std + block.target_mean
                np.testing.assert_allclose(sparse, reference, rtol=2e-6, atol=2e-5)
            finally:
                handle.close()

    def test_pending_duplicate_unknown_or_ineligible_rows_are_rejected(self):
        ref, normalization = self.ref(), self.normalization()
        changes = [("state", (0,), 0), ("selected_native_voxel_ids", (0, 1), 1000),
                   ("selected_native_voxel_ids", (0, 0), 9999),
                   ("selected_native_voxel_ids", (0, 0), 1599),
                   ("weights", (0, 0), np.nan), ("target_std", (0,), 0),
                   ("full_layer_Fortran_index", (0,), 0), ("full_layer_C_index", (0,), 0)]
        with tempfile.TemporaryDirectory() as temporary:
            for index, (name, where, value) in enumerate(changes):
                with self.subTest(dataset=name, value=value):
                    handle, group = self.fragment(Path(temporary) / f"{index}.h5", ref)
                    try:
                        group[name][where] = value
                        with self.assertRaises(ValueError):
                            assembly.read_sparse_block(group, ref.span.start, ref, normalization, ref.span.start, ref.span.stop)
                    finally:
                        handle.close()

    def test_partial_group_reads_only_exact_committed_rows(self):
        ref, normalization = self.ref(), self.normalization()
        partial = assembly.FragmentRef(ref.path, ref.format, ref.group, ref.status_path,
            assembly.coverage.Span(*ref.span.key, ref.span.start, ref.span.start + 1))
        with tempfile.TemporaryDirectory() as temporary:
            handle, group = self.fragment(Path(temporary) / "partial.h5", ref)
            try:
                group["state"][1:] = 0
                group["weights"][1:] = np.nan
                admitted, origin = assembly.fragment_header(handle, partial)
                block = assembly.read_sparse_block(admitted, origin, partial, normalization, partial.span.start, partial.span.stop)
                self.assertEqual(len(block.bias), 1)
                with self.assertRaises(ValueError):
                    assembly.read_sparse_block(group, origin, ref, normalization, ref.span.start, ref.span.stop)
            finally:
                handle.close()

    def test_terminal_record_binds_exact_rows_and_rejects_uncertain_publication(self):
        ref = self.ref()
        state = assembly.required_state("sub-01")
        record = {"schema_version": "ep21.decoder_work_item.v1", "status": "partial_work_item_stopped",
            "participant": "sub-01", "training_role": "subject_unique", "training_images": 4712,
            "predictions_or_scores_computed": False, "tau_train_values_read": False,
            "tau_test_values_read": False, "OOD_access": False,
            "shared_state_reference": str(assembly.BASE / state.path), "committed_spans": [ref.span.record()]}
        assembly.validate_terminal_record(ref, record, state)
        for update in ({"status": "running"}, {"committed_spans": []}, {"completed_units_is_lower_bound": True},
                       {"uncertain_publication": {"phase": "marker"}},
                       {"closeout_errors": [{"affects_fragment": True}]},
                       {"shared_state_reference": "other-participant.npz"}):
            with self.assertRaises(ValueError):
                assembly.validate_terminal_record(ref, dict(record, **update), state)

    def test_soft_or_external_HDF5_links_cannot_expand_access(self):
        ref = self.ref()
        with tempfile.TemporaryDirectory() as temporary:
            handle, group = self.fragment(Path(temporary) / "links.h5", ref)
            try:
                del group["weights"]
                group["weights"] = h5py.SoftLink("/unpermitted")
                with self.assertRaisesRegex(ValueError, "link forbidden"):
                    assembly.fragment_header(handle, ref)
            finally:
                handle.close()
        for path in ("/tmp/outside.h5", "outputs/decoder_training/../closed.h5", "inputs/data.h5"):
            with self.assertRaises(ValueError):
                assembly.safe_relative(path)

    def test_sparse_reader_requires_caller_compute_context_before_access(self):
        with mock.patch.object(assembly, "_read_json") as read:
            stream = assembly.iter_sparse_blocks(None, budget=None,
                                                participant="sub-01", layer="conv1_1")
            with self.assertRaisesRegex(ValueError, "verified compute budget"):
                next(stream)
            read.assert_not_called()
        budget = mock.Mock(allocation={"job_id": "12345", "hostname": "sh04-01n01",
            "job_end_epoch": time.time() + 1000, "allocated_cpus": 16, "memory_gib": 64, "gpus": 1},
            deadline=time.monotonic() + 100)
        with mock.patch.dict(assembly.os.environ, {"SLURM_JOB_ID": "12345"}), \
                mock.patch.object(assembly.socket, "gethostname", return_value="sh04-01n01.stanford.edu"):
            assembly._require_reader_context(budget)
            budget.check.assert_called_once()
            budget.allocation["job_id"] = "12346"
            with self.assertRaisesRegex(ValueError, "actual compute shell"):
                assembly._require_reader_context(budget)

    def test_grouped_scan_opens_one_file_and_status_once_for_multiple_channels(self):
        first, normalization = self.ref(), self.normalization()
        second = self.ref(channel=510)
        state = assembly.required_state("sub-01")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / first.path
            path.parent.mkdir(parents=True)
            handle, group = self.fragment(path, first)
            handle.copy(group, handle["conv5_1"], name="channel_510")
            other = handle[second.group]
            other.attrs["channel"] = 510
            coordinates = [assembly.coverage.coordinates(second.span.shape, 510, i) for i in range(193, 196)]
            other["full_layer_Fortran_index"][:] = [row[1] for row in coordinates]
            other["full_layer_C_index"][:] = [row[2] for row in coordinates]
            handle.close()
            record = {"schema_version": "ep21.decoder_work_item.v1", "status": "complete_work_item_only",
                "participant": "sub-01", "training_role": "subject_unique", "training_images": 4712,
                "predictions_or_scores_computed": False, "tau_train_values_read": False,
                "tau_test_values_read": False, "OOD_access": False,
                "shared_state_reference": str(assembly.BASE / state.path),
                "committed_spans": [first.span.record(), second.span.record()]}
            (Path(temporary) / first.status_path).write_text(json.dumps(record))
            with mock.patch.object(assembly, "_read_json", wraps=assembly._read_json) as read, \
                    mock.patch.object(h5py, "File", wraps=h5py.File) as opened:
                blocks = list(assembly._scan_fragments(temporary, [first, second], state, normalization, lambda: None, 2))
                self.assertEqual(sum(len(block.bias) for block in blocks), 6)
                self.assertEqual(read.call_count, 1)
                self.assertEqual(opened.call_count, 1)

    def test_new_manifest_publication_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            assembly._publish_new_json(path, {"complete_decoder": False}, 1000)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "once-only"):
                assembly._publish_new_json(path, {"complete_decoder": True}, 1000)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_suffix(".json.building").exists())


if __name__ == "__main__":
    unittest.main()
