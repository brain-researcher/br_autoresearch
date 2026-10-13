"""Five focused prediction checks; execute only in coordinated Slurm compute.

All 224 test identities and complete tiny native axes are fabricated. Existing
artifact fixtures supply model structure and completion metadata, without fitting
or opening payloads. The numerical references implement the adopted EP21 sequence:
content retains its tagged float32 physical roundtrip; brain uses the common
float32 decoder input, a disclosed difference from literal NCC brain inference.
No fixture qualifies empirical axes, a full decoder collection or test access.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import decoder_interface as interface
import finding_a_prediction as prediction
import test_final_brain_artifact as brain_fixtures
import test_final_content_artifact as content_fixtures
import test_schedule as source_fixtures


TEST_IDS = tuple(f"fabricated-tau-test-{index:03d}" for index in range(224))


def axis_from_artifact(record):
    return interface.NativeAxis(record["participant"],
        *(record[name].numpy().copy() for name in (
            "full_native_ids", "eligible_native_ids", "eligible_full_columns", "mean", "std")))


def fixture():
    helper = content_fixtures.FinalContentArtifactTests()
    content = helper.prepare(helper.fixture())
    source_axis = axis_from_artifact(content["source_axis"])
    target_axis = axis_from_artifact(content["target_axis"])
    width = len(source_axis.full_native_ids)
    values = (np.arange(224 * width, dtype=np.float32).reshape(224, width) / np.float32(137)
              - np.float32(2.75))
    responses = prediction.TauTestResponses(source_axis.participant, TEST_IDS,
        tuple(source_axis.full_native_ids.tolist()), values)

    brain_helper = brain_fixtures.FinalBrainArtifactTests()
    fit = brain_helper.fixture()
    target_width = len(target_axis.full_native_ids)
    matrix = (np.arange((width + 1) * target_width, dtype=np.float32)
              .reshape(width + 1, target_width) % np.float32(17) - np.float32(8)) / np.float32(31)
    fit = replace(fit, source_native_voxel_ids=tuple(source_axis.full_native_ids.tolist()),
        target_native_voxel_ids=tuple(target_axis.full_native_ids.tolist()), matrix=matrix,
        source_mean=np.linspace(-.3, .7, width, dtype=np.float32)[None],
        source_std=np.linspace(.7, 1.9, width, dtype=np.float32)[None],
        target_mean=np.linspace(3., 11., target_width, dtype=np.float32)[None],
        target_std=np.linspace(.8, 2.3, target_width, dtype=np.float32)[None])
    brain = brain_helper.prepare(fit, source_native_ids=fit.source_native_voxel_ids,
                                 target_native_ids=fit.target_native_voxel_ids)
    return SimpleNamespace(content=content, brain=brain, source_axis=source_axis,
                           target_axis=target_axis, responses=responses)


def within(fix, **changes):
    arguments = dict(decoder_training_image_ids=fix.content["source_training_image_ids"],
                     expected_test_image_ids=TEST_IDS, check=lambda: None)
    arguments.update(changes)
    return prediction.prepare_within_source(fix.responses, fix.source_axis, **arguments)


def decoder_block(axis, *, cancellation=False):
    # The block spans the conv5_1 row/column wrap: F and C feature indices have
    # different orders. Three ranked native selections per unit remain unique.
    columns = np.array([[0, 1, 2], [3, 0, 2], [1, 3, 0]], dtype=np.int64)
    weights = np.array([[.2, -.4, .8], [.125, -.75, 1.5], [.3, -.2, .5]], dtype=np.float32)
    if cancellation:
        weights[0] = [1.e8, 1., -1.e8]
    return interface.DecoderBlock(axis.participant, "conv5_1", 3, 13,
        np.array([6659, 7171, 7683], dtype=np.int64),
        np.array([770, 589, 603], dtype=np.int64),
        axis.eligible_native_ids[columns], columns, weights,
        np.array([.125, -.75, 1.5], dtype=np.float32),
        np.array([13., 21., -2.], dtype=np.float32),
        np.array([2., .5, 3.], dtype=np.float32))


def literal_decoder(normalized_input, block):
    """Default FastL2LiR arithmetic in native order, independent of bridge code."""
    weights = np.zeros((normalized_input.shape[1], len(block.bias)), dtype=np.float32)
    for feature in range(len(block.bias)):
        for rank, column in enumerate(block.eligible_columns[feature]):
            weights[int(column), feature] = block.weights[feature, rank]
    # Preserve default float64 X and appended ones while retaining fitted W/b.
    standardized = (np.asarray(normalized_input, dtype=np.float64) @ weights
                    + np.ones((len(normalized_input), 1), dtype=np.float64) @ block.bias[None])
    return standardized, standardized * block.target_std + block.target_mean


class FindingAPredictionTests(unittest.TestCase):
    def test_exact_224_image_order_and_complete_native_binding_are_retained(self):
        fix = fixture()
        actual = within(fix)
        self.assertEqual(actual.image_ids, TEST_IDS)
        self.assertEqual(actual.method, "within_source")
        self.assertEqual(actual.source, fix.source_axis.participant)
        self.assertIsNone(actual.target)
        self.assertEqual(actual.values.shape, (224, len(fix.source_axis.eligible_native_ids)))
        np.testing.assert_array_equal(actual.decoder_axis.full_native_ids, fix.source_axis.full_native_ids)
        self.assertGreater(len(actual.decoder_axis.full_native_ids), actual.values.shape[1])
        for ids in (TEST_IDS[::-1], TEST_IDS[:-1], TEST_IDS[:-1] + TEST_IDS[:1]):
            changed = replace(fix.responses, image_ids=ids)
            with self.subTest(ids=len(ids)), self.assertRaises(ValueError):
                prediction.prepare_within_source(changed, fix.source_axis,
                    decoder_training_image_ids=fix.content["source_training_image_ids"],
                    expected_test_image_ids=TEST_IDS, check=lambda: None)
        for native in (fix.responses.native_voxel_ids[::-1], fix.responses.native_voxel_ids[:-1],
                       (99999,) + fix.responses.native_voxel_ids[1:]):
            with self.subTest(native=native), self.assertRaises(ValueError):
                prediction.prepare_within_source(replace(fix.responses, native_voxel_ids=native),
                    fix.source_axis, decoder_training_image_ids=fix.content["source_training_image_ids"],
                    expected_test_image_ids=TEST_IDS, check=lambda: None)

    def test_within_source_moments_float64_bias_feature_restoration_and_F_C_indices(self):
        fix = fixture()
        # Exact unit-valued normalized inputs expose a float32 projection's
        # 1e8 + 1 - 1e8 cancellation instead of the default float64 result.
        values = fix.responses.values.copy()
        for eligible_column in range(3):
            full_column = fix.source_axis.eligible_full_columns[eligible_column]
            values[:, full_column] = fix.source_axis.mean[eligible_column] + fix.source_axis.std[eligible_column]
        fix.responses = replace(fix.responses, values=values)
        expected_input = ((values[:, fix.source_axis.eligible_full_columns] - fix.source_axis.mean)
                          / fix.source_axis.std).astype(np.float32)
        prepared = within(fix)
        self.assertEqual(prepared.values.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(prepared.values, expected_input)
        block = decoder_block(fix.source_axis, cancellation=True)
        expected_standardized, expected_features = literal_decoder(expected_input, block)
        actual = prediction.predict_decoder_block(prepared, block, check=lambda: None, row_chunk_size=17)
        self.assertEqual(actual.standardized_features.dtype, np.dtype("float64"))
        self.assertEqual(actual.features.dtype, np.dtype("float64"))
        self.assertEqual(actual.source_file_features.dtype, np.dtype("float32"))
        np.testing.assert_allclose(actual.standardized_features, expected_standardized, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.features, expected_features, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual.source_file_features, expected_features.astype(np.float32))
        self.assertTrue(np.any(expected_features != expected_features.astype(np.float32).astype(np.float64)))
        np.testing.assert_array_equal(actual.standardized_features[:, 0], np.full(224, 1.125))
        np.testing.assert_array_equal(actual.features[:, 0], np.full(224, 15.25))
        np.testing.assert_array_equal(actual.full_layer_Fortran_index, [6659, 7171, 7683])
        np.testing.assert_array_equal(actual.full_layer_C_index, [770, 589, 603])
        self.assertEqual(actual.image_ids, TEST_IDS)
        self.assertEqual(actual.decoder_participant, fix.source_axis.participant)
        self.assertEqual((actual.layer, actual.channel), ("conv5_1", 3))

    def test_content_preserves_float32_physical_roundtrip_without_double_normalizing(self):
        fix = fixture()
        artifact = deepcopy(fix.content)
        # Large training means make the tagged float32 physical roundtrip
        # observably different from directly forwarding standardized output.
        artifact["target_axis"]["mean"].fill_(float(2 ** 25))
        artifact["target_axis"]["std"].fill_(2.)
        target_axis = axis_from_artifact(artifact["target_axis"])
        source_input = ((fix.responses.values[:, fix.source_axis.eligible_full_columns]
                         - fix.source_axis.mean) / fix.source_axis.std).astype(np.float32)
        with torch.random.fork_rng(devices=[]):
            model = source_fixtures.SourceConverter(len(fix.source_axis.eligible_native_ids),
                                                    len(target_axis.eligible_native_ids))
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            standardized_output = model(torch.from_numpy(source_input)).numpy()
        physical = standardized_output * target_axis.std + target_axis.mean
        expected = ((physical - target_axis.mean) / target_axis.std).astype(np.float32)
        self.assertEqual(physical.dtype, np.dtype("float32"))
        self.assertGreater(float(np.max(np.abs(expected - standardized_output))), 1e-3)
        wrong_double_normalization = (standardized_output - target_axis.mean) / target_axis.std
        self.assertGreater(float(np.max(np.abs(expected - wrong_double_normalization))), 1e6)
        prepared = prediction.prepare_content(fix.responses, artifact, target_axis,
            expected_test_image_ids=TEST_IDS, converter_factory=source_fixtures.SourceConverter,
            check=lambda: None)
        self.assertEqual(prepared.method, "content_loss")
        self.assertEqual((prepared.source, prepared.target),
                         (fix.source_axis.participant, target_axis.participant))
        np.testing.assert_array_equal(prepared.values, expected)
        block = decoder_block(target_axis)
        actual = prediction.predict_decoder_block(prepared, block, check=lambda: None)
        standardized, features = literal_decoder(expected, block)
        np.testing.assert_allclose(actual.standardized_features, standardized, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.features, features, rtol=0, atol=1e-12)

    def test_brain_tau_physical_and_subject_unique_scales_follow_adopted_dtype_binding(self):
        fix = fixture()
        model = fix.brain
        # Literal adopted sequence: tau normalization, float64 augmented ridge,
        # physical target restoration, then subject-unique decoder normalization.
        tau_source = ((fix.responses.values - model["source_mean"].numpy())
                      / model["source_std"].numpy())
        augmented = np.concatenate((tau_source, np.ones((224, 1))), axis=1)
        tau_target = augmented @ model["matrix"].numpy()
        physical = tau_target * model["target_std"].numpy() + model["target_mean"].numpy()
        decoder_float64 = ((physical[:, fix.target_axis.eligible_full_columns] - fix.target_axis.mean)
                           / fix.target_axis.std)
        expected = decoder_float64.astype(np.float32)
        self.assertEqual(tau_source.dtype, np.dtype("float32"))
        self.assertEqual(tau_target.dtype, np.dtype("float64"))
        prepared = prediction.prepare_brain(fix.responses, model, fix.target_axis,
            target_decoder_training_image_ids=fix.content["target_decoder_training_image_ids"],
            expected_test_image_ids=TEST_IDS, check=lambda: None)
        self.assertEqual(prepared.method, "brain_loss")
        self.assertEqual(prepared.values.dtype, np.dtype("float32"))
        np.testing.assert_allclose(prepared.values, expected, rtol=0, atol=2e-6)
        # Brain-native columns need not share the canonical decoder order.
        # Rebind every target quantity by the same permutation; decoding must
        # still select by native identity, never by the old positional index.
        order = np.roll(np.arange(len(model["target_native_voxel_ids"])), 3)
        reordered = deepcopy(model)
        reordered["target_native_voxel_ids"] = tuple(model["target_native_voxel_ids"][index] for index in order)
        columns = torch.as_tensor(order, dtype=torch.long)
        for name in ("matrix", "target_mean", "target_std"):
            reordered[name] = model[name][:, columns].clone()
        remapped = prediction.prepare_brain(fix.responses, reordered, fix.target_axis,
            target_decoder_training_image_ids=fix.content["target_decoder_training_image_ids"],
            expected_test_image_ids=TEST_IDS, check=lambda: None)
        np.testing.assert_allclose(remapped.values, prepared.values, rtol=0, atol=2e-6)
        self.assertGreater(float(np.max(np.abs(expected - tau_target[:, fix.target_axis.eligible_full_columns]))), 1.)
        block = decoder_block(fix.target_axis)
        standardized, features = literal_decoder(expected, block)
        direct_float64, _ = literal_decoder(decoder_float64, block)
        self.assertGreater(float(np.max(np.abs(standardized - direct_float64))), 1e-9)
        actual = prediction.predict_decoder_block(prepared, block, check=lambda: None, row_chunk_size=29)
        np.testing.assert_allclose(actual.standardized_features, standardized, rtol=0, atol=2e-6)
        np.testing.assert_allclose(actual.features, features, rtol=0, atol=6e-6)
        self.assertEqual(actual.decoder_participant, fix.target_axis.participant)
        self.assertEqual((actual.source, actual.target),
                         (fix.source_axis.participant, fix.target_axis.participant))

    def test_mismatch_overlap_nonfinite_and_cancellation_fail_without_dropping_columns(self):
        fix = fixture()
        prepared = within(fix)
        block = decoder_block(fix.source_axis)
        corrupt_weights = block.weights.copy()
        corrupt_weights[1, 0] = np.nan
        corrupt_ids = block.selected_native_voxel_ids.copy()
        corrupt_ids[0, 0] = fix.source_axis.full_native_ids[1]  # Frozen excluded voxel.
        for bad in (replace(block, participant=fix.target_axis.participant),
                    replace(block, full_layer_C_index=block.full_layer_C_index[::-1]),
                    replace(block, selected_native_voxel_ids=corrupt_ids),
                    replace(block, weights=corrupt_weights),
                    replace(block, target_std=np.array([2., 0., 3.], dtype=np.float32))):
            with self.subTest(bad=bad.participant), self.assertRaises(ValueError):
                prediction.predict_decoder_block(prepared, bad, check=lambda: None)
        corrupt_values = fix.responses.values.copy()
        corrupt_values[17, fix.source_axis.eligible_full_columns[-1]] = np.inf
        with self.assertRaises(ValueError):
            prediction.prepare_within_source(replace(fix.responses, values=corrupt_values), fix.source_axis,
                decoder_training_image_ids=fix.content["source_training_image_ids"],
                expected_test_image_ids=TEST_IDS, check=lambda: None)
        for method, overlap in (("within", fix.content["source_training_image_ids"][0]),
                                ("content", fix.content["target_decoder_training_image_ids"][0]),
                                ("brain", fix.brain["training_image_ids"][0])):
            ids = (overlap,) + TEST_IDS[1:]
            response = replace(fix.responses, image_ids=ids)
            with self.subTest(method=method), self.assertRaises(ValueError):
                if method == "within":
                    prediction.prepare_within_source(response, fix.source_axis,
                        decoder_training_image_ids=fix.content["source_training_image_ids"],
                        expected_test_image_ids=ids, check=lambda: None)
                elif method == "content":
                    prediction.prepare_content(response, fix.content, fix.target_axis,
                        expected_test_image_ids=ids, converter_factory=source_fixtures.SourceConverter,
                        check=lambda: None)
                else:
                    prediction.prepare_brain(response, fix.brain, fix.target_axis,
                        target_decoder_training_image_ids=fix.content["target_decoder_training_image_ids"],
                        expected_test_image_ids=ids, check=lambda: None)
        with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
            within(fix, check=mock.Mock(side_effect=TimeoutError("fixture deadline")))
        calls = []
        def check():
            calls.append(None)
            if len(calls) == 4:
                raise TimeoutError("fixture deadline")
        with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
            prediction.predict_decoder_block(prepared, block, check=check, row_chunk_size=17)
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
