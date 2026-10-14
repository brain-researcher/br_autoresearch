"""Dormant prediction bridge for the separately labeled constant comparator.

No reader, scoring, grant or activation. Caller-admitted arrays retain all224
test identities and the entire model-native axes. Only the frozen decoder
eligibility chooses decoder columns; tau replacement masks never do so.
"""
from __future__ import annotations

import numpy as np

import final_brain_constant_artifact as artifacts
import finding_a_prediction as prediction


MODEL_POLICY = "exact_constant_brain_comparator_all20_fresh_fits"


def prepare_brain(responses, artifact, target_axis, *, target_decoder_training_image_ids,
                  expected_test_image_ids, check):
    """Restore physical target units, then apply the unchanged decoder scale.

    Source centering/scaling stays float32; default float64 appended ones,
    column-block projection and physical restoration follow the original
    bridge. Target decoder normalization stays float64 until its float32 cast.
    """
    check()
    artifacts._validate_artifact(artifact, check)
    source, target = artifact["summary"]["source"], artifact["summary"]["target"]
    target_axis = prediction._axis(target_axis, check)
    prediction._require(target_axis.participant == target, "amended_brain_target_decoder_participant_required")
    target_ids = artifact["target_native_voxel_ids"]
    prediction._require(set(target_ids) == set(target_axis.full_native_ids.tolist()),
                        "complete_amended_brain_target_and_decoder_native_axes_required")
    positions = {native: column for column, native in enumerate(target_ids)}
    eligible = np.asarray([positions[native] for native in target_axis.eligible_native_ids], dtype=np.int64)
    values = prediction._responses(responses, source, artifact["source_native_voxel_ids"],
                                   expected_test_image_ids, check)
    prediction._disjoint(expected_test_image_ids, artifact["training_image_ids"], 897)
    prediction._disjoint(expected_test_image_ids, target_decoder_training_image_ids, 4712)
    source_state, target_state = artifact["source_normalization"], artifact["target_normalization"]
    source_mean, source_scale = (source_state[name].numpy() for name in ("mean", "effective_scale"))
    tau_source = np.empty_like(values)
    for first in range(0, prediction.TEST_IMAGES, prediction._ROWS):
        check()
        tau_source[first:first + prediction._ROWS] = (
            values[first:first + prediction._ROWS] - source_mean) / source_scale
    prediction._finite(tau_source, check)
    augmented = np.hstack((tau_source, np.ones((prediction.TEST_IMAGES, 1))))
    matrix = artifact["matrix"].numpy()
    target_mean, target_scale = (target_state[name].numpy() for name in ("mean", "effective_scale"))
    physical = np.empty((prediction.TEST_IMAGES, len(target_ids)), dtype=np.float64)
    for first in range(0, len(target_ids), prediction.assembly.MAX_BLOCK_UNITS):
        check()
        last = first + prediction.assembly.MAX_BLOCK_UNITS
        projected = np.matmul(augmented, matrix[:, first:last])
        physical[:, first:last] = projected * target_scale[:, first:last] + target_mean[:, first:last]
        check()
    prediction._finite(physical, check)
    normalized = np.empty((prediction.TEST_IMAGES, len(eligible)), dtype=np.float32)
    for first in range(0, prediction.TEST_IMAGES, prediction._ROWS):
        check()
        normalized[first:first + prediction._ROWS] = (
            physical[first:first + prediction._ROWS, eligible] - target_axis.mean) / target_axis.std
    prediction._finite(normalized, check)
    return prediction.PreparedDecoderInput("brain_loss", source, target, target_axis,
        expected_test_image_ids, prediction._readonly(normalized))
