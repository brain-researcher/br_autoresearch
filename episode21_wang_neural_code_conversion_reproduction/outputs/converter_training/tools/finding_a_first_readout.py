"""First original Finding A composition under its separately selected runner.

Only sub-03 within-source and the original final sub-03 -> sub-01 content
model are decoded: all 224 tau-test images, all channels of all 19 layers.
The other 817 slots remain explicit. Original brain pair4 keeps its observed
not-evaluable prerequisite; seven other sub-01 brain pairs retain that shared
source-inferred prerequisite, explicitly not executed. No amended model or
available-pair reweighting is substituted. The existing full 1,000-image-draw
and 1,000-dyad-draw procedures
and target-incoming reduction run unchanged before collection publication.

The caller must separately admit this exact evaluation, its input roles and
finite compute/storage resources. It owns an isolated collection process with
finite RLIMIT_FSIZE and default SIGXFSZ disposition, as required by the existing
native-HDF5 writer. This module changes no limits, starts no process and offers
no point-only shortcut, retry, resume, resource extension or next pair.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import decoder_interface as interface
import final_content_artifact as artifacts
import finding_a_collection as collection
import finding_a_empirical_inputs as inputs
import finding_a_inference as inference
import finding_a_batched_inference as batched
import finding_a_prediction as prediction
import finding_a_secondary as secondary
import schedule


BASE = Path(__file__).resolve().parents[3]
OPERATION = "run_finding_a_first_readout"
SOURCE, TARGET, PAIR_INDEX = "sub-03", "sub-01", 4
FINAL_MODEL = BASE / "outputs/runs/content_pair_training/cuda001/final_content_pair.pt"
FINAL_MODEL_BYTES = 1073741824
BRAIN_DIAGNOSTIC = BASE / "outputs/runs/brain_pair_prefix_diagnostic/attempt002/report.json"
BRAIN_REASON = (
    "not_evaluable_original_brain_pair4_target_fold0_zero_or_nonfinite_required_training_scale;"
    "evidence=outputs/runs/brain_pair_prefix_diagnostic/attempt002/report.json"
)
NOT_SELECTED = "not_in_this_readout"
SELECTED_STREAMS = (("within_source", SOURCE, None), ("content_loss", SOURCE, TARGET))
assembly = interface.assembly
require = interface.require


def _brain_disposition(check):
    """Keep the existing original-model failure; never infer zero SD alone."""
    check()
    require(BRAIN_DIAGNOSTIC.resolve() == BRAIN_DIAGNOSTIC.absolute(),
            "fixed_original_brain_diagnostic_path_required")
    with BRAIN_DIAGNOSTIC.open("rb") as stream:
        raw = stream.read(65537)
    require(len(raw) <= 65536, "bounded_original_brain_diagnostic_required")
    value = json.loads(raw)
    require(isinstance(value, dict)
            and value.get("kind") == "ep21_brain_first_fold_diagnostic"
            and value.get("source") == SOURCE and value.get("target") == TARGET
            and value.get("diagnostic_stage") == "reconstruction-002"
            and value.get("status") == "localized"
            and value.get("failed_stage") == "first_fold_normalization"
            and value.get("error_type") == "BrainLossNotEvaluable"
            and type(value.get("fits_performed")) is int and value["fits_performed"] == 0
            and value.get("scores_produced") is False and value.get("model_published") is False
            and value.get("brain_loss_failure") == {
                "context": "fold_training", "fold": 0,
                "reason_code": "zero_or_nonfinite_required_training_scale", "role": "target"},
            "observed_original_pair4_scale_disposition_required")
    check()
    return BRAIN_REASON


def _load_original_model(budget, check):
    """Require the published fixed original model before tau-test input binding."""
    check()
    require(FINAL_MODEL.resolve() == FINAL_MODEL.absolute(),
            "fixed_original_content_model_path_required")
    artifact = artifacts.load_final_content_artifact(
        FINAL_MODEL, budget=budget, maximum_bytes=FINAL_MODEL_BYTES)
    initial = artifact["initialization"]
    require(type(initial["pair_index"]) is int and initial["pair_index"] == PAIR_INDEX
            and (initial["source"], initial["target"]) == (SOURCE, TARGET)
            and initial["input_width"] == 12897 and initial["output_width"] == 12294
            and artifact["completion"]["backend"] == "sparse",
            "original_complete_sparse_content_pair4_required")
    # The existing loader also validates the 1,024 updates/all4,712 training
    # rows, full nineteen-layer schedule, axes, weights and canonical RNG state.
    check()
    return artifact


def _unavailable_reason(stream, observed_brain_reason):
    """Retain known original prerequisites without inventing executed failures."""
    method, source, target = stream
    if stream == ("brain_loss", SOURCE, TARGET):
        return observed_brain_reason
    if method == "brain_loss" and TARGET in (source, target):
        index = schedule.PAIR_ORDER.index((source, target))
        role = "source" if source == TARGET else "target"
        return (
            f"not_evaluable_source_inferred_not_executed_original_brain_pair{index}_"
            f"sub01_{role}_fold0_zero_or_nonfinite_required_training_scale;"
            "premise=unchanged_participant_inputs_full12295_native_axis_common_folds_runtime_and_float32_moments;"
            "shared_prerequisite_evidence=outputs/runs/brain_pair_prefix_diagnostic/attempt002/report.json;"
            "source_proof=outputs/experiment_log.md#2026-10-11-0449-pacific--interindividual_converter_training--shared-sub01-prerequisite-establishes-eight-brain-pair-dispositions"
        )
    return NOT_SELECTED


def _decoder_metadata(manifest, budget, check):
    """Read all five fixed normalization records, never normalized_X or Gram."""
    axes, training = {}, {}
    for owner in schedule.PARTICIPANTS:
        check()
        _, normalization = interface._state_context(manifest, owner, budget)
        axes[owner] = interface.native_axis(normalization)
        ids = tuple(normalization.training_image_ids.tolist())
        interface.require_image_order(ids, ids)
        training[owner] = ids
    check()
    return axes, training


def _prepare_predictions(bound, artifact, axes, training, budget, check):
    require(bound.source_responses.participant == SOURCE
            and bound.source_responses.role == "tau_test"
            and len(bound.source_responses.native_voxel_ids) == 12897
            and bound.source_responses.values.shape == (224, 12897)
            and bound.source_responses.image_ids == bound.image_ids,
            "fixed_source03_complete_224_image_native_support_required")
    require(tuple(artifact["source_training_image_ids"]) == training[SOURCE]
            and tuple(artifact["target_decoder_training_image_ids"]) == training[TARGET],
            "original_model_and_current_decoder_training_identities_required")
    for role, owner in (("source_axis", SOURCE), ("target_axis", TARGET)):
        prediction._same_axis(axes[owner], prediction._artifact_axis(artifact[role], check))
    check()
    within = prediction.prepare_within_source(bound.source_responses, axes[SOURCE],
        decoder_training_image_ids=training[SOURCE], expected_test_image_ids=bound.image_ids,
        check=check)
    content = prediction.prepare_content(bound.source_responses, artifact, axes[TARGET],
        expected_test_image_ids=bound.image_ids,
        converter_factory=schedule.load_pinned_converter_factory(), check=check)
    check()
    return {SELECTED_STREAMS[0]: within, SELECTED_STREAMS[1]: content}


def _prediction_blocks(manifest, prepared, axis, layer, budget, check):
    """Reuse the real complete-decoder reader; construct no training pair."""
    require(prepared.decoder_axis.participant in (SOURCE, TARGET)
            and prepared.decoder_axis.participant == axis.participant,
            "only_source03_or_target01_coefficient_payloads_allowed")
    shape = assembly.coverage.prefix.SHAPES[collection.LAYERS.index(layer)]
    # Whole-layer traversal preserves all fragments and every channel. The
    # original reader revalidates complete catalog/recovery evidence and owns
    # coefficient-file lifetimes. Explicit F/C identities allow file order.
    raw_blocks = assembly.iter_sparse_blocks(manifest, participant=axis.participant,
        layer=layer, budget=budget, block_units=256)
    try:
        for raw in raw_blocks:
            check()
            require(raw.participant == axis.participant and raw.layer == layer,
                    "exact_admitted_decoder_owner_and_layer_required")
            channel = (int(raw.full_layer_Fortran_index[0] % shape[0])
                       if len(shape) == 3 else None)
            decoder = interface.map_decoder_block(raw, axis, channel=channel)
            predicted = prediction.predict_decoder_block(prepared, decoder,
                check=check, row_chunk_size=64)
            yield predicted, decoder
            check()
    finally:
        close = getattr(raw_blocks, "close", None)
        if callable(close):
            close()


def run_finding_a_first_readout(execution_context, *, require_admitted_context,
        budget, collection_path, maximum_bytes):
    """Publish the existing full-format record for this fixed partial roster.

    Returns the existing collection publisher's (HDF5, manifest, report) paths.
    Missing required pairs remain not evaluable; no complete quantitative
    finding or comparability claim follows from successful artifact publication.
    Scientific/IO/cancellation failures propagate and preserve partial outputs.
    """
    require(execution_context is not None and callable(require_admitted_context),
            "explicit_first_readout_context_and_resolver_required")
    require(type(maximum_bytes) is int and maximum_bytes > 0,
            "explicit_finite_first_readout_collection_cap_required")
    path = Path(collection_path)
    require(path.is_absolute() and path.suffix == ".h5",
            "explicit_absolute_first_readout_hdf5_path_required")
    options = MappingProxyType({"collection_path": path, "maximum_bytes": maximum_bytes})
    allocation_identity = None
    bound = None

    def check():
        nonlocal allocation_identity
        result = require_admitted_context(execution_context, operation=OPERATION,
            source=SOURCE, target=TARGET, budget=budget, execution_options=options)
        require(result is None, "external_resolver_must_raise_or_return_None")
        assembly._require_reader_context(budget)
        actual = budget.allocation["job_id"], budget.allocation["hostname"]
        if allocation_identity is None:
            allocation_identity = actual
        require(actual == allocation_identity, "first_readout_cannot_move_allocations")
        if bound is not None:
            bound.check()

    check()
    # These are caller process prerequisites, checked without changing either.
    collection._require_file_limit(maximum_bytes)
    brain_reason = _brain_disposition(check)
    artifact = _load_original_model(budget, check)
    manifest = assembly.load_complete_manifest(assembly.BASE / assembly.MANIFEST, budget=budget)
    axes, training = _decoder_metadata(manifest, budget, check)
    try:
        bound = inputs.bind_finding_a_inputs(execution_context,
            require_admitted_context=require_admitted_context, budget=budget)
        check()
        prepared = _prepare_predictions(bound, artifact, axes, training, budget, check)
        del artifact
        fitted = {stream: None for stream in collection.STREAMS}
        fitted[SELECTED_STREAMS[0]] = assembly.MANIFEST + "#participant=" + SOURCE
        fitted[SELECTED_STREAMS[1]] = str(FINAL_MODEL.relative_to(BASE))
        with collection.FindingACollection(path, image_ids=bound.image_ids,
                decoder_axes_by_owner=axes, decoder_training_image_ids_by_owner=training,
                fitted_object_ids=fitted, feature_identity=bound.feature_identity,
                budget=budget, maximum_bytes=maximum_bytes, require_admitted=check) as writer:
            for method, source, target, layer in collection.SLOTS:
                check()
                stream = method, source, target
                if stream in SELECTED_STREAMS:
                    model_input = prepared[stream]
                    writer.write_layer(method=method, source=source, target=target, layer=layer,
                        blocks=_prediction_blocks(manifest, model_input,
                            axes[model_input.decoder_axis.participant], layer, budget, check))
                else:
                    writer.unavailable_layer(method=method, source=source, target=target,
                        layer=layer, reason=_unavailable_reason(stream, brain_reason))
            writer.seal_decoded()
            del prepared

            def provider(*, method, source, target, layer):
                check()
                return writer.block_provider(method=method, source=source, target=target,
                    layer=layer, truth_provider=bound.truth_reader)

            primary = batched.bootstrap_panel(provider, training, bound.image_ids,
                check=check, result_sink=writer.result_sink)
            incoming = secondary.target_incoming(primary.point, check=check)
            dyadic = secondary.dyadic_bootstrap(primary.point, check=check)
            check()
            return writer.publish(primary, incoming, dyadic)
    finally:
        if bound is not None:
            bound.close()
