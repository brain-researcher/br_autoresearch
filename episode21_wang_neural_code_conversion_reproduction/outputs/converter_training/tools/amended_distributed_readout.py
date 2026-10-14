"""Dormant adapted decode/point, 63 draw tasks and complete final join.

Separate operations require future readout authority and explicit finite caps.
No current training grant admits them. Reuse original numerical/statistical
and persistence helpers; keep all224 images,19 layers,855 slots,1000 image
draws,1000 dyadic draws, genuine fitted IDs and the original missingness rules.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path

import amended_full_models as models
import amended_readout_publication as publication
import finding_a_distributed_readout as original


MODEL_POLICY = models.MODEL_POLICY
DECODE = "run_amended_finding_a_decoded_point"
DRAW = "run_amended_finding_a_draw_batch"
FINALIZE = "finalize_amended_finding_a_readout"
collection, inputs, batches = original.collection, original.inputs, original.batches
artifacts, inference, secondary = original.artifacts, original.inference, original.secondary
assembly, require = original.assembly, original.require
_decoder_blocks = original.full._decoder_blocks
_decoder_metadata = original.first._decoder_metadata
_bind_inputs = inputs.bind_full_finding_a_inputs
_prepare_models = models.prepare_full_models


@dataclass(frozen=True)
class AmendedPhaseResult:
    """Future operational reports must retain both phase and model_policy."""
    phase: str
    artifacts: tuple[Path, ...]
    model_policy: str = MODEL_POLICY


def _result(phase, paths):
    return AmendedPhaseResult(phase, tuple(Path(path) for path in paths))


def _expected_fitted():
    result = {}
    for stream in collection.STREAMS:
        method, source, target = stream
        result[stream] = (assembly.MANIFEST + "#participant=" + source if method == "within_source" else
            str(models._model_path(method, source, target).relative_to(models.BASE)))
    return result


def _full_stage(reader, check):
    check()
    require(set(reader.records) == set(collection.SLOTS), "all_855_adapted_stage_slots_required")
    require(dict(reader.fitted_object_ids) == _expected_fitted(), "exact_45_adapted_fitted_streams_required")
    for owner in collection.PARTICIPANTS:
        axis = reader.axes[owner]
        require(axis.participant == owner
                and len(axis.full_native_ids) == models.original.NATIVE_VOXELS[owner]
                and len(axis.eligible_native_ids) == models.original.ELIGIBLE_VOXELS[owner],
                "unchanged_full_native_and_decoder_eligible_axes_required")
    for key in collection.SLOTS:
        check()
        record = reader.records[key]
        require((record["method"], record["source"], record["target"], record["layer"]) == key
                and record["status"] == "decoded" and record["reason"] is None,
                "all_855_expected_adapted_model_layers_must_be_decoded")


@contextmanager
def _decoded_collection(execution_context, *, require_admitted_context, budget,
                        stage_h5, maximum_bytes, authenticated_tau_train_image_ids):
    path = original._absolute(stage_h5)
    require(path.suffix == ".h5", "explicit_adapted_stage_hdf5_path_required")
    check_admission = original._admission(execution_context, require_admitted_context, budget, DECODE,
        dict(collection_path=path, maximum_bytes=original._cap(maximum_bytes), model_policy=MODEL_POLICY))
    bound = prepared = None

    def check():
        check_admission()
        if bound is not None:
            bound.check()

    check()
    collection._require_file_limit(maximum_bytes)
    manifest = assembly.load_complete_manifest(assembly.BASE / assembly.MANIFEST, budget=budget)
    axes, training = _decoder_metadata(manifest, budget, check)
    try:
        bound = _bind_inputs(execution_context, require_admitted_context=require_admitted_context, budget=budget)
        prepared = _prepare_models(bound, axes, training,
            authenticated_tau_train_image_ids=authenticated_tau_train_image_ids,
            execution_context=execution_context, require_admitted_context=require_admitted_context, budget=budget)
        prepared.check()
        require(prepared.model_policy == MODEL_POLICY and not prepared.unavailable_reasons
                and set(prepared.prepared) == set(collection.STREAMS)
                and dict(prepared.fitted_object_ids) == _expected_fitted(),
                "complete_separately_labeled_45_stream_adapted_panel_required")
        with publication.AmendedStagedCollection(path, image_ids=bound.image_ids,
                decoder_axes_by_owner=axes, decoder_training_image_ids_by_owner=training,
                fitted_object_ids=prepared.fitted_object_ids, feature_identity=bound.feature_identity,
                budget=budget, maximum_bytes=maximum_bytes, require_admitted=check) as writer:
            for layer in collection.LAYERS:
                for owner in collection.PARTICIPANTS:
                    check()
                    prepared.check()
                    members = {stream: value for stream, value in prepared.prepared.items()
                               if value.decoder_axis.participant == owner}
                    writer.write_owner_layer(owner=owner, layer=layer, prepared_by_stream=members,
                        blocks=_decoder_blocks(manifest, axes[owner], layer, budget, check))
            writer.seal_decoded()
            prepared = None
            del members
            yield writer, bound, training, check
    finally:
        if bound is not None:
            bound.close()


def run_decoded_point(execution_context, *, require_admitted_context, budget,
                      stage_h5, maximum_bytes, authenticated_tau_train_image_ids):
    """Publish decoded point only; intervals and final scientific report pending."""
    with _decoded_collection(execution_context, require_admitted_context=require_admitted_context,
            budget=budget, stage_h5=stage_h5, maximum_bytes=maximum_bytes,
            authenticated_tau_train_image_ids=authenticated_tau_train_image_ids) as (writer, bound, training, check):
        def provider(**kwargs):
            check()
            return writer.block_provider(**kwargs, truth_provider=bound.truth_reader)
        point = inference.evaluate_panel(provider, training, bound.image_ids,
            check=check, result_sink=writer.result_sink)
        check()
        return _result("decode", writer.publish_decoded_point(point))


def run_draw_batch(execution_context, *, require_admitted_context, budget,
        stage_h5, maximum_collection_bytes, batch_index, batch_path, maximum_batch_bytes):
    """Revalidate bound HDF policy, then evaluate one unchanged canonical batch."""
    batches._batch_start(batch_index)
    stage_h5, batch_path = original._absolute(stage_h5), original._absolute(batch_path)
    options = dict(stage_h5=stage_h5, maximum_collection_bytes=original._cap(maximum_collection_bytes),
        batch_index=batch_index, batch_path=batch_path, maximum_batch_bytes=original._cap(maximum_batch_bytes),
        model_policy=MODEL_POLICY)
    check = original._admission(execution_context, require_admitted_context, budget, DRAW, options)
    with ExitStack() as stack:
        reader = publication.open_decoded_point(stage_h5, budget=budget,
            maximum_bytes=maximum_collection_bytes, require_admitted=check)
        stack.callback(reader.close)
        _full_stage(reader, check)
        truth = inputs.TruthReader(inputs.PROV / "regular_vgg19_sealed/tau_test",
            image_ids=reader.image_ids, feature_identity=reader.feature_identity, check=check)
        stack.callback(truth.close)
        def provider(**kwargs):
            check()
            return reader.block_provider(**kwargs, truth_provider=truth)
        result = batches.evaluate_draw_batch(provider, reader.training, reader.image_ids,
                                             batch_index=batch_index, check=check)
        stage_identity, image_ids = reader.stage_identity, reader.image_ids
        check()
    check()
    path = artifacts.write_draw_batch(batch_path, result, expected_stage_identity=stage_identity,
        expected_batch_index=batch_index, expected_image_ids=image_ids,
        maximum_bytes=maximum_batch_bytes, check=check)
    return _result("draw", (path,))


def run_finalizer(execution_context, *, require_admitted_context, budget,
        stage_h5, final_h5, batch_directory, maximum_collection_bytes,
        maximum_batch_bytes, require_readers_closed):
    """Join all63 batches and unchanged1000-draw inference before one writer."""
    stage_h5, final_h5, batch_directory = map(original._absolute, (stage_h5, final_h5, batch_directory))
    require(callable(require_readers_closed), "external_closed_worker_dependency_required")
    options = dict(stage_h5=stage_h5, final_h5=final_h5, batch_directory=batch_directory,
        maximum_collection_bytes=original._cap(maximum_collection_bytes),
        maximum_batch_bytes=original._cap(maximum_batch_bytes), model_policy=MODEL_POLICY)
    check = original._admission(execution_context, require_admitted_context, budget, FINALIZE, options)
    def closed():
        check()
        require(require_readers_closed() is None, "closed_worker_callback_must_raise_or_return_None")
    closed()
    reader = publication.open_decoded_point(stage_h5, budget=budget,
        maximum_bytes=maximum_collection_bytes, require_admitted=check)
    try:
        _full_stage(reader, check)
        point, identity, image_ids = reader.point, reader.stage_identity, reader.image_ids
    finally:
        reader.close()
    def completed_batches():
        for index in range(batches.BATCH_COUNT):
            check()
            yield artifacts.read_draw_batch(batch_directory / f"draw_batch_{index:03d}.json",
                expected_stage_identity=identity, expected_batch_index=index,
                expected_image_ids=image_ids, maximum_bytes=maximum_batch_bytes, check=check)
    primary = batches.assemble_bootstrap(point, completed_batches(), check=check)
    incoming = secondary.target_incoming(primary.point, check=check)
    dyadic = secondary.dyadic_bootstrap(primary.point, check=check)
    closed()
    paths = publication.finalize_decoded_point(stage_h5, final_h5, primary, incoming, dyadic,
        budget=budget, maximum_bytes=maximum_collection_bytes,
        require_admitted=check, require_readers_closed=closed)
    return _result("finalize", paths)
