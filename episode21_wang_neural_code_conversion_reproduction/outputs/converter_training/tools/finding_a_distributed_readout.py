"""Dormant explicit decode/point, 63 draw tasks and final-join composition.

Each function is a separately admitted bounded stage, not a scheduler or a
retry/resume mechanism. The first stage publishes no scientific completion;
draw tasks read only that successful collection and common locked truth. The
finalizer requires every original batch and all readers closed before the sole
writer adds resampling and publishes the complete requested report. A missing
batch or expected model cannot produce a smaller analysis. There is one large
HDF inode, with no prediction regeneration or full collection copy.

The caller must authenticate the exact paths, original inputs, allocation,
finite file/resource caps and task dependencies. Aggregate storage accounting
must count hardlink aliases by unique inode. No full-readout execution grant,
resource qualification, automatic submission or process control is provided.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType

import finding_a_collection as collection
import finding_a_draw_artifacts as artifacts
import finding_a_draw_batches as batches
import finding_a_first_readout as first
import finding_a_full_empirical_inputs as inputs
import finding_a_full_models as models
import finding_a_full_readout as full
import finding_a_inference as inference
import finding_a_secondary as secondary
import finding_a_staged_collection as staged


DECODE = "run_full_finding_a_decoded_point"
DRAW = "run_full_finding_a_draw_batch"
FINALIZE = "finalize_full_finding_a_readout"
require, assembly = full.require, full.assembly


def _cap(value):
    require(type(value) is int and value > 0, "explicit_positive_stage_byte_cap_required")
    return value


def _absolute(value):
    path = Path(value)
    require(path.is_absolute(), "explicit_absolute_stage_path_required")
    return path


def _admission(context, resolver, budget, operation, options):
    require(context is not None and callable(resolver), "explicit_distributed_stage_admission_required")
    identity = None
    options = MappingProxyType(dict(options))

    def check():
        nonlocal identity
        result = resolver(context, operation=operation, source=None, target=None,
                          budget=budget, execution_options=options)
        require(result is None, "external_resolver_must_raise_or_return_None")
        assembly._require_reader_context(budget)
        actual = budget.allocation["job_id"], budget.allocation["hostname"]
        if identity is None:
            identity = actual
        require(actual == identity, "distributed_stage_cannot_move_allocations")

    check()
    return check


def _full_stage(reader, check):
    """A component fixture/partial roster cannot masquerade as this full stage."""
    check()
    require(set(reader.records) == set(collection.SLOTS), "full_855_stage_slots_required")
    expected = {}
    for stream in collection.STREAMS:
        method, source, target = stream
        if method == "brain_loss" and "sub-01" in (source, target):
            expected[stream] = None
        elif method == "within_source":
            expected[stream] = assembly.MANIFEST + "#participant=" + source
        else:
            expected[stream] = str(models._model_path(method, source, target).relative_to(models.BASE))
    require(dict(reader.fitted_object_ids) == expected, "exact_original_37_fitted_and_eight_absent_streams_required")
    for owner in collection.PARTICIPANTS:
        axis = reader.axes[owner]
        require(axis.participant == owner
                and len(axis.full_native_ids) == models.NATIVE_VOXELS[owner]
                and len(axis.eligible_native_ids) == models.ELIGIBLE_VOXELS[owner],
                "original_stage_full_native_and_eligible_axes_required")
    decoded = missing = 0
    for key in collection.SLOTS:
        check()
        record = reader.records[key]
        require((record["method"], record["source"], record["target"], record["layer"]) == key,
                "stage_record_original_slot_identity_required")
        if expected[key[:3]] is None:
            require(record["status"] == "unavailable"
                    and record["reason"] == first._unavailable_reason(key[:3], first.BRAIN_REASON),
                    "original_observed_or_inferred_brain_disposition_required")
            missing += 1
        else:
            require(record["status"] == "decoded", "every_expected_original_model_layer_must_be_decoded")
            decoded += 1
    require((decoded, missing) == (703, 152), "original_703_decoded_152_unavailable_stage_required")


def run_decoded_point(execution_context, *, require_admitted_context, budget,
        stage_h5, maximum_bytes, authenticated_tau_train_image_ids):
    """Publish complete decoding and original point vectors, inference pending."""
    with full._decoded_full_collection(execution_context,
            require_admitted_context=require_admitted_context, budget=budget,
            collection_path=stage_h5, maximum_bytes=maximum_bytes,
            authenticated_tau_train_image_ids=authenticated_tau_train_image_ids,
            operation=DECODE, collection_class=staged.StagedFindingACollection) as (writer, bound, training, check):

        def provider(**kwargs):
            check()
            return writer.block_provider(**kwargs, truth_provider=bound.truth_reader)

        point = inference.evaluate_panel(provider, training, bound.image_ids,
            check=check, result_sink=writer.result_sink)
        check()
        return writer.publish_decoded_point(point)


def run_draw_batch(execution_context, *, require_admitted_context, budget,
        stage_h5, maximum_collection_bytes, batch_index, batch_path, maximum_batch_bytes):
    """Evaluate exactly one original batch; publish only after readers close."""
    batches._batch_start(batch_index)
    stage_h5, batch_path = _absolute(stage_h5), _absolute(batch_path)
    options = dict(stage_h5=stage_h5, maximum_collection_bytes=_cap(maximum_collection_bytes),
                   batch_index=batch_index, batch_path=batch_path,
                   maximum_batch_bytes=_cap(maximum_batch_bytes))
    check = _admission(execution_context, require_admitted_context, budget, DRAW, options)
    with ExitStack() as stack:
        reader = staged.open_decoded_point(stage_h5, budget=budget,
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
    return artifacts.write_draw_batch(batch_path, result, expected_stage_identity=stage_identity,
        expected_batch_index=batch_index, expected_image_ids=image_ids,
        maximum_bytes=maximum_batch_bytes, check=check)


def run_finalizer(execution_context, *, require_admitted_context, budget,
        stage_h5, final_h5, batch_directory, maximum_collection_bytes,
        maximum_batch_bytes, require_readers_closed):
    """Join every batch and publish once, after caller-proven reader completion.

    Batch names are fixed draw_batch_000.json through draw_batch_062.json.
    The readers-closed callback is external dependency authority: it must
    raise on missing/failed/live work and return None only for successful
    closure of all63 workers. This library never polls the scheduler.
    """
    stage_h5, final_h5, batch_directory = map(_absolute, (stage_h5, final_h5, batch_directory))
    require(callable(require_readers_closed), "external_closed_worker_dependency_required")
    options = dict(stage_h5=stage_h5, final_h5=final_h5, batch_directory=batch_directory,
                   maximum_collection_bytes=_cap(maximum_collection_bytes),
                   maximum_batch_bytes=_cap(maximum_batch_bytes))
    check = _admission(execution_context, require_admitted_context, budget, FINALIZE, options)

    def closed():
        check()
        require(require_readers_closed() is None, "closed_worker_callback_must_raise_or_return_None")

    closed()
    reader = staged.open_decoded_point(stage_h5, budget=budget,
        maximum_bytes=maximum_collection_bytes, require_admitted=check)
    try:
        _full_stage(reader, check)
        point, stage_identity, image_ids = reader.point, reader.stage_identity, reader.image_ids
    finally:
        reader.close()

    def completed_batches():
        for index in range(batches.BATCH_COUNT):
            check()
            yield artifacts.read_draw_batch(batch_directory / f"draw_batch_{index:03d}.json",
                expected_stage_identity=stage_identity, expected_batch_index=index,
                expected_image_ids=image_ids, maximum_bytes=maximum_batch_bytes, check=check)

    primary = batches.assemble_bootstrap(point, completed_batches(), check=check)
    incoming = secondary.target_incoming(primary.point, check=check)
    dyadic = secondary.dyadic_bootstrap(primary.point, check=check)
    closed()
    return staged.finalize_decoded_point(stage_h5, final_h5, primary, incoming, dyadic,
        budget=budget, maximum_bytes=maximum_collection_bytes,
        require_admitted=check, require_readers_closed=closed)
