"""Dormant complete original Finding A composition; no CLI or execution grant.

The fixed available roster is five within-source, twenty content and twelve
brain streams. All eight original sub01 brain prerequisites remain explicit,
including the observed versus source-inferred distinction. Every one of the
855 original method/pair/layer slots is retained. Missing expected models
fail preparation; they cannot become an available-case replacement analysis.

Each owner/layer decoder is read once and applied separately to its prepared
streams by GroupedFindingACollection. Original inference, point vectors,
1,000 image draws, 1,000 dyadic draws and incoming summaries are retained.
This composition uses the existing bounded batched inference in one process;
distributed persistence/dispatch is separate unfinished engineering. Full
runtime and storage remain unqualified. The first readout's grant does not
admit this function or its inputs. An external caller must select full finite
resources, input roles and isolated fatal native-I/O policy before execution.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import decoder_interface as interface
import finding_a_batched_inference as batched
import finding_a_collection as collection
import finding_a_first_readout as first
import finding_a_full_empirical_inputs as inputs
import finding_a_full_models as models
import finding_a_grouped_collection as grouped
import finding_a_secondary as secondary


OPERATION = "run_full_finding_a_readout"
assembly, require = interface.assembly, interface.require


def _decoder_blocks(manifest, axis, layer, budget, check):
    """Visit each owner's complete layer once; preserve reader block ordering."""
    require(axis.participant in collection.PARTICIPANTS,
            "original_five_owner_decoder_required")
    shape = assembly.coverage.prefix.SHAPES[collection.LAYERS.index(layer)]
    check()
    raw_blocks = assembly.iter_sparse_blocks(manifest, participant=axis.participant,
        layer=layer, budget=budget, block_units=256)
    try:
        for raw in raw_blocks:
            check()
            require(raw.participant == axis.participant and raw.layer == layer,
                    "exact_full_readout_decoder_owner_and_layer_required")
            channel = (int(raw.full_layer_Fortran_index[0] % shape[0])
                       if len(shape) == 3 else None)
            yield interface.map_decoder_block(raw, axis, channel=channel)
            check()
    finally:
        close = getattr(raw_blocks, "close", None)
        if callable(close):
            close()


@contextmanager
def _decoded_full_collection(execution_context, *, require_admitted_context,
        budget, collection_path, maximum_bytes, authenticated_tau_train_image_ids,
        operation, collection_class):
    """Internal shared decode lifetime for separately admitted full stages."""
    require(execution_context is not None and callable(require_admitted_context),
            "explicit_full_readout_context_and_resolver_required")
    require(type(maximum_bytes) is int and maximum_bytes > 0,
            "explicit_finite_full_readout_collection_cap_required")
    path = Path(collection_path)
    require(path.is_absolute() and path.suffix == ".h5",
            "explicit_absolute_full_readout_hdf5_path_required")
    options = MappingProxyType({"collection_path": path, "maximum_bytes": maximum_bytes})
    identity = None
    bound = prepared = None

    def check():
        nonlocal identity
        result = require_admitted_context(execution_context, operation=operation,
            source=None, target=None, budget=budget, execution_options=options)
        require(result is None, "external_resolver_must_raise_or_return_None")
        assembly._require_reader_context(budget)
        actual = budget.allocation["job_id"], budget.allocation["hostname"]
        if identity is None:
            identity = actual
        require(actual == identity, "full_readout_cannot_move_allocations")
        if bound is not None:
            bound.check()

    check()
    collection._require_file_limit(maximum_bytes)
    manifest = assembly.load_complete_manifest(assembly.BASE / assembly.MANIFEST, budget=budget)
    axes, training = first._decoder_metadata(manifest, budget, check)
    try:
        bound = inputs.bind_full_finding_a_inputs(execution_context,
            require_admitted_context=require_admitted_context, budget=budget)
        prepared = models.prepare_full_models(bound, axes, training,
            authenticated_tau_train_image_ids=authenticated_tau_train_image_ids,
            execution_context=execution_context, require_admitted_context=require_admitted_context,
            budget=budget)
        prepared.check()
        check()
        with collection_class(path, image_ids=bound.image_ids,
                decoder_axes_by_owner=axes, decoder_training_image_ids_by_owner=training,
                fitted_object_ids=prepared.fitted_object_ids,
                feature_identity=bound.feature_identity, budget=budget,
                maximum_bytes=maximum_bytes, require_admitted=check) as writer:
            for layer in collection.LAYERS:
                for owner in collection.PARTICIPANTS:
                    check()
                    prepared.check()
                    members = {stream: value for stream, value in prepared.prepared.items()
                               if value.decoder_axis.participant == owner}
                    writer.write_owner_layer(owner=owner, layer=layer,
                        prepared_by_stream=members,
                        blocks=_decoder_blocks(manifest, axes[owner], layer, budget, check))
                for stream, reason in prepared.unavailable_reasons.items():
                    check()
                    method, source, target = stream
                    writer.unavailable_layer(method=method, source=source, target=target,
                        layer=layer, reason=reason)
            writer.seal_decoded()
            prepared = None
            del members
            yield writer, bound, training, check
    finally:
        if bound is not None:
            bound.close()


def run_full_finding_a_readout(execution_context, *, require_admitted_context,
        budget, collection_path, maximum_bytes, authenticated_tau_train_image_ids):
    """Complete original records with unchanged unavailable comparisons.

    The caller authenticates the original897 tau-train image identities as
    metadata, separately from4712 decoder training IDs. No training response
    payload, fitting, OOD or scientific amendment is opened by this composition.
    Returns the existing publisher's HDF5/manifest/report tuple only after all
    requested inference. No point-only success, retry or automatic successor.
    """
    with _decoded_full_collection(execution_context,
            require_admitted_context=require_admitted_context, budget=budget,
            collection_path=collection_path, maximum_bytes=maximum_bytes,
            authenticated_tau_train_image_ids=authenticated_tau_train_image_ids,
            operation=OPERATION, collection_class=grouped.GroupedFindingACollection) as (writer, bound, training, check):

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
