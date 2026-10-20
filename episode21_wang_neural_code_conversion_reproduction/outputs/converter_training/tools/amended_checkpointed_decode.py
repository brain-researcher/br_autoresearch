"""Unactivated fresh-decode composition with observed cooperative closure.

Preserve amended_distributed_readout's complete decode -> point -> publication
sequence. This never opens or reuses an old partial and provides no grant, CLI,
scheduler action, or selection of resources. The caller must separately admit
this new top-level operation and all unchanged input/model suboperations.

The supplied budget must check lifecycle.check_work() before its original
resource/allocation checks, including inside helpers that receive only budget.
This composition also checks every resolver/shared callback. The caller owns a
fresh root, actual job/dependency bindings, fatal file policy, both finite clocks
and the hard outer timeout AFTER the cleanup reserve. Lifecycle.run is owned
here, not by the caller. Successful closure is not scientific completion.

Scientific dependencies load only inside admitted execution. A supplied
dependencies object supports control-flow fixtures, not empirical substitutes.
"""
from pathlib import Path
from types import MappingProxyType

from amended_decode_lifecycle import observed_collection_class


OPERATION = "run_checkpointed_amended_finding_a_decoded_point"
MODEL_POLICY = "exact_constant_brain_comparator_all20_fresh_fits"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_dependencies():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import amended_distributed_readout
    return amended_distributed_readout


def run_decoded_point(execution_context, *, require_admitted_context, budget,
                      stage_h5, maximum_bytes, authenticated_tau_train_image_ids,
                      lifecycle, handle_sigterm=False, dependencies=None):
    """Return the original AmendedPhaseResult after lifecycle closeout succeeds.

    Top-level execution_options are the collection path/cap/model policy plus
    work_stop_epoch, hard_stop_epoch and closure_record_path. All delegated
    suboperation names/options and the original budget identity are unchanged.
    """
    path = Path(stage_h5)
    _require(execution_context is not None and callable(require_admitted_context),
             "explicit_checkpointed_decode_admission_required")
    _require(path.is_absolute() and path.suffix == ".h5" and path == lifecycle.collection_path,
             "fresh_lifecycle_collection_path_required")
    _require(type(maximum_bytes) is int and maximum_bytes > 0,
             "explicit_positive_stage_byte_cap_required")
    _require(budget.allocation["job_id"] == lifecycle.job_id,
             "same_lifecycle_and_budget_allocation_required")
    options = MappingProxyType(dict(collection_path=path, maximum_bytes=maximum_bytes,
        model_policy=MODEL_POLICY, work_stop_epoch=lifecycle.work_stop_epoch,
        hard_stop_epoch=lifecycle.hard_stop_epoch, closure_record_path=lifecycle.record_path))

    def checked_resolver(context, **kwargs):
        lifecycle.check_work()
        result = require_admitted_context(context, **kwargs)
        _require(result is None, "external_resolver_must_raise_or_return_None")

    with lifecycle.run(handle_sigterm=handle_sigterm):
        # Reject before importing the scientific dependency tree or opening inputs.
        checked_resolver(execution_context, operation=OPERATION, source=None, target=None,
                         budget=budget, execution_options=options)
        budget.check()
        active = _load_dependencies() if dependencies is None else dependencies
        lifecycle.check_work()
        _require(active.MODEL_POLICY == MODEL_POLICY, "unchanged_amended_model_policy_required")
        check_admission = active.original._admission(execution_context, checked_resolver,
            budget, OPERATION, options)
        bound = prepared = None

        def check():
            lifecycle.check_work()
            check_admission()
            if bound is not None:
                bound.check()

        try:
            check()
            active.collection._require_file_limit(maximum_bytes)
            manifest = active.assembly.load_complete_manifest(
                active.assembly.BASE / active.assembly.MANIFEST, budget=budget)
            check()
            axes, training = active._decoder_metadata(manifest, budget, check)
            bound = active._bind_inputs(execution_context,
                require_admitted_context=checked_resolver, budget=budget)
            check()
            prepared = active._prepare_models(bound, axes, training,
                authenticated_tau_train_image_ids=authenticated_tau_train_image_ids,
                execution_context=execution_context,
                require_admitted_context=checked_resolver, budget=budget)
            check()
            prepared.check()
            active.require(prepared.model_policy == MODEL_POLICY and not prepared.unavailable_reasons
                and set(prepared.prepared) == set(active.collection.STREAMS)
                and dict(prepared.fitted_object_ids) == active._expected_fitted(),
                "complete_separately_labeled_45_stream_adapted_panel_required")
            observed = observed_collection_class(active.publication.AmendedStagedCollection)
            with observed(path, lifecycle=lifecycle, image_ids=bound.image_ids,
                    decoder_axes_by_owner=axes, decoder_training_image_ids_by_owner=training,
                    fitted_object_ids=prepared.fitted_object_ids, feature_identity=bound.feature_identity,
                    budget=budget, maximum_bytes=maximum_bytes, require_admitted=check) as writer:
                for layer in active.collection.LAYERS:
                    for owner in active.collection.PARTICIPANTS:
                        check()
                        prepared.check()
                        members = {stream: value for stream, value in prepared.prepared.items()
                                   if value.decoder_axis.participant == owner}
                        writer.write_owner_layer(owner=owner, layer=layer, prepared_by_stream=members,
                            blocks=active._decoder_blocks(manifest, axes[owner], layer, budget, check))
                writer.seal_decoded()
                prepared = None
                del members

                def provider(**kwargs):
                    check()
                    return writer.block_provider(**kwargs, truth_provider=bound.truth_reader)

                point = active.inference.evaluate_panel(provider, training, bound.image_ids,
                    check=check, result_sink=writer.result_sink)
                check()
                result = active._result("decode", writer.publish_decoded_point(point))
        finally:
            if bound is not None:
                bound.close()
    return result
