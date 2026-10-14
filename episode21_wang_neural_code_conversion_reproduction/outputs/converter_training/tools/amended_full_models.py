"""Dormant all20 amended-brain model binding under separate readout admission.

Reuse the original20 content models and five within-source transformations;
load all20 fresh exact-constant comparator artifacts, one at a time. Missing,
invalid or unavailable expected models fail this complete-panel preparation.
There is no original-brain fallback, available-case panel, scoring, empirical
input reader, resource selection or readout authorization in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

import amended_brain_readout as amended
import final_brain_constant_artifact as artifacts
import finding_a_full_models as original


BASE = original.BASE
PARTICIPANTS, PAIR_ORDER, STREAMS = original.PARTICIPANTS, original.PAIR_ORDER, original.STREAMS
MODEL_POLICY = amended.MODEL_POLICY
BIND = "prepare_amended_full_finding_a_models"
READ_CONTENT = original.READ_CONTENT
READ_BRAIN = "read_amended_full_finding_a_brain_model"
BRAIN_BYTES = original.BRAIN_BYTES
_require = original._require
_require_reader_context = original._require_reader_context
_load_brain = artifacts.load_final_brain_constant_artifact
_prepare_brain = amended.prepare_brain


@dataclass(frozen=True)
class PreparedAmendedFullModels:
    model_policy: str
    prepared: Mapping[tuple, original.prediction.PreparedDecoderInput]
    fitted_object_ids: Mapping[tuple, str | None]
    unavailable_reasons: Mapping[tuple, str]
    image_ids: tuple[str, ...]
    allocation_identity: tuple[str, str]
    _check: Callable = field(repr=False)

    def check(self):
        self._check()


def _model_path(method, source, target):
    _require((source, target) in PAIR_ORDER, "fixed_directed_pair_required")
    if method == "content_loss":
        return original._model_path(method, source, target)
    _require(method == "brain_loss", "fixed_amended_brain_method_required")
    index = PAIR_ORDER.index((source, target))
    stem = f"pair{index:03d}_{source.replace('-', '')}_to_{target.replace('-', '')}_001"
    return BASE / "outputs/runs/brain_constant_training" / stem / "final_brain_constant_pair.pt"


def _brain(source, target, bound, axes, training, tau_ids, budget, check):
    path = _model_path("brain_loss", source, target)
    read_check = lambda: check(READ_BRAIN, source, target)
    read_check()
    _require(path.resolve() == path.absolute(), "fixed_amended_final_model_path_required")
    artifact = None
    try:
        artifact = _load_brain(path, budget=budget, maximum_bytes=BRAIN_BYTES)
        read_check()
        _require(artifact["kind"] == artifacts.KIND and artifact["normalization"] == artifacts.NORMALIZATION,
                 "distinct_adopted_constant_comparator_required")
        _require(type(artifact["pair_index"]) is int
                 and artifact["pair_index"] == PAIR_ORDER.index((source, target))
                 and (artifact["summary"]["source"], artifact["summary"]["target"]) == (source, target),
                 "complete_amended_brain_pair_binding_required")
        _require(artifact["training_image_ids"] == tau_ids,
                 "authenticated_897_tau_train_image_identity_required")
        _require(artifact["source_native_voxel_ids"] == tuple(axes[source].full_native_ids)
                 and artifact["target_native_voxel_ids"] == tuple(axes[target].full_native_ids),
                 "complete_amended_brain_and_decoder_native_identity_required")
        prepared = _prepare_brain(bound.responses_by_participant[source], artifact, axes[target],
            target_decoder_training_image_ids=training[target],
            expected_test_image_ids=bound.image_ids, check=read_check)
        read_check()
        return prepared, str(path.relative_to(BASE))
    finally:
        artifact = None


def prepare_full_models(bound_full_inputs, axes, training, *, authenticated_tau_train_image_ids,
                        execution_context, require_admitted_context, budget):
    """Return all45 complete prepared streams, or propagate the first failure.

    The caller must separately admit the new bind/read-brain operations and
    existing input/content reads. A training grant never admits this call.
    The returned policy label must be retained by future adapted collection
    publication; this object is not an original-protocol result.
    """
    _require(execution_context is not None and callable(require_admitted_context),
             "explicit_amended_model_context_and_resolver_required")
    identity = None

    def check(operation=BIND, source=None, target=None):
        nonlocal identity
        _require((operation == BIND and source is None and target is None)
                 or (operation in (READ_CONTENT, READ_BRAIN) and (source, target) in PAIR_ORDER),
                 "fixed_amended_model_operation_and_pair_required")
        result = require_admitted_context(execution_context, operation=operation,
            source=source, target=target, budget=budget, execution_options=None)
        _require(result is None, "external_resolver_must_raise_or_return_None")
        _require_reader_context(budget)
        actual = budget.allocation["job_id"], budget.allocation["hostname"]
        if identity is None:
            identity = actual
        _require(actual == identity, "amended_model_preparation_cannot_move_allocations")
        _require(isinstance(bound_full_inputs, original.inputs.BoundFullFindingAInputs)
                 and bound_full_inputs.execution_context is execution_context
                 and bound_full_inputs.require_admitted_context is require_admitted_context
                 and bound_full_inputs.budget is budget
                 and bound_full_inputs.allocation_identity == identity,
                 "full_input_context_resolver_and_allocation_required")
        bound_full_inputs.check()

    check()
    fixed_axes, fixed_training, tau_ids = original._support(bound_full_inputs, axes, training,
        authenticated_tau_train_image_ids, check)
    prepared, fitted = {}, {}
    for stream in STREAMS:
        check()
        method, source, target = stream
        if method == "within_source":
            value = original._prepare_within(bound_full_inputs.responses_by_participant[source], fixed_axes[source],
                decoder_training_image_ids=fixed_training[source],
                expected_test_image_ids=bound_full_inputs.image_ids, check=check)
            fitted[stream] = original.first.assembly.MANIFEST + "#participant=" + source
        elif method == "content_loss":
            value, fitted[stream] = original._content(source, target, bound_full_inputs, fixed_axes,
                                                     fixed_training, budget, check)
        else:
            value, fitted[stream] = _brain(source, target, bound_full_inputs, fixed_axes,
                                         fixed_training, tau_ids, budget, check)
        _require(isinstance(value, original.prediction.PreparedDecoderInput)
                 and (value.method, value.source, value.target) == stream
                 and value.image_ids == bound_full_inputs.image_ids,
                 "exact_prepared_amended_stream_and_test_image_binding_required")
        owner = source if target is None else target
        original._same_axis(value.decoder_axis, fixed_axes[owner])
        _require(isinstance(value.values, np.ndarray) and value.values.dtype == np.dtype("float32")
                 and value.values.shape == (224, original.ELIGIBLE_VOXELS[owner])
                 and not value.values.flags.writeable,
                 "complete_readonly_prepared_224_image_decoder_input_required")
        prepared[stream] = value
    check()
    return PreparedAmendedFullModels(MODEL_POLICY, MappingProxyType(prepared), MappingProxyType(fitted),
        MappingProxyType({}), bound_full_inputs.image_ids, identity, check)
