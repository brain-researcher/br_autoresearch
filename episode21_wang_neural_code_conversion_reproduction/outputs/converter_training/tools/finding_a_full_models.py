"""Dormant original Finding A model preparation under external admission.

The caller supplies already admitted full five-owner tau-test inputs, frozen
decoder axes/4,712-image identities, and separately authenticated 897 tau-train
image identities (metadata only). This module authenticates their joins, not
their provenance. It opens only the fixed published final models and the
existing categorical brain diagnostic; it reads no new neural/feature arrays.

Five within-source, twenty content and twelve original brain inputs retain
all 224 test images. Eight sub01 brain streams remain explicitly unavailable
under the unchanged original rule. Missing expected final models fail the
whole preparation, with no fallback or available-pair reweighting. Each model
is loaded/prepared/released before the next; only prepared float32 arrays and
small identities remain. No fitting, scoring, grant, launcher or publication.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

import final_brain_artifact as brain_artifacts
import final_content_artifact as content_artifacts
import finding_a_first_readout as first
import finding_a_full_empirical_inputs as inputs
import finding_a_prediction as prediction
import schedule


BASE = Path(__file__).resolve().parents[3]
PARTICIPANTS, PAIR_ORDER = schedule.PARTICIPANTS, schedule.PAIR_ORDER
STREAMS = first.collection.STREAMS
NATIVE_VOXELS = inputs.NATIVE_VOXELS
ELIGIBLE_VOXELS = MappingProxyType({**NATIVE_VOXELS, "sub-01": 12294})
CONTENT_BYTES, BRAIN_BYTES = 1073741824, 805306368
BIND = "prepare_full_finding_a_models"
READ_CONTENT = "read_full_finding_a_content_model"
READ_BRAIN = "read_full_finding_a_brain_model"

# Local aliases permit source-free composition fixtures without changing any
# existing reader, loader, prediction implementation or live module globals.
_require = first.require
_require_reader_context = first.assembly._require_reader_context
_brain_disposition = first._brain_disposition
_unavailable_reason = first._unavailable_reason
_load_content = content_artifacts.load_final_content_artifact
_load_brain = brain_artifacts.load_final_brain_artifact
_axis, _artifact_axis, _same_axis = prediction._axis, prediction._artifact_axis, prediction._same_axis
_prepare_within = prediction.prepare_within_source
_prepare_content = prediction.prepare_content
_prepare_brain = prediction.prepare_brain
_converter_factory = schedule.load_pinned_converter_factory


@dataclass(frozen=True)
class PreparedFullModels:
    prepared: Mapping[tuple, prediction.PreparedDecoderInput]
    fitted_object_ids: Mapping[tuple, str | None]
    unavailable_reasons: Mapping[tuple, str]
    image_ids: tuple[str, ...]
    allocation_identity: tuple[str, str]
    _check: Callable = field(repr=False)

    def check(self):
        self._check()


def _model_path(method, source, target):
    """Fixed original attempts only; no discovery, retry or path argument."""
    _require((source, target) in PAIR_ORDER, "original_directed_pair_required")
    stem = f"sub{source[-2:]}_to_sub{target[-2:]}"
    if method == "content_loss":
        run = "cuda001" if (source, target) == ("sub-03", "sub-01") else stem + "_cuda001"
        return BASE / "outputs/runs/content_pair_training" / run / "final_content_pair.pt"
    _require(method == "brain_loss" and "sub-01" not in (source, target),
             "only_twelve_original_evaluable_brain_paths_allowed")
    run = stem + ("_mem16_002" if (source, target) == ("sub-03", "sub-05") else "_001")
    return BASE / "outputs/runs/brain_pair_training" / run / "final_brain_pair.pt"


def _support(bound, axes, training, tau_ids, check):
    """Bind all supplied identities before the first diagnostic/model open."""
    _require(isinstance(bound, inputs.BoundFullFindingAInputs)
             and bound.participants == PARTICIPANTS
             and set(bound.responses_by_participant) == set(PARTICIPANTS)
             and set(axes) == set(training) == set(PARTICIPANTS),
             "complete_five_owner_bound_inputs_axes_and_training_required")
    prediction._ids(bound.image_ids, 224)
    prediction._disjoint(bound.image_ids, tau_ids, 897)
    tau = brain_artifacts.make_common_folds(tau_ids).image_ids
    frozen_axes, frozen_training, unique_training = {}, {}, set()
    for owner in PARTICIPANTS:
        check()
        axis = _axis(axes[owner], check)
        _require(axis.participant == owner and len(axis.full_native_ids) == NATIVE_VOXELS[owner]
                 and len(axis.eligible_native_ids) == ELIGIBLE_VOXELS[owner],
                 "fixed_original_full_native_and_decoder_eligible_widths_required")
        prediction._responses(bound.responses_by_participant[owner], owner,
                              axis.full_native_ids, bound.image_ids, check)
        ids = training[owner]
        prediction._disjoint(bound.image_ids, ids, 4712)
        _require(not set(ids).intersection(tau) and not unique_training.intersection(ids),
                 "original_tau_and_disjoint_subject_unique_training_roles_required")
        unique_training.update(ids)
        frozen_axes[owner], frozen_training[owner] = axis, ids
    check()
    return MappingProxyType(frozen_axes), MappingProxyType(frozen_training), tau


def _content(source, target, bound, axes, training, budget, check):
    path = _model_path("content_loss", source, target)
    read_check = lambda: check(READ_CONTENT, source, target)
    read_check()
    _require(path.resolve() == path.absolute(), "fixed_original_final_model_path_required")
    artifact = None
    try:
        artifact = _load_content(path, budget=budget, maximum_bytes=CONTENT_BYTES)
        read_check()
        initial = artifact["initialization"]
        _require(type(initial["pair_index"]) is int
                 and initial["pair_index"] == PAIR_ORDER.index((source, target))
                 and (initial["source"], initial["target"]) == (source, target)
                 and initial["input_width"] == ELIGIBLE_VOXELS[source]
                 and initial["output_width"] == ELIGIBLE_VOXELS[target]
                 and artifact["completion"]["backend"] == "sparse",
                 "original_complete_sparse_content_pair_required")
        _require(artifact["source_training_image_ids"] == training[source]
                 and artifact["target_decoder_training_image_ids"] == training[target],
                 "original_model_and_decoder_training_identities_required")
        for role, owner in (("source_axis", source), ("target_axis", target)):
            _same_axis(axes[owner], _artifact_axis(artifact[role], read_check))
        prepared = _prepare_content(bound.responses_by_participant[source], artifact, axes[target],
            expected_test_image_ids=bound.image_ids, converter_factory=_converter_factory(), check=read_check)
        read_check()
        return prepared, str(path.relative_to(BASE))
    finally:
        # The unchanged prediction helper retains no model or artifact in its
        # result. Clear this last caller reference even on cancellation/failure.
        artifact = None


def _brain(source, target, bound, axes, training, tau_ids, budget, check):
    path = _model_path("brain_loss", source, target)
    read_check = lambda: check(READ_BRAIN, source, target)
    read_check()
    _require(path.resolve() == path.absolute(), "fixed_original_final_model_path_required")
    artifact = None
    try:
        artifact = _load_brain(path, budget=budget, maximum_bytes=BRAIN_BYTES)
        read_check()
        _require(type(artifact["pair_index"]) is int
                 and artifact["pair_index"] == PAIR_ORDER.index((source, target))
                 and (artifact["summary"]["source"], artifact["summary"]["target"]) == (source, target),
                 "original_complete_brain_pair_required")
        _require(artifact["training_image_ids"] == tau_ids,
                 "original_authenticated_897_tau_train_image_identity_required")
        _require(artifact["source_native_voxel_ids"] == tuple(axes[source].full_native_ids)
                 and artifact["target_native_voxel_ids"] == tuple(axes[target].full_native_ids),
                 "original_complete_brain_and_decoder_native_identity_required")
        prepared = _prepare_brain(bound.responses_by_participant[source], artifact, axes[target],
            target_decoder_training_image_ids=training[target],
            expected_test_image_ids=bound.image_ids, check=read_check)
        read_check()
        return prepared, str(path.relative_to(BASE))
    finally:
        artifact = None


def prepare_full_models(bound_full_inputs, axes, training, *, authenticated_tau_train_image_ids,
                        execution_context, require_admitted_context, budget):
    """Prepare the entire original, explicitly incomplete three-method panel.

    The caller authenticates the 897 tau-train IDs independently of each model
    and supplies the frozen five-owner decoder metadata, not response matrices
    from training. New model-binding/read operations must be separately admitted;
    neither the old first-readout grant nor a training grant admits this call.
    Missing/invalid expected models or revoked access raise without returning a
    partial panel. This helper does not close caller-owned full input bindings.
    """
    _require(execution_context is not None and callable(require_admitted_context),
             "explicit_full_model_context_and_resolver_required")
    identity = None

    def check(operation=BIND, source=None, target=None):
        nonlocal identity
        _require((operation == BIND and source is None and target is None)
                 or (operation in (READ_CONTENT, READ_BRAIN) and (source, target) in PAIR_ORDER
                     and (operation != READ_BRAIN or "sub-01" not in (source, target))),
                 "fixed_full_model_operation_and_original_pair_required")
        result = require_admitted_context(execution_context, operation=operation,
            source=source, target=target, budget=budget, execution_options=None)
        _require(result is None, "external_resolver_must_raise_or_return_None")
        _require_reader_context(budget)
        actual = budget.allocation["job_id"], budget.allocation["hostname"]
        if identity is None:
            identity = actual
        _require(actual == identity, "full_model_preparation_cannot_move_allocations")
        _require(isinstance(bound_full_inputs, inputs.BoundFullFindingAInputs)
                 and bound_full_inputs.execution_context is execution_context
                 and bound_full_inputs.require_admitted_context is require_admitted_context
                 and bound_full_inputs.budget is budget
                 and bound_full_inputs.allocation_identity == identity,
                 "original_full_input_context_resolver_and_allocation_required")
        bound_full_inputs.check()

    check()
    fixed_axes, fixed_training, tau_ids = _support(bound_full_inputs, axes, training,
                                                 authenticated_tau_train_image_ids, check)
    observed_brain_reason = _brain_disposition(check)
    prepared, fitted, unavailable = {}, {stream: None for stream in STREAMS}, {}
    for stream in STREAMS:
        check()
        method, source, target = stream
        if method == "brain_loss" and "sub-01" in (source, target):
            unavailable[stream] = _unavailable_reason(stream, observed_brain_reason)
            continue
        if method == "within_source":
            value = _prepare_within(bound_full_inputs.responses_by_participant[source], fixed_axes[source],
                decoder_training_image_ids=fixed_training[source],
                expected_test_image_ids=bound_full_inputs.image_ids, check=check)
            fitted[stream] = first.assembly.MANIFEST + "#participant=" + source
        elif method == "content_loss":
            value, fitted[stream] = _content(source, target, bound_full_inputs, fixed_axes,
                                           fixed_training, budget, check)
        else:
            value, fitted[stream] = _brain(source, target, bound_full_inputs, fixed_axes,
                                         fixed_training, tau_ids, budget, check)
        _require(isinstance(value, prediction.PreparedDecoderInput)
                 and (value.method, value.source, value.target) == stream
                 and value.image_ids == bound_full_inputs.image_ids,
                 "exact_prepared_original_stream_and_test_image_binding_required")
        _same_axis(value.decoder_axis, fixed_axes[source if target is None else target])
        _require(isinstance(value.values, np.ndarray) and value.values.dtype == np.dtype("float32")
                 and value.values.shape == (224, ELIGIBLE_VOXELS[source if target is None else target])
                 and not value.values.flags.writeable,
                 "complete_readonly_prepared_224_image_decoder_input_required")
        prepared[stream] = value
    check()
    return PreparedFullModels(MappingProxyType(prepared), MappingProxyType(fitted),
        MappingProxyType(unavailable), bound_full_inputs.image_ids, identity, check)
