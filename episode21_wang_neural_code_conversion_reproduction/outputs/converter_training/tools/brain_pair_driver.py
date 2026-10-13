"""UNACTIVATED single-pair brain-loss fitting and final-only publication.

Consumes the existing admitted tau-train input container, then separately
resolves the complete fit/publication operation. No CLI, grant constructor,
reader, panel scheduler, retry, scoring or successor is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

import brain_loss as brain
import empirical_brain_inputs as inputs
import final_brain_artifact as final
from schedule import PAIR_ORDER


@dataclass(frozen=True)
class PublishedBrainPair:
    source: str
    target: str
    pair_index: int
    path: Path
    summary: brain.BrainPairSummary


def _admit(bound, options):
    require = inputs.require
    require(isinstance(bound, inputs.BoundBrainInputs)
            and isinstance(bound.source, brain.TauTrainResponses)
            and isinstance(bound.target, brain.TauTrainResponses), "bound_brain_inputs_required")
    pair = bound.source.participant, bound.target.participant
    require(pair in PAIR_ORDER and bound.execution_context is not None
            and callable(bound.require_admitted_context), "external_brain_pair_operation_resolver_required")
    result = bound.require_admitted_context(bound.execution_context,
        operation="train_and_publish_brain_pair", source=pair[0], target=pair[1],
        budget=bound.budget, execution_options=options)
    require(result is None, "external_resolver_must_raise_or_return_None")
    inputs.assembly._require_reader_context(bound.budget)
    identity = bound.budget.allocation["job_id"], bound.budget.allocation["hostname"]
    require(identity == bound.allocation_identity, "brain_pair_cannot_move_to_another_allocation")
    return pair


def train_and_publish_brain_pair(bound, *, final_path, maximum_bytes) -> PublishedBrainPair:
    """Perform the fixed 50 CV fits plus final refit, then publish exactly once.

    The supplied resolver must authorize this operation, not just input
    binding. Existing fit and artifact helpers enforce all897 training images,
    full native axes, complete CV and the final normalization/mapping. No rows,
    columns, folds, alphas or fits are reduced to make a failure publishable.

    Existing final/partial files refuse before fitting. The writer's exclusive
    operation handles any later race. Numerical, deadline and publication
    exceptions propagate; no success object or retry follows a failure.
    """
    options = MappingProxyType({"final_path": Path(final_path), "maximum_bytes": maximum_bytes})
    pair = _admit(bound, options)
    path = final._file_context(options["final_path"], bound.budget, maximum_bytes)
    inputs.require(path.parent.is_dir(), "existing_final_artifact_parent_directory_required")
    for existing in (path, path.with_name(path.name + ".partial")):
        if os.path.lexists(existing):
            raise FileExistsError(existing)
    fit = brain.fit_brain_pair(bound.source, bound.target, bound.folds,
        source_native_ids=bound.source.native_voxel_ids, target_native_ids=bound.target.native_voxel_ids,
        check=bound.budget.check)
    # Fitting cannot extend the original operation or change its allocation.
    _admit(bound, options)
    artifact = final.prepare_final_brain_artifact(fit, expected_pair=pair,
        authenticated_tau_train_image_ids=bound.folds.image_ids,
        source_native_ids=bound.source.native_voxel_ids, target_native_ids=bound.target.native_voxel_ids,
        check=bound.budget.check)
    summary = fit.summary
    del fit
    # Copying the final model also consumes time/memory; recheck immediately
    # before the first publication side effect under the same destination cap.
    _admit(bound, options)
    published = final.write_final_brain_artifact(path, artifact, budget=bound.budget,
                                                maximum_bytes=maximum_bytes)
    return PublishedBrainPair(pair[0], pair[1], PAIR_ORDER.index(pair), published, summary)
