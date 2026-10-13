"""UNACTIVATED tau-train input adapter for the fixed brain-loss comparator.

No admission constructor, CLI, fitting, scoring, publication or automatic
continuation. The future external resolver must admit this exact operation and
its resources before I/O. Fixed per-role filenames restrict this reader's I/O;
this helper does not create process isolation or grant filesystem permissions.
No tau-test, subject-unique, feature or decoder payload is requested.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from brain_loss import (
    ALPHAS, TRAINING_IMAGES, BrainLossNotEvaluable, CommonImageFolds,
    TauTrainResponses, make_common_folds,
)
from schedule import PAIR_ORDER, PARTICIPANTS

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / "outputs/decoder_training/tools"))
import decoder_assembly as assembly

PROV = BASE / "outputs/steward_provisioning"
TRIAL_FIELDS = ["output_row_index", "participant", "session", "run", "beta_index",
                "beta_volume_index", "image_id", "role"]
require = assembly.require


@dataclass(frozen=True)
class ParticipantInputMetadata:
    participant: str
    root: Path
    rows: int
    voxels: int
    mask_index_sha256: str
    grid_shape: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BoundBrainInputs:
    """Loaded inputs, not permission to fit or evaluate them."""
    execution_context: object
    require_admitted_context: Callable
    budget: object
    allocation_identity: tuple[str, str]
    folds: CommonImageFolds
    source: TauTrainResponses
    target: TauTrainResponses
    source_repetition_counts: tuple[int, ...]
    target_repetition_counts: tuple[int, ...]


def _admit(context, resolver, source, target, budget, identity=None):
    require(context is not None and callable(resolver), "explicit_external_context_resolver_required")
    require((source, target) in PAIR_ORDER, "canonical_nonself_brain_pair_required")
    result = resolver(context, operation="bind_brain_inputs", source=source, target=target,
                      budget=budget, execution_options=None)
    require(result is None, "external_resolver_must_raise_or_return_None")
    assembly._require_reader_context(budget)
    current = budget.allocation["job_id"], budget.allocation["hostname"]
    require(identity is None or current == identity, "brain_inputs_cannot_move_to_another_allocation")
    return current


def _path(path):
    require(path.resolve() == path.absolute(), "symlinked_brain_input_refused")
    return path


def _read_metadata(path, check):
    check()
    with _path(path).open("rb") as stream:
        raw = stream.read(10_000_001)
    require(len(raw) <= 10_000_000, "brain_input_metadata_exceeds_bound")
    check()
    return json.loads(raw)


def _trial_groups(root, participant, rows, image_ids, check):
    groups = {image: [] for image in image_ids}
    identities = set()
    with _path(root / "tau_train_trials.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == TRIAL_FIELDS, "tau_train_trial_header_mismatch")
        count = 0
        for index, row in enumerate(reader):
            if index % 128 == 0:
                check()
            require(index < rows and row["output_row_index"] == str(index)
                    and row["participant"] == participant and row["role"] == "tau_train"
                    and row["image_id"] in groups, "tau_train_trial_role_or_identity_mismatch")
            identity = row["session"], row["beta_volume_index"]
            require(all(row[key] for key in ("session", "run", "beta_index", "beta_volume_index"))
                    and identity not in identities, "duplicate_or_missing_native_tau_trial_identity")
            identities.add(identity)
            groups[row["image_id"]].append(index)
            count += 1
    require(count == rows and all(groups.values()), "all_tau_train_images_and_repetitions_required")
    return tuple(tuple(groups[image]) for image in image_ids)


def _image_roles():
    # Existing role admission authenticates source_index.csv against the
    # accepted regular-only export and provider tau variant; IDs are metadata.
    sys.path.insert(0, str(BASE / "outputs/pipeline_conformance/tools"))
    import feature_protocol
    return feature_protocol.image_roles()


def _metadata(source, target, check):
    check()
    _, roles, alignment, _ = _image_roles()
    lock = _read_metadata(BASE / "outputs/protocol_lock.yaml", check)
    require(lock.get("state") == "valid_prospective_lock" and lock.get("job_id") == "46424955"
            and lock.get("neural_execution_authorized") is False and lock.get("OOD_access_authorized") is False
            and lock["module_states"]["laion_brain_loss_converters"] == "runnable_locked"
            and alignment == lock["regular_image_binding"], "existing_brain_protocol_and_role_binding_required")
    selection = lock["resolved_protocol_table"]["brain_loss_contract"]["regularization_value_selection_and_intercept"]["selection"]
    require(tuple(selection["alpha_grid"]) == ALPHAS and selection["n_feat"] == 0
            and selection["dtype"] == "float32"
            and selection["required_pairing"] == "separately_image_average_source_and_target_then_exact_join_on_tau_train_image_ID",
            "fixed_brain_comparator_input_policy_required")
    folds = make_common_folds(tuple(row["image_id"] for _, row in roles["tau_train"]))
    report = _read_metadata(PROV / "regular_neural_structural_report.json", check)
    masks = _read_metadata(PROV / "independent_visual_masks/mask_report.json", check)
    require(report.get("publication") == "complete" and report.get("contains_neural_values") is False
            and report.get("OOD_arrays_formed_retained_summarized") is False
            and {row["participant"] for row in report["subjects"]} == set(PARTICIPANTS),
            "complete_regular_neural_structural_export_required")
    result = []
    for participant in (source, target):
        subjects = [row for row in report["subjects"] if row["participant"] == participant]
        mask_subjects = [row for row in masks["subjects"] if row["participant"] == participant]
        require(len(subjects) == len(mask_subjects) == 1, "one_fixed_participant_native_binding_required")
        subject, mask = subjects[0], mask_subjects[0]
        support = subject["roles"]["tau_train"]
        rows, voxels = support["rows"], support["voxels"]
        require(type(rows) is int and rows >= TRAINING_IMAGES and type(voxels) is int and voxels > 0
                and support["dtype"] == "float32" and support["repetitions_preserved"] is True
                and support["finite_mask_bitorder"] == "little"
                and subject["column_order"] == "ascending_full_grid_C_order"
                and voxels == mask["union_voxel_count"]
                and subject["mask_index_sha256"] == mask["voxel_indices_sha256"]
                and subject["native_grid"] == mask["grid"], "complete_frozen_native_tau_support_required")
        if support["nonfinite_elements"] != 0:
            raise BrainLossNotEvaluable("nonfinite_required_tau_train_export:" + participant)
        root = PROV / "regular_neural_sealed" / participant
        groups = _trial_groups(root, participant, rows, folds.image_ids, check)
        result.append(ParticipantInputMetadata(participant, root, rows, voxels,
            subject["mask_index_sha256"], tuple(subject["native_grid"]["shape"]), groups))
    return folds, tuple(result)


def _read_participant(meta, image_ids, check):
    check()
    # Hash only the existing small native-axis binding, never response values.
    with _path(meta.root / "native_voxel_indices.npy").open("rb") as stream:
        raw = stream.read(meta.voxels * 8 + 4097)
    require(len(raw) <= meta.voxels * 8 + 4096
            and hashlib.sha256(raw).hexdigest() == meta.mask_index_sha256,
            "existing_native_index_source_binding_required")
    native = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
    require(native.shape == (meta.voxels,) and native.dtype.kind in "iu"
            and bool((native >= 0).all()) and bool((native[1:] > native[:-1]).all())
            and int(native[-1]) < math.prod(meta.grid_shape), "complete_frozen_native_axis_required")
    values = flags = None
    try:
        check()
        values = np.load(_path(meta.root / "tau_train_responses.npy"), mmap_mode="r", allow_pickle=False,
                         max_header_size=4096)
        flags = np.load(_path(meta.root / "tau_train_finite_packed.npy"), mmap_mode="r", allow_pickle=False,
                        max_header_size=4096)
        require(values.shape == (meta.rows, meta.voxels) and values.dtype.str == "<f4"
                and flags.shape == (meta.rows, (meta.voxels + 7) // 8) and flags.dtype.str == "|u1",
                "full_tau_train_response_and_finite_headers_required")
        averaged = np.empty((TRAINING_IMAGES, meta.voxels), dtype=np.float32)
        # Same averaging arithmetic as train_retained_prefix.aggregate_images;
        # do not invoke its subject-unique normalization/eligible-column filter.
        for image, indices in enumerate(meta.groups):
            check()
            trials = values[list(indices), :]
            finite = np.unpackbits(flags[list(indices), :], axis=1, bitorder="little")[:, :meta.voxels]
            if not finite.all() or not np.isfinite(trials).all():
                raise BrainLossNotEvaluable("nonfinite_required_tau_train_repetition:" + meta.participant)
            with np.errstate(over="ignore", invalid="ignore"):
                averaged[image] = np.mean(trials, axis=0)
            if not np.isfinite(averaged[image]).all():
                raise BrainLossNotEvaluable("nonfinite_required_tau_train_image_mean:" + meta.participant)
        check()
        averaged.setflags(write=False)
        return TauTrainResponses(meta.participant, image_ids, tuple(int(value) for value in native), averaged)
    finally:
        for array in (values, flags):
            if array is not None and getattr(array, "_mmap", None) is not None:
                array._mmap.close()


def bind_brain_inputs(execution_context, *, require_admitted_context, source, target, budget):
    """Admit and separately average the two participants, preserving all897 IDs.

    The resolver owns access/resource admission; the budget owns actual compute
    identity, memory/deadline checks and any I/O accounting. Both participants'
    complete role/row metadata are validated before either response read.
    Fitting, final publication and evaluation require their own later admission.
    """
    identity = _admit(execution_context, require_admitted_context, source, target, budget)
    folds, metadata = _metadata(source, target, budget.check)
    responses = []
    for meta in metadata:
        _admit(execution_context, require_admitted_context, source, target, budget, identity)
        responses.append(_read_participant(meta, folds.image_ids, budget.check))
    _admit(execution_context, require_admitted_context, source, target, budget, identity)
    return BoundBrainInputs(execution_context, require_admitted_context, budget, identity,
        folds, *responses, *(tuple(map(len, meta.groups)) for meta in metadata))
