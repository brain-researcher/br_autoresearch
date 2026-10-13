"""Dormant five-owner Finding A inputs under a separate external admission.

All five complete native tau-test axes are retained. In particular sub-01
has 12,295 raw columns; its 12,294 content-eligible columns are selected only
by the existing prediction normalization using the frozen decoder axis.
All owner metadata/trial joins complete before any neural array is opened.
Then each owner's repetitions are averaged independently in the existing
float32 arithmetic and its read-only memory maps close before the next owner.

This module opens no model, decoder coefficient, training array, image or OOD
input. The one common TruthReader retains its existing single-channel bound.
Importing performs no data I/O and does not activate the full readout.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import io
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

import finding_a_empirical_inputs as first
import finding_a_prediction as prediction


BASE, PROV = first.BASE, first.PROV
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
NATIVE_VOXELS = MappingProxyType({"sub-01": 12295, "sub-03": 12897, "sub-05": 10891,
                                "sub-06": 12023, "sub-07": 12057})
TEST_IMAGES = 224
BIND = "bind_full_finding_a_inputs"
READ_OWNER = "read_full_finding_a_tau_test"
READ_TRUTH = "read_full_finding_a_truth"
OPERATIONS = (BIND, READ_OWNER, READ_TRUTH)

# Reuse unchanged bounded helpers; none of these aliases changes the first
# readout's globals or its fixed sub03 admission/response-reader behavior.
_require, _path, _readonly = first._require, first._path, first._readonly
_read_json = first._read_json
_common_metadata = first._metadata
_require_reader_context = first.interface.assembly._require_reader_context
_load_array = np.load
TruthReader = first.TruthReader


@dataclass(frozen=True)
class OwnerMetadata:
    participant: str
    source: first.SourceMetadata


@dataclass(frozen=True)
class _Admission:
    context: object
    resolver: Callable
    budget: object
    identity: tuple[str, str] | None = None

    def check(self, operation=BIND, participant=None):
        _require(self.context is not None and callable(self.resolver) and operation in OPERATIONS,
                 "external_full_finding_a_admission_required")
        _require((operation == READ_OWNER and participant in PARTICIPANTS)
                 or (operation != READ_OWNER and participant is None),
                 "explicit_fixed_owner_for_tau_test_read_only")
        result = self.resolver(self.context, operation=operation, source=participant, target=None,
                               budget=self.budget, execution_options=None)
        _require(result is None, "external_resolver_must_raise_or_return_None")
        _require_reader_context(self.budget)
        actual = self.budget.allocation["job_id"], self.budget.allocation["hostname"]
        _require(self.identity is None or actual == self.identity,
                 "full_finding_a_inputs_cannot_move_to_another_allocation")
        return actual


@dataclass(frozen=True)
class BoundFullFindingAInputs:
    execution_context: object
    require_admitted_context: Callable
    budget: object
    allocation_identity: tuple[str, str]
    participants: tuple[str, ...]
    image_ids: tuple[str, ...]
    responses_by_participant: Mapping[str, prediction.TauTestResponses]
    repetition_counts_by_participant: Mapping[str, tuple[int, ...]]
    feature_identity: Mapping
    truth_reader: first.TruthReader
    _admission: _Admission = field(repr=False)

    def check(self):
        _require(self.execution_context is self._admission.context
                 and self.require_admitted_context is self._admission.resolver
                 and self.budget is self._admission.budget
                 and self.allocation_identity == self._admission.identity
                 and self.participants == PARTICIPANTS,
                 "original_bound_full_finding_a_context_required")
        self._admission.check()

    def close(self):
        self.truth_reader.close()

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, *unused):
        self.close()


def _trial_groups(root, participant, rows, image_ids, check):
    """Bind this owner's role before grouping; never pair raw repetitions."""
    groups, identities = {image: [] for image in image_ids}, set()
    count = 0
    check()
    with _path(root / "tau_test_trials.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(reader.fieldnames == first.TRIAL_FIELDS, "exact_tau_test_trial_header_required")
        for index, row in enumerate(reader):
            if index % 128 == 0:
                check()
            _require(index < rows and row["output_row_index"] == str(index)
                     and row["participant"] == participant and row["role"] == "tau_test"
                     and row["image_id"] in groups, "fixed_owner_tau_test_row_binding_required")
            identity = row["session"], row["beta_volume_index"]
            _require(all(row[name] for name in ("session", "run", "beta_index", "beta_volume_index"))
                     and identity not in identities, "unique_complete_tau_test_repetitions_required")
            identities.add(identity)
            groups[row["image_id"]].append(index)
            count += 1
    _require(count == rows and all(groups.values()), "all_tau_test_images_and_repetitions_required")
    check()
    return tuple(tuple(groups[image]) for image in image_ids)


def _metadata(check):
    # The unchanged helper authenticates the common locked feature recipe and
    # image order, and the mandatory sub03 record (metadata only). Its fixed
    # sub03 trial binding is reused only for that owner, never retagged.
    check()
    source03, truth = _common_metadata(check)
    _require(isinstance(source03, first.SourceMetadata) and isinstance(truth, first.TruthMetadata)
             and source03.image_ids == truth.image_ids
             and truth.root == PROV / "regular_vgg19_sealed/tau_test",
             "unchanged_common_tau_test_metadata_required")
    prediction._ids(truth.image_ids, TEST_IMAGES)
    neural = _read_json(PROV / "regular_neural_structural_report.json", check)
    masks = _read_json(PROV / "independent_visual_masks/mask_report.json", check)
    _require(neural.get("publication") == "complete" and neural.get("contains_neural_values") is False
             and neural.get("OOD_arrays_formed_retained_summarized") is False,
             "completed_regular_neural_export_required")
    for records in (neural["subjects"], masks["subjects"]):
        _require(len(records) == len(PARTICIPANTS)
                 and {row["participant"] for row in records} == set(PARTICIPANTS),
                 "exact_five_owner_metadata_roster_required")
    owners = {}
    for participant in PARTICIPANTS:
        check()
        subject = next(row for row in neural["subjects"] if row["participant"] == participant)
        mask = next(row for row in masks["subjects"] if row["participant"] == participant)
        support, width = subject["roles"]["tau_test"], NATIVE_VOXELS[participant]
        _require(type(support["rows"]) is int and support["rows"] >= TEST_IMAGES
                 and support["voxels"] == width == mask["union_voxel_count"]
                 and support["dtype"] == "float32" and support["repetitions_preserved"] is True
                 and support["finite_mask_bitorder"] == "little" and support["nonfinite_elements"] == 0
                 and subject["column_order"] == "ascending_full_grid_C_order"
                 and subject["mask_index_sha256"] == mask["voxel_indices_sha256"]
                 and subject["native_grid"] == mask["grid"],
                 "complete_fixed_owner_tau_test_native_support_required")
        root = PROV / "regular_neural_sealed" / participant
        if participant == "sub-03":
            source = source03
            _require(source.root == root and source.rows == support["rows"] and source.voxels == width
                     and source.mask_index_sha256 == subject["mask_index_sha256"]
                     and source.grid_shape == tuple(subject["native_grid"]["shape"]),
                     "common_sub03_metadata_must_bind_its_original_owner")
        else:
            groups = _trial_groups(root, participant, support["rows"], truth.image_ids, check)
            source = first.SourceMetadata(root, support["rows"], width,
                subject["mask_index_sha256"], tuple(subject["native_grid"]["shape"]), groups, truth.image_ids)
        owners[participant] = OwnerMetadata(participant, source)
    check()
    return MappingProxyType(owners), truth


def _read_owner(owner, check):
    """Retain the full native axis, using the existing float32 mean rule."""
    _require(isinstance(owner, OwnerMetadata) and owner.participant in PARTICIPANTS
             and isinstance(owner.source, first.SourceMetadata), "fixed_owner_metadata_required")
    participant, meta = owner.participant, owner.source
    _require(meta.root == PROV / "regular_neural_sealed" / participant
             and meta.voxels == NATIVE_VOXELS[participant]
             and len(meta.groups) == TEST_IMAGES and all(meta.groups),
             "complete_fixed_owner_tau_test_metadata_required")
    prediction._ids(meta.image_ids, TEST_IMAGES)
    positions = [position for group in meta.groups for position in group]
    _require(len(positions) == meta.rows and sorted(positions) == list(range(meta.rows)),
             "every_owner_tau_test_repetition_exactly_once_required")
    check()
    with _path(meta.root / "native_voxel_indices.npy").open("rb") as stream:
        raw = stream.read(meta.voxels * 8 + 4097)
    _require(len(raw) <= meta.voxels * 8 + 4096
             and hashlib.sha256(raw).hexdigest() == meta.mask_index_sha256,
             "existing_owner_native_axis_identity_required")
    check()
    native = _load_array(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
    _require(native.shape == (meta.voxels,) and native.dtype.kind in "iu"
             and bool((native >= 0).all()) and bool((native[1:] > native[:-1]).all())
             and int(native[-1]) < math.prod(meta.grid_shape), "complete_owner_native_axis_required")
    values = flags = None
    try:
        check()
        values = _load_array(_path(meta.root / "tau_test_responses.npy"), mmap_mode="r",
                             allow_pickle=False, max_header_size=4096)
        check()
        flags = _load_array(_path(meta.root / "tau_test_finite_packed.npy"), mmap_mode="r",
                            allow_pickle=False, max_header_size=4096)
        _require(values.shape == (meta.rows, meta.voxels) and values.dtype.str == "<f4"
                 and flags.shape == (meta.rows, (meta.voxels + 7) // 8) and flags.dtype.str == "|u1"
                 and values.flags.c_contiguous and flags.flags.c_contiguous,
                 "exact_owner_tau_test_response_and_flag_headers_required")
        averaged = np.empty((TEST_IMAGES, meta.voxels), dtype=np.float32)
        for image, group in enumerate(meta.groups):
            check()
            trials = values[list(group), :]
            check()
            finite = np.unpackbits(flags[list(group), :], axis=1, bitorder="little")[:, :meta.voxels]
            _require(bool(finite.all()) and bool(np.isfinite(trials).all()),
                     "nonfinite_required_owner_tau_test_repetition")
            with np.errstate(over="ignore", invalid="ignore"):
                averaged[image] = np.mean(trials, axis=0)
            _require(bool(np.isfinite(averaged[image]).all()), "nonfinite_required_owner_tau_test_image_mean")
        check()
        return prediction.TauTestResponses(participant, meta.image_ids,
            tuple(int(value) for value in native), _readonly(averaged))
    finally:
        for array in (values, flags):
            if array is not None and getattr(array, "_mmap", None) is not None:
                array._mmap.close()


def bind_full_finding_a_inputs(execution_context, *, require_admitted_context, budget):
    """Admit the fixed full input roster, then bind all metadata before values.

    Resolver operations are BIND/READ_TRUTH with source=target=None, and
    READ_OWNER with source equal to one fixed participant and target=None.
    Every operation uses execution_options=None and the original allocation.
    No subset, input path, averaging option or eligibility override is accepted.
    """
    initial = _Admission(execution_context, require_admitted_context, budget)
    identity = initial.check()
    admitted = _Admission(execution_context, require_admitted_context, budget, identity)
    owners, truth = _metadata(admitted.check)
    _require(tuple(owners) == PARTICIPANTS, "complete_ordered_full_finding_a_owner_roster_required")
    for participant in PARTICIPANTS:
        _require(owners[participant].participant == participant
                 and owners[participant].source.image_ids == truth.image_ids,
                 "owner_and_common_tau_test_image_identity_required")
    reader = TruthReader(truth.root, image_ids=truth.image_ids, feature_identity=truth.feature_identity,
                         check=lambda: admitted.check(READ_TRUTH))
    responses, counts = {}, {}
    try:
        for participant in PARTICIPANTS:
            responses[participant] = _read_owner(owners[participant],
                lambda owner=participant: admitted.check(READ_OWNER, owner))
            counts[participant] = tuple(map(len, owners[participant].source.groups))
        admitted.check()
        return BoundFullFindingAInputs(execution_context, require_admitted_context, budget, identity,
            PARTICIPANTS, truth.image_ids, MappingProxyType(responses), MappingProxyType(counts),
            truth.feature_identity, reader, admitted)
    except BaseException:
        reader.close()
        raise
