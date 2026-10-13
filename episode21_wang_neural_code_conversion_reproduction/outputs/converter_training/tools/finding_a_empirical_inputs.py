"""First Finding A input adapter under its separately admitted external runner.

The external resolver must admit the fixed sub03 -> sub01 readout in its
actual allocation. Only sub03 tau-test responses and common tau-test feature
values are read here. No model, decoder coefficients, training arrays, target
neural responses, images or OOD data are opened. Importing performs no I/O.

The truth reader retains at most one physical convolutional channel or one
fully connected layer, not a whole-layer convolutional or 19-layer bank.
It returns the requested explicit C-unit order, including original finite
flags; prediction/metric code owns the separate Fortran-unit association.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import h5py
import numpy as np

import decoder_interface as interface
import finding_a_collection as collection
import finding_a_prediction as prediction
from empirical_brain_inputs import _image_roles


BASE = Path(__file__).resolve().parents[3]
PROV = BASE / "outputs/steward_provisioning"
SOURCE, TARGET = "sub-03", "sub-01"
TEST_IMAGES, NATIVE_VOXELS = 224, 12897
LAYERS = tuple(interface.assembly.coverage.prefix.LAYERS)
SHAPES = tuple(interface.assembly.coverage.prefix.SHAPES)
MAX_BLOCK_UNITS = interface.assembly.MAX_BLOCK_UNITS
MAX_TRUTH_BANK_BYTES = 64 * 1024**2
OPERATIONS = ("bind_finding_a_inputs", "read_finding_a_tau_test", "read_finding_a_truth")
TRIAL_FIELDS = ["output_row_index", "participant", "session", "run", "beta_index",
                "beta_volume_index", "image_id", "role"]
FEATURE_FIELDS = ["output_row_index", "regular_HDF5_row", "image_id", "participant", "role"]


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _readonly(value):
    value.setflags(write=False)
    return value


def _path(path):
    path = Path(path)
    _require(path.is_absolute() and path.resolve() == path,
             "fixed_nonsymlink_finding_a_input_required")
    return path


def _read_json(path, check):
    check()
    with _path(path).open("rb") as stream:
        data = stream.read(10_000_001)
    _require(len(data) <= 10_000_000, "bounded_finding_a_metadata_required")
    check()
    return json.loads(data)


@dataclass(frozen=True)
class SourceMetadata:
    root: Path
    rows: int
    voxels: int
    mask_index_sha256: str
    grid_shape: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class TruthMetadata:
    root: Path
    image_ids: tuple[str, ...]
    feature_identity: Mapping


@dataclass(frozen=True)
class _Admission:
    context: object
    resolver: Callable
    budget: object
    identity: tuple[str, str] | None = None

    def check(self, operation=OPERATIONS[0]):
        _require(self.context is not None and callable(self.resolver) and operation in OPERATIONS,
                 "external_fixed_finding_a_admission_required")
        result = self.resolver(self.context, operation=operation, source=SOURCE, target=TARGET,
                               budget=self.budget, execution_options=None)
        _require(result is None, "external_resolver_must_raise_or_return_None")
        interface.assembly._require_reader_context(self.budget)
        actual = self.budget.allocation["job_id"], self.budget.allocation["hostname"]
        _require(self.identity is None or actual == self.identity,
                 "finding_a_inputs_cannot_move_to_another_allocation")
        return actual


@dataclass(frozen=True)
class BoundFindingAInputs:
    execution_context: object
    require_admitted_context: Callable
    budget: object
    allocation_identity: tuple[str, str]
    image_ids: tuple[str, ...]
    source_responses: prediction.TauTestResponses
    source_repetition_counts: tuple[int, ...]
    feature_identity: Mapping
    truth_reader: TruthReader
    _admission: _Admission = field(repr=False)

    def check(self):
        _require(self.execution_context is self._admission.context
                 and self.require_admitted_context is self._admission.resolver
                 and self.budget is self._admission.budget
                 and self.allocation_identity == self._admission.identity,
                 "original_bound_finding_a_context_required")
        self._admission.check()

    def close(self):
        self.truth_reader.close()

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, *unused):
        self.close()


def _trial_groups(root, rows, image_ids, check):
    groups = {image: [] for image in image_ids}
    identities = set()
    count = 0
    check()
    with _path(root / "tau_test_trials.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(reader.fieldnames == TRIAL_FIELDS, "exact_tau_test_trial_header_required")
        for index, row in enumerate(reader):
            if index % 128 == 0:
                check()
            _require(index < rows and row["output_row_index"] == str(index)
                     and row["participant"] == SOURCE and row["role"] == "tau_test"
                     and row["image_id"] in groups, "fixed_source_tau_test_row_binding_required")
            identity = row["session"], row["beta_volume_index"]
            _require(all(row[name] for name in ("session", "run", "beta_index", "beta_volume_index"))
                     and identity not in identities, "unique_complete_tau_test_repetitions_required")
            identities.add(identity)
            groups[row["image_id"]].append(index)
            count += 1
    _require(count == rows and all(groups.values()), "all_tau_test_images_and_repetitions_required")
    check()
    return tuple(tuple(groups[image]) for image in image_ids)


def _feature_index(root, selected, check):
    check()
    with _path(root / "image_index.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(reader.fieldnames == FEATURE_FIELDS, "exact_tau_test_feature_index_header_required")
        for index, (ordinal, expected) in enumerate(selected):
            check()
            row = next(reader, None)
            _require(row == {"output_row_index": str(index), "regular_HDF5_row": str(ordinal),
                            "image_id": expected["image_id"], "participant": expected["participant"],
                            "role": "tau_test"}, "exact_tau_test_feature_image_order_required")
        _require(next(reader, None) is None, "no_additional_tau_test_feature_rows_allowed")


def _metadata(check):
    check()
    _, roles, alignment, _ = _image_roles()
    check()
    lock = _read_json(BASE / "outputs/protocol_lock.yaml", check)
    _require(lock.get("state") == "valid_prospective_lock" and lock.get("job_id") == "46424955"
             and lock.get("neural_execution_authorized") is False
             and lock.get("OOD_access_authorized") is False
             and lock["module_states"]["finding_a_feature_decoding"] == "runnable_locked"
             and lock["regular_image_binding"] == alignment, "unchanged_original_finding_a_lock_required")
    selected = roles["tau_test"]
    image_ids = tuple(row["image_id"] for _, row in selected)
    _require(len(image_ids) == TEST_IMAGES and len(set(image_ids)) == TEST_IMAGES
             and all(row["role"] == "tau_test" for _, row in selected),
             "exact_authenticated_224_tau_test_image_universe_required")
    for key, rows in roles.items():
        if key != "tau_test":
            _require(not set(image_ids).intersection(row["image_id"] for _, row in rows),
                     "tau_test_training_role_overlap_forbidden")

    neural = _read_json(PROV / "regular_neural_structural_report.json", check)
    masks = _read_json(PROV / "independent_visual_masks/mask_report.json", check)
    _require(neural.get("publication") == "complete" and neural.get("contains_neural_values") is False
             and neural.get("OOD_arrays_formed_retained_summarized") is False,
             "completed_regular_neural_export_required")
    subjects = [value for value in neural["subjects"] if value["participant"] == SOURCE]
    axes = [value for value in masks["subjects"] if value["participant"] == SOURCE]
    _require(len(subjects) == len(axes) == 1, "one_fixed_source_native_metadata_binding_required")
    subject, mask = subjects[0], axes[0]
    support = subject["roles"]["tau_test"]
    _require(type(support["rows"]) is int and support["rows"] >= TEST_IMAGES
             and support["voxels"] == NATIVE_VOXELS == mask["union_voxel_count"]
             and support["dtype"] == "float32" and support["repetitions_preserved"] is True
             and support["finite_mask_bitorder"] == "little" and support["nonfinite_elements"] == 0
             and subject["column_order"] == "ascending_full_grid_C_order"
             and subject["mask_index_sha256"] == mask["voxel_indices_sha256"]
             and subject["native_grid"] == mask["grid"], "complete_fixed_sub03_tau_test_native_support_required")
    root = PROV / "regular_neural_sealed" / SOURCE
    groups = _trial_groups(root, support["rows"], image_ids, check)
    source = SourceMetadata(root, support["rows"], NATIVE_VOXELS,
        subject["mask_index_sha256"], tuple(subject["native_grid"]["shape"]), groups, image_ids)

    features = _read_json(PROV / "regular_vgg19_structural_report.json", check)
    recipe = lock["feature_recipe"]
    _require(features.get("publication") == "complete" and features.get("OOD_access") is False
             and features.get("feature_recipe") == recipe
             and features.get("feature_dtype") == "float32"
             and features.get("finite_mask_bitorder") == "little"
             and features.get("feature_axes") == "NCHW_or_Nvector"
             and features.get("HDF5_row_rule") == "source_index_CSV_ordinal"
             and features.get("layer_shapes") == dict(zip(LAYERS, map(list, SHAPES)))
             and recipe["layers"] == list(LAYERS) and recipe["shapes"] == list(map(list, SHAPES))
             and recipe["dtype"] == "float32" and recipe["finite_mask_bitorder"] == "little",
             "unchanged_completed_tau_test_feature_recipe_required")
    partitions = features["partitions"]
    _require([row["partition"] for row in partitions] == list(roles)
             and all(row["images"] == len(roles[row["partition"]]) and row["nonfinite_elements"] == 0
                     for row in partitions), "complete_original_feature_partition_binding_required")
    truth_root = PROV / "regular_vgg19_sealed/tau_test"
    _feature_index(truth_root, selected, check)
    identity = MappingProxyType({"role": "tau_test", "dtype": "float32", "extraction_status": "complete",
        "checkpoint": json.dumps(recipe["checkpoint_members"], sort_keys=True, separators=(",", ":")),
        "preprocessing": recipe["input_rule"], "feature_axes": "NCHW_or_Nvector",
        "finite_mask_bitorder": "little", "image_order": "source_index_CSV_ordinal"})
    check()
    return source, TruthMetadata(truth_root, image_ids, identity)


def _read_source(meta, check):
    """Read exactly the selected role; caller already admitted this metadata."""
    _require(isinstance(meta, SourceMetadata) and len(meta.image_ids) == TEST_IMAGES
             and len(set(meta.image_ids)) == TEST_IMAGES and len(meta.groups) == TEST_IMAGES
             and all(meta.groups), "complete_tau_test_metadata_required")
    positions = [position for group in meta.groups for position in group]
    _require(len(positions) == meta.rows and sorted(positions) == list(range(meta.rows)),
             "every_tau_test_repetition_exactly_once_required")
    check()
    with _path(meta.root / "native_voxel_indices.npy").open("rb") as stream:
        raw = stream.read(meta.voxels * 8 + 4097)
    _require(len(raw) <= meta.voxels * 8 + 4096
             and hashlib.sha256(raw).hexdigest() == meta.mask_index_sha256,
             "existing_native_axis_identity_required")
    check()
    native = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
    _require(native.shape == (meta.voxels,) and native.dtype.kind in "iu"
             and bool((native >= 0).all()) and bool((native[1:] > native[:-1]).all())
             and int(native[-1]) < math.prod(meta.grid_shape), "complete_source_native_axis_required")
    values = flags = None
    try:
        check()
        values = np.load(_path(meta.root / "tau_test_responses.npy"), mmap_mode="r", allow_pickle=False,
                         max_header_size=4096)
        check()
        flags = np.load(_path(meta.root / "tau_test_finite_packed.npy"), mmap_mode="r", allow_pickle=False,
                        max_header_size=4096)
        _require(values.shape == (meta.rows, meta.voxels) and values.dtype.str == "<f4"
                 and flags.shape == (meta.rows, (meta.voxels + 7) // 8) and flags.dtype.str == "|u1"
                 and values.flags.c_contiguous and flags.flags.c_contiguous,
                 "exact_tau_test_native_response_and_flag_headers_required")
        averaged = np.empty((TEST_IMAGES, meta.voxels), dtype=np.float32)
        for image, group in enumerate(meta.groups):
            check()
            trials = values[list(group), :]
            check()
            finite = np.unpackbits(flags[list(group), :], axis=1, bitorder="little")[:, :meta.voxels]
            _require(bool(finite.all()) and bool(np.isfinite(trials).all()),
                     "nonfinite_required_tau_test_repetition")
            with np.errstate(over="ignore", invalid="ignore"):
                averaged[image] = np.mean(trials, axis=0)
            _require(bool(np.isfinite(averaged[image]).all()), "nonfinite_required_tau_test_image_mean")
        check()
        return prediction.TauTestResponses(SOURCE, meta.image_ids,
            tuple(int(value) for value in native), _readonly(averaged))
    finally:
        for array in (values, flags):
            if array is not None and getattr(array, "_mmap", None) is not None:
                array._mmap.close()


class TruthReader:
    """Caller-admitted bounded common-feature reader; close after every failure.

    The supplied check must resolve the fixed read operation in its original
    allocation. Only bind_finding_a_inputs wires it to the external resolver;
    constructing this helper does not independently establish permission.
    """
    def __init__(self, root, *, image_ids, feature_identity, check):
        _require(callable(check) and isinstance(image_ids, tuple) and len(image_ids) == TEST_IMAGES
                 and len(set(image_ids)) == TEST_IMAGES, "fixed_truth_images_and_admission_check_required")
        self.root, self.image_ids = Path(root), image_ids
        self.feature_identity = MappingProxyType(dict(feature_identity))
        collection._identity(self.feature_identity)
        self._identity = collection._identity(self.feature_identity)
        self._check = check
        self._features = self._flags = None
        self._bank = self._packed = self._key = None
        self._first_byte = 0
        self._closed = False

    def _open(self):
        if self._features is not None:
            return
        self._check()
        self._features = h5py.File(_path(self.root / "features.h5"), "r")
        self._check()
        self._flags = h5py.File(_path(self.root / "finite_packed.h5"), "r")
        _require(set(self._features) == set(self._flags) == set(LAYERS), "exact_truth_layer_vocabulary_required")
        for layer, shape in zip(LAYERS, SHAPES):
            self._check()
            _require(isinstance(self._features.get(layer, getlink=True), h5py.HardLink)
                     and isinstance(self._flags.get(layer, getlink=True), h5py.HardLink),
                     "local_truth_dataset_links_required")
            data, flags = self._features[layer], self._flags[layer]
            _require(isinstance(data, h5py.Dataset) and isinstance(flags, h5py.Dataset)
                     and data.shape == (TEST_IMAGES,) + tuple(shape) and data.dtype.str == "<f4"
                     and flags.shape == (TEST_IMAGES, (math.prod(shape) + 7) // 8) and flags.dtype.str == "|u1"
                     and all(not value.is_virtual and value.external is None for value in (data, flags)),
                     "exact_truth_feature_and_finite_headers_required")

    def _load_bank(self, layer, shape, channel):
        key = layer, channel
        if key == self._key:
            self._check()
            return
        self._key = self._bank = self._packed = None
        spatial = math.prod(shape[1:]) if len(shape) == 3 else shape[0]
        first = channel * spatial if len(shape) == 3 else 0
        first_byte, stop_byte = first // 8, (first + spatial + 7) // 8
        size = TEST_IMAGES * (spatial * 4 + stop_byte - first_byte)
        _require(size <= MAX_TRUTH_BANK_BYTES, "truth_channel_bank_exceeds_fixed_byte_cap")
        self._check()
        data = self._features[layer]
        bank = (data[:, channel, :, :].reshape(TEST_IMAGES, spatial)
                if len(shape) == 3 else data[:, :])
        self._check()
        packed = self._flags[layer][:, first_byte:stop_byte]
        _require(bank.dtype == np.dtype("float32") and bank.shape == (TEST_IMAGES, spatial)
                 and packed.dtype == np.dtype("uint8")
                 and bank.nbytes + packed.nbytes == size, "exact_bounded_truth_bank_required")
        self._check()
        self._bank, self._packed = _readonly(bank), _readonly(packed)
        self._first_byte, self._key = first_byte, key

    def __call__(self, *, layer, image_ids, full_layer_C_index, feature_identity):
        try:
            _require(not self._closed, "truth_reader_closed_or_failed")
            self._check()
            _require(image_ids == self.image_ids and collection._identity(feature_identity) == self._identity,
                     "exact_truth_image_and_extraction_binding_required")
            _require(layer in LAYERS, "fixed_truth_layer_required")
            shape = SHAPES[LAYERS.index(layer)]
            supplied = full_layer_C_index
            _require(isinstance(supplied, np.ndarray) and supplied.ndim == 1 and supplied.dtype.kind in "iu"
                     and 1 <= len(supplied) <= MAX_BLOCK_UNITS
                     and bool((supplied >= 0).all()) and bool((supplied < math.prod(shape)).all())
                     and len(np.unique(supplied)) == len(supplied), "bounded_unique_truth_C_units_required")
            ids = np.array(supplied, dtype=np.int64, copy=True)
            self._open()
            output = np.empty((TEST_IMAGES, len(ids)), dtype=np.float32)
            valid = np.empty(output.shape, dtype=bool)
            spatial = math.prod(shape[1:]) if len(shape) == 3 else shape[0]
            channels = ids // spatial if len(shape) == 3 else np.zeros(len(ids), dtype=np.int64)
            for channel in np.unique(channels):
                self._load_bank(layer, shape, int(channel))
                positions = np.flatnonzero(channels == channel)
                selected = ids[positions]
                output[:, positions] = self._bank[:, selected % spatial]
                byte_ids = selected // 8 - self._first_byte
                valid[:, positions] = ((self._packed[:, byte_ids] >> (selected % 8)[None, :]) & 1).astype(bool)
                self._check()
            _require(np.array_equal(valid, np.isfinite(output)), "truth_finite_flags_disagree_with_values")
            self._check()
            return collection.TruthBlock(self.image_ids, layer, _readonly(ids), _readonly(output),
                                         _readonly(valid), self.feature_identity)
        except BaseException:
            self.close()
            raise

    def close(self):
        self._closed = True
        self._key = self._bank = self._packed = None
        try:
            if self._flags is not None:
                self._flags.close()
        finally:
            self._flags = None
            if self._features is not None:
                self._features.close()
            self._features = None

    def __enter__(self):
        _require(not self._closed, "truth_reader_closed_or_failed")
        self._check()
        return self

    def __exit__(self, *unused):
        self.close()


def bind_finding_a_inputs(execution_context, *, require_admitted_context, budget):
    """Admit the fixed first readout, bind metadata, then read source tau-test."""
    initial = _Admission(execution_context, require_admitted_context, budget)
    identity = initial.check()
    admitted = _Admission(execution_context, require_admitted_context, budget, identity)
    source, truth = _metadata(admitted.check)
    _require(source.image_ids == truth.image_ids and source.voxels == NATIVE_VOXELS
             and source.root == PROV / "regular_neural_sealed" / SOURCE
             and truth.root == PROV / "regular_vgg19_sealed/tau_test",
             "fixed_first_readout_payload_roots_and_axis_required")
    reader = TruthReader(truth.root, image_ids=truth.image_ids, feature_identity=truth.feature_identity,
                         check=lambda: admitted.check(OPERATIONS[2]))
    try:
        responses = _read_source(source, lambda: admitted.check(OPERATIONS[1]))
        admitted.check()
        return BoundFindingAInputs(execution_context, require_admitted_context, budget, identity,
            source.image_ids, responses, tuple(map(len, source.groups)), truth.feature_identity, reader, admitted)
    except BaseException:
        reader.close()
        raise
