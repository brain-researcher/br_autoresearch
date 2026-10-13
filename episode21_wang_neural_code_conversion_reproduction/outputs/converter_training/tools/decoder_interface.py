"""UNACTIVATED content-converter data interface; no CLI or execution grant.

The future caller must first admit its own converter-stage access, corrected
loss evidence and resources. A complete decoder manifest and a live compute
budget are required here, but neither grants that later-stage permission.
The existing empirical execution adapter remains closed. No work is launched.

All 4,712 source images remain present in every feature block. Blocking limits
I/O/working memory only; it does not introduce minibatches or optimizer steps.
Converted target responses are already standardized: weights/bias must be
applied directly, without another target-neural centering/scaling operation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "decoder_training/tools"))
import decoder_assembly as assembly

require = assembly.require
IMAGES = 4712
FEATURE_STRIPE_BYTES = 67_108_864


@dataclass(frozen=True)
class NativeAxis:
    participant: str
    full_native_ids: object
    eligible_native_ids: object
    eligible_full_columns: object
    mean: object
    std: object


@dataclass(frozen=True)
class SourceTraining:
    axis: NativeAxis
    image_ids: tuple
    normalized_X: object
    feature_root: Path


@dataclass(frozen=True)
class ContentPair:
    manifest: assembly.CompleteDecoderManifest
    source: SourceTraining
    target_state: assembly.StateRef
    target_normalization: assembly.Normalization
    target_axis: NativeAxis


@dataclass(frozen=True)
class DecoderBlock:
    participant: str
    layer: str
    channel: int | None
    unit_start: int
    full_layer_Fortran_index: object
    full_layer_C_index: object
    selected_native_voxel_ids: object
    eligible_columns: object
    weights: object
    bias: object
    target_mean: object
    target_std: object


@dataclass(frozen=True)
class ContentBlock:
    decoder: DecoderBlock
    standardized_features: object


def _readonly(array):
    array.setflags(write=False)
    return array


def native_axis(normalization):
    """Preserve the already selected native IDs; never select voxels again."""
    import numpy as np
    ids, eligible, mask = (normalization.native_ids, normalization.eligible_ids,
                           normalization.eligible_mask)
    require(np.array_equal(ids[mask], eligible), "frozen eligible native axis required")
    columns = np.flatnonzero(mask)
    return NativeAxis(normalization.participant, _readonly(ids), _readonly(eligible),
        _readonly(columns), _readonly(normalization.mean[columns]),
        _readonly(normalization.std[columns]))


def require_image_order(actual, expected):
    actual, expected = tuple(actual), tuple(expected)
    require(len(actual) == IMAGES and len(set(actual)) == IMAGES
            and actual == expected, "exact 4712 source image identities and order required")
    return actual


def _state_context(manifest, participant, budget):
    assembly._require_reader_context(budget)
    require(isinstance(manifest, assembly.CompleteDecoderManifest),
            "validated complete decoder manifest required")
    assembly.verify_catalog(manifest.fragments, manifest.states)
    require(participant in assembly.coverage.PARTICIPANTS, "fixed participant required")
    state = next(value for value in manifest.states if value.participant == participant)
    return state, assembly._admit_normalization(state, budget.check)


def _read_normalized_X(path, eligible_count, check):
    """Read exactly one bounded NPZ member; never request penalized_Gram."""
    import numpy as np
    expected = (IMAGES, eligible_count)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("normalized_X.npy")
        require(info.file_size <= IMAGES * eligible_count * 4 + 4096,
                "normalized source member exceeds expected byte bound")
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "bounded NPY header version required")
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, _, dtype = reader(stream)
            require(shape == expected and dtype.str == "<f4" and not dtype.hasobject,
                    "full-source eligible-axis float32 shape required")
        check()
        with archive.open(info) as stream:
            values = np.lib.format.read_array(stream, allow_pickle=False)
    for first in range(0, IMAGES, 64):
        check()
        require(bool(np.isfinite(values[first:first + 64]).all()), "nonfinite normalized source state")
    return _readonly(values)


def load_source_training(manifest, *, participant, expected_image_ids, budget):
    """Library read inside an already authorized converter stage; no Gram load."""
    state, normalization = _state_context(manifest, participant, budget)
    # Existing admission authenticates complete feature export, subject_unique
    # feature-index order, role-disjoint IDs and preserved trial-row bindings.
    import participant_training_state as states
    metadata = states.admit_participant(participant)
    ids = require_image_order(normalization.training_image_ids.tolist(), metadata["ids"])
    require_image_order(ids, expected_image_ids)
    axis = native_axis(normalization)
    values = _read_normalized_X(assembly.physical_path(assembly.BASE, state.path),
                                len(axis.eligible_native_ids), budget.check)
    return SourceTraining(axis, ids, values, metadata["feature_root"])


def bind_content_pair(manifest, source, *, target, budget):
    """Bind a fixed eligible target axis; no target response matrix is read."""
    require(isinstance(source, SourceTraining) and source.axis.participant != target,
            "admitted nonself source-target pair required")
    state, normalization = _state_context(manifest, target, budget)
    require_image_order(source.image_ids, source.image_ids)
    require(source.normalized_X.shape == (IMAGES, len(source.axis.eligible_native_ids)),
            "complete eligible source matrix required")
    require(not set(source.image_ids).intersection(normalization.training_image_ids.tolist()),
            "source converter and target decoder training IDs must be disjoint")
    return ContentPair(manifest, source, state, normalization, native_axis(normalization))


def active_shape(layer, channel):
    require(layer in assembly.coverage.prefix.LAYERS, "fixed VGG19 layer required")
    shape = assembly.coverage.prefix.SHAPES[assembly.coverage.prefix.LAYERS.index(layer)]
    if len(shape) == 3:
        require(type(channel) is int and 0 <= channel < shape[0], "exact convolutional channel required")
        return shape, math.prod(shape[1:])
    require(channel is None, "FC layers use all units with channel None")
    return shape, shape[0]


def active_layer_units(layer, channel_or_None):
    """Public metadata-only active width, with the same exact channel checks."""
    return active_shape(layer, channel_or_None)[1]


def _channel_refs(pair, layer, channel):
    _, units = active_shape(layer, channel)
    storage_channel = 0 if channel is None else channel
    refs = sorted((ref for ref in pair.manifest.fragments
                   if ref.span.participant == pair.target_axis.participant
                   and ref.span.layer == layer and ref.span.channel == storage_channel),
                  key=lambda ref: ref.span.start)
    end = 0
    for ref in refs:
        require(ref.span.start == end, "complete nonoverlapping active-channel coverage required")
        end = ref.span.stop
    require(end == units, "complete active-channel tail required")
    return refs


def map_decoder_block(raw, axis, *, channel):
    """Convert full-native positions to fixed eligible positions, retaining W/b."""
    import numpy as np
    shape, units = active_shape(raw.layer, channel)
    require(raw.participant == axis.participant, "target participant binding required")
    ids = np.asarray(raw.selected_native_voxel_ids)
    positions = np.searchsorted(axis.eligible_native_ids, ids)
    require(bool((positions < len(axis.eligible_native_ids)).all()), "selected target ID absent from eligible axis")
    require(np.array_equal(axis.eligible_native_ids[positions], ids), "selected target ID absent from eligible axis")
    require(np.array_equal(axis.full_native_ids[raw.native_columns], ids)
            and np.array_equal(axis.eligible_full_columns[positions], raw.native_columns),
            "full-native and eligible target coordinates disagree")
    full_f = np.asarray(raw.full_layer_Fortran_index)
    spatial = full_f if len(shape) == 1 else full_f // shape[0]
    require(len(spatial) > 0 and bool((spatial >= 0).all()) and bool((spatial < units).all())
            and np.array_equal(spatial, np.arange(spatial[0], spatial[0] + len(spatial)))
            and (len(shape) == 1 or bool((full_f % shape[0] == channel).all())),
            "contiguous selected-channel Fortran coordinates required")
    full_c = (spatial if len(shape) == 1 else channel * units
              + (spatial % shape[1]) * shape[2] + spatial // shape[1])
    require(np.array_equal(raw.full_layer_C_index, full_c), "C/Fortran feature coordinates disagree")
    return DecoderBlock(raw.participant, raw.layer, channel, int(spatial[0]),
        _readonly(full_f), _readonly(np.asarray(raw.full_layer_C_index)), _readonly(ids),
        _readonly(positions), _readonly(raw.weights), _readonly(raw.bias),
        _readonly(raw.target_mean), _readonly(raw.target_std))


def _admit_decoder_collection(pair, budget):
    """Fresh complete-catalog and recovery-evidence admission, without payloads."""
    assembly._require_reader_context(budget)
    require(isinstance(pair, ContentPair), "bound content pair required")
    manifest = pair.manifest
    require(isinstance(manifest, assembly.CompleteDecoderManifest),
            "validated complete manifest required")
    # Match iter_sparse_blocks' complete-catalog/recovery admission before
    # reading any coefficient. A recovered prefix keeps its exact original
    # review; a missing, changed or no-longer-valid review cannot be bypassed.
    assembly.verify_catalog(manifest.fragments, manifest.states)
    assembly._validate_recovery_catalog(manifest.fragments, manifest.recovery_review)
    if manifest.recovery_review is not None:
        assembly._revalidate_recovery_review(manifest.recovery_review, budget.check)


def iter_decoder_blocks(pair, *, layer, channel, budget, block_units=64):
    """Standalone reads freshly admit the whole collection on every call."""
    require(type(block_units) is int and 1 <= block_units <= assembly.MAX_BLOCK_UNITS,
            "bounded decoder block required")
    _admit_decoder_collection(pair, budget)
    yield from _iter_decoder_blocks(pair, layer=layer, channel=channel,
                                    check=budget.check, block_units=block_units)


def _iter_decoder_blocks(pair, *, layer, channel, check, block_units):
    """Ordered physical scans after this caller's collection admission."""
    manifest = pair.manifest
    # The validated complete manifest is immutable. Canonical span order can
    # differ from file order (e.g. retained prefix); never buffer an entire
    # channel just to sort it. The public iterator groups by file path, so
    # retain the existing physical-row reader per spatially ordered ref here.
    next_unit = 0
    for ref in _channel_refs(pair, layer, channel):
        scan = (assembly._scan_fragments(assembly.BASE, [ref], pair.target_state,
                    pair.target_normalization, check, block_units)
                if manifest.recovery_review is None else
                assembly._scan_fragments(assembly.BASE, [ref], pair.target_state,
                    pair.target_normalization, check, block_units,
                    recovery_review=manifest.recovery_review))
        for raw in scan:
            block = map_decoder_block(raw, pair.target_axis, channel=channel)
            require(block.unit_start == next_unit, "ordered complete decoder block coverage required")
            next_unit += len(block.bias)
            yield block
    require(next_unit == active_shape(layer, channel)[1], "complete active-channel decoder rows required")


def _read_feature_units(data, finite, block, check):
    """Read requested units only; finite-bit addressing is C, output columns F."""
    import numpy as np
    shape, _ = active_shape(block.layer, block.channel)
    first, stop = block.unit_start, block.unit_start + len(block.bias)
    values = np.empty((data.shape[0], stop - first), dtype=np.float32)
    byte_ids, inverse = np.unique(block.full_layer_C_index // 8, return_inverse=True)
    check()
    packed = finite[:, byte_ids]
    valid = (packed[:, inverse] >> (block.full_layer_C_index % 8)) & 1
    require(bool(valid.all()), "nonfinite required source-feature flag")
    if len(shape) == 1:
        check()
        values[:] = data[:, first:stop]
    else:
        unit = first
        while unit < stop:
            check()
            row, column = unit % shape[1], unit // shape[1]
            count = min(stop - unit, shape[1] - row)
            values[:, unit - first:unit - first + count] = data[:, block.channel, row:row + count, column]
            unit += count
    require(bool(np.isfinite(values).all()), "nonfinite required source feature")
    require(bool(np.isfinite(block.target_mean).all()) and bool(np.isfinite(block.target_std).all())
            and bool((block.target_std > 0).all()), "fixed positive target-decoder feature scales required")
    values -= block.target_mean
    values /= block.target_std
    require(bool(np.isfinite(values).all()), "nonfinite target-standardized source feature")
    return _readonly(values)


class _FeatureStripeReader:
    """One pass's active-channel stripe; the thin reader remains the oracle.

    Only storage reads are grouped. Returned blocks retain every image, the
    original Fortran unit order, and the original float32 normalization. The
    packed buffer covers one channel's C-order bits (including boundary bytes);
    only requested bits/values are checked, even when a stripe reads ahead.
    Neither buffer survives the owning pass. Fully connected layers continue
    to use the unchanged thin reader.
    """

    def __init__(self, data, finite, *, layer, channel, check):
        self.data, self.finite = data, finite
        self.layer, self.channel, self.check = layer, channel, check
        self.shape, _ = active_shape(layer, channel)
        self._stripe = self._packed = None
        self._stripe_start = self._stripe_stop = 0
        self._closed = False
        if len(self.shape) == 3:
            _, height, width = self.shape
            require(type(FEATURE_STRIPE_BYTES) is int and FEATURE_STRIPE_BYTES > 0,
                    "positive feature stripe byte bound required")
            self._columns = min(width, FEATURE_STRIPE_BYTES // (data.shape[0] * height * 4))
            require(self._columns >= 1, "feature stripe bound must hold one complete column")
            self._first_byte = channel * height * width // 8
            self._stop_byte = ((channel + 1) * height * width + 7) // 8

    def close(self):
        self._stripe = self._packed = None
        self.data = self.finite = None
        self._closed = True

    def read(self, block):
        import numpy as np
        require(not self._closed, "feature stripe reader is closed")
        require((block.layer, block.channel) == (self.layer, self.channel),
                "feature block must match the active layer and channel")
        if len(self.shape) == 1:
            return _read_feature_units(self.data, self.finite, block, self.check)

        _, height, width = self.shape
        first, stop = block.unit_start, block.unit_start + len(block.bias)
        values = np.empty((self.data.shape[0], stop - first), dtype=np.float32)
        byte_ids, inverse = np.unique(block.full_layer_C_index // 8, return_inverse=True)
        self.check()
        if self._packed is None:
            self._packed = self.finite[:, self._first_byte:self._stop_byte]
            self.check()
        packed = self._packed[:, byte_ids - self._first_byte]
        valid = (packed[:, inverse] >> (block.full_layer_C_index % 8)) & 1
        require(bool(valid.all()), "nonfinite required source-feature flag")

        unit = first
        while unit < stop:
            self.check()
            row, column = unit % height, unit // height
            if self._stripe is None or not self._stripe_start <= column < self._stripe_stop:
                # Drop the previous allocation BEFORE requesting the next one.
                self._stripe = None
                self._stripe_start = column // self._columns * self._columns
                self._stripe_stop = min(width, self._stripe_start + self._columns)
                self.check()
                self._stripe = self.data[:, self.channel, :, self._stripe_start:self._stripe_stop]
                self.check()
            count = min(stop - unit, height - row)
            values[:, unit - first:unit - first + count] = self._stripe[
                :, row:row + count, column - self._stripe_start]
            unit += count
        require(bool(np.isfinite(values).all()), "nonfinite required source feature")
        require(bool(np.isfinite(block.target_mean).all()) and bool(np.isfinite(block.target_std).all())
                and bool((block.target_std > 0).all()), "fixed positive target-decoder feature scales required")
        values -= block.target_mean
        values /= block.target_std
        require(bool(np.isfinite(values).all()), "nonfinite target-standardized source feature")
        return _readonly(values)


def iter_content_blocks(pair, *, layer, channel, budget, block_units=64):
    """Source images/rows + target decoder moments; never target training images.

    Callers may iterate again for complete-block norms then loss/adjoint.
    This function supplies no optimization, prediction or resource authority.
    """
    assembly._require_reader_context(budget)
    require(isinstance(pair, ContentPair), "bound content pair required")
    blocks = iter_decoder_blocks(pair, layer=layer, channel=channel,
                                  budget=budget, block_units=block_units)
    yield from _iter_content_blocks(pair, layer=layer, channel=channel,
                                    decoder_blocks=blocks, check=budget.check)


def _iter_content_blocks(pair, *, layer, channel, decoder_blocks, check, reader_factory=None):
    """Reopen admitted headers, then construct the selected bounded reader."""
    reader = None
    try:
        check()
        require(reader_factory is None or callable(reader_factory), "feature_reader_factory_must_be_callable")
        import h5py
        # Reuse the source-only structural admission and exact row binding done by
        # load_source_training; do not reread all regular metadata every iteration.
        root = assembly.BASE / "outputs/steward_provisioning/regular_vgg19_sealed" / pair.source.axis.participant
        require(root == pair.source.feature_root,
                "fixed subject-unique source feature partition required")
        shape, _ = active_shape(layer, channel)
        with h5py.File(root / "features.h5", "r") as features, \
                h5py.File(root / "finite_packed.h5", "r") as flags:
            require(isinstance(features.get(layer, getlink=True), h5py.HardLink)
                    and isinstance(flags.get(layer, getlink=True), h5py.HardLink),
                    "ordinary in-file source feature datasets required")
            data, finite = features[layer], flags[layer]
            require(isinstance(data, h5py.Dataset) and isinstance(finite, h5py.Dataset)
                    and data.shape == (IMAGES,) + shape and data.dtype.str == "<f4"
                    and data.chunks is None and data.compression is None
                    and finite.shape == (IMAGES, (math.prod(shape) + 7) // 8)
                    and finite.dtype.str == "|u1"
                    and all(not value.is_virtual and value.external is None for value in (data, finite)),
                    "bounded source-unique feature and finite-mask layout required")
            factory = _FeatureStripeReader if reader_factory is None else reader_factory
            reader = factory(data, finite, layer=layer, channel=channel, check=check)
            for block in decoder_blocks:
                yield ContentBlock(block, reader.read(block))
    finally:
        if reader is not None:
            reader.close()
        close = getattr(decoder_blocks, "close", None)
        if callable(close):
            close()


def _owned_recovery_review(review):
    """Bound one private copy using the recovery module's existing metadata cap."""
    if review is None:
        return None
    import reviewed_io_recovery as recovery
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False).iterencode(review):
        size += len(chunk.encode("utf-8"))
        require(size <= recovery.METADATA_BYTES, "recovery review exceeds existing metadata bound")
    return deepcopy(review)


def content_block_provider(pair, *, budget, block_units=64, reader_factory=None):
    """Once-bound ActiveLayer -> LossBlock reader for one immutable collection.

    Construction revalidates the complete catalog/recovery evidence ONCE under
    the caller's actual allocation. It reads operational metadata, never neural,
    feature or coefficient values, and grants no converter-stage permission.

    Each pass checks the caller's review against a privately owned admitted copy
    at entry and exhausted-pass exit, after budget callbacks. Between-pass
    mutation refuses before payloads. Mid-pass mutation cannot change the private
    authority used for scans and refuses before the pass completes (therefore
    before the existing loss engine can advance its optimizer). The full review
    is not walked at every small block. Every ordinary block budget, allocation,
    terminal-record, HDF5 identity, coefficient and feature check remains live.
    A caller must exhaust each pass, as the full-batch engine already does.
    The optional reader factory runs only after the existing HDF header checks;
    the default stripe path is unchanged. The returned callable's check_pass is
    this exact private check_binding closure, for explicitly selected two-pass
    retention without another admission or a copied guard implementation.
    This provider itself adds no coefficient cache, durable ticket or retry.
    """
    assembly._require_reader_context(budget)
    require(isinstance(pair, ContentPair), "bound content pair required")
    require(type(block_units) is int and 1 <= block_units <= assembly.MAX_BLOCK_UNITS,
            "bounded content provider block required")
    require(reader_factory is None or callable(reader_factory), "feature_reader_factory_must_be_callable")
    require(isinstance(pair.manifest, assembly.CompleteDecoderManifest),
            "validated complete manifest required")
    import reviewed_io_recovery as recovery
    identity = (budget.allocation["job_id"], budget.allocation["hostname"])
    snapshot = _owned_recovery_review(pair.manifest.recovery_review)
    bound_pair = replace(pair, manifest=replace(pair.manifest, recovery_review=snapshot))

    def check_allocation():
        budget.check()
        require((budget.allocation.get("job_id"), budget.allocation.get("hostname")) == identity,
                "content provider cannot move to another allocation")

    def check_binding():
        assembly._require_reader_context(budget)
        check_allocation()
        require(recovery._same(pair.manifest.recovery_review, snapshot),
                "caller recovery review changed after provider binding")

    # Snapshot first: a callback that mutates the caller-owned dict during
    # admission cannot silently replace the evidence that was authenticated.
    _admit_decoder_collection(bound_pair, budget)
    check_binding()

    def provider(active):
        from content_loss import LossBlock
        check_binding()
        require(active.units == active_layer_units(active.layer, active.channel),
                "complete active layer size required")
        blocks = _iter_decoder_blocks(bound_pair, layer=active.layer, channel=active.channel,
                                       check=check_allocation, block_units=block_units)
        reader_options = {} if reader_factory is None else {"reader_factory": reader_factory}
        content = _iter_content_blocks(bound_pair, layer=active.layer, channel=active.channel,
                                        decoder_blocks=blocks, check=check_allocation, **reader_options)
        try:
            for block in content:
                decoder = block.decoder
                yield LossBlock(decoder.unit_start, decoder.eligible_columns, decoder.weights,
                                decoder.bias, block.standardized_features)
            check_binding()
        finally:
            close = getattr(content, "close", None)
            if callable(close):
                close()

    provider.check_pass = check_binding
    return provider
