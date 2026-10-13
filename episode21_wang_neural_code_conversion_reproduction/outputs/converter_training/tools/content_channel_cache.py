"""Explicitly selected exact-read cache; no empirical admission or launcher.

make_cached_provider reuses decoder_interface's original complete admission,
reader factory after HDF header checks, and exact check_binding closure exposed
as check_pass. A separately admitted caller selects it through the content-pair
driver's optional provider factory; all ordinary defaults remain unchanged.
This module does not duplicate guards, monkeypatch existing code, generate
schedules, read target responses or expand decoder matrices.

The reader retains one selected channel and its packed flags below 1 GiB. The
cache retains only exact LossBlock arrays, below 4 GiB, for the existing engine's
19 norm passes followed by the matching 19 gradient passes. Raw channel/flag
buffers, incoming blocks and one newly copied block are ADDITIONAL transient
RSS, as are existing models, optimizer and loss-engine tensors. These storage
bounds are not a claim that a full empirical process meets an RSS/time budget.
"""
from __future__ import annotations

import numpy as np

import decoder_interface as interface
from content_loss import ActiveLayer, LossBlock
from schedule import LAYER_ORDER


MAX_CHANNEL_BYTES = 1024**3
MAX_RETAINED_BYTES = 4 * 1024**3
require = interface.require


class ContiguousChannelReader:
    """Selected-channel hyperslab reader, after the EXISTING header admission.

    Construction inspects shapes/dtypes only. The first read loads exactly the
    selected convolutional channel, or all units of the selected FC layer, and
    only the packed bytes covering it. Output blocks retain every source row,
    Fortran column order and the original float32 target normalization order.
    Boundary bits outside a non-byte-aligned channel are never required.
    """

    def __init__(self, data, finite, *, layer, channel, check):
        require(callable(check), "explicit_channel_budget_check_required")
        check()
        self.shape, self.units = interface.active_shape(layer, channel)
        require(data.shape == (interface.IMAGES,) + self.shape
                and data.dtype == np.dtype("float32")
                and finite.shape == (interface.IMAGES, (int(np.prod(self.shape)) + 7) // 8)
                and finite.dtype == np.dtype("uint8"),
                "admitted_full_source_feature_and_flag_shapes_required")
        self.data, self.finite = data, finite
        self.layer, self.channel, self.check = layer, channel, check
        self._first_c = 0 if channel is None else channel * self.units
        self._first_byte = self._first_c // 8
        self._stop_byte = (self._first_c + self.units + 7) // 8
        self._bank_bytes = interface.IMAGES * self.units * 4
        self._flag_bytes = interface.IMAGES * (self._stop_byte - self._first_byte)
        require(self._bank_bytes + self._flag_bytes <= MAX_CHANNEL_BYTES,
                "selected_channel_and_flags_exceed_1GiB_before_read")
        self._bank = self._packed = None
        self._closed = False
        self.retained_bytes = 0

    def close(self):
        self._bank = self._packed = None
        self.data = self.finite = None
        self.retained_bytes = 0
        self._closed = True

    def _load(self):
        if self._bank is not None:
            return
        self.check()
        bank = (self.data[:, :] if self.channel is None else
                self.data[:, self.channel, :, :])
        expected = ((interface.IMAGES, self.units) if self.channel is None else
                    (interface.IMAGES,) + self.shape[1:])
        require(isinstance(bank, np.ndarray) and bank.shape == expected
                and bank.dtype == np.dtype("float32") and bank.flags.owndata
                and bank.flags.c_contiguous and bank.nbytes == self._bank_bytes,
                "owned_bounded_contiguous_selected_channel_required")
        self._bank = bank
        self.retained_bytes = bank.nbytes
        self.check()
        packed = self.finite[:, self._first_byte:self._stop_byte]
        require(isinstance(packed, np.ndarray)
                and packed.shape == (interface.IMAGES, self._stop_byte - self._first_byte)
                and packed.dtype == np.dtype("uint8") and packed.flags.owndata
                and packed.flags.c_contiguous and packed.nbytes == self._flag_bytes,
                "owned_bounded_selected_channel_flag_bytes_required")
        self._packed = packed
        self.retained_bytes += packed.nbytes
        require(self.retained_bytes <= MAX_CHANNEL_BYTES,
                "selected_channel_and_flags_exceed_1GiB")
        self.check()

    def read(self, block):
        try:
            require(not self._closed, "contiguous_channel_reader_closed")
            require((block.layer, block.channel) == (self.layer, self.channel),
                    "feature_block_must_match_selected_channel")
            first, count = block.unit_start, len(block.bias)
            require(type(first) is int and 0 <= first and 0 < count <= interface.assembly.MAX_BLOCK_UNITS
                    and first + count <= self.units, "bounded_selected_channel_block_required")
            spatial = np.arange(first, first + count, dtype=np.int64)
            expected_c = (spatial if self.channel is None else self._first_c
                          + (spatial % self.shape[1]) * self.shape[2] + spatial // self.shape[1])
            expected_f = (spatial if self.channel is None else
                          self.channel + self.shape[0] * spatial)
            require(np.array_equal(block.full_layer_C_index, expected_c)
                    and np.array_equal(block.full_layer_Fortran_index, expected_f),
                    "exact_selected_channel_F_C_coordinates_required")
            self._load()
            self.check()
            # Address packed flags by ORIGINAL full-layer C bit position.
            byte_ids, inverse = np.unique(expected_c // 8, return_inverse=True)
            packed = self._packed[:, byte_ids - self._first_byte]
            valid = (packed[:, inverse] >> (expected_c % 8)) & 1
            require(bool(valid.all()), "nonfinite required source-feature flag")
            values = np.empty((interface.IMAGES, count), dtype=np.float32)
            if self.channel is None:
                values[:] = self._bank[:, first:first + count]
            else:
                unit, stop = first, first + count
                while unit < stop:
                    self.check()
                    row, column = unit % self.shape[1], unit // self.shape[1]
                    take = min(stop - unit, self.shape[1] - row)
                    values[:, unit - first:unit - first + take] = self._bank[
                        :, row:row + take, column]
                    unit += take
            require(bool(np.isfinite(values).all()), "nonfinite required source feature")
            require(bool(np.isfinite(block.target_mean).all())
                    and bool(np.isfinite(block.target_std).all())
                    and bool((block.target_std > 0).all()),
                    "fixed positive target-decoder feature scales required")
            values -= block.target_mean
            values /= block.target_std
            require(bool(np.isfinite(values).all()), "nonfinite target-standardized source feature")
            self.check()
            return interface._readonly(values)
        except BaseException:
            self.close()
            raise


class _TwoPassCache:
    """Array-only state machine; public subclass fixes the scientific roster."""

    def __init__(self, provider, *, check_pass, check, layer_order, expected_rows,
                 validate_active, maximum_bytes):
        require(callable(provider) and callable(check_pass) and callable(check)
                and callable(validate_active), "existing_admitted_provider_and_checks_required")
        require(type(maximum_bytes) is int and 0 < maximum_bytes <= MAX_RETAINED_BYTES,
                "positive_retained_cache_bound_at_most_4GiB_required")
        require(type(expected_rows) is int and expected_rows > 0,
                "positive_complete_row_count_required")
        self._order = tuple(layer_order)
        require(self._order and len(set(self._order)) == len(self._order), "unique_layer_roster_required")
        self._provider, self._check_pass, self._check = provider, check_pass, check
        self._rows, self._validate_active, self._maximum = expected_rows, validate_active, maximum_bytes
        self._active, self._blocks = [], []
        self._index = self._retained = self._peak = self.completed_iterations = 0
        self._phase = "collect"
        self._busy = self._failed = self._closed = False
        self._upstream = None

    @property
    def retained_bytes(self):
        return self._retained

    @property
    def peak_retained_bytes(self):
        return self._peak

    def _clear(self):
        self._active.clear()
        # Suspended iterators may still hold an individual saved-list reference.
        # Empty those lists as well; a currently yielded block is caller-held.
        for saved in self._blocks:
            saved.clear()
        self._blocks.clear()
        self._retained = 0

    def _fail(self):
        # Never recursively close the currently executing wrapper generator.
        self._failed = True
        self._clear()

    @staticmethod
    def _close_iterator(iterator, original_error=None):
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as error:
                if original_error is None:
                    raise
                original_error.add_note("cache upstream close failed: " + type(error).__name__)

    def close(self):
        self._closed = True
        self._clear()
        upstream, self._upstream = self._upstream, None
        self._close_iterator(upstream)

    def _copy(self, block, cursor, active):
        require(isinstance(block, LossBlock) and type(block.unit_start) is int
                and block.unit_start == cursor, "ordered_contiguous_LossBlock_required")
        arrays = (block.eligible_columns, block.weights, block.bias, block.standardized_features)
        require(all(isinstance(value, np.ndarray) for value in arrays),
                "admitted_NumPy_LossBlock_arrays_required")
        columns, weights, bias, targets = arrays
        require(columns.dtype == np.dtype("int64") and columns.ndim == 2
                and 0 < columns.shape[0] <= interface.assembly.MAX_BLOCK_UNITS
                and columns.shape[1] > 0 and weights.dtype == np.dtype("float32")
                and weights.shape == columns.shape and bias.dtype == np.dtype("float32")
                and bias.shape == (columns.shape[0],) and targets.dtype == np.dtype("float32")
                and targets.shape == (self._rows, columns.shape[0])
                and cursor + columns.shape[0] <= active.units,
                "complete_dtype_shape_and_active_support_required")
        size = sum(value.nbytes for value in arrays)
        require(self._retained + size <= self._maximum,
                "retained_cache_byte_cap_before_copy")
        self._check()
        # Owning copies avoid retaining a decoder/whole-channel base that is
        # larger than the counted four arrays. No dtype or arithmetic changes.
        owned = tuple(interface._readonly(value.copy(order="C")) for value in arrays)
        require(all(value.flags.owndata and value.flags.c_contiguous for value in owned)
                and sum(value.nbytes for value in owned) == size,
                "exact_owning_cache_copies_required")
        self._retained += size
        self._peak = max(self._peak, self._retained)
        self._check()
        return LossBlock(cursor, *owned)

    def __call__(self, active):
        # All state changes occur on actual iteration, never generator creation.
        upstream, owns_pass = None, False
        try:
            require(not self._closed and not self._failed, "cache_closed_or_failed")
            require(not self._busy, "one_exhausted_cache_pass_at_a_time_required")
            self._busy, owns_pass = True, True
            self._check_pass()
            self._check()
            require(isinstance(active, ActiveLayer), "explicit_ActiveLayer_required")
            self._validate_active(active)
            require(active.layer == self._order[self._index], "fixed_layer_pass_order_required")
            if self._phase == "collect":
                saved, cursor = [], 0
                self._blocks.append(saved)
                self._active.append(active)
                upstream = iter(self._provider(active))
                self._upstream = upstream
                for block in upstream:
                    self._check()
                    require(not self._closed and not self._failed, "cache_closed_or_failed")
                    copied = self._copy(block, cursor, active)
                    saved.append(copied)
                    cursor += copied.bias.shape[0]
                    yield copied
                require(cursor == active.units, "complete_active_layer_before_cache_replay_required")
                self._close_iterator(upstream)
                upstream = self._upstream = None
            else:
                require(active == self._active[self._index], "same_iteration_layer_channel_units_required")
                for block in self._blocks[self._index]:
                    self._check()
                    require(not self._closed and not self._failed, "cache_closed_or_failed")
                    yield block
            # A close immediately after the last yield is still incomplete.
            # Review/allocation checks occur after genuine iterator exhaustion.
            self._check_pass()
            self._check()
            require(not self._closed and not self._failed, "cache_closed_or_failed")
            self._index += 1
            if self._index == len(self._order):
                self._index = 0
                if self._phase == "collect":
                    self._phase = "replay"
                else:
                    self._clear()
                    self._phase = "collect"
                    self.completed_iterations += 1
        except BaseException as error:
            self._fail()
            self._close_iterator(upstream, error)
            if self._upstream is not upstream:
                self._close_iterator(self._upstream, error)
            raise
        finally:
            if owns_pass:
                self._upstream = None
                self._busy = False


def _validate_full_active(active):
    require(active.units == interface.active_layer_units(active.layer, active.channel),
            "complete_original_active_layer_size_required")


class TwoPassLossCache(_TwoPassCache):
    """Fixed19-layer, all4712-row two-pass cache for an admitted provider.

    check_pass MUST be the original bound provider's exposed check_binding,
    retaining the same private recovery snapshot/allocation checks. check is
    its allocation-budget callback. Neither callback is reimplemented here.
    Each iteration's changed channel schedule is accepted only after all19
    previous matching replay iterators were exhausted. Failure poisons this
    instance; there is no cache eviction, reread fallback, resume or retry.
    """

    def __init__(self, provider, *, check_pass, check):
        super().__init__(provider, check_pass=check_pass, check=check,
                         layer_order=LAYER_ORDER,
                         expected_rows=interface.IMAGES, validate_active=_validate_full_active,
                         maximum_bytes=MAX_RETAINED_BYTES)


def make_cached_provider(pair, *, budget, block_units=64):
    """Compose the fixed caches only through the original admitted provider.

    This factory does not grant a stage or broaden data roles. Admission occurs
    once in content_block_provider; each retained pass reuses its same private
    binding/allocation guard. The driver owns close(), including measurement
    callback stops and failures. No cap or numerical settings are configurable.
    """
    provider = interface.content_block_provider(pair, budget=budget, block_units=block_units,
                                                reader_factory=ContiguousChannelReader)
    return TwoPassLossCache(provider, check_pass=provider.check_pass, check=budget.check)
