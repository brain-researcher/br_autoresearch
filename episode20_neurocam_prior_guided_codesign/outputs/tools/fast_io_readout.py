"""O2: vectorized EP20 readout with compact, complete typed output.

Inputs are one fast_io_planner.WindowPlan and the hold means of its split
intervals, hold_means[split, adc] in the frontend's ADC (hold-state) order.
The arithmetic is that of acquisition_readout.AcquisitionReadout, which the
streaming readout reuses: sequential aperture integrals sum(mean*(end-start))
in chronological order, division by the aperture duration, the same
quantizer thresholds with the upper-tie rule (searchsorted side='right' is
bisect_right), and math.fsum digital combinations (bit-exact replica). IDs are
integers: global record index = window * per-window count + local index.

The compact writer keeps every legacy output quantity. Raw aperture means,
codes and validity/clipping flags, and published observation values and
flags are stored per window; identities, timestamps and aperture supports
are regenerated exactly from the validated plan. Per-window and cumulative
per-record energy totals are stored; per-interval work rows are not.
"""
from dataclasses import dataclass
import json
import math

import numpy as np

from fast_io_exact import float_bits, fsum_rows


WORK_FIELDS = (
    "field_source_boundary_work_joule", "supplied_charge_boundary_work_joule",
    "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
    "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule",
    "noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule",
    "physical_boundary_energy_change_joule", "physical_interval_energy_change_joule")


class ReadoutError(ValueError):
    pass


@dataclass(frozen=True)
class QuantizerTable:
    """Array view of one legacy QuantizerTransfer (same thresholds and codes)."""
    input_min_volts: float
    input_max_volts: float
    thresholds: np.ndarray
    levels: np.ndarray

    @classmethod
    def from_transfer(cls, transfer):
        thresholds = np.array(transfer.thresholds_volts, dtype=np.float64)
        levels = np.array(transfer.code_volts, dtype=np.float64)
        thresholds.setflags(write=False)
        levels.setflags(write=False)
        return cls(float(transfer.input_min_volts), float(transfer.input_max_volts), thresholds, levels)

    def convert(self, volts):
        """QuantizerTransfer.convert for arrays: (code, decoded volts, clipped)."""
        volts = np.asarray(volts, dtype=np.float64)
        if not np.all(np.isfinite(volts)):
            raise ReadoutError("conversion input must be a finite real number")
        code = np.searchsorted(self.thresholds, volts, side="right").astype(np.int64)
        return code, self.levels[code], (volts < self.input_min_volts) | (volts > self.input_max_volts)


def _dense(pointer, members, weights):
    counts = np.diff(pointer)
    width = int(counts.max()) if len(counts) else 0
    rows = np.repeat(np.arange(len(counts)), counts)
    columns = np.arange(len(members)) - np.repeat(pointer[:-1], counts)
    index = np.zeros((len(counts), max(width, 1)), dtype=np.int64)
    weight = np.zeros((len(counts), max(width, 1)))
    index[rows, columns] = members
    weight[rows, columns] = weights
    return index, weight, counts


@dataclass(frozen=True)
class WindowReadout:
    """All readout outputs of one window as typed arrays with integer IDs."""
    index: int
    raw_id: np.ndarray
    aperture_mean: np.ndarray
    code: np.ndarray
    decoded: np.ndarray
    raw_valid: np.ndarray
    raw_clipped: np.ndarray
    returned_id: np.ndarray
    returned_value: np.ndarray
    returned_valid: np.ndarray
    returned_clipped: np.ndarray
    observation_id: np.ndarray
    site: np.ndarray
    role: np.ndarray
    value: np.ndarray
    has_value: np.ndarray
    valid: np.ndarray
    clipped: np.ndarray
    assigned: np.ndarray
    available: np.ndarray
    support_ptr: np.ndarray
    support_raw: np.ndarray
    support_start: np.ndarray
    support_end: np.ndarray

    def raw_publication_order(self, completed):
        """Legacy emission order of RawValue: heap order (completed_at, 'r:<id>')."""
        keys = [f"r:{value}" for value in self.raw_id.tolist()]
        return sorted(range(len(keys)), key=lambda i: (float(completed[i]), keys[i]))

    def observation_publication_order(self):
        """Legacy emission order of PublishedValue: (available_at, 'o:<id>')."""
        keys = [f"o:{value}" for value in self.observation_id.tolist()]
        available = self.available.tolist()
        return sorted(range(len(keys)), key=lambda i: (available[i], keys[i]))


class VectorizedReadout:
    """Bit-exact array readout for one validated template (no nuisance overlay)."""

    def __init__(self, template):
        self.template = template
        self.quantizers = tuple(QuantizerTable.from_transfer(q) for q in template.quantizers)
        self.conv_adc = template.input_adc[template.conv_input]
        self.ret_index, self.ret_weight, self.ret_count = _dense(template.ret_conv_ptr, template.ret_conv,
                                                                 template.ret_weights)
        self.obs_index, self.obs_weight, self.obs_count = _dense(template.obs_sample_ptr, template.obs_sample,
                                                                 template.obs_weights)
        supports = []
        for o in range(template.observations):
            samples = template.obs_sample[template.obs_sample_ptr[o]:template.obs_sample_ptr[o + 1]]
            raw = sorted({int(c) for s in samples
                          for c in template.ret_conv[template.ret_conv_ptr[s]:template.ret_conv_ptr[s + 1]]})
            supports.append(raw)
        self.support_ptr = np.zeros(template.observations + 1, dtype=np.int64)
        self.support_ptr[1:] = np.cumsum([len(s) for s in supports])
        self.support_local = np.array([c for s in supports for c in s], dtype=np.int64)

    def aperture_means(self, window, hold_means):
        """Legacy aperture integrals: integral += mean * (end - start), in order."""
        t = self.template
        means = np.asarray(hold_means, dtype=np.float64)
        splits = len(window.split_hold)
        if means.shape != (splits, len(t.adc_ids)):
            raise ReadoutError("hold means must be [split interval, ADC hold] for this window")
        durations = window.split_bounds[1:] - window.split_bounds[:-1]
        first, count = window.conv_split_first, window.conv_split_count
        integral = np.zeros(t.conversions)
        for offset in range(int(count.max())):
            use = offset < count
            split = first[use] + offset
            value = means[split, self.conv_adc[use]]
            if not np.all(np.isfinite(value)):
                raise ReadoutError("analog interval mean must be a finite real number")
            integral[use] = integral[use] + value * durations[split]
        if not np.all(np.isfinite(integral)):
            raise ReadoutError("aperture integral is not representable")
        return integral / (window.conv_aperture_end - window.conv_aperture_start)

    def quantize(self, mean):
        t = self.template
        code = np.zeros(t.conversions, dtype=np.int64)
        decoded = np.zeros(t.conversions)
        clipped = np.zeros(t.conversions, dtype=bool)
        for q, table in enumerate(self.quantizers):
            rows = t.conv_quantizer == q
            if np.any(rows):
                code[rows], decoded[rows], clipped[rows] = table.convert(mean[rows])
        return code, decoded, clipped

    def decode(self, code):
        t = self.template
        decoded = np.zeros(t.conversions)
        for q, table in enumerate(self.quantizers):
            rows = t.conv_quantizer == q
            decoded[rows] = table.levels[code[rows]]
        return decoded

    def read_window(self, window, hold_means):
        mean = self.aperture_means(window, hold_means)
        code, decoded, clipped = self.quantize(mean)
        return self.assemble(window, mean, code, decoded, self.template.conv_valid.copy(), clipped)

    def assemble(self, window, mean, code, decoded, raw_valid, clipped):
        """Returned words and observations from raw outputs (legacy _returned/_publish)."""
        t = self.template
        returned = fsum_rows(self.ret_weight * decoded[self.ret_index], self.ret_count) / t.ret_divisor
        if not np.all(np.isfinite(returned)):
            raise ReadoutError("returned sample must be a finite real number")
        member = np.arange(self.ret_index.shape[1])[None, :] < self.ret_count[:, None]
        returned_valid = np.all(np.where(member, raw_valid[self.ret_index], True), axis=1)
        returned_clipped = np.any(member & clipped[self.ret_index], axis=1)
        for s in np.flatnonzero(t.ret_output_quantizer >= 0):
            table = self.quantizers[t.ret_output_quantizer[s]]
            _, level, extra = table.convert(returned[s:s + 1])
            returned[s], returned_clipped[s] = level[0], returned_clipped[s] or bool(extra[0])
        member = np.arange(self.obs_index.shape[1])[None, :] < self.obs_count[:, None]
        valid = np.all(np.where(member, returned_valid[self.obs_index], True), axis=1)
        observation_clipped = np.any(member & returned_clipped[self.obs_index], axis=1)
        total = fsum_rows(self.obs_weight * returned[self.obs_index], self.obs_count) / t.obs_divisor
        if not np.all(np.isfinite(total[valid])):
            raise ReadoutError("published observation must be a finite real number")
        value = np.where(valid, total, np.nan)
        support_raw = window.conversion_id[self.support_local]
        support_raw = self._legacy_support_order(support_raw)
        local = support_raw - window.conversion_id[0]
        return WindowReadout(
            index=window.index, raw_id=window.conversion_id, aperture_mean=mean, code=code, decoded=decoded,
            raw_valid=raw_valid, raw_clipped=clipped, returned_id=window.returned_id,
            returned_value=returned, returned_valid=returned_valid, returned_clipped=returned_clipped,
            observation_id=window.observation_id, site=t.obs_site, role=t.obs_role, value=value,
            has_value=valid.copy(), valid=valid, clipped=observation_clipped,
            assigned=window.obs_assigned, available=window.obs_available,
            support_ptr=self.support_ptr, support_raw=support_raw,
            support_start=window.conv_aperture_start[local], support_end=window.conv_aperture_end[local])

    def _legacy_support_order(self, support_raw):
        """AcquisitionReadout._apertures sorts raw IDs as strings 'r:<id>'.

        Numeric order equals string order when every ID of one observation has
        the same number of decimal digits; only mixed-digit groups are resorted.
        """
        result = support_raw.copy()
        if not len(result):
            return result
        powers = 10 ** np.arange(1, 19, dtype=np.int64)
        digits = np.searchsorted(powers, support_raw, side="right")
        starts = self.support_ptr[:-1]
        nonempty = np.diff(self.support_ptr) > 0
        low = np.minimum.reduceat(digits, starts[nonempty])
        high = np.maximum.reduceat(digits, starts[nonempty])
        for o in np.flatnonzero(nonempty)[low != high]:
            a, b = self.support_ptr[o], self.support_ptr[o + 1]
            result[a:b] = sorted(support_raw[a:b].tolist(), key=lambda value: f"r:{value}")
        return result


def legacy_raw_values(readout):
    """Materialize legacy RawValue records (interoperability and tests)."""
    from acquisition_readout import RawValue
    return tuple(RawValue(f"r:{i}", m, c, d, v, k) for i, m, c, d, v, k in zip(
        readout.raw_id.tolist(), readout.aperture_mean.tolist(), readout.code.tolist(),
        readout.decoded.tolist(), readout.raw_valid.tolist(), readout.raw_clipped.tolist()))


def legacy_published_values(readout, template):
    """Materialize legacy PublishedValue records (interoperability and tests)."""
    from acquisition_readout import PublishedValue
    result = []
    ptr = readout.support_ptr.tolist()
    support = list(zip(readout.support_raw.tolist(), readout.support_start.tolist(), readout.support_end.tolist()))
    values, valid = readout.value.tolist(), readout.valid.tolist()
    for o, identifier in enumerate(readout.observation_id.tolist()):
        apertures = tuple((f"r:{raw}", start, end) for raw, start, end in support[ptr[o]:ptr[o + 1]])
        result.append(PublishedValue(f"o:{identifier}", template.site_labels[readout.site[o]],
                                     template.observation_roles[readout.role[o]],
                                     values[o] if valid[o] else None, bool(valid[o]),
                                     bool(readout.clipped[o]), float(readout.assigned[o]),
                                     float(readout.available[o]), apertures))
    return tuple(result)


class CompactRecordWriter:
    """One exclusive HDF5 record of per-window typed readout and energy rows.

    Stored per window: raw aperture means (float64), raw codes (uint32),
    packed raw valid/clipped bits, observation values (float64, NaN where no
    value) and packed has-value/valid/clipped bits, per-window energy totals,
    the running per-record totals after the window, and optional per-window
    accounting scalars. Identities, timestamps, sites, roles, supports and
    returned words are regenerated exactly from the plan whose digests are
    stored. ``record_complete`` is set only by finish().
    """

    def __init__(self, path, *, template, plan_digests, windows=4000, accounting_columns=(),
                 retain_aperture_means=True, compression=None):
        import h5py
        if type(retain_aperture_means) is not bool:
            raise ReadoutError("aperture-mean retention must be explicit")
        self.template, self.windows = template, int(windows)
        self.accounting_columns = tuple(accounting_columns)
        self.retain_aperture_means = retain_aperture_means
        self._file = h5py.File(path, "x")
        self._closed = self._finished = False
        r, o = template.conversions, template.observations
        options = {} if compression is None else dict(compression=compression, shuffle=True)
        f = self._file
        f.attrs["record_complete"] = False
        f.attrs["simulation_status"] = "incomplete"
        f.attrs["plan_digests_json"] = json.dumps(plan_digests, sort_keys=True)
        f.attrs["work_fields_json"] = json.dumps(WORK_FIELDS)
        f.attrs["accounting_columns_json"] = json.dumps(self.accounting_columns)
        f.attrs["identity_rule"] = "global_id = window * per_window_count + local_index"
        f.attrs["retain_aperture_means"] = retain_aperture_means
        if retain_aperture_means:
            f.create_dataset("raw/aperture_mean", (self.windows, r), dtype="<f8", chunks=(1, r), **options)
        f.create_dataset("raw/code", (self.windows, r), dtype="<u4", chunks=(1, r), **options)
        f.create_dataset("raw/valid_bits", (self.windows, (r + 7) // 8), dtype="u1")
        f.create_dataset("raw/clipped_bits", (self.windows, (r + 7) // 8), dtype="u1")
        f.create_dataset("observations/value", (self.windows, o), dtype="<f8", chunks=(1, o), **options)
        f.create_dataset("observations/flags", (self.windows, o), dtype="u1", chunks=(1, o), **options)
        f.create_dataset("energy/window_totals", (self.windows, len(WORK_FIELDS)), dtype="<f8")
        f.create_dataset("energy/cumulative_totals", (self.windows, len(WORK_FIELDS)), dtype="<f8")
        if self.accounting_columns:
            f.create_dataset("accounting/window", (self.windows, len(self.accounting_columns)), dtype="<f8")
        f.create_dataset("windows_written", (self.windows,), dtype="?")
        self._next = 0

    def write_window(self, readout, *, energy_window, energy_cumulative, accounting=None):
        if self._closed or self._finished:
            raise ReadoutError("record writer is closed")
        k = readout.index
        if k != self._next or not 0 <= k < self.windows:
            raise ReadoutError("windows must be written once, in order")
        energy_window = np.array([float(energy_window[name]) for name in WORK_FIELDS])
        energy_cumulative = np.array([float(energy_cumulative[name]) for name in WORK_FIELDS])
        if not (np.all(np.isfinite(energy_window)) and np.all(np.isfinite(energy_cumulative))):
            raise ReadoutError("energy totals must be finite")
        if np.any(readout.code < 0) or np.any(readout.code >= 2**32):
            raise ReadoutError("raw code cannot be represented in uint32")
        f = self._file
        if self.retain_aperture_means:
            f["raw/aperture_mean"][k] = readout.aperture_mean
        f["raw/code"][k] = readout.code.astype("<u4")
        f["raw/valid_bits"][k] = np.packbits(readout.raw_valid)
        f["raw/clipped_bits"][k] = np.packbits(readout.raw_clipped)
        f["observations/value"][k] = readout.value
        f["observations/flags"][k] = (readout.has_value.astype(np.uint8) | (readout.valid.astype(np.uint8) << 1)
                                      | (readout.clipped.astype(np.uint8) << 2))
        f["energy/window_totals"][k] = energy_window
        f["energy/cumulative_totals"][k] = energy_cumulative
        if self.accounting_columns:
            values = np.array([float(accounting[name]) for name in self.accounting_columns])
            if not np.all(np.isfinite(values)):
                raise ReadoutError("accounting scalars must be finite")
            f["accounting/window"][k] = values
        f["windows_written"][k] = True
        self._next += 1

    def finish(self, *, status, metadata):
        if status not in ("complete", "failed") or self._closed or self._finished:
            raise ReadoutError("explicit terminal status on an open writer required")
        f = self._file
        f.attrs["terminal_metadata_json"] = json.dumps(metadata, sort_keys=True, default=repr)
        f.attrs["windows_completed"] = self._next
        if self._next:
            f.attrs["record_energy_totals_json"] = json.dumps(dict(zip(
                WORK_FIELDS, f["energy/cumulative_totals"][self._next - 1].tolist())))
        f.attrs["simulation_status"] = status
        f.flush()
        f.attrs["record_complete"] = status == "complete" and self._next == self.windows
        f.flush()
        self._finished = True

    def close(self):
        if not self._closed:
            self._file.close()
            self._closed = True


def read_window(path, plan, k):
    """Regenerate one WindowReadout from a compact record and its plan.

    Identities, timestamps and supports come from the plan; returned words and
    observations are recomputed from the stored codes with the same exact
    arithmetic and must equal the stored observation values and flags, and
    stored means must reproduce the stored codes (integrity check, no repair).
    """
    import h5py
    t = plan.template
    readout = VectorizedReadout(t)
    window = plan.window(k)
    with h5py.File(path, "r") as f:
        if json.loads(f.attrs["plan_digests_json"]).get("template_digest") != t.digest:
            raise ReadoutError("record and plan template digests differ")
        if not bool(f["windows_written"][k]):
            raise ReadoutError("window was not written")
        code = f["raw/code"][k].astype(np.int64)
        mean = f["raw/aperture_mean"][k] if "raw/aperture_mean" in f else None
        raw_valid = np.unpackbits(f["raw/valid_bits"][k])[:t.conversions].astype(bool)
        raw_clipped = np.unpackbits(f["raw/clipped_bits"][k])[:t.conversions].astype(bool)
        value = f["observations/value"][k]
        flags = f["observations/flags"][k]
    if mean is not None:
        recoded, _, reclipped = readout.quantize(mean)
        if not np.array_equal(recoded, code) or not np.array_equal(reclipped, raw_clipped):
            raise ReadoutError("stored aperture means and codes disagree")
    else:
        mean = np.full(t.conversions, np.nan)
    result = readout.assemble(window, mean, code, readout.decode(code), raw_valid, raw_clipped)
    if (not np.array_equal(float_bits(result.value), float_bits(value))
            or not np.array_equal(result.has_value, (flags & 1).astype(bool))
            or not np.array_equal(result.valid, ((flags >> 1) & 1).astype(bool))
            or not np.array_equal(result.clipped, ((flags >> 2) & 1).astype(bool))):
        raise ReadoutError("stored observations disagree with their stored codes")
    return result


def readouts_bitwise_equal(left, right, *, include_means=True):
    """Field-by-field bitwise comparison of two WindowReadout objects."""
    names = [name for name in WindowReadout.__dataclass_fields__ if name != "index"]
    if not include_means:
        names.remove("aperture_mean")
    differing = []
    for name in names:
        a, b = np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
        if a.shape != b.shape:
            differing.append(name)
        elif a.dtype.kind == "f" or b.dtype.kind == "f":
            if not np.array_equal(float_bits(a), float_bits(b)):
                differing.append(name)
        elif not np.array_equal(a, b):
            differing.append(name)
    return differing
