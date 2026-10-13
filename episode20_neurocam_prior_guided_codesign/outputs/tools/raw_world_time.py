"""Analytic raw-time extension of the selected EP20 development-world recipe.

No data loading, random draws, world regeneration, fitting, FFT evaluation
filter, hold approximation, electrical state advance or DAQ clock is selected.
Only the already generated raw recipe and its saved amplitude coefficients are
replayed. Times belong to [0,16); the endpoint16 is a boundary, not a sample.
M4 and dropout use right-continuous, half-open intervals. All observation-side
nuisances remain separate from raw truth and contact voltage.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from development_worlds import CONDITIONS, DevelopmentWorld, RECIPE, REPLICATE_SEEDS
from surface_to_contact import ContactFactors

DURATION_S = 16.
NATIVE_DT_S = .002
NATIVE_COUNT = 8000
# Numerical replay consistency only, never physical calibration acceptance.
# The per-mode absolute tolerance is this fraction times max(1uV,native peak).
REPLAY_TOLERANCE = 5e-12
# The next binary64 value above the usual nearest representation of pi;
# it is strictly greater than mathematical pi and the generator's np.pi.
PI_UPPER = float.fromhex("0x1.921fb54442d19p+1")


def _upper_result(value, exact_zero=False):
    """Outward round a correctly-rounded nonnegative IEEE754 basic operation."""
    result = 0. if exact_zero else math.nextafter(float(value), math.inf)
    if not math.isfinite(result) or result < 0:
        raise ValueError("nonnegative bound arithmetic is not representable")
    return result


def _upper_mul(left, right):
    return _upper_result(left * right, left == 0 or right == 0)


def _upper_add(left, right):
    return _upper_result(left + right, left == 0 and right == 0)


def _upper_div(left, right):
    if right <= 0:
        raise ValueError("bound denominator must be positive")
    return _upper_result(left / right, left == 0)


def _limit(max_elements, *sizes):
    if type(max_elements) is not int or max_elements < 1 or any(size > max_elements for size in sizes):
        raise ValueError("explicit per-array max_elements exceeded")


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or positive and result <= 0:
        raise ValueError(f"{name} must be finite and positive when required")
    return result


def _array(value, name, ndim, max_elements):
    if (not isinstance(value, np.ndarray) or value.ndim != ndim or not value.size
            or value.dtype.kind not in "fiu"):
        raise ValueError(f"{name} must be an explicit nonempty real array")
    _limit(max_elements, value.size)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return np.asarray(value, dtype=np.float64)


def _quadrature(times, frequency, phase=0.):
    angle = 2 * np.pi * frequency * times + phase
    return np.stack((np.cos(angle), np.sin(angle)))


class _BoundContactTime:
    """A validated, owned spatial snapshot; no repeated native-record scans."""
    def __init__(self, raw_time, modes):
        self._raw_time = raw_time
        self._modes = modes.copy()
        self._modes.setflags(write=False)
        self._absolute_modes = np.abs(self._modes)
        self._absolute_modes.setflags(write=False)

    def volts(self, times_s):
        times = self._raw_time._times(times_s)
        _limit(self._raw_time._max_elements, 4096 * len(times))
        result = (self._modes @ self._raw_time.drivers_uv(times)) * 1e-6
        if not np.isfinite(result).all():
            raise ValueError("contact voltage is not representable")
        return result

    def derivative_bound_volts_per_second(self, start, end):
        """Conservative whole-contact Lipschitz bound on one raw smooth segment.

Each nonnegative multiply and add is rounded upward separately. BLAS sums and
norm heuristics are not used. This encloses the analytic factor sum; it does
not certify transcendental waveform evaluation or an electrical solver.
"""
        rates = self._raw_time.derivative_bounds(start, end)
        total = np.zeros(4096)
        for mode in np.flatnonzero(rates):
            self._raw_time._check()
            column = self._absolute_modes[:, mode]
            product = np.where(column == 0., 0., np.nextafter(column * rates[mode], np.inf))
            summed = total + product
            total = np.where(summed == 0., 0., np.nextafter(summed, np.inf))
        if not np.isfinite(total).all():
            raise ValueError("contact derivative bound is not representable")
        return _upper_div(float(total.max()), 1e6)


class RawWorldTime:
    """Immutable recipe snapshot with bounded native-sample replay validation.

The spatial grid can be a declared private generator fixture because only
raw time is evaluated; its production/fixture status is retained. The temporal
record must be the selected8000 samples at500Hz. No empirical/final role is
accepted, no changed amplitude is fitted and no saved artifact is modified.
"""
    def __init__(self, world, *, max_elements, check=None):
        if not isinstance(world, DevelopmentWorld):
            raise ValueError("explicit DevelopmentWorld required; this component reads no files")
        if check is None:
            check = lambda: None
        if not callable(check):
            raise ValueError("check must be callable")
        _limit(max_elements, NATIVE_COUNT)
        self._max_elements, self._check = max_elements, check
        check()
        meta = deepcopy(world.metadata)
        condition = meta.get("condition_id")
        seed = meta.get("replicate_seed")
        if (meta.get("recipe") != RECIPE or condition not in CONDITIONS
                or type(seed) is not int or seed not in REPLICATE_SEEDS
                or meta.get("world_id") != f"{condition}:{seed}"
                or meta.get("independence_group") != f"development_replicate_{seed}"
                or meta.get("role") != "development_preparation_unassigned"
                or meta.get("final_world_access") is not False
                or meta.get("grid_shape_time_y_x", [None])[0] != NATIVE_COUNT):
            raise ValueError("selected development recipe, temporal grid, identity and unassigned role required")
        native = _array(world.raw_drivers, "saved raw drivers", 2, max_elements)
        if native.shape[1] != NATIVE_COUNT or meta.get("factor_modes") != len(native):
            raise ValueError("saved raw driver shape must match declared mode count and time grid")
        self._native = native.copy()
        self._native.setflags(write=False)
        self._condition, self._rank = condition, len(native)
        self._details = deepcopy(meta["details"])
        self._world_id, self._group = meta["world_id"], meta["independence_group"]
        self._production_grid = bool(meta["production_grid"])
        self._scales, self._phases, self._events = None, None, ()
        self._support, self._tapers, self._jumps = [], [], []
        self._packed = None
        self._line_scale = None
        self._configure()
        self._configure_nuisance(world.nuisance_arrays)
        peaks = np.maximum(1., np.max(np.abs(self._native), axis=1))
        largest = 0.
        for first in range(0, NATIVE_COUNT, 512):
            check()
            last = min(first + 512, NATIVE_COUNT)
            times = np.arange(first, last) * NATIVE_DT_S
            replay = self.drivers_uv(times)
            error = np.abs(replay - self._native[:, first:last])
            allowance = REPLAY_TOLERANCE * (peaks[:, None] + np.abs(self._native[:, first:last]))
            if np.any(error > allowance):
                raise ValueError("saved raw drivers disagree with analytic recipe; no refit or fallback permitted")
            largest = max(largest, float(error.max()))
        self._report = {"world_id": self._world_id, "independence_group": self._group,
            "role": "development_preparation_unassigned", "production_grid": self._production_grid,
            "time_domain_s": [0., DURATION_S], "right_endpoint_excluded": True,
            "native_dt_s": NATIVE_DT_S, "native_sample_count": NATIVE_COUNT,
            "native_replay_max_absolute_error_uv": largest, "native_replay_tolerance": REPLAY_TOLERANCE,
            "replay_absolute_tolerance_rule": "tolerance*max(1uV,mode_native_peak)",
            "raw_amplitudes_refitted": False, "primary_band_drivers_used": False,
            "hold_or_interpolation_selected": False, "electrical_state_advanced": False,
            "nuisances_applied_to_truth": False, "actual_DAQ_identified": False,
            "M4_extension": "analytic_selected_bin_aligned_common_envelope_carriers_before_exact_saved_steps"}

    def _configure(self):
        details, condition = self._details, self._condition
        if condition[0] in "BNW":
            if self._rank != 2:
                raise ValueError("carrier quadratures require two modes")
            self._frequency = _scalar(details["carrier_hz"], "carrier_hz", positive=True)
            self._scale = _scalar(details["applied_scale"], "applied_scale", positive=True)
            expected = {"B1": 3., "B2": 11., "B3": 37., "B4": 80.,
                        "W1": 7., "W2": 11., "W3": 37., "W4": 37.}.get(condition, 11.)
            if self._frequency != expected:
                raise ValueError("carrier differs from selected condition")
        elif condition[0] == "I":
            phases = np.asarray(details["phases_rad"], dtype=float)
            if phases.shape != (self._rank,) or not np.isfinite(phases).all() or self._rank != (4 if condition == "I4" else 2):
                raise ValueError("interaction phase count must match selected stored modes")
            self._phases = phases.copy()
            self._scale = _scalar(details["applied_scale"], "applied_scale", positive=True)
            self._frequency = _scalar(details["carrier_hz"], "carrier_hz", positive=True)
            if self._frequency != 11.:
                raise ValueError("interaction carrier differs from selected recipe")
        elif condition[0] == "M":
            if self._rank != 4:
                raise ValueError("mixtures require four quadrature modes")
            if condition != "M4":
                self._scales = tuple(_scalar(details[name]["applied_scale"], name, positive=True)
                    for name in ("broad_normalization", "local_normalization"))
            else:
                self._phases = tuple(_scalar(details[name], name) for name in
                    ("common_envelope_phase_rad", "common_carrier_phase_rad"))
                self._scales = np.array([details[name]["interval_amplitudes"] for name in
                    ("broad_switch_normalization", "local_switch_normalization")], dtype=float)
                if self._scales.shape != (2, 16) or not np.isfinite(self._scales).all() or np.any(self._scales <= 0):
                    raise ValueError("M4 requires two exact positive16-interval amplitude vectors")
                self._jumps = list(map(float, range(1, 16)))
        else:
            events = details["events"]
            if not isinstance(events, list) or not events or len(events) != details["complete_event_count"]:
                raise ValueError("complete saved event records are required")
            _limit(self._max_elements, len(events) * 6)
            retained, seen = [], set()
            minimum = _scalar(details["minimum_local_prelude_s"], "minimum_local_prelude_s")
            for event in events:
                self._check()
                mode = event["mode_index"]
                start = _scalar(event["support_start_s"], "support_start_s")
                end = _scalar(event["support_end_s"], "support_end_s")
                width = _scalar(event["nominal_envelope_fwhm_s"], "nominal_envelope_fwhm_s", positive=True)
                scale = _scalar(event["applied_scale"], "event applied_scale", positive=True)
                sigma = width / (2 * math.sqrt(2 * math.log(2)))
                if (type(mode) is not int or not 0 <= mode < self._rank or not 0 <= minimum <= start < end
                        or end > (NATIVE_COUNT - 1) * NATIVE_DT_S
                        or not math.isclose(end, start + 6 * sigma, rel_tol=0., abs_tol=2e-14)):
                    raise ValueError("event support, mode or complete-window convention is inconsistent")
                retained.append((mode, start, end, sigma, scale))
                seen.add(mode)
                self._support.extend((start, end))
                self._tapers.extend((start + .5 * sigma, start + 5.5 * sigma))
            if seen != set(range(self._rank)):
                raise ValueError("every stored event mode needs explicit recipe parameters")
            # Generator sums events within a mode in chronological order. Restore
            # that order even though its metadata is globally sorted by onset.
            self._events = tuple(sorted(retained, key=lambda event: (event[0], event[1])))

    def _configure_nuisance(self, arrays):
        if self._condition[0] != "N":
            return
        nuisance = self._details["nuisance"]
        if nuisance.get("observation_side_only") is not True:
            raise ValueError("nuisance must remain observation-side only")
        if self._condition in ("N1", "N2"):
            frequency, rms = (60., 5.) if self._condition == "N1" else (50., 50.)
            self._line_frequency = _scalar(nuisance["line_hz"], "line_hz", positive=True)
            self._line_phase = _scalar(nuisance["line_phase_rad"], "line_phase_rad")
            if self._line_frequency != frequency or not math.isclose(float(nuisance["line_rms_uv"]), rms, rel_tol=2e-14):
                raise ValueError("line nuisance differs from selected recipe")
            native_times = np.arange(NATIVE_COUNT) * NATIVE_DT_S
            native_sine = np.sin(2 * np.pi * frequency * native_times + self._line_phase)
            # Reproduce the generator's fixed finite-grid normalization exactly;
            # this is not a fit to samples or an inferred physical noise level.
            self._line_scale = rms / math.sqrt(float(np.mean(native_sine**2)))
            supplied = _array(arrays["spatially_common_line_uv"], "saved line nuisance", 1, self._max_elements)
            if supplied.shape != (NATIVE_COUNT,) or not np.allclose(supplied, native_sine * self._line_scale,
                                                                    rtol=REPLAY_TOLERANCE, atol=rms * REPLAY_TOLERANCE):
                raise ValueError("saved line nuisance disagrees with its fixed recipe")
        elif self._condition == "N3":
            packed = arrays["missing_sites_packed_little"]
            if (not isinstance(packed, np.ndarray) or packed.dtype != np.uint8 or packed.shape != (NATIVE_COUNT, 512)
                    or nuisance.get("nuisance_clock_hz") != 500.
                    or nuisance.get("bitorder") != "little" or nuisance.get("one_means") != "missing"
                    or nuisance.get("physical_site_shape") != [64, 64]):
                raise ValueError("saved500Hz little-endian site mask and explicit metadata required")
            _limit(self._max_elements, packed.size)
            self._packed = packed.copy()
            self._packed.setflags(write=False)
            # Keep the exact represented clock edges once. Streaming pixel
            # queries must not recreate an 8000-element clock per observation.
            self._nuisance_edges = np.arange(NATIVE_COUNT) * NATIVE_DT_S
            self._nuisance_edges.setflags(write=False)
        else:
            row = nuisance["dead_physical_row_zero_based"]
            if type(row) is not int or not 0 <= row < 64:
                raise ValueError("N4 requires one saved physical row")
            self._dead_row = row

    def _times(self, times_s):
        times = _array(times_s, "query times", 1, self._max_elements)
        if np.any(times < 0.) or np.any(times >= DURATION_S):
            raise ValueError("queries must lie in the finite half-open interval[0,16)")
        _limit(self._max_elements, self._rank * len(times))
        return times

    @property
    def metadata(self):
        return deepcopy(self._report)

    def drivers_uv(self, times_s):
        """Return[mode,time] raw drivers; duplicate/unsorted query times allowed."""
        times = self._times(times_s)
        self._check()
        condition = self._condition
        if condition[0] in "BNW":
            result = _quadrature(times, self._frequency) * self._scale
        elif condition[0] == "I":
            result = np.cos(2 * np.pi * self._frequency * times[None, :] + self._phases[:, None]) * self._scale
        elif condition[0] == "M":
            broad, local = _quadrature(times, 3.), _quadrature(times, 37.)
            if condition == "M4":
                phase_envelope, phase_carrier = self._phases
                envelope = 1 + .25 * np.cos(2 * np.pi * .5 * times + phase_envelope)
                intervals = np.floor(times).astype(int)
                broad = _quadrature(times, 3., phase_carrier) * envelope[None, :] * self._scales[0, intervals]
                local = _quadrature(times, 37., phase_carrier) * envelope[None, :] * self._scales[1, intervals]
            else:
                broad *= self._scales[0]
                local *= self._scales[1]
            result = np.concatenate((broad, local))
        else:
            result = np.zeros((self._rank, len(times)))
            for mode, start, end, sigma, scale in self._events:
                self._check()
                active = (times > start) & (times < end)
                if not np.any(active):
                    continue
                z = (times[active] - start) / sigma - 3
                absolute = np.abs(z)
                taper = np.ones(len(z))
                outer = absolute > 2.5
                u = 2 * (absolute[outer] - 2.5)
                taper[outer] = np.clip((1 - u)**3 * (1 + 3 * u + 6 * u**2), 0., 1.)
                # Support endpoints are exactly zero; no DFT interpolation can
                # leak future event information into a quiet prelude.
                taper[absolute >= 3.] = 0.
                result[mode, active] += (-z * taper * np.exp(-.5 * z**2)) * scale
        if not np.isfinite(result).all():
            raise ValueError("analytic raw driver is not representable")
        return result

    def bind_contact_factors(self, contact_factors):
        """Validate identity/raw equality once and own an immutable mode snapshot.

The returned evaluator's volts(times_s) performs bounded raw evaluation without
rescanning the native8000-sample record or the full spatial-mode matrix on each
interval. It chooses no evaluation times or interval-hold convention.
"""
        if not isinstance(contact_factors, ContactFactors):
            raise ValueError("explicit existing ContactFactors required")
        if (contact_factors.metadata.get("world_id") != self._world_id
                or contact_factors.metadata.get("independence_group") != self._group
                or contact_factors.metadata.get("role") != "development_preparation_unassigned"
                or not np.array_equal(contact_factors.raw_drivers, self._native)):
            raise ValueError("contact factors must preserve this world's exact saved raw drivers and role")
        modes = _array(contact_factors.contact_modes, "contact modes", 2, self._max_elements)
        if modes.shape != (4096, self._rank):
            raise ValueError("all4096contact sites and matching modes required")
        self._check()
        return _BoundContactTime(self, modes)

    def derivative_bounds(self, start, end):
        """Upper bounds[mode] in microvolts/second on one half-open raw segment.

Every interior support/taper/M4 breakpoint must be split first. Sinusoids use
2*pi*f*scale; M4 includes the envelope derivative and its unchanged interval
coefficient. For a finite event w(z)=-z*exp(-z*z/2)*T(z), |w'(z)|<=1:
the untapered core has this bound; in2.5<|z|<3, |T|<=1 and |T'|<=15/4 give
|w'| <= (8+45/4)*exp(-25/8)<1. Summed event bounds retain every overlap.
All scalar nonnegative bound operations round upward. These are bounds for
the analytic waveform, separately from floating evaluation/circuit error.
"""
        start = _scalar(start, "derivative start")
        end = _scalar(end, "derivative end")
        if not 0 <= start < end <= DURATION_S:
            raise ValueError("derivative interval must lie in[0,16]")
        if any(start < point < end for point in self._support + self._tapers + self._jumps):
            raise ValueError("split every raw support/taper/M4 breakpoint before bounding")
        self._check()
        condition = self._condition
        result = np.zeros(self._rank)
        if condition[0] in "BNWI":
            omega = _upper_mul(_upper_mul(2., PI_UPPER), self._frequency)
            result[:] = _upper_mul(omega, self._scale)
        elif condition[0] == "M":
            for component, frequency in enumerate((3., 37.)):
                omega = _upper_mul(_upper_mul(2., PI_UPPER), frequency)
                if condition == "M4":
                    interval = math.floor(start)
                    scale = float(self._scales[component, interval])
                    rate = _upper_mul(scale, _upper_add(_upper_mul(1.25, omega),
                                                        _upper_mul(.25, PI_UPPER)))
                else:
                    rate = _upper_mul(omega, self._scales[component])
                result[2 * component:2 * component + 2] = rate
        else:
            for mode, event_start, event_end, sigma, scale in self._events:
                if start < event_end and end > event_start:
                    result[mode] = _upper_add(float(result[mode]), _upper_div(scale, sigma))
        return result

    def contact_volts(self, contact_factors, times_s):
        """One-off convenience; repeated interval callers should bind once."""
        return self.bind_contact_factors(contact_factors).volts(times_s)

    def nuisance_at(self, times_s, *, site_indices=None):
        """Return separate nuisance values; never inject them into truth/contact modes."""
        times = self._times(times_s)
        self._check()
        output = {"observation_side_only": True, "applied_to_truth": False}
        if self._condition in ("N1", "N2"):
            output["spatially_common_line_uv"] = self._line_scale * np.sin(2 * np.pi * self._line_frequency * times + self._line_phase)
        elif self._condition in ("N3", "N4"):
            if (not isinstance(site_indices, np.ndarray) or site_indices.ndim != 1 or not site_indices.size
                    or site_indices.dtype.kind not in "iu" or np.any(site_indices < 0) or np.any(site_indices >= 4096)):
                raise ValueError("explicit bounded integer row-major physical site indices required")
            _limit(self._max_elements, site_indices.size, len(times) * len(site_indices))
            if self._condition == "N3":
                # Search the actual represented binary clock edges rather than
                # multiplying by500 and accidentally crossing a rounded boundary.
                bins = np.searchsorted(self._nuisance_edges, times, side="right") - 1
                bytes_at_site = self._packed[bins[:, None], site_indices[None, :] // 8]
                missing = ((bytes_at_site >> (site_indices[None, :] % 8)) & 1).astype(bool)
            else:
                missing = np.broadcast_to((site_indices // 64 == self._dead_row)[None, :],
                                           (len(times), len(site_indices))).copy()
            output["missing_sites"] = missing
        return output

    def nuisance_at_pairs(self, times_s, *, site_indices):
        """One observation-side nuisance for each explicit time/site pair.

        Both inputs are equal-length one-dimensional arrays. This is a paired
        lookup, never the Cartesian product returned by ``nuisance_at``. The
        same saved phases, finite-grid line normalization, packed mask, exact
        represented clock edges and dead row are reused without draws/refits.
        """
        times = self._times(times_s)
        if (not isinstance(site_indices, np.ndarray) or site_indices.ndim != 1
                or not site_indices.size or site_indices.shape != times.shape
                or site_indices.dtype.kind not in "iu" or np.any(site_indices < 0)
                or np.any(site_indices >= 4096)):
            raise ValueError("paired nuisance query needs equal-length bounded time/site arrays")
        _limit(self._max_elements, site_indices.size)
        self._check()
        output = {"observation_side_only": True, "applied_to_truth": False}
        if self._condition in ("N1", "N2"):
            output["spatially_common_line_uv"] = self._line_scale * np.sin(
                2 * np.pi * self._line_frequency * times + self._line_phase)
        elif self._condition == "N3":
            bins = np.searchsorted(self._nuisance_edges, times, side="right") - 1
            bytes_at_site = self._packed[bins, site_indices // 8]
            output["missing_sites"] = ((bytes_at_site >> (site_indices % 8)) & 1).astype(bool)
        elif self._condition == "N4":
            output["missing_sites"] = site_indices // 64 == self._dead_row
        return output

    def breakpoint_metadata(self, *, include_nuisance=False):
        """Declared raw changes only; contains no proposed hold/DAQ schedule."""
        if type(include_nuisance) is not bool:
            raise ValueError("include_nuisance must be explicit boolean")
        nuisance = (np.arange(1, NATIVE_COUNT) * NATIVE_DT_S).tolist() if include_nuisance and self._condition == "N3" else []
        _limit(self._max_elements, len(self._support) + len(self._tapers) + len(self._jumps) + len(nuisance) + 2)
        return {"window_endpoints_s": [0., DURATION_S],
                "support_boundaries_s": sorted(set(self._support)),
                "taper_piece_boundaries_s": sorted(set(self._tapers)),
                "raw_amplitude_discontinuities_s": list(self._jumps),
                "observation_nuisance_clock_boundaries_s": nuisance,
                "boundary_convention": "right_continuous_M4_and_dropout;finite_event_endpoints_zero",
                "electrical_hold_or_DAQ_schedule": "not_selected"}

    def breakpoints(self, *, include_nuisance=False):
        report = self.breakpoint_metadata(include_nuisance=include_nuisance)
        values = set()
        for key in ("window_endpoints_s", "support_boundaries_s", "taper_piece_boundaries_s",
                    "raw_amplitude_discontinuities_s", "observation_nuisance_clock_boundaries_s"):
            values.update(report[key])
        return np.array(sorted(values), dtype=float)
