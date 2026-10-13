"""Prospective EP20 90%-power scales on a declared regular rectangular grid.

Array-only development component; no reader, generator, calibration, candidate
execution or independent-final implementation. Input is truth[time, y, x], not
a candidate reconstruction. The supplied dx/dy/dt define uniform coordinates.
Only a full Boolean common mask is supported: holes and submasks are refused,
never filled with zeros or silently cropped. The finite supplied aperture uses
rectangular windows and DFT periodic extension, without taper or padding.

lambda = 1/(2*k90) retains the spatial common mode and uses inclusive 1--100 Hz
temporal DFT bins. tau = 1/(2*f90) uses all positive temporal bins after removing
each spatial sample's temporal mean. Correctly weighted rFFT energies include
Nyquist once. For lambda, temporal-negative spatial spectra are the conjugate
spatial reflection of their positive partners; their radial shell energies are
equal, permitting the same doubling after complete radial-shell aggregation.

Execution belongs in a caller-owned compute allocation. This function keeps
one full temporal spectrum plus finite-aperture arrays; full-world resource
sizing is unresolved. Optional check() calls permit the caller's budget checks
before transforms and between frequency planes, without granting stage access.
"""
from __future__ import annotations

import math
import numbers

import numpy as np


ESTIMATOR = "ep20_rectangular_dft_power90_v1"
POWER_FRACTION = 0.90
PRIMARY_BAND_HZ = (1.0, 100.0)


def _spacing(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an explicit positive finite scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be an explicit positive finite scalar")
    return value


def _power_cutoff(frequencies, power):
    """Smallest represented frequency whose whole-bin cumulative power is 90%."""
    if not len(frequencies):
        return {"status": "undefined_no_frequency_bins", "cutoff": None, "total_power": 0.0,
                "power_fraction_below_cutoff": None, "power_fraction_through_cutoff": None}
    if not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("spectral power must remain finite and nonnegative")
    cumulative = np.cumsum(power, dtype=np.float64)
    total = float(cumulative[-1])
    if not math.isfinite(total):
        raise ValueError("total spectral power is not representable in float64")
    if total == 0:
        return {"status": "undefined_zero_power", "cutoff": None, "total_power": 0.0,
                "power_fraction_below_cutoff": None, "power_fraction_through_cutoff": None}
    index = int(np.searchsorted(cumulative, POWER_FRACTION * total, side="left"))
    cutoff = float(frequencies[index])
    return {"status": "infinite_zero_cutoff" if cutoff == 0 else "finite", "cutoff": cutoff,
            "total_power": total,
            "power_fraction_below_cutoff": float(cumulative[index - 1] / total) if index else 0.0,
            "power_fraction_through_cutoff": float(cumulative[index] / total)}


def _scale(cutoff):
    return None if cutoff is None else (math.inf if cutoff == 0 else 1 / (2 * cutoff))


def field_power_scales(truth, *, dx_micrometres, dy_micrometres, dt_seconds, common_mask, check=None):
    """Return measured scales plus the exact estimator/grid conventions.

    Each grid sample has equal power weight; pixels/time points are not treated
    as independent replicates. Missing samples, irregular masks, implicit grid
    interpolation and a preselected candidate reconstruction are not supported.
    The caller is responsible for the truth/pilot role and declared uniform
    sampling. No coordinates, aperture or voltage-reference correction is inferred.

    ``None`` means undefined (zero power or no primary-band bins), whereas
    ``math.inf`` means defined power whose 90% spatial cutoff is zero. Pixels
    are reported only for exactly equal dx and dy; anisotropic grids retain
    their separate physical spacings and a micrometre scale.
    """
    dx = _spacing("dx_micrometres", dx_micrometres)
    dy = _spacing("dy_micrometres", dy_micrometres)
    dt = _spacing("dt_seconds", dt_seconds)
    if check is None:
        check = lambda: None
    if not callable(check):
        raise ValueError("check must be a callable allocation check")
    check()
    supplied = np.asarray(truth)
    if supplied.dtype.kind not in "fiu" or np.iscomplexobj(supplied):
        raise ValueError("truth must be a real numeric array")
    if supplied.ndim != 3 or supplied.shape[0] < 2 or min(supplied.shape[1:]) < 1:
        raise ValueError("truth must have nonempty [time>=2, y, x] regular-grid axes")
    if not np.isfinite(supplied).all():
        raise ValueError("truth must be finite on every common grid sample")
    mask = np.asarray(common_mask)
    if mask.dtype.kind != "b" or mask.shape != supplied.shape[1:]:
        raise ValueError("common_mask must be an explicit Boolean [y, x] grid")
    if not mask.all():
        raise ValueError("only the full rectangular common mask is implemented; no holes, zero filling or implicit crop")
    values = np.asarray(supplied, dtype=np.float64)
    nt, ny, nx = values.shape
    temporal_frequencies = np.fft.rfftfreq(nt, d=dt)
    kx, ky = np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dy)
    with np.errstate(over="raise", invalid="raise"):
        try:
            radial_squared = ky[:, None] ** 2 + kx[None, :] ** 2
            # Identical computed squared radii form indivisible shells. No
            # user-selected radial bins, tolerance merging or interpolation.
            shells_squared, groups = np.unique(radial_squared.ravel(), return_inverse=True)
            radial_frequencies = np.sqrt(shells_squared)
            if not (np.isfinite(temporal_frequencies).all() and np.isfinite(radial_frequencies).all()
                    and temporal_frequencies[1] > 0):
                raise ValueError("declared frequency grid is not representable")
            check()
            demeaned = values - values.mean(axis=0, keepdims=True)
            spectrum = np.fft.rfft(demeaned, axis=0, norm="ortho")
            del demeaned
            temporal_power = np.zeros(len(temporal_frequencies) - 1, dtype=np.float64)
            radial_power = np.zeros(len(radial_frequencies), dtype=np.float64)
            in_band = ((temporal_frequencies >= PRIMARY_BAND_HZ[0])
                       & (temporal_frequencies <= PRIMARY_BAND_HZ[1]))
            for index in range(1, len(temporal_frequencies)):
                check()
                plane = spectrum[index]
                if not np.isfinite(plane).all():
                    raise ValueError("temporal transform is not finite")
                multiplier = 1.0 if nt % 2 == 0 and index == nt // 2 else 2.0
                temporal_power[index - 1] = multiplier * np.sum(plane.real**2 + plane.imag**2)
                if in_band[index]:
                    transformed = np.fft.fft2(plane, norm="ortho")
                    power = transformed.real**2 + transformed.imag**2
                    radial_power += multiplier * np.bincount(groups, weights=power.ravel(),
                                                             minlength=len(radial_frequencies))
            check()
        except FloatingPointError as error:
            raise ValueError("field power is not representable in float64") from error
    spatial = _power_cutoff(radial_frequencies, radial_power)
    if not in_band.any():
        spatial["status"] = "undefined_no_primary_band_bins"
    temporal = _power_cutoff(temporal_frequencies[1:], temporal_power)
    spatial["k90_cycles_per_micrometre"] = spatial.pop("cutoff")
    wavelength = _scale(spatial["k90_cycles_per_micrometre"])
    # Retain the actual cutoff as well as the scale: infinity is not a missing
    # field scale and zero power must not be mistaken for an infinite scale.
    spatial["micrometres"] = wavelength
    spatial["pixels"] = wavelength / dx if dx == dy and wavelength is not None else None
    spatial["pixel_scale_status"] = "square_spacing" if dx == dy else "undefined_anisotropic_spacing"
    spatial["selected_temporal_bin_count"] = int(np.count_nonzero(in_band))
    temporal["f90_hz"] = temporal.pop("cutoff")
    temporal["seconds"] = _scale(temporal["f90_hz"])
    return {
        "estimator": ESTIMATOR,
        "lambda": spatial,
        "tau": temporal,
        "grid": {
            "axis_order": ["time", "y", "x"], "shape": [nt, ny, nx],
            "dx_micrometres": dx, "dy_micrometres": dy, "dt_seconds": dt,
            "spatial_DFT_period_micrometres_yx": [ny * dy, nx * dx],
            "spatial_sample_centre_span_micrometres_yx": [(ny - 1) * dy, (nx - 1) * dx],
            "temporal_DFT_period_seconds": nt * dt, "temporal_sample_span_seconds": (nt - 1) * dt,
            "maximum_positive_temporal_bin_hz": float(temporal_frequencies[-1]),
            "sampling_nyquist_hz": 1 / (2 * dt),
            "primary_band_upper_edge_at_or_below_nyquist": 100 <= 1 / (2 * dt),
            "common_mask": "full_supplied_rectangle", "crop_applied": False,
        },
        "conventions": {
            "power_fraction": POWER_FRACTION,
            "lambda_temporal_support_hz_inclusive": list(PRIMARY_BAND_HZ),
            "tau_temporal_support": "all_positive_DFT_bins_including_even_length_Nyquist",
            "temporal_demeaning": "separate_time_mean_at_each_spatial_sample_before_transform",
            "lambda_temporal_demeaning_note": "positive_1_to_100_Hz_bins_unchanged_in_exact_arithmetic",
            "spatial_common_mode": "retained_no_spatial_demeaning",
            "window": "rectangular_finite_spatial_and_temporal_aperture",
            "padding": "none", "taper": "none", "boundary": "DFT_periodic_extension",
            "frequency_units": "cycles_per_micrometre_and_cycles_per_second_not_angular",
            "energy": "orthonormal_DFT_squared_modulus_summed_over_all_common_samples",
            "real_signal_energy": "positive_temporal_bins_doubled_except_even_length_Nyquist",
            "radial_grouping": "exact_equal_computed_squared_radii_without_bin_approximation",
            "cutoff": "smallest_observed_frequency_reaching_90_percent_including_complete_tied_shell",
            "zero_power": "undefined_None_no_epsilon_floor", "zero_cutoff": "infinite_scale",
            "irregular_or_partial_mask": "unimplemented_rejected",
            "status": "prospective_development_estimator_no_candidate_or_hardware_claim",
        },
    }
