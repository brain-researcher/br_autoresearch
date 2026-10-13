"""Exact finite-record power90 statistics for separable development fields.

truth[t,y,x] = sum_r spatial_modes[r,y,x] * temporal_drivers[r,t].
The estimator, aperture, bins and shell grouping are those of field_power_scales;
factorization changes evaluation cost, not the scientific statistic. All cross
terms remain. No dense time-by-space world, eigenvalue truncation, power floor,
new taper or approximation is introduced. Cancellation-sensitive Gram products
are recomputed directly from the factors in bounded blocks, not clipped.

Array-only implementation. Execution belongs in a coordinated Slurm allocation.
max_elements bounds each numeric input/work array, not their aggregate bytes;
the caller must reserve a constant multiple, including complex128 spatial FFTs.
"""
from __future__ import annotations

import numpy as np

from field_power_scales import (
    ESTIMATOR, POWER_FRACTION, PRIMARY_BAND_HZ, _power_cutoff, _scale, _spacing,
)


def _array(value, name, ndim, max_elements):
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an explicit NumPy array")
    if (value.ndim != ndim or any(size < 1 for size in value.shape)
            or value.dtype.kind not in "fiu" or value.size > max_elements):
        raise ValueError(f"{name} has invalid real shape/dtype or exceeds max_elements")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return np.asarray(value, dtype=np.float64)


def _powers(gram, vectors, *, gram_sum_length, direct, check):
    """Squared norms through a complete Gram, with direct cancellation fallback.

The bound only selects a more stable calculation; it never sets power to zero.
The direct callback returns the squared norm of the reconstructed coefficient
vector, retaining arbitrarily small positive power and exact cancellation.
"""
    product = gram @ vectors
    powers = np.real(np.sum(vectors.conj() * product, axis=0))
    magnitude = np.abs(vectors)
    absolute_sum = np.sum(magnitude * (np.abs(gram) @ magnitude), axis=0)
    if not np.isfinite(powers).all() or not np.isfinite(absolute_sum).all():
        raise ValueError("factorized field power is not representable in float64")
    eps = np.finfo(np.float64).eps
    operations = 32 * (gram_sum_length + len(gram) + 2)
    if operations * eps >= 0.5:
        raise ValueError("factor dimensions exceed finite error-bound arithmetic")
    gamma = operations * eps / (1 - operations * eps)
    uncertain = (powers < 0) | ((absolute_sum > 0) & (powers <= gamma * absolute_sum))
    for index in np.flatnonzero(uncertain):
        check()
        powers[index] = direct(vectors[:, index])
    if not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError("direct factorized field power must be finite and nonnegative")
    return powers


def factorized_field_power_scales(spatial_modes, temporal_drivers, *,
        dx_micrometres, dy_micrometres, dt_seconds, common_mask,
        max_elements, check=None):
    """Return the existing power90 report without materializing full truth.

Inputs have [mode,y,x] and [mode,time] axes. The full common rectangular mask
is mandatory. Drivers are raw truth, not band-filtered evaluation targets:
lambda selects 1--100 Hz bins but tau uses every positive temporal bin.
Finite-record time demeaning is applied to each driver, which commutes with
the spatial sum. Summation roundoff can differ from dense evaluation.
"""
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be an explicit positive integer")
    dx = _spacing("dx_micrometres", dx_micrometres)
    dy = _spacing("dy_micrometres", dy_micrometres)
    dt = _spacing("dt_seconds", dt_seconds)
    if check is None:
        check = lambda: None
    if not callable(check):
        raise ValueError("check must be a callable allocation check")
    check()
    spatial = _array(spatial_modes, "spatial_modes", 3, max_elements)
    temporal = _array(temporal_drivers, "temporal_drivers", 2, max_elements)
    modes, ny, nx = spatial.shape
    nt = temporal.shape[1]
    if temporal.shape[0] != modes or nt < 2 or modes * modes > max_elements:
        raise ValueError("matching factor modes,time>=2 and bounded Gram size required")
    if (not isinstance(common_mask, np.ndarray) or common_mask.dtype != np.bool_
            or common_mask.shape != (ny, nx) or not common_mask.all()):
        raise ValueError("only the full rectangular common mask is implemented")
    pixels = ny * nx
    temporal_frequencies = np.fft.rfftfreq(nt, d=dt)
    kx, ky = np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dy)
    in_band = ((temporal_frequencies >= PRIMARY_BAND_HZ[0])
               & (temporal_frequencies <= PRIMARY_BAND_HZ[1]))
    # Each block's mode-by-location/frequency matrix remains bounded. Choosing
    # smaller blocks affects only evaluation order, never frequency support.
    block_size = max(1, min(2048, max_elements // modes))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            radial_squared = ky[:, None] ** 2 + kx[None, :] ** 2
            shells_squared, groups = np.unique(radial_squared.ravel(), return_inverse=True)
            radial_frequencies = np.sqrt(shells_squared)
            if not (np.isfinite(temporal_frequencies).all()
                    and np.isfinite(radial_frequencies).all() and temporal_frequencies[1] > 0):
                raise ValueError("declared frequency grid is not representable")
            check()
            demeaned = temporal - temporal.mean(axis=1, keepdims=True)
            spectrum = np.fft.rfft(demeaned, axis=1, norm="ortho")
            del demeaned
            if not np.isfinite(spectrum).all():
                raise ValueError("factor temporal transform is not finite")
            weights = np.full(len(temporal_frequencies), 2.0)
            weights[0] = 0.0
            if nt % 2 == 0:
                weights[-1] = 1.0
            flat = spatial.reshape(modes, pixels)
            check()
            spatial_gram = flat @ flat.T

            def temporal_direct(coefficients):
                power = 0.0
                for start in range(0, pixels, block_size):
                    check()
                    values = coefficients @ flat[:, start:start + block_size]
                    power += float(np.sum(values.real**2 + values.imag**2))
                return power

            temporal_power = np.zeros(len(temporal_frequencies) - 1, dtype=np.float64)
            for start in range(1, len(temporal_frequencies), block_size):
                check()
                stop = min(start + block_size, len(temporal_frequencies))
                temporal_power[start - 1:stop - 1] = weights[start:stop] * _powers(
                    spatial_gram, spectrum[:, start:stop], gram_sum_length=pixels,
                    direct=temporal_direct, check=check)
            del spatial_gram
            radial_power = np.zeros(len(radial_frequencies), dtype=np.float64)
            if in_band.any():
                check()
                band = spectrum[:, in_band] * np.sqrt(weights[in_band])[None, :]
                # Field frequency coefficients are band.T @ spatial_spectrum.
                # Consequently the quadratic Gram is conj(band) @ band.T.
                temporal_gram = band.conj() @ band.T

                def spatial_direct(coefficients):
                    power = 0.0
                    for start in range(0, band.shape[1], block_size):
                        check()
                        values = band[:, start:start + block_size].T @ coefficients
                        power += float(np.sum(values.real**2 + values.imag**2))
                    return power

                check()
                spatial_spectrum = np.fft.fft2(spatial, axes=(-2, -1), norm="ortho").reshape(modes, pixels)
                if not np.isfinite(spatial_spectrum).all():
                    raise ValueError("factor spatial transform is not finite")
                for start in range(0, pixels, block_size):
                    check()
                    stop = min(start + block_size, pixels)
                    power = _powers(temporal_gram, spatial_spectrum[:, start:stop],
                        gram_sum_length=band.shape[1], direct=spatial_direct, check=check)
                    radial_power += np.bincount(groups[start:stop], weights=power,
                                                minlength=len(radial_frequencies))
            check()
    except FloatingPointError as error:
        raise ValueError("factorized field power is not representable in float64") from error
    spatial_result = _power_cutoff(radial_frequencies, radial_power)
    if not in_band.any():
        spatial_result["status"] = "undefined_no_primary_band_bins"
    temporal_result = _power_cutoff(temporal_frequencies[1:], temporal_power)
    spatial_result["k90_cycles_per_micrometre"] = spatial_result.pop("cutoff")
    wavelength = _scale(spatial_result["k90_cycles_per_micrometre"])
    spatial_result["micrometres"] = wavelength
    spatial_result["pixels"] = wavelength / dx if dx == dy and wavelength is not None else None
    spatial_result["pixel_scale_status"] = "square_spacing" if dx == dy else "undefined_anisotropic_spacing"
    spatial_result["selected_temporal_bin_count"] = int(np.count_nonzero(in_band))
    temporal_result["f90_hz"] = temporal_result.pop("cutoff")
    temporal_result["seconds"] = _scale(temporal_result["f90_hz"])
    return {
        "estimator": ESTIMATOR,
        "lambda": spatial_result, "tau": temporal_result,
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
        "evaluation": {"method": "exact_factorized_Gram_with_direct_cancellation_fallback",
            "factor_modes": modes, "all_cross_terms_retained": True,
            "dense_time_by_space_materialized": False, "max_elements": max_elements},
    }
