"""Bounded direct-surface field primitives; no draw law, calibration or runner.

Positions are micrometres, times seconds, voltages microvolts, angles radians.
Every array is an explicit NumPy ndarray. ``max_elements`` bounds each input
and output array before a new product is allocated; work uses O(max_elements)
memory with a small constant, not a full unbounded spatial covariance. Callers
must also bound their supplied transform's workspace. No random draws occur.

The primary-band transform must be a fixed linear 1--100 Hz operator with
caller-selected boundaries and reference convention, common to every design.
It receives copies of the raw field and times. Scaling raw and transformed
fields together relies on that linearity; no filter is selected here.
See ../development_field_primitives.md for unresolved prospective choices.
"""

from dataclasses import dataclass
import math

import numpy as np


def _limit(max_elements, *sizes):
    if isinstance(max_elements, bool) or not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError("max_elements must be an explicit positive integer")
    if any(size > max_elements for size in sizes):
        raise ValueError("array exceeds max_elements; use smaller explicit blocks")


def _array(value, name, ndim, max_elements):
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an explicit NumPy array")
    _limit(max_elements, value.size)
    if value.ndim != ndim or 0 in value.shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} has invalid dimensions or real numeric dtype")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"invalid {name}")
    return result


def _points(value, name, max_elements):
    result = _array(value, name, 2, max_elements)
    if result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (points, 2)")
    return result


def _times(value, max_elements):
    result = _array(value, "times_s", 1, max_elements)
    if result.size < 2 or np.any(np.diff(result) <= 0):
        raise ValueError("times_s must be strictly increasing with at least two samples")
    return result


def _mask(value, shape, name, max_elements):
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be an explicit boolean NumPy array")
    _limit(max_elements, value.size)
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{name} must be boolean with the exact field shape")
    return value


def _axes(minor_width, axis_ratio, orientation_rad):
    minor = _scalar(minor_width, "minor_width", positive=True)
    ratio = _scalar(axis_ratio, "axis_ratio", positive=True)
    if ratio < 1:
        raise ValueError("axis_ratio is major/minor and must be at least one")
    angle = _scalar(orientation_rad, "orientation_rad")
    major = _scalar(minor * ratio, "major_width", positive=True)
    return minor, major, math.cos(angle), math.sin(angle)


def gaussian_covariance_block(left_points_um, right_points_um, *, ell_um,
                              axis_ratio, orientation_rad, max_elements):
    """Unit covariance block; ell is minor SD and orientation is major axis.

    Returns only the caller's left-by-right block. No sampling, jitter,
    Cholesky factor, spatial periodicity or boundary law is introduced.
    """
    left = _points(left_points_um, "left_points_um", max_elements)
    right = _points(right_points_um, "right_points_um", max_elements)
    _limit(max_elements, len(left) * len(right))
    minor, major, c, s = _axes(ell_um, axis_ratio, orientation_rad)
    dx = left[:, 0, None] - right[None, :, 0]
    dy = left[:, 1, None] - right[None, :, 1]
    covariance = np.exp(-0.5 * (((c * dx + s * dy) / major) ** 2
                                + ((-s * dx + c * dy) / minor) ** 2))
    return _array(covariance, "covariance block", 2, max_elements)


def gaussian_spatial_envelope(points_um, *, center_um, fwhm_um, axis_ratio,
                              orientation_rad, max_elements):
    """Peak-one envelope, with minor-axis pre-truncation Gaussian FWHM."""
    points = _points(points_um, "points_um", max_elements)
    center = _array(center_um, "center_um", 1, max_elements)
    if center.shape != (2,):
        raise ValueError("center_um must contain two coordinates")
    fwhm = _scalar(fwhm_um, "fwhm_um", positive=True)
    return gaussian_covariance_block(
        points, center[None, :], ell_um=fwhm / (2 * math.sqrt(2 * math.log(2))),
        axis_ratio=axis_ratio, orientation_rad=orientation_rad,
        max_elements=max_elements)[:, 0]


def combine_spatial_modes(spatial_modes, temporal_drivers_uv, *, max_elements):
    """Explicit N-by-C modes times C-by-T drivers, retaining interference.

    Spatial modes are dimensionless and supplied by the caller; this function
    does not claim that arbitrary modes realize the Gaussian covariance.
    """
    modes = _array(spatial_modes, "spatial_modes", 2, max_elements)
    drivers = _array(temporal_drivers_uv, "temporal_drivers_uv", 2, max_elements)
    if modes.shape[1] != drivers.shape[0]:
        raise ValueError("mode and temporal-driver component counts differ")
    _limit(max_elements, modes.shape[0] * drivers.shape[1])
    return _array(modes @ drivers, "combined field", 2, max_elements)


def focal_event(points_um, times_s, *, center_um, onset_s, spatial_fwhm_um,
                axis_ratio, orientation_rad, temporal_fwhm_s,
                biphasic_driver_uv, outer_taper, max_elements):
    """One event with exact finite-support raw onset; sum events to keep overlap.

    ``onset_s`` is support START, not envelope centre. The centre is onset+3 SD;
    support ends at onset+6 SD. Caller driver and taper are sampled on times_s.
    Taper is exactly one inside 2.5 SD, zero outside/on 3 SD, and in [0,1]
    between. Its continuous smoothness and waveform are caller responsibilities.
    Driver must have both signs within sampled effective support. The return is
    raw truth, not post-filter amplitude normalization or achieved waveform width.
    """
    points = _points(points_um, "points_um", max_elements)
    times = _times(times_s, max_elements)
    _limit(max_elements, len(points) * len(times))
    onset = _scalar(onset_s, "onset_s")
    fwhm = _scalar(temporal_fwhm_s, "temporal_fwhm_s", positive=True)
    driver = _array(biphasic_driver_uv, "biphasic_driver_uv", 1, max_elements)
    taper = _array(outer_taper, "outer_taper", 1, max_elements)
    if driver.shape != times.shape or taper.shape != times.shape:
        raise ValueError("driver and taper must match times_s")
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    z = (times - onset) / sigma - 3
    absolute = np.abs(z)
    if (np.any((taper < 0) | (taper > 1))
            or np.any(taper[absolute <= 2.5] != 1)
            or np.any(taper[absolute >= 3] != 0)):
        raise ValueError("taper must retain the core and vanish outside finite support")
    active = (absolute < 3) & (taper > 0)
    if not np.any(driver[active] > 0) or not np.any(driver[active] < 0):
        raise ValueError("sampled active driver must contain both biphasic lobes")
    temporal = np.zeros(times.shape, dtype=float)
    temporal[active] = driver[active] * taper[active] * np.exp(-0.5 * z[active] ** 2)
    spatial = gaussian_spatial_envelope(
        points, center_um=center_um, fwhm_um=spatial_fwhm_um,
        axis_ratio=axis_ratio, orientation_rad=orientation_rad,
        max_elements=max_elements)
    return _array(spatial[:, None] * temporal[None, :], "focal field", 2, max_elements)


def planar_phase(points_um, times_s, *, wavelength_um, frequency_hz,
                 direction_rad, phase_rad, origin_um, reference_time_s,
                 max_elements):
    """Unwrapped radians, 2*pi*(projected_distance/wavelength - f*dt)+phase.

    Positive frequency translates in the declared direction at f*wavelength.
    Amplitude, envelope, onset and boundary handling remain explicit downstream.
    """
    points = _points(points_um, "points_um", max_elements)
    times = _times(times_s, max_elements)
    origin = _array(origin_um, "origin_um", 1, max_elements)
    if origin.shape != (2,):
        raise ValueError("origin_um must contain two coordinates")
    _limit(max_elements, len(points) * len(times))
    wavelength = _scalar(wavelength_um, "wavelength_um", positive=True)
    frequency = _scalar(frequency_hz, "frequency_hz", positive=True)
    direction = _scalar(direction_rad, "direction_rad")
    phase = _scalar(phase_rad, "phase_rad")
    reference = _scalar(reference_time_s, "reference_time_s")
    delta = points - origin
    distance = delta[:, 0] * math.cos(direction) + delta[:, 1] * math.sin(direction)
    result = 2 * math.pi * (distance[:, None] / wavelength
                            - frequency * (times[None, :] - reference)) + phase
    return _array(result, "planar phase", 2, max_elements)


def expanding_phase(points_um, times_s, *, center_um, wavelength_um,
                    frequency_hz, reference_radius_um, reference_time_s,
                    phase_rad, max_elements):
    """Unwrapped outward radial phase; reference radius at explicit time.

    This is a phase field, not a singular radial Green function or an adopted
    central-amplitude law. Caller supplies any envelope and onset separately.
    """
    points = _points(points_um, "points_um", max_elements)
    times = _times(times_s, max_elements)
    center = _array(center_um, "center_um", 1, max_elements)
    if center.shape != (2,):
        raise ValueError("center_um must contain two coordinates")
    _limit(max_elements, len(points) * len(times))
    wavelength = _scalar(wavelength_um, "wavelength_um", positive=True)
    frequency = _scalar(frequency_hz, "frequency_hz", positive=True)
    radius = _scalar(reference_radius_um, "reference_radius_um", nonnegative=True)
    reference = _scalar(reference_time_s, "reference_time_s")
    phase = _scalar(phase_rad, "phase_rad")
    delta = points - center
    distance = np.hypot(delta[:, 0], delta[:, 1])
    result = 2 * math.pi * ((distance[:, None] - radius) / wavelength
                            - frequency * (times[None, :] - reference)) + phase
    return _array(result, "expanding phase", 2, max_elements)


def _band(field_uv, times_s, common_mask, primary_band_transform, max_elements):
    raw = _array(field_uv, "field_uv", 2, max_elements)
    times = _times(times_s, max_elements)
    if raw.shape[1] != len(times):
        raise ValueError("field time axis differs from times_s")
    mask = _mask(common_mask, raw.shape, "common_mask", max_elements)
    if not np.any(mask):
        raise ValueError("common_mask must contain target support")
    if not callable(primary_band_transform):
        raise ValueError("supply an explicit fixed linear primary-band transform")
    band = _array(primary_band_transform(raw.copy(), times.copy()),
                  "primary-band field", 2, max_elements)
    if band.shape != raw.shape:
        raise ValueError("primary-band transform must preserve the field shape")
    return raw, band, mask


def _power(values):
    return _scalar(np.mean(values ** 2), "mean-square power", nonnegative=True)


@dataclass(frozen=True)
class FieldResult:
    raw_uv: np.ndarray
    primary_band_uv: np.ndarray
    report: dict


def normalize_primary_band(field_uv, times_s, *, common_mask,
                           primary_band_transform, target_uv, measure,
                           max_elements):
    """Scale using supplied common-mask primary-band RMS or global range.

    ``peak_to_peak`` means max minus min across ALL selected space-time samples,
    not a silently chosen per-location event statistic. Its use for a selected
    condition requires that estimand to be prospectively accepted. Raw finite
    support is preserved, while any transform pre-onset ringing is retained.
    """
    raw, band, mask = _band(field_uv, times_s, common_mask,
                            primary_band_transform, max_elements)
    target = _scalar(target_uv, "target_uv", positive=True)
    selected = band[mask]
    if measure == "rms":
        before = math.sqrt(_power(selected))
    elif measure == "peak_to_peak":
        before = _scalar(np.max(selected) - np.min(selected), "peak-to-peak", nonnegative=True)
    else:
        raise ValueError("measure must be rms or peak_to_peak")
    if before == 0:
        raise ValueError("normalization is undefined for zero eligible amplitude")
    scale = _scalar(target / before, "normalization scale", positive=True)
    normalized = _array(raw * scale, "normalized raw field", 2, max_elements)
    normalized_band = _array(band * scale, "normalized band field", 2, max_elements)
    return FieldResult(normalized, normalized_band,
                       {"measure": measure, "input_amplitude_uv": before,
                        "target_amplitude_uv": target, "scale": scale,
                        "common_mask_samples": int(np.count_nonzero(mask))})


def mix_components(broad_uv, local_uv, times_s, *, common_mask,
                   primary_band_transform, local_fraction, component_rms_uv,
                   max_elements):
    """Set P_local/(P_broad+P_local), then sum without renormalizing cancellation.

    component_rms_uv**2 is the sum of the two component mean-square powers,
    BEFORE cross terms. Centered covariance and uncentered cross moment are
    both reported: total power = P_broad + P_local + 2*cross_moment.
    Endpoint fractions permit an inactive zero-power component, but an active
    zero-power component is undefined and rejected without epsilon or redraw.
    """
    fraction = _scalar(local_fraction, "local_fraction", nonnegative=True)
    if fraction > 1:
        raise ValueError("local_fraction must be at most one")
    rms = _scalar(component_rms_uv, "component_rms_uv", positive=True)
    broad, broad_band, mask = _band(broad_uv, times_s, common_mask,
                                    primary_band_transform, max_elements)
    local, local_band, _ = _band(local_uv, times_s, common_mask,
                                  primary_band_transform, max_elements)
    if local.shape != broad.shape:
        raise ValueError("components must have the same shape")
    target_power = _scalar(rms * rms, "component power", positive=True)
    scales = []
    for band, weight in ((broad_band, 1 - fraction), (local_band, fraction)):
        power = _power(band[mask])
        if weight and power == 0:
            raise ValueError("active component has zero eligible primary-band power")
        scale = math.sqrt(target_power * weight / power) if weight else 0.0
        scales.append(_scalar(scale, "component scale", nonnegative=True))
    b = broad_band * scales[0]
    l = local_band * scales[1]
    selected_b, selected_l = b[mask], l[mask]
    cross = _scalar(np.mean(selected_b * selected_l), "cross moment")
    covariance = _scalar(np.mean((selected_b - np.mean(selected_b))
                                 * (selected_l - np.mean(selected_l))), "covariance")
    summed = _array(broad * scales[0] + local * scales[1], "mixed raw field", 2, max_elements)
    summed_band = _array(b + l, "mixed band field", 2, max_elements)
    pb, pl = _power(selected_b), _power(selected_l)
    return FieldResult(summed, summed_band,
                       {"broad_power_uv2": pb, "local_power_uv2": pl,
                        "component_power_fraction": pl / (pb + pl),
                        "cross_moment_uv2": cross, "covariance_uv2": covariance,
                        "cross_term_uv2": 2 * cross,
                        "summed_power_uv2": _power(summed_band[mask]),
                        "broad_scale": scales[0], "local_scale": scales[1],
                        "common_mask_samples": int(np.count_nonzero(mask))})


@dataclass(frozen=True)
class ObservationResult:
    truth_uv: np.ndarray
    additive_nuisance_uv: np.ndarray
    observed_uv: np.ndarray
    observation_mask: np.ndarray
    evaluation_mask: np.ndarray


def add_nuisance_and_missingness(truth_uv, additive_nuisance_uv, *,
                                 observation_mask, evaluation_mask, max_elements):
    """Explicit additive observation nuisance and missingness; truth unchanged.

    Missing observations are NaN, including dead rows. Evaluation support is
    copied exactly and is never intersected with observation availability.
    This helper does not model contact drift, saturation or device transfer.
    """
    truth = _array(truth_uv, "truth_uv", 2, max_elements)
    nuisance = _array(additive_nuisance_uv, "additive_nuisance_uv", 2, max_elements)
    if nuisance.shape != truth.shape:
        raise ValueError("nuisance must have the exact truth shape")
    observed_mask = _mask(observation_mask, truth.shape, "observation_mask", max_elements)
    target_mask = _mask(evaluation_mask, truth.shape, "evaluation_mask", max_elements)
    if not np.any(target_mask):
        raise ValueError("evaluation_mask must contain target support")
    contaminated = _array(truth + nuisance, "contaminated observation", 2, max_elements)
    return ObservationResult(truth.copy(), nuisance.copy(),
                             np.where(observed_mask, contaminated, np.nan),
                             observed_mask.copy(), target_mask.copy())
