"""Selected F1--F4/S1--S4 direct-surface inputs, with finite raw support.

No hardware simulation, calibration, scoring or held-out access. Return exact
separable factors; primary-band evaluation ringing never changes raw support.
The caller supplies the fixed linear primary-band transform and owns compute,
world publication, complete-panel accounting and later whole-group data roles.
"""
from __future__ import annotations

import math

import numpy as np

from development_fields import focal_event, gaussian_spatial_envelope


FOCAL = {
    "F1": (2400., 0.300, 200., 0.5),
    "F2": (600., 0.150, 2000., 3.0),
    "F3": (150., 0.020, 200., 0.5),
    "F4": (600., 0.050, 1000., 3.0),
}
SURPRISE = {"S1": ("F2", 0.5), "S2": ("F2", 2.0),
            "S3": ("F3", 0.5), "S4": ("F4", 2.0)}
STREAMS = {"paired_uniform_locations": 100, "paired_onset_phase": 101,
           "edge_locations": 102, "four_fixed_locations": 103,
           "four_onset_phases": 104}
APERTURE_UM = 9600.0
BORDER_UM = 600.0


def _rng(seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([20, seed, stream])))


def _vector(value, name, max_elements):
    if (not isinstance(value, np.ndarray) or value.ndim != 1 or len(value) < 2
            or value.dtype.kind not in "fiu" or value.size > max_elements
            or not np.isfinite(value).all() or np.any(np.diff(value) <= 0)):
        raise ValueError(f"{name} must be a bounded increasing finite real array")
    result = np.asarray(value, dtype=np.float64)
    step = result[1] - result[0]
    if not np.allclose(np.diff(result), step, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be uniformly sampled")
    return result, float(step)


def _starts(times, prelude, offset, rate, support):
    first = prelude + offset
    count = max(0, math.floor((float(times[-1]) - support - first) * rate) + 1)
    starts = [first + index / rate for index in range(count)]
    # This uses time support only; no event magnitude or outcome is inspected.
    return [start for start in starts if start >= times[0] and start + support <= times[-1]]


def _border_centres(rng, count):
    # Two full-width horizontal strips and two nonoverlapping interior strips.
    rectangles = np.array([
        [0., APERTURE_UM, 0., BORDER_UM],
        [0., APERTURE_UM, APERTURE_UM - BORDER_UM, APERTURE_UM],
        [0., BORDER_UM, BORDER_UM, APERTURE_UM - BORDER_UM],
        [APERTURE_UM - BORDER_UM, APERTURE_UM, BORDER_UM, APERTURE_UM - BORDER_UM],
    ])
    areas = (rectangles[:, 1] - rectangles[:, 0]) * (rectangles[:, 3] - rectangles[:, 2])
    draws = rng.random((count, 3))
    cumulative = np.cumsum(areas / areas.sum())
    cumulative[-1] = 1.0
    labels = np.searchsorted(cumulative, draws[:, 0], side="right")
    selected = rectangles[labels]
    return np.column_stack((selected[:, 0] + draws[:, 1] * (selected[:, 1] - selected[:, 0]),
                            selected[:, 2] + draws[:, 2] * (selected[:, 3] - selected[:, 2])))


def _half_height_span(driver, times):
    absolute = np.abs(driver)
    peak = float(absolute.max())
    if peak == 0:
        raise ValueError("event half-height span is undefined for a zero waveform")
    eligible = np.flatnonzero(absolute >= 0.5 * peak)
    return float(times[eligible[-1]] - times[eligible[0]])


def build_focal_world(condition_id, *, x_um, y_um, times_s, replicate_seed,
                      primary_band_transform, check, max_elements):
    """Return (spatial[R,Y,X], raw_drivers[R,T], metadata).

The transform accepts a [driver,time] array and returns the same shape. The
production caller supplies the selected full-record 1--100-Hz DFT projection.
Smaller regular grids are permitted for analytic fixtures, but they must span
the same 9600-micrometre rectangular aperture. The production recipe fixes
25-micrometre spatial spacing and a 500-Hz,16-second time record.
"""
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be an explicit positive integer")
    if type(replicate_seed) is not int or replicate_seed < 0:
        raise ValueError("replicate_seed must be an explicit nonnegative integer")
    if not callable(check) or not callable(primary_band_transform):
        raise ValueError("explicit budget check and fixed primary-band transform required")
    check()
    x, dx = _vector(x_um, "x_um", max_elements)
    y, dy = _vector(y_um, "y_um", max_elements)
    times, dt = _vector(times_s, "times_s", max_elements)
    if not (math.isclose(x[0], dx / 2, rel_tol=0, abs_tol=1e-9)
            and math.isclose(y[0], dy / 2, rel_tol=0, abs_tol=1e-9)
            and math.isclose(x[-1] + dx / 2, APERTURE_UM, rel_tol=0, abs_tol=1e-9)
            and math.isclose(y[-1] + dy / 2, APERTURE_UM, rel_tol=0, abs_tol=1e-9)
            and times[0] == 0):
        raise ValueError("explicit centre-sampled9600um aperture and time origin zero required")
    if condition_id in FOCAL:
        focal_id, prelude = condition_id, 0.0
    elif condition_id in SURPRISE:
        focal_id, prelude = SURPRISE[condition_id]
    else:
        raise ValueError("condition_id must be one of the selected F1--F4/S1--S4")
    spatial_fwhm, temporal_fwhm, target_pp, rate = FOCAL[focal_id]
    sigma = temporal_fwhm / (2 * math.sqrt(2 * math.log(2)))
    support = 6 * sigma
    if condition_id == "S4":
        centres = _rng(replicate_seed, STREAMS["four_fixed_locations"]).uniform(
            0, APERTURE_UM, size=(4, 2))
        offsets = _rng(replicate_seed, STREAMS["four_onset_phases"]).random(4) / rate
        grouped_starts = [_starts(times, prelude, offset, rate, support) for offset in offsets]
        if any(not starts for starts in grouped_starts):
            raise ValueError("finite record has no complete event for a selected location")
    else:
        offset = float(_rng(replicate_seed, STREAMS["paired_onset_phase"]).random() / rate)
        starts = _starts(times, prelude, offset, rate, support)
        if not starts:
            raise ValueError("finite record has no complete selected event")
        if condition_id == "S2":
            centres = _border_centres(_rng(replicate_seed, STREAMS["edge_locations"]), len(starts))
        else:
            centres = _rng(replicate_seed, STREAMS["paired_uniform_locations"]).uniform(
                0, APERTURE_UM, size=(len(starts), 2))
        grouped_starts = [[start] for start in starts]
        offsets = np.array([offset])
    modes, pixels, nt = len(centres), len(x) * len(y), len(times)
    if max(2 * pixels, modes * pixels, modes * nt) > max_elements:
        raise ValueError("focal factor arrays exceed max_elements")
    check()
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    del xx, yy
    spatial = np.empty((modes, len(y), len(x)), dtype=np.float64)
    drivers = np.zeros((modes, nt), dtype=np.float64)
    event_records = []
    for mode, (centre, starts) in enumerate(zip(centres, grouped_starts)):
        check()
        envelope = gaussian_spatial_envelope(points, center_um=centre,
            fwhm_um=spatial_fwhm, axis_ratio=1., orientation_rad=0., max_elements=max_elements)
        spatial[mode] = envelope.reshape(len(y), len(x))
        spatial_peak = float(envelope.max())
        if spatial_peak == 0:
            raise ValueError("selected event is unresolved on the supplied target grid")
        for event_index, start in enumerate(starts):
            check()
            z = (times - start) / sigma - 3
            absolute = np.abs(z)
            taper = np.zeros(nt)
            taper[absolute <= 2.5] = 1.
            outer = (absolute > 2.5) & (absolute < 3.)
            u = 2 * (absolute[outer] - 2.5)
            # Factored quintic is nonnegative; clip only its roundoff to its
            # analytic [0,1] range. Its value and first two derivatives join.
            taper[outer] = np.clip((1 - u)**3 * (1 + 3 * u + 6 * u**2), 0., 1.)
            raw = focal_event(centre[None, :], times, center_um=centre,
                onset_s=start, spatial_fwhm_um=spatial_fwhm, axis_ratio=1., orientation_rad=0.,
                temporal_fwhm_s=temporal_fwhm, biphasic_driver_uv=-z,
                outer_taper=taper, max_elements=max_elements)[0]
            check()
            transformed = primary_band_transform(raw[None, :].copy())
            if (not isinstance(transformed, np.ndarray) or transformed.shape != (1, nt)
                    or transformed.dtype.kind not in "fiu" or not np.isfinite(transformed).all()):
                raise ValueError("primary-band transform must preserve finite real driver shape")
            band = np.asarray(transformed[0], dtype=np.float64)
            if not (float(band.min()) < 0 < float(band.max())):
                raise ValueError("isolated primary-band event must retain both polarities")
            before_pp = spatial_peak * float(band.max() - band.min())
            scale = target_pp / before_pp
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("isolated event normalization is not finite")
            scaled = raw * scale
            if not np.isfinite(scaled).all():
                raise ValueError("scaled event is not finite")
            drivers[mode] += scaled
            if not np.isfinite(drivers[mode]).all():
                raise ValueError("summed event driver is not finite")
            event_records.append({
                "mode_index": mode, "event_index_at_location": event_index,
                "centre_um_xy": centre.tolist(), "support_start_s": float(start),
                "support_end_s": float(start + support), "nominal_envelope_fwhm_s": temporal_fwhm,
                "raw_half_height_span_s": _half_height_span(raw, times),
                "primary_band_half_height_span_s": _half_height_span(band, times),
                "sampled_spatial_peak": spatial_peak, "applied_scale": scale,
                "primary_band_peak_to_peak_uv_before_scale": before_pp,
                "primary_band_peak_to_peak_uv": before_pp * scale,
                "raw_peak_to_peak_uv": spatial_peak * float(raw.max() - raw.min()) * scale,
                "raw_prelude_exactly_zero": bool(np.all(raw[times < start] == 0)),
            })
    event_records.sort(key=lambda event: (event["support_start_s"], event["mode_index"]))
    metadata = {
        "condition_id": condition_id, "focal_template": focal_id, "replicate_seed": replicate_seed,
        "generator": "ep20_selected_focal_surprise_factors_v1",
        "factor_axis_order": {"spatial": ["mode", "y", "x"], "raw_drivers": ["mode", "time"]},
        "grid_spacing_um_xy": [dx, dy], "dt_seconds": dt,
        "nominal_spatial_fwhm_um": spatial_fwhm, "nominal_temporal_envelope_fwhm_s": temporal_fwhm,
        "nominal_rate_hz_per_location_train": rate, "minimum_local_prelude_s": prelude,
        "onset_offsets_s": offsets.tolist(), "complete_event_count": len(event_records),
        "realized_count_per_second": len(event_records) / (nt * dt),
        "factor_modes": modes, "rng": "PCG64_SeedSequence_20_replicate_stream",
        "stream_ids": dict(STREAMS),
        "shared_innovation_group": f"ep20_focal_surprise_seed_{replicate_seed}",
        "split_linked_condition_worlds_together": True,
        "event_amplitude_estimand": "isolated_event_global_sampled_space_time_primary_band_peak_to_peak",
        "target_event_peak_to_peak_uv": target_pp, "sum_renormalized": False,
        "raw_support_preserved": True, "primary_band_ringing_available_to_scheduler": False,
        "S4_identical_spatial_modes_merged_by_driver_sum": condition_id == "S4",
        "events": event_records, "evaluation_mask": "unchanged_full_target_rectangle",
    }
    check()
    return spatial, drivers, metadata
