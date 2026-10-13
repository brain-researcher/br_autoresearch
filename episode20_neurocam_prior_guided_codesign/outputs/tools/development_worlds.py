"""Prospective EP20 direct-surface development panel, never a calibrated device.

Production fields use the fixed recipe grid; no empirical data, candidate score,
DAQ inference, final-world role or automatic job successor enters this module.
Execute only in the caller's coordinated Slurm allocation. See the recipe for
finite-grid limitations, exact condition meanings and paired-role constraints.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from development_fields import gaussian_spatial_envelope, planar_phase
from development_focal_worlds import build_focal_world
from factorized_field_power_scales import factorized_field_power_scales

CONDITIONS = tuple(f"{family}{index}" for family in "BFWMISN" for index in range(1, 5))
REPLICATE_SEEDS = (2001, 2002, 2003)
MAX_ELEMENTS = 20_000_000
RECIPE = "ep20_direct_surface_development_2026_10_09_v1"
BROAD = {"B1": (4800., 3., 50., 1.), "B2": (1200., 11., 50., 1.),
         "B3": (300., 37., 50., 1.), "B4": (600., 80., 10., 4.)}
WAVES = {"W1": (7., 400., 0.), "W2": (11., 100., 90.),
         "W3": (37., 100., 0.), "W4": (37., 100., 90.)}
STREAMS = {"broad_quadratures": 1, "wave_phase": 10,
           "mixture_broad": 20, "mixture_local": 21, "M4_common_maps": 22,
           "M4_envelope_and_carrier_phase": 23, "interaction_geometry": 30,
           "interaction_phases": 31, "I4_geometry": 32, "I4_phases": 33,
           "N1_line_phase": 40, "N2_line_phase": 41, "N3_missingness": 42,
           "N4_dead_row": 43}


@dataclass(frozen=True)
class _Grid:
    """Only private analytic tests change nx/ny; public generation is fixed."""
    nx: int = 384
    ny: int = 384
    nt: int = 8000
    dt: float = 0.002

    @property
    def x(self):
        return (np.arange(self.nx) + 0.5) * (9600. / self.nx)

    @property
    def y(self):
        return (np.arange(self.ny) + 0.5) * (9600. / self.ny)

    @property
    def times(self):
        return np.arange(self.nt) * self.dt


@dataclass
class DevelopmentWorld:
    spatial_modes: np.ndarray
    raw_drivers: np.ndarray
    primary_band_drivers: np.ndarray
    metadata: dict
    nuisance_arrays: dict


class GenerationDeadline(RuntimeError):
    pass


class UnresolvedNormalization(ValueError):
    pass


def _limit(max_elements, *sizes):
    if type(max_elements) is not int or max_elements < 1 or any(n > max_elements for n in sizes):
        raise ValueError("explicit per-array max_elements exceeded")


def _rng(seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([20, seed, stream])))


def _project(drivers, dt, low=1., high=100., *, check, max_elements):
    _limit(max_elements, drivers.size)
    check()
    spectrum = np.fft.rfft(drivers, axis=-1, norm="ortho")
    frequency = np.fft.rfftfreq(drivers.shape[-1], d=dt)
    spectrum[..., (frequency < low) | (frequency > high)] = 0
    result = np.fft.irfft(spectrum, n=drivers.shape[-1], axis=-1, norm="ortho")
    if not np.isfinite(result).all():
        raise ValueError("nonfinite transformed driver")
    return result


def _moments(spatial, drivers, *, check, max_elements):
    """Exact full-mask second moment and mean, retaining every cross term."""
    modes, ny, nx = spatial.shape
    _limit(max_elements, spatial.size, drivers.size, modes * modes)
    check()
    flat = spatial.reshape(modes, ny * nx)
    gram = flat @ flat.T / flat.shape[1]
    temporal = drivers @ drivers.T / drivers.shape[1]
    power = float(np.sum(gram * temporal))
    absolute = float(np.sum(np.abs(gram * temporal)))
    # Reconstruct bounded blocks when cancellation makes the Gram result
    # unreliable; no clipping, epsilon floor or change to the target sum.
    if power <= 64 * np.finfo(float).eps * (flat.shape[1] + drivers.shape[1] + modes) * absolute:
        power = 0.
        time_block = min(128, drivers.shape[1], max_elements)
        space_block = max(1, min(1024, max_elements // time_block))
        for start in range(0, flat.shape[1], space_block):
            for first in range(0, drivers.shape[1], time_block):
                check()
                values = flat[:, start:start + space_block].T @ drivers[:, first:first + time_block]
                power += float(np.sum(values * values))
        power /= flat.shape[1] * drivers.shape[1]
    mean = float(np.dot(flat.mean(axis=1), drivers.mean(axis=1)))
    if not math.isfinite(power) or power < 0 or not math.isfinite(mean):
        raise ValueError("field moments are not representable")
    return power, mean


def _normalize(spatial, raw, target_rms, grid, *, check, max_elements):
    band = _project(raw, grid.dt, check=check, max_elements=max_elements)
    power, _ = _moments(spatial, band, check=check, max_elements=max_elements)
    if power <= 0:
        raise UnresolvedNormalization("selected realization has zero primary-band power")
    scale = target_rms / math.sqrt(power)
    raw = raw * scale
    if not np.isfinite(raw).all():
        raise UnresolvedNormalization("normalization is not finite")
    return raw, {"target_primary_band_rms_uv": target_rms,
                 "unnormalized_primary_band_power_uv2": power, "applied_scale": scale}


def _axis_root(coordinates, ell, *, check, max_elements):
    _limit(max_elements, len(coordinates)**2)
    covariance = np.exp(-0.5 * ((coordinates[:, None] - coordinates[None, :]) / ell)**2)
    check()
    values, vectors = np.linalg.eigh(covariance)
    tolerance = 128 * np.finfo(float).eps * float(values[-1])
    if values[0] < -tolerance:
        raise ValueError("Gaussian covariance has more than roundoff-negative eigenvalue")
    negative = values < 0
    root = (vectors * np.sqrt(np.maximum(values, 0.))[None, :]) @ vectors.T
    check()
    error = float(np.max(np.abs(root @ root.T - covariance)))
    return root, {"ell_um": ell, "roundoff_clipped_count": int(negative.sum()),
                  "maximum_negative_magnitude": max(0., -float(values[0])),
                  "negative_eigenvalue_tolerance": tolerance,
                  "max_absolute_covariance_reconstruction_error": error}


def _quadrature_maps(grid, ell, ratio, seed, stream, *, check, max_elements):
    _limit(max_elements, 2 * grid.nx * grid.ny)
    # For this registered panel the anisotropic major axis is exactly y.
    lx, report_x = _axis_root(grid.x, ell, check=check, max_elements=max_elements)
    ly, report_y = _axis_root(grid.y, ell * ratio, check=check, max_elements=max_elements)
    random = _rng(seed, stream)
    maps = np.empty((2, grid.ny, grid.nx))
    for index in range(2):
        check()
        innovation = random.standard_normal((grid.ny, grid.nx))
        maps[index] = ly @ innovation @ lx.T
    return maps, {"x_covariance": report_x, "y_covariance": report_y,
                  "innovation_stream": stream, "axis_ratio": ratio,
                  "major_axis": "y", "pre_normalization_Gaussian_draw": True}


def _carrier(times, frequency, phase=0.):
    angle = 2 * np.pi * frequency * times + phase
    return np.stack((np.cos(angle), np.sin(angle)))


def _broad(condition, seed, grid, *, check, max_elements):
    ell, frequency, target, ratio = BROAD[condition]
    spatial, meta = _quadrature_maps(grid, ell, ratio, seed, STREAMS["broad_quadratures"],
                                    check=check, max_elements=max_elements)
    raw, normalization = _normalize(spatial, _carrier(grid.times, frequency), target, grid,
                                    check=check, max_elements=max_elements)
    meta.update(normalization)
    meta.update(nominal_ell_um=ell, carrier_hz=frequency,
                normalized_ensemble_claim="fixed_realization_RMS_not_exact_Gaussian_covariance")
    return spatial, raw, meta


def _wave(condition, seed, grid, *, check, max_elements):
    frequency, speed, direction = WAVES[condition]
    wavelength = speed * 1000. / frequency
    phase = float(_rng(seed, STREAMS["wave_phase"]).uniform(0, 2 * np.pi))
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    _limit(max_elements, points.size)
    check()
    # Two reference times satisfy the existing primitive's time-axis contract;
    # its first column is the exact spatial phase at reference time zero.
    spatial_phase = planar_phase(points, grid.times[:2], wavelength_um=wavelength,
        frequency_hz=frequency, direction_rad=math.radians(direction), phase_rad=phase,
        origin_um=np.array([4800., 4800.]), reference_time_s=0., max_elements=max_elements)[:, 0]
    spatial = np.stack((np.cos(spatial_phase), np.sin(spatial_phase))).reshape(2, grid.ny, grid.nx)
    # cos(spatial_phase - omega*t) = cos* cos + sin* sin.
    raw, normalization = _normalize(spatial, _carrier(grid.times, frequency), 50., grid,
                                    check=check, max_elements=max_elements)
    return spatial, raw, dict(normalization, carrier_hz=frequency, speed_mm_per_s=speed,
                              direction_degrees=direction, wavelength_um=wavelength,
                              phase_rad=phase, phase_stream=STREAMS["wave_phase"])


def _cross_report(spatial_a, drivers_a, spatial_b, drivers_b, *, check, max_elements):
    check()
    a = spatial_a.reshape(len(spatial_a), -1)
    b = spatial_b.reshape(len(spatial_b), -1)
    _limit(max_elements, a.size, b.size, drivers_a.size, drivers_b.size)
    cross = float(np.sum((a @ b.T / a.shape[1]) * (drivers_a @ drivers_b.T / drivers_a.shape[1])))
    pa, ma = _moments(spatial_a, drivers_a, check=check, max_elements=max_elements)
    pb, mb = _moments(spatial_b, drivers_b, check=check, max_elements=max_elements)
    return {"broad_power_uv2": pa, "local_power_uv2": pb,
            "local_component_power_fraction": pb / (pa + pb),
            "uncentred_cross_moment_uv2": cross, "centred_covariance_uv2": cross - ma * mb,
            "sum_power_uv2": pa + pb + 2 * cross}


def _switch_normalize(spatial, unswitched, targets, grid, *, check, max_elements):
    """Exact post-projection interval powers, with analytic log-scale Jacobian."""
    labels = np.floor(grid.times).astype(int)
    intervals = len(targets)
    if set(labels.tolist()) != set(range(intervals)):
        raise ValueError("M4 requires all selected one-second intervals")
    rank, nt = unswitched.shape
    _limit(max_elements, intervals * rank * nt, intervals**3)
    basis = np.zeros((intervals, rank, nt))
    for interval in range(intervals):
        basis[interval, :, labels == interval] = unswitched[:, labels == interval].T
    band_basis = _project(basis.reshape(intervals * rank, nt), grid.dt,
                          check=check, max_elements=max_elements).reshape(intervals, rank, nt)
    flat = spatial.reshape(rank, -1)
    gram = flat @ flat.T / flat.shape[1]
    quadratics = np.empty((intervals, intervals, intervals))
    initial_power = np.empty(intervals)
    unswitched_band = _project(unswitched, grid.dt, check=check, max_elements=max_elements)
    for interval in range(intervals):
        check()
        selection = labels == interval
        block = band_basis[:, :, selection]
        quadratics[interval] = np.einsum("kat,ab,lbt->kl", block, gram, block,
                                        optimize=True) / selection.sum()
        initial_power[interval] = _moments(spatial, unswitched_band[:, selection],
                                           check=check, max_elements=max_elements)[0]
    if np.any(initial_power <= 0) or not np.isfinite(initial_power).all():
        raise UnresolvedNormalization("M4 unswitched interval has nonpositive power")
    initial = np.sqrt(targets / initial_power)

    def evaluate(log_multiplier):
        check()
        weights = initial * np.exp(log_multiplier)
        product = quadratics @ weights
        powers = product @ weights
        if np.any(powers <= 0) or not np.isfinite(powers).all():
            raise UnresolvedNormalization("M4 quadratic powers are nonpositive/nonfinite")
        return weights, product, powers

    def objective(log_multiplier):
        _, _, powers = evaluate(log_multiplier)
        return np.log(powers / targets)

    def jacobian(log_multiplier):
        weights, product, powers = evaluate(log_multiplier)
        return 2 * product * weights[None, :] / powers[:, None]

    result = least_squares(objective, np.zeros(intervals), jac=jacobian,
        bounds=(math.log(1e-6), math.log(1e6)), method="trf", ftol=1e-12, xtol=1e-12,
        gtol=1e-12, x_scale="jac", loss="linear", max_nfev=200)
    weights, _, powers = evaluate(result.x)
    raw = unswitched * weights[labels][None, :]
    band = _project(raw, grid.dt, check=check, max_elements=max_elements)
    actual = np.array([_moments(spatial, band[:, labels == j], check=check,
                               max_elements=max_elements)[0] for j in range(intervals)])
    relative_error = np.abs(actual / targets - 1)
    if not result.success or not np.isfinite(relative_error).all() or np.any(relative_error > 1e-6):
        raise UnresolvedNormalization("M4 selected bounded normalization unresolved; retain seed, no redraw")
    return raw, {"interval_amplitudes": weights.tolist(), "target_interval_powers_uv2": targets.tolist(),
                 "achieved_interval_powers_uv2": actual.tolist(), "relative_errors": relative_error.tolist(),
                 "quadratic_predicted_powers_uv2": powers.tolist(), "solver_success": bool(result.success),
                 "solver_status": int(result.status), "solver_message": str(result.message),
                 "function_evaluations": int(result.nfev), "log_multiplier_bounds": [math.log(1e-6), math.log(1e6)],
                 "max_nfev": 200, "ftol_xtol_gtol": 1e-12, "accepted_relative_error": 1e-6}


def _mixture(condition, seed, grid, *, check, max_elements):
    changing = condition == "M4"
    broad_stream = STREAMS["M4_common_maps"] if changing else STREAMS["mixture_broad"]
    local_stream = broad_stream if changing else STREAMS["mixture_local"]
    broad, broad_meta = _quadrature_maps(grid, 2400., 1., seed, broad_stream,
                                        check=check, max_elements=max_elements)
    local, local_meta = _quadrature_maps(grid, 300., 1., seed, local_stream,
                                        check=check, max_elements=max_elements)
    meta = {"broad": broad_meta, "local": local_meta, "sum_renormalized": False}
    if changing:
        phases = _rng(seed, STREAMS["M4_envelope_and_carrier_phase"]).uniform(0, 2 * np.pi, 2)
        envelope = 1 + 0.25 * np.cos(2 * np.pi * .5 * grid.times + phases[0])
        common = envelope[None, :] * (_carrier(grid.times, 3., phases[1]) + _carrier(grid.times, 37., phases[1]))
        raw_b = _project(common, grid.dt, 2., 4., check=check, max_elements=max_elements)
        raw_l = _project(common, grid.dt, 36., 38., check=check, max_elements=max_elements)
        fractions = np.where(np.arange(16) % 2 == 0, .1, .9)
        raw_b, report_b = _switch_normalize(broad, raw_b, 2500 * (1 - fractions), grid,
                                           check=check, max_elements=max_elements)
        raw_l, report_l = _switch_normalize(local, raw_l, 2500 * fractions, grid,
                                           check=check, max_elements=max_elements)
        meta.update(broad_switch_normalization=report_b, local_switch_normalization=report_l,
                    common_envelope_phase_rad=float(phases[0]), common_carrier_phase_rad=float(phases[1]))
    else:
        fraction = {"M1": .1, "M2": .5, "M3": .9}[condition]
        raw_b, report_b = _normalize(broad, _carrier(grid.times, 3.), math.sqrt(2500 * (1 - fraction)),
                                    grid, check=check, max_elements=max_elements)
        raw_l, report_l = _normalize(local, _carrier(grid.times, 37.), math.sqrt(2500 * fraction),
                                    grid, check=check, max_elements=max_elements)
        meta.update(selected_local_fraction=fraction, broad_normalization=report_b, local_normalization=report_l)
    band_b = _project(raw_b, grid.dt, check=check, max_elements=max_elements)
    band_l = _project(raw_l, grid.dt, check=check, max_elements=max_elements)
    meta["whole_record_mixture"] = _cross_report(broad, band_b, local, band_l, check=check, max_elements=max_elements)
    if changing:
        meta["interval_mixtures"] = [_cross_report(broad, band_b[:, (grid.times >= j) & (grid.times < j + 1)],
            local, band_l[:, (grid.times >= j) & (grid.times < j + 1)], check=check,
            max_elements=max_elements) for j in range(16)]
    return np.concatenate((broad, local)), np.concatenate((raw_b, raw_l)), meta


def _interaction(condition, seed, grid, *, check, max_elements):
    four = condition == "I4"
    random = _rng(seed, STREAMS["I4_geometry" if four else "interaction_geometry"])
    centroid = random.uniform(2400., 7200., 2)
    angle = float(random.uniform(0, 2 * np.pi))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if four:
        offsets = np.array([[-150., -150.], [-150., 150.], [150., -150.], [150., 150.]])
        phases = _rng(seed, STREAMS["I4_phases"]).uniform(0, 2 * np.pi, 4)
    else:
        separation = {"I1": 4800., "I2": 300., "I3": 1200.}[condition]
        offsets = np.array([[-separation / 2, 0.], [separation / 2, 0.]])
        phases = _rng(seed, STREAMS["interaction_phases"]).uniform(0, 2 * np.pi, 2)
        if condition != "I1":
            phases[1] = phases[0] + (np.pi if condition == "I3" else 0.)
    centres = centroid + offsets @ rotation.T
    xx, yy = np.meshgrid(grid.x, grid.y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    spatial = np.stack([gaussian_spatial_envelope(points, center_um=centre, fwhm_um=600.,
        axis_ratio=1., orientation_rad=0., max_elements=max_elements).reshape(grid.ny, grid.nx) for centre in centres])
    raw = np.cos(2 * np.pi * 11 * grid.times[None, :] + phases[:, None])
    raw, normalization = _normalize(spatial, raw, 50., grid, check=check, max_elements=max_elements)
    return spatial, raw, dict(normalization, centres_um_xy=centres.tolist(), phases_rad=phases.tolist(),
                              carrier_hz=11., spatial_fwhm_um=600., full_sum_normalized=True)


def _nuisance(condition, seed, grid, *, check, max_elements):
    arrays = {}
    metadata = {"truth_template": "B2", "observation_side_only": True,
                "target_mask_unchanged": True, "DAQ_and_readout_application_qualified": False}
    if condition in ("N1", "N2"):
        frequency, rms = (60., 5.) if condition == "N1" else (50., 50.)
        phase = float(_rng(seed, STREAMS[condition + "_line_phase"]).uniform(0, 2 * np.pi))
        signal = np.sin(2 * np.pi * frequency * grid.times + phase)
        signal *= rms / math.sqrt(float(np.mean(signal**2)))
        arrays["spatially_common_line_uv"] = signal
        metadata.update(line_hz=frequency, line_rms_uv=float(np.sqrt(np.mean(signal**2))), line_phase_rad=phase)
    elif condition == "N3":
        _limit(max_elements, grid.nt * 512, 128 * 4096)
        missing = np.empty((grid.nt, 512), dtype=np.uint8)
        random = _rng(seed, STREAMS["N3_missingness"])
        count = 0
        for start in range(0, grid.nt, 128):
            check()
            block = random.random((min(128, grid.nt - start), 4096)) < .01
            count += int(block.sum())
            missing[start:start + len(block)] = np.packbits(block, axis=1, bitorder="little")
        arrays["missing_sites_packed_little"] = missing
        metadata.update(nuisance_clock_hz=1 / grid.dt, physical_site_shape=[64, 64],
                        flattened_site_order="row_major", bitorder="little", one_means="missing",
                        state_intervals="[n*0.002,(n+1)*0.002)", nominal_missing_probability=.01,
                        realized_missing_count=count, realized_missing_fraction=count / (grid.nt * 4096))
    else:
        metadata["dead_physical_row_zero_based"] = int(_rng(seed, STREAMS["N4_dead_row"]).integers(0, 64))
    return arrays, metadata


def _build(condition_id, replicate_seed, grid, *, check, max_elements, include_scales=True):
    if condition_id not in CONDITIONS or type(replicate_seed) is not int or replicate_seed not in REPLICATE_SEEDS:
        raise ValueError("only the selected 28 conditions and three development seeds are admitted")
    _limit(max_elements, grid.nx * grid.ny * 4, grid.nt * 4)
    check()
    arguments = dict(check=check, max_elements=max_elements)
    nuisance = {}
    if condition_id[0] in "FS":
        spatial, raw, details = build_focal_world(condition_id, x_um=grid.x, y_um=grid.y, times_s=grid.times,
            replicate_seed=replicate_seed, primary_band_transform=lambda x: _project(x, grid.dt, **arguments), **arguments)
    elif condition_id[0] == "B":
        spatial, raw, details = _broad(condition_id, replicate_seed, grid, **arguments)
    elif condition_id[0] == "W":
        spatial, raw, details = _wave(condition_id, replicate_seed, grid, **arguments)
    elif condition_id[0] == "M":
        spatial, raw, details = _mixture(condition_id, replicate_seed, grid, **arguments)
    elif condition_id[0] == "I":
        spatial, raw, details = _interaction(condition_id, replicate_seed, grid, **arguments)
    else:
        spatial, raw, details = _broad("B2", replicate_seed, grid, **arguments)
        nuisance, details["nuisance"] = _nuisance(condition_id, replicate_seed, grid, **arguments)
    _limit(max_elements, spatial.size, raw.size)
    if not np.isfinite(spatial).all() or not np.isfinite(raw).all():
        raise ValueError("nonfinite generated factor")
    band = _project(raw, grid.dt, **arguments)
    power, mean = _moments(spatial, band, **arguments)
    metadata = {"recipe": RECIPE, "condition_id": condition_id, "replicate_seed": replicate_seed,
        "world_id": f"{condition_id}:{replicate_seed}", "independence_group": f"development_replicate_{replicate_seed}",
        "all_same_seed_conditions_must_share_role": True, "independent_seed_groups_in_initial_panel": 3,
        "role": "development_preparation_unassigned", "source_to_surface_model": "not_implemented_by_this_panel",
        "physical_DAQ_calibration_claim": False, "final_world_access": False, "factor_modes": len(spatial),
        "production_grid": grid == _Grid(), "grid_shape_time_y_x": [grid.nt, grid.ny, grid.nx],
        "primary_band_hz": [1., 100.], "primary_band_boundary": "finite_record_DFT_no_padding_no_taper",
        "common_mask": "full_rectangle", "raw_support_separate_from_evaluation_ringing": True,
        "achieved_primary_band_rms_uv": math.sqrt(power), "primary_band_mean_uv": mean,
        "rng": "PCG64_SeedSequence_[20,replicate_seed,stream_id]", "stream_ids": STREAMS,
        "details": details}
    if include_scales:
        metadata["power_scales"] = factorized_field_power_scales(spatial, raw,
            dx_micrometres=9600. / grid.nx, dy_micrometres=9600. / grid.ny, dt_seconds=grid.dt,
            common_mask=np.ones((grid.ny, grid.nx), dtype=bool), **arguments)
    check()
    return DevelopmentWorld(spatial, raw, band, metadata, nuisance)


def build_development_world(condition_id, replicate_seed, *, check=None, max_elements=MAX_ELEMENTS):
    """Build one fixed-grid registered world; return factors and full diagnostics."""
    if check is None:
        check = lambda: None
    if not callable(check):
        raise ValueError("check must be callable")
    return _build(condition_id, replicate_seed, _Grid(), check=check, max_elements=max_elements)


def _json_value(value):
    # k90=0 has an infinite spatial scale by the retained estimator, not an
    # undefined/error result. Preserve it explicitly in strict portable JSON.
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN may not be published")
        return "+Infinity" if value > 0 else "-Infinity"
    return value


def _json_text(value):
    return json.dumps(_json_value(value), sort_keys=True, indent=2, allow_nan=False)


def _save_world(container, world, check):
    key = world.metadata["world_id"].replace(":", "_seed_")
    group = container.create_group(key)
    group.attrs["complete"] = False
    for name in ("spatial_modes", "raw_drivers", "primary_band_drivers"):
        check()
        group.create_dataset(name, data=getattr(world, name))
    nuisances = group.create_group("observation_side_nuisance_inputs")
    for name, value in world.nuisance_arrays.items():
        check()
        nuisances.create_dataset(name, data=value)
    group.create_dataset("metadata_json", data=_json_text(world.metadata))
    container.flush()
    check()
    group.attrs["complete"] = True
    container.flush()


def generate_panel(output_root, world_ids, *, max_seconds=6900., builder=build_development_world):
    """Fresh explicit tranche; preserve completed groups and exact remaining IDs.

Exit status is recorded by main: complete0, deadline75, unresolved1. A failed
world is retained by identity and deterministic recipe; it is never redrawn.
Partial HDF5 groups remain marked incomplete. No automatic successor exists.
"""
    import h5py
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 6900:
        raise ValueError("generation deadline must be positive and at most 6900 seconds")
    identities = list(world_ids)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("a nonempty, duplicate-free explicit world list is required")
    for condition, seed in identities:
        if condition not in CONDITIONS or seed not in REPLICATE_SEEDS:
            raise ValueError("unknown condition or seed in tranche")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed = []
    summary = {"recipe": RECIPE, "requested_world_ids": [f"{c}:{s}" for c, s in identities],
               "completed_world_ids": completed, "status": "running", "max_seconds": max_seconds,
               "nonfinite_JSON_scale_encoding": "+Infinity/-Infinity strings; undefined remains null",
               "scope": "constructed_direct_surface_development_inputs_only",
               "independent_seed_groups": sorted({s for _, s in identities}),
               "all_same_seed_conditions_must_share_role": True}

    def check():
        if time.monotonic() - started >= max_seconds:
            raise GenerationDeadline("caller-owned bounded generation deadline reached")

    current = None
    try:
        with h5py.File(output / "development_worlds.h5", "x") as container:
            container.attrs["recipe"] = RECIPE
            container.attrs["scope"] = summary["scope"]
            coordinates = container.create_group("common_coordinates")
            for name, values in (("x_um", _Grid().x), ("y_um", _Grid().y), ("times_s", _Grid().times)):
                coordinates.create_dataset(name, data=values)
            for condition, seed in identities:
                current = f"{condition}:{seed}"
                check()
                world = builder(condition, seed, check=check, max_elements=MAX_ELEMENTS)
                if world.metadata.get("world_id") != current:
                    raise ValueError("builder returned a different world identity")
                _save_world(container, world, check)
                completed.append(current)
                del world
            summary["status"] = "complete"
    except GenerationDeadline as error:
        summary.update(status="deadline_partial", interrupted_world_id=current, error=str(error))
    except Exception as error:
        summary.update(status="unresolved", failed_world_id=current,
                       error_type=type(error).__name__, error=str(error))
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["remaining_world_ids"] = [item for item in summary["requested_world_ids"] if item not in completed]
        (output / "summary.json").write_text(_json_text(summary) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-seconds", type=float, default=6900.)
    parser.add_argument("--world-ids", help="Explicit comma-separated condition:seed identities; default all 84")
    args = parser.parse_args(argv)
    if args.world_ids:
        try:
            identities = [(item.split(":")[0], int(item.split(":")[1])) for item in args.world_ids.split(",")
                          if len(item.split(":")) == 2]
            if len(identities) != len(args.world_ids.split(",")):
                raise ValueError("malformed identity")
        except ValueError as error:
            parser.error(str(error))
    else:
        identities = [(condition, seed) for seed in REPLICATE_SEEDS for condition in CONDITIONS]
    summary = generate_panel(args.output_root, identities, max_seconds=args.max_seconds)
    print(_json_text(summary), flush=True)
    return {"complete": 0, "deadline_partial": 75, "unresolved": 1}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
