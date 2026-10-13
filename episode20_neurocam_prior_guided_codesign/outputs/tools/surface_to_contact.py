"""Once-only spatial averaging of selected EP20 zero-gap development fields.

No temporal interpolation, contact-gap model, calibration, actual DAQ mapping,
physical-eligibility decision or reference subtraction is supplied. Geometry is
an explicitly oriented mathematical lattice. Voltages remain microvolts until
a requested bounded raw-driver sample block is converted once to volts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from development_worlds import CONDITIONS, DevelopmentWorld, RECIPE

APERTURE_UM = 9600.
PAD_DIMENSIONS = {"L": (70., 135.), "M": (50., 95.), "S": (35., 70.)}
GEOMETRIES = {"G0": (None, None), "G-I2-M": ("M", 2), "G-T8-M": ("M", 8),
              "G-I2-S": ("S", 2), "G-T8-S": ("S", 8)}


@dataclass(frozen=True)
class PadGeometry:
    centres_um_xy: np.ndarray
    dimensions_um_xy: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class ContactFactors:
    """contact_modes[site,mode] times unchanged raw_drivers[mode,time]."""
    contact_modes: np.ndarray
    raw_drivers: np.ndarray
    metadata: dict

    def raw_voltage_block(self, start, stop, *, max_elements, check=None):
        """Values at existing driver sample indices only; no holding/interpolation."""
        check = _check(check)
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.raw_drivers.shape[1]):
            raise ValueError("explicit nonempty in-range raw sample block required")
        _limit(max_elements, self.contact_modes.size, self.raw_drivers.size,
               self.contact_modes.shape[0] * (stop - start))
        check()
        volts = (self.contact_modes @ self.raw_drivers[:, start:stop]) * 1e-6
        if not np.isfinite(volts).all():
            raise ValueError("contact voltages are not representable")
        return volts


def _check(check):
    if check is None:
        return lambda: None
    if not callable(check):
        raise ValueError("check must be callable")
    return check


def _limit(max_elements, *sizes):
    if type(max_elements) is not int or max_elements < 1 or any(n > max_elements for n in sizes):
        raise ValueError("explicit per-array max_elements exceeded")


def _real_array(value, name, ndim, max_elements):
    if (not isinstance(value, np.ndarray) or value.ndim != ndim or value.dtype.kind not in "fiu"
            or not value.size):
        raise ValueError(f"{name} must be an explicit finite real array")
    _limit(max_elements, value.size)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return np.asarray(value, dtype=np.float64)


def registered_pad_geometry(geometry_id, *, short_axis, max_elements, check=None):
    """Select a catalogue map; x/y registration is explicitly unverified."""
    check = _check(check)
    if geometry_id not in GEOMETRIES or short_axis not in ("x", "y"):
        raise ValueError("registered geometry and explicit short_axis x or y required")
    _limit(max_elements, 4096 * 2)
    check()
    row, column = np.indices((64, 64))
    small, block = GEOMETRIES[geometry_id]
    smaller = np.zeros((64, 64), dtype=bool) if small is None else ((row // block + column // block) % 2 == 1)
    centres = np.column_stack(((column.ravel() + .5) * 150., (row.ravel() + .5) * 150.))
    dimensions = np.tile(PAD_DIMENSIONS["L"], (4096, 1))
    if small is not None:
        dimensions[smaller.ravel()] = PAD_DIMENSIONS[small]
    if short_axis == "y":
        dimensions = dimensions[:, ::-1].copy()
    counts = {"L": int((~smaller).sum())}
    if small:
        counts[small] = int(smaller.sum())
    return PadGeometry(centres, dimensions, {
        "geometry_id": geometry_id, "short_axis": short_axis,
        "long_axis": "y" if short_axis == "x" else "x", "class_counts": counts,
        "site_order": "row_major_64_by_64", "row_axis": "y", "column_axis": "x",
        "centres_um": "x=(column+0.5)*150; y=(row+0.5)*150",
        "centre_origin_and_axis_registration": "prospective_virtual_convention_unverified_physical_mapping",
        "published_orientation_alignment_verified": False, "physical_eligibility_established": False})


def _rectangles(geometry, max_elements):
    if not isinstance(geometry, PadGeometry):
        raise ValueError("explicit PadGeometry required")
    centres = _real_array(geometry.centres_um_xy, "pad centres", 2, max_elements)
    dimensions = _real_array(geometry.dimensions_um_xy, "pad dimensions", 2, max_elements)
    if centres.shape[1] != 2 or dimensions.shape != centres.shape or np.any(dimensions <= 0):
        raise ValueError("centres and positive dimensions need matching [site,xy] shape")
    left, right = centres - dimensions / 2, centres + dimensions / 2
    if np.any(left < 0) or np.any(right > APERTURE_UM):
        raise ValueError("complete rectangles must stay inside the fixed aperture")
    return centres, dimensions, left, right


def _axis_nodes(values, max_elements):
    values = _real_array(values, "target axis", 1, max_elements)
    if len(values) < 2:
        raise ValueError("at least two target centres required")
    expected = (np.arange(len(values)) + .5) * (APERTURE_UM / len(values))
    if not np.allclose(values, expected, rtol=0., atol=1e-9):
        raise ValueError("target axis must centre-sample the entire9600um aperture")
    return values


def _axis_weights(nodes, left, right, *, max_elements, check):
    """Exact integral of linear interpolation plus constant outer half-cells."""
    edges = np.concatenate(([0.], nodes, [APERTURE_UM]))
    rows, columns, values = [], [], []
    for row, (low, high) in enumerate(zip(left, right)):
        check()
        first = max(0, int(np.searchsorted(edges, low, side="right")) - 1)
        last = min(len(edges) - 2, int(np.searchsorted(edges, high, side="left")))
        for segment in range(first, last + 1):
            a, b = max(low, edges[segment]), min(high, edges[segment + 1])
            if b <= a:
                continue
            width = (b - a) / (high - low)
            if segment == 0 or segment == len(edges) - 2:
                rows.append(row); columns.append(0 if segment == 0 else len(nodes) - 1); values.append(width)
            else:
                fraction = ((a + b) / 2 - edges[segment]) / (edges[segment + 1] - edges[segment])
                rows.extend((row, row)); columns.extend((segment - 1, segment))
                values.extend((width * (1 - fraction), width * fraction))
            _limit(max_elements, len(values))
    result = csr_matrix((values, (rows, columns)), shape=(len(left), len(nodes)))
    result.sum_duplicates()
    # Do not silently renormalize a defective quadrature/geometry.
    if not np.allclose(np.asarray(result.sum(axis=1)).ravel(), 1., rtol=0., atol=2e-13):
        raise ValueError("rectangle interpolation weights fail constant preservation")
    return result


def _gaussian_interval_average(low, high, centre, sigma):
    a, b = (low - centre) / sigma, (high - centre) / sigma
    # Use the survival-tail identity on the positive side to avoid subtracting
    # two values rounded to one. No finite tail is deliberately clipped.
    probability = np.where(a >= 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    return math.sqrt(2 * math.pi) * sigma * probability / (high - low)


def integrate_spatial_modes(spatial_modes, *, x_um, y_um, descriptors, geometry,
                            max_elements, check=None):
    """Integrate explicit modes; analytic modes require exact supplied parameters.

Descriptors, one per mode: {'kind':'raster_bilinear'}, or isotropic
{'kind':'gaussian','centre_um_xy':[x,y],'fwhm_um':f}, or
{'kind':'wave_cos'/'wave_sin','kx_cycles_per_um':kx,'ky_cycles_per_um':ky,
 'origin_um_xy':[x,y],'phase_rad':phase}. Analytic modes must match supplied
raster samples within fixed roundoff tolerances; mismatches fail, never fit or
fall back to inferred parameters. Raw temporal amplitudes are handled elsewhere.
"""
    check = _check(check)
    spatial = _real_array(spatial_modes, "spatial modes", 3, max_elements)
    x, y = _axis_nodes(x_um, max_elements), _axis_nodes(y_um, max_elements)
    if spatial.shape[1:] != (len(y), len(x)) or len(descriptors) != len(spatial):
        raise ValueError("one descriptor per mode and matching target axes required")
    centres, dimensions, left, right = _rectangles(geometry, max_elements)
    _limit(max_elements, len(centres) * len(spatial), len(centres) * len(x), 2 * len(x) * len(y))
    check()
    contacts = np.empty((len(centres), len(spatial)))
    wx = wy = xrows = None
    for index, descriptor in enumerate(descriptors):
        check()
        kind = descriptor.get("kind")
        if kind == "raster_bilinear":
            if wx is None:
                wx = _axis_weights(x, left[:, 0], right[:, 0], max_elements=max_elements, check=check)
                wy = _axis_weights(y, left[:, 1], right[:, 1], max_elements=max_elements, check=check)
                xrows = np.repeat(np.arange(len(centres)), np.diff(wx.indptr))
            intermediate = wy @ spatial[index]
            contacts[:, index] = np.bincount(xrows,
                weights=intermediate[xrows, wx.indices] * wx.data, minlength=len(centres))
            continue
        if kind == "gaussian":
            centre = np.asarray(descriptor.get("centre_um_xy"), dtype=float)
            fwhm = float(descriptor.get("fwhm_um", float("nan")))
            if centre.shape != (2,) or not np.isfinite(centre).all() or not math.isfinite(fwhm) or fwhm <= 0:
                raise ValueError("analytic Gaussian needs explicit finite centre and positive FWHM")
            sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
            sampled = np.exp(-.5 * (((x[None, :] - centre[0]) / sigma)**2 + ((y[:, None] - centre[1]) / sigma)**2))
            averaged = (_gaussian_interval_average(left[:, 0], right[:, 0], centre[0], sigma)
                        * _gaussian_interval_average(left[:, 1], right[:, 1], centre[1], sigma))
        elif kind in ("wave_cos", "wave_sin"):
            kx = float(descriptor.get("kx_cycles_per_um", float("nan")))
            ky = float(descriptor.get("ky_cycles_per_um", float("nan")))
            phase = float(descriptor.get("phase_rad", float("nan")))
            origin = np.asarray(descriptor.get("origin_um_xy"), dtype=float)
            if origin.shape != (2,) or not np.isfinite(origin).all() or not all(map(math.isfinite, (kx, ky, phase))):
                raise ValueError("analytic wave needs explicit finite cyclic wavevector, phase and origin")
            function = np.cos if kind == "wave_cos" else np.sin
            sampled = function(2 * np.pi * (kx * (x[None, :] - origin[0]) + ky * (y[:, None] - origin[1])) + phase)
            averaged = function(2 * np.pi * (kx * (centres[:, 0] - origin[0]) + ky * (centres[:, 1] - origin[1])) + phase)
            averaged *= np.sinc(kx * dimensions[:, 0]) * np.sinc(ky * dimensions[:, 1])
        else:
            raise ValueError("unsupported or missing explicit spatial descriptor")
        if not np.allclose(sampled, spatial[index], rtol=2e-13, atol=2e-14):
            raise ValueError("analytic parameters do not reproduce the stored unscaled spatial mode")
        contacts[:, index] = averaged
    if not np.isfinite(contacts).all():
        raise ValueError("averaged spatial modes are not representable")
    return contacts


def _descriptors(metadata, rank):
    condition, details = metadata["condition_id"], metadata["details"]
    if condition not in CONDITIONS:
        raise ValueError("unknown selected condition")
    if condition[0] in "BMN":
        return [{"kind": "raster_bilinear"} for _ in range(rank)]
    if condition[0] == "W":
        wavelength = float(details["wavelength_um"])
        direction = math.radians(float(details["direction_degrees"]))
        if rank != 2 or not math.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("selected wave requires two modes and positive wavelength")
        base = {"kx_cycles_per_um": math.cos(direction) / wavelength,
                "ky_cycles_per_um": math.sin(direction) / wavelength,
                "origin_um_xy": [4800., 4800.], "phase_rad": details["phase_rad"]}
        return [dict(base, kind=kind) for kind in ("wave_cos", "wave_sin")]
    if condition[0] == "I":
        centres, width = details["centres_um_xy"], details["spatial_fwhm_um"]
        if len(centres) != rank:
            raise ValueError("interaction centre count must match stored mode count")
    elif condition[0] in "FS":
        width = details["nominal_spatial_fwhm_um"]
        centres = [None] * rank
        for event in details["events"]:
            index, centre = event["mode_index"], event["centre_um_xy"]
            if type(index) is not int or not 0 <= index < rank:
                raise ValueError("event mode index is invalid")
            if centres[index] is not None and centres[index] != centre:
                raise ValueError("merged events disagree on exact spatial centre")
            centres[index] = centre
        if any(centre is None for centre in centres):
            raise ValueError("exact event centre missing for a stored mode")
    else:
        raise ValueError("unsupported condition")
    return [{"kind": "gaussian", "centre_um_xy": centre, "fwhm_um": width} for centre in centres]


def build_contact_factors(world, geometry, *, max_elements, check=None):
    """Connect a complete production recipe world to once-averaged raw factors."""
    check = _check(check)
    if not isinstance(world, DevelopmentWorld):
        raise ValueError("explicit DevelopmentWorld required; no file/payload inference")
    meta = world.metadata
    if (meta.get("recipe") != RECIPE or meta.get("production_grid") is not True
            or meta.get("grid_shape_time_y_x") != [8000, 384, 384]
            or meta.get("common_mask") != "full_rectangle"):
        raise ValueError("only the fixed selected zero-gap direct-surface production recipe is supported")
    spatial = _real_array(world.spatial_modes, "spatial modes", 3, max_elements)
    raw = _real_array(world.raw_drivers, "raw drivers", 2, max_elements)
    if spatial.shape[1:] != (384, 384) or raw.shape != (len(spatial), 8000):
        raise ValueError("production factors do not match declared grid")
    if not isinstance(geometry, PadGeometry) or geometry.centres_um_xy.shape != (4096, 2):
        raise ValueError("registered full4096site geometry required")
    canonical = registered_pad_geometry(geometry.metadata.get("geometry_id"),
        short_axis=geometry.metadata.get("short_axis"), max_elements=max_elements, check=check)
    if (not np.array_equal(geometry.centres_um_xy, canonical.centres_um_xy)
            or not np.array_equal(geometry.dimensions_um_xy, canonical.dimensions_um_xy)):
        raise ValueError("geometry arrays do not match their declared registered map")
    descriptors = _descriptors(meta, len(spatial))
    nodes = (np.arange(384) + .5) * 25.
    contacts = integrate_spatial_modes(spatial, x_um=nodes, y_um=nodes,
        descriptors=descriptors, geometry=geometry, max_elements=max_elements, check=check)
    # Owning a copy prevents subsequent mutation of the world's raw factors
    # from changing this operator's stored input. No filtering or rescaling.
    drivers = raw.copy()
    drivers.setflags(write=False)
    contacts.setflags(write=False)
    return ContactFactors(contacts, drivers, {
        "world_id": meta["world_id"], "independence_group": meta["independence_group"],
        "role": meta["role"], "geometry": dict(canonical.metadata),
        "spatial_averaging": "uniform_rectangle_once", "gap_um": 0.,
        "raster_interpolant": "bilinear_constant_outer_half_cell_extension",
        "analytic_mode_matching_rtol": 2e-13, "analytic_mode_matching_atol": 2e-14,
        "mode_descriptors": descriptors, "raw_drivers_unchanged": True,
        "reference_subtracted": False, "spatial_common_mode_retained": True,
        "temporal_operation": "existing_raw_sample_indices_only_no_interpolation_or_holding",
        "raw_driver_units": "microvolts", "returned_block_units": "volts",
        "nuisance_applied": False, "contact_impedance_or_electronics_applied": False,
        "source_to_surface_or_contact_gap_model": False, "calibrated_device_claim": False,
        "physical_eligibility_established": False})
