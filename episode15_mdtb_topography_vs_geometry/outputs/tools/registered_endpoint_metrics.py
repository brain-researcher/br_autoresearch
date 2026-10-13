"""Selected EP15 endpoint arithmetic; no reader, fitting or decision rule.

Implements paper_plan's 2026-10-05 selected metric and signed contrasts.
The caller supplies aligned participants, the complete condition order below,
development-only source geometries/scales, and already estimated native
signatures. This module does not establish those inputs' roles or provenance.
It supplies no confidence bounds, scientific margins, class decisions or
development nuisance refits. Passing numeric margins is not their approval.
"""
import numpy as np

from participant_cross_replicate_covariance import cross_replicate_covariance


DOMAINS = (
    ("CPRO",),
    ("Prediction", "PredictViol", "PredictScram"),
    ("SpatialMapEasy", "SpatialMapMed", "SpatialMedDiff"),
    ("NatureMovie", "RomanceMovie", "LandscapeMovie"),
    ("MentalRotEasy", "MentalRotMed", "MentalRotDiff"),
    ("BodyMotionIntact", "BodyMotionScram"),
    ("RespAltEasy", "RespAltMed", "RespAltDiff"),
)
CONDITIONS = tuple(condition for domain in DOMAINS for condition in domain)
CLASSES = ("M1", "M2", "M3", "M4")
MARGINS = ("tau_signal", "tau_remaining", "tau_increment", "tau_geometry",
           "tau_boundary", "tau_alignment", "tau_ablation")


def _finite(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric data")
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite; no coordinates may be dropped")
    return value


def _scalar(value, name, *, positive=False):
    value = _finite(value, name)
    if value.ndim or (positive and not value > 0):
        raise ValueError(f"{name} must be a finite{' positive' if positive else ''} scalar")
    return float(value)


def condition_weights():
    """All 18 conditions, each of seven domains receiving exactly 1/7."""
    return np.asarray([1 / (7 * len(domain)) for domain in DOMAINS
                       for _ in domain], dtype=np.float64)


def geometry_weights():
    """Full-matrix weights equivalent to the 171-entry upper-triangle metric.

    Each of 28 unordered domain pairs has weight 1/28. Off-diagonal
    symmetric copies each receive half their unique-entry weight.
    """
    labels = np.asarray([g for g, domain in enumerate(DOMAINS) for _ in domain])
    weights = np.zeros((18, 18), dtype=np.float64)
    for c in range(18):
        for d in range(c, 18):
            g, h = labels[c], labels[d]
            size = (len(DOMAINS[g]) * len(DOMAINS[h]) if g != h
                    else len(DOMAINS[g]) * (len(DOMAINS[g]) + 1) // 2)
            weight = 1 / (28 * size)
            if c == d:
                weights[c, d] = weight
            else:
                weights[c, d] = weights[d, c] = weight / 2
    return weights


def _geometry(value, name):
    value = _finite(value, name)
    if value.ndim < 2 or value.shape[-2:] != (18, 18):
        raise ValueError(f"{name} must preserve the full 18-by-18 condition geometry")
    # Geometry construction stores identical values for the two symmetric
    # copies. Do not silently symmetrize an incompatible caller input or use
    # this structural requirement as an operator-qualification tolerance.
    if not np.array_equal(value, np.swapaxes(value, -1, -2)):
        raise ValueError(f"{name} must store the same value in both symmetric copies")
    return value


def crossvalidated_geometries(partition_maps):
    """Return (participant, two replicates, 18, 18) from four run partitions.

    Input axes are participant, partitions 1..4, condition, common voxel.
    Each map is spatially centered; pairings are exactly 1x2 and 3x4.
    Negative crossvalidated diagonals and all signed entries are preserved.
    """
    maps = _finite(partition_maps, "partition_maps")
    if maps.ndim != 4 or maps.shape[1:3] != (4, 18) or maps.shape[3] < 1:
        raise ValueError("maps require participant, four partitions, 18 conditions, voxels")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        centered = maps - maps.mean(axis=-1, keepdims=True)
        cross = centered[:, (0, 2)] @ np.swapaxes(centered[:, (1, 3)], -1, -2)
        result = (cross + np.swapaxes(cross, -1, -2)) / (2 * maps.shape[3])
    return _geometry(result, "crossvalidated_geometry")


def geometry_inner(first, second):
    """Metric inner product, keeping every broadcast leading coordinate."""
    first = _geometry(first, "first")
    second = _geometry(second, "second")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = np.sum(first * second * geometry_weights(), axis=(-2, -1))
    return _finite(result, "geometry_inner_product")


def project_gain_direction(geometry, source_geometry):
    """Apply P0 to supplied geometry, without fitting or replacing K0.

    Supply K-K0 or Khat-K0 to obtain registered deviations. A common audit
    K0 has shape (18,18); fold-specific K0 must have explicitly broadcast
    leading axes, e.g. (participant,1,18,18) for two observed replicates.
    Zero/nonfinite source energy is refused, not replaced by a trace scale.
    """
    geometry = _geometry(geometry, "geometry")
    source = _geometry(source_geometry, "source_geometry")
    energy = geometry_inner(source, source)
    if np.any(energy <= 0):
        raise ValueError("every supplied source geometry must have positive metric energy")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        coefficient = geometry_inner(geometry, source) / energy
        result = geometry - coefficient[..., None, None] * source
    return _geometry(result, "gain_projected_geometry")


def map_covariance(first, second):
    """Seven-domain/equal-voxel contraction of matched participant covariance.

    Arrays are group-relative or model-residual maps with axes
    (participant,18 conditions,common voxel). Spatial means are removed here.
    """
    first, second = _finite(first, "first"), _finite(second, "second")
    if first.shape != second.shape or first.ndim != 3 or first.shape[1] != 18 or first.shape[2] < 1:
        raise ValueError("matched maps require participant, 18 conditions, common voxels")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        first = first - first.mean(axis=-1, keepdims=True)
        second = second - second.mean(axis=-1, keepdims=True)
        covariance = cross_replicate_covariance(first, second)
        result = np.sum(covariance.mean(axis=-1) * condition_weights())
    return _scalar(result, "map_covariance")


def geometry_covariance(first, second):
    """Domain-pair contraction for already P0-projected matched deviations."""
    first, second = _geometry(first, "first"), _geometry(second, "second")
    if first.ndim != 3 or first.shape != second.shape:
        raise ValueError("matched geometry requires participant, 18, 18 axes")
    covariance = cross_replicate_covariance(first, second)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = np.sum(covariance * geometry_weights())
    return _scalar(result, "geometry_covariance")


def oriented_contrasts(*, v_map, v_geometry, s_map, s_geometry,
                       r_map, r_geometry, margins, boundary, interior,
                       alignment_advantage, ablation_gains):
    """Return the selected 18+2Q point contrasts, not bounds or class support.

    Both remaining-variation inputs name M1..M4. Geometry adequacy is emitted
    only for M1,M2,M4; R_geom(M3) is used in the paired M4 increment. M3's
    geometry assertions use equivalence and opposite signs of M4 increments,
    without adding a geometry-adequacy primitive for M3.
    boundary/interior each contain V, R (M1), and positive development S.
    alignment_advantage is the equal-participant mean M2-minus-matched-warp
    signature. ablation_gains maps every predeclared mode ID to its two
    equal-participant observed-geometry error-reduction estimates. Model-only
    geometry changes are not these input estimates. No mode selection occurs.
    """
    if set(margins) != set(MARGINS):
        raise ValueError("all seven symbolic margins must be explicitly supplied")
    tau = {key: _scalar(margins[key], key) for key in MARGINS}
    if any(value < 0 for value in tau.values()):
        raise ValueError("margins cannot be negative")
    if set(r_map) != set(CLASSES) or set(r_geometry) != set(CLASSES):
        raise ValueError("both remaining-variation inputs must name all four classes")
    vm, vg = _scalar(v_map, "V_map"), _scalar(v_geometry, "V_geom")
    sm, sg = _scalar(s_map, "S_map", positive=True), _scalar(s_geometry, "S_geom", positive=True)
    rm = {key: _scalar(r_map[key], "R_map_" + key) for key in CLASSES}
    rg = {key: _scalar(r_geometry[key], "R_geom_" + key) for key in CLASSES}
    out = {"G_map": vm - tau["tau_signal"] * sm,
           "G_geom": vg - tau["tau_geometry"] * sg}
    for label, variance, remaining, classes in (("map", vm, rm, CLASSES),
            ("geom", vg, rg, ("M1", "M2", "M4"))):
        for model in classes:
            out[f"A_{label},{model}"] = tau["tau_remaining"] * variance - remaining[model]
        for reference in ("M1", "M2", "M3"):
            out[f"I_{label},M4|{reference}"] = remaining[reference] - remaining["M4"] - tau["tau_increment"] * variance
    native = []
    for name, values in (("boundary", boundary), ("interior", interior)):
        if set(values) != {"V", "R", "S"}:
            raise ValueError(f"{name} must supply its V, M1 R, and development-only S")
        native.append((_scalar(values["V"], name + " V"),
                       _scalar(values["R"], name + " R"),
                       _scalar(values["S"], name + " S", positive=True)))
    (vb, rb, sb), (vi, ri, si) = native
    out["M1_interior_adequacy"] = tau["tau_remaining"] * vi - ri
    out["M1_boundary_enrichment"] = (vb - rb) / sb - (vi - ri) / si - tau["tau_boundary"]
    out["M2_alignment_advantage"] = _scalar(alignment_advantage, "alignment_advantage") - tau["tau_alignment"]
    for mode, gains in ablation_gains.items():
        if not isinstance(mode, str) or not mode:
            raise ValueError("each declared ablation mode needs an explicit identity")
        gains = _finite(gains, "ablation_gains")
        if gains.shape != (2,):
            raise ValueError("each declared mode requires both observed-geometry replicates")
        for replicate in range(2):
            out[f"M4_ablation,{mode},replicate{replicate + 1}"] = float(gains[replicate]) - tau["tau_ablation"]
    if not np.isfinite(list(out.values())).all():
        raise ArithmeticError("nonfinite oriented contrast; no clipping or substitution")
    return out
