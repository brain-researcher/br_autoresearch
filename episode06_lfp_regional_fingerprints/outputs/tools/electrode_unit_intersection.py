"""Prospective same-electrode/unit restriction for EP06 development controls.

The completed reader preserves native_unit_guide=(electrode, unit) and names
its native spike axes from those exact coordinates. This shared electrode
coordinate is also explicit in the source authors' fCCA.m lines44-47:
https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/MainAnalysis/fCCA.m
No channel offset, geometry reconstruction or approximate label match is used.

Freeze a paired plan from TRAINING DevelopmentRole metadata, before arrays.
Keep all fifteen already selected LFP electrodes; retain every native spike
unit whose guide electrode occurs among those fifteen, in original unit order.
Multiple units and unit label0 remain if their electrode matches. There is no
new electrode draw, quality/value filter, trial deletion or band selection.
This is a units_on_selected_LFP_electrodes sensitivity: selected LFP electrodes
without a matching unit remain. The plan reports which selected electrodes
have units; it does not establish a bidirectional electrode match or satisfy
all electrode/unit, reliability and population-capacity matching requirements.

Feature restriction changes native-unit design metadata only and never reads
the LFP array. Activity restriction selects that fixed count-axis subset and
keeps every whole trial/native bin. Regions retain different native unit sets;
their later population fits must both use the fixed rank2. Fewer than two
retained units makes the whole control unavailable at the metadata stage.
Numerical rank is intentionally NOT qualified here: the future caller must
translate its existing fit_bandwise/_pca rank2 failure to whole-control
unavailability, with no lower rank, replacement electrode or excluded date.

No file reader, scientific recipe, fitter, scorer or opening authority exists
in this module. Calibration/evaluation values require their separate admitted
readers; this helper cannot authorize access merely from a supplied role.
"""
from dataclasses import dataclass, replace
import math
from numbers import Real

import numpy as np

from bandwise_prediction import _array, _lfp_metadata, _readonly
from component_scoring import BoundActivity, FrozenDesign
from development_views import DevelopmentRole
from prepare_development_views import SESSIONS


class IntersectionUnavailable(ValueError):
    """One region lacks the metadata support for the complete paired control."""


@dataclass(frozen=True)
class RegionUnitIntersection:
    training_metadata: DevelopmentRole
    retained_unit_indices: tuple
    retained_unit_guide: tuple
    retained_axis_ids: tuple
    selected_electrodes_with_units: tuple


@dataclass(frozen=True)
class PairedUnitIntersection:
    m1: RegionUnitIntersection
    pmd: RegionUnitIntersection
    required_population_rank: int = 2


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _region_plan(metadata):
    if not isinstance(metadata, DevelopmentRole) or not isinstance(metadata.design, FrozenDesign):
        raise TypeError("the completed reader's explicit DevelopmentRole metadata is required")
    if metadata.region not in ("M1", "PMd") or metadata.design.role != "training":
        raise ValueError("unit inclusion must be frozen from training metadata in the named region")
    matches = tuple(spec for spec in SESSIONS if spec.filename == metadata.design.session_id)
    if len(matches) != 1 or metadata.animal != matches[0].animal:
        raise ValueError("only the seven declared development sessions have admitted metadata")
    electrodes = metadata.native_electrode_ids
    positions = metadata.selected_electrode_positions
    selected = metadata.selected_electrode_ids
    if (electrodes != tuple(float(value) for value in range(1, 97))
            or not isinstance(positions, tuple) or len(positions) != 15
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                   or not 0 <= value < 96 for value in positions)
            or tuple(sorted(set(positions))) != positions
            or not isinstance(selected, tuple) or selected != tuple(electrodes[index] for index in positions)):
        raise ValueError("the existing fixed fifteen physical electrodes must retain their native identities")
    guide = metadata.native_unit_guide
    if (not isinstance(guide, tuple) or not guide
            or any(not isinstance(row, tuple) or len(row) != 2
                   or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                          or not math.isfinite(value) for value in row) for row in guide)
            or len(set(guide)) != len(guide)):
        raise ValueError("a complete unique finite native electrode/unit guide is required")
    normalized = tuple(tuple(float(value) for value in row) for row in guide)
    axes = tuple(f"{metadata.region}:electrode_{electrode!r}:unit_{unit!r}"
                 for electrode, unit in normalized)
    if metadata.design.axis_ids != axes:
        raise ValueError("the native spike axes do not bind the supplied electrode/unit guide")
    indices = tuple(index for index, (electrode, _) in enumerate(normalized) if electrode in selected)
    if len(indices) < 2:
        raise IntersectionUnavailable(
            f"{metadata.region} retains {len(indices)} guide-matched units; rank2 whole control unavailable")
    occupied = tuple(electrode for electrode in selected
                     if any(normalized[index][0] == electrode for index in indices))
    return RegionUnitIntersection(metadata, indices, tuple(guide[index] for index in indices),
                                  tuple(axes[index] for index in indices), occupied)


def _paired_designs(left, right):
    if not isinstance(left, FrozenDesign) or not isinstance(right, FrozenDesign):
        raise TypeError("both regions require explicitly bound native designs")
    if any(getattr(left, name) != getattr(right, name) for name in (
            "session_id", "role", "directions", "trial_ids", "trial_directions", "time_ids")):
        raise ValueError("paired regions must retain exact whole-trial, role, direction and native-time support")


def plan_paired_unit_intersection(m1_metadata, pmd_metadata, *, check):
    """Declare both guide-based unit subsets without accessing neural arrays."""
    _check(check)
    left, right = _region_plan(m1_metadata), _region_plan(pmd_metadata)
    if m1_metadata.region != "M1" or pmd_metadata.region != "PMd" or m1_metadata.animal != pmd_metadata.animal:
        raise ValueError("one explicitly ordered same-animal M1/PMd metadata pair is required")
    _paired_designs(m1_metadata.design, pmd_metadata.design)
    for name in ("source_trial_indices", "direction_raw", "calibration_halves", "window_starts", "window_stops"):
        if getattr(m1_metadata, name) != getattr(pmd_metadata, name):
            raise ValueError("paired original trial/window metadata differs")
    _check(check)
    return PairedUnitIntersection(left, right)


def _validate_plan(plan, check):
    _check(check)
    if not isinstance(plan, PairedUnitIntersection) or plan.required_population_rank != 2:
        raise TypeError("one paired fixed-rank2 intersection plan is required")
    if not isinstance(plan.m1, RegionUnitIntersection) or not isinstance(plan.pmd, RegionUnitIntersection):
        raise TypeError("both regional unit-selection declarations are required")
    expected = plan_paired_unit_intersection(plan.m1.training_metadata, plan.pmd.training_metadata, check=check)
    if plan != expected:
        raise ValueError("unit selection differs from the exact guide intersection; no manual filtering or reorder")


def _restricted_design(design, regional):
    trained = regional.training_metadata.design
    if not isinstance(design, FrozenDesign):
        raise TypeError("the native whole-trial design must be explicit")
    for name in ("session_id", "directions", "time_ids", "axis_ids"):
        if getattr(design, name) != getattr(trained, name):
            raise ValueError("native scope/axis identities differ from the frozen training unit guide")
    if design.role == "training":
        if design != trained:
            raise ValueError("training restriction must preserve the exact planned whole-trial identities")
    elif set(design.trial_ids) & set(trained.trial_ids):
        raise ValueError("nontraining role overlaps the planned training whole trials")
    return replace(design, axis_ids=regional.retained_axis_ids)


def restrict_paired_features(m1, pmd, *, plan, check):
    """Bind restricted native-unit metadata; leave both LFP arrays untouched."""
    _validate_plan(plan, check)
    left, left_electrodes, left_bands = _lfp_metadata(m1)
    right, right_electrodes, right_bands = _lfp_metadata(pmd)
    if m1.region != "M1" or pmd.region != "PMd":
        raise ValueError("feature inputs must be ordered M1 then PMd")
    _paired_designs(left, right)
    if (left_electrodes != plan.m1.training_metadata.selected_electrode_ids
            or right_electrodes != plan.pmd.training_metadata.selected_electrode_ids
            or left_bands != right_bands):
        raise ValueError("keep the existing paired band order and each region's fixed fifteen electrodes")
    designs = (_restricted_design(left, plan.m1), _restricted_design(right, plan.pmd))
    _check(check)
    return replace(m1, design=designs[0]), replace(pmd, design=designs[1])


def restrict_paired_activity(m1, pmd, *, plan, check):
    """Select fixed native unit columns only, without fitting or rank selection."""
    _validate_plan(plan, check)
    if not isinstance(m1, BoundActivity) or not isinstance(pmd, BoundActivity):
        raise TypeError("both native count arrays require explicit BoundActivity bindings")
    _paired_designs(m1.design, pmd.design)
    designs = (_restricted_design(m1.design, plan.m1), _restricted_design(pmd.design, plan.pmd))
    # BOTH complete metadata bindings precede either native count value read.
    result = []
    for supplied, regional, design in zip((m1, pmd), (plan.m1, plan.pmd), designs):
        _check(check)
        values = _array(supplied.values, supplied.design.shape, counts=True)
        result.append(BoundActivity(design, _readonly(values[:, :, regional.retained_unit_indices])))
    _check(check)
    return tuple(result)
