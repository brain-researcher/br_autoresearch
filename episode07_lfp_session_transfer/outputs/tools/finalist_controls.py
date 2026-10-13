"""Pure EP07 ten-control planning and bounded member refitting.

The caller admits the selected finalist, original fit/development-feature
packs, original member predictions and any FitMismatch from derange_fit_lfp.
No file, scorer, outcome reader, finalist selection or opening authority is
provided here. The fixed recipe, seed, rank, member identities and mixing
weights never adapt to a control result. A failed fit propagates without a
retry, rank fallback, substitute member or favorable source subset.

Fit one member/setting at a time. Full-panel composition validates every one
of the ten settings for every member before averaging native predictions.
Control refits are not new evaluated base candidates: no feedback or completed
publication record is manufactured. Exact reused-member and comparator
identity is checked; the surrounding coordinator must also compare a wholly
reused ensemble with its originally published finalist and record the paired
control contact. Score sensitivity and scientific pass/fail are outside this
module. Mismatch checks bind metadata, unchanged counts and actual donor LFP
rows to the supplied original packs; admitting those originals is caller-owned.
"""
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
import re

import numpy as np

import candidate_predictions as publication
from adaptive_recipes import adaptive_pair
from ensemble_predictions import _same_cell
from fit_mismatch import FitMismatch, _roster as mismatch_roster
from native_trials import (DayTrials, EARLIER, NativePredictions, _snapshot,
                           fit_calibration_mean, predict_calibration_mean)
from primary_control import (PrimaryControl, fit_primary_control,
                             predict_primary_control_development)
from primary_fit_scope import bind_fit_days
from primary_model import (PrimaryRecipe, fit_primary_model,
                           predict_primary_development)
from source_weighting import SourceWeightRecipe, SourceWeights
from starting_recipes import starting_pair
from weighted_primary_model import (fit_weighted_primary_model,
                                    predict_weighted_primary_development)


@dataclass(frozen=True)
class FinalistMember:
    candidate_id: str
    recipe: PrimaryRecipe
    weight_recipe: SourceWeightRecipe | None
    weight: float


@dataclass(frozen=True)
class FinalistSpec:
    candidate_id: str
    stream: str
    members: tuple[FinalistMember, ...]


@dataclass(frozen=True)
class MemberControlPlan:
    member: FinalistMember
    action: str
    recipe: PrimaryRecipe
    weight_recipe: SourceWeightRecipe | None
    control: PrimaryControl | None = None
    mismatch_scope: str | None = None


@dataclass(frozen=True)
class FinalistControlPlan:
    number: int
    finalist: FinalistSpec
    optional_component: str | None
    members: tuple[MemberControlPlan, ...]


@dataclass(frozen=True)
class ControlDiagnostics:
    member_id: str
    action: str
    original_recipe: PrimaryRecipe
    original_weight_recipe: SourceWeightRecipe | None
    intervened_recipe: PrimaryRecipe
    weight_recipe: SourceWeightRecipe | None
    control: PrimaryControl | None
    effective_recipe: PrimaryRecipe | None
    retained_source_days: tuple[str, ...]
    source_weights: SourceWeights | None
    donor_trial_ids: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    display_aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MemberControlCell:
    member_id: str
    control_number: int
    prediction: NativePredictions
    calibration_mean: NativePredictions
    diagnostics: ControlDiagnostics


@dataclass(frozen=True)
class ControlPredictions:
    finalist_id: str
    stream: str
    control_number: int
    member_ids: tuple[str, ...]
    member_weights: tuple[float, ...]
    predictions: tuple[NativePredictions, ...]
    calibration_means: tuple[NativePredictions, ...]
    member_diagnostics: tuple[tuple[ControlDiagnostics, ...], ...]


def _weights(weights):
    if (not isinstance(weights, tuple) or len(weights) not in (1, 2)
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   or not math.isfinite(float(value)) or value <= 0 for value in weights)
            or math.fsum(weights) != 1.):
        raise ValueError("one or two fixed finite positive member weights must sum exactly to one")
    return tuple(float(value) for value in weights)


def _declaration(candidate_id, stream):
    index = 0 if stream == "history_assisted" else 1
    if not isinstance(candidate_id, str):
        raise ValueError("one declared starting/adaptive candidate ID is required")
    match = re.fullmatch(r"(starting|adaptive)_([0-9]{2})", candidate_id)
    if match is None:
        raise ValueError("one declared starting/adaptive candidate ID is required")
    kind, number = match.group(1), int(match.group(2))
    if kind == "starting":
        if number == 8:
            raise ValueError("exhausted unscored starting_08 cannot be a finalist or member")
        recipe = starting_pair(number)[index]
        if number == 10:
            recipe = replace(recipe, regression=replace(recipe.regression, max_iterations=100000))
        return recipe, None, None, None
    declaration = adaptive_pair(number).streams[index]
    return (declaration.primary, declaration.source_weighting,
            declaration.member_ids, declaration.member_weights)


def resolve_finalist(candidate_id, *, stream):
    """Resolve an admitted choice; this does not select or qualify a finalist."""
    if stream not in ("history_assisted", "today_only"):
        raise ValueError("one declared primary stream is required")
    recipe, weighting, ids, weights = _declaration(candidate_id, stream)
    if ids is None:
        return FinalistSpec(candidate_id, stream, (FinalistMember(candidate_id, recipe, weighting, 1.),))
    weights = _weights(weights)
    if len(set(ids)) != 2 or len(ids) != len(weights):
        raise ValueError("exactly two distinct base members are required")
    members = []
    for member_id, weight in zip(ids, weights):
        recipe, weighting, nested, _ = _declaration(member_id, stream)
        if nested is not None:
            raise ValueError("nested ensembles are not declared finalist members")
        members.append(FinalistMember(member_id, recipe, weighting, weight))
    return FinalistSpec(candidate_id, stream, tuple(members))


def plan_control(finalist, number):
    """Choose a single global intervention category before any data or score."""
    if (not isinstance(finalist, FinalistSpec)
            or finalist != resolve_finalist(finalist.candidate_id, stream=finalist.stream)):
        raise ValueError("finalist must retain its exact declared members and recipes")
    if isinstance(number, (bool, np.bool_)) or not isinstance(number, Integral) or number not in range(1, 11):
        raise ValueError("control number must be1..10")
    history = finalist.stream == "history_assisted"
    optional = None
    if number == 10:
        optional = ("alignment" if any(member.recipe.alignment != "none" for member in finalist.members)
                    else "nuisance" if any(member.recipe.adjustment != "none" for member in finalist.members)
                    else None)
    members = []
    for member in finalist.members:
        recipe, weighting = member.recipe, member.weight_recipe
        action, control, mismatch = "reuse", None, None
        if (number == 2 and history) or number == 3:
            action, mismatch = "mismatch", "earlier" if number == 2 else "calibration"
        elif number == 4 and history:
            action, control = "source_control", PrimaryControl("remove_history")
        elif number == 5:
            action = "template"
        elif number == 6 and history:
            if weighting is not None and weighting.family != "equal":
                recipe = replace(recipe, source_masses=(weighting.source_total_mass / 3,) * 3,
                                 calibration_mass=weighting.target_mass)
                weighting, action = None, "refit"
            elif weighting is None and len(set(recipe.source_masses)) != 1:
                recipe = replace(recipe, source_masses=(math.fsum(recipe.source_masses) / 3,) * 3)
                action = "refit"
        elif number in (7, 8, 9) and history:
            action, control = "source_control", PrimaryControl("omit_source", int(number) - 6)
        elif number == 10:
            if optional == "alignment" and recipe.alignment != "none":
                recipe = replace(recipe, alignment="none", alignment_regularization=None)
                action = "refit"
            elif optional == "nuisance" and recipe.adjustment != "none":
                action, control = "source_control", PrimaryControl("remove_nuisance")
        members.append(MemberControlPlan(member, action, recipe, weighting, control, mismatch))
    return FinalistControlPlan(int(number), finalist, optional, tuple(members))


def _plan(plan):
    if (not isinstance(plan, FinalistControlPlan)
            or plan != plan_control(plan.finalist, plan.number)):
        raise ValueError("the complete frozen control plan must retain its declared intervention")


def _mismatched_days(value, days, scope, check):
    seed = 7007101 if scope == "earlier" else 7007102
    if not isinstance(value, FitMismatch) or (value.scope, value.seed) != (scope, seed):
        raise ValueError("the fixed mismatch scope/seed must match this control")
    animal, target, _, _ = mismatch_roster(value.packs, check)
    if (animal, target) != (days[-1].animal, days[-1].target_day):
        raise ValueError("mismatch packs belong to a different original target")
    if not isinstance(value.lfp_source_trial_ids, tuple) or len(value.lfp_source_trial_ids) != len(value.packs):
        raise ValueError("every mismatch pack needs its explicit donor tuple")
    shared, selected, donors = {}, {}, {}
    for pack, donor in zip(value.packs, value.lfp_source_trial_ids):
        publication._check(check)
        if (not isinstance(donor, tuple) or len(donor) != len(pack.trial_ids)
                or set(donor) != set(pack.trial_ids)):
            raise ValueError("donors must be a whole-trial permutation of the original identities")
        positions = {name: index for index, name in enumerate(pack.trial_ids)}
        affected = pack.day in EARLIER[animal] if scope == "earlier" else pack.day == target
        if any((recipient == source if affected else recipient != source)
               or pack.directions[index] != pack.directions[positions[source]]
               for index, (recipient, source) in enumerate(zip(pack.trial_ids, donor))):
            raise ValueError("mismatch donors must preserve direction and the selected nonidentity scope")
        if pack.day in shared and donor != shared[pack.day]:
            raise ValueError("regions/streams must retain exactly the same original-day donors")
        shared[pack.day] = donor
        if (pack.region, pack.stream) == (days[-1].region, days[-1].stream):
            selected[pack.day], donors[pack.day] = pack, donor
    for original in days:
        changed = selected[original.day]
        if (changed.trial_ids != original.trial_ids
                or any(not np.array_equal(getattr(changed, name), getattr(original, name))
                       for name in ("directions", "bin_offsets", "lfp_guide", "unit_guide"))):
            raise ValueError("mismatch recipient metadata differs from the supplied original fitting pack")
        if (changed.spikes.dtype != original.spikes.dtype
                or not np.array_equal(changed.spikes, original.spikes)):
            raise ValueError("mismatch must retain the supplied original spike counts exactly")
        if changed.lfp.dtype != original.lfp.dtype or changed.lfp.shape != original.lfp.shape:
            raise ValueError("mismatch must retain the original LFP dtype and native dimensions")
        positions = {name: index for index, name in enumerate(original.trial_ids)}
        # Compare one whole native trajectory at a time, avoiding a second
        # full advanced-indexed copy of an earlier day's LFP tensor.
        for row, donor in enumerate(donors[original.day]):
            publication._check(check)
            if not np.array_equal(changed.lfp[row], original.lfp[positions[donor]]):
                raise ValueError("mismatch LFP values do not match their declared original donor rows")
    return (tuple(selected[day.day] for day in days),
            tuple((day.day, day.trial_ids, donors[day.day]) for day in days))


def run_member_cell(plan, member_id, *, fit_days, development, original_prediction,
                    calibration_mean, mismatch=None, check):
    """Refit/reuse one bounded member cell, propagating every technical failure."""
    publication._check(check)
    _plan(plan)
    matches = [value for value in plan.members if value.member.candidate_id == member_id]
    if len(matches) != 1:
        raise ValueError("one declared finalist member is required")
    action = matches[0]
    days = bind_fit_days(fit_days)
    target = days[-1]
    if (target.stream != plan.finalist.stream or not isinstance(development, DayTrials)
            or development.view_role != "development_features" or development.role != "development"
            or development.spikes is not None or development.unit_guide is not None
            or (development.animal, development.target_day, development.day,
                development.region, development.stream) !=
               (target.animal, target.target_day, target.target_day, target.region, target.stream)):
        raise PermissionError("original fit packs and feature-only development must bind the selected stream")
    donor_ids = ()
    inputs = days
    if plan.number in (2, 3):
        # The unchanged today-only slot in control2 still binds the common
        # full paired mismatch and records its identity donor mapping.
        scope = "earlier" if plan.number == 2 else "calibration"
        inputs, donor_ids = _mismatched_days(mismatch, days, scope, check)
    elif mismatch is not None:
        raise ValueError("this member action does not consume a mismatch result")
    # A fresh target-only comparator supplies the expected output axes/roles
    # even when the requested control simply reuses an existing prediction.
    fresh = predict_calibration_mean(fit_calibration_mean(target, check=check), development, check=check)
    original, baseline = publication._pair(original_prediction, calibration_mean)
    _same_cell((original, baseline), (fresh, fresh))
    effective, weights = None, None
    retained = tuple(day.day for day in days[:-1])
    if action.action == "reuse":
        prediction = original
    elif action.action == "template":
        prediction, retained = fresh, ()
    elif action.action == "source_control":
        fitted = fit_primary_control(days, recipe=action.recipe, control=action.control,
                                     weight_recipe=action.weight_recipe, check=check)
        prediction = predict_primary_control_development(fitted, development, check=check)
        effective, weights = fitted.effective_recipe, fitted.source_weights
        retained = fitted.retained_source_days
    elif action.action in ("refit", "mismatch"):
        if action.weight_recipe is None:
            fitted = fit_primary_model(inputs, recipe=action.recipe, check=check)
            prediction = predict_primary_development(fitted, development, check=check)
            effective = fitted.recipe
        else:
            fitted = fit_weighted_primary_model(inputs, recipe=action.recipe,
                                               weight_recipe=action.weight_recipe, check=check)
            prediction = predict_weighted_primary_development(fitted, development, check=check)
            effective, weights = fitted.effective_recipe, fitted.source_weights
    else:
        raise ValueError("unknown declared member action")
    prediction, baseline = publication._pair(prediction, fresh)
    aliases = (tuple((day, f"source{(index + 1) % 3 + 1}")
                     for index, day in enumerate(EARLIER[target.animal]))
               if plan.number == 1 and target.stream == "history_assisted" else ())
    diagnostics = ControlDiagnostics(member_id, action.action, action.member.recipe,
        action.member.weight_recipe, action.recipe, action.weight_recipe, action.control, effective, retained, weights,
        donor_ids, aliases)
    publication._check(check)
    return MemberControlCell(member_id, plan.number, prediction, baseline, diagnostics)


def compose_control_panel(plan, member_cells, *, check):
    """Validate complete member panels, then average their fixed native outputs."""
    publication._check(check)
    _plan(plan)
    weights = _weights(tuple(value.member.weight for value in plan.members))
    if not isinstance(member_cells, tuple) or len(member_cells) != len(plan.members):
        raise ValueError("one complete cell tuple per declared member is required")
    panels, diagnostics = [], []
    for member, cells in zip(plan.members, member_cells):
        if not isinstance(cells, tuple) or len(cells) != len(publication.PANEL_KEYS):
            raise ValueError("every member requires the complete ten-cell control panel")
        values, records, sessions = {}, {}, {}
        for cell in cells:
            publication._check(check)
            if (not isinstance(cell, MemberControlCell) or cell.member_id != member.member.candidate_id
                    or cell.control_number != plan.number
                    or cell.diagnostics.member_id != cell.member_id
                    or cell.diagnostics.action != member.action):
                raise ValueError("member control identity/action differs from the frozen plan")
            pair = publication._pair(cell.prediction, cell.calibration_mean)
            key = pair[0].animal, pair[0].target_day, pair[0].region
            if key in values:
                raise ValueError("a control panel cannot duplicate a setting")
            publication._session_binding(sessions, pair[0])
            values[key], records[key] = pair, cell.diagnostics
        if set(values) != set(publication.PANEL_KEYS):
            raise ValueError("all fixed control settings are required")
        panels.append(tuple(values[key] for key in publication.PANEL_KEYS))
        diagnostics.append(tuple(records[key] for key in publication.PANEL_KEYS))
    for panel in panels[1:]:
        for left, right in zip(panels[0], panel):
            _same_cell(left, right)
    template_only = all(member.action == "template" for member in plan.members)
    if template_only:
        for panel in panels:
            for prediction, baseline in panel:
                publication._check(check)
                if not np.array_equal(prediction.values, baseline.values):
                    raise ValueError("every template member must equal the calibration comparator exactly")
    predictions, baselines = [], []
    for index in range(len(publication.PANEL_KEYS)):
        publication._check(check)
        first, baseline = panels[0][index]
        if template_only:
            # The comparator identity is the intervention itself. Recombining
            # identical arrays with arbitrary convex weights can round it.
            prediction = baseline
        elif len(panels) == 1:
            prediction = first
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                values = weights[0] * first.values + weights[1] * panels[1][index][0].values
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite fixed-weight native control ensemble")
            prediction = replace(first, values=_snapshot(values))
        predictions.append(prediction)
        baselines.append(baseline)
    return ControlPredictions(plan.finalist.candidate_id, plan.finalist.stream, plan.number,
        tuple(value.member.candidate_id for value in plan.members), weights,
        tuple(predictions), tuple(baselines), tuple(diagnostics))
