"""Freeze both complete EP07 primary finalists before held-out feature access.

Only training/calibration packs enter fitting. Deployment uses the existing
explicit held-out feature wrappers and frozen member weights, including learned
source weights. No selection, scientific qualification, outcome reader, scorer
or opening authority is supplied here. The coordinator owns those boundaries.

Pickle is ONLY for this task's trusted, self-produced, fixed-path fitted state;
it is not an interchange format or a safe loader for external files. The saved
objects contain fitted parameters and fitting identities, never raw trial packs.
Publication is exclusive and leaves failed partials intact. Complete state is
validated before publication and after loading, and arrays are made immutable.
"""
from dataclasses import dataclass, fields, replace
from pathlib import Path
import os
import pickle

import numpy as np

from candidate_predictions import PANEL_KEYS
from finalist_controls import FinalistSpec, FinalistMember, resolve_finalist
from lfp_features import FeatureRecipe, FeatureAxis, FittedLFPFeatures
from native_trials import (DayTrials, CalibrationMean, NativePredictions, EARLIER,
    _snapshot, _trial_ids, fit_calibration_mean, predict_calibration_mean_held_out)
from primary_fit_scope import bind_fit_days
from primary_model import (PrimaryRecipe, FittedPrimary, fit_primary_model,
                           predict_primary_held_out)
from primary_design import PrimaryDesign, DayMeans, DesignAxis
from primary_population import PrimaryPopulation, PrimaryBasis
from regression import RegressionRecipe, FittedRegression, NetworkParameters
from source_weighting import SourceWeightRecipe, SourceWeights
from weighted_primary_model import FittedWeightedPrimary, fit_weighted_primary_model


STREAMS = ("history_assisted", "today_only")
FINAL_PAIR_PATH = (Path(__file__).resolve().parents[1] / "runs" / "final_fit"
                   / "attempt001" / "frozen_final_pair.pkl")


@dataclass(frozen=True)
class FrozenFinalMember:
    member_id: str
    fitted: FittedPrimary | FittedWeightedPrimary


@dataclass(frozen=True)
class FrozenFinalCell:
    finalist: FinalistSpec
    animal: str
    target_day: str
    region: str
    members: tuple[FrozenFinalMember, ...]
    calibration: CalibrationMean

    @property
    def stream(self):
        return self.finalist.stream


@dataclass(frozen=True)
class FrozenFinalPair:
    finalists: tuple[FinalistSpec, FinalistSpec]
    cells: tuple[FrozenFinalCell, ...]


_STATE_CLASSES = (FrozenFinalMember, FrozenFinalCell, FrozenFinalPair,
    FinalistSpec, FinalistMember, FeatureRecipe, FeatureAxis, FittedLFPFeatures,
    CalibrationMean, PrimaryRecipe, FittedPrimary, PrimaryDesign, DayMeans,
    DesignAxis, PrimaryPopulation, PrimaryBasis, RegressionRecipe,
    FittedRegression, NetworkParameters, SourceWeightRecipe, SourceWeights,
    FittedWeightedPrimary)


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource/cancellation check required")
    check()


def _freeze(value, check):
    """Copy only declared fitted fields; exclude extra attributes/raw packs."""
    _check(check)
    if type(value) in _STATE_CLASSES:
        return type(value)(**{field.name: _freeze(getattr(value, field.name), check)
                              for field in fields(value)})
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("fitted state requires finite numeric arrays")
        return _snapshot(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item, check) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError("only declared fitted parameters and identities may be saved")


def _spec(spec):
    if not isinstance(spec, FinalistSpec) or spec.stream not in STREAMS:
        raise TypeError("a resolved globally fixed primary finalist is required")
    if spec != resolve_finalist(spec.candidate_id, stream=spec.stream):
        raise ValueError("finalist differs from its fixed member/recipe declaration")
    return spec


def _same(left, right, name):
    if not np.array_equal(left, right):
        raise ValueError(f"frozen {name} differ")


def _array(value, shape, name):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype.kind not in "biuf" or not np.isfinite(value).all()):
        raise ValueError(f"malformed fitted {name}")


def _ids(ids):
    _trial_ids(ids)


def _primary(member, declaration, cell, check):
    state = member.fitted
    if declaration.weight_recipe is None:
        if not isinstance(state, FittedPrimary) or state.recipe != declaration.recipe:
            raise ValueError("member is unfitted or has a different primary recipe")
        primary = state
    else:
        if (not isinstance(state, FittedWeightedPrimary)
                or not isinstance(state.source_weights, SourceWeights)
                or state.base_recipe != declaration.recipe
                or state.source_weights.recipe != declaration.weight_recipe
                or cell.stream != "history_assisted"):
            raise ValueError("member lacks its exact history-weighted fitting state")
        expected = replace(state.base_recipe, source_masses=state.source_weights.source_masses,
                           calibration_mass=state.source_weights.target_mass)
        if state.effective_recipe != expected:
            raise ValueError("learned masses differ from the effective primary recipe")
        primary = state.primary
        if not isinstance(primary, FittedPrimary) or primary.recipe != expected:
            raise ValueError("weighted primary state is missing or mismatched")
    recipe = primary.recipe
    f, p, d, r = primary.features, primary.population, primary.design, primary.regression
    if not (isinstance(f, FittedLFPFeatures) and isinstance(p, PrimaryPopulation)
            and isinstance(d, PrimaryDesign) and isinstance(r, FittedRegression)):
        raise ValueError("all feature/population/design/regression objects must be fitted")
    scope = cell.animal, cell.target_day, cell.region, cell.stream
    for value in (f, p, d):
        if (value.animal, value.target_day, value.region, value.stream) != scope:
            raise ValueError("fitted state crosses animal/target/region/stream")
    if (f.recipe != recipe.features or p.rank != recipe.population_rank
            or p.alignment != recipe.alignment or p.regularization != recipe.alignment_regularization
            or d.adjustment != recipe.adjustment or r.recipe != recipe.regression
            or f.axes != d.feature_axes):
        raise ValueError("component parameters differ from the frozen member recipe")
    days = (EARLIER[cell.animal] if cell.stream == "history_assisted" else ()) + (cell.target_day,)
    if (tuple(b.day for b in p.bases) != days or tuple(m.day for m in d.day_means) != days
            or tuple(day for day, _ in f.fit_trial_ids) != days
            or len(p.source_maps) != len(days) - 1):
        raise ValueError("ordinary complete fitting roster is required; no control omission")
    bins, width, rank = len(cell.calibration.bin_offsets), len(f.axes), p.rank
    for value in (f.bin_offsets, d.bin_offsets):
        _same(value, cell.calibration.bin_offsets, "native bin offsets")
    _array(f.center, (width,), "feature center")
    _array(f.scale, (width,), "feature scale")
    if np.any(f.scale <= 0):
        raise ValueError("frozen feature scales must be positive")
    for basis, means, (_, trial_ids) in zip(p.bases, d.day_means, f.fit_trial_ids):
        _check(check)
        _ids(trial_ids)
        role = "calibration" if basis.day == cell.target_day else "training"
        if ((basis.animal, basis.target_day, basis.region, basis.stream) != scope
                or basis.role != role or means.role != role
                or basis.trial_ids != trial_ids or means.trial_ids != trial_ids):
            raise ValueError("member fitting identities/roles differ between components")
        _same(basis.bin_offsets, cell.calibration.bin_offsets, "basis native bins")
        _same(basis.unit_guide, means.unit_guide, "day-specific native neuron axis")
        units = len(basis.unit_guide)
        _array(basis.unit_guide, (units, 2), "native neuron guide")
        if len({tuple(row) for row in basis.unit_guide}) != units:
            raise ValueError("native neuron identities must be unique within each day")
        _array(basis.mean, (units,), "population mean")
        _array(basis.components, (units, rank), "population components")
        _array(basis.centered_anchors, (8 * bins, rank), "population anchors")
        _array(means.directions, (len(trial_ids),), "training directions")
        if means.directions.dtype.kind not in "iu" or np.any((means.directions < 0) | (means.directions > 7)):
            raise ValueError("frozen training directions must be integers 0..7")
        counts = np.bincount(means.directions, minlength=8)
        _same(means.trials_per_direction, counts, "per-direction fitting support")
        if np.any(counts < (3 if role == "calibration" else 15)):
            raise ValueError("complete-role minimum fitting support is missing")
        _array(means.feature_means, (8, bins, width), "feature means")
        _array(means.response_means, (8, bins, units), "native response means")
    for matrix in p.source_maps:
        _array(matrix, (rank, rank), "source population map")
    target = d.day_means[-1]
    if target.trial_ids != cell.calibration.calibration_trial_ids:
        raise ValueError("member target calibration trial IDs differ from baseline")
    _same(target.unit_guide, cell.calibration.unit_guide, "target native axis")
    _same(target.response_means, cell.calibration.means, "target calibration means")
    _same(target.trials_per_direction, cell.calibration.trials_per_direction, "calibration support")
    columns = len(d.axes)
    _array(r.input_mean, (columns,), "regression input mean")
    _array(r.output_mean, (rank,), "regression response mean")
    if recipe.regression.family == "shallow_network":
        net, hidden = r.network, recipe.regression.width
        if not isinstance(net, NetworkParameters) or r.coefficients is not None:
            raise ValueError("frozen network parameters required")
        for value, shape in ((net.hidden_kernel, (columns, hidden)), (net.hidden_bias, (hidden,)),
                             (net.output_kernel, (hidden, rank)), (net.output_bias, (rank,))):
            _array(value, shape, "network parameter")
    else:
        if r.network is not None:
            raise ValueError("linear member cannot contain a network")
        _array(r.coefficients, (columns, rank), "regression coefficients")
    expected_rows = tuple((day, len(ids) * bins) for day, ids in f.fit_trial_ids)
    raw_masses = np.asarray(recipe.source_masses + (recipe.calibration_mass,), dtype=np.float64)
    raw_masses /= raw_masses.max()
    raw_masses /= raw_masses.sum()
    if (primary.rows_per_day != expected_rows or tuple(day for day, _ in primary.day_masses) != days
            or r.training_rows != sum(rows for _, rows in expected_rows)
            or r.positive_weight_rows != r.training_rows):
        raise ValueError("fitted regression rows/day identities are incomplete")
    _same(np.asarray([mass for _, mass in primary.day_masses]), raw_masses, "regression day masses")
    if declaration.weight_recipe is not None:
        weights = state.source_weights
        if ((weights.animal, weights.target_day, weights.region, weights.stream) != scope
                or weights.fit_trial_ids != f.fit_trial_ids or weights.source_days != days[:-1]):
            raise ValueError("learned weights are bound to different fitting data")
        _same(weights.bin_offsets, cell.calibration.bin_offsets, "weight native bins")
    return primary, target


def _validate_cell(cell, check):
    _check(check)
    if not isinstance(cell, FrozenFinalCell):
        raise TypeError("one complete frozen final cell is required")
    spec = _spec(cell.finalist)
    if (cell.animal, cell.target_day, cell.region) not in PANEL_KEYS:
        raise ValueError("unknown final day/region setting")
    if (not isinstance(cell.members, tuple)
            or tuple(m.member_id for m in cell.members) != tuple(m.candidate_id for m in spec.members)):
        raise ValueError("frozen members must equal the global finalist declaration")
    baseline = cell.calibration
    if (not isinstance(baseline, CalibrationMean)
            or (baseline.animal, baseline.target_day, baseline.region) !=
               (cell.animal, cell.target_day, cell.region)):
        raise ValueError("independently fitted calibration baseline missing/mismatched")
    _ids(baseline.calibration_trial_ids)
    offsets = np.arange(5, 15) if cell.region == "M1" else np.arange(-15, 0)
    _same(baseline.bin_offsets, offsets, "required native offsets")
    units = len(baseline.unit_guide)
    _array(baseline.unit_guide, (units, 2), "baseline native axis")
    _array(baseline.means, (8, len(offsets), units), "baseline means")
    targets = tuple(_primary(m, declaration, cell, check)[1]
                    for m, declaration in zip(cell.members, spec.members))
    for target in targets[1:]:
        _same(target.directions, targets[0].directions, "member calibration trial directions")
    return targets[0]


def fit_final_cell(finalist, fit_days, *, check):
    """Fit selected members and baseline from original complete fit packs only."""
    _check(check)
    finalist = _spec(finalist)
    days = bind_fit_days(fit_days)
    target = days[-1]
    if target.stream != finalist.stream:
        raise PermissionError("finalist stream and fitting data differ")
    baseline = fit_calibration_mean(target, check=check)
    members = []
    for declaration in finalist.members:
        _check(check)
        fitted = (fit_primary_model(days, recipe=declaration.recipe, check=check)
                  if declaration.weight_recipe is None else
                  fit_weighted_primary_model(days, recipe=declaration.recipe,
                    weight_recipe=declaration.weight_recipe, check=check))
        members.append(FrozenFinalMember(declaration.candidate_id, fitted))
    result = _freeze(FrozenFinalCell(finalist, target.animal, target.target_day,
                                   target.region, tuple(members), baseline), check)
    _validate_cell(result, check)
    return result


def assemble_final_pair(finalists, cells, *, check):
    """Require both full ten-cell panels, sharing target calibration identities."""
    _check(check)
    finalists, cells = tuple(finalists), tuple(cells)
    if len(finalists) != 2 or tuple(_spec(s).stream for s in finalists) != STREAMS:
        raise ValueError("exact globally fixed history then today finalists required")
    by_key, sessions, native = {}, {}, {}
    for cell in cells:
        target = _validate_cell(cell, check)
        if cell.finalist != finalists[STREAMS.index(cell.stream)]:
            raise ValueError("a local setting cannot choose a different finalist")
        key = (cell.stream, cell.animal, cell.target_day, cell.region)
        if key in by_key:
            raise ValueError("duplicate final fitting cell")
        by_key[key] = cell
        session_key = cell.animal, cell.target_day
        identity = (target.trial_ids, tuple(target.directions), tuple(target.trials_per_direction))
        if session_key in sessions and sessions[session_key] != identity:
            raise ValueError("both streams and regions must share calibration trial roles")
        sessions[session_key] = identity
        native_key = cell.animal, cell.target_day, cell.region
        if native_key in native:
            for name in ("unit_guide", "bin_offsets", "means"):
                _same(getattr(cell.calibration, name), getattr(native[native_key], name),
                      "cross-stream calibration " + name)
        native[native_key] = cell.calibration
    keys = tuple((stream, *key) for stream in STREAMS for key in PANEL_KEYS)
    if set(by_key) != set(keys):
        raise ValueError("both finalists require all twenty day/region/stream cells")
    return FrozenFinalPair(finalists, tuple(by_key[key] for key in keys))


def _path(path):
    path = Path(path)
    if not path.is_absolute() or path != FINAL_PAIR_PATH or path.is_symlink():
        raise ValueError("only the exact trusted self-produced final-fit path is supported")
    return path


def save_final_pair(path, pair, *, check):
    """Exclusively publish protocol-5 fitted state; failed partials are retained."""
    _check(check)
    path = _path(path)
    partial = path.with_name(path.name + ".partial")
    if os.path.lexists(path) or os.path.lexists(partial):
        raise FileExistsError("final or partial frozen fitting state already exists")
    if not isinstance(pair, FrozenFinalPair):
        raise TypeError("complete final pair required")
    pair = _freeze(assemble_final_pair(pair.finalists, pair.cells, check=check), check)
    with partial.open("xb") as handle:
        _check(check)
        pickle.dump(pair, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    _check(check)
    os.link(partial, path)
    partial.unlink()


def load_final_pair(path, expected_finalists, *, check):
    """Load ONLY this task's trusted fixed artifact; never use on external pickle."""
    _check(check)
    path = _path(path)
    expected_finalists = tuple(_spec(spec) for spec in expected_finalists)
    with path.open("rb") as handle:
        pair = pickle.load(handle)
        if handle.read(1):
            raise ValueError("unexpected bytes after complete fitted state")
    _check(check)
    if not isinstance(pair, FrozenFinalPair) or pair.finalists != expected_finalists:
        raise ValueError("saved state does not match both selected finalists")
    pair = _freeze(pair, check)
    return assemble_final_pair(pair.finalists, pair.cells, check=check)


def predict_final_cell(cell, held_features, *, check):
    """Apply frozen members and calibration baseline; never fit or open outcomes."""
    _check(check)
    if (not isinstance(held_features, DayTrials)
            or (held_features.view_role, held_features.role) != ("held_out_features", "held_out")
            or held_features.spikes is not None or held_features.unit_guide is not None):
        raise PermissionError("only explicit held-out feature-only packs are accepted")
    _validate_cell(cell, check)
    if (held_features.animal, held_features.target_day, held_features.region, held_features.stream) != (
            cell.animal, cell.target_day, cell.region, cell.stream):
        raise PermissionError("held-out features differ from the frozen cell scope")
    baseline = predict_calibration_mean_held_out(cell.calibration, held_features, check=check)
    values = np.zeros_like(baseline.values, dtype=np.float64)
    for member, declaration in zip(cell.members, cell.finalist.members):
        _check(check)
        fitted = member.fitted.primary if isinstance(member.fitted, FittedWeightedPrimary) else member.fitted
        prediction = predict_primary_held_out(fitted, held_features, check=check)
        if ((prediction.animal, prediction.target_day, prediction.region, prediction.trial_ids,
             prediction.calibration_trial_ids) != (baseline.animal, baseline.target_day,
             baseline.region, baseline.trial_ids, baseline.calibration_trial_ids)):
            raise ValueError("member prediction and baseline identities differ")
        for name in ("directions", "bin_offsets", "unit_guide"):
            _same(getattr(prediction, name), getattr(baseline, name), "prediction " + name)
        _array(prediction.values, values.shape, "native prediction")
        values += declaration.weight * prediction.values
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite fixed-weight final prediction")
    _check(check)
    return replace(baseline, values=_snapshot(values)), baseline
