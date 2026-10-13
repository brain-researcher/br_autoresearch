"""Complete EP19 development selection of the source-specific non-neural model.

This is an in-memory composition of the existing three fitting backends, seed
ensemble, four-parameter calibration and horizon scorer. It performs no source
I/O, sensor/clock/onset construction, EEG search, final refit or access release.
All supplied rows must already be the complete admitted natural-prevalence risk
grid. The caller owns source semantics, the first-32-event prefix endpoint and
the history guard; counts here cannot certify any of those physical facts.

WAY runs all six series folds separately for each of all twelve people. The
self-paced source runs all five whole-person folds and uses calibrated32 as its
fixed primary condition; no-label predictions share the exact scored suffix.
All nine recipes and three seeds are required. An unresolved cell raises without
returning a partial ranking. Ties use an explicit, prospectively fixed caller
measurement of each actual fitted three-seed ensemble, never family proxies.
Each recipe's count and latency are the respective maxima over required folds.

Inputs and the existing array-only fitters remain memory-resident. Recipes and
ensembles run sequentially; at most three fitted seed models and the current
and provisional best OOF
prediction panels are retained. This is not a streaming fitter or a resource
qualification. The caller must hold inputs fixed and supply its allocation's
budget callback and outer execution limit; existing classical fits have no
internal callback and cannot be interrupted by a check between fits.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from boosted_hazard_backend import BoostedHazardModel, boosted_hazard_recipes
from classical_forecast_backend import FittingScope, GroupIdentity, UnresolvedFit, _labels
from ensemble_forecast_backend import (
    SEEDS, _row_keys, fit_ensemble_calibration, mean_seed_log_probabilities,
)
from forecast_components import calibrated_log_probabilities, censoring_loss_and_logit_gradient
from forecast_scoring import HORIZONS, ScoringScope, UnresolvedScore, horizon_log_scores
from model_recipes import self_paced_folds, way_inner_folds
from non_neural_forecast_backend import (
    BaselineFittingExample, ContextFeatures, LogisticHazardModel, _schema,
    logistic_hazard_recipes,
)
from temporal_hazard_backend import (
    ContextSequence, TemporalFittingExample, TemporalHazardModel, temporal_hazard_recipes,
)


@dataclass(frozen=True)
class PreparedBaselineRow:
    identity: GroupIdentity
    decision_time: float
    context: ContextFeatures
    sequence: ContextSequence
    observed_category: int
    complete_bins: int
    observed_through_bins: int


@dataclass(frozen=True)
class CalibrationPartition:
    """Already qualified chronological prefix, omitted guard, and full suffix.

    Prefix rows are exactly rows[:prefix_row_count]; suffix rows are exactly
    rows[scored_suffix_start:]. Labels below are censored at the prefix endpoint,
    rather than copied from full-record future labels. The caller establishes
    32 complete eligible movements and sufficient guard using authentic timing.
    Prefix/suffix values may have different acquisition-boundary clock origins.
    """
    prefix_row_count: int
    prefix_observed_category: tuple[int, ...]
    prefix_complete_bins: tuple[int, ...]
    scored_suffix_start: int


@dataclass(frozen=True)
class PreparedParticipant:
    participant: Any
    rows: tuple[PreparedBaselineRow, ...]
    calibration: CalibrationPartition | None


@dataclass(frozen=True)
class SelectionCost:
    parameter_count: int
    latency_seconds: float


def baseline_recipes(registry, *, source):
    """The complete unchanged nine-configuration baseline panel."""
    return (*logistic_hazard_recipes(registry, source=source),
            *boosted_hazard_recipes(registry, source=source),
            *temporal_hazard_recipes(registry, source=source))


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _cost(cost):
    if (not isinstance(cost, SelectionCost) or not _integer(cost.parameter_count)
            or cost.parameter_count <= 0 or not isinstance(cost.latency_seconds, Real)
            or isinstance(cost.latency_seconds, (bool, np.bool_))
            or not np.isfinite(cost.latency_seconds) or cost.latency_seconds <= 0):
        raise UnresolvedFit("actual_ensemble_parameter_count_and_latency_measurement_required")
    return SelectionCost(int(cost.parameter_count), float(cost.latency_seconds))


def _prepare(registry, source, participants, summary_schema, sequence_schema, rate, check):
    scope = ScoringScope.from_registry(registry, source=source, stage="development")
    people = tuple(participants)
    if (any(not isinstance(person, PreparedParticipant) for person in people)
            or len(people) != len(scope.participants)
            or {person.participant for person in people} != set(scope.participants)):
        raise UnresolvedFit("complete_registered_development_cohort_required")
    lookup, all_boundaries = {}, {}
    for person in people:
        check()
        rows = tuple(person.rows)
        if not rows or any(not isinstance(row, PreparedBaselineRow) for row in rows):
            raise ValueError("each participant requires its complete prepared natural-grid rows")
        keys = _row_keys((row.identity, row.decision_time) for row in rows)
        closed, previous, last_time = set(), None, -np.inf
        for row in rows:
            identity = row.identity
            if (identity.source != source or identity.participant != person.participant
                    or (source == "WAY" and identity.series_or_run not in scope.allowed_series)):
                raise ValueError("prepared row crosses its source, person or development-series role")
            group = (source, person.participant, identity.series_or_run)
            if all_boundaries.setdefault(identity.boundary_id, group) != group:
                raise ValueError("one boundary may not name multiple recording groups")
            if identity.boundary_id != previous:
                if identity.boundary_id in closed:
                    raise ValueError("rows must retain contiguous chronological reset-boundary blocks")
                closed.add(identity.boundary_id)
                previous, last_time = identity.boundary_id, -np.inf
            if row.decision_time <= last_time:
                raise ValueError("decisions must increase within each reset boundary")
            last_time = row.decision_time
            if not isinstance(row.context, ContextFeatures) or not isinstance(row.sequence, ContextSequence):
                raise ValueError("the existing summary and temporal context interfaces are required")
            row.context.validated(schema=summary_schema, source=source, participant=person.participant,
                boundary_id=identity.boundary_id, decision_time=row.decision_time)
            row.sequence.validated(schema=sequence_schema, source=source, participant=person.participant,
                boundary_id=identity.boundary_id, decision_time=row.decision_time, sample_rate_hz=rate)
        category, complete = _labels([row.observed_category for row in rows],
                                    [row.complete_bins for row in rows], len(rows))
        _, coverage = _labels(category, [row.observed_through_bins for row in rows], len(rows))
        if np.any((category >= 0) & (coverage < category)):
            raise ValueError("onset category contradicts full-follow-up coverage")
        if source == "WAY":
            if person.calibration is not None:
                raise ValueError("WAY uses unadapted participant-specific fits")
            if {row.identity.series_or_run for row in rows} != set(scope.allowed_series):
                raise UnresolvedFit("all_six_WAY_series_required_for_each_person")
            start, prefix = 0, None
        else:
            partition = person.calibration
            if (not isinstance(partition, CalibrationPartition)
                    or not _integer(partition.prefix_row_count) or not _integer(partition.scored_suffix_start)
                    or not 0 < partition.prefix_row_count <= partition.scored_suffix_start < len(rows)):
                raise ValueError("qualified nonempty prefix, excluded history guard and full suffix required")
            prefix = _labels(partition.prefix_observed_category, partition.prefix_complete_bins,
                             partition.prefix_row_count)
            # Censoring may remove future labels, never introduce or change an
            # observed onset/category, or gain complete risk intervals.
            n = partition.prefix_row_count
            maximum_risk = np.where(category[:n] >= 0, category[:n], complete[:n])
            if (np.any((prefix[0] >= 0) & (prefix[0] != category[:n]))
                    or np.any((prefix[0] < 0) & (prefix[1] > maximum_risk))):
                raise ValueError("prefix labels must be a truncation of the full-record labels")
            start = int(partition.scored_suffix_start)
        if not np.any(coverage[start:] >= HORIZONS["primary_strict"][1]):
            raise UnresolvedScore("every development person requires primary-horizon follow-up")
        lookup[person.participant] = {"person": person, "rows": rows, "keys": keys,
            "category": category, "complete": complete, "coverage": coverage,
            "start": start, "prefix_labels": prefix}
    return scope, lookup


def _folds(registry, source, scope):
    if source == "WAY":
        return tuple((person, fold) for person in scope.participants
                     for fold in way_inner_folds(registry, person))
    return tuple((None, fold) for fold in self_paced_folds(registry))


def _new_model(recipe):
    if recipe.family == "regularized_logistic_discrete_hazard":
        return LogisticHazardModel()
    if recipe.family == "gradient_boosted_discrete_hazard":
        return BoostedHazardModel()
    if recipe.family == "small_past_only_temporal_convolutional_discrete_hazard":
        return TemporalHazardModel()
    raise ValueError("unregistered baseline family")


def _fit(model, rows, *, recipe, seed, registry, scope, summary_schema, sequence_schema, rate, check):
    check()
    common = dict(recipe=recipe, seed=seed, registry=registry, fitting_scope=scope)
    if recipe.family == "small_past_only_temporal_convolutional_discrete_hazard":
        examples = (TemporalFittingExample(row.sequence, row.context, row.decision_time,
                    row.identity, row.observed_category, row.complete_bins) for row in rows)
        report = model.fit((examples,), **common, sequence_schema=sequence_schema,
            summary_schema=summary_schema, analysis_sample_rate_hz=rate, budget_check=check)
    else:
        examples = (BaselineFittingExample(row.context, row.decision_time, row.identity,
                    row.observed_category, row.complete_bins) for row in rows)
        report = model.fit((examples,), **common, feature_schema=summary_schema)
    check()
    if (not getattr(model, "fitted", False) or not isinstance(report, Mapping)
            or not (report.get("completed") is True or report.get("converged") is True)):
        raise UnresolvedFit(f"baseline_cell_unresolved:{recipe.recipe_id}:{scope.fold_id}:{seed}")
    if (report.get("recipe_id") != recipe.recipe_id or report.get("seed") != seed
            or report.get("fold_id") != scope.fold_id or getattr(model, "fitting_scope", None) != scope):
        raise ValueError("backend returned a different recipe, seed or fitting scope")
    return dict(report)


def _predict(model, rows, *, recipe, check):
    result = np.empty((len(rows), 19), dtype=np.float64)
    boundary = None
    for index, row in enumerate(rows):
        check()
        if row.identity.boundary_id != boundary:
            boundary = row.identity.boundary_id
            model.reset(boundary_id=boundary)
        if recipe.family == "small_past_only_temporal_convolutional_discrete_hazard":
            result[index] = model.predict_log_probabilities(row.sequence, row.context,
                                                            decision_time=row.decision_time)
        else:
            result[index] = model.predict_log_probabilities(row.context, decision_time=row.decision_time)
    return result


def _mean(predictions, keys):
    return mean_seed_log_probabilities(predictions, seed_row_keys={seed: keys for seed in SEEDS},
                                       expected_row_keys=keys)


def _summarize(scope, prepared, predictions):
    """Pool rows within person, then equal-weight people; never equal-weight folds."""
    result = {}
    for condition in ("primary", "zero_label"):
        horizon_results = {}
        for horizon in HORIZONS:
            per_person = []
            for person in scope.participants:
                item, values = prepared[person], predictions[person][condition]
                start = item["start"]
                scores = horizon_log_scores(values, item["category"][start:], item["coverage"][start:],
                                             horizon=horizon)
                count = int(np.count_nonzero(scores["eligible"]))
                per_person.append({"participant": person, "eligible_rows": count,
                    "log_score": float(scores["log_scores"][scores["eligible"]].mean()) if count else None})
            complete = all(row["eligible_rows"] for row in per_person)
            if horizon == "primary_strict" and not complete:
                raise UnresolvedScore("all people require complete primary-horizon score support")
            horizon_results[horizon] = {"complete": complete, "participants": per_person,
                "participant_macro_log_score": (float(np.mean([row["log_score"] for row in per_person]))
                                                if complete else None)}
        nll = []
        for person in scope.participants:
            item, values = prepared[person], predictions[person][condition]
            start = item["start"]
            loss, _ = censoring_loss_and_logit_gradient(values, item["category"][start:],
                                                       item["complete"][start:])
            nll.append({"participant": person, "natural_grid_rows": len(loss),
                        "censoring_aware_19_category_nll": float(loss.mean())})
        result[condition] = {"horizons": horizon_results, "nll_participants": nll,
            "participant_macro_19_category_nll": float(np.mean([
                row["censoring_aware_19_category_nll"] for row in nll]))}
    return result


def run_development_baseline_selection(*, registry, source, participants, summary_schema,
        sequence_schema, analysis_sample_rate_hz, measure_ensemble, parameter_count_basis,
        latency_basis, budget_check):
    """Fit/select the full nine-recipe development panel for ONE complete source.

    No score or fitted model is exposed until every required cell completes.
    Returns complete selected OOF probability/support arrays and all recipe
    summaries/reports. It does not retain fitted weights, construct EEG offsets,
    perform a final refit, choose margins, or declare the method empirically frozen.
    Tie-break NLL is the existing censored likelihood on all natural-grid scored
    rows (including zero-information censored rows), pooled within person, then
    averaged across people, under the same primary adaptation as the log score.
    Secondary horizons with absent support remain explicitly unresolved.

    The prospectively fixed ``measure_ensemble(models=..., recipe=...,
    fitting_scope=...)`` receives the actual three fitted seed models, with no
    validation rows or labels. It must return SelectionCost for that ensemble
    under the supplied common conventions. It may exercise inference state,
    but must not update fitted parameters; this driver explicitly resets before
    every subsequent prefix/suffix prediction. No benchmark or parameter-count
    approximation is implemented here. Both count and latency aggregate by
    maximum across the complete required fold set; all fold measurements remain
    in the result. Missing measurements leave the selection unresolved.
    """
    if not callable(budget_check):
        raise ValueError("explicit allocation-owned budget callback required")
    if not callable(measure_ensemble) or any(not isinstance(value, str) or not value.strip()
            for value in (parameter_count_basis, latency_basis)):
        raise ValueError("fixed ensemble measurement callable and named common cost conventions required")
    recipes = baseline_recipes(registry, source=source)
    summary_schema, sequence_schema = _schema(summary_schema), _schema(sequence_schema)
    scope, prepared = _prepare(registry, source, participants, summary_schema, sequence_schema,
                                analysis_sample_rate_hz, budget_check)
    folds = _folds(registry, source, scope)
    summaries, best_key, best_predictions, best_id = [], None, None, None
    for recipe in recipes:
        predictions = {person: {condition: np.full((len(item["rows"]) - item["start"], 19), np.nan)
                               for condition in ("primary", "zero_label")}
                       for person, item in prepared.items()}
        reports, calibration_reports, fold_costs = [], [], []
        for way_person, fold in folds:
            fitting_scope = FittingScope.development(registry, source=source, fold_id=fold.fold_id,
                                                     participant=way_person)
            if source == "WAY":
                fitting_rows = tuple(row for row in prepared[way_person]["rows"]
                                     if row.identity.series_or_run in fold.fit)
                validation_people = (way_person,)
                validation_indices = {way_person: tuple(index for index, row in enumerate(
                    prepared[way_person]["rows"]) if row.identity.series_or_run in fold.validation)}
            else:
                fitting_rows = tuple(row for person in fold.fit for row in prepared[person]["rows"])
                validation_people = fold.validation
                validation_indices = {person: tuple(range(prepared[person]["start"],
                    len(prepared[person]["rows"]))) for person in validation_people}
            suffix_seeds = {person: {} for person in validation_people}
            prefix_seeds = {person: {} for person in validation_people}
            models = []
            for seed in SEEDS:
                model = _new_model(recipe)
                reports.append(_fit(model, fitting_rows, recipe=recipe, seed=seed, registry=registry,
                    scope=fitting_scope, summary_schema=summary_schema, sequence_schema=sequence_schema,
                    rate=analysis_sample_rate_hz, check=budget_check))
                models.append(model)
            budget_check()
            cost = _cost(measure_ensemble(models=tuple(models), recipe=recipe, fitting_scope=fitting_scope))
            budget_check()
            fold_costs.append({"fold_id": fold.fold_id, "parameter_count": cost.parameter_count,
                               "latency_seconds": cost.latency_seconds})
            for seed, model in zip(SEEDS, models):
                for person in validation_people:
                    item = prepared[person]
                    if source == "self_paced":
                        count = item["person"].calibration.prefix_row_count
                        prefix_seeds[person][seed] = _predict(model, item["rows"][:count],
                                                            recipe=recipe, check=budget_check)
                    # _predict explicitly resets even when prefix and suffix
                    # share a boundary; both adaptation conditions reuse this
                    # one suffix evaluation, never a second state advance.
                    rows = tuple(item["rows"][index] for index in validation_indices[person])
                    suffix_seeds[person][seed] = _predict(model, rows, recipe=recipe, check=budget_check)
            del model, models
            for person in validation_people:
                item, indices = prepared[person], validation_indices[person]
                keys = tuple(item["keys"][index] for index in indices)
                zero_label = _mean(suffix_seeds[person], keys)
                if source == "self_paced":
                    count = item["person"].calibration.prefix_row_count
                    prefix_keys = item["keys"][:count]
                    budget_check()
                    calibration = fit_ensemble_calibration(prefix_seeds[person],
                        seed_row_keys={seed: prefix_keys for seed in SEEDS}, prefix_row_keys=prefix_keys,
                        observed_category=item["prefix_labels"][0], complete_bins=item["prefix_labels"][1],
                        weights=np.ones(count, dtype=np.float64))
                    budget_check()
                    primary = calibrated_log_probabilities(zero_label, calibration["parameters"])
                    calibration_reports.append({"participant": person, "fold_id": fold.fold_id, **calibration})
                else:
                    primary = zero_label
                positions = np.asarray(indices) - item["start"]
                predictions[person]["primary"][positions] = primary
                predictions[person]["zero_label"][positions] = zero_label
        scores = _summarize(scope, prepared, predictions)
        cost = SelectionCost(max(item["parameter_count"] for item in fold_costs),
                             max(item["latency_seconds"] for item in fold_costs))
        primary_score = scores["primary"]["horizons"]["primary_strict"]["participant_macro_log_score"]
        nll = scores["primary"]["participant_macro_19_category_nll"]
        key = (-primary_score, nll, cost.parameter_count, cost.latency_seconds, recipe.recipe_id)
        summaries.append({"recipe_id": recipe.recipe_id, "family": recipe.family,
            "scores": scores, "fit_reports": reports, "calibration_reports": calibration_reports,
            "fold_costs": fold_costs,
            "parameter_count": cost.parameter_count, "latency_seconds": cost.latency_seconds})
        if best_key is None or key < best_key:
            best_key, best_predictions, best_id = key, predictions, recipe.recipe_id
    budget_check()
    selected = []
    for person in scope.participants:
        item, values = prepared[person], best_predictions[person]
        start = item["start"]
        selected.append({"participant": person, "row_keys": item["keys"][start:],
            "observed_category": item["category"][start:].copy(),
            "complete_bins": item["complete"][start:].copy(),
            "observed_through_bins": item["coverage"][start:].copy(), **values})
    return {"source": source, "stage": "development_inner_folds", "completed": True,
        "primary_adaptation": "calibrated32" if source == "self_paced" else "unadapted",
        "selected_recipe_id": best_id, "recipe_results": summaries, "selected_oof": selected,
        "recipes": len(recipes), "seeds": SEEDS, "folds_per_recipe": len(folds),
        "fits_completed": len(recipes) * len(SEEDS) * len(folds),
        "parameter_count_basis": parameter_count_basis, "latency_basis": latency_basis,
        "cost_metadata_origin": "caller_measurement_of_actual_three_seed_ensemble_before_validation",
        "cost_aggregation": "separate_maximum_count_and_latency_across_all_required_folds",
        "source_admission_established": False, "empirical_method_freeze_established": False,
        "final_refit_performed": False, "EEG_candidates_scored": False}
