"""Final source-specific EP19 baseline refits on caller-admitted arrays.

An already selected recipe is fitted three times per WAY person on series1–7,
or three times on all15 self-paced development people. This module neither
selects that recipe nor opens a source, feedback packet or final evaluation.
The existing fitters retain their numerical schedules and participant weights.

The common EEG offset is the log of the uncalibrated probability mean of the
three fitted baseline seeds. Its predictions on fitting rows are explicitly TRAINING
predictions, not out-of-fold evidence. This is the declared stagewise fusion:
freeze the baseline, then fit each matched EEG correction on identical offsets.
Calibration belongs to each complete model after fusion/seed aggregation; no
standalone baseline calibration is inserted into these fitting offsets.

All inputs, fitted models and offset arrays remain memory resident. This is not
a streaming fitter or production resource qualification. The caller supplies
an allocation budget callback and outer process limit; classical fitting has
no internal callback. No partial result is returned after a required fit fails.
"""
from dataclasses import dataclass

import numpy as np

from classical_forecast_backend import FittingScope, UnresolvedFit, _labels
from development_baseline_driver import (
    PreparedBaselineRow, PreparedParticipant, _fit, _mean, _new_model, _predict,
    baseline_recipes,
)
from ensemble_forecast_backend import SEEDS, _row_keys
from non_neural_forecast_backend import ContextFeatures, _canonical_scope, _schema
from temporal_hazard_backend import ContextSequence


@dataclass(frozen=True)
class FittedBaselineEnsemble:
    """Three actual fitted baseline models retained for a later admitted use.

    Exposing these in-memory objects is not source access or a final evaluation
    grant. The final refit composition performs only the fitting-row prediction
    recorded in its result. Subsequent consumers must preserve this frozen fit.
    """
    recipe: object
    fitting_scope: FittingScope
    models: tuple
    fit_reports: tuple


@dataclass(frozen=True)
class FrozenFittingOffsets:
    participant: object
    row_keys: tuple
    log_probabilities: np.ndarray
    recipe_id: str
    fitting_scope: FittingScope
    prediction_role: str = "training_rows_of_frozen_baseline_not_out_of_fold"
    calibration: str = "uncalibrated_before_EEG_fusion"
    seed_aggregation: str = "mean_of_three_probability_vectors"


def _prepare_rows(source, participants, scopes, summary_schema, sequence_schema, rate, check):
    expected = tuple(scopes)
    people = tuple(participants)
    if (any(not isinstance(person, PreparedParticipant) for person in people)
            or len(people) != len(expected)
            or {person.participant for person in people} != set(expected)):
        raise UnresolvedFit("complete_registered_fitting_cohort_required")
    lookup = {person.participant: person for person in people}
    boundaries, prepared = {}, {}
    for participant in expected:
        check()
        person = lookup[participant]
        rows = tuple(person.rows)
        if not rows or any(not isinstance(row, PreparedBaselineRow) for row in rows):
            raise UnresolvedFit("complete_prepared_fitting_rows_required")
        scope = scopes[participant]
        keys = _row_keys((row.identity, row.decision_time) for row in rows)
        closed, previous, last_time = set(), None, -np.inf
        for row in rows:
            check()
            identity = row.identity
            if (identity.source != source or identity.participant != participant
                    or (scope.allowed_series is not None
                        and identity.series_or_run not in scope.allowed_series)):
                raise ValueError("fitting row crosses its source, person or series role")
            group = (source, participant, identity.series_or_run)
            if boundaries.setdefault(identity.boundary_id, group) != group:
                raise ValueError("one boundary cannot name different recording groups")
            if identity.boundary_id != previous:
                if identity.boundary_id in closed:
                    raise ValueError("each reset boundary must form one contiguous row block")
                closed.add(identity.boundary_id)
                previous, last_time = identity.boundary_id, -np.inf
            if row.decision_time <= last_time:
                raise ValueError("decisions must increase within each reset boundary")
            last_time = row.decision_time
            if not isinstance(row.context, ContextFeatures) or not isinstance(row.sequence, ContextSequence):
                raise ValueError("existing summary and temporal context interfaces required")
            binding = dict(source=source, participant=participant,
                           boundary_id=identity.boundary_id, decision_time=row.decision_time)
            row.context.validated(schema=summary_schema, **binding)
            row.sequence.validated(schema=sequence_schema, sample_rate_hz=rate, **binding)
        _labels([row.observed_category for row in rows], [row.complete_bins for row in rows], len(rows))
        # Calibration partitions and scored follow-up do not select fitting rows.
        # Every admitted fitting point, including prefixes/guards, stays in scope.
        prepared[participant] = (rows, keys, scope)
    if source == "WAY":
        for rows, _, scope in prepared.values():
            scope.validate([row.identity for row in rows])
    else:
        scopes[expected[0]].validate([row.identity for person in expected for row in prepared[person][0]])
    return expected, prepared


def _fit_group(registry, recipe, fitting_people, prepared, summary_schema, sequence_schema,
               rate, check):
    rows = tuple(row for person in fitting_people for row in prepared[person][0])
    scope = prepared[fitting_people[0]][2]
    models, reports = [], []
    for seed in SEEDS:
        model = _new_model(recipe)
        report = _fit(model, rows, recipe=recipe, seed=seed, registry=registry,
            scope=scope, summary_schema=summary_schema, sequence_schema=sequence_schema,
            rate=rate, check=check)
        models.append(model)
        reports.append(report)
    ensemble = FittedBaselineEnsemble(recipe, scope, tuple(models), tuple(reports))
    offsets = []
    for person in fitting_people:
        person_rows, keys, _ = prepared[person]
        predictions = {seed: _predict(model, person_rows, recipe=recipe, check=check)
                       for seed, model in zip(SEEDS, models)}
        values = _mean(predictions, keys)
        values.setflags(write=False)
        offsets.append(FrozenFittingOffsets(person, keys, values, recipe.recipe_id, scope))
    return ensemble, tuple(offsets)


def _recipe(registry, source, selected_recipe_id):
    recipes = {recipe.recipe_id: recipe for recipe in baseline_recipes(registry, source=source)}
    if selected_recipe_id not in recipes:
        raise ValueError("one already selected registered source-specific recipe required")
    return recipes[selected_recipe_id]


def fit_fitting_fold_baseline(*, registry, fitting_scope, selected_recipe_id, participants,
                              summary_schema, sequence_schema, analysis_sample_rate_hz,
                              budget_check):
    """Fit a selected baseline within ONE existing EEG development fitting fold.

    All fitting rows receive frozen training offsets, including self-paced
    prefix/guard rows. No validation row is accepted or predicted by this call.
    The returned three-seed models must also remain frozen when the caller later
    evaluates its held-out development fold. Recipe selection on development
    data is not itself independent final-performance evidence.
    """
    if not callable(budget_check):
        raise ValueError("explicit allocation-owned budget callback required")
    if not isinstance(fitting_scope, FittingScope):
        raise ValueError("existing canonical development fitting scope required")
    scope = _canonical_scope(registry, fitting_scope, fitting_scope.source)
    recipe = _recipe(registry, scope.source, selected_recipe_id)
    summary_schema, sequence_schema = _schema(summary_schema), _schema(sequence_schema)
    scopes = {person: scope for person in scope.allowed_participants}
    people, prepared = _prepare_rows(scope.source, participants, scopes, summary_schema,
        sequence_schema, analysis_sample_rate_hz, budget_check)
    ensemble, offsets = _fit_group(registry, recipe, people, prepared, summary_schema,
        sequence_schema, analysis_sample_rate_hz, budget_check)
    budget_check()
    return {"source": scope.source, "stage": "development_baseline_inside_EEG_fitting_fold",
        "completed": True, "selected_recipe_id": recipe.recipe_id, "seeds": SEEDS,
        "fits_completed": len(SEEDS), "ensemble": ensemble, "fitting_offsets": offsets,
        "offsets_are_out_of_fold": False, "calibration_performed": False,
        "validation_evaluation_performed": False, "source_admission_established": False}


def fit_final_baselines(*, registry, source, selected_recipe_id, participants,
                       summary_schema, sequence_schema, analysis_sample_rate_hz,
                       budget_check):
    """Fit the one frozen recipe for the complete registered final fitting cohort.

    The caller establishes that ``selected_recipe_id`` was selected before EEG
    candidate scores and frozen for this stage. This API accepts no selection
    scores and performs no search or cost tie-break. Model-cost conventions from
    development selection are unchanged; no final latency measurement is claimed.
    The provided rows are fitting rows only. No held-out person or series is
    accepted, even to produce predictions.
    """
    if not callable(budget_check):
        raise ValueError("explicit allocation-owned budget callback required")
    recipe = _recipe(registry, source, selected_recipe_id)
    summary_schema, sequence_schema = _schema(summary_schema), _schema(sequence_schema)
    if source == "WAY":
        scopes = {person: FittingScope.final_refit(registry, source=source, participant=person)
                  for person in registry["folds"]["WAY_participants"]}
    else:
        scope = FittingScope.final_refit(registry, source=source)
        scopes = {person: scope for person in scope.allowed_participants}
    people, prepared = _prepare_rows(source, participants, scopes, summary_schema,
        sequence_schema, analysis_sample_rate_hz, budget_check)
    batches = ((person,) for person in people) if source == "WAY" else (people,)
    ensembles, offsets = [], []
    for fitting_people in batches:
        ensemble, group_offsets = _fit_group(registry, recipe, fitting_people, prepared,
            summary_schema, sequence_schema, analysis_sample_rate_hz, budget_check)
        offsets.extend(group_offsets)
        ensembles.append(ensemble)
    budget_check()
    return {"source": source, "stage": "final_baseline_refit_on_development_people",
        "completed": True, "selected_recipe_id": recipe.recipe_id, "seeds": SEEDS,
        "fits_completed": len(ensembles) * len(SEEDS), "ensembles": tuple(ensembles),
        "fitting_offsets": tuple(offsets), "offsets_are_out_of_fold": False,
        "calibration_performed": False, "final_evaluation_performed": False,
        "source_admission_established": False, "empirical_method_freeze_established": False}
