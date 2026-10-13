"""Actual fitted non-neural ensemble costs for EP19 development selection.

Caller-selected, label-free fitting-fold probes are copied once and reused in
the same order for all recipes. Declared identities and availability are checked;
this does not certify that the caller supplied authentic fitting data or valid
sensor clocks/units. There is no source I/O, fitting, calibration or scoring.

Counts cover retained fitted inference coordinates across all three seeds:
linear coefficients/intercepts, tree split feature choices and thresholds plus
leaf scores, or every neural-network parameter, and fitted scaler means/scales.
Fixed hyperparameters, tree-index bookkeeping, unused internal-node values,
optimizer state and four-parameter post-ensemble calibration are excluded.
These are explicit counting units, not equivalent statistical degrees of freedom.

Latency is the observed nearest-rank p95 for one prepared-feature decision:
three actual predictions (including backend standardization) followed by the
existing probability mean. Resets, warmup, budget checks and source feature
construction are outside timing. This is not end-to-end latency qualification.
"""
from dataclasses import dataclass
from numbers import Real
from time import perf_counter_ns
from typing import Mapping

import numpy as np

from boosted_hazard_backend import BoostedHazardModel
from classical_forecast_backend import FittingScope, GroupIdentity, UnresolvedFit, exact_samples
from development_baseline_driver import SelectionCost, baseline_recipes
from ensemble_forecast_backend import SEEDS, mean_seed_log_probabilities
from non_neural_forecast_backend import ContextFeatures, LogisticHazardModel, _canonical_scope, _schema
from temporal_hazard_backend import ContextSequence, TemporalHazardModel


PARAMETER_COUNT_BASIS = "all_three_seed_fitted_inference_coordinates_and_scaler_mean_scale_v1"
LATENCY_BASIS = "CPU_prepared_feature_three_seed_prediction_and_probability_mean_nearest_rank_p95_v1"
BACKENDS = {
    "regularized_logistic_discrete_hazard": LogisticHazardModel,
    "gradient_boosted_discrete_hazard": BoostedHazardModel,
    "small_past_only_temporal_convolutional_discrete_hazard": TemporalHazardModel,
}


@dataclass(frozen=True)
class BaselineInferenceProbe:
    identity: GroupIdentity
    decision_time: float
    context: ContextFeatures
    sequence: ContextSequence


def _owned(value):
    if np.iscomplexobj(value):
        raise ValueError("real prepared probe values required")
    copied = np.array(value, dtype=np.float64, copy=True)
    copied.flags.writeable = False
    return copied


def _scaler_count(scaler, size):
    mean, scale = np.asarray(scaler.mean), np.asarray(scaler.scale)
    if (mean.shape != (size,) or scale.shape != (size,) or np.iscomplexobj(mean)
            or np.iscomplexobj(scale) or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise UnresolvedFit("complete_finite_fitted_scaler_required_for_cost")
    return int(mean.size + scale.size)


def _parameter_components(model, check):
    check()
    if type(model) is LogisticHazardModel:
        size = len(model.feature_schema)
        fit = model.fit_result
        coefficients, intercept = np.asarray(fit.coefficients), np.asarray(fit.intercept)
        if (not fit.converged or coefficients.shape != (size, 18) or intercept.shape != (18,)
                or any(np.iscomplexobj(value) or not np.all(np.isfinite(value))
                    for value in (coefficients, intercept))):
            raise UnresolvedFit("complete_finite_logistic_fit_required_for_cost")
        return dict(coefficients=int(coefficients.size), intercepts=int(intercept.size),
            scaler_mean_and_scale=_scaler_count(model.standardizer, size))
    if type(model) is BoostedHazardModel:
        fit, size = model.fit_result, len(model.feature_schema)
        if not fit.completed or len(fit.heads) != 18 or fit.feature_count != size:
            raise UnresolvedFit("complete_eighteen_head_boosted_fit_required_for_cost")
        splits = leaves = 0
        for head in fit.heads:
            check()
            if (head.init != "zero" or head.learning_rate != model.recipe.learning_rate
                    or head.n_estimators_ != model.recipe.trees
                    or head.estimators_.shape != (model.recipe.trees, 1)):
                raise UnresolvedFit("complete_registered_boosted_trees_required_for_cost")
            for estimator in head.estimators_[:, 0]:
                tree = estimator.tree_
                internal = tree.children_left != -1
                if (np.any((tree.children_right != -1) != internal)
                        or tree.value.shape != (tree.node_count, 1, 1)
                        or np.any(tree.feature[internal] < 0) or np.any(tree.feature[internal] >= size)
                        or not np.all(np.isfinite(tree.threshold[internal]))
                        or not np.all(np.isfinite(tree.value[~internal]))):
                    raise UnresolvedFit("finite_scalar_regression_tree_parameters_required_for_cost")
                splits += int(np.count_nonzero(internal))
                leaves += int(tree.value[~internal].size)
        return dict(split_feature_choices=splits, split_thresholds=splits, leaf_scores=leaves,
            scaler_mean_and_scale=_scaler_count(model.standardizer, size))
    if type(model) is TemporalHazardModel:
        if any(module.training for module in model.network.modules()):
            raise UnresolvedFit("fitted_temporal_network_must_already_be_in_evaluation_mode")
        count = 0
        for parameter in model.network.parameters():
            check()
            if parameter.device.type != "cpu" or not np.all(np.isfinite(parameter.detach().numpy())):
                raise UnresolvedFit("finite_CPU_temporal_parameters_required_for_cost")
            # Training has frozen these tensors; requires_grad is not a count filter.
            count += int(parameter.numel())
        if count <= 0:
            raise UnresolvedFit("actual_fitted_temporal_parameters_required_for_cost")
        return dict(network_parameters=count,
            sequence_scaler_mean_and_scale=_scaler_count(model.sequence_standardizer, len(model.sequence_schema)),
            summary_scaler_mean_and_scale=_scaler_count(model.summary_standardizer, len(model.summary_schema)))
    raise ValueError("only the three existing non-neural backends are supported")


class BaselineEnsembleCostMeasurer:
    """Callable supplied unchanged to run_development_baseline_selection.

    The caller prospectively selects finite warmup and measured pass counts,
    complete prepared probes and an allocation-owned budget callback. Each pass
    replays the same probes once. Every probe is role-checked but no labels are
    accepted. A probe panel may be a bounded fitting-data subset; it is not an
    empirical risk-set sample, scoring panel or full-fold fit qualification.
    `measurements` retains completed count breakdowns and every measured latency,
    without predictions. The existing driver independently resets for scoring.
    """
    parameter_count_basis = PARAMETER_COUNT_BASIS
    latency_basis = LATENCY_BASIS

    def __init__(self, *, registry, probes_by_scope, analysis_sample_rate_hz,
                 warmup_passes, measurement_passes, budget_check):
        if (type(warmup_passes) is not int or warmup_passes < 0
                or type(measurement_passes) is not int or measurement_passes < 1
                or not callable(budget_check)):
            raise ValueError("explicit finite pass counts and allocation budget callback required")
        if (isinstance(analysis_sample_rate_hz, bool) or not isinstance(analysis_sample_rate_hz, Real)
                or not np.isfinite(analysis_sample_rate_hz)):
            raise ValueError("explicit finite analysis sample rate required")
        exact_samples(2., analysis_sample_rate_hz)
        if not isinstance(probes_by_scope, Mapping) or not probes_by_scope:
            raise ValueError("explicit nonempty fitting-scope probe mapping required")
        sources = {scope.source for scope in probes_by_scope if isinstance(scope, FittingScope)}
        if len(sources) != 1 or any(not isinstance(scope, FittingScope) for scope in probes_by_scope):
            raise ValueError("one source's canonical fitting scopes are required")
        self._recipes = {recipe.recipe_id: recipe for recipe in baseline_recipes(registry, source=sources.pop())}
        self._probes = {}
        self._schemas = None
        for scope, supplied in probes_by_scope.items():
            budget_check()
            _canonical_scope(registry, scope, scope.source)
            if not isinstance(supplied, tuple) or not supplied:
                raise ValueError("each scope requires an explicit finite nonempty probe tuple")
            owned, groups, closed, previous, last = [], {}, set(), None, -np.inf
            for probe in supplied:
                budget_check()
                if (type(probe) is not BaselineInferenceProbe or not isinstance(probe.identity, GroupIdentity)
                        or not isinstance(probe.context, ContextFeatures)
                        or not isinstance(probe.sequence, ContextSequence)):
                    raise ValueError("explicit label-free baseline probes required")
                identity = probe.identity
                if (identity.source != scope.source or identity.participant not in scope.allowed_participants
                        or identity.series_or_run is None
                        or (scope.allowed_series is not None and identity.series_or_run not in scope.allowed_series)
                        or not isinstance(identity.boundary_id, str) or not identity.boundary_id):
                    raise ValueError("cost probes must stay inside their declared fitting fold")
                group = (identity.source, identity.participant, identity.series_or_run)
                if groups.setdefault(identity.boundary_id, group) != group:
                    raise ValueError("one probe boundary cannot name different recording groups")
                if identity.boundary_id != previous:
                    if identity.boundary_id in closed:
                        raise ValueError("probe boundary blocks must be contiguous")
                    closed.add(identity.boundary_id)
                    previous, last = identity.boundary_id, -np.inf
                if (isinstance(probe.decision_time, bool) or not isinstance(probe.decision_time, Real)
                        or not np.isfinite(probe.decision_time) or probe.decision_time <= last):
                    raise ValueError("finite increasing decision times required within each probe boundary")
                last = probe.decision_time
                schemas = (_schema(probe.context.feature_schema), _schema(probe.sequence.feature_schema))
                if self._schemas is not None and schemas != self._schemas:
                    raise ValueError("all recipes and scopes require the same source feature schemas")
                self._schemas = schemas
                binding = dict(source=scope.source, participant=identity.participant,
                    boundary_id=identity.boundary_id, decision_time=probe.decision_time)
                probe.context.validated(schema=schemas[0], **binding)
                probe.sequence.validated(schema=schemas[1], sample_rate_hz=analysis_sample_rate_hz, **binding)
                context = ContextFeatures(_owned(probe.context.values), schemas[0], scope.source,
                    identity.participant, identity.boundary_id, float(probe.context.available_through))
                sequence = ContextSequence(_owned(probe.sequence.values), schemas[1], scope.source,
                    identity.participant, tuple(probe.sequence.boundary_ids),
                    _owned(probe.sequence.sample_times), _owned(probe.sequence.available_through))
                owned.append(BaselineInferenceProbe(identity, float(probe.decision_time), context, sequence))
            self._probes[scope] = tuple(owned)
        self._rate = analysis_sample_rate_hz
        self._warmup, self._passes, self._check = warmup_passes, measurement_passes, budget_check
        self.measurements = []

    def __call__(self, *, models, recipe, fitting_scope):
        self._check()
        if (recipe.recipe_id not in self._recipes or recipe != self._recipes[recipe.recipe_id]
                or fitting_scope not in self._probes):
            raise ValueError("the fixed recipe and fitting-scope probe panel are required")
        models = tuple(models)
        expected_type = BACKENDS[recipe.family]
        if (len(models) != 3 or len({id(model) for model in models}) != 3
                or any(type(model) is not expected_type or getattr(model, "fitted", False) is not True
                       or getattr(model, "seed", None) != seed or getattr(model, "recipe", None) != recipe
                       or getattr(model, "fitting_scope", None) != fitting_scope
                       for seed, model in zip(SEEDS, models))):
            raise UnresolvedFit("three_distinct_fitted_models_with_exact_seed_recipe_scope_required")
        temporal = expected_type is TemporalHazardModel
        for model in models:
            schema = model.summary_schema if temporal else model.feature_schema
            if (tuple(schema) != self._schemas[0] or (temporal and
                    (tuple(model.sequence_schema) != self._schemas[1] or model.analysis_sample_rate_hz != self._rate))):
                raise ValueError("actual fitted schemas and analysis grid must match fixed probes")
        probes = self._probes[fitting_scope]
        first_boundary = probes[0].identity.boundary_id
        durations = []
        try:
            components = [_parameter_components(model, self._check) for model in models]
            for pass_index in range(self._warmup + self._passes):
                boundary = None
                for probe in probes:
                    self._check()
                    if probe.identity.boundary_id != boundary:
                        boundary = probe.identity.boundary_id
                        for model in models:
                            model.reset(boundary_id=boundary)
                    keys = ((probe.identity, probe.decision_time),)
                    seed_keys = {seed: keys for seed in SEEDS}
                    measured = pass_index >= self._warmup
                    start = perf_counter_ns() if measured else None
                    predictions = {}
                    for seed, model in zip(SEEDS, models):
                        values = (model.predict_log_probabilities(probe.sequence, probe.context,
                                    decision_time=probe.decision_time) if temporal else
                                  model.predict_log_probabilities(probe.context, decision_time=probe.decision_time))
                        predictions[seed] = np.asarray(values)[None, :]
                    mean_seed_log_probabilities(predictions, seed_row_keys=seed_keys, expected_row_keys=keys)
                    if measured:
                        elapsed = perf_counter_ns() - start
                        if elapsed <= 0:
                            raise UnresolvedFit("positive_observed_ensemble_latency_required")
                        durations.append(elapsed)
                    self._check()
        finally:
            # No fitted values are updated. Even failed/cancelled measurement
            # leaves every admitted model reset before the driver's next use.
            for model in models:
                model.reset(boundary_id=first_boundary)
        ordered = sorted(durations)
        rank = (95 * len(ordered) + 99) // 100
        parameter_count = sum(sum(item.values()) for item in components)
        cost = SelectionCost(parameter_count, ordered[rank - 1] / 1000000000.)
        self.measurements.append(dict(source=recipe.source, fold_id=fitting_scope.fold_id,
            fitting_participants=fitting_scope.allowed_participants, recipe_id=recipe.recipe_id,
            seeds=SEEDS, per_seed_components=tuple(components), parameter_count=parameter_count,
            probe_count=len(probes), warmup_passes=self._warmup, measurement_passes=self._passes,
            latency_nanoseconds=tuple(durations), p95_observed_rank_one_based=rank,
            latency_seconds=cost.latency_seconds, parameter_count_basis=self.parameter_count_basis,
            latency_basis=self.latency_basis, probes_shared_across_recipes=True,
            latency_excludes="reset_warmup_budget_checks_source_feature_construction_calibration_and_IO",
            authentic_probe_source_admission_established=False,
            end_to_end_latency_qualified=False))
        return cost
