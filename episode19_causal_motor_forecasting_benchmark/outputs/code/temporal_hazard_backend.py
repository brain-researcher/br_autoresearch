"""Prospective EP19 small non-neural TCN; no source I/O or empirical grant.

The caller supplies a complete regular two-second grid [t-2,t), prepared
summaries and admitted source semantics. Left zeros define each convolution
outside that complete window; they never replace missing samples inside it.
Training is the fixed 100-epoch CPU/float32 schedule, not a convergence search.
Only canonical development folds are supported; final refits remain separate.
This array-only fitter materializes the entire fitting fold's windows. A
production reader and memory qualification are still required; it is not a
streaming large-data fitting implementation.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from boosted_hazard_backend import conditional_hazard_targets
from classical_forecast_backend import (
    FittingStandardizer, GroupIdentity, UnresolvedFit, exact_samples,
    hazard_log_probabilities,
)
from non_neural_forecast_backend import ContextFeatures, _canonical_scope, _schema

FAMILY = "small_past_only_temporal_convolutional_discrete_hazard"
SEEDS = (1901, 1902, 1903)
DILATIONS = (1, 2, 4, 8, 16)
EPOCHS, BATCH_SIZE = 100, 128


@dataclass(frozen=True)
class TemporalHazardRecipe:
    recipe_id: str
    source: str
    weight_decay: float
    seeds: tuple[int, ...] = SEEDS
    channels: int = 16
    family: str = FAMILY


def temporal_hazard_recipes(registry, *, source):
    definition = registry["non_neural_baselines"]
    expected = [{"channels": 16, "weight_decay": value} for value in (.001, .01, .1)]
    training = {"neural_optimizer": "AdamW", "neural_learning_rate": .001,
        "neural_epochs": EPOCHS, "batch_examples": BATCH_SIZE, "neural_dropout": .25,
        "gradient_global_norm_maximum": 1., "class_rebalancing": False}
    if (source not in ("WAY", "self_paced") or source not in definition["sources"]
            or definition["families"][FAMILY] != expected or tuple(registry["seeds"]) != SEEDS
            or definition["history_seconds"] != 2. or definition["hazard_bins"] != 18
            or any(registry["training"][key] != value for key, value in training.items())
            or registry["reference_features"]["TCN_kernel_samples"] != 3
            or tuple(registry["reference_features"]["TCN_dilations"]) != DILATIONS
            or registry["reference_features"]["TCN_convolutions_per_block"] != 2):
        raise ValueError("unchanged temporal recipes, seed bank and fixed training schedule required")
    return tuple(TemporalHazardRecipe(f"{source}.{FAMILY}.c{index:02d}", source,
                    setting["weight_decay"]) for index, setting in enumerate(expected))


@dataclass(frozen=True)
class ContextSequence:
    """Prepared values [time, channel]; all availability facts belong to caller.

    Channel schemas are source-specific. Per-row features must be available by
    that row's time, not merely by the later forecast decision. This declaration
    cannot establish physical clocks, sensor units or upstream source access.
    """
    values: Any
    feature_schema: tuple[str, ...]
    source: str
    participant: Any
    boundary_ids: tuple[str, ...]
    sample_times: Any
    available_through: Any

    def validated(self, *, schema, source, participant, boundary_id, decision_time, sample_rate_hz):
        count = exact_samples(2., sample_rate_hz)
        if self.source != source or _schema(self.feature_schema) != schema:
            raise ValueError("sequence source and feature order must match the fitted schema")
        if self.participant is None or self.participant != participant:
            raise ValueError("sequence participant must match the example or fitted WAY person")
        if not boundary_id or tuple(self.boundary_ids) != (boundary_id,) * count:
            raise ValueError("every sequence sample must belong to the same reset boundary")
        if any(np.iscomplexobj(value) for value in (self.values, self.sample_times, self.available_through)):
            raise ValueError("sequence values, times and availability must be real, not complex")
        values = np.asarray(self.values, dtype=np.float64)
        times = np.asarray(self.sample_times, dtype=np.float64)
        available = np.asarray(self.available_through, dtype=np.float64)
        if (values.shape != (count, len(schema)) or not np.all(np.isfinite(values))
                or times.shape != (count,) or not np.all(np.isfinite(times))
                or available.shape != (count,) or not np.all(np.isfinite(available))
                or not np.isfinite(decision_time)):
            raise ValueError("complete finite two-second sequence required; no history/value imputation")
        wanted = decision_time - np.arange(count, 0, -1) / sample_rate_hz
        tolerance = max(1e-10, abs(decision_time) * np.finfo(float).eps * 8)
        if (np.any(times >= decision_time) or np.any(np.diff(times) <= 0)
                or not np.allclose(times, wanted, atol=tolerance, rtol=0)):
            raise ValueError("sequence must be the exact regular past grid [t-2,t)")
        if np.any(available > times):
            raise ValueError("each sequence feature must be available by its sample time")
        return values


@dataclass(frozen=True)
class TemporalFittingExample:
    sequence: ContextSequence
    summary: ContextFeatures
    decision_time: float
    identity: GroupIdentity
    observed_category: int
    complete_bins: int


class _CausalResidual(nn.Module):
    def __init__(self, input_channels, dilation):
        super().__init__()
        self.left = 2 * dilation
        settings = {"kernel_size": 3, "dilation": dilation, "padding": 0,
                    "device": "cpu", "dtype": torch.float32}
        self.first = nn.Conv1d(input_channels, 16, **settings)
        self.second = nn.Conv1d(16, 16, **settings)
        self.shortcut = (nn.Identity() if input_channels == 16 else
                         nn.Conv1d(input_channels, 16, 1, device="cpu", dtype=torch.float32))
        self.dropout = nn.Dropout(.25)

    def forward(self, values):
        hidden = self.dropout(F.relu(self.first(F.pad(values, (self.left, 0)))))
        hidden = self.dropout(F.relu(self.second(F.pad(hidden, (self.left, 0)))))
        return F.relu(hidden + self.shortcut(values))


class TemporalHazardNetwork(nn.Module):
    def __init__(self, sequence_channels, summary_features):
        super().__init__()
        self.blocks = nn.Sequential(*[_CausalResidual(sequence_channels if index == 0 else 16, dilation)
                                     for index, dilation in enumerate(DILATIONS)])
        self.readout = nn.Linear(16 + summary_features, 18, device="cpu", dtype=torch.float32)

    def hidden(self, sequence):
        return self.blocks(sequence)

    def forward(self, sequence, summary):
        return self.readout(torch.cat((self.hidden(sequence).mean(dim=2), summary), dim=1))


def censored_hazard_losses(logits, risk, events):
    """Per-example conditional NLL; unobserved intervals contribute zero."""
    if (logits.ndim != 2 or logits.shape[1] != 18 or risk.shape != logits.shape
            or events.shape != logits.shape or risk.dtype != torch.bool or events.dtype != torch.bool):
        raise ValueError("18 logits and matching boolean observed/event masks required")
    if not bool(torch.isfinite(logits).all()):
        raise UnresolvedFit("nonfinite_temporal_logits")
    return (torch.where(events, F.softplus(-logits), F.softplus(logits)) * risk).sum(dim=1)


def _minibatch_objective(logits, risk, events, global_weights, total_examples):
    # Uniform sampling without replacement; do NOT renormalize within a batch.
    return total_examples / len(logits) * torch.sum(global_weights * censored_hazard_losses(logits, risk, events))


def _float32(value):
    result = torch.tensor(value, dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise UnresolvedFit("nonfinite_temporal_float32_input_or_scale")
    return result


def _train_fixed_schedule(sequence, summary, risk, events, weights, *, recipe, seed, budget_check):
    if torch.get_num_threads() != 1:
        raise ValueError("caller must configure one CPU numerical thread within its allocation")
    sequence = _float32(sequence).transpose(1, 2).contiguous()
    summary, weights = _float32(summary), _float32(weights)
    if not bool((weights > 0).all()):
        raise UnresolvedFit("positive_example_weight_underflowed_in_float32")
    risk, events = torch.tensor(risk, dtype=torch.bool), torch.tensor(events, dtype=torch.bool)
    rows = len(sequence)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    # Native initialization and dropout share this seed; permutation has its own
    # fixed CPU generator. Neither ambient CPU state nor CUDA RNG is changed.
    with torch.random.fork_rng(devices=[]):
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.random.default_generator.manual_seed(seed)
            permutation = torch.Generator(device="cpu").manual_seed(seed)
            network = TemporalHazardNetwork(sequence.shape[1], summary.shape[1])
            optimizer = torch.optim.AdamW(network.parameters(), lr=.001, betas=(.9, .999),
                eps=1e-8, weight_decay=recipe.weight_decay, amsgrad=False, maximize=False,
                foreach=False, capturable=False, differentiable=False, fused=False)
            network.train()
            steps = 0
            for _ in range(EPOCHS):
                budget_check()
                order = torch.randperm(rows, generator=permutation)
                for start in range(0, rows, BATCH_SIZE):
                    budget_check()
                    indices = order[start:start + BATCH_SIZE]
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(sequence[indices], summary[indices])
                    objective = _minibatch_objective(logits, risk[indices], events[indices], weights[indices], rows)
                    if not bool(torch.isfinite(objective)):
                        raise UnresolvedFit("nonfinite_temporal_objective")
                    objective.backward()
                    try:
                        nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                    except RuntimeError as error:
                        raise UnresolvedFit("nonfinite_temporal_gradient") from error
                    budget_check()  # Required allocation/deadline check immediately before mutation.
                    optimizer.step()
                    steps += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                        raise UnresolvedFit("nonfinite_temporal_parameter")
            network.eval()
            objective = 0.
            with torch.no_grad():
                for start in range(0, rows, BATCH_SIZE):
                    budget_check()
                    stop = start + BATCH_SIZE
                    losses = censored_hazard_losses(network(sequence[start:stop], summary[start:stop]),
                                                    risk[start:stop], events[start:stop])
                    objective += float(torch.sum(weights[start:stop] * losses))
            if not np.isfinite(objective):
                raise UnresolvedFit("nonfinite_temporal_final_objective")
            for parameter in network.parameters():
                parameter.grad = None
                parameter.requires_grad_(False)
            return network, steps, objective
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)


class TemporalHazardModel:
    """Frozen development/final-refit baseline; no evaluation or source access granted.

    The caller's budget_check must raise when its admitted allocation or deadline
    no longer permits fitting. Its exceptions propagate without a retry or fit.
    """
    def fit(self, batches, *, recipe, seed, registry, fitting_scope, sequence_schema,
            summary_schema, analysis_sample_rate_hz, budget_check):
        self.fitted = False
        recipes = {item.recipe_id: item for item in temporal_hazard_recipes(registry, source=recipe.source)}
        if recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id] or type(seed) is not int or seed not in SEEDS:
            raise ValueError("exact registered temporal recipe and seed required")
        scope = _canonical_scope(registry, fitting_scope, recipe.source, allow_final_refit=True)
        sequence_schema, summary_schema = _schema(sequence_schema), _schema(summary_schema)
        count = exact_samples(2., analysis_sample_rate_hz)
        if not callable(budget_check):
            raise ValueError("caller allocation/deadline callback is required")
        budget_check()
        examples = []
        for batch in batches:
            budget_check()
            examples.extend(batch)
        weights = scope.validate([example.identity for example in examples])
        sequences, summaries = [], []
        for example in examples:
            budget_check()
            binding = dict(source=recipe.source, participant=example.identity.participant,
                boundary_id=example.identity.boundary_id, decision_time=example.decision_time)
            if np.iscomplexobj(example.summary.values):
                raise ValueError("prepared summaries must be real, not complex")
            sequences.append(example.sequence.validated(schema=sequence_schema,
                sample_rate_hz=analysis_sample_rate_hz, **binding))
            summaries.append(example.summary.validated(schema=summary_schema, **binding))
        risk, events = conditional_hazard_targets([example.observed_category for example in examples],
            [example.complete_bins for example in examples], len(examples))
        absent = np.flatnonzero(~risk.any(axis=0))
        if len(absent):
            raise UnresolvedFit(f"no_observed_conditional_intervals_heads_{absent.tolist()}")
        single_class = [index for index in range(18) if np.unique(events[risk[:, index], index]).size == 1]
        sequences, summaries = np.asarray(sequences), np.asarray(summaries)
        budget_check()
        flat = sequences.reshape(-1, len(sequence_schema))
        self.sequence_standardizer = FittingStandardizer().fit(flat, np.repeat(weights / count, count))
        self.summary_standardizer = FittingStandardizer().fit(summaries, weights)
        for standardizer in (self.sequence_standardizer, self.summary_standardizer):
            if (not np.all(np.isfinite(standardizer.mean)) or not np.all(np.isfinite(standardizer.scale))
                    or np.any(standardizer.scale <= 0)):
                raise UnresolvedFit("nonfinite_temporal_fitting_scale")
        prepared_sequence = self.sequence_standardizer.transform(flat).reshape(sequences.shape)
        prepared_summary = self.summary_standardizer.transform(summaries)
        self.network, steps, objective = _train_fixed_schedule(prepared_sequence, prepared_summary,
            risk, events, weights, recipe=recipe, seed=seed, budget_check=budget_check)
        self.recipe, self.seed, self.fitting_scope = recipe, seed, scope
        self.sequence_schema, self.summary_schema = sequence_schema, summary_schema
        self.analysis_sample_rate_hz = analysis_sample_rate_hz
        self.boundary_id, self.last_decision_time = None, -np.inf
        self.fitted = True
        return {"completed": True, "reason": "fixed_100_epoch_schedule_complete", "epochs": EPOCHS,
            "optimizer_steps": steps, "objective": objective, "single_class_observed_heads": single_class,
            "recipe_id": recipe.recipe_id, "seed": seed, "source": recipe.source, "fold_id": scope.fold_id,
            "examples": len(examples), "sequence_schema": list(sequence_schema), "summary_schema": list(summary_schema),
            "analysis_sample_rate_hz": analysis_sample_rate_hz, "history_samples": count,
            "sequence_interval": "[t-2,t)", "boundary_operator": "causal_left_zero_extension_of_complete_window",
            "source_admission_established_by_backend": False}

    def reset(self, *, boundary_id):
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("an explicit nonempty reset boundary is required")
        self.boundary_id, self.last_decision_time = boundary_id, -np.inf

    def predict_log_probabilities(self, sequence, summary, *, decision_time):
        if not getattr(self, "fitted", False):
            raise UnresolvedFit("an incomplete temporal baseline cannot supply EEG offsets")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        if torch.get_num_threads() != 1:
            raise ValueError("caller must configure one CPU numerical thread within its allocation")
        participant = self.fitting_scope.allowed_participants[0] if self.recipe.source == "WAY" else sequence.participant
        binding = dict(source=self.recipe.source, participant=participant,
                       boundary_id=self.boundary_id, decision_time=decision_time)
        if np.iscomplexobj(summary.values):
            raise ValueError("prepared summaries must be real, not complex")
        values = sequence.validated(schema=self.sequence_schema, sample_rate_hz=self.analysis_sample_rate_hz, **binding)
        summary_values = summary.validated(schema=self.summary_schema, **binding)
        x = _float32(self.sequence_standardizer.transform(values)[None, :, :]).transpose(1, 2)
        z = _float32(self.summary_standardizer.transform(summary_values[None, :]))
        with torch.no_grad():
            logits = self.network(x, z).numpy()
        if not np.all(np.isfinite(logits)):
            raise UnresolvedFit("nonfinite_temporal_prediction")
        logp = hazard_log_probabilities(logits)[0]
        self.last_decision_time = decision_time
        return logp
