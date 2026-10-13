"""EP19 development GRU correction over caller-admitted, already scaled EEG.

No source I/O, clocks, resampling, labels, voltage scale or masks are inferred.
PastWindow/FittingExample and the existing grouped-fold interface are reused.
The fixed non-neural log probabilities are offsets, never trainable parameters.
This array-only backend materializes fitting windows; production readers and
memory qualification remain separate. It does not authorize empirical scoring.
"""
from typing import Any, Mapping

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from classical_forecast_backend import (
    FittingScope, UnresolvedFit, _labels, exact_samples,
    normalized_log_probabilities, canonical_reference_scope,
)
from model_recipes import reference_recipes

FAMILY = "small_one_directional_GRU"
SEEDS = (1901, 1902, 1903)
EPOCHS, BATCH_SIZE = 100, 128
GRU_SUMMARY = "last_hidden_per_slot"
GRU_INITIAL_STATE = "zero_per_window_and_factorial_slot"
CHANNELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
            "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")


def gru_recipes(registry):
    grid, training = registry["reference_grid"], registry["training"]
    expected = {"output_categories": 19, "loss": "existing_censoring_aware_19_category_NLL",
        "class_rebalancing": False, "neural_optimizer": "AdamW", "neural_learning_rate": .001,
        "neural_epochs": EPOCHS, "batch_examples": BATCH_SIZE, "neural_dropout": .25,
        "gradient_global_norm_maximum": 1.,
        "participant_weighting": "equal_total_per_fitting_participant_then_equal_eligible_grid_points_within_person"}
    if (grid["history_seconds"] != [1., 2.] or grid["regularization"] != [.001, .01, .1]
            or grid["family_variants"][FAMILY] != [{"hidden_units": 16}, {"hidden_units": 32}]
            or tuple(registry["seeds"]) != SEEDS
            or registry["reference_features"]["GRU_layers"] != 1
            or registry["reference_features"]["bidirectional_or_unmasked_cache"] is not False
            or any(training[key] != value for key, value in expected.items())):
        raise ValueError("unchanged GRU recipes, seed bank and fixed training schedule required")
    return tuple(recipe for recipe in reference_recipes(registry) if recipe.family == FAMILY)


def _offsets(value, rows):
    if np.iscomplexobj(value):
        raise ValueError("fixed baseline offsets must be real")
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (rows, 19) or not np.all(np.isfinite(result))
            or not np.allclose(logsumexp(result, axis=1), 0., atol=1e-12, rtol=0)):
        raise ValueError("finite normalized fitting-fold baseline log probabilities required")
    return result


def _tensor(value, *, dtype):
    result = torch.tensor(value, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise UnresolvedFit("nonfinite_GRU_input_or_float32_overflow")
    return result


class GRUCorrectionNetwork(nn.Module):
    """One shared encoder and two last-hidden summaries, with no hidden cache.

    Original recurrence carries the older state into the recent slot. The
    factorial adapter resets the recent slot to zero *after* caller masking.
    Both use precisely the same parameters and concatenated-summary head.
    """
    def __init__(self, hidden_units, *, recent_samples, adapter):
        super().__init__()
        if hidden_units not in (16, 32) or type(recent_samples) is not int or recent_samples < 1:
            raise ValueError("registered hidden size and positive recent sample count required")
        if adapter not in ("original", "factorial"):
            raise ValueError("explicit original/factorial adapter required")
        self.recent_samples, self.adapter = recent_samples, adapter
        self.encoder = nn.GRU(29, hidden_units, num_layers=1, batch_first=True,
            bidirectional=False, dropout=0., device="cpu", dtype=torch.float32)
        self.dropout = nn.Dropout(.25)
        self.readout = nn.Linear(2 * hidden_units, 19, device="cpu", dtype=torch.float32)

    def summaries(self, sequence):
        if sequence.ndim != 3 or sequence.shape[2] != 29 or sequence.shape[1] <= self.recent_samples:
            raise ValueError("complete nonempty older and recent EEG slots required")
        cut = sequence.shape[1] - self.recent_samples
        _, older = self.encoder(sequence[:, :cut])  # Native omitted h0 is all zeros.
        _, recent = self.encoder(sequence[:, cut:], older if self.adapter == "original" else None)
        return torch.cat((older[0], recent[0]), dim=1)

    def forward(self, sequence):
        return self.readout(self.dropout(self.summaries(sequence)))


def censored_category_losses(logits, category, complete):
    """Differentiable existing19-category NLL, including zero-bin censoring."""
    if (logits.ndim != 2 or logits.shape[1] != 19 or category.shape != (len(logits),)
            or complete.shape != category.shape or category.dtype != torch.int64
            or complete.dtype != torch.int64 or bool(((category < -1) | (category > 18)).any())
            or bool(((complete < 0) | (complete > 18)).any())):
        raise ValueError("19 logits and integer observed categories/complete bins required")
    if not bool(torch.isfinite(logits).all()):
        raise UnresolvedFit("nonfinite_GRU_logits")
    normalizer = torch.logsumexp(logits, dim=1)
    observed = normalizer - logits.gather(1, category.clamp(min=0)[:, None])[:, 0]
    allowed = torch.arange(19, device=logits.device)[None, :] >= complete[:, None]
    censored = normalizer - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)
    return torch.where(category >= 0, observed, censored)


def _minibatch_objective(correction, offsets, category, complete, weights, total_examples):
    # Keep fixed log offsets and natural prevalence weights in float64; the
    # encoder/gradients remain float32. Never renormalize weights per minibatch.
    logits = correction.to(dtype=torch.float64) + offsets
    return total_examples / len(logits) * torch.sum(
        weights * censored_category_losses(logits, category, complete))


def _train_fixed_schedule(windows, offsets, category, complete, weights, *,
                          recipe, seed, recent_samples, adapter, budget_check):
    if torch.get_num_threads() != 1:
        raise ValueError("caller must configure one CPU numerical thread within its allocation")
    windows = _tensor(windows, dtype=torch.float32)
    offsets, weights = _tensor(offsets, dtype=torch.float64), _tensor(weights, dtype=torch.float64)
    category, complete = torch.tensor(category, dtype=torch.int64), torch.tensor(complete, dtype=torch.int64)
    rows = len(windows)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.random.default_generator.manual_seed(seed)
            permutation = torch.Generator(device="cpu").manual_seed(seed)
            network = GRUCorrectionNetwork(recipe.family_parameters["hidden_units"],
                recent_samples=recent_samples, adapter=adapter)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.001, betas=(.9, .999),
                eps=1e-8, weight_decay=recipe.regularization, amsgrad=False, maximize=False,
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
                    objective = _minibatch_objective(network(windows[indices]), offsets[indices],
                        category[indices], complete[indices], weights[indices], rows)
                    if not bool(torch.isfinite(objective)):
                        raise UnresolvedFit("nonfinite_GRU_objective")
                    objective.backward()
                    try:
                        nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                    except RuntimeError as error:
                        raise UnresolvedFit("nonfinite_GRU_gradient") from error
                    budget_check()  # Required immediately before parameter mutation.
                    optimizer.step()
                    steps += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                        raise UnresolvedFit("nonfinite_GRU_parameter")
            network.eval()
            objective = 0.
            with torch.no_grad():
                for start in range(0, rows, BATCH_SIZE):
                    budget_check()
                    stop = start + BATCH_SIZE
                    logits = network(windows[start:stop]).to(torch.float64) + offsets[start:stop]
                    objective += float(torch.sum(weights[start:stop] * censored_category_losses(
                        logits, category[start:stop], complete[start:stop])))
            if not np.isfinite(objective):
                raise UnresolvedFit("nonfinite_GRU_final_objective")
            for parameter in network.parameters():
                parameter.grad = None
                parameter.requires_grad_(False)
            return network, steps, objective
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)


class GRUForecastModel:
    """Same fit/reset/predict interface as the existing classical references.

    Semantics explicitly supplies registry, canonical fitting_scope,
    analysis_sample_rate_hz, channel_order, adapter and budget_check. The fixed
    selected GRU_SUMMARY and GRU_INITIAL_STATE are not search parameters.
    The supplied PastWindow already contains the common
    fitting-only voltage transform and any condition mask/reconstruction. No
    further centering/scaling is fit here: a masked zero remains exactly zero.
    Window availability and source admission still belong to the caller.
    fitting_stage defaults to development; final_refit must be explicit.
    """
    def fit(self, batches, *, recipe, seed, semantics: Mapping[str, Any]):
        self.fitted = False
        registry = semantics["registry"]
        recipes = {item.recipe_id: item for item in gru_recipes(registry)}
        if recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id] or type(seed) is not int or seed not in SEEDS:
            raise ValueError("exact registered GRU recipe and seed required")
        scope = semantics["fitting_scope"]
        if not isinstance(scope, FittingScope):
            raise ValueError("canonical development fitting scope required")
        self.fitting_stage = semantics.get("fitting_stage", "development")
        scope = canonical_reference_scope(registry, scope, fitting_stage=self.fitting_stage)
        if tuple(semantics["channel_order"]) != CHANNELS:
            raise ValueError("unchanged scientific 29-channel order required")
        if (semantics.get("gru_summary", GRU_SUMMARY) != GRU_SUMMARY
                or semantics.get("gru_initial_state", GRU_INITIAL_STATE) != GRU_INITIAL_STATE):
            raise ValueError("selected last-hidden summary and zero boundary initialization are fixed")
        self.adapter = semantics["adapter"]
        if self.adapter not in ("original", "factorial"):
            raise ValueError("explicit original/factorial adapter required")
        self.rate, self.recipe = semantics["analysis_sample_rate_hz"], recipe
        history, recent = exact_samples(recipe.history_seconds, self.rate), exact_samples(.2, self.rate)
        if history <= recent:
            raise ValueError("both older and recent slots must have supported samples")
        budget_check = semantics["budget_check"]
        if not callable(budget_check):
            raise ValueError("caller allocation/deadline callback is required")
        budget_check()
        examples = []
        for batch in batches:
            budget_check()
            examples.extend(batch)
        weights = scope.validate([example.identity for example in examples])
        windows = []
        for example in examples:
            budget_check()
            windows.append(self._window(example.past_input, example.decision_time, example.identity.boundary_id))
        offsets = _offsets([example.context_log_probabilities for example in examples], len(examples))
        category, complete = _labels([example.observed_category for example in examples],
                                    [example.complete_bins for example in examples], len(examples))
        if not np.any((category >= 0) | (complete > 0)):
            raise UnresolvedFit("no_observed_GRU_likelihood_information")
        self.network, steps, objective = _train_fixed_schedule(np.asarray(windows), offsets,
            category, complete, weights, recipe=recipe, seed=seed, recent_samples=recent,
            adapter=self.adapter, budget_check=budget_check)
        self.seed, self.fitting_scope = seed, scope
        self.boundary_id, self.last_decision_time = None, -np.inf
        self.fitted = True
        return {"completed": True, "reason": "fixed_100_epoch_schedule_complete", "epochs": EPOCHS,
            "optimizer_steps": steps, "objective": objective, "recipe_id": recipe.recipe_id,
            "seed": seed, "source": scope.source, "fold_id": scope.fold_id, "examples": len(examples),
            "fitting_stage": self.fitting_stage,
            "analysis_sample_rate_hz": self.rate, "history_samples": history, "recent_samples": recent,
            "channel_order": list(CHANNELS), "adapter": self.adapter,
            "gru_summary": GRU_SUMMARY, "gru_initial_state": GRU_INITIAL_STATE,
            "voltage_transform": "caller_supplied_shared_training_only_before_masking",
            "source_admission_established_by_backend": False}

    def _window(self, window, decision_time, boundary_id):
        if np.iscomplexobj(window.values) or np.iscomplexobj(window.sample_times):
            raise ValueError("EEG values and timestamps must be real")
        values = window.validated(decision_time=decision_time, sample_rate_hz=self.rate,
            history_seconds=self.recipe.history_seconds, boundary_id=boundary_id)
        if np.any(np.diff(np.asarray(window.sample_times, dtype=np.float64)) <= 0):
            raise ValueError("sample times must increase strictly")
        return values

    def reset(self, *, boundary_id):
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("explicit nonempty reset boundary required")
        self.boundary_id, self.last_decision_time = boundary_id, -np.inf

    def predict_log_probabilities(self, past_input, *, decision_time, context_log_probabilities):
        if not getattr(self, "fitted", False):
            raise UnresolvedFit("cannot predict from an incomplete GRU fit")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        if torch.get_num_threads() != 1:
            raise ValueError("caller must configure one CPU numerical thread within its allocation")
        window = self._window(past_input, decision_time, self.boundary_id)
        offsets = _offsets(np.asarray(context_log_probabilities)[None, :], 1)
        with torch.no_grad():
            correction = self.network(_tensor(window[None, :, :], dtype=torch.float32)).numpy()
        if not np.all(np.isfinite(correction)):
            raise UnresolvedFit("nonfinite_GRU_prediction")
        prediction = normalized_log_probabilities(correction.astype(np.float64) + offsets)[0]
        self.last_decision_time = decision_time
        return prediction
