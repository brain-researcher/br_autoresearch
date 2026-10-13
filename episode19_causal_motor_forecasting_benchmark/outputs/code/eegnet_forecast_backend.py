"""Prospective EP19 compact causal EEGNet on caller-admitted EEG windows.

No source I/O, sampling-rate default, resampling, input scaler or mask is added.
The two pooling grids are anchored to the full window's final sample, including
when the factorial adapter resets operators at the older/recent cutoff. Fixed
left zeros define operators outside complete admitted slots, not missing raw
history. Production readers and memory/source qualification remain upstream.
"""
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from classical_forecast_backend import (
    FittingScope, UnresolvedFit, _labels, exact_samples, normalized_log_probabilities,
    canonical_reference_scope,
)
from gru_forecast_backend import CHANNELS, _offsets, _minibatch_objective, censored_category_losses
from model_recipes import reference_recipes

FAMILY = "past_only_compact_EEGNet"
SEEDS = (1901, 1902, 1903)
EPOCHS, BATCH_SIZE = 100, 128
EEGNET_SUMMARY = "mean_encoded_positions_per_slot"
EEGNET_BOUNDARY = "fixed_left_zero_per_complete_window_and_factorial_slot"
EEGNET_POOL_PHASE = "full_window_last_sample_at_both_pooling_stages"


def eegnet_recipes(registry):
    grid, training, features = registry["reference_grid"], registry["training"], registry["reference_features"]
    expected = {"output_categories": 19, "loss": "existing_censoring_aware_19_category_NLL",
        "class_rebalancing": False, "neural_optimizer": "AdamW", "neural_learning_rate": .001,
        "neural_epochs": EPOCHS, "batch_examples": BATCH_SIZE, "neural_dropout": .25,
        "gradient_global_norm_maximum": 1.,
        "participant_weighting": "equal_total_per_fitting_participant_then_equal_eligible_grid_points_within_person"}
    if (grid["history_seconds"] != [1., 2.] or grid["regularization"] != [.001, .01, .1]
            or grid["family_variants"][FAMILY] != [
                {"temporal_filters": 4, "depth_multiplier": 2},
                {"temporal_filters": 8, "depth_multiplier": 2}]
            or tuple(registry["seeds"]) != SEEDS
            or features["EEGNet_temporal_kernel_ms"] != 128
            or features["EEGNet_separable_kernel_ms"] != 64
            or features["EEGNet_pool_widths"] != [4, 8]
            or features["bidirectional_or_unmasked_cache"] is not False
            or any(training[key] != value for key, value in expected.items())):
        raise ValueError("unchanged EEGNet recipes, seed bank and fixed training schedule required")
    return tuple(recipe for recipe in reference_recipes(registry) if recipe.family == FAMILY)


def kernel_samples(analysis_sample_rate_hz):
    """The second duration applies to the post-pool4 grid, not the raw grid."""
    return (exact_samples(.128, analysis_sample_rate_hz),
            exact_samples(.064, analysis_sample_rate_hz / 4.))


def _tensor(value, *, dtype):
    result = torch.tensor(value, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise UnresolvedFit("nonfinite_EEGNet_input_or_float32_overflow")
    return result


def _past_average(values, width):
    # Explicit zeros are part of the fixed operator; divide by the full width
    # even at the left boundary. No future padding or partial-support renormalization.
    return F.avg_pool1d(F.pad(values, (width - 1, 0)), kernel_size=width, stride=1)


class EEGNetCorrectionNetwork(nn.Module):
    """Shared compact encoder, with equal-capacity original/factorial adapters."""
    def __init__(self, temporal_filters, *, analysis_sample_rate_hz, recent_samples, adapter):
        super().__init__()
        if temporal_filters not in (4, 8) or type(recent_samples) is not int or recent_samples < 1:
            raise ValueError("registered temporal filters and positive recent sample count required")
        if adapter not in ("original", "factorial"):
            raise ValueError("explicit original/factorial adapter required")
        self.temporal_samples, self.separable_samples = kernel_samples(analysis_sample_rate_hz)
        self.recent_samples, self.adapter = recent_samples, adapter
        self.channels = 2 * temporal_filters
        settings = {"padding": 0, "bias": True, "device": "cpu", "dtype": torch.float32}
        self.temporal = nn.Conv2d(1, temporal_filters, (1, self.temporal_samples), **settings)
        self.spatial = nn.Conv2d(temporal_filters, self.channels, (29, 1),
                                 groups=temporal_filters, **settings)
        self.depthwise = nn.Conv1d(self.channels, self.channels, self.separable_samples,
                                   groups=self.channels, **settings)
        self.pointwise = nn.Conv1d(self.channels, self.channels, 1, **settings)
        self.first_dropout, self.second_dropout = nn.Dropout(.25), nn.Dropout(.25)
        self.readout = nn.Linear(2 * self.channels, 19, device="cpu", dtype=torch.float32)

    def encoded(self, sequence, *, start_sample=0, full_window_samples=None):
        """Return encoded values and their full-window raw-sample right edges.

        Every operator starts at this supplied slot's fixed zero boundary. Pool
        endpoints keep the original full-window phase: N-1-4k, then N-1-32k.
        Neither endpoint grid is independently reanchored to the slot's end.
        """
        if sequence.ndim != 3 or sequence.shape[2] != 29 or sequence.shape[1] < 1:
            raise ValueError("nonempty supplied time-by-29-channel sequence required")
        length = sequence.shape[1]
        if full_window_samples is None:
            full_window_samples = length
        if (type(start_sample) is not int or type(full_window_samples) is not int
                or start_sample < 0 or full_window_samples < start_sample + length):
            raise ValueError("slot coordinates must lie within the complete full window")
        first_offset = (full_window_samples - 1 - start_sample) % 4
        if first_offset >= length:
            raise ValueError("slot has no encoded positions on the fixed pooling grid")
        first_indices = torch.arange(first_offset, length, 4, device="cpu")
        first_edges = start_sample + first_indices
        final_indices = torch.nonzero((full_window_samples - 1 - first_edges) % 32 == 0,
                                      as_tuple=False).flatten()
        if not len(first_indices) or not len(final_indices):
            raise ValueError("slot has no encoded positions on the fixed pooling grid")
        values = sequence.transpose(1, 2).unsqueeze(1)
        hidden = self.temporal(F.pad(values, (self.temporal_samples - 1, 0, 0, 0)))
        hidden = F.elu(self.spatial(hidden).squeeze(2))
        hidden = self.first_dropout(_past_average(hidden, 4)[:, :, first_indices])
        hidden = self.depthwise(F.pad(hidden, (self.separable_samples - 1, 0)))
        hidden = F.elu(self.pointwise(hidden))
        hidden = self.second_dropout(_past_average(hidden, 8)[:, :, final_indices])
        return hidden, first_edges[final_indices]

    def summaries(self, sequence):
        if sequence.ndim != 3 or sequence.shape[2] != 29 or sequence.shape[1] <= self.recent_samples:
            raise ValueError("complete nonempty older and recent EEG slots required")
        length = sequence.shape[1]
        cut = length - self.recent_samples
        if self.adapter == "original":
            hidden, edges = self.encoded(sequence)
            older, recent = hidden[:, :, edges < cut], hidden[:, :, edges >= cut]
        else:
            older, _ = self.encoded(sequence[:, :cut], full_window_samples=length)
            recent, _ = self.encoded(sequence[:, cut:], start_sample=cut, full_window_samples=length)
        if older.shape[2] == 0 or recent.shape[2] == 0:
            raise ValueError("both slots require encoded positions on the fixed pooling grid")
        return torch.cat((older.mean(dim=2), recent.mean(dim=2)), dim=1)

    def forward(self, sequence):
        return self.readout(self.summaries(sequence))


def _train_fixed_schedule(windows, offsets, category, complete, weights, *,
                          recipe, seed, analysis_sample_rate_hz, recent_samples, adapter, budget_check):
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
            network = EEGNetCorrectionNetwork(recipe.family_parameters["temporal_filters"],
                analysis_sample_rate_hz=analysis_sample_rate_hz, recent_samples=recent_samples, adapter=adapter)
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
                        raise UnresolvedFit("nonfinite_EEGNet_objective")
                    objective.backward()
                    try:
                        nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                    except RuntimeError as error:
                        raise UnresolvedFit("nonfinite_EEGNet_gradient") from error
                    budget_check()
                    optimizer.step()
                    steps += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                        raise UnresolvedFit("nonfinite_EEGNet_parameter")
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
                raise UnresolvedFit("nonfinite_EEGNet_final_objective")
            for parameter in network.parameters():
                parameter.grad = None
                parameter.requires_grad_(False)
            return network, steps, objective
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)


class EEGNetForecastModel:
    """Existing interface; final_refit requires an explicit fitting_stage."""
    def fit(self, batches, *, recipe, seed, semantics: Mapping[str, Any]):
        self.fitted = False
        registry = semantics["registry"]
        recipes = {item.recipe_id: item for item in eegnet_recipes(registry)}
        if recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id] or type(seed) is not int or seed not in SEEDS:
            raise ValueError("exact registered EEGNet recipe and seed required")
        scope = semantics["fitting_scope"]
        if not isinstance(scope, FittingScope):
            raise ValueError("canonical development fitting scope required")
        self.fitting_stage = semantics.get("fitting_stage", "development")
        scope = canonical_reference_scope(registry, scope, fitting_stage=self.fitting_stage)
        if tuple(semantics["channel_order"]) != CHANNELS:
            raise ValueError("unchanged scientific 29-channel order required")
        for name, selected in (("eegnet_summary", EEGNET_SUMMARY), ("eegnet_boundary", EEGNET_BOUNDARY),
                               ("eegnet_pool_phase", EEGNET_POOL_PHASE)):
            if semantics.get(name, selected) != selected:
                raise ValueError("selected EEGNet summary, boundary and pooling phase are fixed")
        self.adapter = semantics["adapter"]
        if self.adapter not in ("original", "factorial"):
            raise ValueError("explicit original/factorial adapter required")
        self.rate, self.recipe = semantics["analysis_sample_rate_hz"], recipe
        temporal, separable = kernel_samples(self.rate)
        history, recent = exact_samples(recipe.history_seconds, self.rate), exact_samples(.2, self.rate)
        if history <= recent or history - recent <= (history - 1) % 32:
            raise ValueError("both older and recent slots must have encoded positions")
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
            raise UnresolvedFit("no_observed_EEGNet_likelihood_information")
        self.network, steps, objective = _train_fixed_schedule(np.asarray(windows), offsets,
            category, complete, weights, recipe=recipe, seed=seed, analysis_sample_rate_hz=self.rate,
            recent_samples=recent, adapter=self.adapter, budget_check=budget_check)
        self.seed, self.fitting_scope = seed, scope
        self.boundary_id, self.last_decision_time = None, -np.inf
        self.fitted = True
        return {"completed": True, "reason": "fixed_100_epoch_schedule_complete", "epochs": EPOCHS,
            "optimizer_steps": steps, "objective": objective, "recipe_id": recipe.recipe_id,
            "seed": seed, "source": scope.source, "fold_id": scope.fold_id, "examples": len(examples),
            "fitting_stage": self.fitting_stage,
            "analysis_sample_rate_hz": self.rate, "history_samples": history, "recent_samples": recent,
            "channel_order": list(CHANNELS), "adapter": self.adapter,
            "eegnet_summary": EEGNET_SUMMARY, "eegnet_boundary": EEGNET_BOUNDARY,
            "eegnet_pool_phase": EEGNET_POOL_PHASE, "temporal_kernel_raw_samples": temporal,
            "separable_kernel_post_pool4_samples": separable, "pool_widths_and_strides": [4, 8],
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
            raise UnresolvedFit("cannot predict from an incomplete EEGNet fit")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        if torch.get_num_threads() != 1:
            raise ValueError("caller must configure one CPU numerical thread within its allocation")
        window = self._window(past_input, decision_time, self.boundary_id)
        offsets = _offsets(np.asarray(context_log_probabilities)[None, :], 1)
        with torch.no_grad():
            correction = self.network(_tensor(window[None, :, :], dtype=torch.float32)).numpy()
        if not np.all(np.isfinite(correction)):
            raise UnresolvedFit("nonfinite_EEGNet_prediction")
        prediction = normalized_log_probabilities(correction.astype(np.float64) + offsets)[0]
        self.last_decision_time = decision_time
        return prediction
