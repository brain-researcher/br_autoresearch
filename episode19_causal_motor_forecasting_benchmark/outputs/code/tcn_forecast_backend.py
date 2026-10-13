"""Prospective EP19 EEG TCN correction; supplied development windows only.

No source I/O, clocks, resampling, physical scale or condition mask is inferred.
Complete PastWindows already contain the shared fitting-only voltage transform
and any reconstruction/masking. Fixed left zeros define the convolution outside
each complete window/slot; they never fill missing history. There is no minimum
125-sample eligibility requirement and no persistent convolutional cache.
The fitter materializes the fitting fold; production memory qualification and
source admission remain upstream requirements, not granted by this backend.
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

FAMILY = "dilated_past_only_temporal_convolutional_network"
SEEDS = (1901, 1902, 1903)
DILATIONS = (1, 2, 4, 8, 16)
EPOCHS, BATCH_SIZE = 100, 128
TCN_SUMMARY = "mean_hidden_positions_per_slot"
TCN_BOUNDARY = "fixed_left_zero_per_complete_window_and_factorial_slot"
RECEPTIVE_FIELD_SAMPLES = 125


def tcn_recipes(registry):
    grid, training, features = registry["reference_grid"], registry["training"], registry["reference_features"]
    expected = {"output_categories": 19, "loss": "existing_censoring_aware_19_category_NLL",
        "class_rebalancing": False, "neural_optimizer": "AdamW", "neural_learning_rate": .001,
        "neural_epochs": EPOCHS, "batch_examples": BATCH_SIZE, "neural_dropout": .25,
        "gradient_global_norm_maximum": 1.,
        "participant_weighting": "equal_total_per_fitting_participant_then_equal_eligible_grid_points_within_person"}
    if (grid["history_seconds"] != [1., 2.] or grid["regularization"] != [.001, .01, .1]
            or grid["family_variants"][FAMILY] != [{"channels": 16}, {"channels": 32}]
            or tuple(registry["seeds"]) != SEEDS
            or features["TCN_kernel_samples"] != 3
            or tuple(features["TCN_dilations"]) != DILATIONS
            or features["TCN_convolutions_per_block"] != 2
            or features["bidirectional_or_unmasked_cache"] is not False
            or any(training[key] != value for key, value in expected.items())):
        raise ValueError("unchanged TCN recipes, seed bank and fixed training schedule required")
    return tuple(recipe for recipe in reference_recipes(registry) if recipe.family == FAMILY)


def _tensor(value, *, dtype):
    result = torch.tensor(value, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(result).all()):
        raise UnresolvedFit("nonfinite_TCN_input_or_float32_overflow")
    return result


class _CausalResidual(nn.Module):
    """Same selected baseline block, with the registered 16/32 width choice."""
    def __init__(self, input_channels, channels, dilation):
        super().__init__()
        self.left = 2 * dilation
        settings = {"kernel_size": 3, "dilation": dilation, "padding": 0,
                    "device": "cpu", "dtype": torch.float32}
        self.first = nn.Conv1d(input_channels, channels, **settings)
        self.second = nn.Conv1d(channels, channels, **settings)
        self.shortcut = (nn.Identity() if input_channels == channels else
                         nn.Conv1d(input_channels, channels, 1, device="cpu", dtype=torch.float32))
        self.dropout = nn.Dropout(.25)

    def forward(self, values):
        hidden = self.dropout(F.relu(self.first(F.pad(values, (self.left, 0)))))
        hidden = self.dropout(F.relu(self.second(F.pad(hidden, (self.left, 0)))))
        return F.relu(hidden + self.shortcut(values))


class TCNCorrectionNetwork(nn.Module):
    """One encoder, equal two-mean head and no cache or condition indicator.

    Original mode encodes the whole sequence before reducing its two slots.
    Factorial mode resets every convolution's finite left boundary at the slot
    split after caller reconstruction/masking, using precisely the same weights.
    """
    def __init__(self, channels, *, recent_samples, adapter):
        super().__init__()
        if channels not in (16, 32) or type(recent_samples) is not int or recent_samples < 1:
            raise ValueError("registered channels and positive recent sample count required")
        if adapter not in ("original", "factorial"):
            raise ValueError("explicit original/factorial adapter required")
        self.recent_samples, self.adapter = recent_samples, adapter
        self.blocks = nn.Sequential(*[_CausalResidual(29 if index == 0 else channels, channels, dilation)
                                     for index, dilation in enumerate(DILATIONS)])
        self.readout = nn.Linear(2 * channels, 19, device="cpu", dtype=torch.float32)

    def hidden(self, sequence):
        if sequence.ndim != 3 or sequence.shape[2] != 29 or sequence.shape[1] < 1:
            raise ValueError("nonempty supplied time-by-29-channel sequence required")
        return self.blocks(sequence.transpose(1, 2))

    def summaries(self, sequence):
        if sequence.ndim != 3 or sequence.shape[2] != 29 or sequence.shape[1] <= self.recent_samples:
            raise ValueError("complete nonempty older and recent EEG slots required")
        cut = sequence.shape[1] - self.recent_samples
        if self.adapter == "original":
            hidden = self.hidden(sequence)
            older, recent = hidden[:, :, :cut], hidden[:, :, cut:]
        else:
            older, recent = self.hidden(sequence[:, :cut]), self.hidden(sequence[:, cut:])
        return torch.cat((older.mean(dim=2), recent.mean(dim=2)), dim=1)

    def forward(self, sequence):
        return self.readout(self.summaries(sequence))


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
            network = TCNCorrectionNetwork(recipe.family_parameters["channels"],
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
                        raise UnresolvedFit("nonfinite_TCN_objective")
                    objective.backward()
                    try:
                        nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
                    except RuntimeError as error:
                        raise UnresolvedFit("nonfinite_TCN_gradient") from error
                    budget_check()
                    optimizer.step()
                    steps += 1
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
                        raise UnresolvedFit("nonfinite_TCN_parameter")
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
                raise UnresolvedFit("nonfinite_TCN_final_objective")
            for parameter in network.parameters():
                parameter.grad = None
                parameter.requires_grad_(False)
            return network, steps, objective
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn)


class TCNForecastModel:
    """Existing fit/reset/predict interface; no condition-specific input scaler.

    registry, canonical fitting_scope, analysis_sample_rate_hz,
    channel_order, adapter and budget_check are explicitly supplied semantics.
    Summary and finite convolution boundary are selected constants, not knobs.
    Missing acquisition/filter/window qualifications gain no default here.
    fitting_stage defaults to development; final_refit must be explicit.
    """
    def fit(self, batches, *, recipe, seed, semantics: Mapping[str, Any]):
        self.fitted = False
        registry = semantics["registry"]
        recipes = {item.recipe_id: item for item in tcn_recipes(registry)}
        if recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id] or type(seed) is not int or seed not in SEEDS:
            raise ValueError("exact registered TCN recipe and seed required")
        scope = semantics["fitting_scope"]
        if not isinstance(scope, FittingScope):
            raise ValueError("canonical development fitting scope required")
        self.fitting_stage = semantics.get("fitting_stage", "development")
        scope = canonical_reference_scope(registry, scope, fitting_stage=self.fitting_stage)
        if tuple(semantics["channel_order"]) != CHANNELS:
            raise ValueError("unchanged scientific 29-channel order required")
        if (semantics.get("tcn_summary", TCN_SUMMARY) != TCN_SUMMARY
                or semantics.get("tcn_boundary", TCN_BOUNDARY) != TCN_BOUNDARY):
            raise ValueError("selected two-mean summary and finite left-zero boundary are fixed")
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
            raise UnresolvedFit("no_observed_TCN_likelihood_information")
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
            "tcn_summary": TCN_SUMMARY, "tcn_boundary": TCN_BOUNDARY,
            "receptive_field_samples": RECEPTIVE_FIELD_SAMPLES,
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
            raise UnresolvedFit("cannot predict from an incomplete TCN fit")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        if torch.get_num_threads() != 1:
            raise ValueError("caller must configure one CPU numerical thread within its allocation")
        window = self._window(past_input, decision_time, self.boundary_id)
        offsets = _offsets(np.asarray(context_log_probabilities)[None, :], 1)
        with torch.no_grad():
            correction = self.network(_tensor(window[None, :, :], dtype=torch.float32)).numpy()
        if not np.all(np.isfinite(correction)):
            raise UnresolvedFit("nonfinite_TCN_prediction")
        prediction = normalized_log_probabilities(correction.astype(np.float64) + offsets)[0]
        self.last_decision_time = decision_time
        return prediction
