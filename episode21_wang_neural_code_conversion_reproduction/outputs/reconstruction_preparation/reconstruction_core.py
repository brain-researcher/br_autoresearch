"""Unactivated, supplied-model Finding B core; no reader, assets, CLI or grant.

Expressions follow Wang V1.0.0 recon_icnn_image_gd.py and BdPy 0.25
recon/torch/icnn.py, recon/utils.py and feature/feature.py; see source_binding.md.
The caller authenticates the pinned VGG19/AlexNetGenerator, decoder-owned
training means, public feature statistics, latent bounds and empirical roles.
An outcome/asset lock and finite allocation must be admitted outside this core.

All nineteen decoded arrays and means have a leading singleton image axis.
NumPy feature/statistic dtypes are preserved as in the source, including
float64 means/features. Models and the optimized latent are float32; target
dtype is inferred, never silently coerced to resolve a runtime incompatibility.
No norm epsilon, missing-layer fallback, restart or alternate output is used.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LAYERS = ("conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
          "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4",
          "conv5_1", "conv5_2", "conv5_3", "conv5_4", "fc6", "fc7", "fc8")
SHAPES = ((64, 224, 224), (64, 224, 224), (128, 112, 112), (128, 112, 112),
          (256, 56, 56), (256, 56, 56), (256, 56, 56), (256, 56, 56),
          (512, 28, 28), (512, 28, 28), (512, 28, 28), (512, 28, 28),
          (512, 14, 14), (512, 14, 14), (512, 14, 14), (512, 14, 14),
          (4096,), (4096,), (1000,))
_HOOKS = tuple(("features", i) for i in
              (0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34)) + (
              ("classifier", 0), ("classifier", 3), ("classifier", 6))
ALPHA = dict(zip(LAYERS, (
    0.009952859626673973, 0.003886371646003732, 0.00024056214287883663,
    0.08084213863904581, 0.006857140440209977, 0.0004887923569668929,
    0.00019573287900056505, 0.014169279688225732, 0.02880295296139471,
    0.0021716504774059618, 0.004025735147052829, 0.030798346318926202,
    0.08536182104218124, 0.714010791024532, 0.01975314483213067,
    0.5310432634795051, 0.000629031742843134, 0.03871265134313895,
    0.06661523244761258)))
BETA = dict(zip(LAYERS, (
    0.000865427897168344, 0.008627502425502372, 0.12888847991806807,
    0.02472305546171304, 0.0002639605292406699, 0.039778524165843814,
    0.0004859871528150426, 0.024733951474856346, 0.04328555659354602,
    0.03998274022264576, 0.5166895849277143, 0.7100372437615771,
    0.002337825161420017, 0.5074163077203029, 0.0018775814111698145,
    0.0012780195483159135, 0.038457933646483915, 0.044481711593671994,
    0.008304840607742099)))


class ReconstructionUnavailable(ValueError):
    """Required input/numerical support is unavailable; never a partial image."""


@dataclass(frozen=True)
class ReconstructionResult:
    raw_rgb: np.ndarray
    image_uint8: np.ndarray
    operational_metadata: dict


def _require(condition, reason):
    if not condition:
        raise ReconstructionUnavailable(reason)


def _array(value, reason, shape=None):
    _require(isinstance(value, np.ndarray) and value.dtype in
             (np.dtype("float32"), np.dtype("float64")) and value.size > 0,
             reason)
    _require(shape is None or value.shape == shape, reason)
    _require(bool(np.isfinite(value).all()), reason)
    return value


def _scale_feature(feature, decoder_mean, public_std):
    """Exact selected normalization; arguments include the singleton row axis."""
    centered = feature - decoder_mean
    value = centered[0]
    axes = None if value.ndim == 1 else tuple(range(1, value.ndim))
    mean = np.mean(value, keepdims=True)
    std = np.mean(np.std(value, axis=axes, ddof=1, keepdims=True), keepdims=True)
    scale = np.mean(public_std)
    _require(bool(np.isfinite(std).all()) and bool((std > 0).all()),
             "nonpositive_or_nonfinite_decoded_feature_scale")
    _require(bool(np.isfinite(scale)) and bool(scale > 0),
             "nonpositive_or_nonfinite_public_feature_scale")
    normalized = ((value - mean) / std)
    normalized = normalized * scale
    normalized = normalized + mean
    normalized = normalized[np.newaxis] + decoder_mean
    _require(bool(np.isfinite(normalized).all()), "nonfinite_scaled_decoded_features")
    return normalized


def _layer_weights(features):
    # The source computes each norm in its feature dtype, then casts the
    # nineteen norm scalars to float32 before inverse squares and their sum.
    norms = np.array([np.linalg.norm(features[layer]) for layer in LAYERS], dtype="float32")
    _require(bool(np.isfinite(norms).all()) and bool((norms > 0).all()),
             "nonpositive_or_nonfinite_layer_norm")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        weights = 1. / (norms ** 2)
        total = weights.sum()
        weights = weights / total
    _require(bool(np.isfinite(total)) and bool(total > 0)
             and bool(np.isfinite(weights).all()) and bool((weights > 0).all()),
             "unavailable_inverse_squared_norm_weights")
    return dict(zip(LAYERS, weights))


def _dist(x, y, layer):
    if layer.startswith("fc"):
        x_mean, y_mean = x.mean(), y.mean()
        x_var, y_var = ((x - x_mean) ** 2).mean(), ((y - y_mean) ** 2).mean()
        covariance = (x * y).mean() - x_mean * y_mean
    else:
        x_mean, y_mean = x.mean([2, 3], keepdim=True), y.mean([2, 3], keepdim=True)
        x_var = ((x - x_mean) ** 2).mean([2, 3], keepdim=True)
        y_var = ((y - y_mean) ** 2).mean([2, 3], keepdim=True)
        covariance = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
    mean_similarity = (2 * x_mean * y_mean + 1e-6) / (x_mean**2 + y_mean**2 + 1e-6)
    covariance_similarity = (2 * covariance + 1e-6) / (x_var + y_var + 1e-6)
    return -(mean_similarity.mean() * ALPHA[layer]
             + covariance_similarity.mean() * BETA[layer]) * 343639


def _objective(activations, targets, weights, check):
    loss = 0.
    for layer in reversed(LAYERS):
        check()
        activation = activations[layer].clone()
        target = targets[layer].clone()
        layer_loss = F.mse_loss(activation, target, reduction="sum") * weights[layer]
        layer_loss = layer_loss + _dist(activation, target, layer)
        loss += layer_loss
    _require(bool(torch.isfinite(loss).item()), "nonfinite_reconstruction_objective")
    return loss


def _initial_latent():
    # A local legacy RNG gives the same single seed-42 draw to every cell and
    # does not advance a caller's RNG or choose a new image-specific restart.
    return np.random.RandomState(42).normal(0, 1, (4096,)).astype(np.float32)


def _learning_rate(iteration):
    return 2. + iteration * (1e-10 - 2.) / 200


def _decay_and_clip(value, upper):
    value = (1 - 0.01) * value
    value = np.maximum(value, 0.)
    value = np.minimum(value, upper)
    return value.astype(np.float32)


def _crop_generated(image):
    _require(isinstance(image, torch.Tensor) and image.dtype == torch.float32
             and tuple(image.shape) == (1, 3, 256, 256),
             "pinned_generator_float32_256_by_256_output_required")
    _require(bool(torch.isfinite(image).all().item()), "nonfinite_generator_output")
    # Pinned masked_select has this same row-major selection and shape.
    return image[:, :, 16:240, 16:240].contiguous()


def _iterate(generator, encoder, objective, upper, device, check):
    """Fixed latent loop. The private objective seam permits a tiny oracle test."""
    value = _initial_latent()
    latent = torch.tensor(value[np.newaxis], device=device, requires_grad=True)
    optimizer = torch.optim.SGD([latent], lr=2.)
    encoder.zero_grad()
    generator.zero_grad()
    last_image = None
    for iteration in range(200):
        check()
        optimizer.zero_grad()
        encoder.zero_grad()
        generator.zero_grad()
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(iteration)
            group["momentum"] = 0.9
        latent.data = torch.tensor(value[np.newaxis], device=device)
        image = _crop_generated(generator.forward(latent))
        check()
        loss = objective(image)
        _require(isinstance(loss, torch.Tensor) and loss.ndim == 0
                 and bool(torch.isfinite(loss).item()), "finite_scalar_objective_required")
        loss.backward()
        _require(latent.grad is not None and bool(torch.isfinite(latent.grad).all().item()),
                 "nonfinite_or_missing_latent_gradient")
        mean_gradient = torch.abs(latent.grad).mean().cpu().detach()
        _require(bool(torch.isfinite(mean_gradient).item()), "nonfinite_gradient_mean")
        if mean_gradient > 0:
            latent.grad /= mean_gradient
        # This image precedes this iteration's update, including iteration199.
        last_image = image.clone().cpu().detach().numpy()[0]
        optimizer.step()
        value = latent.cpu().detach().numpy()[0]
        _require(bool(np.isfinite(value).all()), "nonfinite_updated_latent")
        value = _decay_and_clip(value, upper)
        _require(bool(np.isfinite(value).all()), "nonfinite_clipped_latent")
        check()
    return last_image


def _postprocess(bgr):
    _array(bgr, "finite_bgr_image_required")
    _require(bgr.ndim == 3 and bgr.shape[0] == 3, "CHW_bgr_image_required")
    raw = np.dstack((bgr + np.reshape(np.float32([104, 117, 123]), (3, 1, 1)))[::-1])
    _require(bool(np.isfinite(raw).all()), "nonfinite_deprocessed_RGB")
    clipped = np.clip(raw, np.percentile(raw, 2.), np.percentile(raw, 98.))
    normalized = clipped - clipped.min()
    if normalized.max() > 0:
        normalized = normalized * (255.0 / normalized.max())
    _require(bool(np.isfinite(normalized).all()), "nonfinite_final_pixel_normalization")
    image = np.uint8(normalized)
    raw.setflags(write=False)
    image.setflags(write=False)
    return raw, image


def _model_device(encoder, generator):
    _require(isinstance(encoder, nn.Module) and isinstance(generator, nn.Module),
             "caller_supplied_pinned_torch_models_required")
    parameters = tuple(encoder.parameters()) + tuple(generator.parameters())
    _require(bool(parameters) and all(p.dtype == torch.float32 for p in parameters),
             "float32_model_parameters_required")
    devices = {p.device for p in parameters}
    _require(len(devices) == 1, "one_shared_model_device_required")
    device = next(iter(devices))
    _require(device.type in ("cpu", "cuda"), "cpu_or_assigned_cuda_model_device_required")
    _require(all(not isinstance(module, nn.ReLU) or not module.inplace
                 for module in encoder.modules()), "pinned_noninplace_VGG_ReLU_required")
    return device


@contextmanager
def _encoder_activations(encoder):
    activations, handles = {}, []
    try:
        for layer, (section, index) in zip(LAYERS, _HOOKS):
            _require(hasattr(encoder, section) and len(getattr(encoder, section)) > index,
                     "pinned_VGG_module_layout_required")
            module = getattr(encoder, section)[index]
            _require(isinstance(module, nn.Linear if layer.startswith("fc") else nn.Conv2d),
                     "pinned_VGG_preReLU_module_required")
            def capture(module, inputs, output, name=layer):
                _require(name not in activations, "VGG_layer_executed_more_than_once")
                activations[name] = output
            handles.append(module.register_forward_hook(capture))
        yield activations
    finally:
        for handle in handles:
            handle.remove()
        activations.clear()


def reconstruct(decoded_features, *, decoder_training_means, public_feature_std,
                encoder, generator, latent_upper_bound, method, source, target,
                decoder_owner, budget_check):
    """Return one cell's raw RGB and final uint8 image, or raise unavailability.

    method is within_source/content_loss/brain_loss. The declared owner applies
    to both supplied decoded values and training means: source for within,
    target for conversions. This checks caller declarations, not provenance.
    Missing/invalid inputs produce no image. Cancellation and runtime failures
    propagate; callers preserve every requested failure rather than drop cells.
    Models are placed in eval mode; their parameters are never in the optimizer.
    The caller controls the already-admitted device and finite process limit.
    """
    _require(callable(budget_check), "explicit_budget_callback_required")
    budget_check()
    _require(method in ("within_source", "content_loss", "brain_loss")
             and isinstance(source, str) and bool(source), "declared_reconstruction_method_required")
    if method == "within_source":
        _require(target is None and decoder_owner == source, "within_source_decoder_mean_owner_required")
    else:
        _require(isinstance(target, str) and bool(target) and target != source
                 and decoder_owner == target, "target_decoder_mean_owner_required")
    for values in (decoded_features, decoder_training_means, public_feature_std):
        _require(isinstance(values, Mapping) and set(values) == set(LAYERS),
                 "all_nineteen_layers_required")
    upper = _array(latent_upper_bound, "finite_4096_latent_upper_bounds_required", (4096,)).copy()
    _require(bool((upper >= 0).all()), "nonnegative_latent_upper_bounds_required")
    scaled = {}
    for layer, shape in zip(LAYERS, SHAPES):
        budget_check()
        feature = _array(decoded_features[layer], "complete_decoded_feature_shape_required", (1,) + shape)
        mean = _array(decoder_training_means[layer], "complete_decoder_training_mean_shape_required", (1,) + shape)
        std = _array(public_feature_std[layer], "finite_public_feature_SD_required")
        _require(bool((std >= 0).all()), "nonnegative_public_feature_SD_required")
        scaled[layer] = _scale_feature(feature, mean, std)
    weights = _layer_weights(scaled)
    device = _model_device(encoder, generator)
    encoder.eval()
    generator.eval()
    targets = {layer: torch.tensor(scaled[layer], device=device) for layer in LAYERS}
    del scaled
    with _encoder_activations(encoder) as activations:
        def objective(image):
            activations.clear()
            encoder.forward(image)
            _require(set(activations) == set(LAYERS), "complete_VGG_activation_roster_required")
            for layer, shape in zip(LAYERS, SHAPES):
                budget_check()
                activation = activations[layer]
                _require(isinstance(activation, torch.Tensor) and activation.dtype == torch.float32
                         and tuple(activation.shape) == (1,) + shape
                         and bool(torch.isfinite(activation).all().item()),
                         "finite_complete_float32_VGG_activations_required")
            return _objective(activations, targets, weights, budget_check)
        with torch.autocast(device_type=device.type, enabled=False):
            bgr = _iterate(generator, encoder, objective, upper, device, budget_check)
    budget_check()
    raw, image = _postprocess(bgr)
    budget_check()
    return ReconstructionResult(raw, image, dict(
        status="completed_cell", completed_updates=200, generator_forwards=200,
        returned_image="iteration199_forward_before_update200", initialization_seed=42,
        initialization="same_unclipped_legacy_numpy_normal_draw_per_cell", layer_count=19,
        method=method, source=source, target=target, decoder_owner=decoder_owner,
        last_learning_rate=_learning_rate(199), model_and_latent_dtype="float32",
        target_dtypes={layer: str(targets[layer].dtype) for layer in LAYERS},
        device=str(device), loss_history_returned=False, model_parameters_optimized=False,
        source_asset_or_empirical_admission_established=False, scientific_success_claim=False))
