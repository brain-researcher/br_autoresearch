"""Full EP17 development image-to-feature delivery, when separately selected.

The caller supplies the authentic frozen development manifest and an admitted
exact-member loader. The loader takes an unchanged provider path and returns
a context manager yielding its opened original PIL image. It must retain the
existing16MiB compressed/uncompressed member caps and single-JPEG/16M-pixel
admission before eager decoding; a post-load PIL check cannot bound allocation
inside the loader. Prefer lazy opening so the checks precede RGB conversion.
This module does
not locate/open an archive, read a credential, establish source identity or
release a reader. The caller owns source authentication, loader lifetime,
physical reader separation, source immutability and actual job admission.

Reuse the registered preprocessing and four fixed checkpoint loaders, including
the already persisted canonical seed17 random artifact. No random artifact is
created. All four CPU backbones are resident together: one image is decoded and
preprocessed once, then evaluated sequentially by every model with batch1.
This engineering memory choice requires measured admission before execution;
it does not reuse the pilot's one-resident-model memory envelope.

All2423 images and all16 layer banks are mandatory. Output is immutable native
float32 FeaturePanels in the original design order. No centering, PCA, fitting,
score, cache persistence, audit image or partial return is supplied here.
Focused fixture success would not qualify authentic empirical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from PIL import Image
import torch

import model_features as features
import nested_prediction as nested
import private_stimulus_feature_pilot as pilot


class FeatureDeliveryError(ValueError):
    """Fixed whole-delivery refusal; never silently omit images or models."""


def require(condition, code):
    if not condition:
        raise FeatureDeliveryError(code)


@dataclass(frozen=True)
class ResourceEvent:
    """Operation metadata only; no image paths, pixels or feature values."""
    operation: str
    when: str = "before"
    model_id: str | None = None
    layer: str | None = None
    image_index: int | None = None
    shape: tuple = ()
    byte_count: int = 0


def _event(check, operation, **context):
    check(ResourceEvent(operation, **context))


def _bind_manifest(design, image_manifest):
    require(isinstance(design, nested.DevelopmentDesign), "complete_development_design_required")
    require(isinstance(image_manifest, (tuple, list))
            and len(image_manifest) == nested.DEVELOPMENT_IMAGES,
            "complete_development_image_manifest_required")
    for index, row in enumerate(image_manifest):
        require(isinstance(row, Mapping), "image_manifest_row_invalid")
        fold, block = design.fold_ids[index], design.block_ids[index]
        require(row.get("provider_image_path") == design.image_ids[index]
                and row.get("concept_name") == design.concept_ids[index]
                and row.get("role") == "development"
                and row.get("eligible_image") == "True"
                and row.get("common_to_four") == "True"
                and row.get("audit_block") == ""
                and row.get("outer_fold") == fold and type(row.get("outer_fold")) is type(fold)
                and row.get("development_block") == block
                and type(row.get("development_block")) is type(block),
                "frozen_eligible_development_image_identity_changed")


def _load_models(check, random_checkpoint_path):
    # Unlike _prepare_random, this call never creates, repairs or rerolls the
    # random artifact. The fixed pilot remains responsible for first creation.
    path = Path(random_checkpoint_path)
    require(path == pilot.RANDOM_CHECKPOINT, "canonical_persisted_random_checkpoint_required")
    _event(check, "require_existing_random_checkpoint")
    info = pilot._ordinary_private_file(path)
    require(0 < info.st_size <= pilot.RANDOM_CAP, "persisted_random_checkpoint_size_changed")
    models = {}
    for model_id in nested.MODEL_IDS:
        _event(check, "load_backbone", model_id=model_id,
               byte_count=4 * features.BACKBONE_PARAMETERS)
        # Trained models use the existing fixed paths. RANDOM loads the saved
        # exact recipe/state after architecture construction; no fresh random
        # initialization is used as an unpersisted representation substitute.
        model = pilot._load_backbone(features, model_id, path)
        require(isinstance(model, features.FeatureBackbone) and model.model_id == model_id,
                "registered_feature_backbone_required")
        require(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                    and not parameter.requires_grad for parameter in model.parameters())
                and all(not module.training for module in model.modules()),
                "fixed_CPU_float32_frozen_backbone_required")
        models[model_id] = model
        _event(check, "load_backbone", when="after", model_id=model_id)
    return models


def _preprocess_one(image_loader, image_id, image_index, check):
    _event(check, "load_image", image_index=image_index)
    # The loader owns closing the yielded source image. The converted RGB
    # object is ours and is closed before feature inference, even on failure.
    with image_loader(image_id) as image:
        require(isinstance(image, Image.Image) and image.format == "JPEG"
                and getattr(image, "n_frames", 1) == 1 and min(image.size) > 0
                and image.width * image.height <= pilot.PIXEL_CAP,
                "source_image_format_or_pixel_cap")
        _event(check, "preprocess_image", image_index=image_index,
               shape=(3, 224, 224), byte_count=3 * 224 * 224 * 4)
        rgb = image.convert("RGB")
        try:
            value = features.preprocess(rgb)
        finally:
            rgb.close()
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.dtype == torch.float32 and tuple(value.shape) == (3, 224, 224)
            and bool(torch.isfinite(value).all().item()), "preprocessed_image_representation_changed")
    _event(check, "preprocess_image", when="after", image_index=image_index)
    return value


def _build_development_features(design, image_manifest, *, image_loader,
                                random_checkpoint_path, resource_check):
    """Return all four complete immutable FeaturePanels; no file/cache release.

    image_manifest is the ordered2423-row eligible development projection of
    image_assignments.csv, with the same exact string fields used by the
    development response adapter. image_loader is callable and supplies only
    the requested authenticated original image; the caller's loader must keep
    archive/decryption details private. The public boundary below suppresses
    underlying exception text; it never serializes provider/decoder errors.

    random_checkpoint_path must be the already existing canonical pilot random
    artifact. Resource callbacks must raise when the coordinated allocation's
    actual memory/time bounds prohibit the next operation. They do not grant
    access or choose a resource budget. Any failure aborts the entire call.
    """
    require(callable(resource_check), "external_resource_callback_required")
    require(callable(image_loader), "admitted_exact_image_loader_required")
    _event(resource_check, "begin")
    _event(resource_check, "bind_metadata")
    _bind_manifest(design, image_manifest)
    features.check_runtime()
    models = _load_models(resource_check, random_checkpoint_path)
    banks = {}
    for model_id in nested.MODEL_IDS:
        banks[model_id] = {}
        for layer in nested.LAYERS:
            shape = (nested.DEVELOPMENT_IMAGES, nested.FEATURE_WIDTHS[layer])
            _event(resource_check, "allocate_features", model_id=model_id, layer=layer,
                   shape=shape, byte_count=4 * shape[0] * shape[1])
            banks[model_id][layer] = np.empty(shape, dtype=np.float32)
    for index, image_id in enumerate(design.image_ids):
        value = _preprocess_one(image_loader, image_id, index, resource_check)
        for model_id in nested.MODEL_IDS:
            _event(resource_check, "infer_image", model_id=model_id, image_index=index,
                   shape=(1, 3, 224, 224))
            vectors = models[model_id](value.unsqueeze(0))
            require(isinstance(vectors, Mapping) and tuple(vectors) == nested.LAYERS,
                    "all_four_feature_layers_in_order_required")
            for layer in nested.LAYERS:
                vector = vectors[layer]
                width = nested.FEATURE_WIDTHS[layer]
                require(isinstance(vector, torch.Tensor) and vector.device.type == "cpu"
                        and vector.dtype == torch.float32 and tuple(vector.shape) == (1, width)
                        and bool(torch.isfinite(vector).all().item()),
                        "feature_representation_or_finiteness_changed")
                banks[model_id][layer][index] = vector.detach().numpy()[0]
            del vector, vectors
            _event(resource_check, "infer_image", when="after", model_id=model_id, image_index=index)
        del value
    # Release all backbones before snapshotting. Pop each mutable bank as its
    # immutable counterpart is retained: peak freezing adds one layer, rather
    # than holding mutable and frozen complete148,869,120-byte banks together.
    models.clear()
    panels = {}
    for model_id in nested.MODEL_IDS:
        layers = {}
        for layer in nested.LAYERS:
            value = banks[model_id][layer]
            _event(resource_check, "freeze_features", model_id=model_id, layer=layer,
                   shape=value.shape, byte_count=value.nbytes)
            layers[layer] = np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(value.shape)
            del banks[model_id][layer], value
        panels[model_id] = nested.FeaturePanel(design, MappingProxyType(layers))
    _event(resource_check, "complete", when="after")
    return MappingProxyType(panels)


def build_development_features(design, image_manifest, *, image_loader,
                               random_checkpoint_path, resource_check):
    """Deliver complete panels; see the internal callable's input contract.

    All loader/checkpoint/decoder/runtime failures become fixed operational
    codes without underlying exception text or a chained public traceback.
    The callable does not log or serialize source objects or exceptions.
    """
    try:
        return _build_development_features(design, image_manifest,
            image_loader=image_loader, random_checkpoint_path=random_checkpoint_path,
            resource_check=resource_check)
    except FeatureDeliveryError:
        raise
    except Exception:
        raise FeatureDeliveryError("private_development_feature_delivery_failed") from None
