"""EP17's fixed feature definition, without image readers or feature writers.

Only the coordinator may execute this on compute. Actual stimulus adapters,
feature pilots, role release and neural linking remain separate operations.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
import os
from pathlib import Path
import socket

import PIL
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import resnet50


EPISODE = Path(__file__).resolve().parents[2]
CHECKPOINTS = EPISODE / "outputs/cache/model_checkpoints"
ASSETS = {
    "DINO_R50": ("dino_resnet50_pretrain.pth", 94345885),
    "SWAV_R50_800": ("swav_800ep_pretrain.pth.tar", 113703565),
    "SUP_R50_V1": ("resnet50-0676ba61.pth", 102530333),
}
RANDOM_ID = "RANDOM_R50_S17"
SEED = 17
PILLOW_VERSION = "12.2.0"
PAIR_SIGNS = {"P1": ("DINO_R50", "SUP_R50_V1"), "P2": ("SWAV_R50_800", "SUP_R50_V1"),
              "P3": ("DINO_R50", "SWAV_R50_800"), "F1": ("SUP_R50_V1", RANDOM_ID)}
LAYERS = ("layer1", "layer2", "layer3", "layer4")
SPATIAL_SHAPES = ((256, 56, 56), (512, 28, 28), (1024, 14, 14), (2048, 7, 7))
BACKBONE_PARAMETERS = 23508032
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FeatureDefinitionError(RuntimeError):
    """Fixed operational codes only, never values or sample representations."""


def require(condition, code):
    if not condition:
        raise FeatureDefinitionError(code)


def check_runtime():
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(), "compute_allocation_required")
    require(torch.__version__.split("+")[0] == "2.4.1" and torchvision.__version__.split("+")[0] == "0.19.1",
            "registered_model_library_versions_required")
    require(PIL.__version__ == PILLOW_VERSION, "registered_Pillow_version_required")
    require(torch.get_default_dtype() == torch.float32, "float32_default_required")
    require(torch.get_num_threads() == 1 and all(os.environ.get(name) == "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
            "one_compute_thread_required")


def runtime_metadata():
    """Report the implementation binding selected before any feature cache."""
    check_runtime()
    return {"torch": torch.__version__, "torchvision": torchvision.__version__, "pillow": PIL.__version__,
            "dtype": "float32", "autocast": False, "cuda_matmul_tf32": False, "cudnn_tf32": False,
            "float32_matmul_precision": "highest", "pillow_cache_binding_selected": True,
            "source_image_reader_provided": False, "feature_cache_writer_provided": False}


def preprocess(image):
    """Already decoded RGB only: no filesystem access, EXIF rotation or adaptive crop."""
    check_runtime()
    require(isinstance(image, Image.Image) and image.mode == "RGB" and min(image.size) > 0,
            "decoded_RGB_image_required")
    transform = transforms.Compose((
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ))
    result = transform(image)
    require(result.dtype == torch.float32 and tuple(result.shape) == (3, 224, 224), "preprocessing_shape_or_dtype_changed")
    return result


@contextmanager
def full_float32(device_type):
    """Set the adopted policy during inference, then restore caller settings."""
    require(device_type in ("cpu", "cuda"), "unsupported_feature_device")
    old_precision = torch.get_float32_matmul_precision()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        # The boolean setter can itself change the precision string. Restore
        # the exact prior mode last, including a caller's distinct 'medium'.
        torch.set_float32_matmul_precision(old_precision)


def fresh_resnet():
    """Exact CPU constructor order, isolated from caller CPU RNG; nothing persisted."""
    check_runtime()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        # The normal classifier is constructed before it is removed. Its RNG
        # consumption is part of the fixed random-network definition.
        with torch.device("cpu"):
            model = resnet50(weights=None, zero_init_residual=False)
    require(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters()),
            "fresh_constructor_representation_changed")
    return model


def load_state(model_id, model, payload):
    """Documented acquisition semantics, with synthetic-state fixtures.

    Existing acquisition already checked learned values and shapes. This is
    the actual future loader, not a repeated checkpoint-qualification job.
    """
    require(model_id in ASSETS and isinstance(payload, Mapping), "unregistered_or_invalid_checkpoint")
    if model_id == "SWAV_R50_800" and "state_dict" in payload:
        payload = payload["state_dict"]
    require(isinstance(payload, Mapping) and all(isinstance(key, str) for key in payload), "invalid_checkpoint_mapping")
    if model_id == "SUP_R50_V1":
        expected = model.state_dict()
        counters = {key for key in expected if key.endswith(".num_batches_tracked")}
        require(len(counters) == 53 and set(payload) == set(expected) - counters, "supervised_legacy_key_set_changed")
        require(all(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(expected[key].shape) and
                    value.dtype == expected[key].dtype
                    for key, value in payload.items()), "supervised_parameter_shape_changed")
        # Keep PyTorch's strict legacy migration: no learned parameter/statistic
        # is supplied from random initialization. Only the 53 zero counters are.
        model.load_state_dict(payload, strict=True)
        require(all(torch.count_nonzero(model.state_dict()[key]).item() == 0 for key in counters),
                "legacy_batch_counters_not_zero")
        model.fc = nn.Identity()
        return
    model.fc = nn.Identity()
    expected = model.state_dict()
    normalized = OrderedDict()
    for key, value in payload.items():
        name = key[len("module."):] if model_id == "SWAV_R50_800" and key.startswith("module.") else key
        if model_id == "SWAV_R50_800" and name.startswith(("projection_head.", "prototypes.")):
            continue
        require(name not in normalized, "checkpoint_prefix_collision")
        normalized[name] = value
    require(set(normalized) == set(expected), "backbone_key_set_changed")
    require(all(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(expected[key].shape) and
                value.dtype == expected[key].dtype
                for key, value in normalized.items()), "backbone_parameter_shape_changed")
    model.load_state_dict(normalized, strict=True)


class FeatureBackbone(nn.Module):
    """Four stage-end arithmetic spatial means; frozen weights and BatchNorm."""
    def __init__(self, model_id, model):
        super().__init__()
        require(model_id in (*ASSETS, RANDOM_ID) and isinstance(model.fc, nn.Identity), "registered_headless_backbone_required")
        require(sum(parameter.numel() for parameter in model.parameters()) == BACKBONE_PARAMETERS,
                "backbone_parameter_count_changed")
        self.model_id = model_id
        self.backbone = model
        self.backbone.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        require(mode is False, "feature_backbone_training_prohibited")
        return super().train(False)

    @torch.inference_mode()
    def forward(self, batch):
        check_runtime()
        require(isinstance(batch, torch.Tensor) and batch.ndim == 4 and batch.shape[0] > 0 and
                tuple(batch.shape[1:]) == (3, 224, 224) and batch.dtype == torch.float32, "feature_input_representation_changed")
        require(all(not module.training for module in self.modules()), "feature_backbone_training_prohibited")
        require(all(parameter.dtype == torch.float32 and parameter.device == batch.device
                    for parameter in self.parameters()), "feature_parameter_precision_or_device_changed")
        require(torch.isfinite(batch).all().item(), "nonfinite_feature_input")
        result = OrderedDict()
        with full_float32(batch.device.type):
            net = self.backbone
            value = net.maxpool(net.relu(net.bn1(net.conv1(batch))))
            for name, shape in zip(LAYERS, SPATIAL_SHAPES):
                value = getattr(net, name)(value)
                require(tuple(value.shape[1:]) == shape and value.dtype == torch.float32, "feature_stage_shape_or_precision_changed")
                vector = value.mean(dim=(-2, -1))
                require(torch.isfinite(vector).all().item(), "nonfinite_feature_vector")
                result[name] = vector
        return result


def build_backbone(model_id):
    """No downloads and no outputs. Only the three fixed paths can be loaded."""
    require(model_id in (*ASSETS, RANDOM_ID), "unregistered_model_id")
    model = fresh_resnet()
    if model_id == RANDOM_ID:
        model.fc = nn.Identity()
    else:
        filename, expected_bytes = ASSETS[model_id]
        path = CHECKPOINTS / filename
        require(path.is_file() and path.stat().st_size == expected_bytes, "registered_checkpoint_missing_or_size_changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        load_state(model_id, model, payload)
        del payload
    return FeatureBackbone(model_id, model)
