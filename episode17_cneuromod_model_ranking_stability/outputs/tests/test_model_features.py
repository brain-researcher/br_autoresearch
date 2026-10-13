"""Source-free implementation checks; coordinator-owned Slurm execution only."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import model_features as method


class ModelFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        method.check_runtime()
        cls.model = method.build_backbone(method.RANDOM_ID)

    def test_preprocessing_matches_independent_PIL_bilinear_crop_and_normalization(self):
        y, x = np.indices((319, 480))
        pixels = np.stack(((7 * x + y) % 256, (x + 3 * y) % 256, (x * y) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(pixels, "RGB")
        image.getexif()[274] = 6  # No automatic EXIF rotation is part of the adopted transform.
        actual = method.preprocess(image)
        manual = image.resize((385, 256), resample=Image.Resampling.BILINEAR).crop((80, 16, 304, 240))
        reference = torch.from_numpy(np.asarray(manual).copy()).permute(2, 0, 1).float().div(255)
        reference = (reference - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        self.assertTrue(torch.equal(actual, reference))
        self.assertEqual(tuple(actual.shape), (3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(image.size, (480, 319))

    def test_preprocess_refuses_unmapped_input_forms(self):
        for invalid in ("/closed/source/image.jpg", Image.new("L", (224, 224)), Image.new("RGBA", (224, 224)),
                        torch.zeros(3, 224, 224)):
            with self.subTest(type=type(invalid).__name__), self.assertRaises(method.FeatureDefinitionError):
                method.preprocess(invalid)

    def test_random_seed_exact_constructor_order_and_cpu_RNG_preserved(self):
        before = torch.random.get_rng_state().clone()
        second = method.build_backbone(method.RANDOM_ID)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            reference = resnet50(weights=None, zero_init_residual=False)
            reference.fc = nn.Identity()
        for name, actual in self.model.backbone.state_dict().items():
            self.assertTrue(torch.equal(actual, second.backbone.state_dict()[name]), name)
            self.assertTrue(torch.equal(actual, reference.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 23508032)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_four_stage_GAP_outputs_freeze_BN_and_ignore_outer_autocast(self):
        captured = {}
        handles = []
        for layer in method.LAYERS:
            def capture(_module, _args, output, name=layer):
                captured[name] = output.detach().clone()
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertEqual(output.dtype, torch.float32)
            handles.append(getattr(self.model.backbone, layer).register_forward_hook(capture))
        buffers_before = {key: value.clone() for key, value in self.model.backbone.named_buffers()}
        batch = torch.linspace(-1, 1, 3 * 224 * 224, dtype=torch.float32).reshape(1, 3, 224, 224)
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                outputs = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(tuple(outputs), ("layer1", "layer2", "layer3", "layer4"))
        for name, shape in zip(method.LAYERS, method.SPATIAL_SHAPES):
            self.assertEqual(tuple(captured[name].shape), (1, *shape))
            self.assertEqual(tuple(outputs[name].shape), (1, shape[0]))
            self.assertTrue(torch.equal(outputs[name], captured[name].mean(dim=(-2, -1))))
            self.assertFalse(outputs[name].requires_grad)
        for key, value in self.model.backbone.named_buffers():
            self.assertTrue(torch.equal(value, buffers_before[key]), key)
        self.assertTrue(all(not module.training for module in self.model.modules()))

    def test_precision_context_restores_even_after_failure(self):
        original = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            for old_mode in ("highest", "high", "medium"):
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision(old_mode)
                old_tf32 = torch.backends.cuda.matmul.allow_tf32
                with method.full_float32("cpu"):
                    self.assertEqual(torch.get_float32_matmul_precision(), "highest")
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertEqual(torch.get_float32_matmul_precision(), old_mode)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                with self.assertRaisesRegex(RuntimeError, "fixture"):
                    with method.full_float32("cpu"):
                        raise RuntimeError("fixture")
                self.assertEqual(torch.get_float32_matmul_precision(), old_mode)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original[1]
            torch.backends.cudnn.allow_tf32 = original[2]
            torch.set_float32_matmul_precision(original[0])

    def test_training_wrong_dtype_or_shape_and_nonfinite_input_refused(self):
        with self.assertRaises(method.FeatureDefinitionError):
            self.model.train()
        for bad in (torch.zeros(1, 3, 112, 112), torch.zeros(1, 3, 224, 224, dtype=torch.float64),
                    torch.full((1, 3, 224, 224), float("nan"))):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(method.FeatureDefinitionError):
                self.model(bad)
        self.model.backbone.bn1.train()
        try:
            with self.assertRaises(method.FeatureDefinitionError):
                self.model(torch.zeros(1, 3, 224, 224))
        finally:
            self.model.eval()

    def test_DINO_and_SwAV_loading_allow_only_exact_backbone_and_named_heads(self):
        state = self.model.backbone.state_dict()
        for model_id in ("DINO_R50", "SWAV_R50_800"):
            target = method.fresh_resnet()
            if model_id == "DINO_R50":
                payload = dict(state)
            else:
                payload = {f"module.{key}": value for key, value in state.items()}
                payload.update({"module.projection_head.0.weight": torch.ones(2, 2), "module.prototypes.weight": torch.ones(2, 2)})
            method.load_state(model_id, target, payload)
            for key, value in target.state_dict().items():
                self.assertTrue(torch.equal(value, state[key]), key)
            del target
        for mutation in ("missing", "extra", "dtype", "shape", "collision"):
            target = method.fresh_resnet()
            payload = {f"module.{key}": value for key, value in state.items()}
            if mutation == "missing":
                payload.pop("module.conv1.weight")
            elif mutation == "extra":
                payload["module.unknown.weight"] = torch.zeros(1)
            elif mutation == "dtype":
                payload["module.conv1.weight"] = payload["module.conv1.weight"].double()
            elif mutation == "shape":
                payload["module.conv1.weight"] = payload["module.conv1.weight"][:1]
            else:
                payload["conv1.weight"] = payload["module.conv1.weight"]
            with self.subTest(mutation=mutation), self.assertRaises(method.FeatureDefinitionError):
                method.load_state("SWAV_R50_800", target, payload)
            del target

    def test_supervised_legacy_supplies_only_53_zero_counters(self):
        reference = method.fresh_resnet()
        # A plain old-format dictionary intentionally has no modern BN metadata.
        legacy = {key: value for key, value in reference.state_dict().items() if not key.endswith(".num_batches_tracked")}
        self.assertEqual(len(legacy), 267)
        target = method.fresh_resnet()
        method.load_state("SUP_R50_V1", target, legacy)
        counters = [value for key, value in target.state_dict().items() if key.endswith(".num_batches_tracked")]
        self.assertEqual(len(counters), 53)
        self.assertTrue(all(torch.count_nonzero(value).item() == 0 for value in counters))
        for key, value in target.state_dict().items():
            if key in legacy:
                self.assertTrue(torch.equal(value, legacy[key]), key)
        broken = dict(legacy)
        broken.pop("bn1.running_var")
        with self.assertRaises(method.FeatureDefinitionError):
            method.load_state("SUP_R50_V1", method.fresh_resnet(), broken)

    def test_random_builder_has_no_file_load_and_registered_load_is_safe_CPU_only(self):
        with patch.object(method.torch, "load", side_effect=AssertionError("random must not load a checkpoint")):
            model = method.build_backbone(method.RANDOM_ID)
        del model
        payload = dict(self.model.backbone.state_dict())
        with patch.object(method.Path, "is_file", return_value=True), \
             patch.object(method.Path, "stat", return_value=type("Metadata", (), {"st_size": 94345885})()), \
             patch.object(method.torch, "load", return_value=payload) as loader:
            model = method.build_backbone("DINO_R50")
        loader.assert_called_once_with(method.CHECKPOINTS / "dino_resnet50_pretrain.pth", map_location="cpu", weights_only=True)
        self.assertEqual(model.model_id, "DINO_R50")
        with self.assertRaises(method.FeatureDefinitionError):
            method.build_backbone("DEFAULT")


if __name__ == "__main__":
    unittest.main()
