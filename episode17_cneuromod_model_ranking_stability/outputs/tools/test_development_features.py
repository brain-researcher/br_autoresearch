"""Complete feature-delivery software fixtures; coordinated Slurm only.

Keep all2423 rows,480 concepts, all4 models and all16 real layer widths.
Replace only admitted image/model I/O and the already-qualified preprocessing
kernel. No acquired source, model checkpoint, private artifact or fit is read.
These fixtures test composition, not empirical feature/rank qualification.
"""
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import traceback
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import development_features as core
import nested_prediction as nested


def fixture():
    concepts = tuple(item for item in range(480) for _ in range(6 if item < 23 else 5))
    design = nested.DevelopmentDesign(
        tuple(f"images/concept-{item:03}/image-{index:04}.jpg" for index, item in enumerate(concepts)),
        tuple(f"concept-{item:03}" for item in concepts), ("development",) * 2423,
        tuple(f"fold{item // 80 + 1:02}" for item in concepts),
        tuple(f"block{item % 24 + 1:02}" for item in concepts))
    manifest = tuple(dict(provider_image_path=image, concept_name=design.concept_ids[index],
        role="development", outer_fold=design.fold_ids[index], development_block=design.block_ids[index],
        audit_block="", eligible_image="True", common_to_four="True")
        for index, image in enumerate(design.image_ids))
    return design, manifest


class FixtureBackbone(core.features.FeatureBackbone):
    def __init__(self, model_id, calls, failure=None):
        # Deliberately replace the already-qualified checkpoint/inference kernel,
        # while satisfying the fixed object's class/device/frozen interface.
        torch.nn.Module.__init__(self)
        self.model_id, self.calls, self.failure = model_id, calls, failure
        self.register_parameter("marker", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        self.eval()

    def forward(self, batch):
        if tuple(batch.shape) != (1, 3, 224, 224) or batch.dtype != torch.float32:
            raise AssertionError("changed batch representation")
        row = int(batch[0, 0, 0, 0].item())
        self.calls.append((self.model_id, row))
        # Binary fractions distinguish every row, model, layer and column.
        base = row + 4096 * nested.MODEL_IDS.index(self.model_id)
        vectors = {layer: torch.arange(nested.FEATURE_WIDTHS[layer], dtype=torch.float32)[None, :] / 8
                   + base + 512 * layer_index
                   for layer_index, layer in enumerate(nested.LAYERS)}
        if self.failure == "layer":
            vectors.pop("layer4")
        elif self.failure == "width":
            vectors["layer4"] = vectors["layer4"][:, :-1]
        elif self.failure == "dtype":
            vectors["layer4"] = vectors["layer4"].double()
        elif self.failure == "nonfinite":
            vectors["layer4"][0, 17] = float("nan")
        return vectors


class DevelopmentFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Do not patch the actual runtime guard: accidental login execution,
        # wrong package versions and unbounded threading remain refusals.
        core.features.check_runtime()
        cls.design, cls.manifest = fixture()

    def setUp(self):
        self.paths, self.closed, self.preprocessed, self.model_calls = [], [], [], []
        self.loaded, self.events = [], []
        self.indices = {path: index for index, path in enumerate(self.design.image_ids)}

    @contextmanager
    def loader(self, path):
        index = self.indices[path]  # Unexpected/non-development path refuses.
        self.paths.append(path)
        source = Image.new("RGB", (2, 2), color=(index % 256, index // 256, 7))
        source.format = "JPEG"
        try:
            yield source
        finally:
            source.close()
            self.closed.append(index)

    def preprocess(self, image):
        self.assertEqual(image.mode, "RGB")
        red, green, blue = image.getpixel((0, 0))
        self.assertEqual(blue, 7)
        index = red + 256 * green
        self.preprocessed.append((index, image))
        return torch.tensor(float(index), dtype=torch.float32).expand(3, 224, 224)

    @contextmanager
    def environment(self, *, model_failure=None, random_missing=False, wrong_model=False):
        def ordinary(path):
            self.assertEqual(path, core.pilot.RANDOM_CHECKPOINT)
            if random_missing:
                raise FileNotFoundError("private-source-detail")
            return SimpleNamespace(st_size=100)

        def load(feature_module, model_id, path):
            self.assertIs(feature_module, core.features)
            self.assertEqual(path, core.pilot.RANDOM_CHECKPOINT)
            self.assertEqual(len(self.paths), 0)
            self.loaded.append(model_id)
            result = FixtureBackbone(model_id, self.model_calls, model_failure)
            if wrong_model:
                result.model_id = "wrong"
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(core.pilot, "_ordinary_private_file", side_effect=ordinary))
            stack.enter_context(patch.object(core.pilot, "_load_backbone", side_effect=load))
            stack.enter_context(patch.object(core.features, "preprocess", side_effect=self.preprocess))
            yield

    def run_delivery(self, **kwargs):
        options = dict(image_loader=self.loader, random_checkpoint_path=core.pilot.RANDOM_CHECKPOINT,
                       resource_check=self.events.append)
        options.update(kwargs)
        return core.build_development_features(self.design, self.manifest, **options)

    def test_full_roster_fixed_models_layers_values_order_single_decode_and_immutability(self):
        with self.environment():
            panels = self.run_delivery()
        self.assertEqual(tuple(panels), nested.MODEL_IDS)
        self.assertEqual(self.loaded, list(nested.MODEL_IDS))
        self.assertEqual(tuple(self.paths), self.design.image_ids)
        self.assertEqual(self.closed, list(range(2423)))
        self.assertEqual([row for row, _ in self.preprocessed], list(range(2423)))
        self.assertEqual(self.model_calls, [(model, row) for row in range(2423) for model in nested.MODEL_IDS])
        for _, rgb in self.preprocessed:
            with self.assertRaises(ValueError):
                rgb.getpixel((0, 0))
        for model_index, model in enumerate(nested.MODEL_IDS):
            self.assertIs(panels[model].rows, self.design)
            self.assertEqual(tuple(panels[model].layers), nested.LAYERS)
            for layer_index, layer in enumerate(nested.LAYERS):
                actual = panels[model].layers[layer]
                width = nested.FEATURE_WIDTHS[layer]
                expected = (np.arange(2423, dtype=np.float32)[:, None] + 4096 * model_index
                            + 512 * layer_index + np.arange(width, dtype=np.float32)[None, :] / 8)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.dtype("float32"))
                with self.assertRaises(ValueError):
                    actual.setflags(write=True)
                with self.assertRaises(TypeError):
                    panels[model].layers[layer] = actual
        with self.assertRaises(TypeError):
            panels["other"] = panels[nested.MODEL_IDS[0]]
        allocations = [event for event in self.events if event.operation == "allocate_features"]
        self.assertEqual(sum(event.byte_count for event in allocations), 148869120)
        self.assertEqual(len(allocations), 16)
        self.assertEqual(self.events[-1].operation, "complete")

    def test_manifest_and_persisted_random_refuse_before_pixels_or_checkpoint_loads(self):
        mutations = ({"provider_image_path": self.design.image_ids[1]}, {"concept_name": "wrong"},
            {"role": "sealed_audit"}, {"eligible_image": "False"}, {"common_to_four": "False"},
            {"outer_fold": 1}, {"development_block": 1}, {"audit_block": "audit01"})
        with self.environment():
            for mutation in mutations:
                manifest = list(self.manifest)
                manifest[-1] = {**manifest[-1], **mutation}
                with self.subTest(mutation=mutation), self.assertRaises(core.FeatureDeliveryError):
                    core.build_development_features(self.design, manifest, image_loader=self.loader,
                        random_checkpoint_path=core.pilot.RANDOM_CHECKPOINT, resource_check=self.events.append)
            with self.assertRaises(core.FeatureDeliveryError):
                self.run_delivery(random_checkpoint_path="/unselected/random.pth")
        with self.environment(random_missing=True):
            with self.assertRaises(core.FeatureDeliveryError):
                self.run_delivery()
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.paths, [])

    def test_incorrect_backbone_or_any_incomplete_nonfinite_layer_aborts_whole_call(self):
        with self.environment(wrong_model=True):
            with self.assertRaises(core.FeatureDeliveryError):
                self.run_delivery()
        self.assertEqual(self.paths, [])
        for failure in ("layer", "width", "dtype", "nonfinite"):
            self.paths.clear()
            with self.subTest(failure=failure), self.environment(model_failure=failure):
                with self.assertRaises(core.FeatureDeliveryError):
                    self.run_delivery()
            self.assertEqual(self.paths, [self.design.image_ids[0]])

    def test_bad_image_and_private_loader_error_are_closed_and_never_exposed(self):
        @contextmanager
        def bad_image(_path):
            source = Image.new("RGB", (2, 2))
            source.format = "PNG"
            try:
                yield source
            finally:
                source.close()
                self.closed.append("bad")
        with self.environment():
            with self.assertRaisesRegex(core.FeatureDeliveryError, "source_image_format_or_pixel_cap"):
                self.run_delivery(image_loader=bad_image)
        self.assertEqual(self.closed, ["bad"])
        self.assertEqual(self.model_calls, [])

        @contextmanager
        def secret_error(_path):
            raise RuntimeError("PRIVATE-provider-password-detail")
            yield  # pragma: no cover - supplies a context manager before failure.
        with self.environment():
            try:
                self.run_delivery(image_loader=secret_error)
            except core.FeatureDeliveryError as error:
                rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                self.assertNotIn("PRIVATE-provider-password-detail", str(error))
                self.assertNotIn("PRIVATE-provider-password-detail", rendered)
                self.assertTrue(error.__suppress_context__)
            else:
                self.fail("private source exception was not refused")

        sources, converted = [], []

        @contextmanager
        def retained_source(path):
            with self.loader(path) as source:
                sources.append(source)
                yield source

        def preprocess_error(rgb):
            converted.append(rgb)
            raise RuntimeError("fixture preprocessing failure")

        with self.environment(), patch.object(core.features, "preprocess", side_effect=preprocess_error):
            with self.assertRaisesRegex(core.FeatureDeliveryError, "private_development_feature_delivery_failed"):
                self.run_delivery(image_loader=retained_source)
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(converted), 1)
        self.assertIsNot(sources[0], converted[0])
        for image in (sources[0], converted[0]):
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))
        self.assertEqual(self.paths, [self.design.image_ids[0]])
        self.assertEqual(self.closed, ["bad", 0])
        self.assertEqual(self.model_calls, [])

    def test_resource_refusal_precedes_metadata_models_allocations_pixels_and_inference(self):
        for operation in ("bind_metadata", "require_existing_random_checkpoint", "load_backbone",
                          "allocate_features", "load_image", "preprocess_image", "infer_image",
                          "freeze_features"):
            self.paths.clear()
            self.loaded.clear()
            self.preprocessed.clear()
            self.model_calls.clear()
            def stop(event):
                self.events.append(event)
                if event.operation == operation and event.when == "before":
                    raise RuntimeError("fixture resource refusal")
            with self.subTest(operation=operation), self.environment():
                with self.assertRaises(core.FeatureDeliveryError):
                    self.run_delivery(resource_check=stop)
            if operation == "freeze_features":
                self.assertEqual(len(self.model_calls), 4 * 2423)
                self.assertEqual(tuple(self.paths), self.design.image_ids)
                self.assertFalse(any(event.operation == "complete" for event in self.events))
                continue
            self.assertEqual(self.model_calls, [])
            if operation in ("bind_metadata", "require_existing_random_checkpoint", "load_backbone"):
                self.assertEqual(self.loaded, [])
            if operation not in ("preprocess_image", "infer_image"):
                self.assertEqual(self.paths, [])
            if operation != "infer_image":
                self.assertEqual(self.preprocessed, [])


if __name__ == "__main__":
    unittest.main()
