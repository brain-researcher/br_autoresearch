"""Full-state checkpoint checks using a compact fabricated prediction bank.

All2423 rows, four models, eight groups, six folds and84 objective entries
are retained; small response widths keep one-container I/O bounded. One producer
join check fits two small feature-only PCAs; no selection search, payload reader
or audit application is run. Execute only
in the coordinator's Slurm allocation; temporary files belong on scratch.
"""
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
import json
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction
import predictor_checkpoint as core


MAX_BYTES = 64 * 1024 * 1024


def fixture():
    concept = tuple(item for item in range(480) for _ in range(6 if item < 23 else 5))
    folds = (1, "1", 2, "two", 3, "three")
    blocks = (1, "1") + tuple(f"block-{index}" for index in range(2, 24))
    design = nested.DevelopmentDesign(tuple(f"image-{index:04}" for index in range(2423)),
        tuple(f"concept-{item:03}" for item in concept), ("development",) * 2423,
        tuple(folds[item // 80] for item in concept), tuple(blocks[item % 24] for item in concept))
    columns, baselines = {}, {}
    predictions = {model: {} for model in nested.MODEL_IDS}
    for group_index, group in enumerate(endpoint.GROUPS):
        width = 2 + group_index % 4
        columns[group] = endpoint.frozen_array(7 + 11 * np.arange(width), integer=True)
        baselines[group] = {}
        for model in nested.MODEL_IDS:
            predictions[model][group] = {}
        for stage_index, stage in enumerate(endpoint.STAGES):
            vector = .1 * group_index + .2 * stage_index + np.arange(width) / 100.
            # Six stored training means, each constant within its heldout fold.
            baseline = vector + np.array([folds.index(label) / 10. for label in design.fold_ids])[:, None]
            baselines[group][stage] = endpoint.frozen_array(baseline)
            for model_index, model in enumerate(nested.MODEL_IDS):
                predictions[model][group][stage] = endpoint.frozen_array(baseline
                    + (model_index + 1) * np.arange(2423)[:, None] / 10000.)

    def selection(setting, marker):
        objectives = {candidate: -3. - index / 1000. - marker / 100.
                      for index, candidate in enumerate(nested.SETTINGS)}
        objectives[setting] = -.5 - marker / 100.
        return nested.Selection(setting, nested._nested_map(objectives))

    def pca(layer, rank, training_images, marker):
        width = nested.FEATURE_WIDTHS[layer]
        mean = np.arange(width, dtype=float) / 1000. + marker
        loadings = np.zeros((width, rank))
        loadings[:rank] = np.eye(rank)
        singular = np.arange(min(training_images, width), 0, -1, dtype=float)
        if len(singular) == training_images:
            singular[-1] = 0.
        numerical_rank = min(training_images - 1, width)
        scale = np.sqrt(training_images / np.sum(singular[:rank] ** 2))
        return prediction.PCAFit(endpoint.frozen_array(mean), endpoint.frozen_array(loadings), float(scale),
            endpoint.frozen_array(singular), numerical_rank, training_images)

    selections, transforms, final = {}, {}, {}
    for model_index, model in enumerate(nested.MODEL_IDS):
        selections[model], transforms[model] = {}, {}
        for fold_index, fold in enumerate(folds):
            setting = nested.Setting(nested.LAYERS[(model_index + fold_index) % 4],
                nested.PCA_RANKS[(model_index + fold_index) % 3],
                nested.RIDGE_PENALTIES[(model_index + 2 * fold_index) % 7])
            selected = selection(setting, 10 * model_index + fold_index)
            if model_index == 0 and fold_index == 0:
                # Exact objective ties must retain the registered ordering:
                # earliest layer, smallest rank, then strongest penalty.
                setting = nested.Setting(nested.LAYERS[0], 32, 100.)
                selected = nested.Selection(setting, nested._nested_map({candidate: -2. for candidate in nested.SETTINGS}))
            selections[model][fold] = selected
            training_images = 2423 - sum(label == fold for label in design.fold_ids)
            transforms[model][fold] = pca(setting.layer, 64, training_images, model_index + fold_index / 10.)
        setting = nested.Setting(nested.LAYERS[model_index], nested.PCA_RANKS[model_index % 3],
                                 nested.RIDGE_PENALTIES[model_index + 2])
        encoding = pca(setting.layer, setting.rank, 2423, model_index + .25)
        cka = pca(setting.layer, 64, 2423, model_index + .25)
        ridges = {}
        for group_index, group in enumerate(endpoint.GROUPS):
            ridges[group] = {}
            width = len(columns[group])
            for stage_index, stage in enumerate(endpoint.STAGES):
                score_mean = np.zeros(setting.rank)
                score_mean[:2] = (.125, -.25)
                # Means are the same across all models, as in the producer.
                target_mean = .1 * group_index + .2 * stage_index + np.arange(width) / 100.
                coefficients = np.zeros((setting.rank, width))
                coefficients[0] = (model_index + 1) * (.2 + np.arange(width) / 100.)
                coefficients[1] = -.03 * (stage_index + 1) + .001 * group_index
                coefficients[-1, -1] = -0.  # Numeric payloads retain even signed zero.
                ridges[group][stage] = prediction.RidgeFit(endpoint.frozen_array(score_mean),
                    endpoint.frozen_array(target_mean), endpoint.frozen_array(coefficients), setting.penalty, 2423)
        final[model] = nested.FrozenPredictor(setting, encoding, cka, nested._nested_map(ridges),
            selection(setting, 100 + model_index), design.image_ids, design.concept_ids)
    return nested.NestedPredictionResult(design, nested._nested_map(columns), nested._nested_map(predictions),
        nested._nested_map(baselines), nested._nested_map(selections), nested._nested_map(transforms),
        nested._nested_map(final))


def rewrite_container(source, destination, *, omit=None, replacement=None, mutate_metadata=None, extra=False):
    """One bounded malformed copy; no extraction, pickle or retained archive bank."""
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream, zipfile.ZipFile(source) as before, \
            zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as after:
        for name in before.namelist():
            if name == omit:
                continue
            data = before.read(name)
            if name == "metadata.json" and mutate_metadata is not None:
                metadata = json.loads(data)
                mutate_metadata(metadata)
                data = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if replacement is not None and name == replacement[0]:
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, replacement[1], version=(1, 0), allow_pickle=False)
                data = buffer.getvalue()
            after.writestr(name, data)
        if extra:
            after.writestr("unexpected.npy", b"unregistered member")


class PredictorCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = os.environ.get("L_SCRATCH_JOB") or os.environ.get("L_SCRATCH")
        if not scratch or not Path(scratch).is_dir():
            raise RuntimeError("An existing L_SCRATCH_JOB or L_SCRATCH directory is required")
        cls.temporary = tempfile.TemporaryDirectory(prefix="ep17-checkpoint-tests-", dir=scratch)
        cls.directory = Path(cls.temporary.name)
        cls.result = fixture()
        cls.checkpoint = cls.directory / "complete.zip"
        with ExitStack() as stack:
            for name in ("fit_pca", "fit_ridge"):
                stack.enter_context(patch.object(prediction, name, side_effect=AssertionError("checkpoint refit")))
            cls.saved = core.save_prediction_checkpoint(cls.result, cls.checkpoint, max_bytes=MAX_BYTES,
                                                        resource_check=lambda event: None)

    @classmethod
    def tearDownClass(cls):
        del cls.result
        cls.temporary.cleanup()

    def assert_exact_state(self, original, loaded):
        self.assertIs(type(loaded), type(original))
        if isinstance(original, np.ndarray):
            self.assertEqual(loaded.dtype, original.dtype)
            self.assertEqual(loaded.shape, original.shape)
            self.assertEqual(loaded.tobytes(order="C"), original.tobytes(order="C"))
            self.assertFalse(loaded.flags.writeable)
            with self.assertRaises(ValueError):
                loaded.setflags(write=True)
        elif is_dataclass(original):
            for field in fields(original):
                self.assert_exact_state(getattr(original, field.name), getattr(loaded, field.name))
        elif isinstance(original, Mapping):
            self.assertEqual({(type(key), key) for key in loaded}, {(type(key), key) for key in original})
            for key in original:
                self.assert_exact_state(original[key], loaded[key])
        elif isinstance(original, tuple):
            self.assertEqual(len(original), len(loaded))
            for first, second in zip(original, loaded):
                self.assert_exact_state(first, second)
        else:
            self.assertEqual(loaded, original)

    def test_full_roundtrip_preserves_arrays_typed_bindings_and_selected_predictions(self):
        with patch.object(prediction, "fit_pca", side_effect=AssertionError("checkpoint refit")), \
                patch.object(prediction, "fit_ridge", side_effect=AssertionError("checkpoint refit")):
            loaded = core.load_prediction_checkpoint(self.checkpoint, max_bytes=MAX_BYTES,
                                                     resource_check=lambda event: None)
        self.assert_exact_state(self.result, loaded)
        self.assertEqual(self.saved.published_path, self.checkpoint)
        self.assertTrue(self.saved.partial_cleanup_complete)
        self.assertEqual(tuple(loaded.outer_selections[nested.MODEL_IDS[0]]), (1, "1", 2, "two", 3, "three"))
        self.assertEqual(loaded.outer_selections[nested.MODEL_IDS[0]][1].setting,
                         nested.Setting("layer1", 32, 100.))
        self.assertEqual(len(loaded.final_predictors[nested.MODEL_IDS[0]].training_concept_ids), 2423)
        with self.assertRaises(TypeError):
            loaded.predictions[nested.MODEL_IDS[0]] = {}
        with self.assertRaises(TypeError):
            loaded.final_predictors[nested.MODEL_IDS[0]].selection.objectives[nested.SETTINGS[0]] = 0.
        model, group, stage = nested.MODEL_IDS[2], endpoint.GROUPS[3], "C"
        before, after = self.result.final_predictors[model], loaded.final_predictors[model]
        features = np.tile(before.pca.feature_mean, (3, 1))
        features[:, 0] += np.array([.25, 1., -.5])
        features[:, 1] += np.array([2., -.125, .75])
        expected = (before.ridges[group][stage].target_mean
            + (before.pca.score_scale * np.array([.25, 1., -.5])[:, None] - .125)
                * before.ridges[group][stage].coefficients[0]
            + (before.pca.score_scale * np.array([2., -.125, .75])[:, None] + .25)
                * before.ridges[group][stage].coefficients[1])
        actual = after.ridges[group][stage].predict(after.pca.transform(features))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(actual, before.ridges[group][stage].predict(before.pca.transform(features)))
        self.assertEqual(stat.S_IMODE(self.checkpoint.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        with zipfile.ZipFile(self.checkpoint) as archive:
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            self.assertIn("metadata.json", archive.namelist())
            self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))
            self.assertTrue(all(name == "metadata.json" or name.endswith(".npy") for name in archive.namelist()))

    def test_incomplete_or_inconsistent_source_state_has_no_published_checkpoint(self):
        result, model = self.result, nested.MODEL_IDS[0]
        fold, group = result.design.fold_labels[0], endpoint.GROUPS[0]
        selected = result.outer_selections[model][fold]
        incomplete_objectives = dict(selected.objectives)
        del incomplete_objectives[nested.SETTINGS[-1]]
        wrong_winner = replace(selected, setting=nested.Setting("layer2", 32, 100.))
        selection_changes = [replace(selected, objectives=incomplete_objectives), wrong_winner]
        variants = [replace(result, predictions={key: value for key, value in result.predictions.items() if key != model})]
        variants += [replace(result, outer_selections={**result.outer_selections, model: {
            **result.outer_selections[model], fold: value}}) for value in selection_changes]
        final = result.final_predictors[model]
        variants += [replace(result, final_predictors={**result.final_predictors, model:
            replace(final, training_image_ids=tuple(reversed(final.training_image_ids)))}),
            replace(result, outer_cka_transforms={**result.outer_cka_transforms, model: {
                **result.outer_cka_transforms[model], fold:
                replace(result.outer_cka_transforms[model][fold], training_images=2423)}}),
            replace(result, baselines={**result.baselines, group: {
                **result.baselines[group], "B": result.baselines[group]["B"][:-1]}}),
            replace(result, baselines={**result.baselines, group: {
                **result.baselines[group], "B": np.asfortranarray(result.baselines[group]["B"])}})]
        with patch.object(prediction, "fit_pca", side_effect=AssertionError("repair by refitting")), \
                patch.object(prediction, "fit_ridge", side_effect=AssertionError("repair by refitting")):
            for index, altered in enumerate(variants):
                destination = self.directory / f"invalid-source-{index}.zip"
                with self.subTest(index=index), self.assertRaises(ValueError):
                    core.save_prediction_checkpoint(altered, destination, max_bytes=MAX_BYTES,
                                                    resource_check=lambda event: None)
                self.assertFalse(destination.exists())
                self.assertEqual(list(self.directory.glob("." + destination.name + ".*.partial")), [])

    def test_derived_rank_loading_views_roundtrip_without_changing_forward_values(self):
        model = nested.MODEL_IDS[0]
        final = self.result.final_predictors[model]
        fold = self.result.design.fold_labels[0]
        features = np.random.default_rng(1729).normal(size=(2423, nested.FEATURE_WIDTHS[final.setting.layer]))
        maximum = prediction.fit_pca(features, 128)
        encoding = nested.derive_rank_pca(maximum, features, 32)
        cka = nested.derive_rank_pca(maximum, features, 64)
        training_rows = np.array([label != fold for label in self.result.design.fold_ids])
        outer_features = features[training_rows]
        outer_maximum = prediction.fit_pca(outer_features, 128)
        outer = nested.derive_rank_pca(outer_maximum, outer_features, 64)
        for derived, source in ((encoding, maximum), (cka, maximum), (outer, outer_maximum)):
            self.assertFalse(derived.loadings.flags.c_contiguous)
            self.assertFalse(derived.loadings.flags.writeable)
            self.assertTrue(np.shares_memory(derived.loadings, source.loadings))
        result = replace(self.result,
            final_predictors=nested._nested_map({**self.result.final_predictors,
                model: replace(final, pca=encoding, cka_pca=cka)}),
            outer_cka_transforms=nested._nested_map({**self.result.outer_cka_transforms, model: {
                **self.result.outer_cka_transforms[model], fold: outer}}))
        destination = self.directory / "derived-rank-views.zip"
        with patch.object(prediction, "fit_pca", side_effect=AssertionError("checkpoint refit")), \
                patch.object(prediction, "fit_ridge", side_effect=AssertionError("checkpoint refit")):
            core.save_prediction_checkpoint(result, destination, max_bytes=MAX_BYTES,
                                            resource_check=lambda event: None)
            loaded = core.load_prediction_checkpoint(destination, max_bytes=MAX_BYTES,
                                                      resource_check=lambda event: None)
        self.assert_exact_state(result, loaded)
        probe_features = features[:3] + .025
        restored = loaded.final_predictors[model]
        for before, after in ((encoding, restored.pca), (cka, restored.cka_pca),
                              (outer, loaded.outer_cka_transforms[model][fold])):
            self.assertFalse(before.loadings.flags.c_contiguous)
            self.assertTrue(after.loadings.flags.c_contiguous)
            np.testing.assert_allclose(after.transform(probe_features), before.transform(probe_features),
                                       rtol=1e-13, atol=1e-13)
        group, stage = endpoint.GROUPS[0], "B"
        expected = final.ridges[group][stage].predict(encoding.transform(probe_features))
        actual = restored.ridges[group][stage].predict(restored.pca.transform(probe_features))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_missing_extra_wrong_dtype_and_incomplete_metadata_refuse_whole_load(self):
        missing = self.directory / "missing-array.zip"
        rewrite_container(self.checkpoint, missing, omit="final/m3/ridge/g7/D/coefficients.npy")
        with self.assertRaisesRegex(ValueError, "checkpoint_member_missing_or_reused"):
            core.load_prediction_checkpoint(missing, max_bytes=MAX_BYTES, resource_check=lambda event: None)
        extra = self.directory / "extra-array.zip"
        rewrite_container(self.checkpoint, extra, extra=True)
        with self.assertRaisesRegex(ValueError, "checkpoint_extra_members"):
            core.load_prediction_checkpoint(extra, max_bytes=MAX_BYTES, resource_check=lambda event: None)
        wrong_dtype = self.directory / "wrong-dtype.zip"
        rewrite_container(self.checkpoint, wrong_dtype,
            replacement=("columns/g0.npy", np.array([7, 18], dtype=np.int32)))
        with patch.object(np, "frombuffer", side_effect=AssertionError("allocation before dtype admission")):
            with self.assertRaisesRegex(ValueError, "checkpoint_array_header_changed"):
                core.load_prediction_checkpoint(wrong_dtype, max_bytes=MAX_BYTES, resource_check=lambda event: None)
        incomplete = self.directory / "incomplete-objectives.zip"
        rewrite_container(self.checkpoint, incomplete,
            mutate_metadata=lambda metadata: metadata["outer"][0][0]["selection"]["objectives"].pop())
        with patch.object(prediction, "fit_pca", side_effect=AssertionError("repair by refitting")):
            with self.assertRaisesRegex(ValueError, "all_84_selection_objectives_incomplete"):
                core.load_prediction_checkpoint(incomplete, max_bytes=MAX_BYTES, resource_check=lambda event: None)

    def test_byte_cap_and_resource_checks_precede_file_header_and_payload_work(self):
        capped = self.directory / "too-small.zip"
        with patch.object(core.tempfile, "mkstemp", side_effect=AssertionError("partial before byte admission")):
            with self.assertRaisesRegex(ValueError, "serialized_byte_cap_exceeded"):
                core.save_prediction_checkpoint(self.result, capped, max_bytes=1, resource_check=lambda event: None)
        self.assertFalse(capped.exists())
        with patch.object(core.zipfile, "ZipFile", side_effect=AssertionError("archive before byte admission")):
            with self.assertRaisesRegex(ValueError, "serialized_byte_cap_exceeded"):
                core.load_prediction_checkpoint(self.checkpoint, max_bytes=self.checkpoint.stat().st_size - 1,
                                                resource_check=lambda event: None)
        def refuse_begin(event):
            if event.operation == "preflight":
                raise RuntimeError("resource refused before file work")
        with patch.object(core, "_private_path", side_effect=AssertionError("path work before admission")):
            for operation, value in ((core.load_prediction_checkpoint, self.checkpoint),
                                     (core.save_prediction_checkpoint, self.result)):
                with self.assertRaisesRegex(RuntimeError, "resource refused before file work"):
                    if operation is core.load_prediction_checkpoint:
                        operation(value, max_bytes=MAX_BYTES, resource_check=refuse_begin)
                    else:
                        operation(value, capped, max_bytes=MAX_BYTES, resource_check=refuse_begin)
        def refuse_header(event):
            if event.operation == "read_header":
                raise RuntimeError("header resource refused")
        with patch.object(np.lib.format, "read_magic", side_effect=AssertionError("header read before admission")):
            with self.assertRaisesRegex(RuntimeError, "header resource refused"):
                core.load_prediction_checkpoint(self.checkpoint, max_bytes=MAX_BYTES, resource_check=refuse_header)
        read = zipfile.ZipExtFile.read
        def forbid_first_payload(member, count=-1):
            if member.name == "columns/g0.npy" and count == 16:
                raise AssertionError("payload allocation before admission")
            return read(member, count)
        def refuse_payload(event):
            if event.operation == "read_member":
                self.assertEqual((event.member, event.byte_count, event.shape), ("columns/g0.npy", 16, (2,)))
                raise RuntimeError("payload resource refused")
        with patch.object(zipfile.ZipExtFile, "read", new=forbid_first_payload):
            with self.assertRaisesRegex(RuntimeError, "payload resource refused"):
                core.load_prediction_checkpoint(self.checkpoint, max_bytes=MAX_BYTES, resource_check=refuse_payload)

    def test_failed_partial_is_preserved_and_existing_or_racing_destination_is_not_overwritten(self):
        with patch.object(core, "_describe", side_effect=AssertionError("existing checkpoint was not refused early")):
            with self.assertRaisesRegex(ValueError, "checkpoint_destination_exists"):
                core.save_prediction_checkpoint(self.result, self.checkpoint, max_bytes=MAX_BYTES,
                                                resource_check=lambda event: None)
        destination = self.directory / "refused-write.zip"
        def refuse_write(event):
            if event.operation == "write_member":
                raise RuntimeError("write resource refused")
        with patch.object(np.lib.format, "write_array", side_effect=AssertionError("array write before admission")):
            with self.assertRaisesRegex(RuntimeError, "write resource refused"):
                core.save_prediction_checkpoint(self.result, destination, max_bytes=MAX_BYTES, resource_check=refuse_write)
        self.assertFalse(destination.exists())
        partials = list(self.directory.glob("." + destination.name + ".*.partial"))
        self.assertEqual(len(partials), 1)
        self.assertEqual(stat.S_IMODE(partials[0].stat().st_mode), 0o600)
        with self.assertRaisesRegex(ValueError, "partial_checkpoint_not_loadable_or_publishable"):
            core.load_prediction_checkpoint(partials[0], max_bytes=MAX_BYTES, resource_check=lambda event: None)
        racing = self.directory / "racing-destination.zip"
        existing = b"caller-created destination must remain unchanged"
        def install_destination(event):
            if event.operation == "publish_checkpoint":
                descriptor = os.open(racing, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(existing)
        with self.assertRaises(FileExistsError):
            core.save_prediction_checkpoint(self.result, racing, max_bytes=MAX_BYTES, resource_check=install_destination)
        self.assertEqual(racing.read_bytes(), existing)
        complete_partials = list(self.directory.glob("." + racing.name + ".*.partial"))
        self.assertEqual(len(complete_partials), 1)
        with patch.object(core.zipfile, "ZipFile", side_effect=AssertionError("reserved partial archive opened")):
            with self.assertRaisesRegex(ValueError, "partial_checkpoint_not_loadable_or_publishable"):
                core.load_prediction_checkpoint(complete_partials[0], max_bytes=MAX_BYTES,
                                                resource_check=lambda event: None)
        committed = self.directory / "committed-with-partial.zip"
        events = []
        with patch.object(Path, "unlink", side_effect=OSError("test partial cleanup refused")):
            saved = core.save_prediction_checkpoint(self.result, committed, max_bytes=MAX_BYTES,
                                                    resource_check=events.append)
        self.assertEqual(saved.published_path, committed)
        self.assertFalse(saved.partial_cleanup_complete)
        self.assertEqual(events[-1].operation, "publish_checkpoint")
        self.assertEqual(len(list(self.directory.glob("." + committed.name + ".*.partial"))), 1)
        # Failure to remove the extra link does not invalidate the committed
        # result or authorize repeating expensive fitting/publication.
        loaded = core.load_prediction_checkpoint(committed, max_bytes=MAX_BYTES, resource_check=lambda event: None)
        np.testing.assert_array_equal(loaded.predictions[nested.MODEL_IDS[0]][endpoint.GROUPS[0]]["B"],
                                      self.result.predictions[nested.MODEL_IDS[0]][endpoint.GROUPS[0]]["B"])


if __name__ == "__main__":
    unittest.main()
