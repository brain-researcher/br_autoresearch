"""Focused amended-comparator checks; execute only in coordinated Slurm.

All responses here are small invented arrays. No empirical asset, model,
decoder, score, admission record or scheduler is read.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import brain_loss as original
import brain_loss_constant as brain
import final_brain_artifact as original_final
import final_brain_constant_artifact as final


def inputs(constants=False):
    ids = tuple(f"image-{index:04d}" for index in range(897))
    t = np.arange(897, dtype=np.float32)
    x = np.column_stack((t % 31 - 15, t % 17 - 8, t % 23 - 11))
    y = np.column_stack((2 * x[:, 0] - np.float32(.3) * x[:, 1],
                         np.float32(.25) * x[:, 2] + x[:, 1]))
    source_ids, target_ids = (12, 3, 8), (900, 4)
    if constants:
        x = np.column_stack((x, np.full(897, 4, dtype=np.float32)))
        y = np.column_stack((y, np.full(897, -2, dtype=np.float32)))
        source_ids, target_ids = source_ids + (1,), target_ids + (99,)
    source, target = original.PAIR_ORDER[5]
    return (original.TauTrainResponses(source, ids, source_ids, x),
            original.TauTrainResponses(target, ids, target_ids, y),
            original.make_common_folds(ids))


def fit(source, target, folds, fitter=brain.fit_brain_pair):
    return fitter(source, target, folds, source_native_ids=source.native_voxel_ids,
                  target_native_ids=target.native_voxel_ids, check=lambda: None)


class BrainConstantComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.target, cls.folds = inputs(constants=True)
        cls.completed = fit(cls.source, cls.target, cls.folds)

    def prepare(self, completed=None):
        completed = self.completed if completed is None else completed
        return final.prepare_final_brain_constant_artifact(completed,
            expected_pair=original.PAIR_ORDER[5], authenticated_tau_train_image_ids=completed.image_ids[::-1],
            source_native_ids=completed.source_native_voxel_ids,
            target_native_ids=completed.target_native_voxel_ids, check=lambda: None)

    def budget(self):
        return SimpleNamespace(allocation={"job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname().split(".")[0], "job_end_epoch": time.time() + 120},
            deadline=time.monotonic() + 60, check=lambda: None)

    def test_exact_guard_preserves_tiny_positive_and_refuses_other_invalid_cases(self):
        values = np.array([[3, -1e-15], [3, 1e-15], [3, -1e-15], [3, 1e-15]], dtype=np.float32)
        state = brain.normalization_state(values, "guard")
        raw = np.std(values, axis=0, ddof=1)[None, :]
        np.testing.assert_array_equal(state.raw_sd, raw)
        np.testing.assert_array_equal(state.effective_scale, np.array([[1, raw[0, 1]]], dtype=np.float32))
        self.assertGreater(raw[0, 1], 0)
        self.assertLess(raw[0, 1], 1e-12)
        self.assertEqual(state.replacement_count, 1)
        np.testing.assert_array_equal(state.replacement_mask, [[True, False]])
        self.assertTrue(all(not value.flags.writeable for value in
            (state.mean, state.raw_sd, state.effective_scale, state.replacement_mask)))
        underflow = np.array([[-1e-30], [1e-30]], dtype=np.float32)
        self.assertEqual(float(np.std(underflow, axis=0, ddof=1)[0]), 0)
        with self.assertRaisesRegex(brain.BrainLossNotEvaluable, "zero_sd_nonconstant"):
            brain.normalization_state(underflow, "varying")
        # Axis-zero float32 accumulation rounds this constant's mean; squared
        # deviations underflow. Exact constancy alone must not admit the scale.
        rounded_constant = np.full((717, 3), np.float32(1e-30), dtype=np.float32)
        rounded_mean = np.mean(rounded_constant, axis=0)
        self.assertTrue(np.any(rounded_mean != rounded_constant[0]))
        self.assertTrue(np.all(np.std(rounded_constant, axis=0, ddof=1) == 0))
        with self.assertRaisesRegex(brain.BrainLossNotEvaluable, "constant_mean_mismatch"):
            brain.normalization_state(rounded_constant, "rounded")
        for bad in (np.array([[np.nan], [0]], dtype=np.float32),
                    np.array([[np.inf], [0]], dtype=np.float32),
                    np.array([[3e38], [-3e38]], dtype=np.float32)):
            with self.subTest(bad=bad), self.assertRaises(brain.BrainLossNotEvaluable):
                brain.normalization_state(bad, "invalid")

    def test_positive_only_fit_is_exactly_original_comparator(self):
        source, target, folds = inputs()
        adapted = fit(source, target, folds)
        baseline = fit(source, target, folds, original.fit_brain_pair)
        self.assertEqual(adapted.summary, baseline.summary)
        np.testing.assert_array_equal(adapted.matrix, baseline.matrix)
        for axis in ("source", "target"):
            state = getattr(adapted, axis + "_normalization")
            np.testing.assert_array_equal(state.mean, getattr(baseline, axis + "_mean"))
            np.testing.assert_array_equal(state.raw_sd, getattr(baseline, axis + "_std"))
            np.testing.assert_array_equal(state.effective_scale, state.raw_sd)
            self.assertEqual(state.replacement_count, 0)
        self.assertFalse(isinstance(adapted, original.BrainLossFit))

    def test_all_51_fits_use_fold_local_guard_and_full_target_denominator(self):
        source, target, folds = inputs()
        values = np.column_stack((target.values, np.full(897, 2, dtype=np.float32)))
        values[np.asarray(folds.validation_indices[0]), -1] = 3
        target_extra = replace(target, values=values, native_voxel_ids=target.native_voxel_ids + (99,))
        with mock.patch.object(original, "_solve_primal", wraps=original._solve_primal) as solve:
            expanded = fit(source, target_extra, folds)
        self.assertEqual(solve.call_count, 51)
        self.assertEqual(tuple(call.args[2] for call in solve.call_args_list[:50]), original.ALPHAS * 5)
        self.assertEqual(solve.call_args_list[-1].args[2], expanded.summary.selected_alpha)
        self.assertEqual(expanded.fold_target_normalization[0].replacement_count, 1)
        self.assertEqual(expanded.fold_target_normalization[0].mean[0, -1], 2)
        self.assertEqual(expanded.fold_target_normalization[0].effective_scale[0, -1], 1)
        self.assertEqual(expanded.target_normalization.replacement_count, 0)
        self.assertGreater(expanded.target_normalization.mean[0, -1], 2)
        reduced = fit(source, target, folds)
        observed = np.asarray(expanded.summary.fold_mse_by_alpha)[:, 0]
        expected = (2 * np.asarray(reduced.summary.fold_mse_by_alpha)[:, 0] + 1) / 3
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(tuple(state.fitting_images for state in expanded.fold_target_normalization),
                         tuple(897 - count for count in original.FOLD_SIZES))
        # Entirely constant, exactly centered axes still retain the full matrix
        # and the existing smaller-alpha tie rule.
        constant_source = replace(source, values=np.full_like(source.values, 4))
        constant_target = replace(target, values=np.full_like(target.values, -2))
        constant_fit = fit(constant_source, constant_target, folds)
        self.assertEqual(constant_fit.summary.selected_alpha, 620)
        self.assertEqual(constant_fit.matrix.shape, (4, 2))
        np.testing.assert_array_equal(constant_fit.matrix, 0)

    def test_distinct_artifact_roundtrip_preserves_all_fold_and_final_normalizers(self):
        artifact = self.prepare()
        with self.assertRaises(ValueError):
            original_final._validate_artifact(artifact, lambda: None)
        with self.assertRaises(ValueError):
            original_final.prepare_final_brain_artifact(self.completed,
                expected_pair=original.PAIR_ORDER[5], authenticated_tau_train_image_ids=self.completed.image_ids,
                source_native_ids=self.completed.source_native_voxel_ids,
                target_native_ids=self.completed.target_native_voxel_ids, check=lambda: None)
        source, target, folds = inputs()
        old_fit = fit(source, target, folds, original.fit_brain_pair)
        old_artifact = original_final.prepare_final_brain_artifact(old_fit,
            expected_pair=original.PAIR_ORDER[5], authenticated_tau_train_image_ids=old_fit.image_ids,
            source_native_ids=old_fit.source_native_voxel_ids,
            target_native_ids=old_fit.target_native_voxel_ids, check=lambda: None)
        with self.assertRaises(ValueError):
            final._validate_artifact(old_artifact, lambda: None)
        with self.assertRaises(ValueError):
            self.prepare(old_fit)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "constant.pt"
            final.write_final_brain_constant_artifact(path, artifact, budget=self.budget(), maximum_bytes=200_000)
            restored = final.load_final_brain_constant_artifact(path, budget=self.budget(), maximum_bytes=200_000)
        self.assertEqual(restored["kind"], final.KIND)
        self.assertEqual(restored["summary"], artifact["summary"])
        np.testing.assert_array_equal(restored["matrix"].numpy(), self.completed.matrix)
        for axis in ("source", "target"):
            states = restored["fold_" + axis + "_normalization"] + (restored[axis + "_normalization"],)
            originals = getattr(self.completed, "fold_" + axis + "_normalization") + (
                getattr(self.completed, axis + "_normalization"),)
            for saved, state in zip(states, originals):
                self.assertEqual(saved["replacement_count"], 1)
                self.assertEqual(saved["fitting_images"], state.fitting_images)
                for name in ("mean", "raw_sd", "effective_scale", "replacement_mask"):
                    np.testing.assert_array_equal(saved[name].numpy(), getattr(state, name))

    def test_inconsistent_scale_mask_fold_count_and_publication_are_refused(self):
        artifact = self.prepare()
        for field, value in (("raw_sd", -1.), ("effective_scale", 2.),
                             ("replacement_mask", False), ("replacement_count", 0),
                             ("fitting_images", 897)):
            broken = deepcopy(artifact)
            state = broken["fold_target_normalization"][0]
            if field in ("replacement_count", "fitting_images"):
                state[field] = value
            else:
                state[field][0, -1] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                final._validate_artifact(broken, lambda: None)
        for existing in ("final", "partial"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "constant.pt"
                partial = path.with_name("constant.pt.partial")
                retained = path if existing == "final" else partial
                retained.write_bytes(b"preserved")
                with self.assertRaises(FileExistsError):
                    final.write_final_brain_constant_artifact(path, artifact,
                        budget=self.budget(), maximum_bytes=200_000)
                self.assertEqual(retained.read_bytes(), b"preserved")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bounded.pt"
            with self.assertRaises((ValueError, RuntimeError)):
                final.write_final_brain_constant_artifact(path, artifact, budget=self.budget(), maximum_bytes=512)
            self.assertFalse(path.exists())
            self.assertLessEqual(path.with_name("bounded.pt.partial").stat().st_size, 512)

    def test_physical_prediction_restores_target_before_separate_decoder_normalization(self):
        sample = self.source.values[:7].copy()
        artifact = self.prepare()
        physical = brain.predict_physical(self.completed, sample,
            source_native_ids=self.source.native_voxel_ids, check=lambda: None)
        restored = final.predict_physical(artifact, sample,
            source_native_ids=self.source.native_voxel_ids, check=lambda: None)
        np.testing.assert_array_equal(physical, restored)
        self.assertEqual(physical.dtype, np.float64)
        self.assertEqual(physical.shape, (7, 3))
        np.testing.assert_array_equal(physical[:, -1], -2)
        source = self.completed.source_normalization
        target = self.completed.target_normalization
        standardized = original._predict_standardized(
            (sample - source.mean) / source.effective_scale, self.completed.matrix)
        expected = standardized * target.effective_scale + target.mean
        np.testing.assert_array_equal(physical, expected)
        # A separate, nonidentical decoder moment transform is still required.
        decoder_mean = np.array([[7., 8., 9.]], dtype=np.float32)
        decoder_scale = np.array([[2., 3., 4.]], dtype=np.float32)
        decoder_input = (physical - decoder_mean) / decoder_scale
        self.assertEqual(decoder_input[0, -1], -2.75)
        self.assertFalse(np.array_equal(decoder_input, standardized))
        with self.assertRaisesRegex(ValueError, "native_voxel_order"):
            brain.predict_physical(self.completed, sample,
                source_native_ids=self.source.native_voxel_ids[::-1], check=lambda: None)

    def test_panel_retains_all_twenty_dispositions_and_propagates_resource_failure(self):
        source, _, folds = inputs()
        responses = {person: replace(source, participant=person) for person in original.PARTICIPANTS}
        axes = {person: source.native_voxel_ids for person in original.PARTICIPANTS}
        visited, published = [], []
        def pair(source, target, *args, **kwargs):
            current = (source.participant, target.participant)
            visited.append(current)
            if current == original.PAIR_ORDER[2]:
                raise brain.BrainLossNotEvaluable("exact_guard_failed")
            return replace(self.completed, summary=replace(self.completed.summary,
                source=current[0], target=current[1]))
        with mock.patch.object(brain, "fit_brain_pair", side_effect=pair):
            summaries = brain.fit_brain_panel(responses, authenticated_tau_train_image_ids=folds.image_ids,
                expected_native_axes=axes, check=lambda: None, on_pair_completed=published.append)
        self.assertEqual(tuple(visited), original.PAIR_ORDER)
        self.assertEqual(len(summaries), 20)
        self.assertEqual(len(published), 19)
        self.assertFalse(summaries[2].available)
        self.assertEqual(summaries[2].reason, "exact_guard_failed")
        with mock.patch.object(brain, "fit_brain_pair", side_effect=TimeoutError("deadline")):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                brain.fit_brain_panel(responses, authenticated_tau_train_image_ids=folds.image_ids,
                    expected_native_axes=axes, check=lambda: None, on_pair_completed=published.append)


if __name__ == "__main__":
    unittest.main()
