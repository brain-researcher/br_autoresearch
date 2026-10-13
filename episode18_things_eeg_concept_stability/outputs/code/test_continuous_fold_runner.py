"""Focused composition checks; run only in coordinator-owned Slurm compute.

Small algebra fixtures exercise the real 72-sequence/36-split roster without
source EEG, images or feature payloads. These are software checks, not a
simulation study or qualification of empirical preprocessing/EDF choices.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as runner
from continuous_sequence_model import EventTerm, JointRidgeFit, SequenceDesign


CHANNELS = tuple(f"channel_{index}" for index in range(62))
LIMITS = runner.SolverLimits(atol=1e-11, btol=1e-11, conlim=1e12, maxiter=200)


def temporal_basis(lags):
    return np.ones((len(lags), 1))


def fixture():
    sequences = []
    for block in range(12):
        for within in range(6):
            count = 12 + within % 3
            times = np.arange(count) * .004
            rows, columns = np.arange(count)[:, None], np.arange(62)[None, :]
            values = (np.sin(.17 * rows + .07 * columns + .11 * block)
                      + .03 * (within + 1) * np.cos(.2 * columns) * rows)
            mask = np.ones(count, dtype=bool)
            mask[[0, -1]] = False
            sequences.append(runner.PhysicalSequence(f"b{block}s{within}", block, within,
                times, values, CHANNELS, mask, (block, within)))
    return tuple(sequences)


def dense_fit(design, targets, weights, penalties, limits, check):
    """Independent normal equation reference, not the matrix-free fit wrapper."""
    check()
    matrix = design @ np.eye(design.shape[1])
    gram = matrix.T @ (weights[:, None] * matrix) + np.diag(penalties)
    beta = np.linalg.solve(gram, matrix.T @ (weights[:, None] * targets))
    return JointRidgeFit(beta, (2,) * 61, (1,) * 61)


def preparation_factory(names, calls=None, whiten_calls=None):
    def prepare(training, *, split, inner_results, budget_check):
        budget_check()
        actual = tuple(item.sequence_id for item in training)
        if calls is not None:
            calls.append((split, actual, inner_results))
        if {item.block for item in training} != set(split.training_blocks):
            raise AssertionError("Preparation received another split's sequences")
        if split.inner_index is not None and inner_results:
            raise AssertionError("An inner preparation received later fold feedback")
        if split.inner_index is None and len(inner_results) != 5:
            raise AssertionError("Outer selection did not receive all five inner folds")

        def build(view):
            if hasattr(view, "sensor_samples") or hasattr(view, "retained_mask"):
                raise AssertionError("Design builder was given EEG or a mutable scoring mask")
            times = view.sample_times_seconds
            nuisance = np.column_stack((np.ones(len(times)), times / .05))
            result = {}
            for name in names:
                if name in ("nuisance", "reduced_task_nuisance"):
                    terms = {}
                    matrix = nuisance if name == "nuisance" else nuisance[:, :1]
                else:
                    width = 1 if name == "image" else 2
                    features = np.array([[1., 2.], [2., -.5]])[:, :width]
                    terms = {"features": EventTerm(np.array([.001, .013]), features,
                                                   temporal_basis, 1, (0., .8))}
                    matrix = nuisance
                result[name] = SequenceDesign(view.sequence_id, times, matrix, terms)
            return result

        def whitening(residuals, weights):
            if whiten_calls is not None:
                whiten_calls.append((split, residuals.copy(), weights.copy()))
            return runner.WhiteningChoices(True, float(weights.sum()), np.eye(61), .3)

        penalties = {}
        for name in names:
            width = 2 if name == "nuisance" else 1 if name == "reduced_task_nuisance" else 3 if name == "image" else 4
            penalties[name] = np.array([0.] + [.1] * (width - 1))
        return runner.FittedFoldPreparation(actual, build, penalties, whitening, .75)
    return prepare


class ContinuousFoldRunnerTests(unittest.TestCase):
    def test_complete_nested_roster_and_both_fixed_model_panels(self):
        sequences = fixture()
        for phase, names in (("baseline", runner.BASELINE_MODELS),
                             ("fixed_panel", runner.FIXED_PANEL_MODELS)):
            calls, retained = [], []
            with patch.object(runner, "_fit", side_effect=dense_fit):
                result = runner.run_participant_folds(participant_id="fixture", sequences=sequences,
                    frozen_channel_order=CHANNELS, phase=phase,
                    prepare_training=preparation_factory(names, calls), solver_limits=LIMITS,
                    retain_fit=retained.append, budget_check=lambda: None)
            self.assertEqual(len(calls), 36)
            self.assertEqual(len(retained), 36)
            self.assertEqual(len(result.outer), 6)
            self.assertEqual(tuple(len(value) for value in result.inner), (5,) * 6)
            heldout_once = []
            for outer_index, error in enumerate(result.outer):
                self.assertEqual(error.split.heldout_blocks, (outer_index, outer_index + 6))
                self.assertEqual(len(error.training_sequence_ids), 60)
                self.assertEqual(len(error.heldout_sequence_ids), 12)
                heldout_once.extend(error.heldout_sequence_ids)
                self.assertEqual(set(error.whitened_mse), set(names))
                self.assertTrue(all(value.shape == (12,) and np.isfinite(value).all()
                                    for value in error.whitened_mse.values()))
                self.assertEqual({item.split.inner_index for item in result.inner[outer_index]},
                                 set(range(6)) - {outer_index})
                for inner in result.inner[outer_index]:
                    self.assertEqual(len(inner.training_sequence_ids), 48)
                    self.assertFalse(set(inner.training_sequence_ids) & set(inner.heldout_sequence_ids))
                    self.assertFalse(set(inner.training_sequence_ids) & set(error.heldout_sequence_ids))
                    self.assertFalse(set(inner.heldout_sequence_ids) & set(error.heldout_sequence_ids))
            self.assertEqual(sorted(heldout_once), sorted(item.sequence_id for item in sequences))
            self.assertTrue(all(set(fit.fits) == set(names) for fit in retained))

    def test_direct_targets_train_only_whitening_shared_masks_and_independent_mse(self):
        sequences = fixture()
        split = runner._split(0)
        sensor = runner.CommonSensorSpace(CHANNELS)
        fits, observed_targets, whitening = [], [], []

        def reference(design, targets, weights, penalties, limits, check):
            observed_targets.append(targets.copy())
            for rows in design.row_slices:
                self.assertAlmostEqual(weights[rows].sum(), .75)
            return dense_fit(design, targets, weights, penalties, limits, check)

        prepare = preparation_factory(runner.BASELINE_MODELS, whiten_calls=whitening)
        with patch.object(runner, "_fit", side_effect=reference):
            error = runner._run_fold(sequences, split, (None,) * 5, sensor,
                runner.BASELINE_MODELS, prepare, LIMITS, fits.append, lambda: None)
        train = [item for item in sequences if item.block not in (0, 6)]
        original_targets = np.concatenate([sensor.apply(item.sensor_samples, channel_order=CHANNELS)
                                           for item in train])
        for supplied in observed_targets:
            np.testing.assert_array_equal(supplied, original_targets)
        fitted = fits[0]
        heldout = [item for item in sequences if item.block in (0, 6)]
        for index, item in enumerate(heldout):
            view = runner.SequencePredictors(item.sequence_id, item.block, item.within_block,
                                             item.sample_times_seconds, item.preparation_input)
            for name, design in fitted.preparation.build_designs(view).items():
                matrix = runner.JointSequenceDesign([design]) @ np.eye(design.shape[1])
                residual = sensor.apply(item.sensor_samples, channel_order=CHANNELS) - matrix @ fitted.fits[name].coefficients
                transformed = residual[item.retained_mask] @ fitted.whitener.inverse_root
                expected = np.sum(transformed ** 2) / (np.count_nonzero(item.retained_mask) * 61)
                self.assertAlmostEqual(error.whitened_mse[name][index], expected, places=12)
        changed = tuple(replace(item, sensor_samples=item.sensor_samples * 100 + np.arange(62))
                        if item.block in (0, 6) else item for item in sequences)
        again = []
        with patch.object(runner, "_fit", side_effect=dense_fit):
            changed_error = runner._run_fold(changed, split, (None,) * 5, sensor,
                runner.BASELINE_MODELS, prepare, LIMITS, again.append, lambda: None)
        for name in runner.BASELINE_MODELS:
            np.testing.assert_array_equal(fitted.fits[name].coefficients, again[0].fits[name].coefficients)
        np.testing.assert_array_equal(fitted.whitener.inverse_root, again[0].whitener.inverse_root)
        np.testing.assert_array_equal(whitening[0][1], whitening[1][1])
        self.assertFalse(np.allclose(error.whitened_mse["image"], changed_error.whitened_mse["image"]))

    def test_invalid_roster_binding_or_design_refuses_before_any_fit(self):
        sequences = fixture()
        valid = preparation_factory(runner.BASELINE_MODELS)
        common = dict(participant_id="fixture", frozen_channel_order=CHANNELS, phase="baseline",
                      solver_limits=LIMITS, retain_fit=lambda value: None, budget_check=lambda: None)
        with patch.object(runner, "_fit") as fitter:
            with self.assertRaises(ValueError):
                runner.run_participant_folds(sequences=sequences[:-1], prepare_training=valid, **common)
            with self.assertRaises(ValueError):
                runner.run_participant_folds(sequences=sequences[:-1] + (sequences[0],),
                                             prepare_training=valid, **common)
            def wrong_binding(*args, **kwargs):
                training = args[0]
                item = training[0]
                original = next(value for value in sequences if value.sequence_id == item.sequence_id)
                before = original.sensor_samples.copy()
                self.assertTrue(np.shares_memory(item.sensor_samples, original.sensor_samples))
                self.assertFalse(item.sensor_samples.flags.writeable)
                with self.assertRaises(ValueError):
                    item.sensor_samples[0, 0] += 1.
                np.testing.assert_array_equal(original.sensor_samples, before)
                return replace(valid(*args, **kwargs), training_sequence_ids=("foreign",))
            with self.assertRaises(ValueError):
                runner.run_participant_folds(sequences=sequences, prepare_training=wrong_binding, **common)
            def wrong_design(*args, **kwargs):
                prepared = valid(*args, **kwargs)
                def build(view):
                    designs = dict(prepared.build_designs(view))
                    del designs["label_capacity_matched"]
                    return designs
                return replace(prepared, build_designs=build)
            with self.assertRaises(ValueError):
                runner.run_participant_folds(sequences=sequences, prepare_training=wrong_design, **common)
            fitter.assert_not_called()

    def test_solver_checks_each_contrast_and_failures_never_retry(self):
        sequence = fixture()[0]
        design = runner.JointSequenceDesign([SequenceDesign(sequence.sequence_id,
            sequence.sample_times_seconds, np.ones((len(sequence.sample_times_seconds), 1)), {})])
        targets = np.column_stack([np.arange(design.shape[0]) * (column + 1.) for column in range(61)])
        checks = []
        fitted = runner._fit(design, targets, np.ones(design.shape[0]), np.array([.1]),
                             LIMITS, lambda: checks.append(True))
        expected = targets.sum(axis=0) / (design.shape[0] + .1)
        np.testing.assert_allclose(fitted.coefficients[0], expected, rtol=1e-12, atol=1e-12)
        self.assertGreaterEqual(len(checks), 62)
        with patch.object(runner, "fit_joint_ridge", side_effect=RuntimeError("numerical failure")) as fitter:
            with self.assertRaisesRegex(RuntimeError, "numerical failure"):
                runner._fit(design, targets, np.ones(design.shape[0]), np.array([.1]), LIMITS, lambda: None)
            self.assertEqual(fitter.call_count, 1)
        with patch.object(runner, "_fit") as fitter:
            with self.assertRaises(TimeoutError):
                runner.run_participant_folds(participant_id="fixture", sequences=fixture(),
                    frozen_channel_order=CHANNELS, phase="baseline", solver_limits=LIMITS,
                    prepare_training=preparation_factory(runner.BASELINE_MODELS), retain_fit=lambda value: None,
                    budget_check=lambda: (_ for _ in ()).throw(TimeoutError("allocation deadline")))
            fitter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
