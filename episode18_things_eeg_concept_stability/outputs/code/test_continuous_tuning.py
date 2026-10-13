"""Focused orchestration regressions; coordinator-owned Slurm compute only."""
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
from continuous_sequence_model import EventTerm, JointRidgeFit, SequenceDesign
from slot_capacity import RIDGE_GRID, SlotCapacityCurve, SlotCapacityResult
from training_transforms import CommonSensorSpace


CHANNELS = tuple(f"channel_{index}" for index in range(62))
LIMITS = folds.SolverLimits(1e-10, 1e-10, 1e12, 100)


def basis(lags):
    return np.ones((len(lags), 1))


def fixture():
    rows = []
    for index in range(72):
        times = np.arange(4) * .004
        channels = np.arange(62)[None, :]
        values = np.sin(.13 * index + .17 * channels + np.arange(4)[:, None] * .2)
        rows.append(folds.PhysicalSequence(f"sequence_{index}", index // 6, index % 6,
            times, values, CHANNELS, np.array([True, True, True, False]), index))
    return tuple(rows)


def preparation(calls):
    def prepare(training, *, split, budget_check):
        calls.append(split)
        if {item.block for item in training} != set(split.training_blocks):
            raise AssertionError("Preparation crossed the fixed training split")

        def build(view):
            if hasattr(view, "sensor_samples"):
                raise AssertionError("Design construction received held-out EEG")
            index = view.preparation_input
            nuisance = np.ones((len(view.sample_times_seconds), 1))
            base = EventTerm(np.array([0.]), np.array([[1 + index % 7]]), basis, 1, (0., .8))
            label = EventTerm(np.array([0.]), np.array([[1 + index % 5]]), basis, 1, (0., .8))
            slot = EventTerm(np.array([0.]), np.array([[1 + index % 11]]), basis, 1, (0., .8))
            result = {}
            for name in folds.FIXED_PANEL_MODELS:
                if name in ("nuisance", "reduced_task_nuisance"):
                    terms = {}
                elif name == "image":
                    terms = {"image_base": base}
                elif name == "label_linear":
                    terms = {"image_base": base, "label_base": label}
                elif name == "image_plus_true_concept":
                    terms = {"image_base": base, "concept_slot": slot}
                else:
                    slot_name = "concept_slot" if name == "true_concept" else name + "_slot"
                    terms = {"image_base": base, "label_base": label, slot_name: slot}
                result[name] = SequenceDesign(view.sequence_id, view.sample_times_seconds, nuisance, terms)
            return result
        return tuning.TuningPreparation(tuple(item.sequence_id for item in training), build)
    return prepare


def recipe(covariance_calls, selections):
    def penalties(view):
        return tuning.BasePenaltyPlan((.1,), (.1,), (.1, .2, .3))

    def covariance(residuals, weights):
        covariance_calls.append((len(residuals), float(weights.sum())))
        return folds.WhiteningChoices(True, float(weights.sum()), np.eye(61), .5)

    def matcher(view):
        # Exact identical slots make the same index feasible; this is an
        # explicit fixture rule, not an empirical selection default.
        index = view.feasible.concept_index
        return tuning.MatchedIndices(index, index, (index,) * 8)

    def selector(view):
        selections.append(view)
        self_keys = set(vars(view))
        if any("false" in key or "true" in key for key in self_keys):
            raise AssertionError("Hidden model errors reached baseline selection")
        index = min(view.feasible_concept_indices, key=lambda i: (view.matched_label_mean_losses[i], i))
        image = min(range(13), key=lambda i: (view.image_mean_losses[i], i))
        return tuning.BaselineChoice(index, image)

    def image_concept(view):
        return view.nuisance_penalties + (RIDGE_GRID[view.selected_image_index],) + (RIDGE_GRID[view.selected_concept_index],)

    return tuning.TuningRecipe(penalties, covariance, matcher, selector, image_concept,
        "mean_sequences", 1., "ratio_of_equal_sequence_means", "smallest_index_after_loss",
        1e-12, 16 * 1024**2)


def dense_fit(design, targets, weights, penalties, limits, check):
    """Small independent solve; no 61 individual iterative solves in fixtures."""
    check()
    matrix = design @ np.eye(design.shape[1])
    gram = matrix.T @ (weights[:, None] * matrix) + np.diag(penalties)
    coefficients = np.linalg.solve(gram, matrix.T @ (weights[:, None] * targets))
    return JointRidgeFit(coefficients, (2,) * 61, (1,) * 61)


def artificial_capacity(values):
    curves = tuple(SlotCapacityCurve(name, 1, (), tuple(vector), tuple(vector), tuple(vector))
                   for name, vector in values.items())
    return SlotCapacityResult(RIDGE_GRID, 0, 0, (), 1e-12, 0., 0., curves)


class ContinuousTuningTests(unittest.TestCase):
    def test_all_original_splits_prepare_once_full_grid_and_baseline_secrecy(self):
        prepared, covariances, selections, retained = [], [], [], []
        with patch.object(folds, "_fit", side_effect=dense_fit) as fit:
            result = tuning.run_tuned_participant(participant_id="fixture", sequences=fixture(),
                frozen_channel_order=CHANNELS, phase="baseline", recipe=recipe(covariances, selections),
                prepare_training=preparation(prepared), solver_limits=LIMITS,
                retain_fit=retained.append, budget_check=lambda: None)
        self.assertEqual(len(prepared), 36)
        self.assertEqual(len(set(prepared)), 36)
        self.assertEqual(len(covariances), 36)
        self.assertEqual(len(selections), 6)
        self.assertEqual(len(retained), 6)
        self.assertEqual(fit.call_count, 30 * (1 + 2 + 2 * 13) + 6 * 5)
        self.assertTrue(all(abs(weight - 1.) < 1e-12 for _, weight in covariances))
        self.assertEqual(set(result.normalized_participant_losses), set(folds.BASELINE_MODELS))
        for outer_index, inner in enumerate(result.inner):
            self.assertEqual(len(inner), 5)
            self.assertEqual({row.split.inner_index for row in inner}, set(range(6)) - {outer_index})
            for row in inner:
                self.assertEqual(set(row.raw_sequence_losses), set(folds.BASELINE_MODELS))
                self.assertEqual(row.raw_sequence_losses["image"].shape, (13, 12))
                self.assertEqual(row.raw_sequence_losses["label_capacity_matched"].shape, (13, 12))
                self.assertFalse(set(row.training_sequence_ids) & set(row.heldout_sequence_ids))
                self.assertEqual(row.capacity.ridge_penalties, RIDGE_GRID)
                self.assertEqual(len(row.capacity.curves), 10)
        for row in result.outer:
            self.assertEqual(row.split.heldout_blocks, (row.split.outer_index, row.split.outer_index + 6))
            self.assertEqual(row.normalized_losses["nuisance"], 1.)

    def test_fixed_panel_grid_outer_refit_and_shared_exact_capacity(self):
        covariances, selections, retained = [], [], []
        selected_recipe = recipe(covariances, selections)
        sequences, sensor = fixture(), CommonSensorSpace(CHANNELS)
        with patch.object(folds, "_fit", side_effect=dense_fit):
            inner_cache = tuning._prepare(sequences, folds._split(0, 1), sensor,
                selected_recipe, preparation([]), LIMITS, lambda: None)
            inner = tuning._inner(inner_cache, selected_recipe, "fixed_panel", LIMITS, lambda: None)
            self.assertEqual(set(inner.raw_sequence_losses), set(folds.FIXED_PANEL_MODELS) - {"image_plus_true_concept"})
            for name in tuning.GRID_MODELS:
                self.assertEqual(inner.raw_sequence_losses[name].shape, (13, 12))
            outer_cache = tuning._prepare(sequences, folds._split(0), sensor,
                selected_recipe, preparation([]), LIMITS, lambda: None)
            choice = tuning.BaselineChoice(6, 6)
            outer = tuning._outer(outer_cache, choice, selected_recipe, "fixed_panel", sensor,
                                  LIMITS, retained.append, lambda: None)
        self.assertEqual(set(outer.errors.whitened_mse), set(folds.FIXED_PANEL_MODELS))
        self.assertEqual(set(retained[0].fits), set(folds.FIXED_PANEL_MODELS))
        self.assertEqual(outer.match, tuning.MatchedIndices(6, 6, (6,) * 8))
        self.assertEqual(len(covariances), 2)
        for name in tuning.MATCHED_MODELS:
            np.testing.assert_array_equal(retained[0].penalties[name][:3], (.1, .2, .3))
            self.assertAlmostEqual(outer.capacity.curve(name).term_edf[6], outer.capacity.curve("true_concept").term_edf[6])
        expected = np.mean(outer.errors.whitened_mse["true_concept"]) / np.mean(outer.errors.whitened_mse["nuisance"])
        self.assertAlmostEqual(outer.normalized_losses["true_concept"], expected)

    def test_joint_feasibility_preserves_all_tuples_and_allows_fold_local_matches(self):
        values = {name: (20.,) * 13 for name in tuning.MATCHED_MODELS}
        options = tuning.all_feasible_tuples(artificial_capacity(values))
        self.assertEqual(len(options), 13)
        self.assertEqual(sum(item.tuple_count for item in options), 13**10)
        # Each comparison can match somewhere while no single concept index
        # admits all nine comparisons; independent pair matches are insufficient.
        values = {name: (200.,) * 13 for name in tuning.MATCHED_MODELS}
        values["true_concept"] = (20., 40.) + (100.,) * 11
        values["label_capacity_matched"] = (20.,) * 13
        for name in tuning.FALSE_MODELS:
            values[name] = (40.,) * 13
        self.assertEqual(tuning.all_feasible_tuples(artificial_capacity(values)), ())
        selected_recipe = recipe([], [])
        rows = tuple(SimpleNamespace(matches={0: tuning.MatchedIndices(0, index, (index,) * 8)},
            normalized_losses={"label_capacity_matched": np.arange(13.) + 1,
                               "image": np.arange(13.) + 2,
                               "true_concept": np.full(13, -1e99)}) for index in range(5))
        chosen = tuning._selection(rows, 0, selected_recipe)
        self.assertEqual(chosen.concept_index, 0)
        self.assertEqual(chosen.image_index, 0)

    def test_zero_denominator_budget_and_outer_mismatch_refuse_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            tuning.normalize_losses([1., 2.], [0., 0.], normalization="ratio_of_equal_sequence_means")
        # Ratio of equal-sequence means, not an average of sequence ratios.
        self.assertEqual(tuning.normalize_losses([2., 4.], [1., 3.],
                         normalization="ratio_of_equal_sequence_means"), 1.5)
        selected_recipe = recipe([], [])
        sensor = CommonSensorSpace(CHANNELS)
        with patch.object(folds, "_fit", side_effect=dense_fit):
            cache = tuning._prepare(fixture(), folds._split(0), sensor, selected_recipe,
                                    preparation([]), LIMITS, lambda: None)
        def bad_prepare(training, **kwargs):
            original = preparation([])(training, **kwargs)
            def build(view):
                models = original.build_designs(view)
                old = models["image"]
                changed = EventTerm(np.array([0.]), np.array([[999.]]), basis, 1, (0., .8))
                models["image"] = SequenceDesign(view.sequence_id, view.sample_times_seconds,
                                                old.nuisance, {"image_base": changed})
                return models
            return tuning.TuningPreparation(original.training_sequence_ids, build)
        with patch.object(folds, "_fit") as fit, self.assertRaisesRegex(ValueError, "identical penalized base"):
            tuning._prepare(fixture(), folds._split(0), sensor, selected_recipe,
                            bad_prepare, LIMITS, lambda: None)
        fit.assert_not_called()
        with patch.object(tuning, "_expand") as expand, self.assertRaises(MemoryError):
            tuning._capacity(cache, replace(selected_recipe, dense_design_byte_limit=1), lambda: None)
        expand.assert_not_called()
        capacity = artificial_capacity({name: (0.,) * 13 for name in tuning.MATCHED_MODELS})
        retained = []
        with patch.object(tuning, "_capacity", return_value=capacity), self.assertRaisesRegex(ValueError, "outer-training"):
            tuning._outer(cache, tuning.BaselineChoice(6, 6), selected_recipe, "fixed_panel", sensor,
                          LIMITS, retained.append, lambda: None)
        self.assertEqual(retained, [])


if __name__ == "__main__":
    unittest.main()
