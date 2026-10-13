"""Focused secondary-composition checks; no source payloads or fitted models.

Small real kernels are compared with literal replicated-row references. The
full binding/dispatch fixture retains the registered roster and instruments
only PCA application and the expensive kernels, without a synthetic fitting
campaign. Run only in the coordinator's Slurm allocation with one thread.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import development_secondary as core
import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction
import support_core as support


MODELS = ("DINO_R50", "SWAV_R50_800", "SUP_R50_V1", "RANDOM_R50_S17")
PAIRS = {"P1": (MODELS[0], MODELS[2]), "P2": (MODELS[1], MODELS[2]),
         "P3": (MODELS[0], MODELS[1]), "F1": (MODELS[2], MODELS[3])}
COMPONENTS = ("B_Aall", "C_Aall", "D_Aall", "D_AN", "D_RN", "D_VNC")


def replicated_reference(features, responses, weights):
    """Row-Gram CKA after actual replication; no weighted-kernel helper."""
    x = np.repeat(features, weights, axis=0)
    y = np.repeat(responses, weights, axis=0)
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    first, second = x @ x.T, y @ y.T
    return float(np.sum(first * second) / np.sqrt(np.sum(first * first) * np.sum(second * second)))


def bank_positions():
    first = np.concatenate([np.arange(64 * bin_index, 64 * bin_index + 32) for bin_index in range(8)])
    second = first + 32
    separated = (first,) * 16 + (second,) * 16
    mixed = tuple(np.concatenate([
        np.r_[64 * bin_index + (np.arange(16) + shift) % 32,
              64 * bin_index + 32 + (np.arange(16) + shift) % 32]
        for bin_index in range(8)]) for shift in range(32))
    return separated, mixed


def fixture():
    concept = np.array([item for item in range(480) for _ in range(6 if item < 23 else 5)])
    folds, block_indices = concept // 80, concept // 20
    labels = (1, "1") + tuple(f"block-{index}" for index in range(2, 24))
    block_order = tuple(reversed(labels))
    design = nested.DevelopmentDesign(tuple(f"image-{index}" for index in range(2423)),
        tuple(f"concept-{item}" for item in concept), ("development",) * 2423,
        tuple(folds.tolist()), tuple(labels[index] for index in block_indices))
    raw_columns = np.arange(512, dtype=int) * 3 + 7
    bins = np.repeat(np.arange(1, 9), 64)
    separated, _ = bank_positions()
    panels, columns, baselines, supports, banks = {}, {}, {}, {}, {}
    predictions = {model: {} for model in MODELS}
    image = np.arange(2423, dtype=float)
    for group_index, group in enumerate(endpoint.GROUPS):
        columns[group] = raw_columns.copy()
        supports[group] = endpoint.FrozenSupport(columns[group], bins.copy(),
            np.full(512, 1. / 512.), (np.arange(512) % 64 < 32).astype(float) / 256.,
            columns[group][::4], .10 + .05 * (np.arange(128) % 3),
            {name: endpoint.InapplicableGain("undefined_calibration_gain") for name in endpoint.H_COMPONENTS})
        banks[group] = support.AnatomicalBank(group[0], group[1], tuple(columns[group].tolist()),
            tuple(support.BINS[index - 1] for index in bins), tuple(columns[group].tolist()),
            tuple(tuple(columns[group][positions].tolist()) for positions in separated), (16,) * 512)
        stages, baselines[group] = {}, {}
        for model in MODELS:
            predictions[model][group] = {}
        for stage_index, stage in enumerate(endpoint.STAGES):
            vector = group_index * 1000. + stage_index * 100. + image / 1000.
            stages[stage] = np.broadcast_to(vector[:, None], (2423, 512))
            baselines[group][stage] = stages[stage]  # Unused by secondary scores.
            for model_index, model in enumerate(MODELS):
                predicted = vector + 10. * (model_index + 1)
                predictions[model][group][stage] = np.broadcast_to(predicted[:, None], (2423, 512))
        panels[group] = nested.ResponsePanel(design, columns[group], stages)
    features, selections, transforms = {}, {}, {}
    for model_index, model in enumerate(MODELS):
        features[model] = nested.FeaturePanel(design, {
            layer: np.broadcast_to((model_index * 100000. + layer_index * 10000. + image)[:, None],
                                   (2423, nested.FEATURE_WIDTHS[layer]))
            for layer_index, layer in enumerate(nested.LAYERS)})
        selections[model], transforms[model] = {}, {}
        for fold in range(6):
            layer = nested.LAYERS[(model_index + fold) % 4]
            width = nested.FEATURE_WIDTHS[layer]
            loadings = np.zeros((width, 64))
            loadings[:64] = np.eye(64)
            selections[model][fold] = nested.Selection(nested.Setting(layer, 32, 1.), {})
            transforms[model][fold] = prediction.PCAFit(np.zeros(width), loadings,
                1. + 100. * model_index + 10. * fold, np.arange(128., 0., -1.), 128,
                int(np.count_nonzero(folds != fold)))
    result = nested.NestedPredictionResult(design, columns, predictions, baselines, selections, transforms, {})
    return result, features, panels, endpoint.EndpointFamily(supports), banks, block_order


class DevelopmentSecondaryTests(unittest.TestCase):
    def test_weighted_fisher_matches_replicated_rows_and_preserves_boundary_rules(self):
        observed = np.array([[1., 4.], [3., -1.], [2., 6.], [8., 2.], [5., -3.], [0., 1.]])
        predicted = np.array([[2., 0.], [1., 5.], [4., 2.], [7., -1.], [6., 3.], [-2., 4.]])
        weights = np.array([2, 0, 3, 1, 2, 1])
        repeated_x, repeated_y = np.repeat(observed, weights, axis=0), np.repeat(predicted, weights, axis=0)
        reference = np.array([np.arctanh(np.corrcoef(repeated_x[:, index], repeated_y[:, index])[0, 1])
                              for index in range(2)])
        np.testing.assert_allclose(core.weighted_fisher_correlation(observed, predicted, weights),
                                   reference, rtol=2e-13, atol=2e-13)
        boundary = core.weighted_fisher_correlation(observed, observed * np.array([2., -3.]), weights)
        np.testing.assert_allclose(boundary, np.arctanh(np.array([1. - 1e-7, -1. + 1e-7])), atol=2e-9)
        with self.assertRaisesRegex(ValueError, "variance_undefined"):
            core.weighted_fisher_correlation(observed, np.ones_like(predicted), weights)
        for first, second in ((observed + 1e8, predicted), (observed, predicted + 1e8)):
            # Positive finite variance is insufficient when it falls below
            # the frozen1e-12 relative mean-square safeguard on either axis.
            with self.assertRaisesRegex(ValueError, "variance_undefined"):
                core.weighted_fisher_correlation(first, second, weights)
        # Only retained rows determine degeneracy; duplicating a single image
        # cannot create correlation support from the omitted images.
        with self.assertRaisesRegex(ValueError, "variance_undefined"):
            core.weighted_fisher_correlation(observed, predicted, np.array([3, 0, 0, 0, 0, 0]))
        for invalid in (np.zeros(6), np.array([1, -1, 1, 1, 1, 1]), np.full(6, .5)):
            with self.subTest(weights=invalid), self.assertRaises(ValueError):
                core.weighted_fisher_correlation(observed, predicted, invalid)

    def test_weighted_cka_matches_replicated_centering_without_whitening(self):
        features = np.array([[1., 4.], [3., -1.], [2., 6.], [8., 2.], [5., -3.], [0., 1.]])
        responses = np.array([[2., 0., 8.], [1., 5., -2.], [4., 2., 1.], [7., -1., 3.],
                              [6., 3., 7.], [-2., 4., 4.]])
        weights = np.array([2, 0, 3, 1, 2, 1])
        expected = replicated_reference(features, responses, weights)
        self.assertAlmostEqual(core.weighted_linear_cka(features, responses, weights), expected, places=13)
        self.assertAlmostEqual(core.weighted_linear_cka(features + np.array([17., -21.]),
            responses + np.array([10., -9., 4.]), weights), expected, places=13)
        scaled = responses * np.array([1., 7., .2])
        self.assertAlmostEqual(core.weighted_linear_cka(features, scaled, weights),
                               replicated_reference(features, scaled, weights), places=13)
        a, b = np.array([-1., -1., 1., 1.]), np.array([-1., 1., -1., 1.])
        # An analytic scale contrast detects silent per-voxel whitening.
        unit = np.ones(4, dtype=int)
        self.assertAlmostEqual(core.weighted_linear_cka(a[:, None], np.column_stack((a, b)), unit),
                               1. / np.sqrt(2.), places=13)
        self.assertAlmostEqual(core.weighted_linear_cka(a[:, None], np.column_stack((a, 7. * b)), unit),
                               1. / np.sqrt(1. + 7. ** 4), places=13)
        with self.assertRaisesRegex(ValueError, "CKA_denominator_undefined"):
            core.weighted_linear_cka(np.ones_like(features), responses, weights)
        with self.assertRaisesRegex(ValueError, "CKA_denominator_undefined"):
            core.weighted_linear_cka(features, responses, np.array([3, 0, 0, 0, 0, 0]))

    def test_all_32_member_ckas_cannot_be_replaced_by_mean_inclusion_weights(self):
        a = np.array([-1., -1., 1., 1.])
        b = np.array([-1., 1., -1., 1.])
        responses = np.tile(np.column_stack((np.tile(a[:, None], (1, 32)),
                                             np.tile(b[:, None], (1, 32)))), (1, 8))
        separated, mixed = bank_positions()
        for banks in (separated, mixed):
            np.testing.assert_array_equal(np.bincount(np.concatenate(banks), minlength=512), np.full(512, 16))
        calls = []
        def record(event):
            if event.operation == "weighted_linear_cka" and event.when == "before":
                calls.append(event)
        first = core.mean_bank_cka(a[:, None], responses, np.ones(4, dtype=int), separated, resource_check=record)
        second = core.mean_bank_cka(a[:, None], responses, np.ones(4, dtype=int), mixed, resource_check=record)
        self.assertEqual(len(calls), 64)
        self.assertAlmostEqual(first, .5, places=13)
        self.assertAlmostEqual(second, 1. / np.sqrt(2.), places=13)
        self.assertGreater(abs(first - second), .2)
        with self.assertRaisesRegex(ValueError, "all_32"):
            core.mean_bank_cka(a[:, None], responses, np.ones(4, dtype=int), separated[:-1], resource_check=record)
        def refuse_kernel(event):
            if event.operation == "weighted_linear_cka" and event.when == "before":
                raise RuntimeError("CKA resource refused")
        with patch.object(core, "weighted_linear_cka", side_effect=AssertionError("kernel preceded admission")):
            with self.assertRaisesRegex(RuntimeError, "CKA resource refused"):
                core.mean_bank_cka(a[:, None], responses, np.ones(4, dtype=int), separated,
                                   resource_check=refuse_kernel)

    def test_complete_dispatch_retains_frozen_layers_supports_pairs_and_resampling_units(self):
        result, features, responses, family, banks, order = fixture()
        folds = np.asarray(result.design.fold_ids)
        block_lookup = {label: index for index, label in enumerate(order)}
        coordinates = np.array([block_lookup[label] for label in result.design.block_ids])
        transform_calls, kernel_calls = [], []
        current_weights = np.ones(2423)

        def transform(fit, values):
            model_index = int(fit.score_scale) // 100
            fold = (int(fit.score_scale) % 100) // 10
            layer_index = (model_index + fold) % 4
            rows = np.flatnonzero(folds == fold)
            self.assertEqual(values.shape, (len(rows), nested.FEATURE_WIDTHS[nested.LAYERS[layer_index]]))
            np.testing.assert_array_equal(values[:, 0], model_index * 100000. + layer_index * 10000. + rows)
            self.assertEqual(fit.rank, 64)
            self.assertEqual(fit.training_images, 2423 - len(rows))
            transform_calls.append((model_index, fold, layer_index))
            scores = np.zeros((len(rows), 64))
            scores[:, 0], scores[:, 1] = model_index * 10 + fold, rows
            return scores

        def fisher(observed, predicted, weights):
            np.testing.assert_array_equal(weights, current_weights)
            group = int(observed[0, 0] // 1000)
            stage = int((observed[0, 0] % 1000) // 100)
            model_number = int(round((predicted[0, 0] - observed[0, 0]) / 10.))
            kernel_calls.append(("fisher", model_number - 1, group, stage, None))
            return model_number * (.01 + .001 * group + .002 * stage + .00001 * np.arange(512))

        def cka(scores, delivered, weights):
            model_index, fold = divmod(int(scores[0, 0]), 10)
            rows = np.flatnonzero(folds == fold)
            np.testing.assert_array_equal(weights, current_weights[rows])
            np.testing.assert_array_equal(scores[:, 1], rows)
            group = int(delivered[0, 0] // 1000)
            stage = int((delivered[0, 0] % 1000) // 100)
            kernel_calls.append(("cka", model_index, group, stage, fold))
            return (model_index + 1) * (.04 + .004 * group + .007 * stage + .003 * fold
                                      + .00001 * delivered.shape[1])

        with patch.object(prediction.PCAFit, "transform", new=transform):
            composed = core.compose_development_secondary(result, features, responses, family, banks,
                block_order=order, resource_check=lambda event: None)
        self.assertEqual(set(transform_calls), {(model, fold, (model + fold) % 4)
                                                for model in range(4) for fold in range(6)})
        self.assertEqual(len(transform_calls), 24)
        self.assertEqual(tuple(block.block_id for block in composed.blocks), order)
        self.assertEqual(set(composed.statistics), {"heldout_correlation", "linear_CKA"})
        people, block_weights = np.array([1, 2, 1, 3]), np.ones(24, dtype=int)
        current_weights = block_weights[coordinates]
        fold_mean = np.sum(np.arange(6) * np.bincount(folds, weights=current_weights)) / current_weights.sum()
        self.assertGreater(abs(fold_mean - 2.5), .01)
        with patch.object(core, "weighted_fisher_correlation", new=fisher), \
             patch.object(core, "weighted_linear_cka", new=cka):
            for scale in ("heldout_correlation", "linear_CKA"):
                self.assertEqual(set(composed.statistics[scale]), set(PAIRS))
                for pair_id, pair in PAIRS.items():
                    statistic = composed.statistics[scale][pair_id]
                    self.assertEqual(statistic.model_order, pair)
                    self.assertEqual(statistic.component_ids, COMPONENTS)
                    self.assertEqual(statistic.score_scales, (scale,) * 6)
                    signed_number = MODELS.index(pair[0]) - MODELS.index(pair[1])
                    group_values = []
                    for group in range(8):
                        stages = (0, 1, 2, 2, 2, 2)
                        if scale == "heldout_correlation":
                            positions = (np.arange(512),) * 4 + (
                                np.flatnonzero(np.arange(512) % 64 < 32), np.arange(0, 512, 4))
                            row = [signed_number * (.01 + .001 * group + .002 * stage + .00001 * pos.mean())
                                   for stage, pos in zip(stages, positions)]
                        else:
                            row = [signed_number * (.04 + .004 * group + .007 * stage + .003 * fold_mean
                                                    + .00001 * width)
                                   for stage, width in zip(stages, (512, 512, 512, 256, 256, 128))]
                        group_values.append(row)
                    person_values = np.array(group_values).reshape(4, 2, 6).mean(axis=1)
                    expected = np.sum(person_values * people[:, None], axis=0) / people.sum()
                    np.testing.assert_allclose(statistic(people, block_weights), expected, rtol=2e-12, atol=2e-12)
                    if scale == "heldout_correlation" and pair_id == "P1":
                        np.testing.assert_allclose(statistic.participant_effects(block_weights), person_values,
                                                   rtol=2e-12, atol=2e-12)
            # One positive-mass block selects only fold0; only sub-02 is present.
            # Absent units must not be evaluated and then declared undefined.
            kernel_calls.clear()
            block_weights = np.zeros(24, dtype=int)
            block_weights[order.index(1)] = 3
            current_weights = block_weights[coordinates]
            for scale in ("heldout_correlation", "linear_CKA"):
                composed.statistics[scale]["P1"](np.array([0, 1, 0, 0]), block_weights)
            self.assertEqual({item[2] for item in kernel_calls}, {2, 3})
            self.assertEqual({item[4] for item in kernel_calls if item[0] == "cka"}, {0})
            self.assertEqual({item[1] for item in kernel_calls}, {0, 2})
            # A retained undefined statistic invalidates the comparison; it
            # cannot be silently removed as though its multiplicity were zero.
            with patch.object(core, "weighted_linear_cka", side_effect=ValueError("retained CKA undefined")):
                with self.assertRaisesRegex(ValueError, "retained CKA undefined"):
                    composed.statistics["linear_CKA"]["P1"](np.array([0, 1, 0, 0]), block_weights)
            # Input mutation after admission cannot alter retained snapshots.
            statistic = composed.statistics["heldout_correlation"]["P1"]
            people = np.array([1, 0, 0, 0])
            before = statistic(people, block_weights)
            value = responses[endpoint.GROUPS[0]].stages["B"]
            original = value[0, 0]
            value.setflags(write=True)
            try:
                value[0, 0] += 500.
                np.testing.assert_array_equal(statistic(people, block_weights), before)
            finally:
                value[0, 0] = original
                value.setflags(write=False)
        # Exercise only the new complete-family delegation, not the already
        # qualified10,000-draw uncertainty engine.
        delegated = []
        def inference(callback, component_ids, score_scales, phase):
            self.assertEqual(component_ids, COMPONENTS)
            self.assertEqual(score_scales, (callback.scale,) * 6)
            self.assertEqual(phase, "development")
            delegated.append((callback.scale, callback.pair_id))
            return "delegated"
        with patch.object(core.uncertainty, "crossed_bounds", new=inference):
            for scale, statistics in composed.statistics.items():
                for pair_id, statistic in statistics.items():
                    self.assertEqual(statistic.crossed_bounds(), "delegated")
        self.assertEqual(set(delegated), {(scale, pair) for scale in composed.statistics for pair in PAIRS})

    def test_bank_transform_and_resource_failures_refuse_before_scoring(self):
        result, features, responses, family, banks, order = fixture()
        group, model = endpoint.GROUPS[0], MODELS[0]
        altered_counts = {**banks, group: replace(banks[group], inclusion_counts=(15,) + (16,) * 511)}
        altered_bins = list(banks[group].spatial_bins)
        altered_bins[0] = "BIN02"
        wrong_bins = {**banks, group: replace(banks[group], spatial_bins=tuple(altered_bins))}
        missing_banks = {**banks, group: replace(banks[group], banks=banks[group].banks[:-1])}
        wrong_fit = replace(result.outer_cka_transforms[model][0], training_images=2423)
        transforms = {**result.outer_cka_transforms,
                      model: {**result.outer_cka_transforms[model], 0: wrong_fit}}
        cases = ((result, altered_counts, "inclusion_counts"), (result, wrong_bins, "spatial_binding"),
                 (result, missing_banks, "all_32"),
                 (replace(result, outer_cka_transforms=transforms), banks, "rank64_transform"))
        with patch.object(core, "weighted_fisher_correlation", side_effect=AssertionError("early score")), \
             patch.object(core, "weighted_linear_cka", side_effect=AssertionError("early score")), \
             patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("unexpected transform")):
            for candidate, bank, code in cases:
                with self.subTest(code=code), self.assertRaisesRegex(ValueError, code):
                    core.compose_development_secondary(candidate, features, responses, family, bank,
                        block_order=order, resource_check=lambda event: None)
        marker = RuntimeError("resource admission refused")
        with patch.object(endpoint, "frozen_array", side_effect=AssertionError("copy before admission")):
            with self.assertRaises(RuntimeError) as caught:
                core.compose_development_secondary(result, features, responses, family, banks,
                    block_order=order, resource_check=lambda event: (_ for _ in ()).throw(marker))
        self.assertIs(caught.exception, marker)


if __name__ == "__main__":
    unittest.main()
