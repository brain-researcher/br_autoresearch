"""Bounded checks of EP17 nested prediction composition, never study payloads.

One test instruments the qualified numerical cores while retaining the complete
four-model, 84-setting, 2,423-image and six-fold production orchestration. It
tests data roles and composition, not empirical rank feasibility or fit quality.
The small cosine-basis check exercises only the new rank-derivation arithmetic.
Run this module in the coordinator's actual Slurm allocation, with one thread.
"""
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

import numpy as np

import nested_prediction as core
import prediction_core as prediction
import endpoint_evaluator as endpoint


MODELS = ("DINO_R50", "SWAV_R50_800", "SUP_R50_V1", "RANDOM_R50_S17")
WIDTHS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}


def fixture():
    concepts = tuple(f"concept_{concept:03}" for concept in range(480)
                     for _ in range(6 if concept < 23 else 5))
    concept_indices = np.array([int(value.rsplit("_", 1)[1]) for value in concepts])
    rows = core.DevelopmentDesign(
        tuple(f"image_{index:04}" for index in range(2423)), concepts,
        ("development",) * 2423, tuple((concept_indices // 80).tolist()),
        tuple((concept_indices % 24).tolist()))
    image_index = np.arange(2423, dtype=float)
    # Broadcast fixture storage is small. Production snapshots/identity checks
    # remain real; the test does not patch the immutable-array boundary.
    features = {model: core.FeaturePanel(rows, {
        layer: np.broadcast_to(image_index[:, None], (2423, width))
        for layer, width in WIDTHS.items()}) for model in MODELS}
    responses, columns = {}, {}
    for person_index, person in enumerate(endpoint.SUBJECTS):
        for roi_index, roi in enumerate(endpoint.ROIS):
            group = person, roi
            count = 1 + person_index + roi_index
            columns[group] = np.arange(count, dtype=int) * 11 + 7
            responses[group] = core.ResponsePanel(rows, columns[group], {
                stage: image_index[:, None] + 10 * person_index + 3 * roi_index
                       + stage_index + np.arange(count)[None, :] / 4.
                for stage_index, stage in enumerate(("B", "C", "D"))})
    return rows, features, responses, columns


class NestedPredictionTests(unittest.TestCase):
    def test_inner_objective_pools_before_dividing_and_weights_groups_equally(self):
        candidates = []
        for ratios in ((-4., .9, -4., -4., -4.), (.6,) * 5):
            partitions = {}
            for person_index, person in enumerate(endpoint.SUBJECTS):
                for roi_index, roi in enumerate(endpoint.ROIS):
                    width = 1 + person_index + roi_index
                    for stage_index, stage in enumerate(("B", "C", "D")):
                        delta = (.002 * person_index + .003 * roi_index
                                 + .004 * stage_index + .0005 * np.arange(width))
                        partitions[person, roi, stage] = tuple(
                            prediction.ErrorSums((1. - ratio - delta) * denominator,
                                                 np.full(width, denominator), images)
                            for ratio, denominator, images in zip(ratios, (1., 100., 1., 1., 1.),
                                                                  (11, 37, 13, 17, 19)))
            candidates.append(core.pooled_inner_objective(partitions))
        # Equal group/stage averaging gives an analytic .009 offset despite
        # unequal voxel counts; concatenating voxels yields another answer.
        self.assertAlmostEqual(candidates[0], 1. - 30. / 104. + .009, places=13)
        self.assertAlmostEqual(candidates[1], .609, places=13)
        self.assertGreater(candidates[0], candidates[1])
        self.assertLess(np.mean((-4., .9, -4., -4., -4.)), .6)
        with self.assertRaises(ValueError):
            core.pooled_inner_objective({key: value for key, value in partitions.items()
                                        if key != ("sub-06", "right", "D")})

    def test_rank_derivation_preserves_unwhitened_training_scale_and_cutoff_refusal(self):
        # Orthogonal, zero-mean cosine columns have descending known energies.
        # Their distinct amplitudes remove sign/cutoff ambiguity by construction.
        n = 160
        frequencies = np.arange(1, 131)
        features = np.cos(np.pi * (np.arange(n)[:, None] + .5) * frequencies / n)
        features *= np.linspace(4., 1., 130)
        maximum = prediction.fit_pca(features, 128)
        for rank in (32, 64, 128):
            derived = core.derive_rank_pca(maximum, features, rank)
            scale = np.sqrt(n / np.sum(features[:, :rank] ** 2))
            np.testing.assert_allclose(derived.transform(features), features[:, :rank] * scale,
                                       rtol=1e-10, atol=1e-10)
            held = features[:3] + .7
            np.testing.assert_allclose(derived.transform(held), held[:, :rank] * scale,
                                       rtol=1e-10, atol=1e-10)
            self.assertEqual(derived.training_images, n)
            self.assertEqual(derived.rank, rank)
        for rank in (32, 64):
            singular = maximum.singular_values.copy()
            singular[rank - 1] = singular[rank]
            with self.assertRaises(ValueError):
                core.derive_rank_pca(replace(maximum, singular_values=singular), features, rank)

    def test_complete_nested_roster_keeps_partitions_shared_and_freezes_final_predictors(self):
        rows, features, responses, columns = fixture()
        folds = tuple(range(6))
        train_indices = {part: np.flatnonzero(np.isin(rows.fold_ids, part))
                         for count in (4, 5, 6) for part in combinations(folds, count)}
        valid_indices = {fold: np.flatnonzero(np.asarray(rows.fold_ids) == fold) for fold in folds}
        grid = {core.Setting(layer, rank, penalty) for layer in WIDTHS
                for rank in (32, 64, 128) for penalty in (.0001, .001, .01, .1, 1., 10., 100.)}
        self.assertEqual(len(grid), 84)
        groups = tuple((person, roi) for person in endpoint.SUBJECTS for roi in endpoint.ROIS)
        offsets = {(group, stage): 10 * endpoint.SUBJECTS.index(group[0])
                   + 3 * endpoint.ROIS.index(group[1]) + ("B", "C", "D").index(stage)
                   for group in groups for stage in ("B", "C", "D")}
        current = {}
        pca_bindings, pca_seen, derivations = {}, set(), set()
        opportunities = defaultdict(set)
        means, ridge_cache, expected_target_rows, expected_baselines = {}, {}, {}, {}
        inner_seen = set()
        objective_seen = set()
        features_ready = False

        def resource(event):
            nonlocal features_ready
            current["event"] = event
            if event.operation == "all_features_feasible":
                self.assertEqual(pca_seen, {(model, part, layer) for model in MODELS
                                           for part in train_indices for layer in WIDTHS})
                self.assertEqual(derivations, {(model, part, layer, rank) for model in MODELS
                                              for part in train_indices for layer in WIDTHS
                                              for rank in (32, 64, 128)})
                features_ready = True
            if event.operation in ("slice_responses", "qualify_response_variance", "inner_error_sums", "fit_ridge"):
                self.assertTrue(features_ready, "neural work preceded complete feature feasibility")
            if event.operation == "fit_ridge" and event.when == "before":
                setting = core.Setting(event.layer, event.rank, event.penalty)
                opportunities[event.model_id, event.training_folds].add((setting, event.group, event.stage))

        def fit_pca(values, rank):
            event = current["event"]
            self.assertEqual((event.operation, event.when, rank), ("fit_pca", "before", 128))
            np.testing.assert_array_equal(values[:, 0], train_indices[event.training_folds])
            self.assertEqual(values.shape[1], WIDTHS[event.layer])
            pca_seen.add((event.model_id, event.training_folds, event.layer))
            signature = (1000 * MODELS.index(event.model_id) + 100 * tuple(WIDTHS).index(event.layer)
                         + sum(1 << fold for fold in event.training_folds))
            pca_bindings[signature] = event.model_id, event.training_folds, event.layer
            mean = np.zeros(values.shape[1])
            mean[0] = signature
            # This marks a fit's identity only. The separate cosine test uses
            # actual PCA; these rank-one fixture features are not qualified data.
            return prediction.PCAFit(mean, np.zeros((values.shape[1], 128)), 1.,
                                     np.arange(values.shape[1], 0, -1, dtype=float),
                                     128, len(values))

        def derive(maximum, values, rank):
            event = current["event"]
            derivations.add((event.model_id, event.training_folds, event.layer, rank))
            self.assertEqual(pca_bindings[maximum.feature_mean[0]],
                             (event.model_id, event.training_folds, event.layer))
            self.assertEqual(maximum.training_images, len(values))
            return replace(maximum, loadings=maximum.loadings[:, :rank])

        def transform(fit, values):
            event = current["event"]
            self.assertEqual(pca_bindings[fit.feature_mean[0]],
                             (event.model_id, event.training_folds, event.layer))
            self.assertEqual(fit.rank, event.rank)
            image_ids = values[:, 0]
            if event.operation in ("transform_training", "transform_refit_training"):
                np.testing.assert_array_equal(image_ids, train_indices[event.training_folds])
            else:
                self.assertIn(event.operation, ("transform_validation", "transform_outer_heldout"))
                heldout = rows.fold_ids[int(image_ids[0])]
                self.assertNotIn(heldout, event.training_folds)
                np.testing.assert_array_equal(image_ids, valid_indices[heldout])
                if event.operation == "transform_outer_heldout":
                    self.assertEqual(len(event.training_folds), 5)
            return np.broadcast_to(image_ids[:, None], (len(values), fit.rank))

        def fit_ridge(scores, targets, penalty):
            event = current["event"]
            part, group, stage = event.training_folds, event.group, event.stage
            key = part, group, stage
            if key not in expected_target_rows:
                expected_target_rows[key] = responses[group].stages[stage][train_indices[part], 0]
            self.assertEqual(scores.shape, (len(train_indices[part]), event.rank))
            self.assertTrue(np.array_equal(scores[:, 0], train_indices[part]))
            self.assertTrue(np.array_equal(targets[:, 0], expected_target_rows[key]))
            self.assertEqual(targets.shape[1], len(columns[group]))
            self.assertEqual(penalty, event.penalty)
            if key not in means:
                means[key] = np.mean(targets, axis=0, dtype=np.float64)
            cache_key = key, event.rank, penalty
            if cache_key not in ridge_cache:
                ridge_cache[cache_key] = prediction.RidgeFit(np.zeros(event.rank), means[key],
                    np.zeros((event.rank, targets.shape[1])), penalty, len(targets))
            return ridge_cache[cache_key]

        def predict(fit, scores):
            return np.broadcast_to(fit.target_mean, (len(scores), len(fit.target_mean)))

        actual_error_sums = prediction.error_sums

        def errors(observed, predicted, baseline):
            event = current["event"]
            if event.operation != "inner_error_sums":
                if event.operation == "qualify_block_SST":
                    key = event.group, event.stage
                    if key not in expected_baselines:
                        original = responses[event.group].stages[event.stage]
                        expected_baselines[key] = np.empty_like(original)
                        for heldout in folds:
                            training = np.flatnonzero(np.asarray(rows.fold_ids) != heldout)
                            expected_baselines[key][valid_indices[heldout]] = np.mean(original[training], axis=0)
                    image_ids = (observed[:, 0] - offsets[key]).astype(int)
                    # These twenty-concept blocks cross outer folds: each
                    # image must retain its own fold's training-only baseline.
                    self.assertGreater(len({rows.fold_ids[index] for index in image_ids}), 1)
                    np.testing.assert_array_equal(baseline, expected_baselines[key][image_ids])
                return actual_error_sums(observed, predicted, baseline)
            group, stage, part = event.group, event.stage, event.training_folds
            first_image = int(observed[0, 0] - offsets[group, stage])
            heldout = rows.fold_ids[first_image]
            self.assertNotIn(heldout, part)
            self.assertTrue(np.array_equal(observed[:, 0],
                responses[group].stages[stage][valid_indices[heldout], 0]))
            self.assertTrue(np.array_equal(baseline, means[part, group, stage]))
            self.assertEqual(predicted.shape, observed.shape)
            binding = part, heldout, group, stage
            self.assertNotIn(binding, inner_seen)
            inner_seen.add(binding)
            return prediction.ErrorSums(np.ones(observed.shape[1]), np.ones(observed.shape[1]), len(observed))

        def objective(partitions):
            event = current["event"]
            selection_folds = event.training_folds
            expected = {(tuple(f for f in selection_folds if f != heldout), heldout, group, stage)
                        for heldout in selection_folds for group in groups for stage in ("B", "C", "D")}
            self.assertEqual(inner_seen, expected)
            inner_seen.clear()
            self.assertEqual(set(partitions), {(*group, stage) for group in groups for stage in ("B", "C", "D")})
            for sums in partitions.values():
                self.assertEqual(tuple(value.images for value in sums),
                                 tuple(len(valid_indices[fold]) for fold in selection_folds))
            setting = core.Setting(event.layer, event.rank, event.penalty)
            objective_seen.add((event.model_id, selection_folds, setting))
            # Distinct from exact ties: this difference is smaller than common
            # numerical tie tolerances and must still select the better setting.
            wins = (event.model_id == "SUP_R50_V1"
                    and setting == core.Setting("layer4", 128, .0001))
            return float(np.nextafter(1., 2.)) if wins else 1.

        # Direct replacements avoid Mock retaining hundreds of thousands of
        # call arguments. Only numerical work/objectives are instrumented;
        # production role checks, snapshots, variance, baselines and loops run.
        with patch.object(prediction, "fit_pca", new=fit_pca), \
             patch.object(core, "derive_rank_pca", new=derive), \
             patch.object(prediction.PCAFit, "transform", new=transform), \
             patch.object(prediction, "fit_ridge", new=fit_ridge), \
             patch.object(prediction.RidgeFit, "predict", new=predict), \
             patch.object(prediction, "error_sums", new=errors), \
             patch.object(core, "pooled_inner_objective", new=objective):
            result = core.run_nested_prediction(rows, features, responses, columns, resource_check=resource)

        self.assertTrue(features_ready)
        self.assertFalse(inner_seen)
        self.assertEqual(result.design, rows)
        self.assertEqual(set(result.predictions), set(MODELS))
        selection_parts = tuple(combinations(folds, 5)) + (folds,)
        self.assertEqual(objective_seen, {(model, part, setting) for model in MODELS
                                         for part in selection_parts for setting in grid})
        full_opportunity = {(setting, group, stage) for setting in grid
                            for group in groups for stage in ("B", "C", "D")}
        for model in MODELS:
            chosen = (core.Setting("layer4", 128, .0001) if model == "SUP_R50_V1"
                      else core.Setting("layer1", 32, 100.))
            self.assertEqual(set(result.outer_selections[model]), set(folds))
            for part in train_indices:
                expected = full_opportunity if len(part) < 6 else {
                    (chosen, group, stage) for group in groups for stage in ("B", "C", "D")}
                self.assertEqual(opportunities[model, part], expected)
            for heldout in folds:
                selection = result.outer_selections[model][heldout]
                self.assertEqual(selection.setting, chosen)
                self.assertEqual(set(selection.objectives), grid)
                cka = result.outer_cka_transforms[model][heldout]
                part = tuple(f for f in folds if f != heldout)
                self.assertEqual(cka.rank, 64)
                self.assertEqual(pca_bindings[cka.feature_mean[0]], (model, part, chosen.layer))
            final = result.final_predictors[model]
            self.assertEqual(final.setting, chosen)
            self.assertEqual(set(final.selection.objectives), grid)
            self.assertEqual(final.training_image_ids, rows.image_ids)
            self.assertEqual(final.training_concept_ids, rows.concept_ids)
            self.assertEqual(final.pca.training_images, 2423)
            self.assertEqual(final.pca.rank, chosen.rank)
            self.assertEqual(final.cka_pca.rank, 64)
            self.assertEqual(pca_bindings[final.pca.feature_mean[0]], (model, folds, chosen.layer))
            self.assertEqual(pca_bindings[final.cka_pca.feature_mean[0]], (model, folds, chosen.layer))
            self.assertEqual(set(result.predictions[model]), set(groups))
            for group in groups:
                self.assertEqual(set(result.predictions[model][group]), {"B", "C", "D"})
                for stage in ("B", "C", "D"):
                    self.assertEqual(final.ridges[group][stage].training_images, 2423)
                    np.testing.assert_array_equal(final.ridges[group][stage].target_mean,
                                                  np.mean(responses[group].stages[stage], axis=0))
                    expected = np.empty_like(responses[group].stages[stage])
                    for heldout in folds:
                        train = np.flatnonzero(np.asarray(rows.fold_ids) != heldout)
                        expected[valid_indices[heldout]] = np.mean(responses[group].stages[stage][train], axis=0)
                    np.testing.assert_array_equal(result.predictions[model][group][stage], expected)
                    np.testing.assert_array_equal(result.baselines[group][stage], expected)
                    with self.assertRaises(ValueError):
                        result.predictions[model][group][stage].setflags(write=True)
        self.assertEqual(set(result.baselines), set(groups))
        for group in groups:
            np.testing.assert_array_equal(result.columns[group], columns[group])
            with self.assertRaises(ValueError):
                result.columns[group].setflags(write=True)
        with self.assertRaises(TypeError):
            result.predictions[MODELS[0]][groups[0]]["extra_support_specific_D"] = np.empty(0)

    def test_role_identity_original_columns_and_complete_rosters_refuse_before_fitting(self):
        rows, features, responses, columns = fixture()
        with self.assertRaisesRegex(ValueError, "development_role_only"):
            replace(rows, roles=("audit",) + rows.roles[1:])
        with self.assertRaisesRegex(ValueError, "concept_crosses_fold_or_block"):
            replace(rows, fold_ids=(1,) + rows.fold_ids[1:])
        with self.assertRaisesRegex(ValueError, "concept_crosses_fold_or_block"):
            replace(rows, block_ids=(1,) + rows.block_ids[1:])
        reordered = replace(rows, image_ids=tuple(reversed(rows.image_ids)))
        bad_feature = {**features, MODELS[0]: replace(features[MODELS[0]], rows=reordered)}
        group = endpoint.GROUPS[0]
        bad_response = {**responses, group: replace(responses[group], rows=reordered)}
        bad_columns = {**columns, group: columns[group] + 1}
        cases = (({key: value for key, value in features.items() if key != MODELS[-1]}, responses,
                  columns, "all_four_registered_models_required"),
                 (features, {key: value for key, value in responses.items() if key != group},
                  columns, "all_eight_response_groups_required"),
                 (bad_feature, responses, columns, "feature_row_binding_changed"),
                 (features, bad_response, columns, "response_row_binding_changed"),
                 (features, responses, bad_columns, "original_column_binding_changed"))
        # These are metadata refusals, so avoid repeated 300 MB feature copies.
        # The full orchestration test above retains real production snapshots.
        with patch.object(core, "_frozen_matrix", new=lambda value, *args: np.asarray(value)), \
             patch.object(prediction, "fit_pca", new=lambda *args: self.fail("fit began after invalid metadata")):
            for feature_panel, response_panel, axis, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    core.run_nested_prediction(rows, feature_panel, response_panel, axis, resource_check=lambda event: None)

    def test_resource_and_rank_failures_abort_without_neural_work_or_partial_result(self):
        rows, features, responses, columns = fixture()
        marker = RuntimeError("caller resource boundary reached")
        seen = []

        def stop(event):
            seen.append(event.operation)
            if event.operation == "snapshot_features":
                raise marker

        with self.assertRaises(RuntimeError) as caught:
            core.run_nested_prediction(rows, features, responses, columns, resource_check=stop)
        self.assertIs(caught.exception, marker)
        self.assertEqual(seen, ["begin", "snapshot_features"])
        with self.assertRaisesRegex(ValueError, "external_resource_callback_required"):
            core.run_nested_prediction(rows, features, responses, columns, resource_check=None)

        def maximum(values, rank):
            self.assertEqual(rank, 128)
            return prediction.PCAFit(np.zeros(values.shape[1]), np.zeros((values.shape[1], 128)),
                                     1., np.arange(values.shape[1], 0, -1, dtype=float), 128, len(values))

        for failed_rank in (32, 64, 128):
            operations = []

            def refuse(maximum_fit, values, rank):
                if rank == failed_rank:
                    raise prediction.PredictionError("required_rank_infeasible")
                return replace(maximum_fit, loadings=maximum_fit.loadings[:, :rank])

            with self.subTest(failed_rank=failed_rank), \
                 patch.object(core, "_frozen_matrix", new=lambda value, *args: np.asarray(value)), \
                 patch.object(prediction, "fit_pca", new=maximum), \
                 patch.object(core, "derive_rank_pca", new=refuse), \
                 patch.object(prediction, "fit_ridge", new=lambda *args: self.fail("ridge began after rank failure")), \
                 self.assertRaisesRegex(ValueError, "required_rank_infeasible"):
                core.run_nested_prediction(rows, features, responses, columns,
                                           resource_check=lambda event: operations.append(event.operation))
            self.assertNotIn("all_features_feasible", operations)
            self.assertNotIn("slice_responses", operations)
            self.assertNotIn("qualify_response_variance", operations)
            self.assertNotIn("inner_objective", operations)
            self.assertNotIn("complete", operations)

    def test_slice_resource_refusal_precedes_copy(self):
        calls = []
        marker = RuntimeError("copy exceeds remaining resource budget")

        class ForbiddenCopy:
            shape = (2423, 2048)

            def __getitem__(self, key):
                calls.append("copy")
                raise AssertionError("row copy occurred before admission")

        def deny(event):
            calls.append(event)
            self.assertEqual((event.operation, event.when, event.shape),
                             ("slice_features", "before", (2, 2048)))
            self.assertEqual(event.model_id, MODELS[0])
            raise marker

        with self.assertRaises(RuntimeError) as caught:
            core._slice_rows(deny, "slice_features", ForbiddenCopy(), np.array([1, 7]),
                             {"model_id": MODELS[0]})
        self.assertIs(caught.exception, marker)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
