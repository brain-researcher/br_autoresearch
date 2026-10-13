"""Focused frozen audit-application checks on fabricated arrays only.

The complete606-image/120-concept/12-block audit roster and2423-image/480-
concept development bindings are retained. Stored PCA/ridge application is
real; no training, bootstrap, secondary scoring or study payload is used.
Run only in the coordinator's Slurm allocation with one numerical thread.
"""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import audit_adjudicator as adjudicator
import audit_prediction as core
import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction
import uncertainty_core as uncertainty


MODELS = ("DINO_R50", "SUP_R50_V1")
SECONDARY = (("heldout_correlation", "D_RN"), ("linear_CKA", "D_AN"))


def fixture():
    development_concepts = np.array([item for item in range(480)
                                     for _ in range(6 if item < 23 else 5)])
    development = nested.DevelopmentDesign(tuple(f"dev-image-{index}" for index in range(2423)),
        tuple(f"dev-concept-{index}" for index in development_concepts), ("development",) * 2423,
        tuple((development_concepts // 80).tolist()), tuple((development_concepts % 24).tolist()))
    concepts = np.array([item for item in range(120) for _ in range(6 if item < 6 else 5)])
    labels = (1, "1") + tuple(f"audit-block-{index}" for index in range(2, 12))
    order = tuple(reversed(labels))
    design = core.AuditDesign(tuple(f"audit-image-{index}" for index in range(606)),
        tuple(f"audit-concept-{index}" for index in concepts), ("sealed_audit",) * 606,
        tuple(labels[index] for index in concepts // 10))
    columns, supports, responses, means = {}, {}, {}, {}
    image = np.arange(606, dtype=float)
    u, v, w = 1. + image / 100., (image % 11.) / 7. - .5, (image % 7.) / 5.
    for group_index, group in enumerate(endpoint.GROUPS):
        per_bin = 32 if group_index % 2 == 0 else 40
        width = per_bin * 8
        voxel = np.arange(width)
        columns[group] = np.roll(voxel * 3 + 7, 13)
        bins = np.repeat(np.arange(1, 9), per_bin)
        within_bin = voxel % per_bin
        counts = np.full(width, 32) if per_bin == 32 else np.where(within_bin < 24, 26, 25)
        supports[group] = endpoint.FrozenSupport(columns[group], bins, counts / 8192.,
            (within_bin < 32).astype(float) / 256., columns[group][::2],
            .10 + .05 * (np.arange(width // 2) % 3),
            {key: endpoint.InapplicableGain("undefined_calibration_gain") for key in endpoint.H_COMPONENTS})
        stages, means[group] = {}, {}
        for stage_index, stage in enumerate(endpoint.STAGES):
            mean = 2. + .1 * group_index + .2 * stage_index + voxel / 100.
            means[group][stage] = mean
            stages[stage] = (mean + .7 * u[:, None] * (1. + .2 * voxel[None, :] / width)
                             + .4 * v[:, None] + .05 * stage_index * w[:, None])
        responses[group] = core.AuditResponsePanel(design, columns[group], stages)
    family = endpoint.EndpointFamily(supports)
    relation = adjudicator.FrozenAuditRelation("P1", "contract_robust_relation", family,
        "F_raw_normalized", family.component_ids, family.primary_endpoint_ids, family.primary_endpoint_ids,
        {key: 1 for key in family.primary_endpoint_ids},
        {key: uncertainty.MARGINS[scale] for key, scale in zip(family.component_ids, family.score_scales)},
        {key: adjudicator.WIDTHS[scale] for key, scale in zip(family.component_ids, family.score_scales)}, None, None)
    features, predictors, expected, expected_cka = {}, {}, {}, {}
    for model_index, model in enumerate(MODELS):
        layer, rank = nested.LAYERS[model_index], (32, 64)[model_index]
        width = nested.FEATURE_WIDTHS[layer]
        feature_mean = np.arange(width) / 10.
        values = np.tile(feature_mean, (606, 1))
        first, second = u + .3 * model_index, v - .2 * model_index
        values[:, 0] += first
        values[:, 1] += second
        values[:, 2] += w
        features[model] = core.AuditFeaturePanel(design, layer, values)
        encoding_scale, cka_scale = (1.25, 1.)[model_index], (.75, 1.)[model_index]
        loadings = np.zeros((width, 64))
        loadings[:64] = np.eye(64)
        singular = np.arange(128., 0., -1.)
        pca = prediction.PCAFit(feature_mean.copy(), loadings[:, :rank].copy(), encoding_scale,
                                singular.copy(), 128, 2423)
        cka = prediction.PCAFit(feature_mean.copy(), loadings.copy(), cka_scale, singular.copy(), 128, 2423)
        wanted_cka = np.zeros((606, 64))
        wanted_cka[:, 0], wanted_cka[:, 1], wanted_cka[:, 2] = first * cka_scale, second * cka_scale, w * cka_scale
        expected_cka[model] = wanted_cka
        setting = nested.Setting(layer, rank, 1.)
        ridges, expected[model] = {}, {}
        for group_index, group in enumerate(endpoint.GROUPS):
            width = len(columns[group])
            voxel = np.arange(width)
            ridges[group], expected[model][group] = {}, {}
            for stage_index, stage in enumerate(endpoint.STAGES):
                score_mean = np.zeros(rank)
                score_mean[:2] = (.2, -.1)
                coefficient = np.zeros((rank, width))
                coefficient[0] = (model_index + 1) * (.1 + .02 * stage_index + .003 * group_index + .0001 * voxel)
                coefficient[1] = -.03 * (model_index + 1) + .00005 * voxel
                ridges[group][stage] = prediction.RidgeFit(score_mean, means[group][stage].copy(), coefficient, 1., 2423)
                # Closed form from the two nonzero coefficient rows. It does
                # not call PCAFit.transform, RidgeFit.predict or error_sums.
                expected[model][group][stage] = (means[group][stage]
                    + (encoding_scale * first[:, None] - .2) * coefficient[0]
                    + (encoding_scale * second[:, None] + .1) * coefficient[1])
        final = nested.FrozenPredictor(setting, pca, cka, ridges, nested.Selection(setting, {}),
                                       development.image_ids, development.concept_ids)
        predictors[model] = core.FrozenAuditPredictor(final, {group: value.copy() for group, value in columns.items()})
    return design, development, predictors, features, responses, relation, order, expected, expected_cka


def apply(fixture_values, *, resource_check=lambda event: None, secondary_checks=SECONDARY, **changes):
    names = ("design", "development_design", "predictors", "features", "responses", "relation")
    arguments = dict(zip(names, fixture_values[:6]))
    arguments.update(changes)
    return core.apply_frozen_audit(**arguments, block_order=fixture_values[6],
                                   secondary_checks=secondary_checks, resource_check=resource_check)


class AuditPredictionTests(unittest.TestCase):
    def test_complete_application_matches_frozen_parameters_and_raw_block_reference_without_fitting(self):
        values = fixture()
        design, development, predictors, features, responses, relation, order, expected, expected_cka = values
        with ExitStack() as stack:
            for module, name in ((prediction, "fit_pca"), (prediction, "fit_ridge"),
                                 (np.linalg, "svd"), (np.linalg, "lstsq"),
                                 (prediction, "linear_cka"), (nested.calibration, "fisher_transform"),
                                 (adjudicator, "adjudicate_primary_audit")):
                stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError("unselected fitting, secondary scoring or adjudication")))
            applied = apply(values)
        self.assertEqual(applied.pair_id, "P1")
        self.assertEqual(applied.design, design)
        self.assertEqual(tuple(block.block_id for block in applied.blocks), order)
        self.assertEqual(applied.secondary_checks, SECONDARY)
        self.assertEqual(set(applied.predictions), set(MODELS))
        self.assertEqual(sum(len(block.image_ids) for block in applied.blocks), 606)
        self.assertGreater(len({len(block.image_ids) for block in applied.blocks}), 1)
        self.assertEqual(applied.statistic.phase, "audit")
        self.assertIs(applied.statistic.family, relation.endpoint_family)
        self.assertEqual(applied.statistic.component_ids, relation.endpoint_family.component_ids)
        rows_by_block = [np.array([index for index, label in enumerate(design.block_ids) if label == block])
                         for block in order]
        block_weights = np.array([1, 3, 0, 2] * 3)
        group_endpoint_values = []
        for model in MODELS:
            np.testing.assert_allclose(applied.cka_features[model], expected_cka[model], rtol=2e-13, atol=2e-13)
        for group in endpoint.GROUPS:
            np.testing.assert_array_equal(applied.columns[group], predictors[MODELS[0]].column_ids[group])
            differences = {}
            for stage in endpoint.STAGES:
                frozen_mean = predictors[MODELS[0]].predictor.ridges[group][stage].target_mean
                np.testing.assert_array_equal(applied.baselines[group][stage], frozen_mean)
                observed = responses[group].stages[stage]
                np.testing.assert_array_equal(applied.responses[group][stage], observed)
                denominator = np.array([np.sum((observed[rows] - frozen_mean) ** 2, axis=0)
                                        for rows in rows_by_block])
                errors = {}
                for model in MODELS:
                    np.testing.assert_allclose(applied.predictions[model][group][stage], expected[model][group][stage],
                                               rtol=2e-13, atol=2e-13)
                    errors[model] = np.array([np.sum((observed[rows] - expected[model][group][stage][rows]) ** 2, axis=0)
                                              for rows in rows_by_block])
                paired = applied.paired_errors[group]
                self.assertEqual(paired.model_order, MODELS)
                np.testing.assert_allclose(paired.sse_a[stage], errors[MODELS[0]], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(paired.sse_b[stage], errors[MODELS[1]], rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(paired.sst[stage], denominator, rtol=2e-13, atol=2e-13)
                differences[stage] = ((block_weights @ errors[MODELS[1]]) - (block_weights @ errors[MODELS[0]])) / (
                    block_weights @ denominator)
            fixed = relation.endpoint_family._supports[group]
            b, c, d = (differences[stage] for stage in endpoint.STAGES)
            selected = d[fixed.ceiling_positions]
            group_endpoint_values.append([b.mean(), c.mean(), d.mean(), fixed.anatomical_weights @ d,
                fixed.reliable_weights @ d, selected.mean(), (selected / fixed.ceiling_denominators).mean()])
        people = np.array([1, 3, 2, 1])
        person_values = np.array(group_endpoint_values).reshape(4, 2, 7).mean(axis=1)
        np.testing.assert_allclose(applied.statistic(people, block_weights), (people @ person_values) / people.sum(),
                                   rtol=2e-12, atol=2e-12)
        group, stage = endpoint.GROUPS[0], "B"
        observed = responses[group].stages[stage]
        wrong = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0)
        correct = applied.paired_errors[group].sst[stage].sum(axis=0)
        self.assertGreater(np.max(np.abs(wrong - correct)), 100.)
        # The returned numerical result stays fixed if the caller later edits
        # its original fitted arrays; immutable outputs cannot be reopened.
        saved = applied.predictions[MODELS[0]][group][stage].copy()
        predictors[MODELS[0]].predictor.ridges[group][stage].coefficients[0, 0] = 999.
        np.testing.assert_array_equal(applied.predictions[MODELS[0]][group][stage], saved)
        saved_response = applied.responses[group][stage].copy()
        responses[group].stages[stage][0, 0] += .125
        np.testing.assert_array_equal(applied.responses[group][stage], saved_response)
        for array in (applied.predictions[MODELS[0]][group][stage], applied.baselines[group][stage],
                      applied.cka_features[MODELS[0]], applied.responses[group][stage]):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_audit_role_and_development_overlap_refuse_before_application(self):
        values = fixture()
        design, development = values[:2]
        with self.assertRaises(ValueError):
            replace(design, roles=("development",) * 606)
        overlapping_images = replace(design, image_ids=(development.image_ids[0],) + design.image_ids[1:])
        overlapping_concepts = replace(design, concept_ids=tuple(
            development.concept_ids[0] if item == design.concept_ids[0] else item for item in design.concept_ids))
        with patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("application before identity check")):
            for altered in (overlapping_images, overlapping_concepts):
                with self.subTest(images=altered.image_ids[0]), self.assertRaisesRegex(ValueError, "identity_overlap"):
                    apply(values, design=altered)
        with self.assertRaises(ValueError):
            replace(design, image_ids=design.image_ids[:-1], concept_ids=design.concept_ids[:-1],
                    roles=design.roles[:-1], block_ids=design.block_ids[:-1])

    def test_predictor_columns_training_roster_shared_means_and_selected_transform_are_bound(self):
        values = fixture()
        predictors = values[2]
        model, group = MODELS[1], endpoint.GROUPS[0]
        wrapper, original = predictors[model], predictors[model].predictor
        changed_columns = {**wrapper.column_ids, group: wrapper.column_ids[group][::-1]}
        wrong_training = replace(original, training_image_ids=tuple(reversed(original.training_image_ids)))
        mean = original.ridges[group]["B"].target_mean.copy()
        mean[0] += .125
        mean_fit = replace(original.ridges[group]["B"], target_mean=mean)
        wrong_means = replace(original, ridges={**original.ridges,
            group: {**original.ridges[group], "B": mean_fit}})
        loading = original.cka_pca.loadings.copy()
        loading[0, 0] *= -1.
        wrong_basis = replace(original, cka_pca=replace(original.cka_pca, loadings=loading))
        wrong_penalty = replace(original, ridges={**original.ridges,
            group: {**original.ridges[group], "B": replace(original.ridges[group]["B"], penalty=10.)}})
        variants = (replace(wrapper, column_ids=changed_columns), replace(wrapper, predictor=wrong_training),
                    replace(wrapper, predictor=wrong_means), replace(wrapper, predictor=wrong_basis),
                    replace(wrapper, predictor=wrong_penalty))
        with patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("application before frozen binding")):
            for candidate in variants:
                with self.subTest(candidate=candidate.predictor.setting), self.assertRaises(ValueError):
                    apply(values, predictors={**predictors, model: candidate})
            wrong_layer = {**values[3], model: replace(values[3][model], layer=nested.LAYERS[0])}
            with self.assertRaises(ValueError):
                apply(values, features=wrong_layer)

    def test_secondary_declarations_are_prelocked_bounded_and_not_additional_scores(self):
        values = fixture()
        with patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("application before declaration binding")):
            for descriptors in (SECONDARY + (("linear_CKA", "B_Aall"),), (SECONDARY[0], SECONDARY[0]),
                                (("linear_CKA", "D_VNC_normalized"),), (("raw_R2", "D_RN"),)):
                with self.subTest(descriptors=descriptors), self.assertRaises(ValueError):
                    apply(values, secondary_checks=descriptors)
            with self.assertRaises(ValueError):
                apply(values, relation=replace(values[5], pair_id="F1"))

    def test_response_variance_failure_preserves_whole_audit_scope(self):
        values = fixture()
        group, stage = endpoint.GROUPS[0], "B"
        panel = values[4][group]
        with patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("application before response qualification")):
            for whole in (True, False):
                altered = panel.stages[stage].copy()
                if whole:
                    altered[:, 0] = altered[0, 0]
                else:
                    chosen = [index for index, label in enumerate(values[0].block_ids) if label == 1]
                    # Oct12: constant primary blocks are admitted only with
                    # positive SST around the frozen development mean.
                    altered[chosen, 0] = values[2][MODELS[0]].predictor.ridges[group][stage].target_mean[0]
                responses = {**values[4], group: replace(panel, stages={**panel.stages, stage: altered})}
                with self.subTest(whole=whole), self.assertRaisesRegex(ValueError, "variance" if whole else "SST"):
                    apply(values, responses=responses)

    def test_resource_refusal_precedes_copies_and_frozen_transform_application(self):
        values = fixture()
        marker = RuntimeError("audit resource refused")
        def refuse_begin(event):
            raise marker
        with patch.object(endpoint, "frozen_array", side_effect=AssertionError("copy before admission")):
            with self.assertRaises(RuntimeError) as caught:
                apply(values, resource_check=refuse_begin)
        self.assertIs(caught.exception, marker)
        events = []
        def refuse_transform(event):
            events.append(event.operation)
            if event.operation == "apply_audit_pca" and event.when == "before":
                raise marker
        with patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("PCA before admission")):
            with self.assertRaises(RuntimeError):
                apply(values, resource_check=refuse_transform)
        self.assertNotIn("complete_audit_application", events)


if __name__ == "__main__":
    unittest.main()
