"""Focused audit-secondary composition checks on fabricated supplied arrays.

The fixture is an already-applied606-image/120-concept/12-block result, with
all eight groups and the selected two models. Expensive CKA dispatch is
instrumented; Fisher composition has a literal replicated-row reference.
No fitting, payload readers or bootstrap draws run. Execute only inside the
coordinator's Slurm allocation with one numerical thread.
"""
from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import audit_prediction as application
import audit_secondary as core
import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction
import support_core as support


MODELS = ("DINO_R50", "SUP_R50_V1")
COMPONENTS = ("B_Aall", "C_Aall", "D_Aall", "D_AN", "D_RN", "D_VNC")
CHECKS = (("heldout_correlation", "D_RN"), ("linear_CKA", "D_AN"))


def fixture():
    concepts = np.array([item for item in range(120) for _ in range(6 if item < 6 else 5)])
    labels = (1, "1") + tuple(f"audit-block-{index}" for index in range(2, 12))
    order = tuple(reversed(labels))
    design = application.AuditDesign(tuple(f"audit-image-{index}" for index in range(606)),
        tuple(f"audit-concept-{index}" for index in concepts), ("sealed_audit",) * 606,
        tuple(labels[index] for index in concepts // 10))
    rows = [np.array([i for i, label in enumerate(design.block_ids) if label == block]) for block in order]
    blocks = tuple(endpoint.ConceptBlock(block, tuple(design.image_ids[i] for i in selected),
        tuple(design.concept_ids[i] for i in selected)) for block, selected in zip(order, rows))
    positions = np.arange(512)
    columns = endpoint.frozen_array(positions * 3 + 7, integer=True)
    bins = np.repeat(np.arange(1, 9), 64)
    first = np.concatenate([np.arange(64 * index, 64 * index + 32) for index in range(8)])
    members = (first,) * 16 + (first + 32,) * 16
    image = np.arange(606, dtype=float)
    u, v, w = (image % 17.) / 4., ((image * 3.) % 13.) / 7., ((image * 7.) % 19.) / 6.
    voxel_class = positions % 5 + 1
    responses, baselines, supports, banks, errors, axes = {}, {}, {}, {}, {}, {}
    predictions = {model: {} for model in MODELS}
    for group_index, group in enumerate(endpoint.GROUPS):
        axes[group] = columns
        supports[group] = endpoint.FrozenSupport(columns, bins, np.full(512, 1. / 512.),
            (positions % 64 < 32).astype(float) / 256., columns[::4],
            .10 + .05 * (np.arange(128) % 3),
            {key: endpoint.InapplicableGain("undefined_calibration_gain") for key in endpoint.H_COMPONENTS})
        banks[group] = support.AnatomicalBank(group[0], group[1], tuple(columns.tolist()),
            tuple(support.BINS[index - 1] for index in bins), tuple(columns.tolist()),
            tuple(tuple(columns[selected].tolist()) for selected in members), (16,) * 512)
        responses[group], baselines[group] = {}, {}
        for model in MODELS:
            predictions[model][group] = {}
        sst, sse_a, sse_b = {}, {}, {}
        for stage_index, stage in enumerate(endpoint.STAGES):
            marker = group_index * 1000. + stage_index * 100.
            observed = (marker + (1. + .01 * group_index + .05 * stage_index) * u[:, None]
                        + .1 * v[:, None] * voxel_class + positions * 1e-5)
            responses[group][stage] = endpoint.frozen_array(observed)
            mean = marker - 1. + positions * 1e-5
            baselines[group][stage] = endpoint.frozen_array(mean)
            sst[stage] = endpoint.frozen_array([np.sum((observed[selected] - mean) ** 2, axis=0)
                                               for selected in rows])
            for model_number, model, target in ((1, MODELS[0], sse_a), (3, MODELS[1], sse_b)):
                predicted = (marker + 10. * model_number + positions * 1e-5
                    + (.5 + .1 * model_number + .02 * stage_index) * u[:, None]
                    + (.1 + .03 * model_number) * v[:, None] * voxel_class
                    + (.2 + .05 * model_number + .01 * group_index) * w[:, None])
                predictions[model][group][stage] = endpoint.frozen_array(predicted)
                target[stage] = endpoint.frozen_array([np.sum((observed[selected] - predicted[selected]) ** 2, axis=0)
                                                      for selected in rows])
        errors[group] = endpoint.PairedBlockErrors(MODELS, columns, blocks,
            nested._nested_map(sse_a), nested._nested_map(sse_b), nested._nested_map(sst))
    features = {}
    for model_number, model in ((1, MODELS[0]), (3, MODELS[1])):
        values = np.zeros((606, 64))
        values[:, 0], values[:, 1], values[:, 2] = model_number, image, u
        features[model] = endpoint.frozen_array(values)
    family = endpoint.EndpointFamily(supports)
    applied = application.AuditPredictionResult(design=design, blocks=blocks, pair_id="P1",
        columns=nested._nested_map(axes), predictions=nested._nested_map(predictions),
        baselines=nested._nested_map(baselines), cka_features=nested._nested_map(features),
        paired_errors=nested._nested_map(errors), statistic=family.bind("audit", errors, pair_id="P1"),
        secondary_checks=CHECKS, responses=nested._nested_map(responses))
    return applied, banks, supports, members


def image_multiplicities(applied, block_weights):
    # Literal metadata order; do not borrow the adapter's coordinate array.
    lookup = {block.block_id: index for index, block in enumerate(applied.blocks)}
    return np.array([block_weights[lookup[label]] for label in applied.design.block_ids], dtype=int)


class AuditSecondaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.applied, cls.banks, cls.supports, cls.members = fixture()

    @classmethod
    def tearDownClass(cls):
        del cls.applied, cls.banks, cls.supports, cls.members

    def test_complete_selected_families_use_single_final_bank_and_all_32_members(self):
        applied = self.applied
        block_weights = np.array([1, 3, 0, 2] * 3)
        image_weights = image_multiplicities(applied, block_weights)
        fisher_calls, cka_calls = [], []

        def fisher(observed, predicted, weights):
            np.testing.assert_array_equal(weights, image_weights)
            group = int(observed[0, 0] // 1000)
            stage = int(observed[0, 0] % 1000 // 100)
            number = int(round((predicted[0, 0] - observed[0, 0]) / 10.))
            self.assertIs(observed, applied.responses[endpoint.GROUPS[group]][endpoint.STAGES[stage]])
            self.assertIs(predicted, applied.predictions[MODELS[0 if number == 1 else 1]][
                endpoint.GROUPS[group]][endpoint.STAGES[stage]])
            fisher_calls.append((number, group, stage))
            return number * (.01 + .001 * group + .002 * stage + .00001 * np.arange(512))

        def cka(features, observed, weights):
            number = int(features[0, 0])
            self.assertIs(features, applied.cka_features[MODELS[0 if number == 1 else 1]])
            self.assertEqual(features.shape, (606, 64))
            np.testing.assert_array_equal(features[:, 1], np.arange(606))
            np.testing.assert_array_equal(weights, image_weights)
            group = int(observed[0, 0] // 1000)
            stage = int(observed[0, 0] % 1000 // 100)
            selected = np.rint((observed[0] - group * 1000. - stage * 100.) * 1e5).astype(int)
            cka_calls.append((number, group, stage, tuple(selected)))
            return number * (.04 + .004 * group + .007 * stage + 1e-7 * selected.mean() ** 2)

        with patch.object(prediction.PCAFit, "transform", side_effect=AssertionError("already transformed")), \
                patch.object(prediction, "fit_pca", side_effect=AssertionError("audit refit")), \
                patch.object(core.secondary, "weighted_fisher_correlation", new=fisher), \
                patch.object(core.secondary, "weighted_linear_cka", new=cka):
            result = core.compose_audit_secondary(applied, self.banks, resource_check=lambda event: None)
            self.assertEqual(result.secondary_checks, CHECKS)
            self.assertEqual(tuple(result.statistics), ("heldout_correlation", "linear_CKA"))
            self.assertEqual((len(fisher_calls), len(cka_calls)), (0, 0))
            for scale, statistic in result.statistics.items():
                self.assertEqual(statistic.pair_id, "P1")
                self.assertEqual(statistic.model_order, MODELS)
                self.assertEqual(statistic.phase, "audit")
                self.assertEqual(statistic.component_ids, COMPONENTS)
                self.assertEqual(statistic.score_scales, (scale,) * 6)
                self.assertEqual(statistic.selected_components, tuple(key for chosen, key in CHECKS if chosen == scale))
                group_values = []
                for group_index, group in enumerate(endpoint.GROUPS):
                    if scale == "heldout_correlation":
                        z = [-2 * (.01 + .001 * group_index + .002 * stage + .00001 * np.arange(512))
                             for stage in range(3)]
                        values = [entry.mean() for entry in z]
                        # Support means from explicit memberships, not the
                        # adapter's aggregate inclusion-weight calculation.
                        values += [np.mean([z[2][member].mean() for member in self.members]),
                                   z[2][self.members[0]].mean(), z[2][::4].mean()]
                    else:
                        score = lambda stage, member: -2 * (.04 + .004 * group_index + .007 * stage
                                                           + 1e-7 * np.mean(member) ** 2)
                        values = [score(stage, np.arange(512)) for stage in range(3)]
                        values += [np.mean([score(2, member) for member in self.members]),
                                   score(2, self.members[0]), score(2, np.arange(512)[::4])]
                    group_values.append(values)
                expected = np.asarray(group_values).reshape(4, 2, 6).mean(axis=1)
                np.testing.assert_allclose(statistic.participant_effects(block_weights), expected, atol=2e-14)
                calls = fisher_calls if scale == "heldout_correlation" else cka_calls
                if scale == "heldout_correlation":
                    self.assertEqual(Counter(calls), Counter((number, group, stage)
                        for number in (1, 3) for group in range(8) for stage in range(3)))
                else:
                    expected_calls = Counter()
                    for number in (1, 3):
                        for group in range(8):
                            for stage in range(3):
                                expected_calls[(number, group, stage, tuple(range(512)))] += 1
                            for selected in self.members + (self.members[0], np.arange(512)[::4]):
                                expected_calls[(number, group, 2, tuple(selected))] += 1
                    self.assertEqual(Counter(calls), expected_calls)
                    self.assertEqual(len(calls), 2 * 8 * 37)
                calls.clear()
                people = np.array([0, 3, 0, 1])
                np.testing.assert_allclose(statistic(people, block_weights), people @ expected / people.sum(), atol=2e-14)
                self.assertEqual({call[1] for call in calls}, {2, 3, 6, 7})
                with patch.object(core.uncertainty, "crossed_bounds", return_value="complete audit bounds") as bounds:
                    self.assertEqual(statistic.crossed_bounds(), "complete audit bounds")
                    bounds.assert_called_once_with(statistic, COMPONENTS, (scale,) * 6, "audit")
            bank = result.statistics["linear_CKA"]._bank
            for group in endpoint.GROUPS:
                for stage in endpoint.STAGES:
                    self.assertIs(bank.responses[group][stage], applied.responses[group][stage])
                    for model in MODELS:
                        self.assertIs(bank.predictions[model][group][stage], applied.predictions[model][group][stage])
            for model in MODELS:
                self.assertIs(bank.features[model], applied.cka_features[model])

    def test_actual_fisher_composition_matches_literal_audit_row_replication(self):
        applied = replace(self.applied, secondary_checks=(("heldout_correlation", "D_RN"),))
        result = core.compose_audit_secondary(applied, None, resource_check=lambda event: None)
        blocks = np.array([1, 3, 0, 2] * 3)
        repeated = np.repeat(np.arange(606), image_multiplicities(applied, blocks))
        group_scores = []
        for group in endpoint.GROUPS[2:4]:
            by_stage = []
            for stage in endpoint.STAGES:
                observed = applied.responses[group][stage]
                per_class = []
                for column in range(5):
                    zs = [np.arctanh(np.corrcoef(observed[repeated, column],
                        applied.predictions[model][group][stage][repeated, column])[0, 1]) for model in MODELS]
                    per_class.append(zs[0] - zs[1])
                # Remaining columns differ only by a constant, so these
                # five explicit reference correlations cover the whole axis.
                by_stage.append(np.asarray(per_class)[np.arange(512) % 5])
            d = by_stage[2]
            group_scores.append([entry.mean() for entry in by_stage] +
                [np.mean([d[member].mean() for member in self.members]), d[self.members[0]].mean(), d[::4].mean()])
        expected = np.mean(group_scores, axis=0)
        with patch.object(core.secondary, "weighted_linear_cka", side_effect=AssertionError("unlocked CKA")):
            actual = result.statistics["heldout_correlation"](np.array([0, 4, 0, 0]), blocks)
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
        self.assertEqual(result.statistics["heldout_correlation"].component_ids, COMPONENTS)

    def test_locked_scales_and_frozen_inapplicability_preserve_complete_families(self):
        empty = replace(self.applied, secondary_checks=(), responses={}, predictions={}, cka_features={})
        with patch.object(core, "_reuse_matrix", side_effect=AssertionError("no selected arrays")), \
                patch.object(core.secondary, "weighted_fisher_correlation", side_effect=AssertionError("unlocked score")), \
                patch.object(core.secondary, "weighted_linear_cka", side_effect=AssertionError("unlocked score")):
            result = core.compose_audit_secondary(empty, None, resource_check=lambda event: None)
            self.assertEqual(result.secondary_checks, ())
            self.assertEqual(dict(result.statistics), {})
        locks = (("heldout_correlation", "D_RN"), ("heldout_correlation", "B_Aall"))
        selected = replace(self.applied, secondary_checks=locks, cka_features={})
        result = core.compose_audit_secondary(selected, None, resource_check=lambda event: None)
        self.assertEqual(tuple(result.statistics), ("heldout_correlation",))
        self.assertEqual(result.statistics["heldout_correlation"].selected_components, ("D_RN", "B_Aall"))
        self.assertEqual(result.statistics["heldout_correlation"].component_ids, COMPONENTS)
        cka_only = replace(self.applied, secondary_checks=(("linear_CKA", "D_AN"),), predictions={})
        result = core.compose_audit_secondary(cka_only, self.banks, resource_check=lambda event: None)
        self.assertEqual(tuple(result.statistics), ("linear_CKA",))
        with self.assertRaisesRegex(ValueError, "complete_anatomical_bank_groups"):
            core.compose_audit_secondary(cka_only, None, resource_check=lambda event: None)
        absent_ceiling = {group: replace(value, V_NC=np.array([], dtype=int),
                                         V_NC_ceiling_denominator=np.array([], dtype=float))
                          for group, value in self.supports.items()}
        family = endpoint.EndpointFamily(absent_ceiling)
        selected = replace(selected, statistic=family.bind("audit", self.applied.paired_errors, pair_id="P1"))
        result = core.compose_audit_secondary(selected, None, resource_check=lambda event: None)
        self.assertEqual(result.statistics["heldout_correlation"].component_ids, COMPONENTS[:-1])
        invalid_locks = (CHECKS + (("linear_CKA", "B_Aall"),), (CHECKS[0], CHECKS[0]),
                         (("heldout_correlation", "D_VNC_normalized"),))
        for locks in invalid_locks:
            with self.subTest(locks=locks), self.assertRaisesRegex(ValueError, "at_most_two"):
                core.compose_audit_secondary(replace(self.applied, secondary_checks=locks), self.banks,
                                             resource_check=lambda event: None)
        with self.assertRaisesRegex(ValueError, "at_most_two"):
            core.compose_audit_secondary(replace(selected, secondary_checks=(("heldout_correlation", "D_VNC"),)),
                                         None, resource_check=lambda event: None)

    def test_primary_result_binding_and_irreversible_array_snapshots_are_required(self):
        applied, group = self.applied, endpoint.GROUPS[0]
        own_readonly = np.array(applied.responses[group]["B"], copy=True)
        own_readonly.setflags(write=False)
        response_changes = ({**applied.responses, group: {**applied.responses[group], "B": own_readonly}},
            {**applied.responses, group: {**applied.responses[group], "B": own_readonly.copy()}})
        bad_prediction = {**applied.predictions, MODELS[0]: {**applied.predictions[MODELS[0]], group: {
            **applied.predictions[MODELS[0]][group], "B": applied.predictions[MODELS[0]][group]["B"].copy()}}}
        bad_features = {**applied.cka_features, MODELS[0]: applied.cka_features[MODELS[0]].copy()}
        variants = [replace(applied, responses=value) for value in response_changes]
        variants += [replace(applied, predictions=bad_prediction), replace(applied, cka_features=bad_features)]
        with patch.object(core.secondary, "weighted_fisher_correlation", side_effect=AssertionError("premature score")), \
                patch.object(core.secondary, "weighted_linear_cka", side_effect=AssertionError("premature score")):
            for altered in variants:
                with self.assertRaisesRegex(ValueError, "existing_immutable_audit_matrix_required"):
                    core.compose_audit_secondary(altered, self.banks, resource_check=lambda event: None)
            for altered, reason in (
                (replace(applied, pair_id="P2"), "primary_audit_statistic_binding_changed"),
                (replace(applied, blocks=tuple(reversed(applied.blocks))), "primary_audit_block_order_changed"),
                (replace(applied, columns={**applied.columns, group: applied.columns[group][::-1]}),
                 "audit_original_column_binding_changed"),
                (replace(applied, design=replace(applied.design,
                    image_ids=("replacement-image",) + applied.design.image_ids[1:])), "audit_design_image_binding_changed"),
                (replace(applied, responses={key: value for key, value in applied.responses.items() if key != group}),
                 "complete_retained_audit_responses_required")):
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    core.compose_audit_secondary(altered, self.banks, resource_check=lambda event: None)
        for array in (applied.responses[group]["B"], applied.predictions[MODELS[0]][group]["B"],
                      applied.cka_features[MODELS[0]]):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_resource_admission_precedes_work_and_positive_mass_failures_are_not_dropped(self):
        def refuse_begin(event):
            if event.operation == "begin_audit_secondary":
                raise RuntimeError("binding resource refused")
        with patch.object(core, "_reuse_matrix", side_effect=AssertionError("validation before admission")):
            with self.assertRaisesRegex(RuntimeError, "binding resource refused"):
                core.compose_audit_secondary(self.applied, self.banks, resource_check=refuse_begin)
        denied = [None]
        def admission(event):
            if event.when == "before" and event.operation == denied[0]:
                raise RuntimeError("selected kernel resource refused")
        result = core.compose_audit_secondary(self.applied, self.banks, resource_check=admission)
        for scale, kernel in (("heldout_correlation", "weighted_fisher_correlation"),
                              ("linear_CKA", "weighted_linear_cka")):
            denied[0] = kernel
            with patch.object(core.secondary, kernel, side_effect=AssertionError("kernel before admission")):
                with self.assertRaisesRegex(RuntimeError, "selected kernel resource refused"):
                    result.statistics[scale](np.ones(4, dtype=int), np.ones(12, dtype=int))
        denied[0] = None
        seen = []
        def selected_undefined(observed, predicted, weights):
            group = int(observed[0, 0] // 1000)
            seen.append(group)
            if group >= 2:
                raise ValueError("positive-mass correlation undefined")
            return np.zeros(512)
        with patch.object(core.secondary, "weighted_fisher_correlation", side_effect=selected_undefined):
            statistic = result.statistics["heldout_correlation"]
            np.testing.assert_array_equal(statistic(np.array([1, 0, 0, 0]), np.ones(12, dtype=int)), np.zeros(6))
            self.assertEqual(set(seen), {0, 1})
            with self.assertRaisesRegex(ValueError, "positive-mass correlation undefined"):
                statistic(np.ones(4, dtype=int), np.ones(12, dtype=int))
            with self.assertRaisesRegex(ValueError, "positive-mass correlation undefined"):
                statistic.participant_effects(np.ones(12, dtype=int))
        with patch.object(core.secondary, "weighted_linear_cka", new=lambda *args: np.nan):
            with self.assertRaisesRegex(ValueError, "audit_secondary_statistic_nonfinite"):
                result.statistics["linear_CKA"](np.array([1, 0, 0, 0]), np.ones(12, dtype=int))


if __name__ == "__main__":
    unittest.main()
