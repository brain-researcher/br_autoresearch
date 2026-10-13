"""Focused prediction-to-endpoint checks, with fabricated arrays only.

The complete production image/concept/fold/block roster is retained. There is
no feature extraction, PCA, regression, bootstrap or study-payload access.
Run only in the coordinator's actual Slurm allocation, with one BLAS thread.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import development_endpoints as core
import endpoint_evaluator as endpoint
import nested_prediction as nested


MODELS = ("DINO_R50", "SWAV_R50_800", "SUP_R50_V1", "RANDOM_R50_S17")
PAIRS = {"P1": (MODELS[0], MODELS[2]), "P2": (MODELS[1], MODELS[2]),
         "P3": (MODELS[0], MODELS[1]), "F1": (MODELS[2], MODELS[3])}
STAGES = ("B", "C", "D")


def fixture():
    concepts = np.array([concept for concept in range(480)
                         for _ in range(6 if concept < 23 else 5)])
    exemplars = np.array([exemplar for concept in range(480)
                          for exemplar in range(6 if concept < 23 else 5)])
    fold_indices, block_indices = concepts // 80, concepts % 24
    # Integer 1 and string "1" are different identities. The frozen order is
    # explicitly supplied, and deliberately differs from row first occurrence.
    labels = (1, "1") + tuple(f"block-{index}" for index in range(2, 24))
    block_order = tuple(reversed(labels))
    design = nested.DevelopmentDesign(
        tuple(f"image-{index}" for index in range(2423)),
        tuple(f"concept-{index}" for index in concepts),
        ("development",) * 2423, tuple(fold_indices.tolist()),
        tuple(labels[index] for index in block_indices))
    image = np.arange(2423, dtype=float)
    row_value = 5. * fold_indices + .07 * (concepts % 17) + .13 * exemplars + .0001 * image
    columns, responses, baselines, supports = {}, {}, {}, {}
    predictions = {model: {} for model in MODELS}
    for group_index, group in enumerate(endpoint.GROUPS):
        per_bin = 32 if group_index % 2 == 0 else 40
        width = per_bin * 8
        voxel = np.arange(width)
        columns[group] = np.roll(voxel * 3 + 7, 17)
        bins = np.repeat(np.arange(1, 9), per_bin)
        within_bin = voxel % per_bin
        counts = np.full(width, 32) if per_bin == 32 else np.where(within_bin < 24, 26, 25)
        reliable = (within_bin < 32).astype(float) / 256.
        ceiling_positions = np.arange(0, width, 3)
        supports[group] = endpoint.FrozenSupport(
            columns[group], bins, counts / 8192., reliable,
            columns[group][ceiling_positions], .10 + .025 * (np.arange(len(ceiling_positions)) % 3),
            {key: endpoint.GainStrata(columns[group][:width // 2], columns[group][width // 2:])
             for key in ("H_BC", "H_CD")})
        stages, baselines[group] = {}, {}
        for model in MODELS:
            predictions[model][group] = {}
        for stage_index, stage in enumerate(STAGES):
            observed = ((1. + .2 * stage_index) * row_value[:, None]
                        * (1. + .3 * voxel[None, :] / width)
                        + group_index * .4 + voxel[None, :] / 19.)
            baseline = np.empty_like(observed)
            for fold in range(6):
                baseline[fold_indices == fold] = observed[fold_indices != fold].mean(axis=0)
            stages[stage], baselines[group][stage] = observed, baseline
            for model_index, model in enumerate(MODELS):
                # Deliberately poor predictions preserve negative R². Error
                # varies with block, stage, group and voxel, so weighting and
                # stage-bank substitutions have visible consequences.
                residual = ((8., 13., 17., 23.)[model_index]
                            * (1. + .7 * block_indices[:, None] / 23. + .3 * fold_indices[:, None] / 5.)
                            * (1. + .04 * stage_index + .02 * group_index + .5 * voxel[None, :] / width))
                predictions[model][group][stage] = observed + residual
        responses[group] = nested.ResponsePanel(design, columns[group], stages)
    result = nested.NestedPredictionResult(design, columns, predictions, baselines, {}, {}, {})
    return result, responses, supports, block_order


def direct_block_sums(result, responses, block_order):
    """Reference from raw rows; no prediction/endpoint helper is called."""
    fold_ids = np.asarray(result.design.fold_ids)
    block_rows = [np.array([index for index, label in enumerate(result.design.block_ids)
                           if label == block], dtype=int) for block in block_order]
    denominator, errors = {}, {model: {} for model in MODELS}
    for group in endpoint.GROUPS:
        denominator[group] = {}
        for model in MODELS:
            errors[model][group] = {}
        for stage in STAGES:
            observed = responses[group].stages[stage]
            # Reconstruct training means independently; do not use the supplied
            # baseline matrix or a within-block/overall response mean.
            means = {fold: observed[fold_ids != fold].mean(axis=0) for fold in range(6)}
            sst = []
            for rows in block_rows:
                selected_baseline = np.array([means[int(fold_ids[row])] for row in rows])
                sst.append(np.sum((observed[rows] - selected_baseline) ** 2, axis=0))
            denominator[group][stage] = np.array(sst)
            for model in MODELS:
                predicted = result.predictions[model][group][stage]
                errors[model][group][stage] = np.array([
                    np.sum((observed[rows] - predicted[rows]) ** 2, axis=0)
                    for rows in block_rows])
    return errors, denominator


def direct_endpoints(supports, errors, denominator, pair_id, people, blocks):
    """Explicit signed pair and endpoint formulae, independent of the bridge."""
    model_a, model_b = PAIRS[pair_id]
    rows = []
    for person in endpoint.SUBJECTS:
        rois = []
        for roi in endpoint.ROIS:
            group = person, roi
            advantages = {}
            for stage in STAGES:
                a = np.sum(errors[model_a][group][stage] * blocks[:, None], axis=0)
                b = np.sum(errors[model_b][group][stage] * blocks[:, None], axis=0)
                sst = np.sum(denominator[group][stage] * blocks[:, None], axis=0)
                advantages[stage] = (b - a) / sst
            b, c, d = (advantages[stage] for stage in STAGES)
            frozen = supports[group]
            lookup = {column: index for index, column in enumerate(frozen.A_all)}
            ceiling = np.array([lookup[column] for column in frozen.V_NC])
            values = [b.mean(), c.mean(), d.mean(), np.dot(frozen.A_N_weights, d),
                      np.dot(frozen.R_N_weights, d), d[ceiling].mean(),
                      (d[ceiling] / frozen.V_NC_ceiling_denominator).mean()]
            for key, change in (("H_BC", c - b), ("H_CD", d - c)):
                low = [lookup[column] for column in frozen.beta_strata[key].low_columns]
                high = [lookup[column] for column in frozen.beta_strata[key].high_columns]
                values.append(change[high].mean() - change[low].mean())
            rois.append(values)
        rows.append(np.mean(rois, axis=0))
    rows = np.array(rows)
    return np.sum(rows * (people / people.sum())[:, None], axis=0), rows


class DevelopmentEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.responses, cls.supports, cls.block_order = fixture()
        cls.family = endpoint.EndpointFamily(cls.supports)

    @classmethod
    def tearDownClass(cls):
        # Release the full roster before any separately selected test module.
        for name in ("result", "responses", "supports", "block_order", "family"):
            delattr(cls, name)

    def compose(self, result=None, responses=None, family=None, resource_check=None, block_order=None):
        return core.compose_development_endpoints(
            self.result if result is None else result,
            self.responses if responses is None else responses,
            self.family if family is None else family,
            block_order=self.block_order if block_order is None else block_order,
            resource_check=(lambda event: None) if resource_check is None else resource_check)

    def test_complete_bank_matches_raw_rows_and_fixed_signed_endpoint_formulae(self):
        composed = self.compose()
        errors, denominator = direct_block_sums(self.result, self.responses, self.block_order)
        self.assertEqual(tuple(block.block_id for block in composed.blocks), self.block_order)
        self.assertNotEqual(self.block_order, self.result.design.block_labels)
        self.assertEqual(set(composed.statistics), set(PAIRS))
        self.assertEqual(set(composed.model_sse), set(MODELS))
        self.assertEqual(sum(len(block.image_ids) for block in composed.blocks), 2423)
        self.assertGreater(len({len(block.image_ids) for block in composed.blocks}), 1)
        for block in composed.blocks:
            indices = [index for index, label in enumerate(self.result.design.block_ids)
                       if label == block.block_id]
            self.assertEqual(block.image_ids, tuple(self.result.design.image_ids[index] for index in indices))
            self.assertEqual(block.concept_ids, tuple(self.result.design.concept_ids[index] for index in indices))
            self.assertEqual(len(set(block.concept_ids)), 20)
            self.assertEqual(len({self.result.design.fold_ids[index] for index in indices}), 6)
        for group in endpoint.GROUPS:
            for stage in STAGES:
                np.testing.assert_allclose(composed.common_sst[group][stage], denominator[group][stage], rtol=2e-13)
                for model in MODELS:
                    np.testing.assert_allclose(composed.model_sse[model][group][stage], errors[model][group][stage], rtol=2e-13)
        people = np.array([2, 0, 1, 3])
        blocks = np.array([1, 3, 0, 2] * 6)
        for pair_id, model_order in PAIRS.items():
            statistic = composed.statistics[pair_id]
            self.assertEqual(statistic.model_order, model_order)
            self.assertEqual(statistic.phase, "development")
            self.assertEqual(statistic.component_ids, endpoint.PRIMARY + endpoint.H_COMPONENTS)
            expected, participant_rows = direct_endpoints(self.supports, errors, denominator, pair_id, people, blocks)
            np.testing.assert_allclose(statistic(people, blocks), expected, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(statistic.participant_effects(blocks), participant_rows, rtol=2e-12, atol=2e-12)
            self.assertGreater(expected[6], 1.)
            for group in endpoint.GROUPS:
                paired = composed.paired_errors[pair_id][group]
                self.assertEqual(paired.model_order, model_order)
                np.testing.assert_array_equal(paired.column_ids, self.result.columns[group])
                for stage in STAGES:
                    np.testing.assert_allclose(paired.sse_a[stage], errors[model_order[0]][group][stage], rtol=2e-13)
                    np.testing.assert_allclose(paired.sse_b[stage], errors[model_order[1]][group][stage], rtol=2e-13)
                    np.testing.assert_allclose(paired.sst[stage], denominator[group][stage], rtol=2e-13)
        # These explicit counterexamples prove the fixture detects the tempting
        # alternative estimands, rather than only confirming dimensions.
        group, stage = endpoint.GROUPS[0], "D"
        observed = self.responses[group].stages[stage]
        first_rows = np.array([index for index, label in enumerate(self.result.design.block_ids)
                               if label == self.block_order[0]])
        for wrong_mean in (observed.mean(axis=0), observed[first_rows].mean(axis=0)):
            wrong = np.sum((observed[first_rows] - wrong_mean) ** 2, axis=0)
            self.assertGreater(np.max(np.abs(wrong - denominator[group][stage][0])), 1.)
        a, b = (errors[model][group][stage] for model in PAIRS["P1"])
        self.assertTrue((1. - b.sum(axis=0) / denominator[group][stage].sum(axis=0) < 0).all())
        right = np.sum((b - a) * blocks[:, None], axis=0) / np.sum(denominator[group][stage] * blocks[:, None], axis=0)
        wrong = np.average((b - a) / denominator[group][stage], axis=0, weights=blocks)
        self.assertGreater(np.max(np.abs(right - wrong)), 1e-4)
        for value in (composed.common_sst[group][stage], composed.model_sse[MODELS[0]][group][stage]):
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_alignment_and_complete_roster_failures_precede_pair_binding(self):
        group = endpoint.GROUPS[0]
        response_columns = dict(self.responses)
        response_columns[group] = replace(self.responses[group], column_ids=self.responses[group].column_ids[::-1])
        result_columns = dict(self.result.columns)
        result_columns[group] = result_columns[group][::-1]
        predictions = dict(self.result.predictions)
        predictions.pop(MODELS[-1])
        missing_group = dict(self.result.predictions[MODELS[0]])
        missing_group.pop(group)
        missing_stage = dict(self.result.predictions[MODELS[0]][group])
        missing_stage.pop("C")
        altered_predictions = {**self.result.predictions,
                               MODELS[0]: {**self.result.predictions[MODELS[0]], group: missing_stage}}
        short_predictions = {**self.result.predictions,
                             MODELS[0]: {**self.result.predictions[MODELS[0]],
                                         group: {**self.result.predictions[MODELS[0]][group],
                                                 "B": self.result.predictions[MODELS[0]][group]["B"][:-1]}}}
        cases = (
            ({"responses": response_columns}, "column"),
            ({"result": replace(self.result, columns=result_columns)}, "column"),
            ({"result": replace(self.result, predictions=predictions)}, "model"),
            ({"result": replace(self.result, predictions={**self.result.predictions, MODELS[0]: missing_group})}, "group"),
            ({"result": replace(self.result, predictions=altered_predictions)}, "B_C_D"),
            ({"result": replace(self.result, predictions=short_predictions)}, "shape"),
            ({"responses": {**self.responses, group: replace(self.responses[group], rows=replace(
                self.result.design, image_ids=tuple(reversed(self.result.design.image_ids))))}}, "row_binding"),
            ({"block_order": self.block_order[:-1]}, "block"),
            ({"block_order": self.block_order[:-1] + (self.block_order[0],)}, "block"),
            ({"block_order": tuple(True if label == 1 and isinstance(label, int) else label
                                    for label in self.block_order)}, "block"),
        )
        with patch.object(endpoint.EndpointFamily, "bind", side_effect=AssertionError("partial pair binding")):
            for arguments, code in cases:
                with self.subTest(code=code), self.assertRaisesRegex(ValueError, code):
                    self.compose(**arguments)
        # Directly cover the changed endpoint label boundary too; the bridge's
        # earlier label check must not be the only guard against bool/int aliasing.
        blocks = tuple(endpoint.ConceptBlock(label,
            tuple(f"image-{block}-{concept}" for concept in range(20)),
            tuple(f"concept-{block}-{concept}" for concept in range(20)))
            for block, label in enumerate(self.block_order))
        self.assertEqual(tuple(block.block_id for block in endpoint.bind_blocks(blocks, "development")), self.block_order)
        with self.assertRaisesRegex(endpoint.EndpointError, "block_identity_invalid"):
            endpoint.bind_blocks(tuple(replace(block, block_id=True) if block.block_id == 1
                                       and isinstance(block.block_id, int) else block for block in blocks), "development")

    def test_altered_baseline_or_nonfinite_prediction_refuses_without_partial_statistics(self):
        group, stage = endpoint.GROUPS[0], "B"
        wrong_baseline = self.result.baselines[group][stage].copy()
        wrong_baseline[0, 0] += .125
        altered = replace(self.result, baselines={**self.result.baselines,
                          group: {**self.result.baselines[group], stage: wrong_baseline}})
        nonfinite = self.result.predictions[MODELS[-1]][group][stage].copy()
        nonfinite[5, 7] = np.nan
        predictions = {**self.result.predictions,
                       MODELS[-1]: {**self.result.predictions[MODELS[-1]],
                                    group: {**self.result.predictions[MODELS[-1]][group], stage: nonfinite}}}
        with patch.object(endpoint.EndpointFamily, "bind", side_effect=AssertionError("partial pair binding")):
            with self.assertRaisesRegex(ValueError, "outer_training_baseline_changed"):
                self.compose(result=altered)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.compose(result=replace(self.result, predictions=predictions))

    def test_resource_refusal_happens_before_array_copy_or_partial_pair_binding(self):
        class BudgetStop(RuntimeError):
            pass

        events = []

        def refuse(event):
            events.append(event)
            raise BudgetStop("resource refused")

        with patch.object(endpoint, "frozen_array", side_effect=AssertionError("copy before admission")):
            with self.assertRaises(BudgetStop):
                self.compose(resource_check=refuse)
        self.assertEqual(len(events), 1)
        events.clear()

        def stop_before_prediction_copy(event):
            if event.operation == "snapshot_prediction" and event.when == "before":
                raise BudgetStop("prediction copy refused")

        with patch.object(core, "_snapshot", wraps=core._snapshot) as snapshot:
            with self.assertRaises(BudgetStop):
                self.compose(resource_check=stop_before_prediction_copy)
            self.assertEqual([call.args[2] for call in snapshot.call_args_list], ["response", "baseline"])

        def stop_at_sse(event):
            events.append(event)
            if event.operation == "block_sse" and event.when == "before":
                raise BudgetStop("resource refused during block reduction")

        with patch.object(endpoint.EndpointFamily, "bind", side_effect=AssertionError("partial pair binding")):
            with self.assertRaises(BudgetStop):
                self.compose(resource_check=stop_at_sse)
        self.assertEqual(events[-1].operation, "block_sse")
        self.assertEqual(events[-1].when, "before")
        self.assertNotIn("complete", [event.operation for event in events])


if __name__ == "__main__":
    unittest.main()
