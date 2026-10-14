"""Focused adopted primary-block exception checks on existing invented rosters.

No empirical files, PCA, ridge fitting, audit opening or bootstrap campaign.
The development response preparation and frozen audit application are real.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
import nested_prediction as nested
import audit_prediction as audit
import development_secondary as secondary
import test_nested_prediction as development_fixture
import test_audit_prediction as audit_fixture


class PrimaryBlockVarianceAmendmentTests(unittest.TestCase):
    def test_only_primary_block_variance_is_relaxed_and_moments_remain_finite(self):
        constant = np.full((9, 2), 3.)
        tiny_variation = constant.copy()
        tiny_variation[:, 0] += np.arange(9) * 1e-8
        for values in (constant, tiny_variation):
            nested._qualify_primary_block_moments(values)
            with self.assertRaisesRegex(nested.NestedPredictionError, "response_partition_variance_invalid"):
                nested._qualify_response_variance(values)
            nested._qualify_sst(values, np.zeros(2))
        for value in (np.nan, np.inf, 1e308):
            damaged = constant.copy()
            damaged[0, 0] = value
            with self.assertRaises(ValueError):
                nested._qualify_primary_block_moments(damaged)
        with self.assertRaisesRegex(nested.NestedPredictionError, "response_partition_SST_invalid"):
            nested._qualify_sst(constant, np.full(2, 3.))
        weights = np.ones(9, dtype=int)
        varied = np.arange(18, dtype=float).reshape(9, 2)
        with self.assertRaises(ValueError):
            secondary.weighted_fisher_correlation(constant, varied, weights)
        with self.assertRaises(ValueError):
            secondary.weighted_linear_cka(varied, constant, weights)

    def test_development_constant_block_passes_but_training_validation_and_block_SST_do_not_relax(self):
        design, _, panels, _ = development_fixture.fixture()
        training, validation, blocks = nested._row_indices(design)
        responses = {group: {stage: value.copy() for stage, value in panel.stages.items()}
                     for group, panel in panels.items()}
        group, stage = nested.GROUPS[0], "B"
        original = responses[group][stage].copy()
        block = next(iter(blocks.values()))
        responses[group][stage][block, 0] = -7.
        events = []
        means, baselines = nested._prepare_responses(design, responses, training, validation, blocks, events.append)
        self.assertEqual(sum(event.operation == "qualify_response_variance" and event.when == "after"
                             for event in events), 24 * (22 + 6))
        self.assertEqual(sum(event.operation == "qualify_primary_block_moments" and event.when == "after"
                             for event in events), 24 * 24)
        self.assertEqual(sum(event.operation == "qualify_block_SST" and event.when == "after"
                             for event in events), 24 * 24)
        self.assertTrue(np.all(np.sum((responses[group][stage][block] - baselines[group][stage][block]) ** 2,
                                     axis=0) > 0))
        for rows in (np.arange(2423), next(iter(validation.values()))):
            responses[group][stage] = original.copy()
            responses[group][stage][rows, 0] = 0.
            with self.assertRaisesRegex(nested.NestedPredictionError, "response_partition_variance_invalid"):
                nested._prepare_responses(design, responses, training, validation, blocks, lambda event: None)
        # Exactly balanced non-block rows make every frozen training mean zero.
        # Each complete fold retains variance, but this block has zero SST.
        values = np.zeros_like(original)
        for rows in validation.values():
            others = np.setdiff1d(rows, block)
            even = len(others) - len(others) % 2
            values[others[:even], 0] = np.tile((-1., 1.), even // 2)
        responses[group][stage] = values
        with self.assertRaisesRegex(nested.NestedPredictionError, "response_partition_SST_invalid"):
            nested._prepare_responses(design, responses, training, validation, blocks, lambda event: None)

    def test_audit_constant_block_keeps_frozen_mean_SST_and_whole_audit_variance(self):
        fixture = audit_fixture.fixture()
        design, _, predictors, _, responses, _, order = fixture[:7]
        group, stage = nested.GROUPS[0], "B"
        rows = np.flatnonzero(np.asarray(design.block_ids, dtype=object) == order[0])
        original = responses[group].stages[stage].copy()
        mean = predictors[audit_fixture.MODELS[0]].predictor.ridges[group][stage].target_mean

        def changed(column):
            panels = dict(responses)
            stages = dict(responses[group].stages)
            stages[stage] = original.copy()
            stages[stage][:, 0] = column
            panels[group] = replace(responses[group], stages=stages)
            return panels

        column = original[:, 0].copy()
        column[rows] = mean[0] + 1.
        events = []
        result = audit_fixture.apply(fixture, responses=changed(column), resource_check=events.append)
        np.testing.assert_allclose(result.paired_errors[group].sst[stage][0, 0], len(rows), rtol=0, atol=0)
        self.assertEqual(sum(event.operation == "qualify_audit_response_variance" and event.when == "after"
                             for event in events), 24)
        self.assertEqual(sum(event.operation == "qualify_audit_block_moments" and event.when == "after"
                             for event in events), 24 * 12)
        column[rows] = mean[0]
        with self.assertRaisesRegex(audit.AuditPredictionError, "audit_block_SST_nonpositive"):
            audit_fixture.apply(fixture, responses=changed(column))
        with self.assertRaisesRegex(nested.NestedPredictionError, "response_partition_variance_invalid"):
            audit_fixture.apply(fixture, responses=changed(np.full(606, mean[0] + 1.)))


if __name__ == "__main__":
    unittest.main()
