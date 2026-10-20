"""Inactive audit gate/adapter checks; coordinated compute only, no audit I/O."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import observed_map_audit_context as context
import observed_map_audit_predictions as prediction
from calibration_falsifiers import TASK_A_HALF_RUNS, TaskACalibrationPacket, condition_relabeling
from development_lopo import FoldContext, validate_source_context
from local_functional_source import TASK_A_CONDITIONS


def policy():
    # In-memory fixture only. No active policy file or adoption is created.
    return {context.ADOPTION_KEY: {"status": "adopted", "specification": context.SPECIFICATION,
            "authority": "test fixture only", "adopted_at_utc": "2000-01-01T00:00:00Z"}}


class ForbiddenInput:
    def __array__(self, *args, **kwargs):
        raise AssertionError("closed input was converted")

    def __iter__(self):
        raise AssertionError("closed input was iterated")


class AuditPredictionAdapterTests(unittest.TestCase):
    def test_absent_or_incomplete_adoption_refuses_before_any_supplied_arrays(self):
        blocked = ForbiddenInput()
        for denied in ({}, {context.ADOPTION_KEY: {"status": "proposed"}},
                       {context.ADOPTION_KEY: {**policy()[context.ADOPTION_KEY], "authority": ""}},
                       {context.ADOPTION_KEY: {**policy()[context.ADOPTION_KEY], "specification": "other.md"}}):
            with self.assertRaises(PermissionError):
                prediction.fit_audit_predictions(blocked, blocked, common_support=blocked,
                    forward_header=blocked, policy=denied, check=lambda: None)
            with self.assertRaises(PermissionError):
                context.validate_prediction_bundle(blocked, policy=denied)

    def test_audit_context_keeps_all_sources_and_legacy_guard_stays_closed(self):
        for target in context.AUDIT_IDS:
            supplied = context.AuditContext(target)
            context.validate_audit_context(supplied)
            with self.assertRaises(ValueError):
                validate_source_context(supplied)
        invalid = (context.AuditContext(context.DEVELOPMENT_IDS[0]),
                   context.AuditContext(context.AUDIT_IDS[0], training_ids=context.DEVELOPMENT_IDS[:-1]),
                   context.AuditContext(context.AUDIT_IDS[0], deleted_id=context.DEVELOPMENT_IDS[0]),
                   context.AuditContext(context.AUDIT_IDS[0], donor_id=context.AUDIT_IDS[0]))
        for supplied in invalid:
            with self.assertRaises(ValueError):
                context.validate_audit_context(supplied)
        pairings = context.audit_identity_pairings()
        self.assertEqual(tuple(person for person, _ in pairings), context.AUDIT_IDS)
        self.assertEqual(set(donor for _, donor in pairings), set(context.AUDIT_IDS))
        self.assertTrue(all(person != donor for person, donor in pairings))

    def test_condition_mapping_retains_original_strata_algorithm_and_seed(self):
        ids = tuple(range(188))
        maps = np.sin(np.arange(29)[:, None] * .13 + np.arange(188)[None, :] * .07)
        halves = np.stack((maps, maps + .03 * np.cos(np.arange(188))[None, :]))
        mean = halves.mean(axis=0)
        # All source people have identical known algebra fixtures, so using
        # twelve rather than eleven cannot change the source median strata.
        packets = tuple(TaskACalibrationPacket(person, TASK_A_CONDITIONS, ids,
            mean, halves, TASK_A_HALF_RUNS) for person in context.DEVELOPMENT_IDS)
        ordinary = FoldContext(context.DEVELOPMENT_IDS[0], context.DEVELOPMENT_IDS[1:], None)
        expected, _ = condition_relabeling(packets[1:], ordinary, ids,
                                            numerical_atol=1e-10, support_size=188)
        self.assertEqual(prediction._condition_mapping(packets, ids), expected)
        self.assertEqual(expected[28], 28)
        self.assertTrue(all(expected[index] != index for index in range(28)))

    def test_matched_warp_keeps_actual_audit_seed_and_all_four_by_eight_draws(self):
        coordinates = np.zeros((188, 3))
        fields = tuple(SimpleNamespace(variant=variant, coordinates=coordinates)
                       for variant in prediction.WARP_VARIANTS)
        source = SimpleNamespace(geometry_state=SimpleNamespace(source_partitions=np.ones((4, 18, 188))))
        predictor = SimpleNamespace(fields=fields, representation=SimpleNamespace(coordinates=coordinates),
            source_recipe=SimpleNamespace(_source=Mock()),
            predict_partitions=lambda field, partitions, coefficients, gain: partitions * gain)
        models = tuple(prediction.M2Prediction(field.variant.name, np.zeros((18, 188)), np.zeros((18, 18)),
            {"training_ids": context.DEVELOPMENT_IDS, "variant": field.variant.name,
             "common_gain": 1., "coefficients": np.arange(24, dtype=float).reshape(8, 3) * .001})
            for field in fields)
        target = context.AuditContext("sub-02")
        with patch.object(prediction, "_geometry", return_value={}):
            results = prediction._matched_warps(predictor, source, models, 1., target, lambda: None)
        self.assertEqual(len(results), 32)
        for index, result in enumerate(results):
            self.assertEqual(result.diagnostics["seed_entropy"], (20261007, 2, index // 8, index % 8))
            self.assertEqual(result.diagnostics["target_id"], "sub-02")
            self.assertEqual(result.diagnostics["training_ids"], context.DEVELOPMENT_IDS)
            self.assertFalse(result.diagnostics["degenerate_identity_control"])
            self.assertFalse(result.diagnostics["omnibus_random_field_null"])
        for model in models:
            model.diagnostics["coefficients"] = np.zeros((8, 3))
        with patch.object(prediction, "_geometry", return_value={}):
            degenerate = prediction._matched_warps(predictor, source, models, 1., target, lambda: None)
        self.assertTrue(all(value.diagnostics["degenerate_identity_control"] for value in degenerate))
        self.assertTrue(all(not np.any(value.diagnostics["coefficients"]) for value in degenerate))
        with patch.object(prediction, "_geometry", side_effect=ArithmeticError("invalid parent field")):
            with self.assertRaisesRegex(ArithmeticError, "invalid parent"):
                prediction._matched_warps(predictor, source, models, 1., target, lambda: None)


if __name__ == "__main__":
    unittest.main()
