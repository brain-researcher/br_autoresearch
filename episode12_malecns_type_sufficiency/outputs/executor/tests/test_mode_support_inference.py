from __future__ import annotations

import copy
import unittest

import numpy as np

from ep12_executor.mode_support_inference import (
    BOOTSTRAP_REPLICATES,
    ModeSupportInferenceError,
    infer_mode_support,
    verify_mode_support_result,
)


CONTRACT_SHA256 = "1" * 64
CONFIGURATION_SHA256 = "2" * 64
RESULT_SHA256 = "3" * 64


class ModeSupportInferenceTests(unittest.TestCase):
    def evidence(self, *, n: int = 16) -> dict[str, object]:
        offset = np.linspace(-0.002, 0.002, n)
        return {
            "comparison_trial_id": "trial_selected",
            "configuration_sha256": CONFIGURATION_SHA256,
            "contract_sha256": CONTRACT_SHA256,
            "result_table_sha256": RESULT_SHA256,
            "meaningful_margin": 0.02,
            "prevalence_parameter_rank_definition": (
                "synthetic_fitted_component_parameter_rank"
            ),
            "whole_provider_type_ids": [f"type_{index:02d}" for index in range(n)],
            "prediction_gain_by_type_direction": np.column_stack(
                (0.08 + offset, 0.075 - offset)
            ),
            "component_symmetric_kl_by_type_direction": np.stack(
                (
                    np.column_stack((0.70 + offset, 0.80 - offset)),
                    np.column_stack((0.75 - offset, 0.85 + offset)),
                ),
                axis=1,
            ),
            "matched_contrast_cosine_by_type": np.column_stack(
                (0.82 + offset, 0.76 - offset)
            ),
            "partner_contrast_by_type_direction": np.stack(
                (
                    np.column_stack((0.50 + offset, -0.40 + offset)),
                    np.column_stack((0.45 - offset, -0.35 - offset)),
                ),
                axis=1,
            ),
            "effective_memberships_by_direction_component": np.tile(
                np.asarray([[[12.0, 11.0], [10.0, 13.0]]]), (n, 1, 1)
            ),
            "fitted_parameter_ranks_by_direction_component": np.tile(
                np.asarray([[[3, 3], [3, 3]]]), (n, 1, 1)
            ),
        }

    def test_true_mode_support_is_joint_and_replayable(self) -> None:
        inputs = self.evidence()
        first = infer_mode_support(**inputs)
        second = infer_mode_support(**inputs)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap"]["replicates"], BOOTSTRAP_REPLICATES)
        self.assertTrue(first["bootstrap"]["directions_resampled_together"])
        self.assertEqual(first["bootstrap"]["unit"], "whole_provider_type")
        self.assertTrue(first["scientific_mode_support"])
        self.assertEqual(first["support_state"], "supported")
        self.assertEqual(
            first["diagnostics"]["partner_contrast"][
                "same_sign_terms_excluding_zero_in_both_source_fits"
            ],
            [0, 1],
        )
        verify_mode_support_result(first)

    def test_no_mode_pattern_does_not_make_a_positive_claim(self) -> None:
        inputs = self.evidence()
        n = len(inputs["whole_provider_type_ids"])
        inputs["component_symmetric_kl_by_type_direction"] = np.zeros((n, 2, 2))
        inputs["matched_contrast_cosine_by_type"] = np.zeros((n, 2))
        partner = np.zeros((n, 2, 2))
        partner[:, 0, 0] = 0.4
        partner[:, 1, 0] = -0.4
        inputs["partner_contrast_by_type_direction"] = partner
        result = infer_mode_support(**inputs)
        self.assertFalse(result["scientific_mode_support"])
        self.assertEqual(result["support_state"], "unresolved")
        self.assertFalse(result["gates"]["component_separation"])
        self.assertFalse(result["gates"]["cross_side_matching"])
        self.assertFalse(result["gates"]["partner_identifiability"])

    def test_weak_point_gain_fails_strict_simultaneous_lower_bound(self) -> None:
        inputs = self.evidence(n=12)
        alternating = np.array([0.000, 0.050] * 6)
        # Both point means exceed 0.02, but the simultaneous lower bounds do not.
        inputs["prediction_gain_by_type_direction"] = np.column_stack(
            (alternating, alternating[::-1])
        )
        result = infer_mode_support(**inputs)
        means = result["diagnostics"]["prediction_gain"]["estimate_by_direction"]
        self.assertTrue(all(value > 0.02 for value in means))
        self.assertFalse(result["gates"]["prediction_gain"])
        self.assertFalse(result["null_eligible_from_mode_support"])

    def test_prevalence_is_strictly_greater_than_fitted_rank(self) -> None:
        inputs = self.evidence()
        memberships = np.asarray(
            inputs["effective_memberships_by_direction_component"]
        ).copy()
        memberships[0, 0, 0] = 3.0
        inputs["effective_memberships_by_direction_component"] = memberships
        result = infer_mode_support(**inputs)
        self.assertFalse(result["gates"]["prevalence"])
        self.assertFalse(result["scientific_mode_support"])

    def test_missing_evidence_is_authenticated_unresolved(self) -> None:
        inputs = self.evidence()
        inputs["matched_contrast_cosine_by_type"] = None
        result = infer_mode_support(**inputs)
        self.assertFalse(result["complete_input"])
        self.assertEqual(result["support_state"], "unresolved")
        self.assertIn("matched_contrast_cosine_by_type", result["missing_inputs"])
        verify_mode_support_result(result)

    def test_result_tamper_and_direction_shape_fail_closed(self) -> None:
        inputs = self.evidence()
        result = infer_mode_support(**inputs)
        tampered = copy.deepcopy(result)
        tampered["gates"]["prediction_gain"] = False
        with self.assertRaisesRegex(ModeSupportInferenceError, "digest mismatch"):
            verify_mode_support_result(tampered)
        inputs["prediction_gain_by_type_direction"] = np.ones((16, 1))
        with self.assertRaisesRegex(ModeSupportInferenceError, "exactly the two"):
            infer_mode_support(**inputs)


if __name__ == "__main__":
    unittest.main()
