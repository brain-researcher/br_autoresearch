from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from ep12_executor.falsifier_controls import (
    FalsifierControlError,
    ablate_nuisance_block,
    component_partner_contrasts,
    composite_control_receipt,
    cross_side_alignment_permutation,
    endpoint_branch_inventory,
    implementation_spec,
    label_invariant_component_score,
    leave_one_type_influence,
    make_branch_manifest,
    neuron_influence_curves,
    nuisance_block_columns,
    partner_family_inventory,
    pooled_multinomial_noise,
    pooled_multinomial_noise_sparse,
    validate_implementation_spec,
)
from ep12_executor.policy import digest_object


EPISODE_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    EPISODE_ROOT
    / "outputs"
    / "executor"
    / "FALSIFIER_CONTROL_IMPLEMENTATION_SPEC.json"
)


class FalsifierControlTests(unittest.TestCase):
    def test_static_spec_matches_frozen_threshold_free_definition(self) -> None:
        payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        validated = validate_implementation_spec(payload)
        self.assertEqual(validated, implementation_spec())
        semantics = validated["control_pass_semantics"]
        self.assertIsNone(semantics["new_scientific_pass_threshold"])
        self.assertFalse(semantics["robustness_or_residual_mode_support_decided_here"])
        self.assertFalse(validated["final_connectivity_accessed"])

    def test_capacity_matched_noise_is_deterministic_and_preserves_every_row_total(self) -> None:
        counts = np.asarray([[4, 1, 0], [0, 3, 5], [2, 2, 2]], dtype=np.int64)
        first, receipt = pooled_multinomial_noise(counts, seed=12012)
        second, second_receipt = pooled_multinomial_noise(counts, seed=12012)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, counts.shape)
        self.assertTrue(np.array_equal(first.sum(axis=1), counts.sum(axis=1)))
        self.assertTrue(receipt["same_shape"])
        self.assertTrue(receipt["row_totals_preserved"])
        self.assertEqual(receipt, second_receipt)
        sparse_first, sparse_receipt = pooled_multinomial_noise_sparse(
            counts, seed=12012
        )
        sparse_second, sparse_second_receipt = pooled_multinomial_noise_sparse(
            counts, seed=12012
        )
        self.assertTrue(np.array_equal(sparse_first.toarray(), first))
        self.assertTrue(np.array_equal(sparse_second.toarray(), first))
        self.assertEqual(sparse_receipt, sparse_second_receipt)
        self.assertTrue(sparse_receipt["total_mass_preserved"])

    def test_all_frozen_nuisance_blocks_are_nonempty_and_common(self) -> None:
        for spline_df, width in ((0, 22), (3, 34), (5, 46)):
            with self.subTest(spline_df=spline_df):
                blocks = nuisance_block_columns(spline_df=spline_df, design_width=width)
                self.assertEqual(
                    set(blocks),
                    {
                        "connection_strength_exposure",
                        "reconstruction_and_status",
                        "prespecified_anatomical_effects",
                        "missingness_indicators",
                    },
                )
                source = np.arange(3 * width, dtype=float).reshape(3, width)
                for block_id, columns in blocks.items():
                    ablated, receipt = ablate_nuisance_block(
                        source, spline_df=spline_df, block_id=block_id
                    )
                    self.assertTrue(np.all(ablated[:, columns] == 0.0))
                    self.assertEqual(receipt["applied_to_models"], ["T", "U", "M"])

    def test_hierarchy_and_endpoint_inventories_exhaust_nonempty_branches(self) -> None:
        vocabulary = [
            "typed::A",
            "typed::B",
            "typed::C",
            "untyped_status::Traced::",
            "untyped_status::Anchor::anchor",
            "missing_annotation",
        ]
        hierarchy = {
            "A": {"subclass": "s1", "class": "c1"},
            "B": {"subclass": "s1", "class": "c2"},
            "C": {"subclass": "s2", "class": "c2"},
        }
        families = partner_family_inventory(hierarchy, vocabulary)
        self.assertEqual(
            [value["branch_id"] for value in families],
            [
                "partner_family::depth1::s1",
                "partner_family::depth1::s2",
                "partner_family::depth2::c1",
                "partner_family::depth2::c2",
            ],
        )
        counts = np.asarray([[1, 2, 0, 3, 0, 4], [0, 0, 1, 0, 2, 0]])
        endpoints = endpoint_branch_inventory(vocabulary, counts)
        self.assertEqual(
            {value["endpoint_bin"] for value in endpoints},
            {"other_typed", "untyped", "proofreading", "missing_annotation"},
        )

    def test_component_contrasts_and_alignment_are_label_invariant(self) -> None:
        basis = np.asarray(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(6)],
                [0.0, -2 / np.sqrt(6)],
            ]
        )
        contrast = component_partner_contrasts(
            basis=basis,
            component_means=np.asarray([[1.0, 0.2], [-1.0, -0.2]]),
            component_weights=np.asarray([0.5, 0.5]),
        )
        weighted = np.asarray(contrast["component_weights"]) @ np.asarray(
            contrast["partner_contrasts"]
        )
        self.assertTrue(np.allclose(weighted, 0.0))
        permuted = np.asarray(contrast["partner_contrasts"])[::-1]
        score = label_invariant_component_score(
            contrast["partner_contrasts"], permuted
        )
        self.assertAlmostEqual(score["score"], 1.0)

        records = [
            {
                "provider_type": name,
                "left": contrast,
                "right": {
                    **contrast,
                    "partner_contrasts": (
                        np.asarray(contrast["partner_contrasts"]) * scale
                    ).tolist(),
                },
            }
            for name, scale in (("A", 1.0), ("B", -1.0), ("C", 0.5))
        ]
        singleton = {
            "component_count": 3,
            "component_weights": [1 / 3, 1 / 3, 1 / 3],
            "partner_contrasts": [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
                [-1.0, -1.0, 2.0],
            ],
        }
        records.append(
            {
                "provider_type": "D",
                "left": singleton,
                "right": singleton,
            }
        )
        receipt = cross_side_alignment_permutation(records, seed=17)
        self.assertEqual(
            [value["branch_id"] for value in receipt["branches"]],
            ["unpermuted", "deterministic_within_K_permutation"],
        )
        permuted_pairs = receipt["branches"][1]["pair_scores"]
        self.assertTrue(
            all(
                value["provider_type"] != value["right_provider_type"]
                for value in permuted_pairs
                if value["component_count"] == 2
            )
        )
        singleton_pair = [
            value for value in permuted_pairs if value["component_count"] == 3
        ]
        self.assertEqual(singleton_pair[0]["right_provider_type"], "D")

    def test_complete_type_and_neuron_influence_curves_add_no_cutoff(self) -> None:
        records = []
        for type_index, provider_type in enumerate(("A", "B", "C"), start=1):
            directional_scores = {"T": 0.1, "U": 0.2, "M": 0.3}
            target = {
                direction: [
                    {
                        "body_id": f"{provider_type}-{direction}-{neuron}",
                        "strength": 10 * type_index + neuron,
                        "T": 0.1 + neuron / 100,
                        "U": 0.2 + neuron / 100,
                        "M": 0.3 + neuron / 100,
                        "delta": 0.1,
                    }
                    for neuron in range(3)
                ]
                for direction in ("left_to_right", "right_to_left")
            }
            records.append(
                {
                    "provider_type": provider_type,
                    "left_to_right_scores": directional_scores,
                    "right_to_left_scores": directional_scores,
                    "left_to_right_delta": 0.1 * type_index,
                    "right_to_left_delta": 0.1 * type_index,
                    "target_neuron_scores": target,
                }
            )
        type_curve = leave_one_type_influence(records)
        self.assertTrue(type_curve["curve_complete"])
        self.assertEqual(len(type_curve["leave_one_type_curve"]), 3)
        neuron_curves = neuron_influence_curves(records)
        self.assertTrue(neuron_curves["leave_one_neuron_curve_complete"])
        self.assertTrue(neuron_curves["high_strength_curve_complete"])
        self.assertEqual(len(neuron_curves["leave_one_neuron_curve"]), 18)
        self.assertEqual(len(neuron_curves["high_strength_cumulative_curve"]), 12)
        self.assertIsNone(neuron_curves["new_concentration_cutoff"])

    def test_branch_manifests_require_exact_complete_set_and_all_checks(self) -> None:
        control_id = "capacity_matched_noise_controls"
        branches = [
            make_branch_manifest(
                control_id=control_id,
                branch_id=branch_id,
                branch_kind="complete_triplet_fit",
                complete_triplet_sha256=digest_object([branch_id, "triplet"]),
                applicability_checks_passed=True,
                conservation_checks_passed=True,
                capacity_checks_passed=True,
                artifacts={"trial_report.json": digest_object([branch_id, "report"])},
            )
            for branch_id in ("weighted_baseline", "pooled_multinomial_noise")
        ]
        receipt = composite_control_receipt(
            control_id=control_id,
            expected_branch_ids=["weighted_baseline", "pooled_multinomial_noise"],
            branches=branches,
            implementation_spec_sha256=implementation_spec()["spec_sha256"],
        )
        self.assertTrue(receipt["control_passed"])
        self.assertFalse(receipt["robustness_or_support_decided"])
        with self.assertRaises(FalsifierControlError):
            composite_control_receipt(
                control_id=control_id,
                expected_branch_ids=["weighted_baseline", "pooled_multinomial_noise"],
                branches=branches[:1],
                implementation_spec_sha256=implementation_spec()["spec_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
