from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ep12_executor.falsifier_controls import (
    component_partner_contrasts,
    make_branch_manifest,
)
from ep12_executor.policy import digest_object
from ep12_executor.production_falsifier_suite import ProductionFalsifierSuite


class SyntheticSuite(ProductionFalsifierSuite):
    def __init__(self) -> None:
        self.snapshot_dir = Path("/synthetic/not-read")
        self.policy = SimpleNamespace(
            cpu_cores_per_trial_maximum=32,
            grammar_choices={
                "partner_vocabulary": ["vocabulary_a", "vocabulary_b"],
                "rank": [2, 4],
            }
        )
        from ep12_executor.falsifier_controls import implementation_spec

        self.spec = implementation_spec()
        basis = np.asarray(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(6)],
                [-1 / np.sqrt(2), 1 / np.sqrt(6)],
                [0.0, -2 / np.sqrt(6)],
            ]
        )
        self.contrast = component_partner_contrasts(
            basis=basis,
            component_means=np.asarray([[1.0, 0.2], [-1.0, -0.2]]),
            component_weights=np.asarray([0.5, 0.5]),
        )
        self.executed: list[str] = []

    def _inventories(self):  # type: ignore[override]
        return (
            [
                {
                    "branch_id": "partner_family::depth1::synthetic",
                    "raw_partner_keys": ["typed::A"],
                }
            ],
            [
                {"branch_id": "endpoint::unknown", "endpoint_bin": "unknown"},
                {"branch_id": "endpoint::fragment", "endpoint_bin": "fragment"},
            ],
        )

    def _synthetic_rows(self) -> list[dict[str, Any]]:
        rows = []
        for type_index, provider_type in enumerate(("A", "B", "C"), start=1):
            scores = {"T": 0.1, "U": 0.2, "M": 0.3}
            targets = {
                direction: [
                    {
                        "body_id": f"{provider_type}-{direction}-{index}",
                        "strength": type_index * 10 + index,
                        "T": 0.1,
                        "U": 0.2,
                        "M": 0.3,
                        "delta": 0.1,
                    }
                    for index in range(3)
                ]
                for direction in ("left_to_right", "right_to_left")
            }
            scale = 1.0 if type_index != 2 else -1.0
            right = {
                **self.contrast,
                "partner_contrasts": (
                    np.asarray(self.contrast["partner_contrasts"]) * scale
                ).tolist(),
            }
            rows.append(
                {
                    "provider_type": provider_type,
                    "left_to_right_scores": scores,
                    "right_to_left_scores": scores,
                    "left_to_right_delta": 0.1 * type_index,
                    "right_to_left_delta": 0.1 * type_index,
                    "target_neuron_scores": targets,
                    "source_fit_metadata": {
                        "left": {"M_component_partner_contrasts": self.contrast},
                        "right": {"M_component_partner_contrasts": right},
                    },
                }
            )
        return rows

    def _run_fit_branch(  # type: ignore[override]
        self,
        *,
        proposal: Mapping[str, Any],
        branch_root: Path,
        index: int,
        branch_id: str,
        overrides: Mapping[str, Any] | None = None,
        control_spec: Mapping[str, Any] | None = None,
        capacity_reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.executed.append(branch_id)
        signature = {
            "models": ["T_type_template", "U_flexible_unimodal", "M_residual_modes"],
            "representation": "count_aware_log_ratio",
            "rank": 2,
            "covariance": "diagonal",
            "covariance_rank": 2,
            "component_count": 2,
            "regularization": 0.1,
        }
        capacity_ok = capacity_reference is None or dict(capacity_reference) == signature
        report_hash = digest_object([proposal["registered_falsifier_id"], branch_id, "report"])
        manifest = make_branch_manifest(
            control_id=str(proposal["registered_falsifier_id"]),
            branch_id=branch_id,
            branch_kind="complete_triplet_fit",
            complete_triplet_sha256=report_hash,
            applicability_checks_passed=True,
            conservation_checks_passed=True,
            capacity_checks_passed=capacity_ok,
            artifacts={"trial_report.json": report_hash},
        )
        result_root = branch_root / f"{index:03d}_synthetic"
        result_root.mkdir(parents=True, exist_ok=False)
        (result_root / "branch_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "branch_id": branch_id,
            "manifest": manifest,
            "report": {
                "complete_type_records": self._synthetic_rows(),
                "wall_seconds": 0.1,
                "cpu_seconds": 0.1,
            },
            "verified": {"trial_report_sha256": report_hash},
            "capacity_signature": signature,
            "result_root": result_root,
        }


class ProductionFalsifierSuiteTests(unittest.TestCase):
    def _proposal(self, control_id: str) -> dict[str, Any]:
        return {
            "trial_id": "synthetic_slot",
            "registered_falsifier_id": control_id,
            "transform_seed": 12012,
            "configuration": {
                "nuisance_spline_df": 0,
                "nuisance": [
                    "connection_strength_exposure",
                    "reconstruction_and_status",
                    "prespecified_anatomical_effects",
                ],
            },
        }

    def test_every_composite_control_dispatches_its_complete_branch_set(self) -> None:
        controls = (
            "cross_side_component_alignment_permutation",
            "binary_weighted_vocabulary_and_rank_sensitivities",
            "nuisance_and_partner_block_ablations",
            "capacity_matched_noise_controls",
            "leave_one_development_type_and_partner_family_influence",
            "few_type_and_high_strength_neuron_concentration_check",
            "status_endpoint_anatomy_and_missingness_checks",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, control_id in enumerate(controls):
                with self.subTest(control_id=control_id):
                    suite = SyntheticSuite()
                    result = suite.execute(
                        proposal=self._proposal(control_id),
                        attempt_dir=root / f"control_{index}",
                    )
                    receipt = result["control_receipt"]
                    self.assertTrue(receipt["control_passed"])
                    self.assertTrue(receipt["complete_prespecified_branch_set_executed"])
                    self.assertTrue(receipt["every_branch_complete_T_U_M"])
                    self.assertTrue(receipt["branch_hashes_valid"])
                    self.assertIsNone(receipt["new_scientific_pass_threshold"])
                    self.assertFalse(receipt["robustness_or_support_decided"])
                    self.assertEqual(
                        receipt["expected_branch_ids"], receipt["observed_branch_ids"]
                    )
                    self.assertAlmostEqual(
                        result["resource_totals"]["branch_wall_seconds_sum"],
                        0.1 * len(suite.executed),
                    )
                    self.assertAlmostEqual(
                        result["resource_totals"]["cpu_seconds"],
                        0.1 * len(suite.executed),
                    )

    def test_worker_count_uses_allocation_without_exceeding_frozen_cap(self) -> None:
        suite = SyntheticSuite()
        prior = os.environ.get("SLURM_CPUS_PER_TASK")
        try:
            os.environ["SLURM_CPUS_PER_TASK"] = "32"
            self.assertEqual(suite._worker_count(74), 32)
            self.assertEqual(suite._worker_count(11), 11)
        finally:
            if prior is None:
                os.environ.pop("SLURM_CPUS_PER_TASK", None)
            else:
                os.environ["SLURM_CPUS_PER_TASK"] = prior


if __name__ == "__main__":
    unittest.main()
