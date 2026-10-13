from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ep12_executor.contract_provenance import ACTIVE_EPOCH, PREFIX_EPOCH
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.post36_tail import ProductionSlotExecutor
from ep12_executor.production_falsifier_suite import ProductionFalsifierSuite


EPISODE_ROOT = Path(__file__).resolve().parents[3]


class ProductionSlotExecutorIntegrationTests(unittest.TestCase):
    def test_composite_control_is_routed_and_every_branch_artifact_is_bundled(self) -> None:
        active_contract_path = (
            EPISODE_ROOT / "outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml"
        )
        active_contract_sha256 = hashlib.sha256(
            active_contract_path.read_bytes()
        ).hexdigest()
        provenance = {
            "prefix_contract": {
                "epoch": PREFIX_EPOCH,
                "sha256": "1" * 64,
            },
            "active_contract": {
                "epoch": ACTIVE_EPOCH,
                "sha256": active_contract_sha256,
            },
            "transition": {
                "semantic_sha256": "2" * 64,
                "file_sha256": "3" * 64,
            },
        }
        active_fields = {
            "contract_epoch": ACTIVE_EPOCH,
            "scientific_contract_sha256": active_contract_sha256,
            "contract_transition_semantic_sha256": "2" * 64,
            "contract_transition_file_sha256": "3" * 64,
        }
        configuration = {"trial_id": "post36_falsifier_037", "synthetic": True}
        proposal = {
            "trial_id": configuration["trial_id"],
            "slot_kind": "mandatory_falsifier",
            "registered_falsifier_id": "cross_side_component_alignment_permutation",
            "transform_seed": 12012,
            "configuration": configuration,
            "configuration_sha256": digest_object(configuration),
            "slot_sha256": digest_object(["synthetic", "slot"]),
            "proposal_context_hash": digest_object(["synthetic", "context"]),
        }

        def fake_suite_execute(
            _suite: ProductionFalsifierSuite, *, proposal, attempt_dir: Path
        ):
            branch_root = attempt_dir / "control_branches"
            artifact_dir = branch_root / "001_weighted_baseline" / "synthetic_branch"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            report_hash = digest_object([proposal["trial_id"], "baseline-report"])
            return {
                "control_receipt": {
                    "schema_version": "ep12.post36_control_receipt.v1",
                    "falsifier_id": proposal["registered_falsifier_id"],
                    "implementation": "ProductionFalsifierSuite",
                    "transforms": [
                        {
                            "transform": "complete_prespecified_control_branch",
                            "branch_id": "unpermuted",
                        },
                        {
                            "transform": "complete_prespecified_control_branch",
                            "branch_id": "deterministic_within_K_permutation",
                        },
                    ],
                    "applicability_checks_passed": True,
                    "conservation_checks_passed": True,
                    "capacity_checks_passed": True,
                    "control_passed": True,
                    "development_roles_only": True,
                    "final_connectivity_accessed": False,
                },
                "representative": {
                    "report": {
                        "aggregate": {"technical_failure_count": 0},
                        **active_fields,
                    },
                    "verified": {
                        "valid_type_count": 3,
                        "unscorable_type_count": 0,
                        "objective": {
                            "left_to_right": 0.1,
                            "right_to_left": 0.1,
                            "bilateral": 0.1,
                            "bilateral_floor": 0.1,
                            "direction_disagreement": 0.0,
                            "types_both_above_margin": 3,
                        },
                        "trial_report_sha256": report_hash,
                        "manifest_sha256": digest_object(
                            [proposal["trial_id"], "baseline-manifest"]
                        ),
                    },
                },
                "branch_root": branch_root,
                "resource_totals": {"wall_seconds": 2.0, "cpu_seconds": 3.0},
            }

        with tempfile.TemporaryDirectory(
            prefix="ep12-slot-executor-test-", dir=EPISODE_ROOT
        ) as isolated_name:
            isolated_root = Path(isolated_name)
            outputs_root = isolated_root / "outputs"
            scratch_root = isolated_root / "scratch"
            durable = outputs_root / "tail"
            scratch = scratch_root / "tail"
            outputs_root.mkdir()
            scratch.mkdir(parents=True)
            snapshot = scratch / "snapshot"
            snapshot.mkdir()
            with mock.patch(
                "ep12_executor.post36_tail.validate_contract_provenance",
                return_value=provenance,
            ), mock.patch(
                "ep12_executor.contract_provenance.validate_contract_provenance",
                return_value=provenance,
            ), mock.patch(
                "ep12_executor.post36_tail.OUTPUTS_ROOT", outputs_root
            ), mock.patch(
                "ep12_executor.post36_tail.SCRATCH_ROOT", scratch_root
            ):
                executor = ProductionSlotExecutor(
                    tail_root=durable,
                    scratch_root=scratch,
                    snapshot_dir=snapshot,
                    policy=EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml"),
                    active_contract_path=active_contract_path,
                    contract_provenance=provenance,
                )
                with mock.patch.object(
                    ProductionFalsifierSuite, "execute", new=fake_suite_execute
                ):
                    realization = executor.execute(proposal)
                recovered = executor.recover(proposal)

            self.assertTrue(realization["control_receipt"]["control_passed"])
            self.assertAlmostEqual(realization["resources"]["wall_hours"], 2 / 3600)
            self.assertAlmostEqual(
                realization["resources"]["cpu_core_hours"], 3 / 3600
            )
            slot_dir = durable / "slots" / proposal["trial_id"]
            branch_relative = (
                "control_branches/001_weighted_baseline/"
                "synthetic_branch/manifest.json"
            )
            manifest = json.loads(
                (slot_dir / "bundle_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn(branch_relative, manifest["files"])
            self.assertTrue((slot_dir / branch_relative).is_file())
            self.assertEqual(
                recovered["evidence"]["bundle_manifest_sha256"],
                realization["evidence"]["bundle_manifest_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
