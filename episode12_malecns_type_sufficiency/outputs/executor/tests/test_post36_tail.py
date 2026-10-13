from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from ep12_executor.falsifier_controls import (
    composite_control_receipt,
    make_branch_manifest,
)
from ep12_executor.policy import EpisodePolicy, digest_bytes
from ep12_executor.post36_tail import (
    assess_observed_search_wall_budget,
    capability_report,
    observed_search_wall_budget,
    procedure_lock_handoff_permitted,
    production_control_capabilities,
    promotion_decision,
    verify_readiness_directory,
    verify_recovered_slot_artifacts,
)
from ep12_executor.production_readiness import REQUIRED_READINESS_CHECK_IDS
from ep12_executor.runtime import EPISODE_ROOT

import observed_tail_plan


def _record(trial_id: str, score: float, configuration_sha256: str) -> dict:
    return {
        "trial_id": trial_id,
        "configuration_sha256": configuration_sha256,
        "objective": {
            "bilateral_floor": score,
            "bilateral": score,
            "direction_disagreement": 0.0,
            "types_both_above_margin": 2,
        },
    }


class Post36TailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    def test_controls_cover_every_registered_countable_falsifier(self) -> None:
        report = capability_report(
            observed_tail_plan.build_plan(), production_control_capabilities()
        )
        self.assertTrue(report["whole_plan_executable"])
        self.assertEqual(report["unsupported_falsifiers"], [])

    def test_control_cannot_replace_the_primary_incumbent(self) -> None:
        incumbent = _record("incumbent", 0.01, "a" * 64)
        stronger_control = _record("control", 1.0, "b" * 64)
        decision = promotion_decision(
            incumbent=incumbent,
            candidate=stronger_control,
            promotion_eligible=False,
        )
        self.assertFalse(decision["improved"])
        self.assertEqual(decision["incumbent_trial_id"], "incumbent")

    def test_wall_deadline_gate_is_outcome_blind(self) -> None:
        budget = observed_search_wall_budget(self.policy)
        early = assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=budget,
            now_utc=datetime(2026, 9, 26, 6, tzinfo=timezone.utc),
        )
        late = assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=budget,
            now_utc=datetime(2026, 10, 1, 5, tzinfo=timezone.utc),
        )
        self.assertTrue(early["can_start_next_tail_slot"])
        self.assertFalse(late["can_start_next_tail_slot"])
        self.assertFalse(late["outcome_values_read_for_decision"])

    def test_procedure_lock_handoff_requires_explicit_normal_stop_permission(self) -> None:
        stop = {
            "completed_promotable_search": True,
            "promotion_gates": {"all_required_gates": True},
            "procedure_lock_submission_permitted": True,
            "next_tail_slot_submission_permitted": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        self.assertTrue(
            procedure_lock_handoff_permitted(action="stopped", stop_record=stop)
        )
        self.assertFalse(
            procedure_lock_handoff_permitted(action="continue", stop_record=stop)
        )
        self.assertFalse(
            procedure_lock_handoff_permitted(
                action="stopped",
                stop_record={**stop, "procedure_lock_submission_permitted": False},
            )
        )
        self.assertFalse(
            procedure_lock_handoff_permitted(
                action="stopped",
                stop_record={
                    **stop,
                    "completed_promotable_search": False,
                    "promotion_gates": {"all_required_gates": False},
                },
            )
        )

    def _passing_readiness_report(self) -> dict:
        return {
            "status": "pass",
            "policy_id": self.policy.policy_id,
            "production_readiness_passed": True,
            "checks": [
                {
                    "check_id": check_id,
                    "passed": True,
                    "evidence": {"verified": True},
                }
                for check_id in REQUIRED_READINESS_CHECK_IDS
            ],
            "failed_foundational_checks": [],
            "claim_guards": {
                "observed_post36_tail_launch": {"permitted": True, "missing": []}
            },
            "observed_outcome_values_read": False,
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }

    def test_readiness_consumer_requires_exact_nonempty_check_inventory(self) -> None:
        mutations = {
            "empty": [],
            "missing": [
                {
                    "check_id": check_id,
                    "passed": True,
                    "evidence": {"verified": True},
                }
                for check_id in REQUIRED_READINESS_CHECK_IDS[:-1]
            ],
            "extra": [
                *self._passing_readiness_report()["checks"],
                {"check_id": "unexpected", "passed": True, "evidence": {}},
            ],
        }
        for mutation, checks in mutations.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="ep12-readiness-consumer-"
            ) as name:
                root = Path(name)
                directory = root / "readiness"
                directory.mkdir()
                report = self._passing_readiness_report()
                report["checks"] = checks
                (directory / "production_readiness.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                with self.assertRaisesRegex(Exception, "checks are missing|inventory"):
                    verify_readiness_directory(directory, outputs_root=root)

    def test_readiness_consumer_accepts_exact_passing_check_inventory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-consumer-ok-") as name:
            root = Path(name)
            directory = root / "readiness"
            directory.mkdir()
            report = self._passing_readiness_report()
            (directory / "production_readiness.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            verified = verify_readiness_directory(directory, outputs_root=root)
            self.assertEqual(verified["policy_id"], self.policy.policy_id)

    def _recovery_fixture(self, root: Path) -> tuple[dict, dict]:
        proposal = {
            "trial_id": "slot001",
            "configuration": {"trial_id": "slot001", "rank": 2},
        }
        control = {"control_passed": True, "falsifier_id": None}
        (root / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
        (root / "control_result.json").write_text(
            json.dumps(control), encoding="utf-8"
        )
        trial_dir = root / "result" / "slot001"
        trial_dir.mkdir(parents=True)
        (trial_dir / "result_table.csv").write_text("cell,value\n", encoding="utf-8")
        realization = {
            "control_result": control,
            "objective": {"bilateral_floor": 0.1},
            "valid_type_count": 8,
            "unscorable_type_count": 0,
            "evidence": {
                "proposal": "proposal.json",
                "control_result": "control_result.json",
                "result_table": "result/slot001/result_table.csv",
            },
        }
        return proposal, realization

    def _composite_recovery_fixture(
        self, root: Path
    ) -> tuple[dict, dict, Path]:
        control_id = "binary_weighted_vocabulary_and_rank_sensitivities"
        proposal = {
            "trial_id": "slot001",
            "registered_falsifier_id": control_id,
            "configuration": {"trial_id": "slot001", "rank": 2},
        }
        (root / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
        branch_root = root / "control_branches"

        def write_branch(index: int, branch_id: str) -> tuple[dict, Path]:
            result_root = branch_root / f"{index:03d}_{branch_id}"
            trial_dir = result_root / f"slot001__branch_{index:03d}"
            trial_dir.mkdir(parents=True)
            contents = {
                "trial_report.json": b'{"status":"completed"}\n',
                "manifest.json": b'{"final_connectivity_accessed":false}\n',
                "result_table.csv": b"cell,value\n",
                "type_direction_receipts.jsonl": b'{"status":"valid"}\n',
            }
            for name, content in contents.items():
                (trial_dir / name).write_bytes(content)
            artifacts = {
                name: digest_bytes(content) for name, content in contents.items()
            }
            manifest = make_branch_manifest(
                control_id=control_id,
                branch_id=branch_id,
                branch_kind="complete_triplet_fit",
                complete_triplet_sha256=artifacts["trial_report.json"],
                applicability_checks_passed=True,
                conservation_checks_passed=True,
                capacity_checks_passed=True,
                artifacts=artifacts,
            )
            (result_root / "branch_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )
            return manifest, trial_dir

        baseline, baseline_trial_dir = write_branch(1, "weighted_baseline")
        alternative, alternative_trial_dir = write_branch(2, "binary")
        control = composite_control_receipt(
            control_id=control_id,
            expected_branch_ids=["weighted_baseline", "binary"],
            branches=[baseline, alternative],
            implementation_spec_sha256="ab" * 32,
        )
        (root / "control_result.json").write_text(
            json.dumps(control), encoding="utf-8"
        )
        realization = {
            "control_result": control,
            "objective": {"bilateral_floor": 0.1},
            "valid_type_count": 8,
            "unscorable_type_count": 0,
            "evidence": {
                "proposal": "proposal.json",
                "control_result": "control_result.json",
                "result_table": str(
                    (baseline_trial_dir / "result_table.csv").relative_to(root)
                ),
            },
        }
        return (
            proposal,
            realization,
            alternative_trial_dir / "type_direction_receipts.jsonl",
        )

    def test_recovery_reauthenticates_scientific_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-recovery-") as name:
            root = Path(name)
            proposal, realization = self._recovery_fixture(root)
            verified = {
                "objective": realization["objective"],
                "valid_type_count": 8,
                "unscorable_type_count": 0,
            }
            with patch(
                "ep12_executor.control_trial.verify_development_trial_artifacts",
                return_value=verified,
            ) as verifier:
                self.assertEqual(
                    verify_recovered_slot_artifacts(
                        directory=root,
                        realization=realization,
                        proposal=proposal,
                    ),
                    verified,
                )
            verifier.assert_called_once()

    def test_composite_recovery_authenticates_every_branch_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-composite-recovery-") as name:
            root = Path(name)
            proposal, realization, _ = self._composite_recovery_fixture(root)
            verified = {
                "objective": realization["objective"],
                "valid_type_count": 8,
                "unscorable_type_count": 0,
            }
            with patch(
                "ep12_executor.control_trial.verify_development_trial_artifacts",
                return_value=verified,
            ) as verifier:
                self.assertEqual(
                    verify_recovered_slot_artifacts(
                        directory=root,
                        realization=realization,
                        proposal=proposal,
                    ),
                    verified,
                )
            verifier.assert_called_once()

    def test_composite_recovery_rejects_tampered_nonrepresentative_branch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-composite-tamper-") as name:
            root = Path(name)
            proposal, realization, artifact = self._composite_recovery_fixture(root)
            artifact.write_text('{"status":"tampered"}\n', encoding="utf-8")
            with (
                patch(
                    "ep12_executor.control_trial.verify_development_trial_artifacts"
                ) as verifier,
                self.assertRaisesRegex(Exception, "branch artifact digest mismatch"),
            ):
                verify_recovered_slot_artifacts(
                    directory=root,
                    realization=realization,
                    proposal=proposal,
                )
            verifier.assert_not_called()

    def test_composite_recovery_rejects_partial_nonrepresentative_branch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-composite-partial-") as name:
            root = Path(name)
            proposal, realization, artifact = self._composite_recovery_fixture(root)
            artifact.unlink()
            with (
                patch(
                    "ep12_executor.control_trial.verify_development_trial_artifacts"
                ) as verifier,
                self.assertRaisesRegex(Exception, "branch artifact.*missing or unsafe"),
            ):
                verify_recovered_slot_artifacts(
                    directory=root,
                    realization=realization,
                    proposal=proposal,
                )
            verifier.assert_not_called()

    def test_composite_recovery_cannot_bypass_branch_checks_by_dropping_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-composite-receipt-") as name:
            root = Path(name)
            proposal, realization, _ = self._composite_recovery_fixture(root)
            control = dict(realization["control_result"])
            control.pop("receipt_sha256")
            realization["control_result"] = control
            (root / "control_result.json").write_text(
                json.dumps(control), encoding="utf-8"
            )
            with (
                patch(
                    "ep12_executor.control_trial.verify_development_trial_artifacts"
                ) as verifier,
                self.assertRaisesRegex(Exception, "receipt presence is inconsistent"),
            ):
                verify_recovered_slot_artifacts(
                    directory=root,
                    realization=realization,
                    proposal=proposal,
                )
            verifier.assert_not_called()

    def test_recovery_rejects_missing_or_inconsistent_scientific_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-tail-recovery-bad-") as name:
            root = Path(name)
            proposal, realization = self._recovery_fixture(root)
            (root / "result" / "slot001" / "result_table.csv").unlink()
            with self.assertRaisesRegex(Exception, "result-table evidence is missing"):
                verify_recovered_slot_artifacts(
                    directory=root,
                    realization=realization,
                    proposal=proposal,
                )

        with tempfile.TemporaryDirectory(prefix="ep12-tail-recovery-objective-") as name:
            root = Path(name)
            proposal, realization = self._recovery_fixture(root)
            verified = {
                "objective": {"bilateral_floor": 0.2},
                "valid_type_count": 8,
                "unscorable_type_count": 0,
            }
            with (
                patch(
                    "ep12_executor.control_trial.verify_development_trial_artifacts",
                    return_value=verified,
                ),
                self.assertRaisesRegex(Exception, "differs from verified"),
            ):
                verify_recovered_slot_artifacts(
                    directory=root,
                    realization=realization,
                    proposal=proposal,
                )
if __name__ == "__main__":
    unittest.main()
