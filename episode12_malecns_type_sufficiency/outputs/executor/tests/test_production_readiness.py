from __future__ import annotations

import unittest

from ep12_executor.production_readiness import (
    claim_is_permitted,
    evaluate_production_readiness,
    verify_contract_roles,
)
from ep12_executor.runtime import EPISODE_ROOT


class ProductionReadinessTests(unittest.TestCase):
    def test_contract_roles_preserve_development_and_final_boundaries(self) -> None:
        executor = EPISODE_ROOT / "outputs" / "executor"
        evidence = verify_contract_roles(
            prefix_contract_path=executor / "SCIENTIFIC_CONTRACT.yaml",
            active_contract_path=executor / "SCIENTIFIC_CONTRACT_POST36.yaml",
        )
        self.assertFalse(evidence["final_connectivity_accessed"])

    def test_current_readiness_stays_revise_for_scientific_qualification(self) -> None:
        outputs = EPISODE_ROOT / "outputs"
        executor = outputs / "executor"
        run_root = outputs / "development_run002"
        report = evaluate_production_readiness(
            qualification_dir=outputs / "prelaunch" / "scientific_qualification_dev007",
            observed_run_root=run_root,
            observed_finalization_dir=run_root / "observed_finalization",
            falsifier_plan_path=executor / "OBSERVED_POST36_TAIL_PLAN.json",
        )
        checks = {row["check_id"]: row for row in report["checks"]}
        self.assertEqual(report["status"], "revise")
        self.assertFalse(checks["full_synthetic_qualification"]["passed"])
        self.assertTrue(checks["observed_36_trial_provisional_chain"]["passed"])
        self.assertTrue(checks["post36_falsifier_plan"]["passed"])
        self.assertFalse(claim_is_permitted(report, "observed_tail"))
        self.assertFalse(claim_is_permitted(report, "final_connectivity"))


if __name__ == "__main__":
    unittest.main()
