from __future__ import annotations

import unittest
from pathlib import Path


SLURM_ROOT = Path(__file__).resolve().parents[1] / "slurm"


class SlurmWrapperTests(unittest.TestCase):
    def test_live_wrappers_do_not_rebuild_retired_provenance_layers(self) -> None:
        forbidden = (
            "source_freeze",
            "source_archive",
            "attest_observed_trials",
            "scientific_transition",
            "slurm_handoff",
            "secure_publish_lock",
            "full_null_launch_permit",
        )
        for path in SLURM_ROOT.glob("*.sbatch"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{path.name} still invokes {token}")

    def test_scientific_execution_wrappers_remain(self) -> None:
        expected = {
            "check_production_readiness.sbatch",
            "qualify_methods.sbatch",
            "run_post36_tail.sbatch",
            "run_procedure_lock.sbatch",
            "run_full_search_null.sbatch",
            "run_full_search_null_aggregate.sbatch",
            "run_coverage_array.sbatch",
            "run_successor_array.sbatch",
        }
        self.assertTrue(expected.issubset({path.name for path in SLURM_ROOT.glob("*.sbatch")}))

    def test_obsolete_pre_null_shortcut_is_removed(self) -> None:
        self.assertFalse((SLURM_ROOT / "pre_null_gate.sbatch").exists())
        self.assertFalse(
            (SLURM_ROOT.parent / "ep12_executor" / "pre_null_gate.py").exists()
        )

    def test_no_final_data_wrapper_exists_before_scientific_completion(self) -> None:
        self.assertFalse((SLURM_ROOT / "run_final_evaluation.sbatch").exists())


if __name__ == "__main__":
    unittest.main()
