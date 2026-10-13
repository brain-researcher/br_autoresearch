#!/usr/bin/env python3
"""Focused resource/qualification gate regressions; no optimizer fits."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fit_v3_m1_real_v1 as runner


class CleanRevalidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refresh_root = Path(os.environ["SCRATCH"]) / "br_autoresearch" / "episode10_single_cell_coprojection" / "v3_m1_affected_numerics_refresh_v1_job_46330399"
        with (cls.refresh_root / "v3_m1_affected_numerics_refresh.json").open() as handle:
            cls.refresh = json.load(handle)
        cls.amendment = runner._read_exact_resource_record(runner.RESOURCE_AMENDMENT_PATH, runner.RESOURCE_AMENDMENT_PATH)
        cls.assessment = runner._read_exact_resource_record(runner.RESOURCE_ASSESSMENT_PATH, runner.RESOURCE_ASSESSMENT_PATH)
        cls.authorization = runner._read_exact_resource_record(runner.WALL_AUTHORIZATION_PATH, runner.WALL_AUTHORIZATION_PATH)

    def assess(self, *, refresh=None, amendment=None, assessment=None, authorization=None, allocated_cpus=3):
        return runner.assess_approved_numerical_refresh(
            self.refresh if refresh is None else refresh,
            self.amendment if amendment is None else amendment,
            self.assessment if assessment is None else assessment,
            self.authorization if authorization is None else authorization,
            self.refresh["numeric_backend"], allocated_cpus,
        )

    def test_actual_numerics_pass_under_approved_limit_preserves_original_fail(self):
        original = copy.deepcopy(self.refresh)
        for allocated in (1, 2, 3, 4):
            result = self.assess(allocated_cpus=allocated)
            self.assertEqual(result["status"], "pass_under_user_authorized_12_hour_request")
            self.assertEqual(result["original_refresh_status_preserved"], "fail")
            self.assertEqual(result["projected_production_wall_hours"], 8.163286204683827)
        self.assertEqual(self.refresh, original)
        with mock.patch.dict(os.environ, SLURM_CPUS_PER_TASK="3"):
            bound = runner.require_approved_numerical_refresh(
                self.refresh_root, runner.RESOURCE_AMENDMENT_PATH,
                runner.RESOURCE_ASSESSMENT_PATH, self.refresh["numeric_backend"],
            )
        self.assertEqual(bound["numerical_refresh_job_id"], "46330399")

    def test_missing_approval_or_changed_request_fails_closed(self):
        with self.assertRaises(runner.RealM1Error):
            self.assess(authorization=dict(self.authorization, status="pending"))
        amendment = copy.deepcopy(self.amendment)
        amendment["production_boundary"]["requested_wall_time"] = "08:00:00"
        with self.assertRaises(runner.RealM1Error):
            self.assess(amendment=amendment)

    def test_numerical_failure_or_stale_refresh_cannot_be_reinterpreted(self):
        for key, value in (("SLURM_job_id", "46330031"), ("failure_message", "line search failed")):
            with self.assertRaises(runner.RealM1Error):
                self.assess(refresh=dict(self.refresh, **{key: value}))
        failed = copy.deepcopy(self.refresh)
        failed["recovery"]["exact_population_recovery"]["status"] = "fail"
        with self.assertRaises(runner.RealM1Error):
            self.assess(refresh=failed)

    def test_existing_epoch_headroom_and_reservation_remain_required(self):
        for key, value in (("reservation_included", False), ("remaining_unreserved_wall_hours", -1),
                           ("projected_production_wall_hours", 7), ("producer_batch_MaxRSS_KiB", 16 * 1024**2)):
            assessment = copy.deepcopy(self.assessment)
            assessment["m1_clean_refit"][key] = value
            with self.assertRaises(runner.RealM1Error):
                self.assess(assessment=assessment)
        with self.assertRaises(runner.RealM1Error):
            self.assess(assessment=dict(self.assessment, accounting_epoch="new_reset"))
        with self.assertRaises(runner.RealM1Error):
            self.assess(allocated_cpus=5)

    def test_historical_M0_opening_is_exact_and_current_job_remains_closed(self):
        readiness = runner._read_readiness()
        qualification_root = self.refresh_root.parent / runner.QUALIFICATION_ROOT_BASENAME
        summary = runner.require_single_thread_qualification(qualification_root)
        self.assertEqual(summary["SLURM_job_id"], "46328672")
        for invalid in (False, "true_job_other_post_M1_mechanical_comparison_only"):
            wrong = copy.deepcopy(readiness)
            wrong["access"]["v3_m1_M0_artifact_opened"] = invalid
            with mock.patch.object(runner, "_read_readiness", return_value=wrong):
                with self.assertRaises(runner.RealM1Error):
                    runner.require_single_thread_qualification(qualification_root)
                with self.assertRaises(runner.RealM1Error):
                    runner.require_actual_k_prerequisite(Path("unused"), membership_opened_by_job_id=runner.PRIOR_MEMBERSHIP_OPENING_JOB_ID)
        identity = runner._execution_identity({"schema_version": runner.EXECUTION_SCHEMA_VERSION}, self.refresh["numeric_backend"])
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, SLURM_JOB_ID="99999999"):
            root = Path(directory)
            runner._write_checkpoint_manifest(root, execution_identity=identity, phase="prerequisites_passed_before_membership", realized_membership_opened=False, completed_warm_paths=0, completed_zero_refits=0)
            manifest = json.loads((root / runner.CHECKPOINT_MANIFEST).read_text())
        self.assertFalse(manifest["M0_prediction_or_fit_artifact_opened"])
        self.assertEqual(manifest["execution_identity"]["historical_M0_artifact_access"], runner.HISTORICAL_M0_ACCESS)


if __name__ == "__main__":
    unittest.main()
