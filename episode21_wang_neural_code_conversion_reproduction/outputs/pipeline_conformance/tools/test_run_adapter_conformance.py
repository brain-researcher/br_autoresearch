#!/usr/bin/env python3
"""Tests for the frozen synthetic adapter-conformance runner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


RUNNER_PATH = pathlib.Path(__file__).with_name("run_adapter_conformance.py")
SPEC = importlib.util.spec_from_file_location("run_adapter_conformance", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class AdapterConformanceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, _ = runner.read_stable_bytes(
            runner.ADAPTER_PATH,
            label="test adapter",
            maximum_bytes=runner.MAX_CODE_BYTES,
            expected_sha256=runner.ADAPTER_SHA256,
        )
        cls.adapter_raw = raw
        cls.adapter = runner.load_verified_module(
            raw,
            path=runner.ADAPTER_PATH,
            module_name="ep21_adapter_runner_tests_verified",
        )

    def test_build_report_is_conservative_and_complete(self):
        report = runner.build_report(6, 1, runner.ATTENDED_LOGIN_CONTEXT)
        self.assertEqual(report["schema_version"], "ep21.adapter_conformance_report.v1")
        self.assertEqual(report["attempt"], 6)
        self.assertEqual(
            report["outcome"],
            "passed_synthetic_adapter_checks_with_protocol_and_access_blockers",
        )
        self.assertGreaterEqual(report["negative_fixture_count"], 49)
        self.assertTrue(all(item["result"] == "rejected" for item in report["negative_fixture_results"]))
        self.assertTrue(all(value == "absent" for value in report["current_gate"].values()))
        self.assertTrue(all(value is False for value in report["boundary"].values()))
        self.assertTrue(all(value is False for value in report["protocol_effect"].values()))
        contexts = report["contexts_exercised"]
        self.assertTrue(contexts["synthetic_only"])
        self.assertFalse(contexts["empirical_context_accepted"])
        self.assertFalse(contexts["real_protocol_lock_present"])
        self.assertFalse(contexts["synthetic_receipts_are_authenticated_handoff_evidence"])
        self.assertTrue(
            contexts["synthetic_worker_count_matches_explicit_runner_limit"]
        )
        execution = report["execution"]
        self.assertEqual(execution["kind"], runner.ATTENDED_LOGIN_CONTEXT)
        self.assertEqual(execution["max_workers"], 1)
        self.assertFalse(execution["slurm_allocation_present"])
        self.assertTrue(execution["synchronous_foreground_required"])

    def test_fixture_coverage_has_dynamic_sizes_modes_and_alpha_scopes(self):
        report = runner.build_report(6, 1, runner.ATTENDED_LOGIN_CONTEXT)
        average = report["synthetic_fixtures"]["dynamic_rows_3_image_average"]
        paired = report["synthetic_fixtures"]["dynamic_rows_7_authenticated_pairing"]
        self.assertEqual(average["summary"]["content_row_count"], 3)
        self.assertEqual(paired["summary"]["content_row_count"], 7)
        self.assertEqual(average["summary"]["repetition_mode"], "image_average_before_fit")
        self.assertEqual(paired["summary"]["repetition_mode"], "authenticated_cross_subject_pairing")
        self.assertEqual(average["alpha_binding_kind"], "fixed_predeclared")
        self.assertEqual(paired["alpha_binding_kind"], "tau_train_only")
        self.assertTrue(average["alpha_value_omitted_from_report"])
        self.assertTrue(paired["alpha_value_omitted_from_report"])
        self.assertNotEqual(average["summary"]["content_row_count"], 4712)
        self.assertNotEqual(paired["summary"]["content_row_count"], 4712)

    def test_negative_fixture_labels_are_unique_and_cover_prior_bypasses(self):
        results = runner.negative_fixtures(self.adapter, 1)
        labels = [item["label"] for item in results]
        self.assertEqual(len(labels), len(set(labels)))
        required = {
            "unlisted_pair_membership",
            "same_count_substituted_feature_row",
            "missing_authenticated_repeat",
            "local_repetition_cannot_replace_alignment_key",
            "alpha_declares_tau_test_or_ood_use",
            "alpha_dotted_policy_names_tau_test",
            "alpha_value_differs_from_context_declaration",
            "alpha_numeric_representation_differs_from_context_declaration",
            "resealed_self_consistent_content_plan",
            "content_plan_negative_tail_dimension",
            "content_plan_bool_feature_dimension",
            "resealed_arbitrary_response_rows_seal",
            "resealed_arbitrary_feature_rows_seal",
            "resealed_brain_group_reorder",
            "brain_plan_bool_index",
            "brain_plan_float_index",
            "brain_plan_nonlist_group_container",
            "brain_plan_falsey_nonlist_aligned_container",
            "resealed_arbitrary_source_rows_seal",
            "resealed_arbitrary_target_rows_seal",
            "resealed_self_consistent_summary",
            "resealed_summary_plan_authority_seal",
            "ordered_rows_zero_tail_dimension",
            "feature_rows_bool_tail_dimension",
            "duck_typed_alpha",
            "duck_typed_content_plan",
            "subclassed_content_plan",
        }
        self.assertTrue(required <= set(labels))

    def test_adapter_static_audit_rejects_import_and_io_call(self):
        with self.assertRaisesRegex(ValueError, "import boundary"):
            runner.adapter_static_audit(b"import torch\n")
        poisoned = self.adapter_raw + b"\nopen('forbidden')\n"
        with self.assertRaisesRegex(ValueError, "forbidden calls"):
            runner.adapter_static_audit(poisoned)

    def test_upstream_semantics_reject_readiness_or_boundary_promotion(self):
        neutral = json.loads(runner.NEUTRAL_REPORT.read_text(encoding="utf-8"))
        method = json.loads(runner.METHOD_REPORT.read_text(encoding="utf-8"))
        protocol = runner.PROTOCOL_TABLE.read_text(encoding="utf-8")
        runner.validate_upstream_bindings(neutral, method, protocol)
        promoted = copy.deepcopy(neutral)
        promoted["empirical_readiness"] = "ready"
        with self.assertRaisesRegex(ValueError, "attempt-5"):
            runner.validate_upstream_bindings(promoted, method, protocol)
        opened = copy.deepcopy(method)
        opened["boundary"]["model_or_data_payload_read"] = True
        with self.assertRaisesRegex(ValueError, "closed-boundary"):
            runner.validate_upstream_bindings(neutral, opened, protocol)

    def test_stable_reader_rejects_wrong_hash_symlink_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            target = directory / "target.json"
            target.write_bytes(b"{}\n")
            digest = hashlib.sha256(b"{}\n").hexdigest()
            payload, identity = runner.read_stable_bytes(
                target, label="fixture", maximum_bytes=100, expected_sha256=digest
            )
            self.assertEqual(payload, b"{}\n")
            self.assertTrue(identity["single_link"])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                runner.read_stable_bytes(
                    target, label="fixture", maximum_bytes=100, expected_sha256="0" * 64
                )
            symlink = directory / "symlink.json"
            symlink.symlink_to(target)
            with self.assertRaises(OSError):
                runner.read_stable_bytes(
                    symlink, label="symlink", maximum_bytes=100, expected_sha256=digest
                )
            hardlink = directory / "hardlink.json"
            os.link(target, hardlink)
            with self.assertRaisesRegex(ValueError, "single-link"):
                runner.read_stable_bytes(
                    target, label="hardlink", maximum_bytes=100, expected_sha256=digest
                )

    def test_final_entry_check_is_metadata_only_and_detects_any_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            candidate = directory / "candidate"
            self.assertTrue(runner.final_entry_absent(candidate))
            candidate.symlink_to(directory / "missing-target")
            self.assertFalse(runner.final_entry_absent(candidate))

    def test_wrong_attempt_worker_limit_or_context_fails(self):
        for attempt, workers, context in (
            (5, 1, runner.ATTENDED_LOGIN_CONTEXT),
            (6, 2, runner.ATTENDED_LOGIN_CONTEXT),
            (6, True, runner.ATTENDED_LOGIN_CONTEXT),
            (6, 1, "slurm"),
        ):
            with self.subTest(attempt=attempt, workers=workers, context=context):
                with self.assertRaises(ValueError):
                    runner.build_report(attempt, workers, context)

    def test_attended_context_rejects_slurm_and_uses_no_fake_cpu_variable(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("SLURM_CPUS_PER_TASK", source)
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=False):
            with self.assertRaisesRegex(ValueError, "active Slurm"):
                runner.build_report(6, 1, runner.ATTENDED_LOGIN_CONTEXT)

    def test_no_report_or_partial_is_created_by_build_report(self):
        output = runner.OUTPUT_DIR / runner.OUTPUT_NAME
        before = output.exists() or output.is_symlink()
        runner.build_report(6, 1, runner.ATTENDED_LOGIN_CONTEXT)
        after = output.exists() or output.is_symlink()
        self.assertEqual(before, after)
        self.assertFalse(after)
        if runner.OUTPUT_DIR.exists():
            partials = list(runner.OUTPUT_DIR.glob(f".{runner.OUTPUT_NAME}.*.partial"))
            self.assertEqual(partials, [])


if __name__ == "__main__":
    unittest.main()
