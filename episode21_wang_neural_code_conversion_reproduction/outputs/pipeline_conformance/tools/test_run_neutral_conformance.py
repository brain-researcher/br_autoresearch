#!/usr/bin/env python3
"""Focused tests for the protocol-neutral EP21 conformance guards."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("run_neutral_conformance.py")
SPEC = importlib.util.spec_from_file_location("run_neutral_conformance", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def blocked_qualification() -> dict:
    return {
        "schema_version": "ep21.source_qualification_report.v2",
        "attempt": 17,
        "qualification_outcome": "completed_with_empirical_and_scientific_decision_blockers",
        "empirical_or_ood_payload_accessed": False,
        "protocol_lock_present": False,
        "laion_readiness": "blocked",
        "external_readiness": "blocked_unprovisioned_optional_branch",
        "direct_interface_state": {
            "laion_regular_feature_handoff": "absent",
            "laion_regular_scoring_handoff": "absent",
            "external_site_feature_handoff": "absent",
            "external_site_scoring_handoff": "absent",
        },
        "paper_code_discrepancy_register": {
            key: {"state": "recorded_unresolved_for_pipeline_conformance"}
            for key in (
                "activation",
                "alexnet_backend_and_input_size",
                "content_loss_weighting",
                "reconstruction_layers",
            )
        },
        "source_pipeline_assets": {
            "unmanifested_or_unpinned_gaps": {
                "alexnet_identification_checkpoint": {
                    "state": "required_unmanifested_public_caffenet_not_substitutable"
                },
                "main_bdpy_dependency": {"state": "declared_but_unpinned"},
                "vgg19_identification_checkpoint_binding": {
                    "state": "public_reconstruction_member_authenticated_evaluator_lab_copy_unbound"
                },
            },
            "public_model_payload_audit": {
                "schema_version": "ep21.public_model_payload_audit.v2",
                "attempt": 17,
                "empirical_or_decoded_payload_accessed": False,
                "ood_or_ep04_payload_accessed": False,
                "checkpoint_deserialization_performed": False,
            },
        },
        "ood_disposition": "not_evaluable_under_current_contract",
        "runtime_and_reconstruction": {"execution_environment_ready": False},
    }


class NeutralConformanceTests(unittest.TestCase):
    def test_archive_identity_is_independently_pinned(self) -> None:
        qualification = {
            "source_identity": {
                "state": "verified",
                "archive_md5": MODULE.PINNED_ARCHIVE_MD5,
                "archive_size_bytes": MODULE.PINNED_ARCHIVE_SIZE_BYTES,
            }
        }
        MODULE.validate_pinned_archive_identity(
            qualification,
            MODULE.PINNED_ARCHIVE_MD5,
            MODULE.PINNED_ARCHIVE_SIZE_BYTES,
        )
        for field, bad_value in (
            ("archive_md5", "0" * 32),
            ("archive_size_bytes", MODULE.PINNED_ARCHIVE_SIZE_BYTES + 1),
        ):
            mutated = {
                "source_identity": dict(qualification["source_identity"])
            }
            mutated["source_identity"][field] = bad_value
            with self.subTest(field=field), self.assertRaises(ValueError):
                MODULE.validate_pinned_archive_identity(
                    mutated,
                    MODULE.PINNED_ARCHIVE_MD5,
                    MODULE.PINNED_ARCHIVE_SIZE_BYTES,
                )
        with self.assertRaises(ValueError):
            MODULE.validate_pinned_archive_identity(
                qualification,
                "0" * 32,
                MODULE.PINNED_ARCHIVE_SIZE_BYTES,
            )

    def test_dynamic_shapes_use_actual_row_count(self) -> None:
        self.assertEqual(
            MODULE.allocate_exact_row_shapes(4712, {"conv": (3, 4), "fc": (8,)}),
            {"conv": [4712, 3, 4], "fc": [4712, 8]},
        )
        with self.assertRaises(ValueError):
            MODULE.allocate_exact_row_shapes(0, {"fc": (8,)})

    def test_qualification_binding_requires_exact_blocked_attempt(self) -> None:
        qualification = blocked_qualification()
        MODULE.validate_qualification_binding(qualification, 17)
        mutations = [
            ("schema_version", "wrong"),
            ("attempt", 16),
            ("qualification_outcome", "ready"),
            ("empirical_or_ood_payload_accessed", True),
            ("protocol_lock_present", True),
        ]
        for field, value in mutations:
            mutated = dict(qualification)
            mutated[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                MODULE.validate_qualification_binding(mutated, 17)
        nested_mutations = []
        mutated = copy.deepcopy(qualification)
        mutated["direct_interface_state"]["laion_regular_feature_handoff"] = "present"
        nested_mutations.append(mutated)
        mutated = copy.deepcopy(qualification)
        mutated["paper_code_discrepancy_register"]["activation"]["state"] = "resolved"
        nested_mutations.append(mutated)
        mutated = copy.deepcopy(qualification)
        mutated["source_pipeline_assets"]["unmanifested_or_unpinned_gaps"][
            "main_bdpy_dependency"
        ]["state"] = "resolved"
        nested_mutations.append(mutated)
        mutated = copy.deepcopy(qualification)
        mutated["source_pipeline_assets"]["public_model_payload_audit"][
            "checkpoint_deserialization_performed"
        ] = True
        nested_mutations.append(mutated)
        mutated = copy.deepcopy(qualification)
        mutated["runtime_and_reconstruction"]["execution_environment_ready"] = True
        nested_mutations.append(mutated)
        for index, mutated in enumerate(nested_mutations):
            with self.subTest(nested=index), self.assertRaises(ValueError):
                MODULE.validate_qualification_binding(mutated, 17)

    def test_qualification_artifact_binding_requires_prepared_path_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            episode = Path(name)
            report = (
                episode
                / "outputs/source_qualification/attempt-017/qualification_report.json"
            )
            report.parent.mkdir(parents=True)
            report.write_bytes(b'{"attempt": 17}\n')
            _, identity = MODULE.read_stable_bytes(
                report, label="qualification report", maximum_bytes=1024
            )
            bound = MODULE.validate_qualification_artifact_binding(
                report, episode, identity, 17, identity["sha256"]
            )
            self.assertEqual(bound, report.resolve())
            with self.assertRaises(ValueError):
                MODULE.validate_qualification_artifact_binding(
                    report, episode, identity, 17, "0" * 64
                )
            alternate = episode / "alternate.json"
            alternate.write_bytes(report.read_bytes())
            with self.assertRaises(ValueError):
                MODULE.validate_qualification_artifact_binding(
                    alternate, episode, identity, 17, identity["sha256"]
                )

    def test_stable_input_reader_rejects_symlink_and_records_hash(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            target = directory / "report.json"
            target.write_bytes(b'{"attempt": 17}\n')
            payload, identity = MODULE.read_stable_bytes(
                target, label="report", maximum_bytes=1024
            )
            self.assertEqual(payload, target.read_bytes())
            self.assertEqual(identity["sha256"], hashlib.sha256(payload).hexdigest())
            link = directory / "link.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                MODULE.read_stable_bytes(link, label="report", maximum_bytes=1024)

    def test_archive_hash_and_member_reads_share_authenticated_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            archive_path = directory / "source.zip"
            replacement = directory / "replacement.zip"

            def write_archive(path: Path, marker: str) -> None:
                with zipfile.ZipFile(path, "w") as archive:
                    for key, suffix in MODULE.SOURCE_MEMBERS.items():
                        archive.writestr(f"release/{suffix}", f"{marker}-{key}\n")

            write_archive(archive_path, "A")
            write_archive(replacement, "B")
            expected_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
            expected_size = archive_path.stat().st_size
            original_stream = MODULE.source_archive_verifier._stream_md5

            def stream_then_replace(handle):
                result = original_stream(handle)
                os.replace(replacement, archive_path)
                return result

            members = None
            try:
                with mock.patch.object(
                    MODULE.source_archive_verifier,
                    "_stream_md5",
                    side_effect=stream_then_replace,
                ), MODULE.source_archive_verifier.verified_zip(
                    archive_path, expected_md5, expected_size
                ) as verified:
                    members = MODULE.read_source_members(verified.archive)
            except ValueError:
                pass
            if members is not None:
                self.assertTrue(
                    all(item["text"].startswith("A-") for item in members.values())
                )
                self.assertFalse(
                    any(item["text"].startswith("B-") for item in members.values())
                )

    def test_runner_requires_expected_qualification_attempt(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'add_argument("--expected-qualification-attempt", type=int, required=True)',
            source,
        )
        self.assertIn(
            'add_argument("--expected-qualification-sha256", required=True)', source
        )

    def test_content_guard_rejects_default_mismatch_rows_and_duplicates(self) -> None:
        ids = [f"id-{index}" for index in range(4712)]
        accepted = MODULE.validate_content_rows(
            ids, 4712, {"conv": 4712, "fc": 4712}, 4712, 4712
        )
        self.assertEqual(accepted["allocation_row_count"], 4712)
        with self.assertRaises(ValueError):
            MODULE.validate_content_rows(ids, 4712, {"fc": 4712}, 6000, 4712)
        with self.assertRaises(ValueError):
            MODULE.validate_content_rows(ids, 4712, {"fc": 4711}, 4712, 4712)
        with self.assertRaises(ValueError):
            MODULE.validate_content_rows(
                [*ids[:-1], ids[0]], 4712, {"fc": 4712}, 4712, 4712
            )

    def test_tau_guard_accepts_exact_disjoint_roles(self) -> None:
        train = [f"train-{index}" for index in range(897)]
        test = [f"test-{index}" for index in range(224)]
        result = MODULE.validate_tau_inputs(
            list(reversed(train)), train, train, test, 897, 224
        )
        self.assertEqual(result["source_fit_count"], 897)
        self.assertEqual(result["train_test_overlap_count"], 0)
        self.assertFalse(result["source_target_input_order_equal"])
        self.assertTrue(result["post_sort_pairwise_label_alignment"])
        self.assertEqual(result["post_sort_pairwise_label_count"], 897)

    def test_tau_guard_rejects_overlap_leak_and_mismatch(self) -> None:
        train = [f"train-{index}" for index in range(897)]
        test = [f"test-{index}" for index in range(224)]
        with self.assertRaises(ValueError):
            MODULE.validate_tau_inputs(
                train, train, train, [train[0], *test[1:]], 897, 224
            )
        with self.assertRaises(ValueError):
            MODULE.validate_tau_inputs(
                [*train[:-1], test[0]], train, train, test, 897, 224
            )
        with self.assertRaises(ValueError):
            MODULE.validate_tau_inputs(train[:-1], train[:-1], train[:-1], test, 897, 224)

    def test_worker_binding_is_explicit_and_fail_closed(self) -> None:
        self.assertEqual(
            MODULE.resolve_worker_count(1, {})["binding_source"], "explicit_cli"
        )
        self.assertEqual(
            MODULE.resolve_worker_count(None, {"SLURM_CPUS_PER_TASK": "4"})[
                "max_workers"
            ],
            4,
        )
        with self.assertRaises(ValueError):
            MODULE.resolve_worker_count(None, {})
        with self.assertRaises(ValueError):
            MODULE.resolve_worker_count(3, {"SLURM_CPUS_PER_TASK": "2"})

    def test_ast_helpers_detect_unused_argument_and_executor(self) -> None:
        source = """
def train_NCconverter(x, num_sample):
    return x

def prepare():
    with ProcessPoolExecutor() as executor:
        return executor
"""
        self.assertFalse(
            MODULE.function_loads_name(source, "train_NCconverter", "num_sample")
        )
        self.assertEqual(
            MODULE.process_pool_calls(source),
            [{"line": 6, "positional_argument_count": 0, "keyword_names": []}],
        )

    def test_argparse_default_and_parameter_table(self) -> None:
        source = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=6000)
"""
        self.assertEqual(MODULE.argparse_default(source, "--batchSize"), 6000)
        table = "Source,Target,Number of samples,Alpha,ROI,Method\na,b,6000,820,VC,wholeVC\n"
        result = MODULE.parse_brain_parameter_table(table)
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["alpha_min"], 820)


if __name__ == "__main__":
    unittest.main()
