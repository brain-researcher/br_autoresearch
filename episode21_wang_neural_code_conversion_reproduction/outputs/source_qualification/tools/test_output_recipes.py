#!/usr/bin/env python3
"""Focused tests for the stable source-qualification output recipes."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
EPISODE = TOOLS.parents[2]


def load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QUALIFY = load("qualify_output")
RENDER = load("render_branch")


def qualification_fixture() -> dict:
    return {
        "attempt": 1,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "process_id": 123,
        "empirical_or_ood_payload_accessed": False,
        "protocol_lock_present": False,
        "source_identity": {"state": "verified"},
        "source_pipeline_assets": {"verified_public_model_asset_count": 4},
        "structural_qualification_matrix": {
            "participant_ids": {"planning_count": 5},
            "directed_laion_pair_table": {"planning_count": 20},
            "prior_exposure": {"classification": "retrospective"},
            "declared_ep21_input_boundary": {
                "physical_exclusion": {"ep21_input_tree_contains_ood_payload": False}
            },
            "regular_feature_handoff": {"state": "absent"},
            "regular_scoring_handoff": {"state": "absent"},
        },
    }


class OutputRecipeTests(unittest.TestCase):
    def test_main_recipe_requires_ledger_supplied_attempt(self) -> None:
        wrapper = (TOOLS / "qualify_output.py").read_text(encoding="utf-8")
        astra = (EPISODE / "outputs/astra/v0.0.14/astra.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'add_argument("--attempt", type=int, required=True)', wrapper
        )
        self.assertIn('--attempt "$EP21_LEDGER_ATTEMPT"', astra)
        for argument, placeholder in [
            ("--goal", "{inputs.goal}"),
            ("--datasets", "{inputs.datasets}"),
            ("--protocol-table", "{inputs.protocol_table}"),
            ("--input-contract", "{inputs.input_contract}"),
        ]:
            self.assertIn(f'add_argument("{argument}", type=Path, required=True)', wrapper)
            self.assertIn(f"{argument} {placeholder}", astra)

    def test_laion_renderer_is_fail_closed(self) -> None:
        report = RENDER.render_branch("laion", qualification_fixture())
        self.assertEqual(report["status"], "blocked_not_runnable")
        self.assertEqual(report["planning_directed_pair_count"], 20)
        self.assertTrue(report["absent"]["protocol_lock"])
        self.assertFalse(report["empirical_data_accessed"])

    def test_laion_renderer_narrows_verified_public_model_unknowns(self) -> None:
        fixture = qualification_fixture()
        fixture["source_pipeline_assets"].update(
            {
                "payload_verified_public_model_asset_count": 2,
                "public_model_payload_audit": {
                    "schema_version": "ep21.public_model_payload_audit.v2"
                },
            }
        )
        unresolved = RENDER.render_branch("laion", fixture)[
            "unknown_or_unresolved"
        ]
        self.assertNotIn("VGG19_archive_to_checkpoint_binding", unresolved)
        self.assertIn(
            "VGG19_evaluator_lab_copy_and_strict_runtime_load", unresolved
        )
        self.assertIn("main_bdpy_revision", unresolved)
        self.assertNotIn("main_bdpy_historical_installed_revision", unresolved)

    def test_external_renderer_does_not_block_laion(self) -> None:
        report = RENDER.render_branch("external", qualification_fixture())
        self.assertEqual(report["status"], "blocked_unprovisioned_optional_branch")
        self.assertFalse(report["within_laion_blocked_by_external_absence"])

    def test_renderer_rejects_open_boundary(self) -> None:
        fixture = qualification_fixture()
        fixture["empirical_or_ood_payload_accessed"] = True
        with self.assertRaises(ValueError):
            RENDER.render_branch("laion", fixture)

    def test_renderer_derives_protocol_lock_absence(self) -> None:
        fixture = qualification_fixture()
        fixture["protocol_lock_present"] = True
        report = RENDER.render_branch("laion", fixture)
        self.assertFalse(report["absent"]["protocol_lock"])

        del fixture["protocol_lock_present"]
        report = RENDER.render_branch("laion", fixture)
        self.assertFalse(report["absent"]["protocol_lock"])

    def test_publish_is_atomic_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            source = directory / "source.json"
            destination = directory / "nested" / "destination.json"
            source.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
            QUALIFY.publish(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_archive_path_must_be_safe_and_relative(self) -> None:
        self.assertEqual(
            str(QUALIFY.safe_relative_archive("KamitaniLab/archive.zip")),
            "KamitaniLab/archive.zip",
        )
        for unsafe in ["/absolute/archive.zip", "../escape.zip", "a/../../escape.zip"]:
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                QUALIFY.safe_relative_archive(unsafe)

    def test_recipe_report_rewrites_transient_attempt_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            source = directory / "source.json"
            destination = directory / "destination.json"
            source.write_text(
                json.dumps(
                    {
                        "evidence_inputs": {
                            "initial_qualification": "outputs/source_qualification/attempt-002/qualification_report.json"
                        }
                    }
                ),
                encoding="utf-8",
            )
            QUALIFY.publish_recipe_report(
                source,
                destination,
                {
                    "goal": {"path": "GOAL.md", "sha256": "1" * 64},
                    "datasets": {"path": "DATASETS.md", "sha256": "2" * 64},
                    "protocol_table": {
                        "path": "PROTOCOL_TABLE.yaml",
                        "sha256": "3" * 64,
                    },
                    "input_contract": {
                        "path": "inputs/README.md",
                        "sha256": "4" * 64,
                    },
                    "conversion_structural_manifest": {
                        "path": "inputs/structure.json",
                        "sha256": "5" * 64,
                    },
                    "upstream_pipeline_asset_inventory": {
                        "path": "inputs/assets.json",
                        "sha256": "6" * 64,
                    },
                },
            )
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertNotIn("initial_qualification", report["evidence_inputs"])
            self.assertEqual(
                report["evidence_inputs"]["public_source_identity_and_inventory"],
                "recipe_intermediate:run_source_qualification.py",
            )
            self.assertNotIn(
                "isolated_scratch_intermediates_removed", report["recipe_replay"]
            )
            self.assertEqual(
                report["recipe_replay"]["ephemeral_recipe_workspace_lifecycle"],
                "cleanup_pending_at_publication_and_automatic_on_normal_context_exit",
            )
            self.assertTrue(report["recipe_replay"]["persistent_public_model_cache_retained"])
            self.assertIn(
                "audit_public_model_payloads.py",
                report["recipe_replay"]["component_tools"],
            )
            self.assertEqual(
                report["evidence_inputs"]["public_model_payload_audit"],
                "recipe_intermediate:audit_public_model_payloads.py",
            )
            self.assertEqual(
                report["evidence_inputs"]["goal"]["sha256"], "1" * 64
            )
            self.assertEqual(
                set(report["recipe_replay"]["declared_contract_inputs_identity_checked"]),
                {"goal", "datasets", "protocol_table", "input_contract"},
            )

    def test_each_declared_contract_changes_recorded_identity(self) -> None:
        roles = {
            "goal": "scientific_question_evidence_roles_and_claim_limits",
            "datasets": "dataset_and_branch_contract",
            "protocol_table": "outcome_independent_protocol_fields",
            "input_contract": "stable_interfaces_and_access_restrictions",
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "declared.txt"
            for input_id, role in roles.items():
                with self.subTest(input_id=input_id):
                    path.write_text(f"{input_id}: first\n", encoding="utf-8")
                    _, first = QUALIFY.read_declared_input(path, role)
                    path.write_text(f"{input_id}: second\n", encoding="utf-8")
                    _, second = QUALIFY.read_declared_input(path, role)
                    self.assertNotEqual(first["sha256"], second["sha256"])
                    self.assertEqual(first["role"], role)

    def test_declared_input_rejects_symlink_and_snapshot_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            target = directory / "target.txt"
            target.write_text("payload\n", encoding="utf-8")
            link = directory / "link.txt"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                QUALIFY.read_declared_input(link, "test")
            snapshot = directory / "private" / "snapshot.json"
            QUALIFY.write_private_snapshot(snapshot, b"{}\n")
            self.assertEqual(snapshot.read_bytes(), b"{}\n")
            self.assertEqual(snapshot.stat().st_mode & 0o777, 0o400)


if __name__ == "__main__":
    unittest.main()
