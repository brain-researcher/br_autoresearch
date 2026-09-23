#!/usr/bin/env python3
"""Mechanical tests for the deterministic Dudman role handoffs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


EPISODE = Path(__file__).resolve().parents[2]
PROJECT = EPISODE.parent
CODE = EPISODE / "outputs" / "code" / "materialize_roles.py"
PUBLIC = Path(
    os.environ.get("EP02_ROLE_PUBLIC_DIR", EPISODE / "outputs" / "role_materialization")
)
DEVELOPMENT = Path(
    os.environ.get(
        "EP02_ROLE_DEVELOPMENT_DIR", EPISODE / "inputs" / "materialized" / "development_full"
    )
)
STRUCTURAL = Path(
    os.environ.get(
        "EP02_ROLE_STRUCTURAL_DIR", EPISODE / "inputs" / "materialized" / "audit_structural"
    )
)
VAULT = Path(
    os.environ.get("EP02_ROLE_VAULT_DIR", PROJECT / ".evaluator_vault" / EPISODE.name)
)

SPEC = importlib.util.spec_from_file_location("materialize_roles", CODE)
assert SPEC and SPEC.loader
MATERIALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATERIALIZER)


class DeterministicEncodingTests(unittest.TestCase):
    def test_observed_on_is_required(self) -> None:
        argv = [
            "materialize_roles.py",
            "--mat", "source.mat",
            "--cohort-map", "cohort.json",
            "--development-dir", "development",
            "--audit-structural-dir", "structural",
            "--vault-root", "vault",
            "--public-output-dir", "public",
            "--expected-bytes", "1",
            "--expected-md5", "0" * 32,
            "--expected-sha256", "0" * 64,
            "--expected-cohort-map-sha256", "0" * 64,
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            MATERIALIZER.parse_args()

    def test_npy_bytes_are_reproducible_pickle_free_and_bit_exact(self) -> None:
        values = np.asarray([[0.0, -0.0, np.nan]], dtype=np.float32)
        first = MATERIALIZER.npy_payload(values)
        second = MATERIALIZER.npy_payload(values)
        self.assertEqual(first, second)
        MATERIALIZER.verify_npy_payload(first, values, "synthetic")

    def test_object_arrays_fail_closed(self) -> None:
        with self.assertRaises(TypeError):
            MATERIALIZER.npy_payload(np.asarray([[object()]], dtype=object))

    def test_independent_synthetic_builds_have_identical_roots(self) -> None:
        dtype = np.dtype([(field, object) for field in MATERIALIZER.EXPECTED_FIELDS])
        sessions = np.empty((1, 24), dtype=dtype)
        lookup = {}
        development = {1, 2, 4, 6, 9, 11, 15, 16, 19}
        boundary = {21, 22, 23, 24}
        for source_index in range(1, 25):
            if source_index in development:
                role = "development_full"
                cohort = "development_control"
            elif source_index in boundary:
                role = "post_primary_boundary_evaluator_only"
                cohort = "boundary_supraphysiological_stim_lick_plus"
            else:
                role = "primary_audit_evaluator_only"
                cohort = "audit_synthetic"
            lookup[source_index] = {"role": role, "cohort_id": cohort}
            for offset, field in enumerate(MATERIALIZER.EXPECTED_FIELDS):
                value = np.asarray(
                    [[source_index, offset, -0.0]],
                    dtype=np.float32 if offset % 2 else np.int32,
                )
                sessions[0, source_index - 1][field] = value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MATERIALIZER.write_role_pack(
                root / "first",
                sessions,
                lookup,
                "development_full",
                "2026-09-22",
                "a" * 64,
                "b" * 64,
                "c" * 64,
            )
            second = MATERIALIZER.write_role_pack(
                root / "second",
                sessions,
                lookup,
                "development_full",
                "2026-09-22",
                "a" * 64,
                "b" * 64,
                "c" * 64,
            )
            self.assertEqual(first, second)
            self.assertEqual(
                (root / "first" / "MANIFEST.json").read_bytes(),
                (root / "second" / "MANIFEST.json").read_bytes(),
            )
            for tree in (root / "first", root / "second"):
                for path in tree.rglob("*"):
                    path.chmod(0o700 if path.is_dir() else 0o600)
                tree.chmod(0o700)


@unittest.skipUnless((PUBLIC / "PUBLIC_ROLE_MANIFEST.json").exists(), "materializer not run")
class MaterializedPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public = json.loads((PUBLIC / "PUBLIC_ROLE_MANIFEST.json").read_text())

    def test_roles_are_disjoint_and_complete(self) -> None:
        commitments = self.public["role_commitments"]
        self.assertEqual(commitments["development_full"]["record_count"], 9)
        self.assertEqual(commitments["primary_audit_evaluator_only"]["record_count"], 11)
        self.assertEqual(
            commitments["post_primary_boundary_evaluator_only"]["record_count"], 4
        )
        self.assertEqual(sum(item["record_count"] for item in commitments.values()), 24)
        self.assertEqual(
            commitments["post_primary_boundary_evaluator_only"]["visibility"],
            "trusted_evaluator_internal_stage_after_primary_subpacket_commit_"
            "before_feedback_in_same_logical_opening",
        )

    @unittest.skipUnless(
        (DEVELOPMENT / "MANIFEST.json").exists(),
        "local ignored development pack is unavailable",
    )
    def test_development_arrays_match_manifest(self) -> None:
        development_manifest = json.loads((DEVELOPMENT / "MANIFEST.json").read_text())
        entries = development_manifest["record_entries"]
        self.assertEqual(len(entries), 9)
        self.assertEqual(len(list(DEVELOPMENT.glob("source_record_*"))), 9)
        for entry in entries:
            self.assertEqual(len(entry["fields"]), 20)
            for field in MATERIALIZER.EXPECTED_FIELDS:
                metadata = entry["fields"][field]
                path = DEVELOPMENT / metadata["relative_path"]
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(observed, metadata["npy_sha256"])

    def test_public_manifest_does_not_disclose_individual_audit_archives(self) -> None:
        text = (PUBLIC / "PUBLIC_ROLE_MANIFEST.json").read_text()
        audit = self.public["audit_structural_disclosure"]
        self.assertFalse(audit["individual_file_paths_or_hashes_disclosed"])
        self.assertFalse(audit["behavioral_or_neural_magnitudes_disclosed"])
        for source_index in (3, 5, 7, 8, 10, 12, 13, 14, 17, 18, 20, 21, 22, 23, 24):
            self.assertNotIn(f"source_record_{source_index:02d}", text)
        self.assertNotIn("/oak/", text)
        self.assertNotIn(".evaluator_vault", text)

    def test_materializer_hash_replays(self) -> None:
        observed_code = hashlib.sha256(CODE.read_bytes()).hexdigest()
        self.assertEqual(self.public["materializer_sha256"], observed_code)

    @unittest.skipUnless(
        (DEVELOPMENT / "MANIFEST.json").exists(),
        "local ignored development pack is unavailable",
    )
    def test_development_pack_root_replays(self) -> None:
        development_manifest = json.loads((DEVELOPMENT / "MANIFEST.json").read_text())
        observed_code = hashlib.sha256(CODE.read_bytes()).hexdigest()
        self.assertEqual(development_manifest["materializer_sha256"], observed_code)
        core = dict(development_manifest)
        expected_root = core.pop("pack_root_sha256")
        replayed = hashlib.sha256(MATERIALIZER.canonical_json_bytes(core)).hexdigest()
        self.assertEqual(replayed, expected_root)
        self.assertEqual(
            self.public["role_commitments"]["development_full"]["pack_root_sha256"],
            expected_root,
        )

    @unittest.skipUnless(
        (STRUCTURAL / "PUBLIC_ROLE_MANIFEST.json").exists(),
        "local ignored structural handoff is unavailable",
    )
    def test_public_copy_is_exact(self) -> None:
        self.assertEqual(
            (PUBLIC / "PUBLIC_ROLE_MANIFEST.json").read_bytes(),
            (STRUCTURAL / "PUBLIC_ROLE_MANIFEST.json").read_bytes(),
        )

    @unittest.skipUnless(
        VAULT.exists()
        and (VAULT / "primary_audit_evaluator_only").exists()
        and (VAULT / "post_primary_boundary_evaluator_only").exists(),
        "local ignored evaluator staging is unavailable",
    )
    def test_audit_directories_have_no_mode_bits(self) -> None:
        self.assertEqual(stat.S_IMODE(VAULT.stat().st_mode), 0o100)
        for role in (
            "primary_audit_evaluator_only",
            "post_primary_boundary_evaluator_only",
        ):
            self.assertEqual(stat.S_IMODE((VAULT / role).stat().st_mode), 0)

    def test_firewall_limit_is_not_overclaimed(self) -> None:
        gates = self.public["gates"]
        self.assertTrue(gates["materialization_complete"])
        self.assertFalse(gates["physical_audit_firewall_complete"])
        self.assertFalse(gates["original_mixed_source_unavailable_to_candidate_uid"])
        self.assertFalse(gates["launch_authorized"])

    def test_public_sha256_manifest(self) -> None:
        for row in (PUBLIC / "SHA256SUMS").read_text().splitlines():
            expected, name = row.split("  ", 1)
            self.assertEqual(hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
