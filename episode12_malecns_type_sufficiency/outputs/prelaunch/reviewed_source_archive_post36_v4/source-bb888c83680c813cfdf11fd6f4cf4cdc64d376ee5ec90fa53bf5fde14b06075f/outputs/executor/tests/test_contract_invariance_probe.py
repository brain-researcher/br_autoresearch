from __future__ import annotations

import base64
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ep12_executor import contract_invariance_probe as probe


SPEC_PATH = (
    Path(probe.__file__).resolve().parents[1]
    / "CONTRACT_TRANSITION_PROBE_SPEC.json"
)


def _diagnostics(*, include_new_rank: bool, rank: int = 10) -> dict:
    cases: dict[str, object] = {}
    for case in probe.EXPECTED_CASES:
        case_id = case["case_id"]
        directions: dict[str, object] = {}
        for direction in probe.EXPECTED_DIRECTIONS:
            models: dict[str, object] = {}
            for family in probe.MODEL_FAMILIES:
                diagnostics: dict[str, object] = {
                    "representation": case["representation"],
                    "covariance_mode": case["covariance"],
                }
                if case_id != "adequate_template_count_aware":
                    diagnostics["representation_rank"] = case["rank"]
                elif include_new_rank:
                    diagnostics["representation_rank"] = rank
                models[family] = {
                    "model_id": family,
                    "diagnostics": diagnostics,
                }
            directions[direction] = {"models": models}
        cases[case_id] = {"directions": directions}
    return {"schema_version": probe.DIAGNOSTICS_SCHEMA, "cases": cases}


def _minimal_run_record(spec_sha: str) -> dict:
    case_ids = [case["case_id"] for case in probe.EXPECTED_CASES]
    input_payload = {
        "schema_version": probe.INPUT_SCHEMA,
        "cases": {case_id: {} for case_id in case_ids},
    }
    scientific_payload = {
        "schema_version": probe.SCIENTIFIC_SCHEMA,
        "cases": {case_id: {} for case_id in case_ids},
    }
    diagnostics_payload = _diagnostics(include_new_rank=False)
    identity = {
        "source_payload_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "spec_sha256": spec_sha,
    }
    record = {
        "schema_version": probe.RUN_SCHEMA,
        "phase": "legacy",
        "outcome_firewall": {
            "generated_examples_only": True,
            "real_connectivity_accessed": False,
            "connectivity_paths_accepted": False,
            "annotation_or_outcome_files_accessed": False,
        },
        "byte_identity": {"start": identity, "end": copy.deepcopy(identity)},
        "environment_payload": {"environment": "fixed"},
        "input_payload": input_payload,
        "scientific_payload": scientific_payload,
        "diagnostics_payload": diagnostics_payload,
    }
    record["environment_payload_sha256"] = probe._sha256_payload(
        record["environment_payload"]
    )
    record["input_payload_sha256"] = probe._sha256_payload(input_payload)
    record["scientific_payload_sha256"] = probe._sha256_payload(
        scientific_payload
    )
    record["diagnostics_payload_sha256"] = probe._sha256_payload(
        diagnostics_payload
    )
    record["artifact_sha256"] = probe._artifact_digest(record)
    return record


class ContractTransitionProbeSpecTests(unittest.TestCase):
    def test_spec_is_exact_and_generated_only(self) -> None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        probe._validate_spec(spec)
        self.assertEqual(tuple(spec["cases"]), probe.EXPECTED_CASES)
        self.assertEqual(spec["data_scope"]["real_connectivity_access"], "forbidden")

    def test_spec_rejects_even_one_changed_fit_choice(self) -> None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        spec["fit"]["integration_draws"] = 255
        with self.assertRaises(probe.ProbeError):
            probe._validate_spec(spec)

    def test_cli_exposes_no_connectivity_or_data_argument(self) -> None:
        parser = probe.build_parser()
        destinations: set[str] = set()
        pending = [parser]
        while pending:
            current = pending.pop()
            for action in current._actions:
                destinations.add(action.dest)
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict):
                    pending.extend(choices.values())
        joined = " ".join(sorted(destinations)).lower()
        self.assertNotIn("connectivity", joined)
        self.assertNotIn("annotation", joined)
        self.assertNotIn("weights", joined)


class ContractTransitionProbePathTests(unittest.TestCase):
    def test_rejects_dataset_shaped_paths_before_reading(self) -> None:
        for path in (
            "/tmp/connectome/value.json",
            "/tmp/connectivity/value.json",
            f"{probe.EPISODE_ROOT}/inputs/value.json",
        ):
            with self.subTest(path=path), self.assertRaises(probe.ProbeError):
                probe._validate_nondataset_path(
                    path, label="test", must_exist=False, output=True
                )

    def test_rejects_symlinked_source_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaises(probe.ProbeError):
                probe._validate_nondataset_path(
                    linked, label="source", must_exist=True, expect_directory=True
                )

    def test_source_identity_detects_a_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executor = Path(temporary) / "executor"
            package = executor / "ep12_executor"
            package.mkdir(parents=True)
            for relative in probe.EXPECTED_MODULES:
                path = executor / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            (executor / "SCIENTIFIC_CONTRACT.yaml").write_text(
                "contract: legacy\n", encoding="utf-8"
            )
            before = probe._source_identity(
                executor, "SCIENTIFIC_CONTRACT.yaml", SPEC_PATH
            )
            (package / "scientific_models.py").write_text(
                "# changed\n", encoding="utf-8"
            )
            after = probe._source_identity(
                executor, "SCIENTIFIC_CONTRACT.yaml", SPEC_PATH
            )
            self.assertNotEqual(before, after)

    def test_scientific_loader_does_not_execute_package_initializer(self) -> None:
        namespace = "_ep12_contract_probe_source"
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            executor = Path(temporary) / "executor"
            package = executor / "ep12_executor"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                "raise RuntimeError('package initializer must not execute')\n",
                encoding="utf-8",
            )
            (package / "scientific_models.py").write_text(
                "SENTINEL = 'models'\n", encoding="utf-8"
            )
            (package / "scientific_synthetic.py").write_text(
                "from .scientific_models import SENTINEL\n",
                encoding="utf-8",
            )
            try:
                models, synthetic, *_ = probe._import_scientific_stack(executor)
                self.assertEqual(models.SENTINEL, "models")
                self.assertEqual(synthetic.SENTINEL, "models")
            finally:
                for name in tuple(sys.modules):
                    if name == namespace or name.startswith(namespace + "."):
                        del sys.modules[name]


class ContractTransitionProbeComparisonTests(unittest.TestCase):
    def _records(self) -> tuple[dict, dict]:
        legacy = {
            "phase": "legacy",
            "input_payload_sha256": "a" * 64,
            "environment_payload_sha256": "b" * 64,
            "scientific_payload_sha256": "c" * 64,
            "diagnostics_payload": _diagnostics(include_new_rank=False),
        }
        post36 = copy.deepcopy(legacy)
        post36["phase"] = "post36"
        post36["diagnostics_payload"] = _diagnostics(include_new_rank=True)
        return legacy, post36

    def test_compare_accepts_only_six_rank_diagnostic_additions(self) -> None:
        legacy, post36 = self._records()
        checks = probe._compare_probe_records(legacy, post36)
        self.assertTrue(checks["scientific_payload_byte_identical"])
        self.assertEqual(
            set(checks["diagnostic_additions"]), probe._expected_rank_additions()
        )

    def test_compare_rejects_changed_scientific_payload(self) -> None:
        legacy, post36 = self._records()
        post36["scientific_payload_sha256"] = "d" * 64
        with self.assertRaises(probe.ProbeError):
            probe._compare_probe_records(legacy, post36)

    def test_compare_rejects_missing_or_wrong_rank_diagnostic(self) -> None:
        legacy, post36 = self._records()
        del post36["diagnostics_payload"]["cases"][
            "adequate_template_count_aware"
        ]["directions"]["left_to_right"]["models"]["T"]["diagnostics"][
            "representation_rank"
        ]
        with self.assertRaises(probe.ProbeError):
            probe._compare_probe_records(legacy, post36)

        legacy, post36 = self._records()
        post36["diagnostics_payload"]["cases"][
            "adequate_template_count_aware"
        ]["directions"]["right_to_left"]["models"]["M"]["diagnostics"][
            "representation_rank"
        ] = 9
        with self.assertRaises(probe.ProbeError):
            probe._compare_probe_records(legacy, post36)

    def test_compare_rejects_any_other_diagnostic_change(self) -> None:
        legacy, post36 = self._records()
        post36["diagnostics_payload"]["cases"][
            "curved_continuum_low_rank_composition"
        ]["directions"]["left_to_right"]["models"]["U"]["diagnostics"][
            "covariance_mode"
        ] = "diagonal"
        with self.assertRaises(probe.ProbeError):
            probe._compare_probe_records(legacy, post36)


class ContractTransitionProbeReceiptTests(unittest.TestCase):
    def test_run_receipt_self_hash_and_payload_hashes_verify(self) -> None:
        spec_sha = probe._sha256_file(SPEC_PATH)
        record = _minimal_run_record(spec_sha)
        probe._verify_run_record(record, spec_sha)

        tampered = copy.deepcopy(record)
        tampered["scientific_payload"]["extra"] = True
        with self.assertRaises(probe.ProbeError):
            probe._verify_run_record(tampered, spec_sha)

    def test_embedded_tensor_bytes_are_checked(self) -> None:
        raw = b"\x00\x01\x02\x03"
        record = {
            "byte_count": len(raw),
            "c_order_bytes_sha256": probe._sha256_bytes(raw),
            "c_order_bytes_base64": base64.b64encode(raw).decode("ascii"),
        }
        probe._validate_embedded_arrays(record)
        record["byte_count"] += 1
        with self.assertRaises(probe.ProbeError):
            probe._validate_embedded_arrays(record)

    def test_environment_must_be_exactly_single_threaded(self) -> None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        with mock.patch.dict(
            os.environ, probe.EXPECTED_REQUIRED_ENVIRONMENT, clear=False
        ):
            probe._require_environment(spec)
            os.environ["OMP_NUM_THREADS"] = "2"
            with self.assertRaises(probe.ProbeError):
                probe._require_environment(spec)


if __name__ == "__main__":
    unittest.main()
