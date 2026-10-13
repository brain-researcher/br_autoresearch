"""Adapter for a qualification launched by the archive-bound downstream DAG."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .qualification_launch import QUALIFICATION_HASH_TARGETS
from . import post36_downstream_launch as _downstream_module
from .post36_downstream_launch import (
    INTENT_RELATIVE as DOWNSTREAM_INTENT_RELATIVE,
    TRANSACTION_RELATIVE as DOWNSTREAM_TRANSACTION_RELATIVE,
    verify_post36_downstream_receipt,
)
from .post36_orchestration import _sha256_file
from .post36_unattended_controller import (
    INTENT_RELATIVE as UPSTREAM_INTENT_RELATIVE,
    verify_post36_start_authorization,
)
from .synthetic import EPISODE_ROOT


EVIDENCE_SCHEMA = "ep12.archive_nested_qualification_launch_evidence.v1"


class ArchiveNestedQualificationError(RuntimeError):
    """The archive-bound qualification launch is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveNestedQualificationError(message)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _source_identity(executor: Path) -> tuple[dict[str, str], dict[str, str]]:
    identity: dict[str, str] = {}
    paths: dict[str, str] = {}
    prefix = Path("outputs/executor")
    for field, episode_relative in QUALIFICATION_HASH_TARGETS.items():
        try:
            relative = episode_relative.relative_to(prefix)
        except ValueError as exc:
            raise ArchiveNestedQualificationError(
                f"qualification source path left executor surface: {episode_relative}"
            ) from exc
        path = executor / relative
        identity[field] = _sha256_file(
            path, root=executor, label=f"staged qualification source {field}"
        )
        paths[field] = str(path)
    return identity, paths


def verify_archive_nested_qualification_launch(
    *,
    episode_root: Path,
    executor_root: Path,
    launch_path: Path,
) -> dict[str, Any]:
    """Return normalized qualification authority from the canonical DAG."""

    root = _absolute(episode_root)
    executor = _absolute(executor_root)
    _require(root == EPISODE_ROOT, "nested qualification root is not canonical")
    expected_module = (
        executor / "ep12_executor/archive_nested_qualification.py"
    ).resolve(strict=True)
    _require(
        Path(__file__).resolve(strict=True) == expected_module,
        "nested qualification adapter was imported outside the authenticated stage",
    )
    expected_downstream = (
        executor / "ep12_executor/post36_downstream_launch.py"
    ).resolve(strict=True)
    _require(
        Path(str(_downstream_module.__file__)).resolve(strict=True)
        == expected_downstream,
        "downstream verifier was imported outside the authenticated stage",
    )

    controller = verify_post36_start_authorization(episode_root=root)
    publisher = controller["stages"]["publisher"]
    coordinator = controller["stages"]["downstream"]
    downstream = verify_post36_downstream_receipt(
        episode_root=root,
        executor_root=executor,
        orchestration_intent=root / UPSTREAM_INTENT_RELATIVE,
        orchestration_intent_sha256=str(controller["intent_sha256"]),
        publisher_job_id=str(publisher["slurm_job_id"]),
        coordinator_job_id=str(coordinator["slurm_job_id"]),
    )
    canonical_launch = (
        root
        / DOWNSTREAM_TRANSACTION_RELATIVE
        / "stages/scientific_qualification/receipt.json"
    )
    _require(
        _absolute(launch_path) == canonical_launch,
        "nested qualification launch path is not canonical",
    )
    test = downstream["stages"]["synthetic_tests"]
    qualification = downstream["stages"]["scientific_qualification"]
    _require(
        qualification["dependency"] == f"afterok:{test['slurm_job_id']}"
        and qualification["predecessor_job_id"] == test["slurm_job_id"],
        "nested qualification dependency differs from the receipted test job",
    )
    arguments = qualification.get("arguments")
    _require(
        isinstance(arguments, list)
        and len(arguments) == 17
        and arguments[7] == str(root / DOWNSTREAM_INTENT_RELATIVE)
        and arguments[10] == str(canonical_launch),
        "nested qualification positional contract changed",
    )
    output_dir = root / "outputs/prelaunch/scientific_qualification_qualification003"
    transition = downstream["scientific_transition"]
    _require(
        arguments[11] == str(output_dir)
        and arguments[12] == "72"
        and arguments[13] == "8"
        and arguments[14] == str(transition["transition_manifest_path"])
        and arguments[15] == str(transition["transition_manifest_file_sha256"])
        and arguments[16] == str(transition["transition_receipt_sha256"]),
        "nested qualification scientific arguments changed",
    )
    source_identity, source_paths = _source_identity(executor)
    _require(
        source_identity["slurm_script_sha256"]
        == qualification["wrapper_sha256"],
        "nested qualification wrapper differs from authenticated staged source",
    )
    _require(
        downstream["archive"] == controller["replacement"]["archive"],
        "nested qualification archive differs from the start authority",
    )
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "launch_path": str(canonical_launch),
        "launch_file_sha256": _sha256_file(
            canonical_launch, root=root, label="nested qualification stage receipt"
        ),
        "launch_sha256": qualification["stage_receipt_sha256"],
        "downstream_intent_path": downstream["intent_path"],
        "downstream_intent_file_sha256": downstream["intent_file_sha256"],
        "downstream_intent_sha256": downstream["intent_sha256"],
        "output_dir": str(output_dir),
        "replicates_per_core_error_cell": 72,
        "whole_types_per_replicate": 8,
        "slurm_script": qualification["wrapper"],
        "slurm_job_id": qualification["slurm_job_id"],
        "slurm_job_name": qualification["slurm_job_name"],
        "submission_user": qualification["submission_user"],
        "test_job_id": test["slurm_job_id"],
        "dependency": qualification["dependency"],
        "kill_on_invalid_dependency": True,
        "scientific_contract_transition": dict(transition),
        "source_identity": source_identity,
        "source_paths": source_paths,
        "archive": dict(downstream["archive"]),
        "stage_receipt_sha256": qualification["stage_receipt_sha256"],
        "release_authorization_file_sha256": qualification[
            "release_authorization_file_sha256"
        ],
        "release_authorization_sha256": qualification[
            "release_authorization_sha256"
        ],
        "release_receipt_file_sha256": qualification[
            "release_receipt_file_sha256"
        ],
        "release_receipt_sha256": qualification["release_receipt_sha256"],
        "all_stage_and_release_receipts_verified": True,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def qualification_launch_view(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Project normalized evidence to fields consumed by readiness checks."""

    source = evidence.get("source_identity")
    _require(isinstance(source, Mapping), "nested qualification lacks source identity")
    return {
        "output_dir": evidence["output_dir"],
        "replicates_per_core_error_cell": evidence[
            "replicates_per_core_error_cell"
        ],
        "whole_types_per_replicate": evidence["whole_types_per_replicate"],
        "scientific_contract_transition": dict(
            evidence["scientific_contract_transition"]
        ),
        "slurm_job_id": evidence["slurm_job_id"],
        "slurm_job_name": evidence["slurm_job_name"],
        "submission_user": evidence["submission_user"],
        "test_job_id": evidence["test_job_id"],
        "dependency": evidence["dependency"],
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
        **dict(source),
    }
