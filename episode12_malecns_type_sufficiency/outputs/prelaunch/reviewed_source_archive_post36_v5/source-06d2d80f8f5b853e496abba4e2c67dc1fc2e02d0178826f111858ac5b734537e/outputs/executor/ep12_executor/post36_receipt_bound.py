"""Runtime authentication for archive-bound post-36 Slurm stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from .policy import canonical_json
from . import post36_unattended_controller as _controller_module
from .post36_unattended_controller import (
    INTENT_RELATIVE,
    STAGE_ORDER,
    TRANSACTION_RELATIVE,
    verify_post36_start_authorization,
)
from .synthetic import EPISODE_ROOT


STAGE_RESULT_SCHEMA = "ep12.source_archive_bootstrap_stage.v1"
RESULT_SCHEMA = "ep12.post36_stage_runtime_evidence.v1"
_ALLOWED_STAGE_STATUS = {"created", "recovered_identical", "reused_identical"}


class Post36ReceiptBoundError(RuntimeError):
    """A running stage differs from its durable controller receipt."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Post36ReceiptBoundError(message)


def _read_regular(path: Path, *, label: str) -> bytes:
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        metadata = os.lstat(current)
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symlink")
    descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"{label} is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _sha(path: Path, *, label: str) -> str:
    return hashlib.sha256(_read_regular(path, label=label)).hexdigest()


def _stage_result(
    *, path: Path, staged_root: Path, archive: Mapping[str, Any]
) -> dict[str, Any]:
    encoded = _read_regular(path, label="bootstrap stage result")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Post36ReceiptBoundError("bootstrap stage result is malformed") from exc
    _require(isinstance(value, Mapping), "bootstrap stage result is not an object")
    result = dict(value)
    _require(
        encoded == (canonical_json(result) + "\n").encode("ascii"),
        "bootstrap stage result is not canonical",
    )
    _require(
        set(result)
        == {
            "schema_version", "stage_status", "staged_root",
            "upstream_archived_root", "archive_receipt_file_sha256",
            "archive_receipt_sha256", "manifest_file_sha256",
            "manifest_sha256", "tree_sha256", "complete",
        }
        and result["schema_version"] == STAGE_RESULT_SCHEMA
        and result["stage_status"] in _ALLOWED_STAGE_STATUS
        and result["staged_root"] == str(staged_root)
        and result["upstream_archived_root"] == archive["archived_root"]
        and result["archive_receipt_file_sha256"]
        == archive["archive_receipt_file_sha256"]
        and result["archive_receipt_sha256"] == archive["archive_receipt_sha256"]
        and result["manifest_file_sha256"] == archive["manifest_file_sha256"]
        and result["manifest_sha256"] == archive["manifest_sha256"]
        and result["tree_sha256"] == archive["tree_sha256"]
        and result["complete"] is True,
        "bootstrap stage result identity changed",
    )
    return result


def _accepted_executed_wrapper(
    *, observed: Path, archived: Path, job_id: str
) -> bool:
    return observed == archived or (
        observed.is_absolute()
        and observed.name == "slurm_script"
        and observed.parent.name == f"job{job_id}"
        and str(observed).startswith("/var/spool/slurmd/")
    )


def verify_stage_runtime(
    *, episode_root: Path, orchestration_intent_path: Path,
    expected_intent_file_sha256: str, expected_intent_sha256: str,
    stage_receipt_path: Path, expected_stage: str, executed_wrapper: Path,
    slurm_job_id: str, slurm_job_name: str,
    bootstrap_stage_result_path: Path, staged_root: Path,
    wrapper_arguments: Sequence[str],
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "runtime episode root is not canonical")
    _require(expected_stage in STAGE_ORDER, "runtime stage is unknown")
    evidence = verify_post36_start_authorization(episode_root=root)
    _require(
        Path(os.path.abspath(os.fspath(orchestration_intent_path)))
        == root / INTENT_RELATIVE
        and evidence["intent_path"] == str(root / INTENT_RELATIVE)
        and evidence["intent_file_sha256"] == expected_intent_file_sha256
        and evidence["intent_sha256"] == expected_intent_sha256,
        "runtime orchestration intent identity changed",
    )
    stage = evidence["stages"][expected_stage]
    canonical_receipt = (
        root / TRANSACTION_RELATIVE / "stages" / expected_stage / "receipt.json"
    )
    _require(
        Path(os.path.abspath(os.fspath(stage_receipt_path))) == canonical_receipt,
        "runtime stage receipt path is not canonical",
    )
    _require(
        slurm_job_id == stage["slurm_job_id"]
        and slurm_job_name == stage["slurm_job_name"],
        "runtime Slurm identity differs from the durable receipt",
    )
    arguments = list(wrapper_arguments)
    _require(arguments == stage["arguments"],
             "runtime positional arguments differ from the durable receipt")
    observed_wrapper = Path(os.path.abspath(os.fspath(executed_wrapper)))
    archived_wrapper = Path(str(stage["wrapper"]))
    _require(
        _accepted_executed_wrapper(
            observed=observed_wrapper, archived=archived_wrapper,
            job_id=slurm_job_id,
        ),
        "runtime wrapper path is neither archived source nor the exact Slurm spool",
    )
    _require(
        _sha(observed_wrapper, label="executed stage wrapper")
        == stage["wrapper_sha256"],
        "executed stage wrapper bytes differ from the archived wrapper",
    )
    stage_result = _stage_result(
        path=bootstrap_stage_result_path,
        staged_root=Path(os.path.abspath(os.fspath(staged_root))),
        archive=evidence["replacement"]["archive"],
    )
    staged_executor = Path(str(stage_result["staged_root"])) / "outputs/executor"
    module_path = Path(__file__).resolve(strict=True)
    expected_module = (
        staged_executor / "ep12_executor/post36_receipt_bound.py"
    ).resolve(strict=True)
    _require(module_path == expected_module,
             "runtime verifier was imported outside the authenticated stage")
    expected_controller = (
        staged_executor / "ep12_executor/post36_unattended_controller.py"
    ).resolve(strict=True)
    _require(
        Path(str(_controller_module.__file__)).resolve(strict=True)
        == expected_controller,
        "runtime controller was imported outside the authenticated stage",
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "stage": expected_stage,
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": slurm_job_name,
        "orchestration_intent_path": evidence["intent_path"],
        "orchestration_intent_file_sha256": expected_intent_file_sha256,
        "orchestration_intent_sha256": expected_intent_sha256,
        "stage_receipt_path": str(canonical_receipt),
        "stage_receipt_file_sha256": _sha(
            canonical_receipt, label="stage receipt"
        ),
        "stage_receipt_sha256": stage["stage_receipt_sha256"],
        "release_authorization_file_sha256": stage[
            "release_authorization_file_sha256"
        ],
        "release_authorization_sha256": stage[
            "release_authorization_sha256"
        ],
        "release_receipt_file_sha256": stage["release_receipt_file_sha256"],
        "release_receipt_sha256": stage["release_receipt_sha256"],
        "executed_wrapper_path": str(observed_wrapper),
        "executed_wrapper_sha256": stage["wrapper_sha256"],
        "wrapper_arguments": arguments,
        "archive": dict(evidence["replacement"]["archive"]),
        "bootstrap_stage_result_path": str(bootstrap_stage_result_path),
        "bootstrap_stage_result_file_sha256": _sha(
            bootstrap_stage_result_path, label="bootstrap stage result"
        ),
        "staged_root": str(staged_root),
        "staged_executor_root": str(staged_executor),
        "runtime_verifier_path": str(module_path),
        "runtime_verifier_under_staged_executor": True,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-stage-runtime")
    verify.add_argument("--episode-root", type=Path, required=True)
    verify.add_argument("--orchestration-intent-path", type=Path, required=True)
    verify.add_argument("--expected-intent-file-sha256", required=True)
    verify.add_argument("--expected-intent-sha256", required=True)
    verify.add_argument("--stage-receipt-path", type=Path, required=True)
    verify.add_argument("--expected-stage", choices=STAGE_ORDER, required=True)
    verify.add_argument("--executed-wrapper", type=Path, required=True)
    verify.add_argument("--slurm-job-id", required=True)
    verify.add_argument("--slurm-job-name", required=True)
    verify.add_argument("--bootstrap-stage-result-path", type=Path, required=True)
    verify.add_argument("--staged-root", type=Path, required=True)
    verify.add_argument("--wrapper-arguments", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    result = verify_stage_runtime(
        episode_root=arguments.episode_root,
        orchestration_intent_path=arguments.orchestration_intent_path,
        expected_intent_file_sha256=arguments.expected_intent_file_sha256,
        expected_intent_sha256=arguments.expected_intent_sha256,
        stage_receipt_path=arguments.stage_receipt_path,
        expected_stage=arguments.expected_stage,
        executed_wrapper=arguments.executed_wrapper,
        slurm_job_id=arguments.slurm_job_id,
        slurm_job_name=arguments.slurm_job_name,
        bootstrap_stage_result_path=arguments.bootstrap_stage_result_path,
        staged_root=arguments.staged_root,
        wrapper_arguments=arguments.wrapper_arguments,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
