"""Runtime authentication for archive-bound downstream Slurm stages.

This verifier is executed only after the standalone source-archive bootstrap
has authenticated and staged the complete reviewed executor.  It binds the
running test, qualification, or readiness job to the canonical downstream
coordinator receipt, including the exact accepted-job and release receipts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from .policy import canonical_json
from . import post36_receipt_bound as _base_runtime_module
from . import post36_downstream_launch as _downstream_module
from . import post36_unattended_controller as _controller_module
from .post36_downstream_launch import (
    INTENT_RELATIVE,
    STAGE_ORDER,
    TRANSACTION_RELATIVE,
    verify_post36_downstream_receipt,
)
from .post36_receipt_bound import (
    _accepted_executed_wrapper,
    _sha,
    _stage_result,
)
from .post36_unattended_controller import (
    INTENT_RELATIVE as UPSTREAM_INTENT_RELATIVE,
    verify_post36_start_authorization,
)
from .synthetic import EPISODE_ROOT


RESULT_SCHEMA = "ep12.post36_downstream_stage_runtime_evidence.v1"


class Post36DownstreamReceiptBoundError(RuntimeError):
    """A running downstream stage differs from its durable receipt."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Post36DownstreamReceiptBoundError(message)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _verify_ep12_import_origins(staged_executor: Path) -> None:
    boundary = staged_executor.resolve(strict=True)
    for name, imported in tuple(sys.modules.items()):
        if name != "ep12_executor" and not name.startswith("ep12_executor."):
            continue
        imported_path = getattr(imported, "__file__", None)
        if imported_path is None:
            continue
        resolved = Path(str(imported_path)).resolve(strict=True)
        _require(
            resolved == boundary or boundary in resolved.parents,
            f"EP12 module was imported outside the authenticated stage: {name}",
        )


def verify_downstream_stage_runtime(
    *,
    episode_root: Path,
    orchestration_intent_path: Path,
    expected_intent_file_sha256: str,
    expected_intent_sha256: str,
    stage_receipt_path: Path,
    expected_stage: str,
    executed_wrapper: Path,
    slurm_job_id: str,
    slurm_job_name: str,
    bootstrap_stage_result_path: Path,
    staged_root: Path,
    wrapper_arguments: Sequence[str],
) -> dict[str, Any]:
    """Authenticate one running canonical downstream stage."""

    root = _absolute(episode_root)
    _require(root == EPISODE_ROOT, "runtime episode root is not canonical")
    _require(expected_stage in STAGE_ORDER, "runtime downstream stage is unknown")

    controller = verify_post36_start_authorization(episode_root=root)
    archive = controller["replacement"]["archive"]
    staged = _absolute(staged_root)
    stage_result = _stage_result(
        path=bootstrap_stage_result_path,
        staged_root=staged,
        archive=archive,
    )
    staged_executor = staged / "outputs/executor"

    module_path = Path(__file__).resolve(strict=True)
    expected_module = (
        staged_executor
        / "ep12_executor/post36_downstream_receipt_bound.py"
    ).resolve(strict=True)
    _require(
        module_path == expected_module,
        "downstream runtime verifier was imported outside the authenticated stage",
    )
    expected_downstream = (
        staged_executor / "ep12_executor/post36_downstream_launch.py"
    ).resolve(strict=True)
    _require(
        Path(str(_downstream_module.__file__)).resolve(strict=True)
        == expected_downstream,
        "downstream coordinator was imported outside the authenticated stage",
    )
    expected_controller = (
        staged_executor / "ep12_executor/post36_unattended_controller.py"
    ).resolve(strict=True)
    _require(
        Path(str(_controller_module.__file__)).resolve(strict=True)
        == expected_controller,
        "top controller was imported outside the authenticated stage",
    )
    expected_base_runtime = (
        staged_executor / "ep12_executor/post36_receipt_bound.py"
    ).resolve(strict=True)
    _require(
        Path(str(_base_runtime_module.__file__)).resolve(strict=True)
        == expected_base_runtime,
        "base runtime verifier was imported outside the authenticated stage",
    )

    publisher = controller["stages"]["publisher"]
    coordinator = controller["stages"]["downstream"]
    evidence = verify_post36_downstream_receipt(
        episode_root=root,
        executor_root=staged_executor,
        orchestration_intent=root / UPSTREAM_INTENT_RELATIVE,
        orchestration_intent_sha256=str(controller["intent_sha256"]),
        publisher_job_id=str(publisher["slurm_job_id"]),
        coordinator_job_id=str(coordinator["slurm_job_id"]),
    )
    _require(
        evidence["archive"] == archive,
        "downstream receipt archive differs from the start authority",
    )

    canonical_intent = root / INTENT_RELATIVE
    _require(
        _absolute(orchestration_intent_path) == canonical_intent
        and evidence["intent_path"] == str(canonical_intent)
        and evidence["intent_file_sha256"] == expected_intent_file_sha256
        and evidence["intent_sha256"] == expected_intent_sha256,
        "runtime downstream intent identity changed",
    )
    stage = evidence["stages"][expected_stage]
    canonical_receipt = (
        root / TRANSACTION_RELATIVE / "stages" / expected_stage / "receipt.json"
    )
    _require(
        _absolute(stage_receipt_path) == canonical_receipt,
        "runtime downstream stage receipt path is not canonical",
    )
    _require(
        slurm_job_id == stage["slurm_job_id"]
        and slurm_job_name == stage["slurm_job_name"],
        "runtime Slurm identity differs from the downstream receipt",
    )
    arguments = list(wrapper_arguments)
    _require(
        arguments == stage["arguments"],
        "runtime positional arguments differ from the downstream receipt",
    )
    observed_wrapper = _absolute(executed_wrapper)
    archived_wrapper = Path(str(stage["wrapper"]))
    _require(
        _accepted_executed_wrapper(
            observed=observed_wrapper,
            archived=archived_wrapper,
            job_id=slurm_job_id,
        ),
        "runtime wrapper path is neither archived source nor the exact Slurm spool",
    )
    _require(
        _sha(observed_wrapper, label="executed downstream wrapper")
        == stage["wrapper_sha256"],
        "executed downstream wrapper bytes differ from the archived wrapper",
    )
    _require(
        stage.get("release_receipt_file_sha256")
        and stage.get("release_receipt_sha256")
        and stage.get("release_authorization_file_sha256")
        and stage.get("release_authorization_sha256"),
        "downstream stage lacks a complete durable release proof",
    )
    _verify_ep12_import_origins(staged_executor)

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
            canonical_receipt, label="downstream stage receipt"
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
        "archive": dict(archive),
        "bootstrap_stage_result_path": str(bootstrap_stage_result_path),
        "bootstrap_stage_result_file_sha256": _sha(
            bootstrap_stage_result_path,
            label="bootstrap stage result",
        ),
        "staged_root": str(staged),
        "staged_executor_root": str(staged_executor),
        "runtime_verifier_path": str(module_path),
        "runtime_verifier_under_staged_executor": True,
        "all_stage_and_release_receipts_verified": True,
        "obsolete_job_is_execution_dependency": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--orchestration-intent-path", type=Path, required=True)
    parser.add_argument("--expected-intent-file-sha256", required=True)
    parser.add_argument("--expected-intent-sha256", required=True)
    parser.add_argument("--stage-receipt-path", type=Path, required=True)
    parser.add_argument("--expected-stage", choices=STAGE_ORDER, required=True)
    parser.add_argument("--executed-wrapper", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--slurm-job-name", required=True)
    parser.add_argument("--bootstrap-stage-result-path", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--wrapper-arguments", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    result = verify_downstream_stage_runtime(
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
