"""Strict cross-stage identities for the EP12 launch chain.

This module keeps downstream launchers from trusting a caller-supplied Slurm
number.  The number must occur in the canonical, self-consistent receipt,
intent, and progress transaction written by the preceding launcher.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from . import test_launch
from .source_freeze import SourceFreezeError


class LaunchChainIdentityError(RuntimeError):
    """A predecessor launch transaction cannot be authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchChainIdentityError(message)


def authenticate_test_launch(
    *,
    episode_root: Path,
    expected_job_id: str | int | None = None,
) -> dict[str, Any]:
    """Authenticate the complete canonical test-launch transaction."""

    try:
        paths = test_launch._paths(episode_root)
        receipt = test_launch.read_test_launch_receipt(
            paths["receipt"], episode_root=paths["root"]
        )
        expected_intent = test_launch._build_intent(
            episode_root=paths["root"], submission_user=test_launch._effective_user()
        )
        intent = test_launch._verify_intent(
            test_launch._read_json(paths["intent"], episode_root=paths["root"]),
            expected_intent,
        )
        state, job_id, progress = test_launch._verify_progress(
            test_launch._read_json(paths["progress"], episode_root=paths["root"]),
            intent=intent,
        )
    except (OSError, SourceFreezeError, test_launch.TestLaunchError) as exc:
        raise LaunchChainIdentityError(str(exc)) from exc
    _require(
        state == "submitted" and isinstance(job_id, str),
        "canonical test-launch progress is not submitted",
    )
    _require(receipt.get("slurm_job_id") == job_id, "test-launch job ID differs from progress")
    expected_receipt = test_launch._receipt_value(
        intent=intent, progress=progress, job_id=job_id
    )
    _require(receipt == expected_receipt, "canonical test-launch receipt differs from its transaction")
    if expected_job_id is not None:
        _require(
            str(expected_job_id) == job_id,
            "caller-supplied test job ID is unrelated to the canonical test launch",
        )
    receipt_bytes = test_launch._read_bytes(
        paths["receipt"], episode_root=paths["root"], label="test-launch receipt"
    )
    source = receipt.get("source_freeze")
    _require(isinstance(source, Mapping), "canonical test launch lacks source-freeze identity")
    return {
        "test_job_id": job_id,
        "test_launch_receipt_path": str(paths["receipt"]),
        "test_launch_receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "test_launch_receipt_sha256": receipt["receipt_sha256"],
        "submission_user": receipt["submission_user"],
        "source_freeze_manifest": source["source_freeze_manifest"],
        "source_freeze_manifest_sha256": source["source_freeze_manifest_sha256"],
        "source_freeze_self_sha256": source["source_freeze_self_sha256"],
    }
