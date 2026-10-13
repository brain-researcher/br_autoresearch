#!/usr/bin/env python3
"""Render one fail-closed EP21 branch qualification from the main report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


LAION_UNRESOLVED = [
    "release_package_identifier",
    "license_and_DUA_permission",
    "Sherlock_Low_or_Moderate_Risk_decision",
    "five_participant_identifiers",
    "session_run_trial_image_repetition_joins",
    "beta_variant_shapes_dtypes_missingness_and_masks",
    "subject_unique_tau_train_tau_test_roles",
    "exact_overlap_and_repetition_gates",
    "eligible_20_directed_pair_table",
    "paper_vs_tagged_method_choice",
    "Sherlock_runtime_environment",
    "exact_AlexNet_identification_checkpoint",
    "VGG19_archive_to_checkpoint_binding",
    "main_bdpy_revision",
]


def laion_unresolved(qualification: Mapping[str, Any]) -> list[str]:
    unresolved = list(LAION_UNRESOLVED)
    assets = qualification.get("source_pipeline_assets", {})
    audit = assets.get("public_model_payload_audit")
    if (
        assets.get("payload_verified_public_model_asset_count") == 2
        and isinstance(audit, Mapping)
        and audit.get("schema_version") == "ep21.public_model_payload_audit.v2"
    ):
        unresolved[unresolved.index("VGG19_archive_to_checkpoint_binding")] = (
            "VGG19_evaluator_lab_copy_and_strict_runtime_load"
        )
    return unresolved


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def render_branch(branch: str, qualification: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        "astra_analysis_id": "source_qualification",
        "attempt": qualification.get("attempt"),
        "started_at": qualification.get("started_at"),
        "completed_at": qualification.get("completed_at"),
        "process_id": qualification.get("process_id"),
        "eligible_directed_pairs": [],
        "empirical_data_accessed": False,
    }
    if qualification.get("empirical_or_ood_payload_accessed") is not False:
        raise ValueError("qualification report does not preserve the closed boundary")
    if qualification.get("source_identity", {}).get("state") != "verified":
        raise ValueError("source identity is not verified")

    if branch == "laion":
        matrix = qualification["structural_qualification_matrix"]
        assets = qualification["source_pipeline_assets"]
        return {
            "schema_version": "ep21.laion_branch_qualification.v2",
            **common,
            "status": "blocked_not_runnable",
            "planning_participant_count": matrix["participant_ids"]["planning_count"],
            "planning_directed_pair_count": matrix["directed_laion_pair_table"][
                "planning_count"
            ],
            "verified": {
                "source_release_identity": True,
                "public_reconstruction_asset_metadata_count": assets[
                    "verified_public_model_asset_count"
                ],
                "prior_exposure": matrix["prior_exposure"]["classification"],
                "declared_input_tree_excludes_ood_interface": matrix[
                    "declared_ep21_input_boundary"
                ]["physical_exclusion"]["ep21_input_tree_contains_ood_payload"]
                is False,
            },
            "absent": {
                "laion_regular_feature_handoff": matrix["regular_feature_handoff"][
                    "state"
                ]
                == "absent",
                "laion_regular_scoring_handoff": matrix["regular_scoring_handoff"][
                    "state"
                ]
                == "absent",
                "protocol_lock": qualification.get("protocol_lock_present") is False,
            },
            "unknown_or_unresolved": laion_unresolved(qualification),
            "next_action": "Obtain DUA/risk-cleared structural metadata and an isolated regular-only feature handoff; resolve the paper-versus-tagged choice in pipeline conformance before any scoring handoff becomes visible.",
        }
    if branch == "external":
        return {
            "schema_version": "ep21.external_branch_qualification.v2",
            **common,
            "status": "blocked_unprovisioned_optional_branch",
            "planning_directed_pair_count": 70,
            "nsd_status": "not_provisioned_as_ep21_asset",
            "things_status": "not_provisioned_as_ep21_asset",
            "within_laion_blocked_by_external_absence": False,
            "next_action": "Provision exact NSD/THINGS releases, permissions, risk decisions, and isolated handoffs only if the optional inter-site branch is desired; otherwise lock it not_evaluable without blocking LAION.",
        }
    raise ValueError(f"unsupported branch: {branch}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["laion", "external"], required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    atomic_json(args.output, render_branch(args.branch, qualification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
