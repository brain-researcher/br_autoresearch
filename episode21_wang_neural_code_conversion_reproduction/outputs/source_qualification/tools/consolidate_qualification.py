#!/usr/bin/env python3
"""Consolidate EP21 source qualification into the three declared ASTRA outputs."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PINNED_SOURCE_RELEASE = {
    "repository": "https://github.com/KamitaniLab/InterSiteNeuralCodeConversion",
    "tag": "V1.0.0",
    "expected_commit": "edbff02edd6a08d3c6c829af45a957fd14bdc460",
    "observed_commit": "edbff02edd6a08d3c6c829af45a957fd14bdc460",
    "zenodo_record": "14910040",
    "zenodo_doi": "10.5281/zenodo.14910040",
    "expected_archive_name": "KamitaniLab/InterSiteNeuralCodeConversion-V1.0.0.zip",
    "observed_archive_size_bytes": 23_110_908,
    "observed_archive_md5": "bb80dcc1a038737d6d8fc2c95b3e5586",
}
DIRECT_INTERFACE_NAMES = (
    "laion_regular_feature_handoff",
    "laion_regular_scoring_handoff",
    "external_site_feature_handoff",
    "external_site_scoring_handoff",
)
PUBLIC_MODEL_PINS = {
    "vgg19": {
        "figshare_file_id": 38225868,
        "filename": "VGG_ILSVRC_19_layers.zip",
        "size_bytes": 533_710_127,
        "md5": "49f4d6dbf1024f33149352dfd309dc4d",
        "checkpoint_member": "VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt",
    },
    "caffenet": {
        "figshare_file_id": 52398902,
        "filename": "bvlc_reference_caffenet.zip",
        "size_bytes": 226_699_018,
        "md5": "2e4fe2126c7ed13554e33d716a72c65f",
        "checkpoint_member": "bvlc_reference_caffenet/bvlc_reference_caffenet.pt",
    },
}
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def validate_pinned_source_release(source_release: dict[str, Any]) -> None:
    """Reject a self-consistent report for anything except the EP21 release."""
    for field, expected in PINNED_SOURCE_RELEASE.items():
        observed = source_release.get(field)
        matches = (
            str(observed) == str(expected)
            if field == "zenodo_record"
            else observed == expected
        )
        if not matches:
            raise ValueError(
                f"source release {field} does not match the EP21 pin: "
                f"expected {expected!r}, got {observed!r}"
            )
    for field in (
        "tag_commit_matches",
        "zenodo_record_matches",
        "zenodo_doi_matches",
        "archive_size_matches",
        "archive_md5_matches",
        "inventory_release_matches_pin",
    ):
        if source_release.get(field) is not True:
            raise ValueError(f"source release check {field} is not true")
    if source_release.get("identity_status") != "verified":
        raise ValueError("source identity is not verified")


def validate_interface_state_reconciliation(
    manifest_interfaces: dict[str, Any], initial_qualification: dict[str, Any]
) -> dict[str, str]:
    """Require manifest claims to match direct existence checks from this replay."""
    observed = initial_qualification.get("declared_interface_state")
    if not isinstance(observed, dict):
        raise ValueError("initial qualification lacks direct interface state")
    for name in DIRECT_INTERFACE_NAMES:
        manifest_state = manifest_interfaces.get(name)
        observed_state = observed.get(name)
        if manifest_state not in {"absent", "present"}:
            raise ValueError(f"manifest interface state is invalid for {name}")
        if observed_state not in {"absent", "present"}:
            raise ValueError(f"direct interface state is invalid for {name}")
        if manifest_state != observed_state:
            raise ValueError(
                f"manifest/direct interface mismatch for {name}: "
                f"manifest={manifest_state!r}, observed={observed_state!r}"
            )
    return {name: observed[name] for name in DIRECT_INTERFACE_NAMES}


def validate_public_model_payload_audit(
    report: dict[str, Any], attempt: int
) -> None:
    if report.get("schema_version") != "ep21.public_model_payload_audit.v2":
        raise ValueError("public-model payload audit schema is not v2")
    if report.get("astra_analysis_id") != "source_qualification":
        raise ValueError("public-model payload audit has the wrong analysis ID")
    if report.get("attempt") != attempt:
        raise ValueError("public-model payload audit attempt does not match replay")
    if report.get("outcome") != (
        "completed_public_reconstruction_assets_authenticated_"
        "evaluator_checkpoints_unbound"
    ):
        raise ValueError("public-model payload audit did not reach its closed outcome")
    for field in (
        "empirical_or_decoded_payload_accessed",
        "ood_or_ep04_payload_accessed",
        "checkpoint_deserialization_performed",
    ):
        if report.get(field) is not False:
            raise ValueError(f"public-model payload boundary field is not false: {field}")
    assets = report.get("assets", {})
    for asset_id, pin in PUBLIC_MODEL_PINS.items():
        asset = assets.get(asset_id, {})
        observed_pin = asset.get("pinned_identity", {})
        for field in ("figshare_file_id", "filename", "size_bytes", "md5"):
            if observed_pin.get(field) != pin[field]:
                raise ValueError(f"public-model pin mismatch: {asset_id}.{field}")
        cache = asset.get("cache", {})
        if not all(
            (
                cache.get("size_matches") is True,
                cache.get("md5_matches") is True,
                cache.get("cache_regular_file_verified") is True,
                cache.get("cache_symlink_rejected") is True,
                cache.get("cache_mode") == "0440",
            )
        ):
            raise ValueError(f"public-model cache validation failed: {asset_id}")
        inventory = asset.get("archive_inventory", {})
        if inventory.get("unsafe_member_names") != []:
            raise ValueError(f"unsafe public-model archive member: {asset_id}")
        if inventory.get("non_regular_member_names") != []:
            raise ValueError(f"non-regular public-model archive member: {asset_id}")
        matches = [
            item
            for item in inventory.get("members", [])
            if item.get("path") == pin["checkpoint_member"]
            and item.get("member_type") == "regular_file"
            and len(item.get("sha256", "")) == 64
        ]
        if len(matches) != 1:
            raise ValueError(f"exact public-model checkpoint member missing: {asset_id}")
    bindings = report.get("public_reconstruction_bindings", {})
    if bindings.get("vgg19", {}).get("checkpoint_member_verified") is not True:
        raise ValueError("VGG public reconstruction checkpoint is not bound")
    if (
        bindings.get("caffenet", {}).get("candidate_checkpoint_member_verified")
        is not True
    ):
        raise ValueError("CaffeNet public candidate checkpoint is not bound")
    evaluator = report.get("evaluator_checkpoint_references", {})
    if evaluator.get("vgg19", {}).get("lab_local_byte_identity_verified") is not False:
        raise ValueError("VGG lab-local byte identity must remain unverified")
    if evaluator.get("vgg19", {}).get("strict_bdpy_runtime_load_verified") is not False:
        raise ValueError("VGG strict runtime load must remain unverified")
    alexnet = evaluator.get("alexnet", {})
    if alexnet.get("exact_public_bvlc_alexnet_named_member_verified") is not False:
        raise ValueError("exact public AlexNet checkpoint unexpectedly verified")
    if alexnet.get("caffenet_equivalence_claim_allowed") is not False:
        raise ValueError("CaffeNet substitution must remain forbidden")


def build_reports(
    attempt: int,
    structural: dict[str, Any],
    initial_qualification: dict[str, Any],
    verified_source: dict[str, Any],
    asset_dependencies: dict[str, Any],
    asset_metadata: dict[str, Any],
    runtime: dict[str, Any],
    unmanifested_assets: dict[str, Any],
    started_at: str,
    process_id: int,
    manifest_metadata: dict[str, Any] | None = None,
    public_model_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_release = verified_source["source_release"]
    validate_pinned_source_release(source_release)
    if verified_source.get("forbidden_payload_accessed") is not False:
        raise ValueError("source audit crossed the forbidden payload boundary")
    if verified_source.get("archive", {}).get("unsafe_member_names") != []:
        raise ValueError("source archive contains unsafe member names")
    if asset_metadata["failed_asset_count"] != 0:
        raise ValueError("public model asset metadata has unresolved failures")
    if asset_dependencies["empirical_or_decoded_payload_accessed"] is not False:
        raise ValueError("asset audit crossed the empirical boundary")
    if unmanifested_assets["empirical_or_decoded_payload_accessed"] is not False:
        raise ValueError("unmanifested-asset audit crossed the empirical boundary")
    if manifest_metadata is not None:
        if manifest_metadata.get("state") != "complete_metadata_only_payload_unopened":
            raise ValueError("full Figshare manifest reconciliation is incomplete")
        if manifest_metadata.get("empirical_or_decoded_payload_accessed") is not False:
            raise ValueError("Figshare reconciliation crossed the empirical boundary")
        if manifest_metadata.get("payload_bytes_downloaded") != 0:
            raise ValueError("Figshare reconciliation downloaded a listed payload")
        if manifest_metadata.get("resolved_file_id_count") != 62:
            raise ValueError("Figshare reconciliation did not resolve all 62 file IDs")
        if manifest_metadata.get("readiness_effect") != "none_laion_remains_blocked":
            raise ValueError("Figshare metadata must not promote LAION readiness")
    if public_model_payload is not None:
        validate_public_model_payload_audit(public_model_payload, attempt)

    completed_at = utc_now()
    laion = structural["laion_fmri"]
    interfaces = structural["stable_interfaces"]
    observed_interfaces = validate_interface_state_reconciliation(
        interfaces, initial_qualification
    )
    protocol_lock_present = structural["authorization"]["protocol_lock_present"]
    declared_input_tree_excludes_ood_interface = (
        structural["physical_exclusion"]["ep21_input_tree_contains_ood_payload"]
        is False
    )
    feature_handoff_absent = interfaces["laion_regular_feature_handoff"] == "absent"
    scoring_handoff_absent = interfaces["laion_regular_scoring_handoff"] == "absent"
    structural_matrix = {
        "release_package_identifier": {
            "state": "unknown",
            "evidence": laion["release_identifier"],
        },
        "license_and_dua": {
            "state": "unresolved",
            "evidence": laion["license_and_data_use_status"],
        },
        "sherlock_risk_classification": {
            "state": "unresolved",
            "evidence": laion["sherlock_risk_classification"],
        },
        "participant_ids": {
            "state": laion["participant_ids"]["status"],
            "planning_count": laion["participant_ids"]["planning_count"],
            "verified_values": laion["participant_ids"]["values"],
        },
        "session_run_trial_image_repetition_joins": {
            "state": "unknown",
            "evidence": laion["release_metadata"][
                "session_run_trial_image_repetition_ids"
            ],
        },
        "beta_variant": {
            "state": "unknown",
            "evidence": laion["release_metadata"]["beta_variant"],
        },
        "array_shapes_dtypes_finiteness_missingness": {
            "state": "unknown",
            "shapes": laion["release_metadata"]["array_shapes"],
            "dtypes": laion["release_metadata"]["dtypes"],
            "finite_nan_masks": laion["release_metadata"][
                "finite_nan_masks_without_values"
            ],
            "coverage_missingness": laion["release_metadata"][
                "coverage_and_missingness"
            ],
        },
        "native_roi_and_mask_ids": {
            "state": "unknown",
            "evidence": laion["release_metadata"][
                "native_space_roi_identifiers"
            ],
        },
        "stimulus_roles_and_tau_partition": {
            "state": "unverified",
            "roles": laion["roles"],
        },
        "exact_overlap_and_repetition_gates": {
            "state": "not_run_no_role_filtered_identifiers",
            "checks": laion["overlap_checks"],
        },
        "directed_laion_pair_table": {
            "state": laion["directed_pairs"]["status"],
            "planning_count": laion["directed_pairs"]["planning_count"],
            "eligible_pairs": laion["directed_pairs"]["eligible_pairs"],
        },
        "regular_feature_handoff": {
            "state": interfaces["laion_regular_feature_handoff"],
        },
        "regular_scoring_handoff": {
            "state": interfaces["laion_regular_scoring_handoff"],
        },
        "prior_exposure": {
            "state": "verified_from_contract_and_campaign_inventory",
            "classification": laion["prior_exposure"],
            "independence": laion["evidence_independence"],
        },
        "declared_ep21_input_boundary": {
            "state": "verified_for_declared_interfaces_only",
            "physical_exclusion": structural["physical_exclusion"],
            "limitation": "No upstream mixed-role or DUA-bound payload was traversed; physical exclusion is established only for declared EP21 inputs.",
        },
    }

    asset_gaps = copy.deepcopy(unmanifested_assets["gaps"])
    if public_model_payload is not None:
        vgg_member = public_model_payload["public_reconstruction_bindings"][
            "vgg19"
        ]["checkpoint_member"]
        caffenet_member = public_model_payload["public_reconstruction_bindings"][
            "caffenet"
        ]["candidate_checkpoint_member"]
        asset_gaps["vgg19_identification_checkpoint_binding"].update(
            {
                "state": "public_reconstruction_member_authenticated_evaluator_lab_copy_unbound",
                "exact_checkpoint_path_bound_by_tagged_manifest_and_config": True,
                "public_checkpoint_member": vgg_member,
                "evaluator_lab_local_byte_identity_verified": False,
                "strict_bdpy_runtime_load_verified": False,
                "required_action": "Authenticate the evaluator's lab-local copy or prospectively bind the authenticated public member, then validate a strict load in the locked runtime.",
            }
        )
        asset_gaps["alexnet_identification_checkpoint"].update(
            {
                "state": "required_unmanifested_public_caffenet_not_substitutable",
                "public_caffenet_candidate_member": caffenet_member,
                "exact_public_bvlc_alexnet_named_member_verified": False,
                "caffenet_equivalence_claim_allowed": False,
                "required_action": "Authenticate bvlc_alexnet.pt or lock the affected identification findings not_evaluable; do not substitute the public CaffeNet artifact.",
            }
        )

    qualification = {
        "schema_version": "ep21.source_qualification_report.v2",
        "astra_analysis_id": "source_qualification",
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": completed_at,
        "process_id": process_id,
        "scope": "consolidated_public_source_asset_and_redacted_structure_qualification",
        "evidence_inputs": {
            "structural_manifest": "inputs/conversion_structural_manifest.json",
            "initial_qualification": "outputs/source_qualification/attempt-002/qualification_report.json",
            "verified_source_inventory": "outputs/source_qualification/attempt-002/verified_public_source_inventory.json",
            "public_asset_dependencies": "outputs/source_qualification/attempt-004/public_asset_dependency_report.json",
            "public_asset_metadata": "outputs/source_qualification/attempt-004/public_asset_metadata_report.json",
            "runtime_environment": "outputs/source_qualification/attempt-004/runtime_environment_report.json",
            "unmanifested_source_assets": "outputs/source_qualification/attempt-006/unmanifested_source_asset_report.json",
        },
        "source_identity": {
            "state": "verified",
            "repository": source_release["repository"],
            "tag": source_release["tag"],
            "expected_commit": source_release["expected_commit"],
            "observed_commit": source_release["observed_commit"],
            "zenodo_record": source_release["zenodo_record"],
            "zenodo_doi": source_release["zenodo_doi"],
            "archive_size_bytes": source_release["observed_archive_size_bytes"],
            "archive_md5": source_release["observed_archive_md5"],
            "archive_file_count": verified_source["archive"]["file_count"],
            "unsafe_archive_member_names": verified_source["archive"][
                "unsafe_member_names"
            ],
        },
        "protocol_lock_present": protocol_lock_present,
        "direct_interface_state": observed_interfaces,
        "structural_qualification_matrix": structural_matrix,
        "source_pipeline_assets": {
            "declared_dependency_count": asset_dependencies["asset_count"],
            "classification_counts": asset_dependencies["classification_counts"],
            "excluded_empirical_derived_or_fitted_asset_count": asset_dependencies[
                "excluded_payload_download_count"
            ],
            "public_model_metadata_method": asset_metadata["method"],
            "public_model_metadata_article": asset_metadata["article"],
            "verified_public_model_asset_count": asset_metadata[
                "verified_asset_count"
            ],
            "public_model_assets": asset_metadata["assets"],
            "payload_bytes_downloaded": asset_metadata["payload_bytes_downloaded"],
            "payload_checksum_limitation": "Names, byte sizes, and supplier MD5 metadata match the tagged manifest; model payload bytes have not been downloaded or independently checksummed.",
            "unmanifested_or_unpinned_gaps": asset_gaps,
        },
        "runtime_and_reconstruction": {
            "declared_environments": runtime["environments"],
            "execution_environment_ready": runtime["execution_environment_ready"],
            "environment_blocker": runtime["blocker"],
            "dists_like_custom_loss": runtime["dists_like_custom_loss"],
        },
        "paper_code_discrepancy_register": {
            "activation": {
                "paper": "ReLU",
                "tagged_code": "two LeakyReLU(0.2) activations",
                "state": "recorded_unresolved_for_pipeline_conformance",
            },
            "content_loss_weighting": {
                "paper": "inverse squared norm of true features",
                "tagged_code": "normalized inverse-squared norm of active per-iteration feature blocks",
                "state": "recorded_unresolved_for_pipeline_conformance",
            },
            "reconstruction_layers": {
                "paper": "eight-layer description",
                "tagged_code": "all 19 VGG19 layers in encoder configuration",
                "state": "recorded_unresolved_for_pipeline_conformance",
            },
            "alexnet_backend_and_input_size": {
                "paper": "official Caffe bvlc_alexnet with 224 by 224 inputs",
                "tagged_code": "bdpy PyTorch AlexNet port with direct 227 by 227 resize",
                "state": "recorded_unresolved_for_pipeline_conformance",
            },
        },
        "initial_attempt_outcome": initial_qualification["outcome"],
        "laion_readiness": "blocked",
        "external_readiness": "blocked_unprovisioned_optional_branch",
        "ood_disposition": "not_evaluable_under_current_contract",
        "empirical_or_ood_payload_accessed": False,
        "qualification_outcome": "completed_with_empirical_and_scientific_decision_blockers",
        "next_analysis": "pipeline_conformance_after_scientist_protocol_choice_and_required_structural_evidence",
    }
    if manifest_metadata is not None:
        qualification["evidence_inputs"]["figshare_manifest_reconciliation"] = (
            "outputs/source_qualification/attempt-009/figshare_manifest_reconciliation.json"
        )
        qualification["source_pipeline_assets"][
            "figshare_manifest_metadata_reconciliation"
        ] = manifest_metadata
        qualification["source_pipeline_assets"]["payload_checksum_limitation"] = (
            "All 62 manifest MD5 strings match both Figshare supplied_md5 and computed_md5 metadata fields; no listed payload body was downloaded or independently hashed by EP21."
        )
    if public_model_payload is not None:
        qualification["evidence_inputs"]["public_model_payload_audit"] = (
            "recipe_intermediate:audit_public_model_payloads.py"
        )
        pipeline_assets = qualification["source_pipeline_assets"]
        pipeline_assets["public_model_payload_audit"] = public_model_payload
        pipeline_assets["payload_verified_public_model_asset_count"] = 2
        pipeline_assets["public_model_payload_bytes_inspected"] = sum(
            pin["size_bytes"] for pin in PUBLIC_MODEL_PINS.values()
        )
        pipeline_assets["public_model_payload_bytes_downloaded_this_attempt"] = (
            public_model_payload["payload_bytes_downloaded_this_attempt"]
        )
        pipeline_assets["payload_checksum_limitation"] = (
            "The two public non-empirical VGG/CaffeNet archives were independently "
            "size/MD5 checked and every regular member was streamed for SHA256; the "
            "other manifest payloads remain metadata-only, checkpoints were not "
            "deserialized, and evaluator lab-local copy identity remains unverified."
        )

    vgg_unresolved = (
        "VGG19_evaluator_lab_copy_and_strict_runtime_load"
        if public_model_payload is not None
        else "VGG19_archive_to_checkpoint_binding"
    )
    bdpy_unresolved = "main_bdpy_revision"

    laion_report = {
        "schema_version": "ep21.laion_branch_qualification.v2",
        "astra_analysis_id": "source_qualification",
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": completed_at,
        "process_id": process_id,
        "status": "blocked_not_runnable",
        "verified": {
            "source_release_identity": True,
            "public_reconstruction_asset_metadata_count": asset_metadata[
                "verified_asset_count"
            ],
            "prior_exposure": laion["prior_exposure"],
            "declared_input_tree_excludes_ood_interface": declared_input_tree_excludes_ood_interface,
        },
        "absent": {
            "laion_regular_feature_handoff": feature_handoff_absent,
            "laion_regular_scoring_handoff": scoring_handoff_absent,
            "protocol_lock": protocol_lock_present is False,
        },
        "unknown_or_unresolved": [
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
            vgg_unresolved,
            bdpy_unresolved,
        ],
        "planning_participant_count": 5,
        "planning_directed_pair_count": 20,
        "eligible_directed_pairs": [],
        "empirical_data_accessed": False,
        "next_action": "Obtain DUA/risk-cleared structural metadata and an isolated regular-only feature handoff; resolve the paper-versus-tagged choice in pipeline conformance before any scoring handoff becomes visible.",
    }

    external_report = {
        "schema_version": "ep21.external_branch_qualification.v2",
        "astra_analysis_id": "source_qualification",
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": completed_at,
        "process_id": process_id,
        "status": "blocked_unprovisioned_optional_branch",
        "nsd_status": structural["external_sites"]["nsd"]["status"],
        "things_status": structural["external_sites"]["things"]["status"],
        "planning_directed_pair_count": structural["external_sites"][
            "directed_pairs"
        ]["planning_count_if_fully_qualified"],
        "eligible_directed_pairs": [],
        "within_laion_blocked_by_external_absence": False,
        "empirical_data_accessed": False,
        "next_action": "Provision exact NSD/THINGS releases, permissions, risk decisions, and isolated handoffs only if the optional inter-site branch is desired; otherwise lock it not_evaluable without blocking LAION.",
    }
    return qualification, laion_report, external_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--structural-manifest", type=Path, required=True)
    parser.add_argument("--attempt-2-dir", type=Path, required=True)
    parser.add_argument("--attempt-4-dir", type=Path, required=True)
    parser.add_argument("--attempt-6-dir", type=Path, required=True)
    parser.add_argument("--manifest-metadata-report", type=Path)
    parser.add_argument("--public-model-payload-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started_at = utc_now()
    print(
        json.dumps(
            {
                "event": "ep21_qualification_consolidation_started",
                "pid": os.getpid(),
                "started_at": started_at,
                "attempt": args.attempt,
            }
        ),
        flush=True,
    )
    reports = build_reports(
        attempt=args.attempt,
        structural=load_json(args.structural_manifest),
        initial_qualification=load_json(args.attempt_2_dir / "qualification_report.json"),
        verified_source=load_json(
            args.attempt_2_dir / "verified_public_source_inventory.json"
        ),
        asset_dependencies=load_json(
            args.attempt_4_dir / "public_asset_dependency_report.json"
        ),
        asset_metadata=load_json(
            args.attempt_4_dir / "public_asset_metadata_report.json"
        ),
        runtime=load_json(args.attempt_4_dir / "runtime_environment_report.json"),
        unmanifested_assets=load_json(
            args.attempt_6_dir / "unmanifested_source_asset_report.json"
        ),
        started_at=started_at,
        process_id=os.getpid(),
        manifest_metadata=(
            load_json(args.manifest_metadata_report)
            if args.manifest_metadata_report is not None
            else None
        ),
        public_model_payload=(
            load_json(args.public_model_payload_report)
            if args.public_model_payload_report is not None
            else None
        ),
    )
    names = [
        "qualification_report.json",
        "laion_branch_qualification.json",
        "external_branch_qualification.json",
    ]
    for name, report in zip(names, reports):
        atomic_json(args.output_dir / name, report)
    print(
        json.dumps(
            {
                "event": "ep21_qualification_consolidation_finished",
                "pid": os.getpid(),
                "attempt": args.attempt,
                "outcome": reports[0]["qualification_outcome"],
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
