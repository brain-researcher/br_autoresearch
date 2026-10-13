#!/usr/bin/env python3
"""Find hard-coded or unpinned EP21 source-pipeline dependencies."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import verified_source_archive as source_archive_verifier


ARCHIVE_MD5 = "bb80dcc1a038737d6d8fc2c95b3e5586"
ARCHIVE_SIZE_BYTES = 23_110_908
ROOT = "KamitaniLab-InterSiteNeuralCodeConversion-edbff02/"
IDENTIFICATION_SCRIPT = (
    ROOT + "evaluation/identification_accuracy/recon_image_eval_dnn.py"
)
MAIN_ENV = ROOT + "env.yaml"
MAIN_MANIFEST = ROOT + "data/files.json"
RECON_MANIFEST = ROOT + "reconstruction/data/files.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def line_evidence(text: str, pattern: str) -> list[dict[str, Any]]:
    regex = re.compile(pattern)
    return [
        {"line": index, "text": line.strip()}
        for index, line in enumerate(text.splitlines(), start=1)
        if regex.search(line)
    ]


def manifest_names(*manifests: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for manifest in manifests:
        for group in manifest.values():
            names.update(item["name"] for item in group["files"])
    return names


def build_report(
    attempt: int,
    process_id: int,
    started_at: str,
    script_text: str,
    environment_text: str,
    declared_names: set[str],
) -> dict[str, Any]:
    alexnet_evidence = line_evidence(
        script_text, r"bvlc_alexnet\.pt|encoder_name == \"AlexNet\"|AlexNet\(\)"
    )
    vgg_evidence = line_evidence(
        script_text, r"VGG_ILSVRC_19_layers\.pt|encoder_name == \"VGG19\"|VGG19\(\)"
    )
    true_image_evidence = line_evidence(script_text, r"data/test_image/source")
    bdpy_entries = line_evidence(environment_text, r"^\s*-\s+bdpy(?:\s|$|=)")
    alexnet_manifested = any("alexnet" in name.lower() for name in declared_names)
    exact_vgg_checkpoint_manifested = "VGG_ILSVRC_19_layers.pt" in declared_names

    return {
        "schema_version": "ep21.unmanifested_source_asset_report.v1",
        "astra_analysis_id": "source_qualification",
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": process_id,
        "scope": "pinned_public_source_code_and_manifests_only",
        "identification_script": "evaluation/identification_accuracy/recon_image_eval_dnn.py",
        "gaps": {
            "alexnet_identification_checkpoint": {
                "state": "required_unmanifested",
                "hard_coded_path": "/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt",
                "evidence": alexnet_evidence,
                "any_alexnet_named_manifest_asset": alexnet_manifested,
                "caffenet_archive_equivalence": "unproven_names_state_dict_layer_map_preprocessing_and_hash_binding_absent",
                "affected_endpoints": ["finding_c", "finding_e"],
                "required_action": "Authenticate and provision the exact AlexNet checkpoint and preprocessing/layer mapping before identification may be runnable; otherwise lock affected findings not_evaluable.",
            },
            "vgg19_identification_checkpoint_binding": {
                "state": "candidate_archive_metadata_verified_payload_binding_unproven",
                "hard_coded_path": "/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt",
                "evidence": vgg_evidence,
                "exact_checkpoint_filename_in_manifest": exact_vgg_checkpoint_manifested,
                "candidate_archive": "VGG_ILSVRC_19_layers.zip",
                "required_action": "Download and checksum the declared public archive in an authorized compute workflow, then verify the exact checkpoint member, state-dict identity, layer mapping, and preprocessing before use.",
            },
            "source_true_image_directory": {
                "state": "source_study_licensed_stimulus_path_not_EP21_input",
                "hard_coded_path": "../../data/test_image/source/",
                "evidence": true_image_evidence,
                "ep21_action": "Do not acquire source-study true images as an EP21 substitute; use only an authorized regular-only LAION feature/stimulus handoff after its access gates pass.",
            },
            "main_bdpy_dependency": {
                "state": "declared_but_unpinned",
                "evidence": bdpy_entries,
                "reconstruction_environment_note": "The separate reconstruction environment pins bdpy==0.25, but the main converter/evaluation environment does not.",
                "required_action": "Authenticate the bdpy revision/API compatibility used by the tagged pipeline before the runtime environment can be locked.",
            },
        },
        "absolute_external_path_scan_complete_for_identification_script": True,
        "payload_bytes_downloaded": 0,
        "empirical_or_decoded_payload_accessed": False,
        "outcome": "completed_with_verified_absent_and_unknown_assets",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started_at = utc_now()
    print(
        json.dumps(
            {
                "event": "ep21_unmanifested_asset_audit_started",
                "pid": os.getpid(),
                "attempt": args.attempt,
                "started_at": started_at,
            }
        ),
        flush=True,
    )
    with source_archive_verifier.verified_zip(
        args.archive, ARCHIVE_MD5, ARCHIVE_SIZE_BYTES
    ) as verified:
        observed_md5 = verified.md5
        archive = verified.archive
        script_text = archive.read(IDENTIFICATION_SCRIPT).decode("utf-8")
        environment_text = archive.read(MAIN_ENV).decode("utf-8")
        declared_names = manifest_names(
            json.loads(archive.read(MAIN_MANIFEST)),
            json.loads(archive.read(RECON_MANIFEST)),
        )
    report = build_report(
        attempt=args.attempt,
        process_id=os.getpid(),
        started_at=started_at,
        script_text=script_text,
        environment_text=environment_text,
        declared_names=declared_names,
    )
    report["archive_md5"] = observed_md5
    atomic_json(args.output_dir / "unmanifested_source_asset_report.json", report)
    print(
        json.dumps(
            {
                "event": "ep21_unmanifested_asset_audit_finished",
                "pid": os.getpid(),
                "attempt": args.attempt,
                "outcome": report["outcome"],
                "payload_bytes_downloaded": 0,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
