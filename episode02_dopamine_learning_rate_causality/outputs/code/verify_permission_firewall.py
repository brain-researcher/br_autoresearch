#!/usr/bin/env python3
"""Verify the EP02 permission-separated firewall without reading outcome data.

The verifier deliberately does not treat chmod, an Apptainer namespace, or a
Slurm allocation running as the same Unix uid as a confidentiality boundary.
A PASS requires a receipt signed by independently pinned infrastructure and
data-steward keys.  The current local probe can only produce BLOCKED.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "ep02.dudman.permission_firewall_assessment.v1"
REQUIRED_CHECKS = (
    "separate_principals",
    "candidate_source_denial",
    "candidate_evaluator_pack_denial",
    "candidate_network_egress_denial",
    "evaluator_network_egress_denial",
    "evaluator_only_mount",
    "atomic_no_partial_output",
)
REQUIRED_SIGNER_ROLES = ("infrastructure_operator", "data_steward")
HEX_SHA256_LENGTH = 64
ED25519_SUBJECT_PUBLIC_KEY_INFO_PREFIX = bytes.fromhex("302a300506032b6570032100")


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HEX_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_contract(contract: Any) -> list[str]:
    failures: list[str] = []
    if not isinstance(contract, dict):
        return ["contract_not_object"]
    if contract.get("schema_version") != "ep02.dudman.permission_firewall_contract.v1":
        failures.append("unsupported_contract_schema")
    if contract.get("contract_id") != "ep02_permission_separated_firewall_v1":
        failures.append("contract_id_mismatch")
    if contract.get("episode_id") != "ep02":
        failures.append("episode_id_mismatch")
    if contract.get("current_environment_status") != "PROVISIONED_EXTERNALLY_ATTESTED":
        failures.append("environment_not_provisioned")
    if contract.get("trusted_signers_status") != "PINNED_AND_REVIEWED":
        failures.append("trusted_signers_not_pinned_and_reviewed")
    if contract.get("same_uid_controls_are_sufficient") is not False:
        failures.append("contract_must_reject_same_uid_controls")
    required_checks = contract.get("required_checks")
    if not isinstance(required_checks, (list, tuple)) or tuple(required_checks) != REQUIRED_CHECKS:
        failures.append("required_checks_mismatch")
    required_signer_roles = contract.get("required_signer_roles")
    if not isinstance(required_signer_roles, (list, tuple)) or tuple(
        required_signer_roles
    ) != REQUIRED_SIGNER_ROLES:
        failures.append("required_signer_roles_mismatch")
    required_distinct_principals = contract.get("required_distinct_principals")
    if not isinstance(required_distinct_principals, (list, tuple)) or tuple(
        required_distinct_principals
    ) != (
        "candidate",
        "evaluator",
        "steward",
    ):
        failures.append("required_distinct_principals_mismatch")
    for field in (
        "source_sha256",
        "primary_audit_pack_root_sha256",
        "policy_sha256",
        "configuration_lock_sha256",
    ):
        if not is_sha256(contract.get(field)):
            failures.append(f"contract_commitment_invalid:{field}")
    validity_seconds = contract.get("receipt_validity_seconds_max")
    if not isinstance(validity_seconds, int) or not 0 < validity_seconds <= 86400:
        failures.append("receipt_validity_seconds_max_invalid")
    clock_skew_seconds = contract.get("clock_skew_seconds_max")
    if not isinstance(clock_skew_seconds, int) or not 0 <= clock_skew_seconds <= 300:
        failures.append("clock_skew_seconds_max_invalid")
    pinned = contract.get("trusted_signers")
    if not isinstance(pinned, dict):
        failures.append("trusted_signers_not_object")
        pinned = {}
    signer_ids: list[str] = []
    fingerprints: list[str] = []
    for role in REQUIRED_SIGNER_ROLES:
        entry = pinned.get(role)
        if not isinstance(entry, dict):
            failures.append(f"trusted_signer_not_pinned:{role}")
            continue
        if not isinstance(entry.get("signer_id"), str) or not entry["signer_id"]:
            failures.append(f"trusted_signer_id_invalid:{role}")
        else:
            signer_ids.append(entry["signer_id"])
        fingerprint = entry.get("ed25519_public_key_der_sha256")
        if not is_sha256(fingerprint):
            failures.append(f"trusted_signer_fingerprint_invalid:{role}")
        else:
            fingerprints.append(fingerprint)
    if len(signer_ids) == len(REQUIRED_SIGNER_ROLES) and len(set(signer_ids)) != len(
        REQUIRED_SIGNER_ROLES
    ):
        failures.append("trusted_signer_ids_not_distinct")
    if len(fingerprints) == len(REQUIRED_SIGNER_ROLES) and len(set(fingerprints)) != len(
        REQUIRED_SIGNER_ROLES
    ):
        failures.append("trusted_signer_keys_not_distinct")
    return failures


def _public_key_der_sha256(openssl: str, public_key_pem: bytes) -> str | None:
    with tempfile.TemporaryDirectory(prefix="ep02-firewall-key-") as temporary:
        key_path = Path(temporary) / "key.pem"
        der_path = Path(temporary) / "key.der"
        key_path.write_bytes(public_key_pem)
        completed = subprocess.run(
            [
                openssl,
                "pkey",
                "-pubin",
                "-in",
                str(key_path),
                "-outform",
                "DER",
                "-out",
                str(der_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or not der_path.exists():
            return None
        der = der_path.read_bytes()
        # RFC 8410 Ed25519 SubjectPublicKeyInfo is exactly the fixed algorithm
        # identifier plus a 32-byte raw public key. Merely accepting any key
        # that `openssl pkey` can parse would also admit RSA or EC keys while
        # the contract specifically requires Ed25519.
        if len(der) != 44 or not der.startswith(ED25519_SUBJECT_PUBLIC_KEY_INFO_PREFIX):
            return None
        return sha256_bytes(der)


def verify_ed25519_attestation(
    openssl: str,
    payload: bytes,
    attestation: dict[str, Any],
    expected_fingerprint: str,
) -> bool:
    try:
        public_key = attestation["public_key_pem"].encode("ascii")
        signature = base64.b64decode(attestation["signature_base64"], validate=True)
    except (KeyError, UnicodeError, ValueError, TypeError):
        return False
    if _public_key_der_sha256(openssl, public_key) != expected_fingerprint:
        return False
    with tempfile.TemporaryDirectory(prefix="ep02-firewall-signature-") as temporary:
        directory = Path(temporary)
        payload_path = directory / "payload.json"
        key_path = directory / "key.pem"
        signature_path = directory / "signature.bin"
        payload_path.write_bytes(payload)
        key_path.write_bytes(public_key)
        signature_path.write_bytes(signature)
        completed = subprocess.run(
            [
                openssl,
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey",
                str(key_path),
                "-sigfile",
                str(signature_path),
                "-in",
                str(payload_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed.returncode == 0


SignatureChecker = Callable[[str, bytes, dict[str, Any], str], bool]


def assess_signed_bundle(
    contract: Any,
    bundle: Any,
    openssl: str,
    signature_checker: SignatureChecker = verify_ed25519_attestation,
    assessment_time: datetime | None = None,
) -> dict[str, Any]:
    failures = validate_contract(contract)
    contract_for_hash = contract
    if not isinstance(contract, dict):
        contract = {}
    if not isinstance(bundle, dict):
        failures.append("receipt_bundle_not_object")
        bundle = {}
    if bundle.get("schema_version") != "ep02.dudman.permission_firewall_receipt_bundle.v1":
        failures.append("unsupported_receipt_bundle_schema")
    if bundle.get("status") != "SIGNED":
        failures.append("receipt_bundle_not_signed")
    payload = bundle.get("receipt_payload")
    if not isinstance(payload, dict):
        failures.append("missing_receipt_payload")
        payload = {}
    expected_contract_hash = sha256_bytes(canonical_json_bytes(contract_for_hash))
    if payload.get("contract_sha256") != expected_contract_hash:
        failures.append("contract_hash_mismatch")
    for field in (
        "episode_id",
        "source_sha256",
        "primary_audit_pack_root_sha256",
        "policy_sha256",
        "configuration_lock_sha256",
    ):
        if payload.get(field) != contract.get(field):
            failures.append(f"{field}_mismatch")

    principals = payload.get("principals")
    if not isinstance(principals, dict):
        failures.append("principals_not_object")
        principals = {}
    principal_ids = [principals.get(role) for role in ("candidate", "evaluator", "steward")]
    if any(not isinstance(item, str) or not item.strip() for item in principal_ids):
        failures.append("principal_identity_missing")
    elif len(set(principal_ids)) != 3:
        failures.append("principals_not_distinct")

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        failures.append("checks_not_object")
        checks = {}
    for check in REQUIRED_CHECKS:
        item = checks.get(check)
        if not isinstance(item, dict) or item.get("status") != "PASS":
            failures.append(f"check_not_passed:{check}")
        elif not isinstance(item.get("external_evidence_id"), str) or not item[
            "external_evidence_id"
        ].strip():
            failures.append(f"external_evidence_missing:{check}")

    candidate_runtime_id = payload.get("candidate_runtime_id")
    evaluator_runtime_id = payload.get("evaluator_runtime_id")
    if not isinstance(candidate_runtime_id, str) or not candidate_runtime_id.strip():
        failures.append("candidate_runtime_id_missing")
    if not isinstance(evaluator_runtime_id, str) or not evaluator_runtime_id.strip():
        failures.append("evaluator_runtime_id_missing")
    observed_at = parse_utc_timestamp(payload.get("observed_at_utc"))
    expires_at = parse_utc_timestamp(payload.get("expires_at_utc"))
    if observed_at is None:
        failures.append("observation_time_invalid")
    if expires_at is None:
        failures.append("expiry_time_invalid")
    if observed_at is not None and expires_at is not None:
        validity_seconds = contract.get("receipt_validity_seconds_max")
        if not isinstance(validity_seconds, int):
            validity_seconds = 0
        maximum_validity = timedelta(seconds=max(validity_seconds, 0))
        if expires_at <= observed_at:
            failures.append("receipt_expiry_not_after_observation")
        elif expires_at - observed_at > maximum_validity:
            failures.append("receipt_validity_window_too_long")
        now = assessment_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        clock_skew_seconds = contract.get("clock_skew_seconds_max")
        if not isinstance(clock_skew_seconds, int):
            clock_skew_seconds = 0
        skew = timedelta(seconds=max(clock_skew_seconds, 0))
        if observed_at > now + skew:
            failures.append("receipt_observation_in_future")
        if expires_at < now - skew:
            failures.append("receipt_expired")

    attestations = bundle.get("attestations")
    if not isinstance(attestations, list):
        failures.append("attestations_not_array")
        attestations = []
    by_role: dict[str, dict[str, Any]] = {}
    role_counts: dict[str, int] = {}
    for item in attestations:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            continue
        role = item["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
        by_role[role] = item
    payload_bytes = canonical_json_bytes(payload)
    trusted = contract.get("trusted_signers")
    if not isinstance(trusted, dict):
        trusted = {}
    for role in REQUIRED_SIGNER_ROLES:
        attestation = by_role.get(role)
        signer = trusted.get(role)
        if role_counts.get(role, 0) != 1:
            failures.append(f"attestation_count_not_one:{role}")
        if not isinstance(attestation, dict) or not isinstance(signer, dict):
            failures.append(f"signed_attestation_missing:{role}")
            continue
        if attestation.get("signer_id") != signer.get("signer_id"):
            failures.append(f"signer_id_mismatch:{role}")
            continue
        if attestation.get("signer_id") == principals.get("candidate"):
            failures.append(f"candidate_cannot_attest:{role}")
            continue
        expected = signer.get("ed25519_public_key_der_sha256", "")
        if not signature_checker(openssl, payload_bytes, attestation, expected):
            failures.append(f"signature_invalid:{role}")

    return {
        "schema_version": SCHEMA_VERSION,
        "assessment_mode": "externally_signed_receipt_bundle",
        "status": "PASS" if not failures else "BLOCKED",
        "launch_authorized": not failures,
        "contract_sha256": expected_contract_hash,
        "receipt_payload_sha256": sha256_bytes(payload_bytes),
        "assessed_at_utc": (assessment_time or datetime.now(timezone.utc))
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "failures": sorted(set(failures)),
        "outcome_files_read": [],
        "same_uid_mode_bits_treated_as_firewall": False,
    }


def assess_current_environment(
    contract: dict[str, Any], source_path: Path | None
) -> dict[str, Any]:
    failures = validate_contract(contract)
    source_probe: dict[str, Any] = {
        "performed": source_path is not None,
        "content_read": False,
    }
    if source_path is not None:
        try:
            stat = source_path.stat()
            source_probe.update(
                {
                    "exists": True,
                    "owner_uid": stat.st_uid,
                    "current_euid": os.geteuid(),
                    "same_uid": stat.st_uid == os.geteuid(),
                    "candidate_readable_by_access_probe": os.access(source_path, os.R_OK),
                }
            )
            if os.access(source_path, os.R_OK):
                failures.append("candidate_can_read_mixed_source")
            if stat.st_uid == os.geteuid():
                failures.append("mixed_source_owned_by_candidate_uid")
        except FileNotFoundError:
            source_probe["exists"] = False
            failures.append("source_probe_target_missing_not_externally_attested_denial")
    failures.extend(
        [
            "external_permission_separation_receipt_missing",
            "external_network_policy_receipt_missing",
            "external_evaluator_mount_receipt_missing",
            "external_atomic_output_receipt_missing",
            "independent_signed_attestations_missing",
        ]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "assessment_mode": "local_noncontent_probe",
        "status": "BLOCKED",
        "launch_authorized": False,
        "contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
        "source_probe": source_probe,
        "failures": sorted(set(failures)),
        "outcome_files_read": [],
        "same_uid_mode_bits_treated_as_firewall": False,
        "note": (
            "A local process cannot self-attest its own confinement. PASS requires "
            "the externally signed receipt bundle defined by the contract."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt-bundle", type=Path)
    parser.add_argument("--probe-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--openssl", default=shutil.which("openssl") or "openssl")
    arguments = parser.parse_args()
    contract = json.loads(arguments.contract.read_text(encoding="utf-8"))
    if arguments.receipt_bundle:
        bundle = json.loads(arguments.receipt_bundle.read_text(encoding="utf-8"))
        assessment = assess_signed_bundle(contract, bundle, arguments.openssl)
    else:
        assessment = assess_current_environment(contract, arguments.probe_source)
    text = json.dumps(assessment, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        atomic_write_text(arguments.output, text)
    else:
        print(text, end="")
    return 0 if assessment["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
