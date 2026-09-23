#!/usr/bin/env python3
"""Trusted, outcome-blind role materializer for Dudman Episode 02.

The provider MAT contains development and audit animals in one compressed
variable.  This program is therefore a *trusted operator* utility, never a
candidate/search utility.  It verifies the frozen source and cohort map,
loads the mixed source once, and writes deterministic, field-preserving NPY
trees into three physically separate handoffs:

* a candidate-readable development pack;
* a primary-audit evaluator pack; and
* a post-primary boundary evaluator pack.

Only structural commitments are copied into the candidate-readable audit
manifest.  The program never computes or emits behavioral, neural, latency,
model-fit, or intervention-effect summaries.  Unix modes provide an
accidental-access guard, not a security boundary when source, candidate, and
evaluator all share the same Unix identity; that limitation is recorded in
the public receipt and remains a launch blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.io import loadmat, whosmat


SCHEMA_VERSION = "ep02.dudman.role_materialization.v1"
PACK_FORMAT = "deterministic_npy_tree_v1"
EXPECTED_FIELDS = (
    "trialID",
    "seshID",
    "lickState",
    "stimState",
    "latency",
    "cueLatency",
    "nac",
    "vta",
    "ds",
    "pupil",
    "lick",
    "nose",
    "whisk",
    "body",
    "iti",
    "predVars",
    "cuePredVars",
    "rewDA",
    "prepLick",
    "baseLick",
)

ROLE_LAYOUT = {
    "development_full": {
        "cohorts": ("development_control",),
        "visibility": "candidate_readable",
    },
    "primary_audit_evaluator_only": {
        "cohorts": (
            "audit_calibrated_stim_lick_minus",
            "audit_calibrated_stim_lick_plus",
        ),
        "visibility": "trusted_evaluator_only_after_configuration_lock",
    },
    "post_primary_boundary_evaluator_only": {
        "cohorts": ("boundary_supraphysiological_stim_lick_plus",),
        "visibility": (
            "trusted_evaluator_internal_stage_after_primary_subpacket_commit_"
            "before_feedback_in_same_logical_opening"
        ),
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_regular_bytes(path: Path) -> bytes:
    """Read one non-symlink regular file and reject descriptor changes."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or len(payload) != before.st_size:
            raise ValueError(f"File changed while being read: {path}")
        return payload
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o440) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any, mode: int = 0o440) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload), mode=mode)


def npy_payload(array: np.ndarray) -> bytes:
    """Return a pickle-free, C-order NPY preserving shape, dtype, and bits."""

    value = np.ascontiguousarray(np.asarray(array))
    if value.dtype.hasobject:
        raise TypeError("Object arrays are forbidden in role packs")
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def verify_npy_payload(payload: bytes, expected: np.ndarray, field: str) -> None:
    """Perform a bit-exact trusted round trip without emitting any values."""

    observed = np.load(io.BytesIO(payload), allow_pickle=False)
    source = np.ascontiguousarray(np.asarray(expected))
    if observed.dtype != source.dtype or observed.shape != source.shape:
        raise ValueError(f"Round-trip schema mismatch for {field}")
    if observed.tobytes(order="C") != source.tobytes(order="C"):
        raise ValueError(f"Round-trip bit-pattern mismatch for {field}")


def load_cohort_map(payload_bytes: bytes, expected_source_sha256: str) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    payload = json.loads(payload_bytes.decode("utf-8"))
    if payload.get("source_index_base") != 1:
        raise ValueError("Cohort map must use one-based indices")
    if payload.get("source_payload_sha256") != expected_source_sha256:
        raise ValueError("Cohort map is not bound to the requested source SHA-256")

    lookup: dict[int, dict[str, str]] = {}
    for cohort in payload.get("cohorts", []):
        for source_index in cohort["source_records_1_based"]:
            if source_index in lookup:
                raise ValueError(f"Duplicate source record {source_index}")
            lookup[source_index] = {
                "cohort_id": cohort["cohort_id"],
                "role": cohort["role"],
            }
    if set(lookup) != set(range(1, 25)):
        raise ValueError("Cohort map must cover source records 1..24 exactly")
    if {item["role"] for item in lookup.values()} != set(ROLE_LAYOUT):
        raise ValueError("Cohort map roles differ from the frozen role layout")
    return payload, lookup


def require_new_directory(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists; refusing to overwrite: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError(f"{label} parent may not be a symlink: {parent}")


def role_records(lookup: dict[int, dict[str, str]], role: str) -> list[int]:
    return [index for index in sorted(lookup) if lookup[index]["role"] == role]


def write_role_pack(
    temporary: Path,
    sessions: np.ndarray,
    lookup: dict[int, dict[str, str]],
    role: str,
    observed_on: str,
    source_sha256: str,
    cohort_map_sha256: str,
    materializer_sha256: str,
) -> dict[str, Any]:
    temporary.mkdir(mode=0o700)
    entries: list[dict[str, Any]] = []
    for source_index in role_records(lookup, role):
        row = sessions[0, source_index - 1]
        fields = {field: np.asarray(row[field]) for field in EXPECTED_FIELDS}
        opaque_id = f"source_record_{source_index:02d}"
        record_dir = temporary / opaque_id
        record_dir.mkdir(mode=0o750)
        schemas: dict[str, Any] = {}
        for field in EXPECTED_FIELDS:
            value = np.ascontiguousarray(fields[field])
            payload = npy_payload(value)
            verify_npy_payload(payload, value, field)
            relative = f"{opaque_id}/{field}.npy"
            atomic_write_bytes(record_dir / f"{field}.npy", payload, mode=0o440)
            schemas[field] = {
                "relative_path": relative,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "npy_bytes": len(payload),
                "npy_sha256": sha256_bytes(payload),
            }
        os.chmod(record_dir, 0o550)
        record_root = sha256_bytes(
            b"".join(
                field.encode("utf-8")
                + b"\0"
                + schemas[field]["npy_sha256"].encode("ascii")
                + b"\n"
                for field in EXPECTED_FIELDS
            )
        )
        entries.append(
            {
                "opaque_animal_id": opaque_id,
                "source_index_1_based": source_index,
                "cohort_id": lookup[source_index]["cohort_id"],
                "record_root_sha256": record_root,
                "fields": schemas,
            }
        )

    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "pack_format": PACK_FORMAT,
        "field_policy": {
            "design_fields": ["trialID", "seshID", "lickState", "stimState"],
            "released_outcome_or_covariate_fields": [
                "latency",
                "cueLatency",
                "nac",
                "vta",
                "ds",
                "pupil",
                "lick",
                "nose",
                "whisk",
                "body",
                "iti",
            ],
            "provider_derived_fields": [
                "predVars",
                "cuePredVars",
                "rewDA",
                "prepLick",
                "baseLick",
            ],
            "packaging_is_analysis_authorization": False,
        },
        "observed_on": observed_on,
        "role": role,
        "visibility": ROLE_LAYOUT[role]["visibility"],
        "source_payload_sha256": source_sha256,
        "cohort_map_sha256": cohort_map_sha256,
        "materializer_sha256": materializer_sha256,
        "record_count": len(entries),
        "record_entries": entries,
        "outcome_summaries_emitted": False,
        "round_trip_verified": True,
    }
    manifest = {
        **manifest_core,
        "pack_root_sha256": sha256_bytes(canonical_json_bytes(manifest_core)),
    }
    atomic_write_json(temporary / "MANIFEST.json", manifest, mode=0o440)
    # Keep the pack root owner-writable until its same-filesystem rename.
    # Stanford OAK rejects moving a non-writable directory even when both
    # parents are writable.  The final root is made read-only after publish.
    os.chmod(temporary, 0o750)
    return manifest


def compact_role_commitment(manifest: dict[str, Any]) -> dict[str, Any]:
    cohorts: dict[str, int] = {}
    schema_signatures: set[str] = set()
    for entry in manifest["record_entries"]:
        cohorts[entry["cohort_id"]] = cohorts.get(entry["cohort_id"], 0) + 1
        schema_only = {
            field: {"shape": item["shape"], "dtype": item["dtype"]}
            for field, item in entry["fields"].items()
        }
        schema_signatures.add(sha256_bytes(canonical_json_bytes(schema_only)))
    return {
        "role": manifest["role"],
        "visibility": manifest["visibility"],
        "record_count": manifest["record_count"],
        "cohort_counts": dict(sorted(cohorts.items())),
        "pack_root_sha256": manifest["pack_root_sha256"],
        "distinct_schema_signature_count": len(schema_signatures),
        "individual_file_paths_or_hashes_disclosed": manifest["role"] == "development_full",
    }


def stage_unsealed_audit_directory(path: Path) -> None:
    """Apply an accidental-access guard; this is explicitly not a security seal."""

    children = list(path.rglob("*"))
    for child in children:
        if child.is_file():
            os.chmod(child, 0)
    for child in sorted((item for item in children if item.is_dir()), reverse=True):
        os.chmod(child, 0)
    os.chmod(path, 0)


def public_manifest(
    args: argparse.Namespace,
    source_sha256: str,
    cohort_map_sha256: str,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    development = manifests["development_full"]
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_on": args.observed_on,
        "episode_id": "ep02_dopamine_learning_rate_causality",
        "source_commitment": {
            "bytes": args.expected_bytes,
            "md5": args.expected_md5,
            "sha256": source_sha256,
        },
        "cohort_map_sha256": cohort_map_sha256,
        "materializer_sha256": manifests["development_full"]["materializer_sha256"],
        "pack_format": PACK_FORMAT,
        "field_policy": manifests["development_full"]["field_policy"],
        "role_commitments": {
            role: compact_role_commitment(manifest)
            for role, manifest in manifests.items()
        },
        "development_records": [
            {
                "opaque_animal_id": item["opaque_animal_id"],
                "source_index_1_based": item["source_index_1_based"],
                "cohort_id": item["cohort_id"],
                "record_root_sha256": item["record_root_sha256"],
            }
            for item in development["record_entries"]
        ],
        "audit_structural_disclosure": {
            "individual_outcome_arrays_disclosed": False,
            "individual_file_paths_or_hashes_disclosed": False,
            "behavioral_or_neural_magnitudes_disclosed": False,
            "cohort_counts_and_role_roots_only": True,
        },
        "gates": {
            "handoff_state": "materialized_unsealed",
            "materialization_integrity_status": "PASS",
            "confidentiality_firewall_status": (
                "BLOCKED_SAME_UID_SOURCE_READABLE_AND_NETWORK_REACQUISITION"
            ),
            "materialization_complete": True,
            "source_and_roles_hash_bound": True,
            "round_trip_verified": True,
            "unix_mode_accidental_access_guard_applied": True,
            "different_unix_or_service_principal_used": False,
            "original_mixed_source_unavailable_to_candidate_uid": False,
            "candidate_network_egress_disabled": False,
            "physical_audit_firewall_complete": False,
            "launch_authorized": False,
        },
        "firewall_limitation": (
            "Candidate, trusted builder, evaluator packs, and the owner-readable mixed source "
            "share one Unix uid. Mode 000 audit directories deter accidental reads inside the "
            "bounded launcher but do not create adversarial isolation; a separate principal or "
            "external evaluator/source service remains required before launch. The candidate "
            "must also have no network egress because the mixed source is publicly downloadable."
        ),
        "audit_outcome_summaries_emitted": False,
    }


def render_report(manifest: dict[str, Any]) -> str:
    commitments = manifest["role_commitments"]
    rows = []
    for role in ROLE_LAYOUT:
        item = commitments[role]
        rows.append(
            f"| `{role}` | {item['record_count']} | `{item['pack_root_sha256']}` | "
            f"{item['visibility']} |"
        )
    return f"""# Trusted role materialization

## Result

The mixed Dudman MAT was split by the frozen, source-index cohort map into a
candidate-readable development pack and two independently committed evaluator
packs.  Every record was round-tripped field-by-field with dtype, shape, NaN,
and value equality checked inside the trusted process.  No outcome summary or
intervention contrast was calculated or emitted.

| Role | Mice | Pack-root SHA-256 | Visibility |
| --- | ---: | --- | --- |
{chr(10).join(rows)}

The public audit-structural handoff exposes only role/cohort counts, schema
signature counts, and role-level hash commitments.  Individual evaluator
field paths, field hashes, and arrays remain in the staged evaluator
handoffs.

## Security boundary

This step creates deterministic handoffs and an accidental-access guard, not
a completed physical audit firewall.  The source, builder, and candidate all
run as the same Unix uid, and the original mixed MAT remains owner-readable.
The evaluator directories were set to mode `0000`, but their owner could
restore access outside the candidate sandbox.  Before launch, an independent
Unix/service principal or external evaluator/key service must make both the
audit packs **and the original mixed source** unavailable to candidate workers,
and the candidate must have no network path to reacquire the public payload.

Consequently `physical_audit_firewall_complete=false` and
`launch_authorized=false` remain frozen in the public manifest.
"""


def validate_path_separation(paths: Iterable[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Materialization destinations must be distinct")
    for left in resolved:
        for right in resolved:
            if left == right:
                continue
            if left in right.parents or right in left.parents:
                raise ValueError(f"Materialization destinations may not nest: {left} and {right}")


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_argument = args.mat.absolute()
    cohort_map_argument = args.cohort_map.absolute()
    if source_argument.is_symlink() or cohort_map_argument.is_symlink():
        raise ValueError("Source and cohort map must not be symlinks")
    source = source_argument.resolve(strict=True)
    cohort_map_path = cohort_map_argument.resolve(strict=True)
    development_dir = args.development_dir.resolve(strict=False)
    audit_structural_dir = args.audit_structural_dir.resolve(strict=False)
    vault_root = args.vault_root.resolve(strict=False)
    public_output_dir = args.public_output_dir.resolve(strict=False)

    validate_path_separation(
        [development_dir, audit_structural_dir, vault_root, public_output_dir]
    )
    if not source.is_file():
        raise FileNotFoundError(f"Source must be a regular, non-symlink file: {source}")
    if source.stat().st_size != args.expected_bytes:
        raise ValueError("Source byte count differs from the frozen value")
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    source_stream = os.fdopen(source_fd, "rb")
    before = os.fstat(source_stream.fileno())
    if not stat.S_ISREG(before.st_mode):
        source_stream.close()
        raise ValueError("Source descriptor is not a regular file")
    md5_digest = hashlib.md5()
    sha256_digest = hashlib.sha256()
    for block in iter(lambda: source_stream.read(8 * 1024 * 1024), b""):
        md5_digest.update(block)
        sha256_digest.update(block)
    observed_md5 = md5_digest.hexdigest()
    observed_sha256 = sha256_digest.hexdigest()
    if observed_md5 != args.expected_md5:
        source_stream.close()
        raise ValueError("Source MD5 differs from the frozen provider value")
    if observed_sha256 != args.expected_sha256:
        source_stream.close()
        raise ValueError("Source SHA-256 differs from the frozen value")
    cohort_map_bytes = read_regular_bytes(cohort_map_path)
    cohort_map_sha256 = sha256_bytes(cohort_map_bytes)
    if cohort_map_sha256 != args.expected_cohort_map_sha256:
        source_stream.close()
        raise ValueError("Cohort-map SHA-256 differs from the frozen value")

    try:
        cohort_payload, lookup = load_cohort_map(cohort_map_bytes, observed_sha256)
    except Exception:
        source_stream.close()
        raise
    expected_whos = [("seshMerge", (1, 24), "struct")]
    source_stream.seek(0)
    if whosmat(source_stream) != expected_whos:
        source_stream.close()
        raise ValueError("Unexpected MAT top-level layout")

    require_new_directory(development_dir, "development pack")
    require_new_directory(audit_structural_dir, "audit-structural pack")
    require_new_directory(vault_root, "evaluator vault")
    require_new_directory(public_output_dir, "public materialization packet")

    source_stream.seek(0)
    loaded = loadmat(
        source_stream,
        variable_names=["seshMerge"],
        struct_as_record=True,
        squeeze_me=False,
    )
    after = os.fstat(source_stream.fileno())
    source_stream.close()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError("Source descriptor metadata changed during materialization")
    sessions = loaded["seshMerge"]
    if sessions.shape != (1, 24) or tuple(sessions.dtype.names or ()) != EXPECTED_FIELDS:
        raise ValueError("Loaded MAT schema differs from the frozen schema")

    dev_temp = Path(tempfile.mkdtemp(prefix=".development_full.", dir=development_dir.parent))
    vault_temp = Path(tempfile.mkdtemp(prefix=".evaluator_vault.", dir=vault_root.parent))
    structural_temp = Path(
        tempfile.mkdtemp(prefix=".audit_structural.", dir=audit_structural_dir.parent)
    )
    public_temp = Path(
        tempfile.mkdtemp(prefix=".role_materialization.", dir=public_output_dir.parent)
    )
    manifests: dict[str, dict[str, Any]] = {}
    materializer_sha256 = file_digest(Path(__file__).resolve())
    finalized = False
    published: list[Path] = []
    try:
        # mkdtemp creates the root; write_role_pack expects to create its target.
        dev_role_temp = dev_temp / "pack"
        manifests["development_full"] = write_role_pack(
            dev_role_temp,
            sessions,
            lookup,
            "development_full",
            args.observed_on,
            observed_sha256,
            cohort_map_sha256,
            materializer_sha256,
        )
        for role in (
            "primary_audit_evaluator_only",
            "post_primary_boundary_evaluator_only",
        ):
            manifests[role] = write_role_pack(
                vault_temp / role,
                sessions,
                lookup,
                role,
                args.observed_on,
                observed_sha256,
                cohort_map_sha256,
                materializer_sha256,
            )

        public = public_manifest(
            args,
            observed_sha256,
            cohort_map_sha256,
            manifests,
        )
        public["cohort_map_schema_version"] = cohort_payload.get("schema_version")
        public["runtime"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "zlib": getattr(__import__("zlib"), "ZLIB_VERSION"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }

        atomic_write_json(structural_temp / "PUBLIC_ROLE_MANIFEST.json", public)
        atomic_write_json(public_temp / "PUBLIC_ROLE_MANIFEST.json", public)
        atomic_write_bytes(
            public_temp / "MATERIALIZATION.md",
            render_report(public).encode("utf-8"),
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "observed_on": args.observed_on,
            "trusted_operator_utility": True,
            "mixed_source_materialized_in_memory": True,
            "source_values_used_only_for_lossless_serialization_and_round_trip": True,
            "behavioral_or_neural_magnitudes_summarized": False,
            "intervention_effect_computed": False,
            "handoff_state": "materialized_unsealed",
            "materialization_integrity_status": "PASS",
            "confidentiality_firewall_status": (
                "BLOCKED_SAME_UID_SOURCE_READABLE_AND_NETWORK_REACQUISITION"
            ),
            "audit_directories_mode_after_seal": "0000",
            "vault_root_mode_after_seal": "0100",
            "same_uid_adversarial_isolation_achieved": False,
            "original_mixed_source_candidate_inaccessible": False,
            "launch_authorized": False,
        }
        atomic_write_json(public_temp / "ACCESS_RECEIPT.json", receipt)
        manifested = sorted(
            item
            for item in public_temp.iterdir()
            if item.is_file() and item.name != "SHA256SUMS"
        )
        sums = "".join(f"{file_digest(item)}  {item.name}\n" for item in manifested)
        atomic_write_bytes(public_temp / "SHA256SUMS", sums.encode("ascii"))

        for role in (
            "primary_audit_evaluator_only",
            "post_primary_boundary_evaluator_only",
        ):
            stage_unsealed_audit_directory(vault_temp / role)
        os.chmod(vault_temp, 0o700)
        os.chmod(structural_temp, 0o750)
        os.chmod(public_temp, 0o770)

        # Every final target was proven absent.  Prepare and verify all four
        # trees before any rename, then publish same-filesystem trees.
        dev_role_temp.replace(development_dir)
        published.append(development_dir)
        os.chmod(development_dir, 0o550)
        dev_temp.rmdir()
        vault_temp.replace(vault_root)
        published.append(vault_root)
        os.chmod(vault_root, 0o100)
        structural_temp.replace(audit_structural_dir)
        published.append(audit_structural_dir)
        os.chmod(audit_structural_dir, 0o550)
        public_temp.replace(public_output_dir)
        published.append(public_output_dir)
        finalized = True
        return public
    finally:
        if not finalized:
            # Only remove paths created by this invocation.  Final targets were
            # all absent on entry, so this cannot overwrite or delete prior data.
            for path in reversed(published):
                if path.exists():
                    try:
                        os.chmod(path, 0o700)
                    except OSError:
                        pass
                    for child in path.rglob("*"):
                        try:
                            os.chmod(child, 0o700 if child.is_dir() else 0o600)
                        except OSError:
                            pass
                    shutil.rmtree(path)
            for temporary in (dev_temp, vault_temp, structural_temp, public_temp):
                if temporary.exists():
                    try:
                        os.chmod(temporary, 0o700)
                    except OSError:
                        pass
                    for child in temporary.rglob("*"):
                        try:
                            os.chmod(child, 0o700 if child.is_dir() else 0o600)
                        except OSError:
                            pass
                    shutil.rmtree(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--cohort-map", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--audit-structural-dir", type=Path, required=True)
    parser.add_argument("--vault-root", type=Path, required=True)
    parser.add_argument("--public-output-dir", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-cohort-map-sha256", required=True)
    parser.add_argument("--observed-on", required=True)
    return parser.parse_args()


def main() -> int:
    materialize(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
