"""Create a separate, outcome-blind attestation for EP12 observed trials.

The immutable v1 trial directories contain outcome-bearing tables and reports.
This module authenticates their bytes, but extracts only structural/status
metadata needed to prove that each declared development configuration ran a
complete T/U/M comparison without technical failure.  It never emits or
compares objectives, scores, deltas, intervals, or selected scientific values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy import EpisodePolicy, digest_bytes, digest_object


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
OUTPUTS_ROOT = EPISODE_ROOT / "outputs"

ATTESTATION_SCHEMA = "ep12.observed_trial_attestation_set.v1"
ATTESTATION_MANIFEST_SCHEMA = "ep12.observed_trial_attestation_set_manifest.v1"
SOURCE_MANIFEST_SCHEMA = "ep12.development_trial_manifest.v1"
SOURCE_REPORT_SCHEMA = "ep12.development_trial.v1"
SOURCE_FILES = {
    "result_table.csv",
    "trial_report.json",
    "type_direction_receipts.jsonl",
}

CHAIN_SOURCES = (
    (
        "coverage",
        18,
        Path("search_plan/coverage_configurations.json"),
        Path("observed"),
        None,
    ),
    (
        "adaptive_cycle_1",
        9,
        Path("adaptive/cycle1/plan/successor_configurations.json"),
        Path("adaptive/cycle1/observed"),
        1,
    ),
    (
        "adaptive_cycle_2",
        9,
        Path("adaptive/cycle2/plan/successor_configurations.json"),
        Path("adaptive/cycle2/observed"),
        2,
    ),
)


class ObservedAttestationError(ValueError):
    """A source trial or separate attestation is unsafe or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservedAttestationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservedAttestationError(f"{label} must be a JSON object")
    return value


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ObservedAttestationError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ObservedAttestationError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ObservedAttestationError(f"malformed {label}: {path}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise ObservedAttestationError(f"missing authenticated source: {path}") from exc
    return digest.hexdigest()


def scientific_configuration_sha256(configuration: Mapping[str, Any]) -> str:
    """Hash all declared configuration fields except identity and stage."""

    return digest_object(
        {
            key: value
            for key, value in configuration.items()
            if key not in {"trial_id", "stage"}
        }
    )


def _verify_plan_manifest(plan_dir: Path, configuration_name: str) -> str:
    manifest_path = plan_dir / "manifest.json"
    manifest = _mapping(_read_json(manifest_path, "plan manifest"), "plan manifest")
    _require(
        manifest.get("final_connectivity_accessed") is False,
        f"plan manifest records final-connectivity access: {plan_dir}",
    )
    files = _mapping(manifest.get("files"), "plan manifest files")
    _require(configuration_name in files, f"plan manifest omits {configuration_name}")
    for name, expected in files.items():
        _require(
            isinstance(name, str) and Path(name).name == name,
            f"unsafe plan artifact name: {name!r}",
        )
        _require(
            _sha256_file(plan_dir / name) == str(expected),
            f"plan artifact hash mismatch: {plan_dir / name}",
        )
    return _sha256_file(manifest_path)


def _load_configurations(path: Path, expected_cycle: int | None) -> list[dict[str, Any]]:
    payload = _read_json(path, "declared configuration file")
    if expected_cycle is None:
        _require(isinstance(payload, list), "coverage configurations must be a list")
        configurations = payload
        recorded_hashes = [digest_object(value) for value in configurations]
    else:
        container = _mapping(payload, "adaptive configuration container")
        _require(
            container.get("schema_version")
            == "ep12.adaptive_successor_configurations.v1",
            f"adaptive cycle {expected_cycle} configuration schema mismatch",
        )
        _require(
            container.get("cycle") == expected_cycle
            and container.get("final_connectivity_accessed") is False,
            f"adaptive cycle {expected_cycle} identity or firewall mismatch",
        )
        configurations = container.get("configurations")
        recorded_hashes = container.get("configuration_sha256")
        _require(
            isinstance(configurations, list)
            and isinstance(recorded_hashes, list)
            and len(configurations) == len(recorded_hashes),
            f"adaptive cycle {expected_cycle} configuration hashes are incomplete",
        )
    normalized: list[dict[str, Any]] = []
    for index, (raw, recorded_hash) in enumerate(zip(configurations, recorded_hashes)):
        configuration = _mapping(raw, f"configuration {index}")
        record = dict(configuration)
        _require(
            digest_object(record) == str(recorded_hash),
            f"declared configuration hash mismatch: {record.get('trial_id', index)}",
        )
        normalized.append(record)
    return normalized


def _validate_configuration(
    configuration: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    stage: str,
) -> None:
    trial_id = configuration.get("trial_id")
    _require(isinstance(trial_id, str) and trial_id, "configuration lacks trial_id")
    _require(
        configuration.get("data_mode") == "development",
        f"{trial_id} is not development-only",
    )
    _require(
        tuple(configuration.get("models", ())) == policy.model_triplet,
        f"{trial_id} omits the complete ordered T/U/M triplet",
    )
    _require(
        tuple(configuration.get("nuisance", ())) == policy.nuisance_terms,
        f"{trial_id} changes nuisance families",
    )
    for name, choices in policy.grammar_choices.items():
        _require(
            configuration.get(name) in choices,
            f"{trial_id} has out-of-grammar field {name}",
        )
    if stage == "coverage":
        _require(
            configuration.get("stage") == "coverage"
            and trial_id.startswith("coverage_"),
            f"coverage trial identity mismatch: {trial_id}",
        )
    else:
        cycle = "1" if stage.endswith("1") else "2"
        _require(
            configuration.get("stage") == "adaptive"
            and trial_id.startswith(f"adaptive_c{cycle}_"),
            f"{stage} trial identity mismatch: {trial_id}",
        )


def _receipt_status_counts(path: Path, trial_id: str) -> dict[str, int]:
    counts = {
        "valid_complete_T_U_M_triplet": 0,
        "unscorable": 0,
        "technical_failure": 0,
    }
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ObservedAttestationError(
                        f"malformed receipt row {trial_id}:{line_number}"
                    ) from exc
                record = _mapping(record, f"receipt row {trial_id}:{line_number}")
                # Deliberately read only the status field.  All score/delta/
                # objective payloads remain unexamined and are never emitted.
                status = record.get("status")
                _require(
                    status in counts,
                    f"unknown receipt status for {trial_id}: {status!r}",
                )
                counts[str(status)] += 1
    except FileNotFoundError as exc:
        raise ObservedAttestationError(f"missing receipt file for {trial_id}") from exc
    _require(sum(counts.values()) > 0, f"trial {trial_id} has no receipt rows")
    return counts


def _attest_trial(
    *,
    run_root: Path,
    trial_dir: Path,
    stage: str,
    configuration: Mapping[str, Any],
    policy: EpisodePolicy,
) -> dict[str, Any]:
    trial_id = str(configuration["trial_id"])
    manifest_path = trial_dir / "manifest.json"
    manifest = _mapping(_read_json(manifest_path, "source trial manifest"), "source trial manifest")
    _require(
        set(manifest) == {"schema_version", "files", "final_connectivity_accessed"},
        f"source manifest fields differ from v1 schema for {trial_id}",
    )
    _require(
        manifest.get("schema_version") == SOURCE_MANIFEST_SCHEMA,
        f"unsupported immutable source manifest for {trial_id}",
    )
    _require(
        manifest.get("final_connectivity_accessed") is False,
        f"source manifest records final access for {trial_id}",
    )
    files = _mapping(manifest.get("files"), "source trial files")
    _require(set(files) == SOURCE_FILES, f"source file set is incomplete for {trial_id}")
    authenticated_files: dict[str, str] = {}
    for name in sorted(SOURCE_FILES):
        _require(Path(name).name == name, f"unsafe source filename for {trial_id}")
        actual = _sha256_file(trial_dir / name)
        _require(actual == str(files[name]), f"source file hash mismatch for {trial_id}/{name}")
        authenticated_files[name] = actual

    # Parsing is restricted to structural/status keys.  No aggregate objective,
    # direction score, delta, interval, or selected value is read or compared.
    report = _mapping(
        _read_json(trial_dir / "trial_report.json", "source trial report"),
        "source trial report",
    )
    _require(report.get("schema_version") == SOURCE_REPORT_SCHEMA, f"report schema mismatch for {trial_id}")
    _require(report.get("trial_id") == trial_id, f"report identity mismatch for {trial_id}")
    _require(report.get("status") == "completed", f"trial {trial_id} is not technically complete")
    _require(report.get("data_role") == "development_only", f"trial {trial_id} has the wrong data role")
    _require(report.get("final_connectivity_accessed") is False, f"report records final access for {trial_id}")
    report_configuration = _mapping(report.get("configuration"), "report configuration")
    configuration_sha256 = digest_object(configuration)
    _require(
        report_configuration.get("configuration_sha256") == configuration_sha256,
        f"report configuration hash mismatch for {trial_id}",
    )
    for name, value in configuration.items():
        _require(
            report_configuration.get(name) == value,
            f"report configuration field mismatch for {trial_id}: {name}",
        )
    _require(
        tuple(report_configuration.get("models", ())) == policy.model_triplet,
        f"report does not record complete T/U/M presence for {trial_id}",
    )
    aggregate = _mapping(report.get("aggregate"), "report aggregate status")
    technical_count = aggregate.get("technical_failure_count")
    valid_count = aggregate.get("valid_type_count")
    unscorable_count = aggregate.get("unscorable_type_count")
    for name, value in (
        ("technical_failure_count", technical_count),
        ("valid_type_count", valid_count),
        ("unscorable_type_count", unscorable_count),
    ):
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"invalid structural count {name} for {trial_id}",
        )
    failures = report.get("failures")
    _require(
        technical_count == 0 and failures == [],
        f"trial {trial_id} has a technical failure",
    )
    status_counts = _receipt_status_counts(
        trial_dir / "type_direction_receipts.jsonl", trial_id
    )
    _require(
        status_counts["technical_failure"] == 0
        and status_counts["valid_complete_T_U_M_triplet"] == valid_count
        and status_counts["unscorable"] == unscorable_count,
        f"receipt/report structural counts disagree for {trial_id}",
    )
    return {
        "stage": stage,
        "trial_id": trial_id,
        "configuration_sha256": configuration_sha256,
        "scientific_configuration_sha256": scientific_configuration_sha256(
            configuration
        ),
        "models": list(policy.model_triplet),
        "complete_T_U_M_triplet": True,
        "technical_failure_count": 0,
        "valid_complete_type_count": valid_count,
        "unscorable_type_count": unscorable_count,
        "receipt_record_count": sum(status_counts.values()),
        "status": "valid_complete_T_U_M_triplet",
        "data_role": "development_only",
        "final_connectivity_accessed": False,
        "source_trial_directory": str(trial_dir.relative_to(run_root)),
        "source_manifest_schema": SOURCE_MANIFEST_SCHEMA,
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_files_sha256": authenticated_files,
        "source_report_schema": SOURCE_REPORT_SCHEMA,
        "source_report_status": "completed",
        "objective_values_read": False,
        "objective_values_compared": False,
        "objective_values_emitted": False,
    }


def build_observed_attestation(
    *,
    run_root: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Reconstruct the exact 18+9+9 status-only attestation in memory."""

    run_root = _inside(run_root, outputs_root, "observed run root")
    trials: list[dict[str, Any]] = []
    configuration_sources: list[dict[str, Any]] = []
    for stage, expected_count, config_relative, trial_relative, cycle in CHAIN_SOURCES:
        configuration_path = run_root / config_relative
        plan_manifest_sha256 = _verify_plan_manifest(
            configuration_path.parent, configuration_path.name
        )
        configurations = _load_configurations(configuration_path, cycle)
        _require(
            len(configurations) == expected_count,
            f"{stage} expected {expected_count} configurations, found {len(configurations)}",
        )
        for configuration in configurations:
            _validate_configuration(configuration, policy=policy, stage=stage)
            trials.append(
                _attest_trial(
                    run_root=run_root,
                    trial_dir=run_root / trial_relative / str(configuration["trial_id"]),
                    stage=stage,
                    configuration=configuration,
                    policy=policy,
                )
            )
        configuration_sources.append(
            {
                "stage": stage,
                "configuration_path": str(config_relative),
                "configuration_file_sha256": _sha256_file(configuration_path),
                "plan_manifest_sha256": plan_manifest_sha256,
                "declared_trial_count": len(configurations),
            }
        )
    trial_ids = [row["trial_id"] for row in trials]
    full_hashes = [row["configuration_sha256"] for row in trials]
    scientific_hashes = [row["scientific_configuration_sha256"] for row in trials]
    _require(len(trials) == policy.minimum_valid_trials == 36, "attestation must contain exactly 36 trials")
    _require(len(set(trial_ids)) == 36, "attestation repeats a trial ID")
    _require(len(set(full_hashes)) == 36, "attestation repeats a full configuration")
    _require(
        len(set(scientific_hashes)) == 36,
        "attestation repeats a scientific configuration after identity/stage exclusion",
    )
    core = {
        "schema_version": ATTESTATION_SCHEMA,
        "episode_id": "ep12",
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_hash,
        "run_root": str(run_root),
        "trial_count": len(trials),
        "stage_trial_counts": {
            "coverage": 18,
            "adaptive_cycle_1": 9,
            "adaptive_cycle_2": 9,
        },
        "configuration_sources": configuration_sources,
        "trials": trials,
        "unique_trial_ids": True,
        "unique_full_configuration_hashes": True,
        "unique_scientific_configuration_hashes": True,
        "source_trial_files_modified": False,
        "objective_values_read": False,
        "objective_values_compared": False,
        "objective_values_emitted": False,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "attestation_sha256": digest_object(core)}


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            view = memoryview(content)
            while view:
                written = handle.write(view)
                if written is None or written <= 0:
                    raise ObservedAttestationError(f"short write while creating {path}")
                view = view[written:]
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_or_verify(path: Path, content: bytes, label: str) -> None:
    if path.exists():
        _require(path.is_file(), f"existing {label} is not a file")
        _require(path.read_bytes() == content, f"existing {label} differs from retry")
        return
    _atomic_write(path, content)


def write_attestation_artifacts(
    *,
    run_root: Path,
    output_dir: Path,
    attestation: Mapping[str, Any],
    outputs_root: Path = OUTPUTS_ROOT,
) -> None:
    """Atomically publish a retry-safe separate attestation; never touch trials."""

    run_root = _inside(run_root, outputs_root, "observed run root")
    output_dir = _inside(output_dir, outputs_root, "observed attestation directory")
    _require(
        output_dir == (run_root / "observed_attestation").resolve(),
        "attestation output must be run_root/observed_attestation",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    attestation_bytes = _json_bytes(attestation)
    attestation_path = output_dir / "observed_trial_attestations.json"
    _write_or_verify(attestation_path, attestation_bytes, "observed attestation")
    manifest = {
        "schema_version": ATTESTATION_MANIFEST_SCHEMA,
        "files": {
            "observed_trial_attestations.json": digest_bytes(attestation_bytes)
        },
        "attestation_sha256": attestation.get("attestation_sha256"),
        "trial_count": attestation.get("trial_count"),
        "source_trial_files_modified": False,
        "objective_values_emitted": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    _write_or_verify(output_dir / "manifest.json", _json_bytes(manifest), "attestation manifest")


def verify_attestation_root(
    *,
    run_root: Path,
    attestation_dir: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Authenticate the separate root and rebuild its status-only semantics."""

    run_root = _inside(run_root, outputs_root, "observed run root")
    attestation_dir = _inside(
        attestation_dir, outputs_root, "observed attestation directory"
    )
    _require(
        attestation_dir == (run_root / "observed_attestation").resolve(),
        "readiness attestation root is not the run's separate attestation path",
    )
    manifest_path = attestation_dir / "manifest.json"
    manifest = _mapping(_read_json(manifest_path, "attestation manifest"), "attestation manifest")
    expected_manifest_fields = {
        "schema_version",
        "files",
        "attestation_sha256",
        "trial_count",
        "source_trial_files_modified",
        "objective_values_emitted",
        "final_connectivity_accessed",
        "final_connectivity_access_authorized",
    }
    _require(set(manifest) == expected_manifest_fields, "attestation manifest fields differ from schema")
    _require(manifest.get("schema_version") == ATTESTATION_MANIFEST_SCHEMA, "attestation manifest schema mismatch")
    files = _mapping(manifest.get("files"), "attestation manifest files")
    _require(set(files) == {"observed_trial_attestations.json"}, "attestation manifest file set mismatch")
    artifact_path = attestation_dir / "observed_trial_attestations.json"
    artifact_hash = _sha256_file(artifact_path)
    _require(artifact_hash == files["observed_trial_attestations.json"], "attestation artifact hash mismatch")
    payload = _mapping(_read_json(artifact_path, "observed attestation"), "observed attestation")
    recorded_hash = payload.get("attestation_sha256")
    _require(
        isinstance(recorded_hash, str)
        and recorded_hash
        == digest_object({key: value for key, value in payload.items() if key != "attestation_sha256"}),
        "observed attestation self-hash mismatch",
    )
    _require(
        manifest.get("attestation_sha256") == recorded_hash
        and manifest.get("trial_count") == 36
        and manifest.get("source_trial_files_modified") is False
        and manifest.get("objective_values_emitted") is False
        and manifest.get("final_connectivity_accessed") is False
        and manifest.get("final_connectivity_access_authorized") is False,
        "attestation manifest safety metadata mismatch",
    )
    expected = build_observed_attestation(
        run_root=run_root, policy=policy, outputs_root=outputs_root
    )
    _require(payload == expected, "observed attestation differs from source status metadata")
    rows = payload["trials"]
    return {
        "attestation_dir": str(attestation_dir),
        "attestation_manifest_sha256": _sha256_file(manifest_path),
        "attestation_artifact_file_sha256": artifact_hash,
        "attestation_sha256": recorded_hash,
        "verified_trial_count": len(rows),
        "stage_trial_counts": payload["stage_trial_counts"],
        "unique_trial_ids": True,
        "unique_full_configuration_hashes": True,
        "unique_scientific_configuration_hashes": True,
        "source_trial_files_modified": False,
        "objective_values_read": False,
        "objective_values_compared": False,
        "objective_values_emitted": False,
        "final_connectivity_accessed": False,
        "_trial_rows_by_id": {row["trial_id"]: row for row in rows},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-attest-observed-trials")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=EPISODE_ROOT / "SEARCH_POLICY.yaml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    policy = EpisodePolicy.load(arguments.policy)
    attestation = build_observed_attestation(
        run_root=arguments.run_root,
        policy=policy,
    )
    write_attestation_artifacts(
        run_root=arguments.run_root,
        output_dir=arguments.output_dir,
        attestation=attestation,
    )
    print(
        json.dumps(
            {
                "attestation_sha256": attestation["attestation_sha256"],
                "trial_count": attestation["trial_count"],
                "objective_values_read": False,
                "objective_values_compared": False,
                "objective_values_emitted": False,
                "source_trial_files_modified": False,
                "final_connectivity_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
