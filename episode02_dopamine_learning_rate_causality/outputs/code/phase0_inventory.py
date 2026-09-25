#!/usr/bin/env python3
"""Trusted structural inventory for the Dudman Episode 02 Phase-0 gate.

The Figshare release stores all mice and all fields in one compressed MATLAB
v5 variable.  Consequently scipy must materialize that variable before nested
fields can be inspected. This program deliberately emits source integrity,
development-only code counts, aggregate audit schema/support facts, and the
access boundary. It never emits per-record audit values, behavioral or neural
magnitudes, treatment-exposure counts, or contingency-derived diagnostics.

The program is a trusted pre-launch extractor.  It is not suitable for a
candidate/search worker because the monolithic input also contains
evaluator-only intervention outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat


SCHEMA_VERSION = "ep02.dudman.phase0_inventory.v2"

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

SUMMARY_FIELDS = ("predVars", "cuePredVars", "rewDA", "prepLick", "baseLick")
RAW_TRACE_FIELDS = ("nac", "vta", "ds", "pupil", "lick", "nose", "whisk", "body")
OUTCOME_FIELDS = (
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


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def json_scalar(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def count_codes(array: np.ndarray) -> dict[str, int]:
    flattened = np.asarray(array).reshape(-1)
    values, counts = np.unique(flattened, return_counts=True)
    return {json_scalar(value): int(count) for value, count in zip(values, counts)}


def finite_missingness(array: np.ndarray) -> dict[str, int]:
    numeric = np.asarray(array)
    if numeric.size == 0:
        return {"elements": 0, "finite": 0, "nonfinite": 0}
    finite = int(np.isfinite(numeric).sum())
    return {
        "elements": int(numeric.size),
        "finite": finite,
        "nonfinite": int(numeric.size - finite),
    }


def array_schema(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def mat_container_layout(path: Path) -> dict[str, Any]:
    """Read only the 128-byte MAT header and first top-level data tag."""

    with path.open("rb") as stream:
        header = stream.read(128)
        tag = stream.read(8)
    if len(header) != 128 or len(tag) != 8:
        raise ValueError("MAT file is too short for a v5 header and top-level tag")
    endian_marker = header[126:128]
    if endian_marker == b"IM":
        byte_order = "little"
        data_type, byte_count = struct.unpack("<II", tag)
    elif endian_marker == b"MI":
        byte_order = "big"
        data_type, byte_count = struct.unpack(">II", tag)
    else:
        raise ValueError(f"Unrecognized MAT endian marker: {endian_marker!r}")
    return {
        "header_description": header[:116].rstrip(b" \x00").decode("latin-1"),
        "version_hex": header[124:126].hex(),
        "endian_marker": endian_marker.decode("ascii"),
        "byte_order": byte_order,
        "top_level_tag_type": int(data_type),
        "top_level_tag_type_name": "miCOMPRESSED" if data_type == 15 else "other",
        "top_level_compressed_bytes": int(byte_count),
        "single_compressed_payload_reaches_eof": path.stat().st_size == 136 + byte_count,
    }


def load_cohort_map(path: Path, record_count: int) -> tuple[dict[str, Any], dict[int, dict[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    index_base = payload.get("source_index_base")
    if index_base != 1:
        raise ValueError("Cohort map must use one-based source indices")

    lookup: dict[int, dict[str, str]] = {}
    for cohort in payload.get("cohorts", []):
        name = cohort["cohort_id"]
        role = cohort["role"]
        for source_index in cohort["source_records_1_based"]:
            if not isinstance(source_index, int):
                raise TypeError(f"Non-integer source index in {name}: {source_index!r}")
            if source_index < 1 or source_index > record_count:
                raise ValueError(f"Out-of-range source index in {name}: {source_index}")
            if source_index in lookup:
                raise ValueError(f"Source index {source_index} appears in multiple cohorts")
            lookup[source_index] = {"cohort_id": name, "role": role}

    expected = set(range(1, record_count + 1))
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"Cohort map does not cover the source exactly: missing={missing}, extra={extra}")
    return payload, lookup


def render_feasibility(report: dict[str, Any]) -> str:
    role = report["role_feasibility"]
    gate = report["gate"]
    integrity = report["source_integrity"]
    cohorts = role["cohorts"]
    checks = "\n".join(
        f"| `{item['check_id']}` | {item['status']} | {item['detail']} |"
        for item in gate["checks"]
    )
    blockers = "\n".join(f"- {item}" for item in gate["audit_opening_blockers"])
    return f"""# Dudman Phase-0 feasibility gate

## Verdict

**{gate['verdict']}** — the dataset can support a whole-animal development /
evaluator-role design, but this result does **not** establish a physical seal
or authorize candidate scoring or audit access.

## Frozen source

- File: `{integrity['source_path']}`
- Bytes: `{integrity['observed_bytes']:,}`
- SHA-256: `{integrity['observed_sha256']}`
- Provider MD5 matched: `{str(integrity['md5_matches']).lower()}`
- Container: one `{integrity['mat_container']['top_level_tag_type_name']}`
  payload holding a `1 x {report['structural_inventory']['record_count']}`
  `seshMerge` struct.

## Whole-animal roles

| Role | Mice | Trials | Use |
| --- | ---: | ---: | --- |
| Development controls | {cohorts['development_control']['mice']} | {cohorts['development_control']['trials']:,} | Model/estimand development only |
| Calibrated `stimLick-` audit | {cohorts['audit_calibrated_stim_lick_minus']['mice']} | {cohorts['audit_calibrated_stim_lick_minus']['trials']:,} | Intended evaluator-only primary contrast |
| Calibrated `stimLick+` audit | {cohorts['audit_calibrated_stim_lick_plus']['mice']} | {cohorts['audit_calibrated_stim_lick_plus']['trials']:,} | Intended evaluator-only primary contrast |
| Supraphysiological boundary | {cohorts['boundary_supraphysiological_stim_lick_plus']['mice']} | {cohorts['boundary_supraphysiological_stim_lick_plus']['trials']:,} | Internal post-primary diagnostic in the same logical opening |

The inferential unit is the mouse, never the trial.  The primary audit has
`n=6` versus `n=5`, so it can adjudicate only a large, predeclared directional
contrast.  Its exact assignment space has
`C(11,6)={role['primary_audit_assignment_count']}` allocations.

## Checks

| Check | Status | Detail |
| --- | --- | --- |
{checks}

## Launch blockers

{blockers}

## Access boundary

The trusted extractor had to materialize the monolithic MAT variable, but this
v2 packet emits detailed records only for development controls. For audit
roles it emits aggregate cohort and schema-signature counts, never per-record
treatment exposure, contingency, missingness, behavioral, neural, or model
values. Candidate workers must never receive the original file. The later
role materializer created exact hash-bound packs, but they remain
`materialized_unsealed` until a separate principal or external evaluator
enforces the physical boundary.

This is the sanitized v2 packet. A superseded, uncommitted local v1 packet
emitted per-record `stimulated_trial_count` and
`declared_contingency_structurally_consistent` before any candidate search or
controller ran. Their values are conservatively treated as exposed regardless
of viewing and are permanently excluded from candidate scoring, model
selection, audit terminals, and boundary interpretation. The current packet
does not emit either field.
"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    source = args.mat.resolve()
    cohort_map_path = args.cohort_map.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    observed_bytes = source.stat().st_size
    observed_md5 = file_digest(source, "md5")
    observed_sha256 = file_digest(source, "sha256")
    container = mat_container_layout(source)
    top_level = [
        {"name": name, "shape": list(shape), "matlab_class": matlab_class}
        for name, shape, matlab_class in whosmat(source)
    ]

    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mat = loadmat(source, variable_names=["seshMerge"], struct_as_record=True, squeeze_me=False)
    sessions = mat["seshMerge"]
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sessions.ndim != 2 or sessions.shape[0] != 1:
        raise ValueError(f"Expected seshMerge shape (1, N), observed {sessions.shape}")

    field_names = tuple(sessions.dtype.names or ())
    cohort_payload, cohort_lookup = load_cohort_map(cohort_map_path, sessions.shape[1])

    records: list[dict[str, Any]] = []
    cohort_accumulator: dict[str, dict[str, Any]] = {}
    for zero_based_index in range(sessions.shape[1]):
        source_index = zero_based_index + 1
        row = sessions[0, zero_based_index]
        assignment = cohort_lookup[source_index]
        cohort_id = assignment["cohort_id"]
        trial_id = np.asarray(row["trialID"]).reshape(-1)
        session_id = np.asarray(row["seshID"]).reshape(-1)
        stim_state = np.asarray(row["stimState"]).reshape(-1)
        if not (trial_id.size == session_id.size == stim_state.size):
            raise ValueError(f"Structural vector length mismatch in source record {source_index}")

        schemas = {field: array_schema(row[field]) for field in field_names}
        is_development = assignment["role"] == "development_full"
        record = {
            "opaque_animal_id": f"source_record_{source_index:02d}",
            "source_index_1_based": source_index,
            "cohort_id": cohort_id,
            "role": assignment["role"],
            "trial_rows": int(trial_id.size),
            "session_count": len(count_codes(session_id)),
            "field_schema": schemas,
        }
        if is_development:
            record.update(
                {
                    "session_id_counts": count_codes(session_id),
                    "trial_id_counts_unresolved_codebook": count_codes(trial_id),
                    "summary_field_missingness_only": {
                        field: finite_missingness(row[field]) for field in SUMMARY_FIELDS
                    },
                }
            )
        records.append(record)

        bucket = cohort_accumulator.setdefault(
            cohort_id,
            {"mice": 0, "trials": 0, "session_counts": [], "trial_rows_per_mouse": []},
        )
        bucket["mice"] += 1
        bucket["trials"] += int(trial_id.size)
        bucket["session_counts"].append(record["session_count"])
        bucket["trial_rows_per_mouse"].append(record["trial_rows"])

    for bucket in cohort_accumulator.values():
        session_counts = bucket.pop("session_counts")
        bucket["session_count_range"] = [min(session_counts), max(session_counts)]
        rows = bucket.pop("trial_rows_per_mouse")
        bucket["trial_rows_per_mouse_range"] = [min(rows), max(rows)]

    expected_counts = {
        "development_control": 9,
        "audit_calibrated_stim_lick_minus": 6,
        "audit_calibrated_stim_lick_plus": 5,
        "boundary_supraphysiological_stim_lick_plus": 4,
    }
    observed_counts = {name: item["mice"] for name, item in cohort_accumulator.items()}

    first_twenty = records[:20]
    first_twenty_summary_support = all(
        record["field_schema"]["predVars"]["shape"] == [800, 8]
        and record["field_schema"]["cuePredVars"]["shape"] == [800, 4]
        and record["field_schema"]["rewDA"]["shape"] == [1, 800]
        and record["field_schema"]["prepLick"]["shape"] == [1, 800]
        and record["field_schema"]["baseLick"]["shape"] == [1, 800]
        for record in first_twenty
    )
    first_twenty_nac_support = all(
        record["field_schema"]["nac"]["shape"]
        == [record["trial_rows"], 701]
        for record in first_twenty
    )
    boundary_summary_support = all(
        record["field_schema"]["rewDA"]["shape"] == [1, 580]
        and record["field_schema"]["prepLick"]["shape"] == [1, 580]
        and record["field_schema"]["baseLick"]["shape"] == [1, 580]
        for record in records[20:]
    )

    integrity_pass = (
        observed_bytes == args.expected_bytes
        and observed_md5 == args.expected_md5
        and observed_sha256 == args.expected_sha256
    )
    structure_pass = (
        top_level == [{"name": "seshMerge", "shape": [1, 24], "matlab_class": "struct"}]
        and field_names == EXPECTED_FIELDS
        and container["top_level_tag_type_name"] == "miCOMPRESSED"
        and container["single_compressed_payload_reaches_eof"]
    )
    cohort_pass = observed_counts == expected_counts
    support_pass = first_twenty_summary_support and first_twenty_nac_support and boundary_summary_support
    audit_role_schema_signatures: dict[str, set[str]] = {}
    for record in records:
        if record["role"] == "development_full":
            continue
        signature = hashlib.sha256(
            json.dumps(record["field_schema"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        audit_role_schema_signatures.setdefault(record["role"], set()).add(signature)
    public_audit_role_summary = {
        role_name: {
            "record_count": sum(record["role"] == role_name for record in records),
            "distinct_field_schema_signature_count": len(signatures),
            "individual_record_details_emitted": False,
        }
        for role_name, signatures in sorted(audit_role_schema_signatures.items())
    }

    checks = [
        {
            "check_id": "source_integrity",
            "status": "PASS" if integrity_pass else "FAIL",
            "detail": "byte count, provider MD5, and local SHA-256 all match" if integrity_pass else "one or more source pins differ",
        },
        {
            "check_id": "monolithic_container_schema",
            "status": "PASS" if structure_pass else "FAIL",
            "detail": "one compressed 1x24 seshMerge struct with the expected 20 fields" if structure_pass else "unexpected MAT layout or field schema",
        },
        {
            "check_id": "whole_animal_cohort_recovery",
            "status": "PASS" if cohort_pass else "FAIL",
            "detail": "sanitized upstream map yields 9 control, 6 stimLick-, 5 stimLick+, and 4 high-amplitude records" if cohort_pass else f"observed cohort counts {observed_counts}",
        },
        {
            "check_id": "primary_modality_support",
            "status": "PASS" if support_pass else "FAIL",
            "detail": "the first 20 mice have 800-trial summaries and complete NAc trace rows; the boundary cohort has 580-trial summaries" if support_pass else "required shape support is incomplete",
        },
        {
            "check_id": "audit_record_nondisclosure",
            "status": "PASS",
            "detail": "audit roles expose only aggregate cohort and schema-signature counts; no per-record treatment, contingency, missingness, or outcome diagnostic is emitted",
        },
        {
            "check_id": "physical_audit_firewall",
            "status": "BLOCKED",
            "detail": "exact role-filtered handoffs exist but remain materialized_unsealed under the same Unix uid with source readability and network reacquisition possible",
        },
        {
            "check_id": "trial_and_time_codebook",
            "status": "BLOCKED",
            "detail": "trialID semantics and the zero/alignment convention for the 701-sample axis are not encoded in the release",
        },
        {
            "check_id": "whole_mouse_information_floor",
            "status": "CONDITIONAL",
            "detail": "generic outcome-blind calibration selected no binding rule; endpoint-faithful bounded/discrete calibration and scientist signoff remain required",
        },
    ]
    hard_pass = all(item["status"] == "PASS" for item in checks[:5])
    verdict = "PASS_WITH_CONDITIONS" if hard_pass else "FAIL"

    report = {
        "schema_version": SCHEMA_VERSION,
        "observed_on": args.observed_on,
        "inventory_scope": "trusted_structural_qc_only",
        "source_integrity": {
            "source_path": str(source),
            "observed_bytes": observed_bytes,
            "expected_bytes": args.expected_bytes,
            "bytes_match": observed_bytes == args.expected_bytes,
            "observed_md5": observed_md5,
            "expected_md5": args.expected_md5,
            "md5_matches": observed_md5 == args.expected_md5,
            "observed_sha256": observed_sha256,
            "expected_sha256": args.expected_sha256,
            "sha256_matches": observed_sha256 == args.expected_sha256,
            "mat_container": container,
            "top_level_variables": top_level,
        },
        "structural_inventory": {
            "record_count": int(sessions.shape[1]),
            "field_count": len(field_names),
            "field_names": list(field_names),
            "total_trial_rows": sum(item["trial_rows"] for item in records),
            "development_records": [
                record for record in records if record["role"] == "development_full"
            ],
            "audit_individual_record_details_emitted": False,
            "audit_role_summary": public_audit_role_summary,
            "raw_trace_fields_shape_only": list(RAW_TRACE_FIELDS),
            "outcome_fields_without_value_summaries": list(OUTCOME_FIELDS),
        },
        "role_feasibility": {
            "cohort_map_path": str(cohort_map_path),
            "cohort_map_sha256": file_digest(cohort_map_path, "sha256"),
            "cohort_map_schema_version": cohort_payload.get("schema_version"),
            "cohorts": dict(sorted(cohort_accumulator.items())),
            "primary_audit_assignment_count": math.comb(11, 6),
            "primary_inferential_unit": "mouse",
            "trial_level_pseudoreplication_allowed": False,
            "proposed_roles": {
                "development": "all nine controls; publication- and phase0-exposed",
                "primary_audit": "six calibrated stimLick- versus five calibrated stimLick+ mice",
                "boundary_audit": "four supraphysiological stim+Lick+ mice, accessed only after the primary subpacket inside the same logical evaluator opening",
            },
        },
        "gate": {
            "verdict": verdict,
            "launch_authorized": False,
            "canonical_state_changed": False,
            "checks": checks,
            "audit_opening_blockers": [
                "Replace same-uid staging with a separate-principal or external evaluator firewall; candidate workers must not read or reacquire the original MAT file or evaluator packs.",
                "Resolve or explicitly freeze the trialID codebook and 701-sample time-axis/alignment contract without using audit effects.",
                "Replace the nonbinding generic calibration with endpoint-faithful binomial/beta-binomial calibration, absolute raw margins, independent validation, and scientist signoff.",
                "Independently review this extractor and prove that the audit evaluator emits no partial outcomes and rejects access before configuration lock.",
                "Bind the qualified author port, configuration/resource manifests, evaluator, and canonical adaptive program before any search or audit opening.",
            ],
        },
        "access_receipt": {
            "trusted_extractor": True,
            "monolithic_variable_materialized_in_process": True,
            "reason": "the complete 1x24 struct is a single miCOMPRESSED MATLAB v5 payload",
            "numeric_values_used_for_structural_checks": [
                "development-only trialID and seshID codes and counts",
                "audit trial/session vector lengths and aggregate session support",
                "development-only finite/nonfinite masks for five provider-derived summary fields",
            ],
            "candidate_discriminating_values_emitted": False,
            "candidate_discriminating_values_emitted_scope": "current_v2_packet",
            "behavioral_or_latency_magnitudes_emitted": False,
            "photometry_magnitudes_emitted": False,
            "perturbation_effects_emitted": False,
            "model_scores_emitted": False,
            "intervention_behavioral_state_codes_used_for_contingency_qc": False,
            "intervention_individual_outcome_magnitudes_inspected": False,
            "publication_level_aggregate_direction_known": True,
            "control_exposure": "the upstream mapping script also exposed outcome-derived control ordering/labels; all controls are development-only",
            "audit_exposure": "calibrated and high-amplitude arrays were materialized by scipy; current v2 output uses array schemas, vector lengths, and aggregate session support but does not index lickState values, count stimState exposure, compute contingency diagnostics, or emit per-record audit details",
            "superseded_v1_prelaunch_exposure": {
                "candidate_search_or_controller_started": False,
                "uncommitted_local_packet_existed": True,
                "per_record_fields_previously_emitted": [
                    "stimulated_trial_count",
                    "declared_contingency_structurally_consistent"
                ],
                "values_must_be_treated_as_exposed_regardless_of_whether_viewed": True,
                "scientific_use": "permanently_excluded_from_candidate_scoring_model_selection_audit_terminal_and_boundary_interpretation",
                "mitigation": "v2_removes_the_fields_and_all_per_record_audit_details_before_formal_search_execution"
            },
        },
        "runtime": {
            "python_executable": os.path.realpath(os.sys.executable),
            "numpy_version": np.__version__,
            "scipy_required": True,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "max_rss_kib_before_load": int(before_rss),
            "max_rss_kib_after_load": int(after_rss),
        },
    }
    return report


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = {
        "source_integrity.json": {
            "schema_version": report["schema_version"],
            "observed_on": report["observed_on"],
            **report["source_integrity"],
        },
        "structural_inventory.json": {
            "schema_version": report["schema_version"],
            "observed_on": report["observed_on"],
            "inventory_scope": report["inventory_scope"],
            **report["structural_inventory"],
        },
        "role_feasibility.json": {
            "schema_version": report["schema_version"],
            "observed_on": report["observed_on"],
            **report["role_feasibility"],
            "gate": report["gate"],
        },
        "access_receipt.json": {
            "schema_version": report["schema_version"],
            "observed_on": report["observed_on"],
            **report["access_receipt"],
            "runtime": report["runtime"],
        },
    }
    for name, payload in parts.items():
        atomic_write_json(output_dir / name, payload)
    atomic_write_text(output_dir / "FEASIBILITY.md", render_feasibility(report))

    manifested = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS"}
    )
    lines = [f"{file_digest(path, 'sha256')}  {path.name}" for path in manifested]
    atomic_write_text(output_dir / "SHA256SUMS", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--cohort-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--observed-on", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    write_outputs(report, args.output_dir.resolve())
    return 0 if report["gate"]["verdict"] == "PASS_WITH_CONDITIONS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
