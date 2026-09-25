#!/usr/bin/env python3
"""Validate an EP11 trusted redacted table and emit aggregate support only.

Real row-level inputs remain inside the trusted steward boundary, require a
redaction receipt and expected SHA-256 before CSV decoding, and are never
candidate-visible. The tool never accepts the mixed-role provider workbook or
morphology archives. Only its identifier-free JSON aggregate may be handed to
the candidate process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "stable_cell_id",
    "authenticated_independent_animal_id",
    "source_region",
    "ccf_soma_x",
    "ccf_soma_y",
    "ccf_soma_z",
    "eligible_under_frozen_metadata_rule",
    "exclusion_reason",
    "role",
}

ALLOWED_COLUMNS = REQUIRED_COLUMNS | {
    "provider_cell_id",
    "provider_brain_id_proxy",
    "specimen_id_if_distinct",
    "duplicate_family_id",
    "source_subregion_if_preregistered",
    "raw_soma_x",
    "raw_soma_y",
    "raw_soma_z",
    "cortical_layer",
    "preregistered_depth",
    "sex",
    "age",
    "strain",
    "genotype_or_cre_line",
    "labeling_strategy",
    "imaging_modality",
    "hemisphere",
    "source_laboratory",
    "acquisition_batch",
    "registration_batch",
    "reconstruction_version",
    "manual_check_status",
    "complete_axon_reconstruction_flag_with_independent_provenance",
    "registration_qc_flag_with_independent_provenance",
    "metadata_missingness_flags",
}

ROLES = {"development", "audit_sealed", "excluded"}
TRUE_VALUES = {"1", "true", "yes"}
FALSE_VALUES = {"0", "false", "no"}


class ValidationError(Exception):
    pass


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValidationError(
        f"row {row_number}: {field} must be one of "
        f"{sorted(TRUE_VALUES | FALSE_VALUES)}"
    )


def parse_coordinate(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValidationError(f"row {row_number}: {field} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValidationError(f"row {row_number}: {field} is not finite")
    return parsed


def verify_receipt(receipt_path: Path, input_hash: str) -> str:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "input_sha256",
        "row_level_input_candidate_visible",
        "projection_derived_values_emitted",
        "audit_row_identities_emitted_to_candidate",
        "trusted_operation_fully_logged",
        "emitted_columns",
    }
    missing = sorted(required - receipt.keys())
    if missing:
        raise ValidationError(f"trusted receipt missing keys: {missing}")
    if receipt["input_sha256"] != input_hash:
        raise ValidationError("trusted receipt input_sha256 does not match input")
    if receipt["row_level_input_candidate_visible"] is not False:
        raise ValidationError("row-level trusted input must not be candidate-visible")
    if receipt["projection_derived_values_emitted"] is not False:
        raise ValidationError("receipt does not exclude projection-derived values")
    if receipt["audit_row_identities_emitted_to_candidate"] is not False:
        raise ValidationError("receipt exposes audit row identities to candidates")
    if receipt["trusted_operation_fully_logged"] is not True:
        raise ValidationError("trusted operation is not fully logged")
    emitted = set(receipt["emitted_columns"])
    if not emitted <= ALLOWED_COLUMNS:
        raise ValidationError("trusted receipt lists non-allow-listed columns")
    return sha256_bytes(receipt_path)


def validate(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input.resolve()
    if input_path.suffix.lower() != ".csv":
        raise ValidationError("input must be a redacted CSV, never XLSX or ZIP")
    input_hash = sha256_bytes(input_path)
    if input_hash != args.expected_sha256:
        raise ValidationError("input SHA-256 differs from --expected-sha256")

    receipt_hash = None
    if args.mode == "trusted-steward":
        if args.trusted_receipt is None:
            raise ValidationError("trusted-steward mode requires --trusted-receipt")
        receipt_hash = verify_receipt(args.trusted_receipt.resolve(), input_hash)
    elif args.trusted_receipt is not None:
        raise ValidationError("synthetic mode must not use a trusted real-data receipt")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        unknown = sorted(columns - ALLOWED_COLUMNS)
        if missing:
            raise ValidationError(f"missing required columns: {missing}")
        if unknown:
            raise ValidationError("non-allow-listed columns detected; fail closed")

        seen_cells: set[str] = set()
        animal_role: dict[str, str] = {}
        by_source_role_animals: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        by_source_role_cells: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        by_source_axis: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"x": [], "y": [], "z": []}
        )
        eligible_rows = 0
        excluded_rows = 0
        total_rows = 0

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            cell_id = row["stable_cell_id"].strip()
            animal_id = row["authenticated_independent_animal_id"].strip()
            source = row["source_region"].strip()
            role = row["role"].strip()
            eligible = parse_bool(
                row["eligible_under_frozen_metadata_rule"],
                "eligible_under_frozen_metadata_rule",
                row_number,
            )

            if not cell_id or cell_id in seen_cells:
                raise ValidationError(f"row {row_number}: missing or duplicate cell ID")
            seen_cells.add(cell_id)
            if role not in ROLES:
                raise ValidationError(f"row {row_number}: invalid role")
            if not source:
                raise ValidationError(f"row {row_number}: missing source region")
            if eligible and (not animal_id or role == "excluded"):
                raise ValidationError(
                    f"row {row_number}: eligible row lacks animal or usable role"
                )
            if not eligible:
                excluded_rows += 1
                if not row["exclusion_reason"].strip():
                    raise ValidationError(
                        f"row {row_number}: excluded row lacks prospective reason"
                    )
                continue

            prior_role = animal_role.setdefault(animal_id, role)
            if prior_role != role:
                raise ValidationError(
                    f"row {row_number}: one animal appears in multiple roles"
                )

            x = parse_coordinate(row["ccf_soma_x"], "ccf_soma_x", row_number)
            y = parse_coordinate(row["ccf_soma_y"], "ccf_soma_y", row_number)
            z = parse_coordinate(row["ccf_soma_z"], "ccf_soma_z", row_number)
            eligible_rows += 1
            by_source_role_animals[source][role].add(animal_id)
            by_source_role_cells[source][role] += 1
            by_source_axis[source]["x"].append(x)
            by_source_axis[source]["y"].append(y)
            by_source_axis[source]["z"].append(z)

    summaries = []
    for source in sorted(by_source_role_animals):
        dev_animals = len(by_source_role_animals[source]["development"])
        audit_animals = len(by_source_role_animals[source]["audit_sealed"])
        axes = by_source_axis[source]
        summaries.append(
            {
                "source_region": source,
                "development_animals": dev_animals,
                "audit_animals": audit_animals,
                "development_cells": by_source_role_cells[source]["development"],
                "audit_cells": by_source_role_cells[source]["audit_sealed"],
                "coordinate_bounds": {
                    axis: [min(values), max(values)] for axis, values in axes.items()
                },
                "passes_current_count_floor_only": (
                    dev_animals >= args.minimum_development_animals
                    and audit_animals >= args.minimum_audit_animals
                ),
                "full_support_qualification": "not_evaluated_by_this_count_validator",
            }
        )

    return {
        "schema_version": "ep11.redacted_metadata_validation.v1",
        "mode": args.mode,
        "input_sha256": input_hash,
        "trusted_receipt_sha256": receipt_hash,
        "row_counts": {
            "total": total_rows,
            "eligible": eligible_rows,
            "excluded": excluded_rows,
        },
        "current_design_count_floor": {
            "development_animals": args.minimum_development_animals,
            "audit_animals": args.minimum_audit_animals,
            "universal_scientific_threshold": False,
        },
        "sources": summaries,
        "contains_cell_or_animal_identifiers": False,
        "outcome_support_conclusion": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--trusted-receipt", type=Path)
    parser.add_argument(
        "--mode", choices=("synthetic", "trusted-steward"), required=True
    )
    parser.add_argument("--minimum-development-animals", type=int, default=12)
    parser.add_argument("--minimum-audit-animals", type=int, default=8)
    return parser


def main() -> int:
    try:
        report = validate(build_parser().parse_args())
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)}))
        return 2
    print(json.dumps({"status": "valid", "report": report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
