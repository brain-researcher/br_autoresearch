#!/usr/bin/env python3
"""Build the frozen EP10 Gao MOp model-eligible development SWC manifest.

This joins only outcome-blind identity, role, and frozen soma-common-support
metadata.  It does not contact the provider or read any SWC content.  The
acquisition manifest is the 200-cell model-eligible subset of the 209 primary
development assignments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path


EXPECTED_ROLE_GROUPS = {"development": 12, "final": 8, "unassigned": 56}
EXPECTED_DEVELOPMENT_ASSIGNMENTS = 209
EXPECTED_ACQUISITION_FILES = 200
EXPECTED_COMMON_SUPPORT_EXCLUSIONS = 9
EXPECTED_SHARED_GROUPS = 15
FROZEN_GEOMETRY_CONDITION = (
    "exact_polyline_radius_500um_depth_tolerance_0.20"
)

SAFE_BRAIN_ID = re.compile(r"^[0-9]+$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9.-]+_[A-Za-z0-9.-]+\.swc$")

ROLE_FIELDS = {
    "fMOST_brain_id",
    "provider_sample_id",
    "all_cell_count",
    "primary_cell_count",
    "EP11_SSp_tr_overlap",
    "episode10_role",
    "geometry_supported_primary_cells",
    "geometry_condition",
    "role_status",
    "projection_outcome_exposed",
}
INVENTORY_FIELDS = {
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "hemisphere",
    "reconstruction_type",
    "primary_axon_dendrite_distinguished_eligible",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
}
OUTPUT_FIELDS = [
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "hemisphere",
    "reconstruction_type",
    "primary_common_support",
    "primary_model_eligible",
    "geometry_condition",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
    "episode10_role",
    "EP11_SSp_tr_overlap",
    "role_status",
    "projection_outcome_exposed",
]
class ManifestError(RuntimeError):
    """The safe inputs cannot produce the frozen development manifest."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--common-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ManifestError(f"{path} lacks required columns: {missing}")
        return list(reader)


def read_support_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if fields != OUTPUT_FIELDS:
            raise ManifestError(
                "common-support ledger columns/order differ from the frozen "
                f"builder output: observed={fields!r} expected={OUTPUT_FIELDS!r}"
            )
        return list(reader)


def build_assignment_rows(
    role_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
    support_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if len(role_rows) != 76:
        raise ManifestError(f"expected 76 frozen role rows, found {len(role_rows)}")

    roles: dict[str, dict[str, str]] = {}
    samples: dict[str, str] = {}
    for row in role_rows:
        brain_id = row["fMOST_brain_id"]
        sample_id = row["provider_sample_id"]
        if not SAFE_BRAIN_ID.fullmatch(brain_id):
            raise ManifestError(f"unsafe fMOST brain ID: {brain_id!r}")
        if not sample_id:
            raise ManifestError(f"empty provider sample ID for {brain_id}")
        if brain_id in roles:
            raise ManifestError(f"duplicate frozen role row: {brain_id}")
        if sample_id in samples:
            raise ManifestError(
                f"provider sample {sample_id!r} maps to both "
                f"{samples[sample_id]} and {brain_id}"
            )
        roles[brain_id] = row
        samples[sample_id] = brain_id

    observed_roles = Counter(row["episode10_role"] for row in role_rows)
    if dict(observed_roles) != EXPECTED_ROLE_GROUPS:
        raise ManifestError(
            f"frozen group roles differ from {EXPECTED_ROLE_GROUPS}: "
            f"{dict(observed_roles)}"
        )

    shared = [row for row in role_rows if row["EP11_SSp_tr_overlap"] == "yes"]
    if len(shared) != EXPECTED_SHARED_GROUPS:
        raise ManifestError(
            f"expected {EXPECTED_SHARED_GROUPS} EP11-shared groups, found "
            f"{len(shared)}"
        )
    shared_not_unassigned = [
        row["fMOST_brain_id"]
        for row in shared
        if row["episode10_role"] != "unassigned"
    ]
    if shared_not_unassigned:
        raise ManifestError(
            "EP11-shared groups are not all unassigned: "
            f"{shared_not_unassigned}"
        )

    if len(support_rows) != EXPECTED_DEVELOPMENT_ASSIGNMENTS:
        raise ManifestError(
            f"expected {EXPECTED_DEVELOPMENT_ASSIGNMENTS} common-support rows, "
            f"found {len(support_rows)}"
        )
    if support_rows != sorted(
        support_rows, key=lambda row: (row["fMOST_brain_id"], row["file"])
    ):
        raise ManifestError("common-support ledger is not canonically sorted")
    support_by_provider: dict[str, dict[str, str]] = {}
    for row in support_rows:
        provider_id = row["provider_neuron_id"]
        if not provider_id.isdigit():
            raise ManifestError(
                f"nonnumeric provider neuron ID in support ledger: {provider_id!r}"
            )
        if provider_id in support_by_provider:
            raise ManifestError(
                f"duplicate provider neuron ID in support ledger: {provider_id}"
            )
        if row["primary_common_support"] not in {"yes", "no"}:
            raise ManifestError(
                f"invalid primary_common_support for {provider_id}: "
                f"{row['primary_common_support']!r}"
            )
        if row["primary_model_eligible"] != row["primary_common_support"]:
            raise ManifestError(
                f"support/model eligibility mismatch for {provider_id}"
            )
        if row["geometry_condition"] != FROZEN_GEOMETRY_CONDITION:
            raise ManifestError(
                f"unexpected support geometry condition for {provider_id}: "
                f"{row['geometry_condition']!r}"
            )
        support_by_provider[provider_id] = row

    if len(inventory_rows) != 837:
        raise ManifestError(
            f"expected 837 safe MOp inventory rows, found {len(inventory_rows)}"
        )
    provider_ids: set[str] = set()
    filenames: set[str] = set()
    observed_cells: Counter[str] = Counter()
    observed_primary: Counter[str] = Counter()
    selected: list[dict[str, str]] = []

    for row in inventory_rows:
        provider_id = row["provider_neuron_id"]
        filename = row["file"]
        brain_id = row["fMOST_brain_id"]
        role = roles.get(brain_id)
        if role is None:
            raise ManifestError(f"inventory brain absent from role ledger: {brain_id}")
        if provider_id in provider_ids:
            raise ManifestError(f"duplicate provider neuron ID: {provider_id}")
        if not provider_id.isdigit():
            raise ManifestError(f"nonnumeric provider neuron ID: {provider_id!r}")
        if filename in filenames:
            raise ManifestError(f"duplicate SWC filename: {filename}")
        provider_ids.add(provider_id)
        filenames.add(filename)
        if not SAFE_FILE.fullmatch(filename):
            raise ManifestError(f"unsafe SWC filename: {filename!r}")
        if filename.split("_", 1)[0] != brain_id:
            raise ManifestError(
                f"filename prefix does not match fMOST brain: {filename} vs {brain_id}"
            )
        if row["provider_sample_id"] != role["provider_sample_id"]:
            raise ManifestError(f"sample ID mismatch for {filename}")
        if not row["source_acronym"].startswith("MOp"):
            raise ManifestError(f"non-MOp source label for {filename}")

        observed_cells[brain_id] += 1
        if row["reconstruction_type"] == "Axon_and_dendrite":
            observed_primary[brain_id] += 1

        if role["episode10_role"] != "development":
            continue
        if role["EP11_SSp_tr_overlap"] != "no":
            raise ManifestError(f"development row overlaps EP11: {filename}")
        if role["role_status"] != "frozen_before_projection_outcomes":
            raise ManifestError(f"development role is not frozen: {brain_id}")
        if role["projection_outcome_exposed"] != "no":
            raise ManifestError(f"development outcome already exposed: {brain_id}")
        if row["reconstruction_type"] != "Axon_and_dendrite":
            raise ManifestError(f"non-primary development reconstruction: {filename}")
        if row["primary_axon_dendrite_distinguished_eligible"] != "yes":
            raise ManifestError(f"development row is not primary eligible: {filename}")
        if row["hemisphere"] not in {"Left", "Right"}:
            raise ManifestError(f"invalid provider hemisphere for {filename}")
        try:
            soma = tuple(
                float(row[field])
                for field in (
                    "soma_x_ccfv3_um",
                    "soma_y_ccfv3_um",
                    "soma_z_ccfv3_um",
                )
            )
        except ValueError as error:
            raise ManifestError(f"nonnumeric soma coordinate for {filename}") from error
        if not all(math.isfinite(value) for value in soma):
            raise ManifestError(f"nonfinite soma coordinate for {filename}")
        support_row = support_by_provider.get(provider_id)
        if support_row is None:
            raise ManifestError(
                f"development row absent from common-support ledger: {filename}"
            )
        selected_row = {
            "provider_neuron_id": provider_id,
            "file": filename,
            "fMOST_brain_id": brain_id,
            "provider_sample_id": row["provider_sample_id"],
            "source_acronym": row["source_acronym"],
            "hemisphere": row["hemisphere"],
            "reconstruction_type": row["reconstruction_type"],
            "primary_common_support": support_row["primary_common_support"],
            "primary_model_eligible": support_row["primary_model_eligible"],
            "geometry_condition": role["geometry_condition"],
            "soma_x_ccfv3_um": row["soma_x_ccfv3_um"],
            "soma_y_ccfv3_um": row["soma_y_ccfv3_um"],
            "soma_z_ccfv3_um": row["soma_z_ccfv3_um"],
            "episode10_role": role["episode10_role"],
            "EP11_SSp_tr_overlap": role["EP11_SSp_tr_overlap"],
            "role_status": role["role_status"],
            "projection_outcome_exposed": role["projection_outcome_exposed"],
        }
        for field in OUTPUT_FIELDS:
            if support_row[field] != selected_row[field]:
                raise ManifestError(
                    f"common-support ledger identity mismatch for {filename}, "
                    f"field {field!r}: {support_row[field]!r} != "
                    f"{selected_row[field]!r}"
                )
        selected.append(selected_row)

    for brain_id, role in roles.items():
        try:
            expected_all = int(role["all_cell_count"])
            expected_primary = int(role["primary_cell_count"])
        except ValueError as error:
            raise ManifestError(f"noninteger role counts for {brain_id}") from error
        if observed_cells[brain_id] != expected_all:
            raise ManifestError(
                f"all-cell count mismatch for {brain_id}: "
                f"{observed_cells[brain_id]} != {expected_all}"
            )
        if observed_primary[brain_id] != expected_primary:
            raise ManifestError(
                f"primary-cell count mismatch for {brain_id}: "
                f"{observed_primary[brain_id]} != {expected_primary}"
            )

    selected.sort(key=lambda row: (row["fMOST_brain_id"], row["file"]))
    selected_groups = {row["fMOST_brain_id"] for row in selected}
    if len(selected) != EXPECTED_DEVELOPMENT_ASSIGNMENTS:
        raise ManifestError(
            f"expected {EXPECTED_DEVELOPMENT_ASSIGNMENTS} development assignments, "
            f"found {len(selected)}"
        )
    if len(selected_groups) != EXPECTED_ROLE_GROUPS["development"]:
        raise ManifestError(
            f"expected {EXPECTED_ROLE_GROUPS['development']} development groups, "
            f"found {len(selected_groups)}"
        )
    if any(row["episode10_role"] != "development" for row in selected):
        raise ManifestError("final or unassigned row entered development manifest")
    if any(row["EP11_SSp_tr_overlap"] != "no" for row in selected):
        raise ManifestError("EP11-shared row entered development manifest")
    if len({row["provider_neuron_id"] for row in selected}) != len(selected):
        raise ManifestError("duplicate development provider neuron ID")
    if len({row["file"] for row in selected}) != len(selected):
        raise ManifestError("duplicate development filename")
    selected_provider_ids = {row["provider_neuron_id"] for row in selected}
    if set(support_by_provider) != selected_provider_ids:
        missing = sorted(selected_provider_ids - set(support_by_provider), key=int)
        extra = sorted(set(support_by_provider) - selected_provider_ids, key=int)
        raise ManifestError(
            "common-support ledger identity set differs from development "
            f"assignments: missing={missing[:5]} extra={extra[:5]}"
        )

    observed_unsupported = {
        row["provider_neuron_id"]
        for row in selected
        if row["primary_common_support"] == "no"
    }
    if len(observed_unsupported) != EXPECTED_COMMON_SUPPORT_EXCLUSIONS:
        raise ManifestError(
            f"expected {EXPECTED_COMMON_SUPPORT_EXCLUSIONS} common-support "
            f"exclusions, found {len(observed_unsupported)}"
        )

    supported_by_group = Counter(
        row["fMOST_brain_id"]
        for row in selected
        if row["primary_common_support"] == "yes"
    )
    for brain_id in selected_groups:
        role = roles[brain_id]
        if role["geometry_condition"] != FROZEN_GEOMETRY_CONDITION:
            raise ManifestError(
                f"unexpected frozen geometry condition for {brain_id}: "
                f"{role['geometry_condition']!r}"
            )
        try:
            expected_supported = int(role["geometry_supported_primary_cells"])
        except ValueError as error:
            raise ManifestError(
                f"noninteger frozen supported-cell count for {brain_id}"
            ) from error
        if supported_by_group[brain_id] != expected_supported:
            raise ManifestError(
                f"common-support count mismatch for {brain_id}: "
                f"{supported_by_group[brain_id]} != {expected_supported}"
            )

    supported = sum(
        row["primary_common_support"] == "yes" for row in selected
    )
    if supported != EXPECTED_ACQUISITION_FILES:
        raise ManifestError(
            f"expected {EXPECTED_ACQUISITION_FILES} common-support development "
            f"cells, found {supported}"
        )
    if any(
        row["primary_model_eligible"] != row["primary_common_support"]
        for row in selected
    ):
        raise ManifestError("primary model eligibility differs from common support")
    return selected


def acquisition_rows(
    assignment_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows = [
        row
        for row in assignment_rows
        if row["primary_common_support"] == "yes"
        and row["primary_model_eligible"] == "yes"
    ]
    if len(rows) != EXPECTED_ACQUISITION_FILES:
        raise ManifestError(
            f"expected {EXPECTED_ACQUISITION_FILES} acquisition rows, found "
            f"{len(rows)}"
        )
    groups = {row["fMOST_brain_id"] for row in rows}
    if len(groups) != EXPECTED_ROLE_GROUPS["development"]:
        raise ManifestError(
            "common-support filtering removed a complete development group"
        )
    if any(row["primary_common_support"] != "yes" for row in rows):
        raise ManifestError("unsupported cell entered acquisition manifest")
    return rows


def build_rows(
    role_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
    support_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return the exact 200-row acquisition/modeling manifest."""

    return acquisition_rows(
        build_assignment_rows(role_rows, inventory_rows, support_rows)
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ManifestError(f"refusing symlink output: {path}")
    partial = path.with_suffix(path.suffix + ".part")
    if partial.is_symlink():
        raise ManifestError(f"refusing symlink partial output: {partial}")
    try:
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    role_rows = read_csv(args.roles, ROLE_FIELDS)
    inventory_rows = read_csv(args.inventory, INVENTORY_FIELDS)
    support_rows = read_support_csv(args.common_support)
    assignments = build_assignment_rows(role_rows, inventory_rows, support_rows)
    rows = acquisition_rows(assignments)
    write_manifest(args.output, rows)
    print(
        json.dumps(
            {
                "development_assignment_files": len(assignments),
                "primary_common_support_yes": len(rows),
                "primary_common_support_no": len(assignments) - len(rows),
                "acquisition_files": len(rows),
                "development_groups": len(
                    {row["fMOST_brain_id"] for row in rows}
                ),
                "EP11_shared_files": sum(
                    row["EP11_SSp_tr_overlap"] == "yes" for row in rows
                ),
                "reconstruction_types": dict(
                    sorted(Counter(row["reconstruction_type"] for row in rows).items())
                ),
                "common_support": str(args.common_support),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
