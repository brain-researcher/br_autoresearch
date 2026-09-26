#!/usr/bin/env python3
"""Derive frozen EP10 MOp development-cell soma common support.

Only the safe soma/identity inventory, frozen role ledger, and pre-existing
Allen cortical-geometry HDF5 assets are read.  No SWC tree, projection outcome,
or network resource is opened.  The exact rule is the already-frozen 500-um
pial-surface geodesic and 0.20 normalized-depth condition with support from at
least two distinct frozen development groups.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import build_mop_development_manifest as manifest_builder


SCREEN_MODULE = Path(__file__).with_name("screen_mop_geometry.py")
EXPECTED_ALL_CELLS = 837
EXPECTED_PRIMARY_CELLS = 833
EXPECTED_DEVELOPMENT_ASSIGNMENTS = 209
EXPECTED_SUPPORTED = 200
EXPECTED_UNSUPPORTED = 9
EXPECTED_DEVELOPMENT_GROUPS = 12
RADIUS_UM = 500
DEPTH_TOLERANCE = 0.20
MINIMUM_DISTINCT_DEVELOPMENT_GROUPS = 2


class SupportDerivationError(RuntimeError):
    """Safe inputs or derived common-support identities are inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate safe identity/role inputs without opening geometry HDF5",
    )
    parser.add_argument(
        "--execute-safe-geometry-derivation",
        action="store_true",
        help="run the outcome-blind soma-geometry derivation",
    )
    return parser.parse_args()


def load_screen_module():
    spec = importlib.util.spec_from_file_location(
        "ep10_common_support_screen", SCREEN_MODULE
    )
    if spec is None or spec.loader is None:
        raise SupportDerivationError("could not load the durable MOp geometry screen")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_safe_inputs(
    roles_path: Path, inventory_path: Path
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    role_rows = manifest_builder.read_csv(
        roles_path, manifest_builder.ROLE_FIELDS
    )
    inventory_rows = manifest_builder.read_csv(
        inventory_path, manifest_builder.INVENTORY_FIELDS
    )
    if len(role_rows) != 76:
        raise SupportDerivationError(
            f"expected 76 frozen role rows, found {len(role_rows)}"
        )
    if dict(Counter(row["episode10_role"] for row in role_rows)) != (
        manifest_builder.EXPECTED_ROLE_GROUPS
    ):
        raise SupportDerivationError("frozen role counts changed")
    roles: dict[str, dict[str, str]] = {}
    for row in role_rows:
        brain_id = row["fMOST_brain_id"]
        if brain_id in roles:
            raise SupportDerivationError(f"duplicate frozen role row: {brain_id}")
        roles[brain_id] = row

    if len(inventory_rows) != EXPECTED_ALL_CELLS:
        raise SupportDerivationError(
            f"expected {EXPECTED_ALL_CELLS} safe inventory rows, "
            f"found {len(inventory_rows)}"
        )
    inventory_by_provider: dict[str, dict[str, str]] = {}
    observed_all: Counter[str] = Counter()
    observed_primary: Counter[str] = Counter()
    development_primary = 0
    for row in inventory_rows:
        provider_id = row["provider_neuron_id"]
        brain_id = row["fMOST_brain_id"]
        role = roles.get(brain_id)
        if role is None:
            raise SupportDerivationError(
                f"inventory brain absent from frozen roles: {brain_id}"
            )
        if provider_id in inventory_by_provider:
            raise SupportDerivationError(
                f"duplicate provider neuron ID: {provider_id}"
            )
        if row["provider_sample_id"] != role["provider_sample_id"]:
            raise SupportDerivationError(
                f"sample identity mismatch for {row['file']}"
            )
        inventory_by_provider[provider_id] = row
        observed_all[brain_id] += 1
        if row["reconstruction_type"] == "Axon_and_dendrite":
            observed_primary[brain_id] += 1
            if role["episode10_role"] == "development":
                development_primary += 1

    if sum(observed_primary.values()) != EXPECTED_PRIMARY_CELLS:
        raise SupportDerivationError("safe inventory no longer has 833 primary cells")
    if development_primary != EXPECTED_DEVELOPMENT_ASSIGNMENTS:
        raise SupportDerivationError(
            "frozen development assignments no longer total 209 primary cells"
        )
    for brain_id, role in roles.items():
        if observed_all[brain_id] != int(role["all_cell_count"]):
            raise SupportDerivationError(f"all-cell count mismatch for {brain_id}")
        if observed_primary[brain_id] != int(role["primary_cell_count"]):
            raise SupportDerivationError(f"primary-cell count mismatch for {brain_id}")
        if role["episode10_role"] == "development":
            if role["geometry_condition"] != manifest_builder.FROZEN_GEOMETRY_CONDITION:
                raise SupportDerivationError(
                    f"geometry condition changed for development group {brain_id}"
                )
            if role["EP11_SSp_tr_overlap"] != "no":
                raise SupportDerivationError(
                    f"development group overlaps EP11: {brain_id}"
                )
            if role["role_status"] != "frozen_before_projection_outcomes":
                raise SupportDerivationError(f"role is not frozen: {brain_id}")
            if role["projection_outcome_exposed"] != "no":
                raise SupportDerivationError(
                    f"projection outcome was exposed for {brain_id}"
                )
    return role_rows, inventory_rows, roles, inventory_by_provider


def derive_rows(
    inventory_path: Path,
    roles: dict[str, dict[str, str]],
    inventory_by_provider: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    screen = load_screen_module()
    geometry = screen.load_geometry_module()
    records = screen.load_records(inventory_path, geometry)
    (
        geometry_records,
        exact_depth,
        _api_depth,
        geodesic,
        _geometry_diagnostics,
        _graph_diagnostics,
    ) = geometry.build_geometry(records)
    if len(geometry_records) != EXPECTED_ALL_CELLS:
        raise SupportDerivationError(
            "geometry eligibility changed; all 837 cells are required"
        )

    primary_positions = geometry.np.asarray(
        [
            index
            for index, record in enumerate(geometry_records)
            if record.primary_axon_dendrite_distinguished_eligible
        ],
        dtype=int,
    )
    primary_records = [geometry_records[int(index)] for index in primary_positions]
    if len(primary_records) != EXPECTED_PRIMARY_CELLS:
        raise SupportDerivationError("geometry primary scope changed from 833 cells")
    primary_geodesic = geodesic[
        geometry.np.ix_(primary_positions, primary_positions)
    ]
    primary_depth = exact_depth[primary_positions]
    samples = sorted({record.brain_id for record in primary_records})
    sample_index = {sample: index for index, sample in enumerate(samples)}
    neighbors = geometry.neighbor_brain_sets(
        primary_records,
        sample_index,
        primary_geodesic,
        primary_depth,
        RADIUS_UM,
        DEPTH_TOLERANCE,
    )

    development_samples = {
        role["provider_sample_id"]
        for role in roles.values()
        if role["episode10_role"] == "development"
    }
    if len(development_samples) != EXPECTED_DEVELOPMENT_GROUPS:
        raise SupportDerivationError("development group count changed from 12")
    development_indexes = {sample_index[sample] for sample in development_samples}

    rows: list[dict[str, str]] = []
    for index, record in enumerate(primary_records):
        if record.brain_id not in development_samples:
            continue
        inventory_row = inventory_by_provider.get(record.cell_id)
        if inventory_row is None:
            raise SupportDerivationError(
                f"geometry cell absent from safe inventory: {record.cell_id}"
            )
        brain_id = inventory_row["fMOST_brain_id"]
        role = roles[brain_id]
        supported = (
            len(neighbors[index] & development_indexes)
            >= MINIMUM_DISTINCT_DEVELOPMENT_GROUPS
        )
        flag = "yes" if supported else "no"
        rows.append(
            {
                "provider_neuron_id": inventory_row["provider_neuron_id"],
                "file": inventory_row["file"],
                "fMOST_brain_id": brain_id,
                "provider_sample_id": inventory_row["provider_sample_id"],
                "source_acronym": inventory_row["source_acronym"],
                "hemisphere": inventory_row["hemisphere"],
                "reconstruction_type": inventory_row["reconstruction_type"],
                "primary_common_support": flag,
                "primary_model_eligible": flag,
                "geometry_condition": role["geometry_condition"],
                "soma_x_ccfv3_um": inventory_row["soma_x_ccfv3_um"],
                "soma_y_ccfv3_um": inventory_row["soma_y_ccfv3_um"],
                "soma_z_ccfv3_um": inventory_row["soma_z_ccfv3_um"],
                "episode10_role": role["episode10_role"],
                "EP11_SSp_tr_overlap": role["EP11_SSp_tr_overlap"],
                "role_status": role["role_status"],
                "projection_outcome_exposed": role["projection_outcome_exposed"],
            }
        )

    rows.sort(key=lambda row: (row["fMOST_brain_id"], row["file"]))
    if len(rows) != EXPECTED_DEVELOPMENT_ASSIGNMENTS:
        raise SupportDerivationError(
            f"derived {len(rows)} development rows instead of 209"
        )
    support_counts = Counter(
        row["fMOST_brain_id"]
        for row in rows
        if row["primary_common_support"] == "yes"
    )
    if sum(support_counts.values()) != EXPECTED_SUPPORTED:
        raise SupportDerivationError(
            f"derived {sum(support_counts.values())} supported cells instead of 200"
        )
    if len(rows) - sum(support_counts.values()) != EXPECTED_UNSUPPORTED:
        raise SupportDerivationError("unsupported-cell count changed from nine")
    for brain_id, role in roles.items():
        if role["episode10_role"] != "development":
            continue
        if support_counts[brain_id] != int(
            role["geometry_supported_primary_cells"]
        ):
            raise SupportDerivationError(
                f"derived support count differs from frozen ledger for {brain_id}"
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.validate_only == args.execute_safe_geometry_derivation:
        raise SupportDerivationError(
            "choose exactly one of --validate-only or "
            "--execute-safe-geometry-derivation"
        )
    _, inventory_rows, roles, inventory_by_provider = load_safe_inputs(
        args.roles, args.inventory
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "development_assignments": EXPECTED_DEVELOPMENT_ASSIGNMENTS,
                    "geometry_hdf5_opened": False,
                    "inventory_rows": len(inventory_rows),
                    "network_requests": 0,
                    "output_created": False,
                    "projection_outcomes_opened": False,
                    "status": "validated_only",
                    "swc_rows_opened": 0,
                },
                sort_keys=True,
            )
        )
        return

    rows = derive_rows(args.inventory, roles, inventory_by_provider)
    manifest_builder.write_manifest(args.output, rows)
    print(
        json.dumps(
            {
                "development_assignments": len(rows),
                "development_groups": len(
                    {row["fMOST_brain_id"] for row in rows}
                ),
                "geometry_condition": manifest_builder.FROZEN_GEOMETRY_CONDITION,
                "network_requests": 0,
                "output": str(args.output),
                "primary_common_support_no": sum(
                    row["primary_common_support"] == "no" for row in rows
                ),
                "primary_common_support_yes": sum(
                    row["primary_common_support"] == "yes" for row in rows
                ),
                "projection_outcomes_opened": False,
                "status": "complete",
                "swc_rows_opened": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
