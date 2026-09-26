#!/usr/bin/env python3
"""Outcome-blind MOp soma common-support feasibility using Allen geometry."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


EP11_GEOMETRY_MODULE = Path(
    os.environ.get(
        "EP10_GEOMETRY_MODULE",
        "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
        "episode11_projection_types_vs_gradients/outputs/stage0/tools/"
        "build_whole_cortex_ssptr_geometry_support.py",
    )
)
EXPECTED_CELLS = 837
EXPECTED_PRIMARY_CELLS = 833
EXPECTED_GROUPS = 76
DEVELOPMENT_GROUPS = 12
FINAL_GROUPS = 8
MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS = 2
PER_GROUP_RETENTION = 0.90
OVERALL_ROLE_RETENTION = 0.95
MINIMUM_MULTI_CELL_DEVELOPMENT_GROUPS = 8
RADII_UM = (250, 500, 750, 1000, 1250, 1500)
DEPTH_TOLERANCES = (0.10, 0.20, 0.30, math.inf)
LAYER_MAP = {
    "MOp1": "L1",
    "MOp2/3": "L2/3",
    "MOp5": "L5",
    "MOp6a": "L6a",
    "MOp6b": "L6b",
}


def load_geometry_module():
    spec = importlib.util.spec_from_file_location("ep11_geometry_reuse", EP11_GEOMETRY_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load EP11 geometry module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EXPECTED_BRAINS = EXPECTED_GROUPS
    module.EXPECTED_SOURCE = "MOp"
    module.RADII_UM = RADII_UM
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=60.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def finite_float(value: str, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite {name}")
    return result


def load_records(path: Path, geometry):
    records = []
    seen_cells: set[str] = set()
    group_samples: dict[str, set[str]] = defaultdict(set)
    group_lines: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "provider_neuron_id",
            "fMOST_brain_id",
            "provider_sample_id",
            "source_acronym",
            "strain_or_cre_line",
            "hemisphere",
            "primary_axon_dendrite_distinguished_eligible",
            "soma_x_ccfv3_um",
            "soma_y_ccfv3_um",
            "soma_z_ccfv3_um",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"inventory missing columns: {missing}")
        for row in reader:
            cell_id = row["provider_neuron_id"].strip()
            fmost_id = row["fMOST_brain_id"].strip()
            sample = row["provider_sample_id"].strip()
            group = sample
            source = row["source_acronym"].strip()
            line = row["strain_or_cre_line"].strip()
            hemisphere = row["hemisphere"].strip()
            primary = row["primary_axon_dendrite_distinguished_eligible"].strip()
            if not cell_id or cell_id in seen_cells:
                raise RuntimeError("missing or duplicate provider neuron ID")
            seen_cells.add(cell_id)
            if not group or not sample or not line:
                raise RuntimeError("missing group, sample, or line")
            if source not in LAYER_MAP:
                raise RuntimeError(f"unexpected MOp source label {source!r}")
            if hemisphere not in {"Left", "Right"}:
                raise RuntimeError(f"unexpected hemisphere {hemisphere!r}")
            if primary not in {"yes", "no_axon_dendrite_not_distinguished"}:
                raise RuntimeError(f"unexpected primary flag {primary!r}")
            coordinate = tuple(
                finite_float(row[f"soma_{axis}_ccfv3_um"], axis)
                for axis in ("x", "y", "z")
            )
            group_samples[fmost_id].add(sample)
            group_lines[group].add(line)
            records.append(
                geometry.CellRecord(
                    cell_id=cell_id,
                    brain_id=group,
                    sample_id=sample,
                    provider_source_label=source,
                    layer=LAYER_MAP[source],
                    line=line,
                    hemisphere=hemisphere,
                    coordinate=coordinate,
                    primary_axon_dendrite_distinguished_eligible=(primary == "yes"),
                )
            )
    if len(records) != EXPECTED_CELLS or len(group_samples) != EXPECTED_GROUPS:
        raise RuntimeError("inventory does not reconcile to 837 cells in 76 groups")
    if len(seen_cells) != EXPECTED_CELLS:
        raise RuntimeError("provider neuron IDs are not unique")
    if any(len(values) != 1 for values in group_samples.values()):
        raise RuntimeError("a specimen prefix maps to multiple sample IDs")
    sample_groups: dict[str, set[str]] = defaultdict(set)
    for group, samples in group_samples.items():
        sample_groups[next(iter(samples))].add(group)
    if len(sample_groups) != EXPECTED_GROUPS or any(
        len(values) != 1 for values in sample_groups.values()
    ):
        raise RuntimeError("sample IDs do not map one-to-one to specimen prefixes")
    if any(len(values) != 1 for values in group_lines.values()):
        raise RuntimeError("a specimen group has multiple strain/Cre lines")
    if sum(record.primary_axon_dendrite_distinguished_eligible for record in records) != EXPECTED_PRIMARY_CELLS:
        raise RuntimeError("primary reconstruction count is not 833")
    return records


def solve_role_feasibility(geometry, records, neighbors, time_limit):
    groups = sorted({record.brain_id for record in records})
    group_index = {group: index for index, group in enumerate(groups)}
    group_cells: dict[int, list[int]] = defaultdict(list)
    group_line: dict[int, str] = {}
    for cell_index, record in enumerate(records):
        group = group_index[record.brain_id]
        group_cells[group].append(cell_index)
        prior = group_line.setdefault(group, record.line)
        if prior != record.line:
            raise RuntimeError("a group has multiple lines")
    line_groups: dict[str, list[int]] = defaultdict(list)
    for group, line in group_line.items():
        line_groups[line].append(group)

    group_count = len(groups)
    cell_count = len(records)
    development_offset = 0
    final_offset = group_count
    supported_development_offset = 2 * group_count
    supported_final_offset = 2 * group_count + cell_count
    variable_count = 2 * group_count + 2 * cell_count

    def development(group: int) -> int:
        return development_offset + group

    def final(group: int) -> int:
        return final_offset + group

    def supported_development(cell: int) -> int:
        return supported_development_offset + cell

    def supported_final(cell: int) -> int:
        return supported_final_offset + cell

    constraints = geometry.ConstraintRows(variable_count)
    constraints.add(
        {development(group): 1.0 for group in range(group_count)},
        DEVELOPMENT_GROUPS,
        DEVELOPMENT_GROUPS,
    )
    constraints.add(
        {final(group): 1.0 for group in range(group_count)},
        FINAL_GROUPS,
        FINAL_GROUPS,
    )
    for group in range(group_count):
        constraints.add({development(group): 1.0, final(group): 1.0}, upper=1.0)

    for cell, record in enumerate(records):
        owner = group_index[record.brain_id]
        constraints.add(
            {supported_development(cell): 1.0, development(owner): -1.0},
            upper=0.0,
        )
        constraints.add(
            {supported_final(cell): 1.0, final(owner): -1.0}, upper=0.0
        )
        development_neighbors = {
            development(group): -1.0 for group in neighbors[cell]
        }
        constraints.add(
            {
                supported_development(cell): float(
                    MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                ),
                **development_neighbors,
            },
            upper=0.0,
        )
        constraints.add(
            {
                supported_final(cell): float(
                    MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                ),
                **development_neighbors,
            },
            upper=0.0,
        )

    for group, cells in group_cells.items():
        required = math.ceil(PER_GROUP_RETENTION * len(cells) - 1e-12)
        development_coefficients = {
            supported_development(cell): 1.0 for cell in cells
        }
        development_coefficients[development(group)] = -float(required)
        constraints.add(development_coefficients, lower=0.0)
        final_coefficients = {supported_final(cell): 1.0 for cell in cells}
        final_coefficients[final(group)] = -float(required)
        constraints.add(final_coefficients, lower=0.0)

    overall_development = {
        supported_development(cell): 1.0 for cell in range(cell_count)
    }
    overall_final = {supported_final(cell): 1.0 for cell in range(cell_count)}
    for group, cells in group_cells.items():
        overall_development[development(group)] = -OVERALL_ROLE_RETENTION * len(cells)
        overall_final[final(group)] = -OVERALL_ROLE_RETENTION * len(cells)
    constraints.add(overall_development, lower=0.0)
    constraints.add(overall_final, lower=0.0)

    multi_cell_groups = [
        group for group, cells in group_cells.items() if len(cells) >= 2
    ]
    constraints.add(
        {development(group): 1.0 for group in multi_cell_groups},
        lower=float(MINIMUM_MULTI_CELL_DEVELOPMENT_GROUPS),
    )

    linear_constraint = constraints.linear_constraint()
    result = geometry.milp(
        c=geometry.np.zeros(variable_count, dtype=float),
        integrality=geometry.np.ones(variable_count, dtype=geometry.np.uint8),
        bounds=geometry.Bounds(
            geometry.np.zeros(variable_count, dtype=float),
            geometry.np.ones(variable_count, dtype=float),
        ),
        constraints=linear_constraint,
        options={"disp": False, "time_limit": float(time_limit)},
    )
    if result.status == 0 and result.x is not None:
        values = geometry.np.rint(result.x).astype(int)
        if geometry.np.max(geometry.np.abs(result.x - values)) > 1e-6:
            raise RuntimeError("MILP returned nonintegral witness")
        evaluated = linear_constraint.A @ values
        finite_lower = geometry.np.isfinite(linear_constraint.lb)
        finite_upper = geometry.np.isfinite(linear_constraint.ub)
        if geometry.np.any(
            evaluated[finite_lower] < linear_constraint.lb[finite_lower] - 1e-7
        ) or geometry.np.any(
            evaluated[finite_upper] > linear_constraint.ub[finite_upper] + 1e-7
        ):
            raise RuntimeError("MILP witness violates a constraint")
        return "yes", "feasible", int(result.status)
    if result.status == 2:
        return "no", "infeasible", int(result.status)
    return "unknown", "indeterminate", int(result.status)


def depth_label(value: float) -> str:
    return "unrestricted_diagnostic" if math.isinf(value) else f"{value:.2f}"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    geometry = load_geometry_module()
    records = load_records(args.inventory, geometry)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "cells": len(records),
                    "groups": len({record.brain_id for record in records}),
                    "primary_cells": sum(
                        record.primary_axon_dendrite_distinguished_eligible
                        for record in records
                    ),
                    "lines": len({record.line for record in records}),
                },
                sort_keys=True,
            )
        )
        return

    (
        geometry_records,
        exact_depth,
        api_depth,
        geodesic,
        geometry_diagnostics,
        graph_diagnostics,
    ) = geometry.build_geometry(records)
    if len(geometry_records) != EXPECTED_CELLS:
        raise RuntimeError(
            "not all 837 cells are atlas-geometry eligible; fail rather than "
            "change role-retention denominators"
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
    if len(primary_records) != EXPECTED_PRIMARY_CELLS or len(
        {record.brain_id for record in primary_records}
    ) != EXPECTED_GROUPS:
        raise RuntimeError("geometry qualification removes a complete primary group")
    primary_geodesic = geodesic[geometry.np.ix_(primary_positions, primary_positions)]
    depth_conventions = (
        ("exact_polyline_closest_point_primary", exact_depth[primary_positions]),
        ("ccf_streamlines_LineString3D_project_sensitivity", api_depth[primary_positions]),
    )
    groups = sorted({record.brain_id for record in primary_records})
    group_index = {group: index for index, group in enumerate(groups)}

    rows: list[dict[str, object]] = []
    for depth_convention, depth_values in depth_conventions:
        for radius in RADII_UM:
            for tolerance in DEPTH_TOLERANCES:
                neighbors = geometry.neighbor_brain_sets(
                    primary_records,
                    group_index,
                    primary_geodesic,
                    depth_values,
                    radius,
                    tolerance,
                )
                feasible, status, status_code = solve_role_feasibility(
                    geometry,
                    primary_records,
                    neighbors,
                    args.milp_time_limit_seconds,
                )
                rows.append(
                    {
                            "source_region": "MOp",
                            "cohort": "Gao_whole_cortex_projectome",
                            "analysis_cells": len(primary_records),
                            "released_brain_sample_units": len(groups),
                            "identity_interpretation": "unique_provider_sample_id_one_to_one_with_fMOST_id_not_certified_animal",
                            "hemispheres_folded": "yes",
                            "tangential_metric": "pial_endpoint_26_neighbor_surface_graph_geodesic_um",
                            "depth_metric": "pia_to_soma_path_length_divided_by_local_streamline_length",
                            "depth_projection_convention": depth_convention,
                            "tangential_radius_um": radius,
                            "normalized_depth_tolerance": depth_label(tolerance),
                            "label_support_track": "line_composition_reported_not_hard_gated",
                            "cells_with_2plus_candidate_neighbor_groups": sum(
                                len(values) >= MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                                for values in neighbors
                            ),
                            "development_groups": DEVELOPMENT_GROUPS,
                            "final_groups": FINAL_GROUPS,
                            "unassigned_groups": len(groups) - DEVELOPMENT_GROUPS - FINAL_GROUPS,
                            "minimum_per_role_group_retention": f"{PER_GROUP_RETENTION:.2f}",
                            "minimum_overall_retention_within_each_role": f"{OVERALL_ROLE_RETENTION:.2f}",
                            "minimum_development_groups_with_2plus_cells": MINIMUM_MULTI_CELL_DEVELOPMENT_GROUPS,
                            "distinct_development_neighbor_groups_per_retained_cell": MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS,
                            "feasible_role_allocation_exists": feasible,
                            "solver_status": status,
                            "solver_status_code": status_code,
                            "role_assignment_emitted": "no",
                            "screen_status": "outcome_blind_geometry_feasibility_no_role_assignment",
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".part")
    with partial_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial_csv, args.output_csv)
    summary = {
        "status": "complete",
        "rows": len(rows),
        "input_cells": len(records),
        "primary_cells_before_geometry": EXPECTED_PRIMARY_CELLS,
        "primary_geometry_eligible_cells": len(primary_records),
        "released_brain_sample_units": EXPECTED_GROUPS,
        "feasible_rows": sum(
            row["feasible_role_allocation_exists"] == "yes" for row in rows
        ),
        "transgenic_line_group_counts": dict(
            sorted(
                (
                    line,
                    len({record.brain_id for record in primary_records if record.line == line}),
                )
                for line in {record.line for record in primary_records}
            )
        ),
        "transgenic_line_role_constraint": "not_hard_gated_nine_lines_exceed_eight_final_groups",
        "geometry_diagnostics": geometry_diagnostics,
        "surface_graph_diagnostics": graph_diagnostics,
        "role_assignment_emitted": False,
        "projection_outcomes_opened": False,
    }
    atomic_json(args.summary_json, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
