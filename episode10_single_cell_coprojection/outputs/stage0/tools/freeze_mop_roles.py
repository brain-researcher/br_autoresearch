#!/usr/bin/env python3
"""Freeze outcome-blind EP10 Gao MOp roles after the soma-geometry gate."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode10_single_cell_coprojection"
)
SCREEN_MODULE = Path(
    os.environ.get(
        "EP10_SCREEN_MODULE",
        str(EPISODE_ROOT / "outputs/stage0/tools/screen_mop_geometry.py"),
    )
)
RADIUS_UM = 500
DEPTH_TOLERANCE = 0.20
DEVELOPMENT_GROUPS = 12
FINAL_GROUPS = 8
PER_GROUP_RETENTION = 0.90
OVERALL_ROLE_RETENTION = 0.95
MINIMUM_MULTI_CELL_DEVELOPMENT_GROUPS = 8
MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS = 2
EXPECTED_CELLS = 837
EXPECTED_PRIMARY_CELLS = 833
EXPECTED_GROUPS = 76
DEFAULT_SEED = 20260926
SHARED_EP11_FMOST_IDS = frozenset(
    {
        "221224",
        "221227",
        "221299",
        "221366",
        "221428",
        "221471",
        "221478",
        "221479",
        "221481",
        "221512",
        "221513",
        "221726",
        "221728",
        "221752",
        "233794",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=300.0)
    return parser.parse_args()


def load_screen_module():
    spec = importlib.util.spec_from_file_location("ep10_mop_geometry_screen", SCREEN_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the durable MOp geometry screen")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_group_metadata(path: Path) -> dict[str, dict[str, object]]:
    required = {
        "provider_neuron_id",
        "fMOST_brain_id",
        "provider_sample_id",
        "strain_or_cre_line",
        "primary_axon_dendrite_distinguished_eligible",
    }
    groups: dict[str, dict[str, object]] = {}
    seen_cells: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"inventory missing role-freeze columns: {missing}")
        for row in reader:
            cell_id = row["provider_neuron_id"].strip()
            fmost = row["fMOST_brain_id"].strip()
            sample = row["provider_sample_id"].strip()
            line = row["strain_or_cre_line"].strip()
            primary_flag = row[
                "primary_axon_dendrite_distinguished_eligible"
            ].strip()
            if not cell_id or cell_id in seen_cells:
                raise RuntimeError("missing or duplicate provider neuron ID")
            seen_cells.add(cell_id)
            if not fmost or not sample or not line:
                raise RuntimeError("missing fMOST, sample, or line identity")
            if primary_flag not in {"yes", "no_axon_dendrite_not_distinguished"}:
                raise RuntimeError("unexpected primary eligibility flag")
            group = groups.setdefault(
                sample,
                {
                    "fMOST_brain_id": fmost,
                    "provider_sample_id": sample,
                    "strain_or_cre_line": line,
                    "all_cell_count": 0,
                    "primary_cell_count": 0,
                },
            )
            if group["fMOST_brain_id"] != fmost or group["strain_or_cre_line"] != line:
                raise RuntimeError("sample identity maps to multiple fMOST IDs or lines")
            group["all_cell_count"] = int(group["all_cell_count"]) + 1
            if primary_flag == "yes":
                group["primary_cell_count"] = int(group["primary_cell_count"]) + 1
    if len(seen_cells) != EXPECTED_CELLS or len(groups) != EXPECTED_GROUPS:
        raise RuntimeError("inventory does not reconcile to 837 cells in 76 groups")
    if sum(int(value["primary_cell_count"]) for value in groups.values()) != EXPECTED_PRIMARY_CELLS:
        raise RuntimeError("inventory does not reconcile to 833 primary cells")
    fmost_ids = [str(value["fMOST_brain_id"]) for value in groups.values()]
    if len(set(fmost_ids)) != EXPECTED_GROUPS:
        raise RuntimeError("fMOST IDs are not one-to-one with sample IDs")
    if not SHARED_EP11_FMOST_IDS.issubset(set(fmost_ids)):
        raise RuntimeError("the 15-unit EP11 overlap is not present in the inventory")
    return groups


def solve_roles(geometry, records, neighbors, group_metadata, seed, time_limit):
    groups = sorted({record.brain_id for record in records})
    if set(groups) != set(group_metadata):
        raise RuntimeError("geometry groups and inventory samples disagree")
    group_index = {group: index for index, group in enumerate(groups)}
    group_cells: dict[int, list[int]] = defaultdict(list)
    group_line: dict[int, str] = {}
    for cell_index, record in enumerate(records):
        group = group_index[record.brain_id]
        group_cells[group].append(cell_index)
        previous = group_line.setdefault(group, record.line)
        if previous != record.line:
            raise RuntimeError("a role-freeze group has multiple lines")

    lines = sorted(set(group_line.values()))
    if len(lines) != 9:
        raise RuntimeError(f"expected 9 strain/Cre lines, found {len(lines)}")
    line_index = {line: index for index, line in enumerate(lines)}
    line_groups: dict[str, list[int]] = defaultdict(list)
    for group, line in group_line.items():
        line_groups[line].append(group)

    group_count = len(groups)
    cell_count = len(records)
    line_count = len(lines)
    development_offset = 0
    final_offset = group_count
    supported_development_offset = 2 * group_count
    supported_final_offset = 2 * group_count + cell_count
    development_line_offset = 2 * group_count + 2 * cell_count
    final_line_offset = development_line_offset + line_count
    variable_count = final_line_offset + line_count

    def development(group: int) -> int:
        return development_offset + group

    def final(group: int) -> int:
        return final_offset + group

    def supported_development(cell: int) -> int:
        return supported_development_offset + cell

    def supported_final(cell: int) -> int:
        return supported_final_offset + cell

    def development_line(line: str) -> int:
        return development_line_offset + line_index[line]

    def final_line(line: str) -> int:
        return final_line_offset + line_index[line]

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
                supported_final(cell): float(MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS),
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

    for line, members in line_groups.items():
        constraints.add(
            {
                development_line(line): 1.0,
                **{development(group): -1.0 for group in members},
            },
            upper=0.0,
        )
        constraints.add(
            {
                final_line(line): 1.0,
                **{final(group): -1.0 for group in members},
            },
            upper=0.0,
        )

    lower_bounds = geometry.np.zeros(variable_count, dtype=float)
    upper_bounds = geometry.np.ones(variable_count, dtype=float)
    shared_groups: set[int] = set()
    for group, sample in enumerate(groups):
        if str(group_metadata[sample]["fMOST_brain_id"]) in SHARED_EP11_FMOST_IDS:
            shared_groups.add(group)
            upper_bounds[development(group)] = 0.0
            upper_bounds[final(group)] = 0.0
    if len(shared_groups) != len(SHARED_EP11_FMOST_IDS):
        raise RuntimeError("did not resolve all 15 shared groups")

    line_objective = geometry.np.zeros(variable_count, dtype=float)
    for line in lines:
        line_objective[development_line(line)] = -10.0
        line_objective[final_line(line)] = -100.0

    linear_constraint = constraints.linear_constraint()
    line_result = geometry.milp(
        c=line_objective,
        integrality=geometry.np.ones(variable_count, dtype=geometry.np.uint8),
        bounds=geometry.Bounds(lower_bounds, upper_bounds),
        constraints=linear_constraint,
        options={
            "disp": False,
            "time_limit": float(time_limit),
            "mip_rel_gap": 0.0,
        },
    )
    if line_result.status != 0 or line_result.x is None:
        raise RuntimeError(
            "no frozen EP10 role allocation found with all 15 EP11-overlap "
            f"units unassigned: solver status {line_result.status}"
        )
    line_values = geometry.np.rint(line_result.x).astype(int)
    if geometry.np.max(geometry.np.abs(line_result.x - line_values)) > 1e-6:
        raise RuntimeError("line-coverage MILP returned a nonintegral witness")
    maximum_development_lines = int(
        sum(line_values[development_line(line)] for line in lines)
    )
    maximum_final_lines = int(sum(line_values[final_line(line)] for line in lines))
    constraints.add(
        {development_line(line): 1.0 for line in lines},
        maximum_development_lines,
        maximum_development_lines,
    )
    constraints.add(
        {final_line(line): 1.0 for line in lines},
        maximum_final_lines,
        maximum_final_lines,
    )

    rng = random.Random(seed)
    priority_weights = rng.sample(range(1, 10_000_000), 2 * group_count)
    objective = geometry.np.zeros(variable_count, dtype=float)
    for group in range(group_count):
        objective[development(group)] = priority_weights[group]
        objective[final(group)] = priority_weights[group_count + group]

    linear_constraint = constraints.linear_constraint()

    def run_assignment_milp():
        return geometry.milp(
            c=objective,
            integrality=geometry.np.ones(variable_count, dtype=geometry.np.uint8),
            bounds=geometry.Bounds(lower_bounds, upper_bounds),
            constraints=linear_constraint,
            options={
                "disp": False,
                "time_limit": float(time_limit),
                "mip_rel_gap": 0.0,
            },
        )

    result = run_assignment_milp()
    reproducibility_result = run_assignment_milp()
    if (
        result.status != 0
        or result.x is None
        or reproducibility_result.status != 0
        or reproducibility_result.x is None
    ):
        raise RuntimeError(
            "fixed-seed assignment MILP did not reach an exact optimum twice: "
            f"statuses {result.status}, {reproducibility_result.status}"
        )
    values = geometry.np.rint(result.x).astype(int)
    reproducibility_values = geometry.np.rint(reproducibility_result.x).astype(int)
    if geometry.np.max(geometry.np.abs(result.x - values)) > 1e-6:
        raise RuntimeError("role-freeze MILP returned a nonintegral witness")
    if geometry.np.max(
        geometry.np.abs(reproducibility_result.x - reproducibility_values)
    ) > 1e-6:
        raise RuntimeError("role-freeze reproducibility MILP returned a nonintegral witness")
    role_stop = 2 * group_count
    if not geometry.np.array_equal(values[:role_stop], reproducibility_values[:role_stop]):
        raise RuntimeError("fixed-seed role assignment was not identical on immediate rerun")
    evaluated = linear_constraint.A @ values
    finite_lower = geometry.np.isfinite(linear_constraint.lb)
    finite_upper = geometry.np.isfinite(linear_constraint.ub)
    if geometry.np.any(
        evaluated[finite_lower] < linear_constraint.lb[finite_lower] - 1e-7
    ) or geometry.np.any(
        evaluated[finite_upper] > linear_constraint.ub[finite_upper] + 1e-7
    ):
        raise RuntimeError("role-freeze MILP witness violates a constraint")

    role_by_sample: dict[str, str] = {}
    witness_supported_by_sample: dict[str, int] = {}
    for group, sample in enumerate(groups):
        if values[development(group)]:
            role = "development"
            supported = sum(values[supported_development(cell)] for cell in group_cells[group])
        elif values[final(group)]:
            role = "final"
            supported = sum(values[supported_final(cell)] for cell in group_cells[group])
        else:
            role = "unassigned"
            supported = 0
        role_by_sample[sample] = role
        witness_supported_by_sample[sample] = int(supported)

    if sum(role == "development" for role in role_by_sample.values()) != DEVELOPMENT_GROUPS:
        raise RuntimeError("frozen role allocation has the wrong development count")
    if sum(role == "final" for role in role_by_sample.values()) != FINAL_GROUPS:
        raise RuntimeError("frozen role allocation has the wrong final count")
    if any(
        role_by_sample[sample] != "unassigned"
        for sample in groups
        if str(group_metadata[sample]["fMOST_brain_id"]) in SHARED_EP11_FMOST_IDS
    ):
        raise RuntimeError("a shared EP11 unit received an EP10 role")

    selected_development_groups = {
        group_index[sample]
        for sample, role in role_by_sample.items()
        if role == "development"
    }
    supported_by_sample: dict[str, int] = {}
    for group, sample in enumerate(groups):
        supported_by_sample[sample] = sum(
            len(neighbors[cell] & selected_development_groups)
            >= MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
            for cell in group_cells[group]
        )
        if role_by_sample[sample] in {"development", "final"}:
            required = math.ceil(PER_GROUP_RETENTION * len(group_cells[group]) - 1e-12)
            if supported_by_sample[sample] < required:
                raise RuntimeError("a selected group violates actual support retention")
            if witness_supported_by_sample[sample] > supported_by_sample[sample]:
                raise RuntimeError("MILP support witness exceeds actual supported cells")

    for role in ("development", "final"):
        selected_samples = [
            sample for sample, assigned in role_by_sample.items() if assigned == role
        ]
        role_cells = sum(len(group_cells[group_index[sample]]) for sample in selected_samples)
        role_supported = sum(supported_by_sample[sample] for sample in selected_samples)
        if role_supported + 1e-12 < OVERALL_ROLE_RETENTION * role_cells:
            raise RuntimeError(f"{role} role violates actual overall support retention")

    actual_development_lines = sorted(
        {
            str(group_metadata[sample]["strain_or_cre_line"])
            for sample, role in role_by_sample.items()
            if role == "development"
        }
    )
    actual_final_lines = sorted(
        {
            str(group_metadata[sample]["strain_or_cre_line"])
            for sample, role in role_by_sample.items()
            if role == "final"
        }
    )
    if len(actual_development_lines) != maximum_development_lines:
        raise RuntimeError("frozen development line coverage differs from optimized count")
    if len(actual_final_lines) != maximum_final_lines:
        raise RuntimeError("frozen final line coverage differs from optimized count")

    diagnostics = {
        "solver_status_code": int(result.status),
        "solver_status": "exact_optimum_reproduced_assignment_frozen",
        "solver_objective": float(result.fun),
        "maximum_development_line_coverage": maximum_development_lines,
        "maximum_final_line_coverage": maximum_final_lines,
        "immediate_role_rerun_identical": True,
        "development_lines": actual_development_lines,
        "final_lines": actual_final_lines,
    }
    return role_by_sample, supported_by_sample, diagnostics


def atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    if args.output_csv.exists() or args.summary_json.exists():
        raise RuntimeError("refusing to overwrite an existing frozen role artifact")
    if args.milp_time_limit_seconds <= 0:
        raise RuntimeError("MILP time limit must be positive")

    group_metadata = load_group_metadata(args.inventory)
    screen = load_screen_module()
    geometry = screen.load_geometry_module()
    records = screen.load_records(args.inventory, geometry)
    (
        geometry_records,
        exact_depth,
        _api_depth,
        geodesic,
        geometry_diagnostics,
        _graph_diagnostics,
    ) = geometry.build_geometry(records)
    if len(geometry_records) != EXPECTED_CELLS:
        raise RuntimeError("not all 837 cells are atlas-geometry eligible")
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
        raise RuntimeError("primary geometry scope is not 833 cells")
    primary_geodesic = geodesic[geometry.np.ix_(primary_positions, primary_positions)]
    primary_depth = exact_depth[primary_positions]
    groups = sorted({record.brain_id for record in primary_records})
    group_index = {group: index for index, group in enumerate(groups)}
    neighbors = geometry.neighbor_brain_sets(
        primary_records,
        group_index,
        primary_geodesic,
        primary_depth,
        RADIUS_UM,
        DEPTH_TOLERANCE,
    )
    candidate_cells = sum(
        len(values) >= MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
        for values in neighbors
    )
    if candidate_cells != 825:
        raise RuntimeError(
            f"the frozen 500-um/0.20 geometry condition expected 825 candidate cells, found {candidate_cells}"
        )

    role_by_sample, supported_by_sample, solver_diagnostics = solve_roles(
        geometry,
        primary_records,
        neighbors,
        group_metadata,
        args.seed,
        args.milp_time_limit_seconds,
    )

    rows: list[dict[str, object]] = []
    for sample, metadata in sorted(
        group_metadata.items(), key=lambda item: int(str(item[1]["fMOST_brain_id"]))
    ):
        fmost = str(metadata["fMOST_brain_id"])
        role = role_by_sample[sample]
        primary_count = int(metadata["primary_cell_count"])
        supported = supported_by_sample[sample]
        rows.append(
            {
                "fMOST_brain_id": fmost,
                "provider_sample_id": sample,
                "strain_or_cre_line": metadata["strain_or_cre_line"],
                "all_cell_count": metadata["all_cell_count"],
                "primary_cell_count": primary_count,
                "EP11_SSp_tr_overlap": "yes" if fmost in SHARED_EP11_FMOST_IDS else "no",
                "episode10_role": role,
                "geometry_supported_primary_cells": supported if role != "unassigned" else "",
                "geometry_supported_fraction": (
                    f"{supported / primary_count:.6f}"
                    if role != "unassigned" and primary_count
                    else ""
                ),
                "geometry_condition": "exact_polyline_radius_500um_depth_tolerance_0.20",
                "animal_identity_status": "unresolved_fMOST_sample_unit",
                "cross_episode_disposition": (
                    "reserved_unassigned_for_EP11" if fmost in SHARED_EP11_FMOST_IDS else "EP10_only"
                ),
                "role_status": "frozen_before_projection_outcomes",
                "projection_outcome_exposed": "no",
            }
        )

    role_primary_totals = {
        role: sum(
            int(group_metadata[sample]["primary_cell_count"])
            for sample, assigned in role_by_sample.items()
            if assigned == role
        )
        for role in ("development", "final")
    }
    role_supported_totals = {
        role: sum(
            supported_by_sample[sample]
            for sample, assigned in role_by_sample.items()
            if assigned == role
        )
        for role in ("development", "final")
    }
    summary = {
        "status": "frozen_before_projection_outcomes",
        "input_cells": EXPECTED_CELLS,
        "primary_cells": EXPECTED_PRIMARY_CELLS,
        "released_fMOST_sample_units": EXPECTED_GROUPS,
        "geometry_eligible_cells": geometry_diagnostics["geometry_eligible_cells"],
        "geometry_condition": {
            "depth_projection_convention": "exact_polyline_closest_point_primary",
            "tangential_radius_um": RADIUS_UM,
            "normalized_depth_tolerance": DEPTH_TOLERANCE,
            "candidate_cells_with_2plus_neighbor_groups": candidate_cells,
            "selection_rule": "radius_first_lexicographic_over_prespecified_finite_grid",
            "pareto_alternative_not_selected": {
                "tangential_radius_um": 750,
                "normalized_depth_tolerance": 0.10,
            },
        },
        "roles": {
            "development_groups": DEVELOPMENT_GROUPS,
            "final_groups": FINAL_GROUPS,
            "unassigned_groups": EXPECTED_GROUPS - DEVELOPMENT_GROUPS - FINAL_GROUPS,
            "development_primary_cells": role_primary_totals["development"],
            "development_supported_primary_cells": role_supported_totals["development"],
            "development_supported_fraction": (
                role_supported_totals["development"] / role_primary_totals["development"]
            ),
            "final_primary_cells": role_primary_totals["final"],
            "final_supported_primary_cells": role_supported_totals["final"],
            "final_supported_fraction": (
                role_supported_totals["final"] / role_primary_totals["final"]
            ),
        },
        "constraints": {
            "minimum_per_group_retention": PER_GROUP_RETENTION,
            "minimum_overall_retention_within_each_role": OVERALL_ROLE_RETENTION,
            "minimum_development_groups_with_2plus_cells": MINIMUM_MULTI_CELL_DEVELOPMENT_GROUPS,
            "distinct_development_neighbor_groups_per_retained_cell": MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS,
            "shared_EP11_units_forced_unassigned": len(SHARED_EP11_FMOST_IDS),
        },
        "shared_EP11_fMOST_ids": sorted(SHARED_EP11_FMOST_IDS, key=int),
        "assignment_policy": {
            "seed": args.seed,
            "tie_break": "metadata_only_fixed_seed_integer_priority_exact_mip_gap",
            "transgenic_line_coverage": "soft_lexicographic_final_then_development_fixed_before_assignment_tie_break",
        },
        "solver": solver_diagnostics,
        "role_assignment_emitted": True,
        "projection_outcomes_opened": False,
        "axon_tree_rows_opened": False,
        "independent_animal_identity_certified": False,
    }
    atomic_write_csv(args.output_csv, rows)
    atomic_write_json(args.summary_json, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
