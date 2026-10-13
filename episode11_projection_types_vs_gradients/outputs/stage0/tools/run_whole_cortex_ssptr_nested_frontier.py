#!/usr/bin/env python3
"""Enumerate the outcome-blind EP11 nested role-count frontier.

For each addition pair ``(x, y)`` with ``x >= 0``, ``y >= 0``, and
``x + y <= 15``, this runner asks whether one hidden allocation can satisfy
both a D0=12/A0=8 baseline and its nested expansion
``D=(D0 + D_add[x])`` / ``A=(A0 + A_add[y])``.  The 35 non-singleton
brain/sample units are partitioned into D0, A0, D_add, A_add, and unassigned;
the singleton Tle4 unit is unassigned.

Every pair is solved independently, and the exact-polyline and LineString3D
depth conventions are also solved independently.  Feasible rows therefore do
not imply that one role assignment works for another count pair or depth
convention.  Only aggregate counts and feasibility statuses are written; role
identities and MILP witnesses are discarded in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode11_projection_types_vs_gradients"
)
DEFAULT_INVENTORY = EPISODE_ROOT / "outputs/stage0/whole_cortex_ssptr_inventory.csv"
DEFAULT_OUTPUT_CSV = (
    EPISODE_ROOT
    / "outputs/stage0/whole_cortex_ssptr_nested_frontier_sensitivity.csv"
)
DEFAULT_OUTPUT_MD = (
    EPISODE_ROOT
    / "outputs/stage0/whole_cortex_ssptr_nested_frontier_sensitivity.md"
)

SOURCE_REGION = "SSp-tr"
PRIMARY_CELLS = 206
TOTAL_UNITS = 36
NON_SINGLETON_UNITS = 35
NON_SINGLETON_LINES = 8
SINGLETON_LINE = "Tle4-2A-CreER"
BASELINE_DEVELOPMENT_UNITS = 12
BASELINE_AUDIT_UNITS = 8
MAXIMUM_ADDITIONS = 15
TANGENTIAL_RADIUS_UM = 500
DEPTH_TOLERANCE = 0.10
MINIMUM_MULTI_CELL_DEVELOPMENT_UNITS = 8
MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS = 2
MINIMUM_PER_ROLE_UNIT_RETENTION = 0.90
MINIMUM_OVERALL_ROLE_RETENTION = 0.95


class FrontierError(RuntimeError):
    """Raised when a fixed contract or solver invariant is violated."""


@dataclass(frozen=True)
class SupportCell:
    """Outcome-blind fields used internally by the role MILP."""

    cell_id: str
    unit_id: str
    line: str


class ConstraintRows:
    """Construct one sparse SciPy ``LinearConstraint``."""

    def __init__(self, variable_count: int):
        self.variable_count = variable_count
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(
        self,
        coefficients: Mapping[int, float],
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.columns.append(int(column))
                self.values.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def linear_constraint(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.lower), self.variable_count),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


class VariableBook:
    """Allocate named blocks of binary variables."""

    def __init__(self) -> None:
        self.size = 0
        self.blocks: dict[str, np.ndarray] = {}

    def add(self, name: str, length: int) -> np.ndarray:
        if name in self.blocks:
            raise FrontierError(f"duplicate variable block: {name}")
        result = np.arange(self.size, self.size + length, dtype=int)
        self.blocks[name] = result
        self.size += length
        return result


def add_coefficients(
    target: dict[int, float], expression: Mapping[int, float], scale: float = 1.0
) -> None:
    """Accumulate a sparse linear expression in place."""

    for variable, coefficient in expression.items():
        target[int(variable)] = target.get(int(variable), 0.0) + scale * coefficient


def scalar_expression(variable: int) -> dict[int, float]:
    return {int(variable): 1.0}


def sum_expressions(expressions: Iterable[Mapping[int, float]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for expression in expressions:
        add_coefficients(result, expression)
    return result


def count_pairs() -> tuple[tuple[int, int], ...]:
    """Return all 136 registered addition-count pairs."""

    return tuple(
        (development_addition, audit_addition)
        for development_addition in range(MAXIMUM_ADDITIONS + 1)
        for audit_addition in range(
            MAXIMUM_ADDITIONS - development_addition + 1
        )
    )


def validate_pair(development_addition: int, audit_addition: int) -> None:
    if isinstance(development_addition, bool) or not isinstance(
        development_addition, int
    ):
        raise FrontierError("development addition must be an integer")
    if isinstance(audit_addition, bool) or not isinstance(audit_addition, int):
        raise FrontierError("audit addition must be an integer")
    if development_addition < 0 or audit_addition < 0:
        raise FrontierError("addition counts must be nonnegative")
    if development_addition + audit_addition > MAXIMUM_ADDITIONS:
        raise FrontierError("addition counts exceed the 15-unit reserve")


def validate_problem(
    cells: Sequence[SupportCell],
    neighbor_units: Sequence[set[int]],
    singleton_line: str,
) -> tuple[
    list[str],
    dict[str, int],
    dict[int, list[int]],
    dict[int, str],
    dict[str, list[int]],
]:
    """Validate outcome-blind unit, line, and neighborhood structure."""

    if len(cells) != len(neighbor_units) or not cells:
        raise FrontierError("cell and neighborhood arrays must be nonempty and aligned")
    units = sorted({cell.unit_id for cell in cells})
    if len(units) != TOTAL_UNITS:
        raise FrontierError("the frontier requires exactly 36 units")
    unit_index = {unit: index for index, unit in enumerate(units)}
    unit_cells: dict[int, list[int]] = defaultdict(list)
    unit_line: dict[int, str] = {}
    for cell_index, cell in enumerate(cells):
        owner = unit_index[cell.unit_id]
        unit_cells[owner].append(cell_index)
        previous = unit_line.setdefault(owner, cell.line)
        if previous != cell.line:
            raise FrontierError("one unit is associated with multiple lines")
        if owner in neighbor_units[cell_index]:
            raise FrontierError("a cell cannot use its own unit as a neighbor")
        if any(unit < 0 or unit >= len(units) for unit in neighbor_units[cell_index]):
            raise FrontierError("a neighbor-unit index is out of range")

    line_units: dict[str, list[int]] = defaultdict(list)
    for unit, line in unit_line.items():
        line_units[line].append(unit)
    singleton_lines = sorted(
        line for line, members in line_units.items() if len(members) == 1
    )
    if singleton_lines != [singleton_line]:
        raise FrontierError("the expected sole singleton line was not recovered")
    non_singleton_lines = [line for line in line_units if line != singleton_line]
    if len(non_singleton_lines) != NON_SINGLETON_LINES:
        raise FrontierError("the cohort must have eight non-singleton lines")
    if sum(len(line_units[line]) for line in non_singleton_lines) != NON_SINGLETON_UNITS:
        raise FrontierError("the non-singleton lines must contain exactly 35 units")
    if any(len(line_units[line]) < 2 for line in non_singleton_lines):
        raise FrontierError("each non-singleton line needs baseline D0 and A0 support")
    if sum(len(indices) >= 2 for indices in unit_cells.values()) < (
        MINIMUM_MULTI_CELL_DEVELOPMENT_UNITS
    ):
        raise FrontierError("fewer than eight units contain at least two cells")
    return units, unit_index, unit_cells, unit_line, line_units


PUBLIC_SOLVER_KEYS = frozenset(
    {
        "solver_status_code",
        "solver_status",
        "nested_pair_feasible",
        "binary_variables",
        "linear_constraints",
        "role_assignment_emitted",
    }
)


def validate_aggregate_solver_result(result: Mapping[str, object]) -> None:
    """Prohibit a role vector or other solver witness in returned results."""

    if set(result) != PUBLIC_SOLVER_KEYS:
        raise FrontierError("solver result contains a non-aggregate field")
    if result["role_assignment_emitted"] is not False:
        raise FrontierError("role assignments must never be emitted")
    for value in result.values():
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            raise FrontierError("solver result contains a structured witness")


def solve_nested_pair(
    cells: Sequence[SupportCell],
    neighbor_units: Sequence[set[int]],
    development_addition: int,
    audit_addition: int,
    *,
    singleton_line: str = SINGLETON_LINE,
    time_limit_seconds: float,
    required_unassigned_units: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Solve one pair and return aggregate feasibility only."""

    validate_pair(development_addition, audit_addition)
    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise FrontierError("MILP time limit must be finite and positive")
    (
        units,
        unit_index,
        unit_cells,
        _unit_line,
        line_units,
    ) = validate_problem(cells, neighbor_units, singleton_line)
    unknown_required = required_unassigned_units - set(units)
    if unknown_required:
        raise FrontierError("a required-unassigned unit is absent")

    unit_count = len(units)
    cell_count = len(cells)
    unassigned_count = (
        MAXIMUM_ADDITIONS - development_addition - audit_addition
    )
    variables = VariableBook()
    d0 = variables.add("baseline_development", unit_count)
    a0 = variables.add("baseline_audit", unit_count)
    d_add = variables.add("added_development", unit_count)
    a_add = variables.add("added_audit", unit_count)
    unassigned = variables.add("unassigned", unit_count)
    supported_baseline_development = variables.add(
        "supported_baseline_development", cell_count
    )
    supported_baseline_audit = variables.add("supported_baseline_audit", cell_count)
    supported_expanded_development = variables.add(
        "supported_expanded_development", cell_count
    )
    supported_expanded_audit = variables.add("supported_expanded_audit", cell_count)

    d0_expr = [scalar_expression(value) for value in d0]
    a0_expr = [scalar_expression(value) for value in a0]
    expanded_development_expr = [
        sum_expressions((scalar_expression(d0[index]), scalar_expression(d_add[index])))
        for index in range(unit_count)
    ]
    expanded_audit_expr = [
        sum_expressions((scalar_expression(a0[index]), scalar_expression(a_add[index])))
        for index in range(unit_count)
    ]
    scenarios = (
        (
            "baseline",
            d0_expr,
            a0_expr,
            supported_baseline_development,
            supported_baseline_audit,
            BASELINE_DEVELOPMENT_UNITS,
            BASELINE_AUDIT_UNITS,
        ),
        (
            "expanded",
            expanded_development_expr,
            expanded_audit_expr,
            supported_expanded_development,
            supported_expanded_audit,
            BASELINE_DEVELOPMENT_UNITS + development_addition,
            BASELINE_AUDIT_UNITS + audit_addition,
        ),
    )

    constraints = ConstraintRows(variables.size)
    constraints.add(
        {int(value): 1.0 for value in d0},
        BASELINE_DEVELOPMENT_UNITS,
        BASELINE_DEVELOPMENT_UNITS,
    )
    constraints.add(
        {int(value): 1.0 for value in a0},
        BASELINE_AUDIT_UNITS,
        BASELINE_AUDIT_UNITS,
    )
    constraints.add(
        {int(value): 1.0 for value in d_add},
        development_addition,
        development_addition,
    )
    constraints.add(
        {int(value): 1.0 for value in a_add}, audit_addition, audit_addition
    )
    constraints.add(
        {int(value): 1.0 for value in unassigned},
        unassigned_count,
        unassigned_count,
    )

    singleton_unit = line_units[singleton_line][0]
    for unit in range(unit_count):
        role_partition = {
            int(d0[unit]): 1.0,
            int(a0[unit]): 1.0,
            int(d_add[unit]): 1.0,
            int(a_add[unit]): 1.0,
            int(unassigned[unit]): 1.0,
        }
        if unit == singleton_unit:
            constraints.add(role_partition, 0.0, 0.0)
        else:
            constraints.add(role_partition, 1.0, 1.0)
    for unit_id in required_unassigned_units:
        constraints.add(
            {int(unassigned[unit_index[unit_id]]): 1.0}, 1.0, 1.0
        )

    multi_cell_units = [
        unit for unit, indices in unit_cells.items() if len(indices) >= 2
    ]
    for (
        _scenario_name,
        development_expr,
        audit_expr,
        supported_development,
        supported_audit,
        expected_development,
        expected_audit,
    ) in scenarios:
        constraints.add(
            sum_expressions(development_expr),
            expected_development,
            expected_development,
        )
        constraints.add(
            sum_expressions(audit_expr), expected_audit, expected_audit
        )

        # Both baseline and expanded roles must represent every non-singleton
        # line. Expanded representation is implied by nesting, but is kept as
        # an explicit auditable constraint.
        for line, members in sorted(line_units.items()):
            if line == singleton_line:
                continue
            constraints.add(
                sum_expressions(development_expr[unit] for unit in members),
                lower=1.0,
            )
            constraints.add(
                sum_expressions(audit_expr[unit] for unit in members), lower=1.0
            )

        for cell_index, cell in enumerate(cells):
            owner = unit_index[cell.unit_id]
            development_owner = {int(supported_development[cell_index]): 1.0}
            add_coefficients(development_owner, development_expr[owner], -1.0)
            constraints.add(development_owner, upper=0.0)
            audit_owner = {int(supported_audit[cell_index]): 1.0}
            add_coefficients(audit_owner, audit_expr[owner], -1.0)
            constraints.add(audit_owner, upper=0.0)

            development_neighbors = sum_expressions(
                development_expr[unit] for unit in neighbor_units[cell_index]
            )
            development_support = {
                int(supported_development[cell_index]): float(
                    MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                )
            }
            add_coefficients(development_support, development_neighbors, -1.0)
            constraints.add(development_support, upper=0.0)
            audit_support = {
                int(supported_audit[cell_index]): float(
                    MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
                )
            }
            add_coefficients(audit_support, development_neighbors, -1.0)
            constraints.add(audit_support, upper=0.0)

        for unit, cell_indices in unit_cells.items():
            required = math.ceil(
                MINIMUM_PER_ROLE_UNIT_RETENTION * len(cell_indices) - 1e-12
            )
            development_retention = {
                int(supported_development[cell]): 1.0 for cell in cell_indices
            }
            add_coefficients(
                development_retention,
                development_expr[unit],
                -float(required),
            )
            constraints.add(development_retention, lower=0.0)
            audit_retention = {
                int(supported_audit[cell]): 1.0 for cell in cell_indices
            }
            add_coefficients(
                audit_retention, audit_expr[unit], -float(required)
            )
            constraints.add(audit_retention, lower=0.0)

        overall_development = {
            int(supported_development[cell]): 1.0 for cell in range(cell_count)
        }
        overall_audit = {
            int(supported_audit[cell]): 1.0 for cell in range(cell_count)
        }
        for unit, cell_indices in unit_cells.items():
            add_coefficients(
                overall_development,
                development_expr[unit],
                -MINIMUM_OVERALL_ROLE_RETENTION * len(cell_indices),
            )
            add_coefficients(
                overall_audit,
                audit_expr[unit],
                -MINIMUM_OVERALL_ROLE_RETENTION * len(cell_indices),
            )
        constraints.add(overall_development, lower=0.0)
        constraints.add(overall_audit, lower=0.0)
        constraints.add(
            sum_expressions(development_expr[unit] for unit in multi_cell_units),
            lower=float(MINIMUM_MULTI_CELL_DEVELOPMENT_UNITS),
        )

    lower_bounds = np.zeros(variables.size, dtype=float)
    upper_bounds = np.ones(variables.size, dtype=float)
    linear_constraint = constraints.linear_constraint()
    solver = milp(
        c=np.zeros(variables.size, dtype=float),
        integrality=np.ones(variables.size, dtype=np.uint8),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=linear_constraint,
        options={"disp": False, "time_limit": float(time_limit_seconds)},
    )

    if solver.status == 0 and solver.x is not None:
        rounded = np.rint(solver.x)
        if np.max(np.abs(solver.x - rounded)) > 1e-6:
            raise FrontierError("MILP returned a non-integral feasible point")
        values = rounded.astype(int)
        evaluated = linear_constraint.A @ values
        finite_lower = np.isfinite(linear_constraint.lb)
        finite_upper = np.isfinite(linear_constraint.ub)
        if np.any(
            evaluated[finite_lower] < linear_constraint.lb[finite_lower] - 1e-7
        ) or np.any(
            evaluated[finite_upper] > linear_constraint.ub[finite_upper] + 1e-7
        ):
            raise FrontierError("MILP witness violates a linear constraint")
        if int(values[d0].sum()) != BASELINE_DEVELOPMENT_UNITS:
            raise FrontierError("MILP witness has the wrong D0 count")
        if int(values[a0].sum()) != BASELINE_AUDIT_UNITS:
            raise FrontierError("MILP witness has the wrong A0 count")
        if int(values[d_add].sum()) != development_addition:
            raise FrontierError("MILP witness has the wrong D_add count")
        if int(values[a_add].sum()) != audit_addition:
            raise FrontierError("MILP witness has the wrong A_add count")
        if int(values[unassigned].sum()) != unassigned_count:
            raise FrontierError("MILP witness has the wrong unassigned count")
        if any(values[unassigned[unit_index[unit_id]]] != 1 for unit_id in required_unassigned_units):
            raise FrontierError("MILP witness violates required unassignment")
        status = "feasible"
        feasible = "yes"
    elif solver.status == 2:
        status = "infeasible"
        feasible = "no"
    else:
        status = "indeterminate"
        feasible = "unknown"

    # The solver vector is deliberately not retained beyond validation.
    result: dict[str, object] = {
        "solver_status_code": int(solver.status),
        "solver_status": status,
        "nested_pair_feasible": feasible,
        "binary_variables": int(variables.size),
        "linear_constraints": int(len(constraints.lower)),
        "role_assignment_emitted": False,
    }
    validate_aggregate_solver_result(result)
    return result


def build_neighbor_units(
    records: Sequence[object],
    geodesic: np.ndarray,
    depth_fraction: np.ndarray,
) -> list[set[int]]:
    """Build registered direct cross-unit neighborhoods."""

    cell_count = len(records)
    if geodesic.shape != (cell_count, cell_count):
        raise FrontierError("geodesic matrix has the wrong shape")
    if depth_fraction.shape != (cell_count,) or not np.all(np.isfinite(depth_fraction)):
        raise FrontierError("depth vector must be finite and cell-aligned")
    units = sorted({str(record.brain_id) for record in records})
    unit_index = {unit: index for index, unit in enumerate(units)}
    neighbors = [set() for _ in records]
    for left in range(cell_count):
        for right in range(left + 1, cell_count):
            if records[left].brain_id == records[right].brain_id:
                continue
            if geodesic[left, right] > TANGENTIAL_RADIUS_UM + 1e-9:
                continue
            if abs(depth_fraction[left] - depth_fraction[right]) > (
                DEPTH_TOLERANCE + 1e-12
            ):
                continue
            neighbors[left].add(unit_index[str(records[right].brain_id)])
            neighbors[right].add(unit_index[str(records[left].brain_id)])
    return neighbors


def load_primary_geometry(
    inventory: Path,
) -> tuple[list[object], np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """Reuse the validated outcome-blind geometry construction."""

    tool_dir = Path(__file__).resolve().parent
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))
    import run_whole_cortex_ssptr_nested_role_sensitivity as nested  # noqa: PLC0415

    records, exact_depth, linestring_depth, geodesic, diagnostics = (
        nested.load_primary_geometry(inventory)
    )
    if len(records) != PRIMARY_CELLS:
        raise FrontierError("primary geometry does not contain 206 cells")
    if len({record.brain_id for record in records}) != TOTAL_UNITS:
        raise FrontierError("primary geometry does not preserve all 36 units")
    return records, exact_depth, linestring_depth, geodesic, diagnostics


def ensure_aggregate_row(row: Mapping[str, object]) -> None:
    """Reject fields or values that could encode a hidden role assignment."""

    forbidden_fragments = (
        "unit_id",
        "cell_id",
        "brain_id",
        "sample_id",
        "selected_id",
        "role_ids",
        "witness",
    )
    for key, value in row.items():
        if any(fragment in key.lower() for fragment in forbidden_fragments):
            raise FrontierError(f"forbidden output field: {key}")
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            raise FrontierError(f"non-aggregate output value in {key}")
    if row.get("role_assignment_emitted") not in (False, "no"):
        raise FrontierError("an output row does not prohibit role identities")


def pareto_pairs(feasible_pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Return coordinate-wise maximal pairs among proved-feasible pairs."""

    return {
        pair
        for pair in feasible_pairs
        if not any(
            other[0] >= pair[0]
            and other[1] >= pair[1]
            and other != pair
            for other in feasible_pairs
        )
    }


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize one depth convention using aggregate count pairs only."""

    if not rows:
        raise FrontierError("cannot summarize an empty convention")
    feasible = {
        (int(row["development_addition_units"]), int(row["audit_addition_units"]))
        for row in rows
        if row["nested_pair_feasible"] == "yes"
    }
    frontier = pareto_pairs(feasible)
    development_only = [x for x, y in feasible if y == 0]
    audit_only = [y for x, y in feasible if x == 0]
    maximum_total = max((x + y for x, y in feasible), default=None)
    return {
        "depth_projection_convention": rows[0]["depth_projection_convention"],
        "pairs_tested": len(rows),
        "feasible_pairs": len(feasible),
        "infeasible_pairs": sum(row["nested_pair_feasible"] == "no" for row in rows),
        "indeterminate_pairs": sum(
            row["nested_pair_feasible"] == "unknown" for row in rows
        ),
        "maximum_development_only_addition": (
            max(development_only) if development_only else None
        ),
        "maximum_audit_only_addition": max(audit_only) if audit_only else None,
        "maximum_total_additions": maximum_total,
        "maximum_total_expanded_assigned_units": (
            BASELINE_DEVELOPMENT_UNITS
            + BASELINE_AUDIT_UNITS
            + maximum_total
            if maximum_total is not None
            else None
        ),
        "pareto_pair_count": len(frontier),
        "pareto_addition_pairs": "; ".join(
            f"({x},{y})" for x, y in sorted(frontier)
        ),
        "pareto_expanded_role_counts": "; ".join(
            f"(D{BASELINE_DEVELOPMENT_UNITS + x},A{BASELINE_AUDIT_UNITS + y})"
            for x, y in sorted(frontier)
        ),
    }


def format_optional(value: object) -> str:
    return "none" if value is None else str(value)


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, object],
    output_csv: Path,
    output_md: Path,
    *,
    protected_identifiers: Iterable[str] = (),
) -> None:
    """Write aggregate pair rows and an aggregate frontier report."""

    if not rows or not summaries:
        raise FrontierError("no frontier results were produced")
    for row in rows:
        ensure_aggregate_row(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# EP11 SSp-tr nested role-count frontier",
        "",
        "This outcome-blind diagnostic enumerates all 136 additions (x,y) with",
        "x>=0, y>=0, and x+y<=15 at 500 um / 0.10 normalized depth. For each",
        "pair, one hidden D0=12/A0=8 baseline and its expanded D=(12+x) and",
        "A=(8+y) roles must simultaneously satisfy the registered geometry rules.",
        "It reads soma geometry, brain/sample-unit identity, and strain/Cre line",
        "only; it does not read a regional projection outcome.",
        "",
        "Each count pair is solved independently, so feasible pairs may use",
        "different hidden allocations. The two depth conventions are also solved",
        "independently. This report does not claim one role assignment works across",
        "different pairs or across both depth conventions. No role identity or MILP",
        "witness is emitted.",
        "",
        "## Fixed rules",
        "",
        "- The 35 non-singleton units are partitioned into D0, A0, D_add, A_add,",
        "  and unassigned; the singleton Tle4 unit is unassigned.",
        "- Baseline and expanded roles separately require two direct development",
        "  neighbors per retained cell, at least 0.90 retention per assigned unit,",
        "  at least 0.95 retention in each role overall, at least eight multi-cell",
        "  development units, and every non-singleton line in both roles.",
        "- The Pareto set below is coordinate-wise maximal among proved-feasible",
        "  count pairs only. Any indeterminate row limits interpretation.",
        "",
        "## Aggregate geometry",
        "",
        f"- Primary cells / brain-sample units: {diagnostics['primary_cells']} / "
        f"{diagnostics['geometry_eligible_units']}.",
        f"- Geometry-excluded primary cells: "
        f"{diagnostics['geometry_excluded_primary_cells']}.",
        "- Exact-versus-LineString3D normalized-depth absolute difference",
        f"  median / maximum: "
        f"{float(diagnostics['exact_vs_linestring_depth_absolute_difference_median']):.4f} / "
        f"{float(diagnostics['exact_vs_linestring_depth_absolute_difference_max']):.4f}.",
        "",
        "## Aggregate frontier",
        "",
        "| Depth convention | Tested | Feasible | Infeasible | Indeterminate | Max dev-only x | Max audit-only y | Max x+y | Max assigned | Pareto pairs | Pareto additions (x,y) | Expanded counts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['depth_projection_convention']} | "
            f"{summary['pairs_tested']} | {summary['feasible_pairs']} | "
            f"{summary['infeasible_pairs']} | {summary['indeterminate_pairs']} | "
            f"{format_optional(summary['maximum_development_only_addition'])} | "
            f"{format_optional(summary['maximum_audit_only_addition'])} | "
            f"{format_optional(summary['maximum_total_additions'])} | "
            f"{format_optional(summary['maximum_total_expanded_assigned_units'])} | "
            f"{summary['pareto_pair_count']} | "
            f"{summary['pareto_addition_pairs'] or 'none'} | "
            f"{summary['pareto_expanded_role_counts'] or 'none'} |"
        )
    lines.extend(
        (
            "",
            "The companion CSV contains all aggregate pairwise statuses. Feasibility",
            "does not choose or lock roles, establish animal independence, authorize",
            "projection-outcome access, or support either groups or a continuum.",
        )
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    combined = output_csv.read_text(encoding="utf-8") + output_md.read_text(
        encoding="utf-8"
    )
    leaked = [identifier for identifier in protected_identifiers if identifier in combined]
    if leaked:
        raise FrontierError("an aggregate output leaked a protected identifier")


def build_output_row(
    convention: str,
    development_addition: int,
    audit_addition: int,
    candidate_supported_cells: int,
    solver: Mapping[str, object],
) -> dict[str, object]:
    """Construct one scalar-only public row."""

    row: dict[str, object] = {
        "source_region": SOURCE_REGION,
        "cohort": "Gao_whole_cortex_projectome",
        "analysis_track": "primary_206_axon_dendrite_distinguished",
        "depth_projection_convention": convention,
        "tangential_radius_um": TANGENTIAL_RADIUS_UM,
        "normalized_depth_tolerance": f"{DEPTH_TOLERANCE:.2f}",
        "allocation_scope": "separate_hidden_allocation_per_pair_and_convention",
        "geometry_eligible_cells": PRIMARY_CELLS,
        "brain_sample_units": TOTAL_UNITS,
        "non_singleton_units_partitioned": NON_SINGLETON_UNITS,
        "singleton_units_forced_unassigned": 1,
        "cells_with_2plus_candidate_neighbor_units": candidate_supported_cells,
        "baseline_development_units": BASELINE_DEVELOPMENT_UNITS,
        "baseline_audit_units": BASELINE_AUDIT_UNITS,
        "development_addition_units": development_addition,
        "audit_addition_units": audit_addition,
        "expanded_development_units": (
            BASELINE_DEVELOPMENT_UNITS + development_addition
        ),
        "expanded_audit_units": BASELINE_AUDIT_UNITS + audit_addition,
        "unassigned_non_singleton_units": (
            MAXIMUM_ADDITIONS - development_addition - audit_addition
        ),
        "expanded_assigned_units": (
            BASELINE_DEVELOPMENT_UNITS
            + BASELINE_AUDIT_UNITS
            + development_addition
            + audit_addition
        ),
        "minimum_distinct_development_neighbor_units": (
            MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
        ),
        "minimum_per_role_unit_retention": (
            f"{MINIMUM_PER_ROLE_UNIT_RETENTION:.2f}"
        ),
        "minimum_overall_role_retention": (
            f"{MINIMUM_OVERALL_ROLE_RETENTION:.2f}"
        ),
        "minimum_multi_cell_development_units": (
            MINIMUM_MULTI_CELL_DEVELOPMENT_UNITS
        ),
        **solver,
        "pareto_maximal_among_proved_feasible": "pending",
    }
    ensure_aggregate_row(row)
    return row


def annotate_pareto(rows: Sequence[dict[str, object]]) -> None:
    """Mark the proved-feasible coordinate-wise frontier in place."""

    by_convention: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_convention[str(row["depth_projection_convention"])].append(row)
    for convention_rows in by_convention.values():
        feasible = {
            (
                int(row["development_addition_units"]),
                int(row["audit_addition_units"]),
            )
            for row in convention_rows
            if row["nested_pair_feasible"] == "yes"
        }
        frontier = pareto_pairs(feasible)
        for row in convention_rows:
            pair = (
                int(row["development_addition_units"]),
                int(row["audit_addition_units"]),
            )
            if row["nested_pair_feasible"] == "yes":
                row["pareto_maximal_among_proved_feasible"] = (
                    "yes" if pair in frontier else "no"
                )
            else:
                row["pareto_maximal_among_proved_feasible"] = "not_applicable"


def build(args: argparse.Namespace) -> dict[str, object]:
    """Run the two independent 136-pair frontiers."""

    records, exact_depth, linestring_depth, geodesic, diagnostics = (
        load_primary_geometry(args.inventory.resolve())
    )
    cells = [
        SupportCell(
            cell_id=str(record.cell_id),
            unit_id=str(record.brain_id),
            line=str(record.line),
        )
        for record in records
    ]
    all_rows: list[dict[str, object]] = []
    conventions = (
        ("exact_polyline_closest_point_primary", exact_depth),
        ("ccf_streamlines_LineString3D_project_sensitivity", linestring_depth),
    )
    for convention, depth in conventions:
        neighbors = build_neighbor_units(records, geodesic, depth)
        candidate_supported = sum(
            len(values) >= MINIMUM_DISTINCT_DEVELOPMENT_NEIGHBORS
            for values in neighbors
        )
        for development_addition, audit_addition in count_pairs():
            solver = solve_nested_pair(
                cells,
                neighbors,
                development_addition,
                audit_addition,
                time_limit_seconds=args.milp_time_limit_seconds,
            )
            all_rows.append(
                build_output_row(
                    convention,
                    development_addition,
                    audit_addition,
                    candidate_supported,
                    solver,
                )
            )
    annotate_pareto(all_rows)
    summaries = [
        summarize_rows(
            [
                row
                for row in all_rows
                if row["depth_projection_convention"] == convention
            ]
        )
        for convention, _depth in conventions
    ]
    write_outputs(
        all_rows,
        summaries,
        diagnostics,
        args.output_csv,
        args.output_md,
    )
    return {
        "status": "complete",
        "pairs_per_depth_convention": len(count_pairs()),
        "depth_conventions": len(conventions),
        "rows": len(all_rows),
        "feasible_rows": sum(row["nested_pair_feasible"] == "yes" for row in all_rows),
        "infeasible_rows": sum(row["nested_pair_feasible"] == "no" for row in all_rows),
        "indeterminate_rows": sum(
            row["nested_pair_feasible"] == "unknown" for row in all_rows
        ),
        "role_assignment_emitted": False,
        "output_csv": str(args.output_csv.resolve()),
        "output_md": str(args.output_md.resolve()),
    }


def make_fixture(
    *, sparse_unit: str | None = None
) -> tuple[list[SupportCell], list[set[int]], str]:
    """Create an asset-free 36-unit fixture with optional sparse geometry."""

    line_sizes = (9, 5, 4, 4, 4, 3, 3, 3)
    cells: list[SupportCell] = []
    unit_ids: list[str] = []
    for line_index, line_size in enumerate(line_sizes):
        for within_line in range(line_size):
            unit = f"fixture_unit_{line_index:02d}_{within_line:02d}"
            unit_ids.append(unit)
            for cell_index in range(2):
                cells.append(
                    SupportCell(
                        cell_id=f"fixture_cell_{line_index:02d}_{within_line:02d}_{cell_index}",
                        unit_id=unit,
                        line=f"fixture_line_{line_index:02d}",
                    )
                )
    singleton_line = "fixture_singleton_line"
    singleton_unit = "fixture_unit_singleton"
    unit_ids.append(singleton_unit)
    for cell_index in range(2):
        cells.append(
            SupportCell(
                cell_id=f"fixture_cell_singleton_{cell_index}",
                unit_id=singleton_unit,
                line=singleton_line,
            )
        )
    sorted_units = sorted(unit_ids)
    unit_index = {unit: index for index, unit in enumerate(sorted_units)}
    neighbors = [
        {
            unit_index[unit]
            for unit in sorted_units
            if unit != cell.unit_id
        }
        for cell in cells
    ]
    if sparse_unit is not None:
        if sparse_unit not in unit_index:
            raise FrontierError("requested sparse fixture unit is absent")
        for index, cell in enumerate(cells):
            if cell.unit_id == sparse_unit:
                neighbors[index] = set()
    return cells, neighbors, singleton_line


def run_self_test() -> dict[str, object]:
    """Exercise all count pairs without inventory or Allen assets."""

    pairs = count_pairs()
    if len(pairs) != 136 or len(set(pairs)) != 136:
        raise FrontierError("the registered count-pair grid is not 136 unique pairs")

    complete_cells, complete_neighbors, singleton_line = make_fixture()
    complete_results: list[dict[str, object]] = []
    for development_addition, audit_addition in pairs:
        result = solve_nested_pair(
            complete_cells,
            complete_neighbors,
            development_addition,
            audit_addition,
            singleton_line=singleton_line,
            time_limit_seconds=10.0,
        )
        if result["nested_pair_feasible"] != "yes":
            raise FrontierError("a complete-neighbor count pair was not feasible")
        complete_results.append(result)

    sparse_unit = "fixture_unit_00_00"
    sparse_cells, sparse_neighbors, sparse_singleton = make_fixture(
        sparse_unit=sparse_unit
    )
    sparse_unassigned = solve_nested_pair(
        sparse_cells,
        sparse_neighbors,
        0,
        0,
        singleton_line=sparse_singleton,
        time_limit_seconds=10.0,
        required_unassigned_units=frozenset({sparse_unit}),
    )
    if sparse_unassigned["nested_pair_feasible"] != "yes":
        raise FrontierError("the sparse unit could not remain unassigned")
    sparse_forced_assigned = solve_nested_pair(
        sparse_cells,
        sparse_neighbors,
        MAXIMUM_ADDITIONS,
        0,
        singleton_line=sparse_singleton,
        time_limit_seconds=10.0,
    )
    if sparse_forced_assigned["nested_pair_feasible"] != "no":
        raise FrontierError("an all-assigned design did not reject the sparse unit")

    with tempfile.TemporaryDirectory() as temporary:
        output_csv = Path(temporary) / "frontier.csv"
        output_md = Path(temporary) / "frontier.md"
        rows = [
            build_output_row(
                "synthetic_complete",
                development_addition,
                audit_addition,
                len(complete_cells),
                result,
            )
            for (development_addition, audit_addition), result in zip(
                pairs, complete_results, strict=True
            )
        ]
        annotate_pareto(rows)
        summaries = [summarize_rows(rows)]
        diagnostics = {
            "primary_cells": len(complete_cells),
            "geometry_eligible_units": TOTAL_UNITS,
            "geometry_excluded_primary_cells": 0,
            "exact_vs_linestring_depth_absolute_difference_median": 0.0,
            "exact_vs_linestring_depth_absolute_difference_max": 0.0,
        }
        protected = {cell.unit_id for cell in complete_cells} | {
            cell.cell_id for cell in complete_cells
        }
        write_outputs(
            rows,
            summaries,
            diagnostics,
            output_csv,
            output_md,
            protected_identifiers=protected,
        )

    return {
        "status": "passed",
        "registered_pairs": len(pairs),
        "complete_neighbor_feasible_pairs": len(complete_results),
        "sparse_unit_can_remain_unassigned": True,
        "sparse_unit_all_35_assigned": "infeasible",
        "role_assignment_emitted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=120.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if (
        not math.isfinite(args.milp_time_limit_seconds)
        or args.milp_time_limit_seconds <= 0
    ):
        parser.error("--milp-time-limit-seconds must be finite and positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = run_self_test() if arguments.self_test else build(arguments)
    print(json.dumps(result, sort_keys=True))
