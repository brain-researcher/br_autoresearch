#!/usr/bin/env python3
"""Outcome-blind nested role-design sensitivity for EP11 Gao SSp-tr.

This runner reuses the already defined soma-geometry construction, restricts
the analysis to the 206 primary ``Axon_and_dendrite`` reconstructions, and
tests only the registered 500-um/0.10 neighborhood under both depth
projection conventions.  It never reads axon geometry or a regional
projection outcome.

The key calculation is a *joint* MILP.  One common 12-development/8-audit
core and a 15-unit reserve must simultaneously support:

* the baseline 12-development/8-audit design;
* an estimator ceiling with the reserve added to development (27/8); and
* a precision ceiling with the same reserve added to audit (12/23).

A second joint system adds a nested 16-development/19-audit compromise.  Its
development role contains the 12-unit core, its audit role contains the
8-unit core, and it has exactly two development units in each of the eight
non-singleton strain/Cre lines.  The singleton Tle4 unit is unassigned in
every design.

Only aggregate counts and feasibility statuses are written.  Solver role
assignments are validated in memory and discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


REPO_EPISODE = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode11_projection_types_vs_gradients"
)
DEFAULT_INVENTORY = (
    REPO_EPISODE / "outputs/stage0/whole_cortex_ssptr_inventory.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_EPISODE
    / "outputs/stage0/whole_cortex_ssptr_nested_role_design_sensitivity.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_EPISODE
    / "outputs/stage0/whole_cortex_ssptr_nested_role_design_sensitivity.md"
)

SOURCE_REGION = "SSp-tr"
PRIMARY_CELLS = 206
TOTAL_UNITS = 36
SINGLETON_LINE = "Tle4-2A-CreER"
TANGENTIAL_RADIUS_UM = 500
DEPTH_TOLERANCE = 0.10


class RoleSensitivityError(RuntimeError):
    """Raised when a fixed-design or solver invariant is violated."""


@dataclass(frozen=True)
class SupportCell:
    """Only the outcome-blind fields required by the role MILP."""

    cell_id: str
    unit_id: str
    line: str


@dataclass(frozen=True)
class FixedDesign:
    """The prospective counts and support thresholds tested here."""

    core_development_units: int = 12
    core_audit_units: int = 8
    reserve_units: int = 15
    estimator_development_units: int = 27
    estimator_audit_units: int = 8
    precision_development_units: int = 12
    precision_audit_units: int = 23
    compromise_development_units: int = 16
    compromise_audit_units: int = 19
    non_singleton_lines: int = 8
    compromise_development_units_per_line: int = 2
    singleton_units_unassigned: int = 1
    minimum_multi_cell_development_units: int = 8
    minimum_distinct_development_neighbors: int = 2
    minimum_per_role_unit_retention: float = 0.90
    minimum_overall_role_retention: float = 0.95

    def validate_fixed_contract(self) -> None:
        """Reject accidental changes to the sensitivity design."""

        if self.core_development_units != 12:
            raise RoleSensitivityError("the common development core must be 12")
        if self.core_audit_units != 8:
            raise RoleSensitivityError("the common audit core must be 8")
        if self.reserve_units != 15:
            raise RoleSensitivityError("the common reserve must be 15")
        if self.singleton_units_unassigned != 1:
            raise RoleSensitivityError("exactly one singleton unit must be unassigned")
        if self.non_singleton_lines != 8:
            raise RoleSensitivityError("the fixed cohort must have eight non-singleton lines")
        if (
            self.core_development_units
            + self.core_audit_units
            + self.reserve_units
            + self.singleton_units_unassigned
            != TOTAL_UNITS
        ):
            raise RoleSensitivityError("core, reserve, and singleton counts must total 36")
        if self.estimator_development_units != (
            self.core_development_units + self.reserve_units
        ) or self.estimator_audit_units != self.core_audit_units:
            raise RoleSensitivityError("estimator ceiling must be D0+R versus A0")
        if self.precision_development_units != self.core_development_units or (
            self.precision_audit_units
            != self.core_audit_units + self.reserve_units
        ):
            raise RoleSensitivityError("precision ceiling must be D0 versus A0+R")
        assigned = TOTAL_UNITS - self.singleton_units_unassigned
        if (
            self.compromise_development_units + self.compromise_audit_units
            != assigned
        ):
            raise RoleSensitivityError("compromise roles must partition 35 units")
        if (
            self.compromise_development_units != 16
            or self.compromise_audit_units != 19
        ):
            raise RoleSensitivityError("the compromise must be D16/A19")
        if (
            self.non_singleton_lines
            * self.compromise_development_units_per_line
            != self.compromise_development_units
        ):
            raise RoleSensitivityError(
                "the compromise must place exactly two development units in "
                "each non-singleton line"
            )
        if self.minimum_distinct_development_neighbors != 2:
            raise RoleSensitivityError("the direct-neighbor minimum must remain two")
        if self.minimum_multi_cell_development_units != 8:
            raise RoleSensitivityError(
                "the multi-cell development-unit minimum must remain eight"
            )
        if not math.isclose(self.minimum_per_role_unit_retention, 0.90):
            raise RoleSensitivityError("the per-unit retention minimum must remain 0.90")
        if not math.isclose(self.minimum_overall_role_retention, 0.95):
            raise RoleSensitivityError("the role-wide retention minimum must remain 0.95")


FIXED_DESIGN = FixedDesign()


class ConstraintRows:
    """Build one sparse SciPy ``LinearConstraint`` without dense matrices."""

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
    """Allocate named binary-variable blocks."""

    def __init__(self) -> None:
        self.size = 0
        self.blocks: dict[str, np.ndarray] = {}

    def add(self, name: str, length: int) -> np.ndarray:
        if name in self.blocks:
            raise RoleSensitivityError(f"duplicate variable block: {name}")
        values = np.arange(self.size, self.size + length, dtype=int)
        self.blocks[name] = values
        self.size += length
        return values


def add_coefficients(
    target: dict[int, float], expression: Mapping[int, float], scale: float = 1.0
) -> None:
    """Accumulate a sparse linear expression in place."""

    for variable, coefficient in expression.items():
        target[variable] = target.get(variable, 0.0) + scale * coefficient


def scalar_expression(variable: int) -> dict[int, float]:
    return {int(variable): 1.0}


def sum_expressions(expressions: Iterable[Mapping[int, float]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for expression in expressions:
        add_coefficients(result, expression)
    return result


PUBLIC_SOLVER_KEYS = frozenset(
    {
        "solver_status_code",
        "solver_status",
        "joint_nested_allocation_exists",
        "binary_variables",
        "linear_constraints",
        "role_assignment_emitted",
    }
)


def validate_aggregate_solver_result(result: Mapping[str, object]) -> None:
    """Ensure no assignment or other solver witness can reach an output."""

    if set(result) != PUBLIC_SOLVER_KEYS:
        raise RoleSensitivityError("solver result contains a non-aggregate field")
    if result["role_assignment_emitted"] is not False:
        raise RoleSensitivityError("role assignments must never be emitted")
    for value in result.values():
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            raise RoleSensitivityError("solver result contains a structured witness")


def validate_problem(
    cells: Sequence[SupportCell],
    neighbor_units: Sequence[set[int]],
    singleton_line: str,
    design: FixedDesign,
) -> tuple[list[str], dict[str, int], dict[int, list[int]], dict[int, str], dict[str, list[int]]]:
    design.validate_fixed_contract()
    if len(cells) != len(neighbor_units):
        raise RoleSensitivityError("cell and neighbor arrays have different lengths")
    if not cells:
        raise RoleSensitivityError("the support problem has no cells")

    units = sorted({cell.unit_id for cell in cells})
    if len(units) != TOTAL_UNITS:
        raise RoleSensitivityError("the nested sensitivity requires exactly 36 units")
    unit_index = {unit: index for index, unit in enumerate(units)}
    unit_cells: dict[int, list[int]] = defaultdict(list)
    unit_line: dict[int, str] = {}
    for cell_index, cell in enumerate(cells):
        unit = unit_index[cell.unit_id]
        unit_cells[unit].append(cell_index)
        prior = unit_line.setdefault(unit, cell.line)
        if prior != cell.line:
            raise RoleSensitivityError("one unit is associated with multiple lines")
        if unit in neighbor_units[cell_index]:
            raise RoleSensitivityError("a cell cannot use its own unit as a neighbor")
        if any(value < 0 or value >= len(units) for value in neighbor_units[cell_index]):
            raise RoleSensitivityError("a neighbor-unit index is out of range")

    line_units: dict[str, list[int]] = defaultdict(list)
    for unit, line in unit_line.items():
        line_units[line].append(unit)
    singleton_lines = [line for line, members in line_units.items() if len(members) == 1]
    if singleton_lines != [singleton_line]:
        raise RoleSensitivityError("the expected sole singleton line was not recovered")
    non_singleton = [line for line in line_units if line != singleton_line]
    if len(non_singleton) != design.non_singleton_lines:
        raise RoleSensitivityError("the design requires eight non-singleton lines")
    if any(
        len(line_units[line])
        < design.compromise_development_units_per_line + 1
        for line in non_singleton
    ):
        raise RoleSensitivityError(
            "every non-singleton line needs two compromise-development and one audit unit"
        )
    return units, unit_index, unit_cells, unit_line, line_units


def solve_joint_nested_roles(
    cells: Sequence[SupportCell],
    neighbor_units: Sequence[set[int]],
    *,
    singleton_line: str = SINGLETON_LINE,
    include_compromise: bool,
    time_limit_seconds: float,
    forced_core_audit_units: frozenset[str] = frozenset(),
    design: FixedDesign = FIXED_DESIGN,
) -> dict[str, object]:
    """Solve a joint nested allocation and return aggregate status only."""

    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise RoleSensitivityError("MILP time limit must be finite and positive")
    (
        units,
        unit_index,
        unit_cells,
        unit_line,
        line_units,
    ) = validate_problem(cells, neighbor_units, singleton_line, design)
    unknown_forced = forced_core_audit_units - set(units)
    if unknown_forced:
        raise RoleSensitivityError("a forced core-audit unit is absent")

    unit_count = len(units)
    cell_count = len(cells)
    variables = VariableBook()
    core_development = variables.add("core_development", unit_count)
    core_audit = variables.add("core_audit", unit_count)
    reserve = variables.add("reserve", unit_count)
    compromise_development = None
    compromise_audit = None
    if include_compromise:
        compromise_development = variables.add("compromise_development", unit_count)
        compromise_audit = variables.add("compromise_audit", unit_count)

    membership: dict[str, tuple[list[dict[int, float]], list[dict[int, float]]]] = {}

    def per_unit(block: np.ndarray) -> list[dict[int, float]]:
        return [scalar_expression(int(value)) for value in block]

    core_development_expr = per_unit(core_development)
    core_audit_expr = per_unit(core_audit)
    reserve_expr = per_unit(reserve)
    membership["baseline_D12_A8"] = (
        core_development_expr,
        core_audit_expr,
    )
    membership["estimator_ceiling_D27_A8"] = (
        [
            sum_expressions((core_development_expr[index], reserve_expr[index]))
            for index in range(unit_count)
        ],
        core_audit_expr,
    )
    membership["precision_ceiling_D12_A23"] = (
        core_development_expr,
        [
            sum_expressions((core_audit_expr[index], reserve_expr[index]))
            for index in range(unit_count)
        ],
    )
    if include_compromise:
        assert compromise_development is not None and compromise_audit is not None
        membership["compromise_D16_A19_exact2_per_line"] = (
            per_unit(compromise_development),
            per_unit(compromise_audit),
        )

    support_variables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in membership:
        support_variables[name] = (
            variables.add(f"{name}_supported_development", cell_count),
            variables.add(f"{name}_supported_audit", cell_count),
        )

    constraints = ConstraintRows(variables.size)
    singleton_unit = line_units[singleton_line][0]
    non_singleton_units = [
        unit for unit in range(unit_count) if unit != singleton_unit
    ]

    # D0, A0, and R form one partition of the 35 non-singleton units.
    constraints.add(
        {int(value): 1.0 for value in core_development},
        design.core_development_units,
        design.core_development_units,
    )
    constraints.add(
        {int(value): 1.0 for value in core_audit},
        design.core_audit_units,
        design.core_audit_units,
    )
    constraints.add(
        {int(value): 1.0 for value in reserve},
        design.reserve_units,
        design.reserve_units,
    )
    for unit in non_singleton_units:
        constraints.add(
            {
                int(core_development[unit]): 1.0,
                int(core_audit[unit]): 1.0,
                int(reserve[unit]): 1.0,
            },
            1.0,
            1.0,
        )
    constraints.add(
        {
            int(core_development[singleton_unit]): 1.0,
            int(core_audit[singleton_unit]): 1.0,
            int(reserve[singleton_unit]): 1.0,
        },
        0.0,
        0.0,
    )
    for forced in forced_core_audit_units:
        constraints.add(
            {int(core_audit[unit_index[forced]]): 1.0}, 1.0, 1.0
        )

    # Core line representation propagates to both ceiling designs.
    for line, members in sorted(line_units.items()):
        if line == singleton_line:
            continue
        constraints.add(
            {int(core_development[unit]): 1.0 for unit in members}, lower=1.0
        )
        constraints.add(
            {int(core_audit[unit]): 1.0 for unit in members}, lower=1.0
        )

    if include_compromise:
        assert compromise_development is not None and compromise_audit is not None
        constraints.add(
            {int(value): 1.0 for value in compromise_development},
            design.compromise_development_units,
            design.compromise_development_units,
        )
        constraints.add(
            {int(value): 1.0 for value in compromise_audit},
            design.compromise_audit_units,
            design.compromise_audit_units,
        )
        for unit in non_singleton_units:
            constraints.add(
                {
                    int(compromise_development[unit]): 1.0,
                    int(compromise_audit[unit]): 1.0,
                },
                1.0,
                1.0,
            )
            # D0 is nested in D16; A0 is nested in A19.
            constraints.add(
                {
                    int(core_development[unit]): 1.0,
                    int(compromise_development[unit]): -1.0,
                },
                upper=0.0,
            )
            constraints.add(
                {
                    int(core_audit[unit]): 1.0,
                    int(compromise_audit[unit]): -1.0,
                },
                upper=0.0,
            )
        constraints.add(
            {
                int(compromise_development[singleton_unit]): 1.0,
                int(compromise_audit[singleton_unit]): 1.0,
            },
            0.0,
            0.0,
        )
        for line, members in sorted(line_units.items()):
            if line == singleton_line:
                continue
            constraints.add(
                {int(compromise_development[unit]): 1.0 for unit in members},
                design.compromise_development_units_per_line,
                design.compromise_development_units_per_line,
            )
            constraints.add(
                {int(compromise_audit[unit]): 1.0 for unit in members}, lower=1.0
            )

    # Apply the identical direct-neighbor and retention rules to every design.
    multi_cell_units = [
        unit for unit, indices in unit_cells.items() if len(indices) >= 2
    ]
    for name, (development_expr, audit_expr) in membership.items():
        supported_development, supported_audit = support_variables[name]
        # Exact role sizes make each design independently auditable, even when
        # they are algebraically implied by D0/A0/R nesting.
        expected_counts = {
            "baseline_D12_A8": (
                design.core_development_units,
                design.core_audit_units,
            ),
            "estimator_ceiling_D27_A8": (
                design.estimator_development_units,
                design.estimator_audit_units,
            ),
            "precision_ceiling_D12_A23": (
                design.precision_development_units,
                design.precision_audit_units,
            ),
            "compromise_D16_A19_exact2_per_line": (
                design.compromise_development_units,
                design.compromise_audit_units,
            ),
        }[name]
        constraints.add(
            sum_expressions(development_expr),
            expected_counts[0],
            expected_counts[0],
        )
        constraints.add(
            sum_expressions(audit_expr), expected_counts[1], expected_counts[1]
        )

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
            development_membership = development_expr[owner]
            audit_membership = audit_expr[owner]
            development_owner_constraint = {
                int(supported_development[cell_index]): 1.0
            }
            add_coefficients(
                development_owner_constraint, development_membership, -1.0
            )
            constraints.add(development_owner_constraint, upper=0.0)
            audit_owner_constraint = {int(supported_audit[cell_index]): 1.0}
            add_coefficients(audit_owner_constraint, audit_membership, -1.0)
            constraints.add(audit_owner_constraint, upper=0.0)

            development_neighbors = sum_expressions(
                development_expr[unit] for unit in neighbor_units[cell_index]
            )
            development_neighbor_constraint = {
                int(supported_development[cell_index]): float(
                    design.minimum_distinct_development_neighbors
                )
            }
            add_coefficients(
                development_neighbor_constraint, development_neighbors, -1.0
            )
            constraints.add(development_neighbor_constraint, upper=0.0)
            audit_neighbor_constraint = {
                int(supported_audit[cell_index]): float(
                    design.minimum_distinct_development_neighbors
                )
            }
            add_coefficients(audit_neighbor_constraint, development_neighbors, -1.0)
            constraints.add(audit_neighbor_constraint, upper=0.0)

        for unit, cell_indices in unit_cells.items():
            required = math.ceil(
                design.minimum_per_role_unit_retention * len(cell_indices) - 1e-12
            )
            development_retention = {
                int(supported_development[cell]): 1.0 for cell in cell_indices
            }
            add_coefficients(
                development_retention, development_expr[unit], -float(required)
            )
            constraints.add(development_retention, lower=0.0)
            audit_retention = {
                int(supported_audit[cell]): 1.0 for cell in cell_indices
            }
            add_coefficients(audit_retention, audit_expr[unit], -float(required))
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
                -design.minimum_overall_role_retention * len(cell_indices),
            )
            add_coefficients(
                overall_audit,
                audit_expr[unit],
                -design.minimum_overall_role_retention * len(cell_indices),
            )
        constraints.add(overall_development, lower=0.0)
        constraints.add(overall_audit, lower=0.0)
        constraints.add(
            sum_expressions(development_expr[unit] for unit in multi_cell_units),
            lower=float(design.minimum_multi_cell_development_units),
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
            raise RoleSensitivityError("MILP returned a non-integral feasible point")
        values = rounded.astype(int)
        evaluated = linear_constraint.A @ values
        finite_lower = np.isfinite(linear_constraint.lb)
        finite_upper = np.isfinite(linear_constraint.ub)
        if np.any(
            evaluated[finite_lower] < linear_constraint.lb[finite_lower] - 1e-7
        ) or np.any(
            evaluated[finite_upper] > linear_constraint.ub[finite_upper] + 1e-7
        ):
            raise RoleSensitivityError("MILP witness violates a linear constraint")
        core_development_values = values[core_development]
        core_audit_values = values[core_audit]
        reserve_values = values[reserve]
        if int(core_development_values.sum()) != design.core_development_units:
            raise RoleSensitivityError("witness has the wrong D0 count")
        if int(core_audit_values.sum()) != design.core_audit_units:
            raise RoleSensitivityError("witness has the wrong A0 count")
        if int(reserve_values.sum()) != design.reserve_units:
            raise RoleSensitivityError("witness has the wrong reserve count")
        if np.any(
            core_development_values[non_singleton_units]
            + core_audit_values[non_singleton_units]
            + reserve_values[non_singleton_units]
            != 1
        ):
            raise RoleSensitivityError("witness does not partition non-singleton units")
        if any(
            values[core_audit[unit_index[unit]]] != 1
            for unit in forced_core_audit_units
        ):
            raise RoleSensitivityError("witness violates a forced core-audit role")
        if include_compromise:
            assert (
                compromise_development is not None
                and compromise_audit is not None
            )
            compromise_development_values = values[compromise_development]
            compromise_audit_values = values[compromise_audit]
            if int(compromise_development_values.sum()) != (
                design.compromise_development_units
            ) or int(compromise_audit_values.sum()) != (
                design.compromise_audit_units
            ):
                raise RoleSensitivityError("witness has the wrong compromise counts")
            if np.any(
                core_development_values > compromise_development_values
            ) or np.any(core_audit_values > compromise_audit_values):
                raise RoleSensitivityError("witness violates compromise nesting")
            for line, members in line_units.items():
                if line == singleton_line:
                    continue
                if int(compromise_development_values[members].sum()) != (
                    design.compromise_development_units_per_line
                ):
                    raise RoleSensitivityError(
                        "witness violates exact-two compromise line balance"
                    )
        status = "feasible"
        exists = "yes"
    elif solver.status == 2:
        status = "infeasible"
        exists = "no"
    else:
        status = "indeterminate"
        exists = "unknown"

    # ``solver.x`` is intentionally not retained beyond internal validation.
    result: dict[str, object] = {
        "solver_status_code": int(solver.status),
        "solver_status": status,
        "joint_nested_allocation_exists": exists,
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
    radius_um: float,
    depth_tolerance: float,
) -> list[set[int]]:
    """Construct direct cross-unit neighborhoods from fixed soma geometry."""

    cell_count = len(records)
    if geodesic.shape != (cell_count, cell_count):
        raise RoleSensitivityError("geodesic matrix has the wrong shape")
    if depth_fraction.shape != (cell_count,) or not np.all(
        np.isfinite(depth_fraction)
    ):
        raise RoleSensitivityError("depth vector must be finite and cell-aligned")
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise RoleSensitivityError("tangential radius must be finite and positive")
    if not math.isfinite(depth_tolerance) or depth_tolerance < 0:
        raise RoleSensitivityError("depth tolerance must be finite and nonnegative")
    units = sorted({str(record.brain_id) for record in records})
    unit_index = {unit: index for index, unit in enumerate(units)}
    neighbors = [set() for _ in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left].brain_id == records[right].brain_id:
                continue
            if geodesic[left, right] > radius_um + 1e-9:
                continue
            if abs(depth_fraction[left] - depth_fraction[right]) > (
                depth_tolerance + 1e-12
            ):
                continue
            neighbors[left].add(unit_index[str(records[right].brain_id)])
            neighbors[right].add(unit_index[str(records[left].brain_id)])
    return neighbors


def load_primary_geometry(
    inventory: Path,
) -> tuple[
    list[object], np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]
]:
    """Load the inventory and build both depth conventions exactly once."""

    # The heavy geometry module is deliberately imported only on the real-data
    # path, keeping ``--self-test`` independent of Allen geometry assets and
    # ccf-streamlines.
    tool_dir = Path(__file__).resolve().parent
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))
    import build_whole_cortex_ssptr_geometry_support as geometry  # noqa: PLC0415

    records, inventory_diagnostics = geometry.load_inventory(inventory.resolve())
    primary_input = [
        record
        for record in records
        if record.primary_axon_dendrite_distinguished_eligible
    ]
    if len(primary_input) != PRIMARY_CELLS:
        raise RoleSensitivityError("inventory does not contain 206 primary cells")
    (
        primary_records,
        exact_depth,
        linestring_depth,
        geodesic,
        geometry_diagnostics,
        _graph_diagnostics,
    ) = geometry.build_geometry(primary_input)
    if len(primary_records) != PRIMARY_CELLS:
        raise RoleSensitivityError("primary geometry does not contain 206 cells")
    if len({record.brain_id for record in primary_records}) != TOTAL_UNITS:
        raise RoleSensitivityError("primary geometry does not preserve all 36 units")
    diagnostics = {
        "downloaded_cells": int(inventory_diagnostics["downloaded_cells"]),
        "primary_cells": len(primary_records),
        "geometry_eligible_cells": len(primary_records),
        "geometry_eligible_units": len(
            {record.brain_id for record in primary_records}
        ),
        "geometry_excluded_primary_cells": PRIMARY_CELLS - len(primary_records),
        "exact_vs_linestring_depth_absolute_difference_median": float(
            geometry_diagnostics["depth_projection_absolute_difference_median"]
        ),
        "exact_vs_linestring_depth_absolute_difference_max": float(
            geometry_diagnostics["depth_projection_absolute_difference_max"]
        ),
    }
    return (
        primary_records,
        exact_depth,
        linestring_depth,
        geodesic,
        diagnostics,
    )


def ensure_aggregate_row(row: Mapping[str, object]) -> None:
    """Reject structured values or fields that could encode a role witness."""

    forbidden_fragments = ("assignment", "witness", "selected_id", "role_ids")
    for key, value in row.items():
        if key != "role_assignment_emitted" and any(
            fragment in key.lower() for fragment in forbidden_fragments
        ):
            raise RoleSensitivityError(f"forbidden output field: {key}")
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            raise RoleSensitivityError(f"non-aggregate output value in {key}")


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    diagnostics: Mapping[str, object],
    output_csv: Path,
    output_md: Path,
) -> None:
    if not rows:
        raise RoleSensitivityError("no sensitivity rows were produced")
    for row in rows:
        ensure_aggregate_row(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# EP11 SSp-tr nested role-design geometry sensitivity",
        "",
        "This is an outcome-blind feasibility diagnostic on the 206 primary",
        "axon/dendrite-distinguished reconstructions. It uses only soma geometry,",
        "brain/sample-unit identity, and strain/Cre line. No regional projection",
        "outcome, role identity, or MILP witness is emitted.",
        "",
        "The common core is D0=12 and A0=8. The remaining 15 non-singleton",
        "units form R. The joint core system requires the baseline D12/A8, the",
        "estimator ceiling (D0+R)/A0 = D27/A8, and the precision ceiling",
        "D0/(A0+R) = D12/A23 to pass simultaneously. The preferred compromise",
        "adds D16/A19 with D0 contained in D16, A0 contained in A19, and exactly",
        "two D16 units in each of the eight non-singleton lines. The singleton",
        "Tle4 unit is unassigned throughout.",
        "",
        "Every design separately enforces direct support from two development",
        "units, at least 0.90 retention in every assigned role unit, at least",
        "0.95 retention in each role overall, at least eight multi-cell",
        "development units, and non-singleton-line representation in both roles.",
        "The two depth conventions are solved separately. Passing both is not",
        "yet proof that one common hidden allocation satisfies both conventions;",
        "that stronger robustness check would require one cross-convention MILP.",
        "",
        "## Aggregate geometry",
        "",
        f"- Primary cells / brain-sample units: {diagnostics['primary_cells']} / "
        f"{diagnostics['geometry_eligible_units']}.",
        f"- Geometry-excluded primary cells: "
        f"{diagnostics['geometry_excluded_primary_cells']}.",
        "- Exact-versus-LineString3D normalized-depth absolute difference",
        f"  median / maximum: "
        f"{diagnostics['exact_vs_linestring_depth_absolute_difference_median']:.4f} / "
        f"{diagnostics['exact_vs_linestring_depth_absolute_difference_max']:.4f}.",
        "",
        "## Aggregate feasibility",
        "",
        "| Depth projection | Joint system | Cells with >=2 candidate neighbor units | Binary variables | Constraints | Solver status | Feasible |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['depth_projection_convention']} | "
            f"{row['joint_system']} | "
            f"{row['cells_with_2plus_candidate_neighbor_units']} | "
            f"{row['binary_variables']} | "
            f"{row['linear_constraints']} | "
            f"{row['solver_status']} | "
            f"{row['joint_nested_allocation_exists']} |"
        )
    lines.extend(
        (
            "",
            "Feasibility establishes only that a nested allocation exists under",
            "the fixed outcome-blind support rule. It does not choose or lock role",
            "identities, establish animal independence, authorize projection-outcome",
            "access, or support either groups or a continuum.",
        )
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, object]:
    FIXED_DESIGN.validate_fixed_contract()
    (
        records,
        exact_depth,
        linestring_depth,
        geodesic,
        diagnostics,
    ) = load_primary_geometry(args.inventory)
    support_cells = [
        SupportCell(
            cell_id=str(record.cell_id),
            unit_id=str(record.brain_id),
            line=str(record.line),
        )
        for record in records
    ]
    rows: list[dict[str, object]] = []
    conventions = (
        ("exact_polyline_closest_point_primary", exact_depth),
        ("ccf_streamlines_LineString3D_project_sensitivity", linestring_depth),
    )
    systems = (
        ("core_plus_estimator_and_precision", False),
        ("core_plus_estimator_precision_and_exact2_compromise", True),
    )
    for convention, depth in conventions:
        neighbors = build_neighbor_units(
            records,
            geodesic,
            depth,
            TANGENTIAL_RADIUS_UM,
            DEPTH_TOLERANCE,
        )
        candidate_supported = sum(
            len(values) >= FIXED_DESIGN.minimum_distinct_development_neighbors
            for values in neighbors
        )
        for system, include_compromise in systems:
            solver = solve_joint_nested_roles(
                support_cells,
                neighbors,
                include_compromise=include_compromise,
                time_limit_seconds=args.milp_time_limit_seconds,
            )
            row = {
                "source_region": SOURCE_REGION,
                "cohort": "Gao_whole_cortex_projectome",
                "analysis_track": "primary_206_axon_dendrite_distinguished",
                "depth_projection_convention": convention,
                "tangential_radius_um": TANGENTIAL_RADIUS_UM,
                "normalized_depth_tolerance": f"{DEPTH_TOLERANCE:.2f}",
                "depth_convention_role_scope": (
                    "separate_existence_not_one_common_cross_convention_allocation"
                ),
                "geometry_eligible_cells": len(records),
                "brain_sample_units": len({record.brain_id for record in records}),
                "non_singleton_lines": FIXED_DESIGN.non_singleton_lines,
                "singleton_units_forced_unassigned": (
                    FIXED_DESIGN.singleton_units_unassigned
                ),
                "cells_with_2plus_candidate_neighbor_units": candidate_supported,
                "joint_system": system,
                "core_development_units": FIXED_DESIGN.core_development_units,
                "core_audit_units": FIXED_DESIGN.core_audit_units,
                "reserve_units": FIXED_DESIGN.reserve_units,
                "estimator_development_units": (
                    FIXED_DESIGN.estimator_development_units
                ),
                "estimator_audit_units": FIXED_DESIGN.estimator_audit_units,
                "precision_development_units": (
                    FIXED_DESIGN.precision_development_units
                ),
                "precision_audit_units": FIXED_DESIGN.precision_audit_units,
                "compromise_included": "yes" if include_compromise else "no",
                "compromise_development_units": (
                    FIXED_DESIGN.compromise_development_units
                    if include_compromise
                    else 0
                ),
                "compromise_audit_units": (
                    FIXED_DESIGN.compromise_audit_units
                    if include_compromise
                    else 0
                ),
                "compromise_development_units_per_non_singleton_line": (
                    FIXED_DESIGN.compromise_development_units_per_line
                    if include_compromise
                    else 0
                ),
                "minimum_distinct_development_neighbor_units": (
                    FIXED_DESIGN.minimum_distinct_development_neighbors
                ),
                "minimum_per_role_unit_retention": (
                    f"{FIXED_DESIGN.minimum_per_role_unit_retention:.2f}"
                ),
                "minimum_overall_role_retention": (
                    f"{FIXED_DESIGN.minimum_overall_role_retention:.2f}"
                ),
                "minimum_multi_cell_development_units": (
                    FIXED_DESIGN.minimum_multi_cell_development_units
                ),
                **solver,
            }
            ensure_aggregate_row(row)
            rows.append(row)

    write_outputs(rows, diagnostics, args.output_csv, args.output_md)
    return {
        "status": "complete",
        "rows": len(rows),
        "primary_cells": len(records),
        "brain_sample_units": len({record.brain_id for record in records}),
        "indeterminate_rows": sum(
            row["joint_nested_allocation_exists"] == "unknown" for row in rows
        ),
        "role_assignment_emitted": False,
        "output_csv": str(args.output_csv.resolve()),
        "output_md": str(args.output_md.resolve()),
    }


def make_complete_fixture(
    *, single_cell_unit: str | None = None
) -> tuple[list[SupportCell], list[set[int]], str]:
    """Create an asset-free 36-unit complete-neighbor fixture."""

    line_sizes = (9, 5, 4, 4, 4, 3, 3, 3)
    cells: list[SupportCell] = []
    unit_ids: list[str] = []
    for line_index, size in enumerate(line_sizes):
        for within_line in range(size):
            unit = f"unit_{line_index:02d}_{within_line:02d}"
            unit_ids.append(unit)
            cell_count = 1 if unit == single_cell_unit else 2
            for cell_index in range(cell_count):
                cells.append(
                    SupportCell(
                        cell_id=f"cell_{unit}_{cell_index}",
                        unit_id=unit,
                        line=f"line_{line_index:02d}",
                    )
                )
    singleton_line = "singleton_line"
    singleton_unit = "unit_singleton"
    unit_ids.append(singleton_unit)
    for cell_index in range(2):
        cells.append(
            SupportCell(
                cell_id=f"cell_{singleton_unit}_{cell_index}",
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
    return cells, neighbors, singleton_line


def run_self_test() -> dict[str, object]:
    """Exercise the MILP without reading inventory or Allen assets."""

    # A complete cross-unit neighborhood must support the full nested system,
    # including the exact-two-per-line compromise.
    cells, neighbors, singleton_line = make_complete_fixture()
    feasible = solve_joint_nested_roles(
        cells,
        neighbors,
        singleton_line=singleton_line,
        include_compromise=True,
        time_limit_seconds=30.0,
    )
    if feasible["joint_nested_allocation_exists"] != "yes":
        raise RoleSensitivityError("complete-neighbor fixture was not feasible")

    # The requested A8 core is a fixed design requirement, not a tunable count.
    a7_rejected = False
    try:
        replace(FIXED_DESIGN, core_audit_units=7).validate_fixed_contract()
    except RoleSensitivityError:
        a7_rejected = True
    if not a7_rejected:
        raise RoleSensitivityError("an A7 core was not rejected")

    # A forced one-cell audit unit with no development neighbors cannot reach
    # either the per-unit or the direct-two-development-unit support rule.
    sparse_unit = "unit_00_00"
    sparse_cells, sparse_neighbors, sparse_singleton = make_complete_fixture(
        single_cell_unit=sparse_unit
    )
    for index, cell in enumerate(sparse_cells):
        if cell.unit_id == sparse_unit:
            sparse_neighbors[index] = set()
    sparse = solve_joint_nested_roles(
        sparse_cells,
        sparse_neighbors,
        singleton_line=sparse_singleton,
        include_compromise=False,
        time_limit_seconds=30.0,
        forced_core_audit_units=frozenset({sparse_unit}),
    )
    if sparse["joint_nested_allocation_exists"] != "no":
        raise RoleSensitivityError("forced sparse assigned unit was not infeasible")

    # Public solver objects and written rows are scalars only; no IDs or role
    # vectors survive internal witness validation.
    validate_aggregate_solver_result(feasible)
    validate_aggregate_solver_result(sparse)
    with tempfile.TemporaryDirectory() as temporary:
        output_csv = Path(temporary) / "aggregate.csv"
        output_md = Path(temporary) / "aggregate.md"
        synthetic_row = {
            "depth_projection_convention": "synthetic_complete",
            "joint_system": "self_test",
            "cells_with_2plus_candidate_neighbor_units": len(cells),
            **feasible,
        }
        synthetic_diagnostics = {
            "primary_cells": len(cells),
            "geometry_eligible_units": TOTAL_UNITS,
            "geometry_excluded_primary_cells": 0,
            "exact_vs_linestring_depth_absolute_difference_median": 0.0,
            "exact_vs_linestring_depth_absolute_difference_max": 0.0,
        }
        write_outputs(
            [synthetic_row], synthetic_diagnostics, output_csv, output_md
        )
        csv_text = output_csv.read_text(encoding="utf-8")
        md_text = output_md.read_text(encoding="utf-8")
        if any(unit in csv_text or unit in md_text for unit in {cell.unit_id for cell in cells}):
            raise RoleSensitivityError("a self-test output leaked role-unit identity")

    return {
        "status": "passed",
        "complete_neighbor_joint_system": "feasible",
        "audit_7_contract": "rejected",
        "forced_sparse_small_assigned_unit": "infeasible",
        "role_assignment_emitted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--milp-time-limit-seconds", type=float, default=900.0)
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
