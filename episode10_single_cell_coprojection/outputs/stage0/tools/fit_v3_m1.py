#!/usr/bin/env python3
"""Target-free gates and exact set-softmax engine for EP10 v3 M1.

The probe modes in this file can address only the frozen target-blind model
interface and atlas-geometry products.  They have no target-root argument and
cannot open observed target calls, SWCs, final data, or unassigned data.

M1 is normalized by exhaustive enumeration of every fixed-K candidate set.
Candidate sufficient statistics are accumulated in fixed batches as 21
dimensions (15 target contrasts plus six geometry features); a candidate by
261 matrix is never constructed.  The exact 261-dimensional Hessian is
expanded from those streamed moments and the cell's 17-dimensional design
row.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import resource
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from v3_m1_triangular_solve import solve_cholesky as _cholesky_factored_solve


SCRIPT_PATH = Path(os.path.abspath(__file__))
STAGE0_DIR = SCRIPT_PATH.parent.parent
CONTRACT_PATH = STAGE0_DIR / "v3_m1_model_contract.yaml"
SYNTHETIC_DESIGN_ADDENDUM_PATH = (
    STAGE0_DIR / "v3_m1_synthetic_design_addendum_v1.yaml"
)
SYNTHETIC_RECOVERY_ADDENDUM_PATH = (
    STAGE0_DIR / "v3_m1_synthetic_recovery_addendum_v2.yaml"
)
SOLVER_ADDENDUM_V1_PATH = STAGE0_DIR / "v3_m1_solver_addendum_v1.yaml"
SOLVER_ADDENDUM_PATH = STAGE0_DIR / "v3_m1_solver_addendum_v2.yaml"
M0_V7_ENGINE_PATH = STAGE0_DIR / "tools" / "fit_v3_m0_v7.py"
WARM_ZERO_V1_OUTPUT_PREFIX = "v3_m1_synthetic_warm_zero_diagnostic_v1"
WARM_ZERO_OUTPUT_PREFIX = "v3_m1_synthetic_warm_zero_diagnostic_v2"

PANEL = (
    "allen_993_MOs__ipsilateral",
    "allen_993_MOs__contralateral",
    "allen_453_SS__ipsilateral",
    "allen_453_SS__contralateral",
    "allen_31_ACA__ipsilateral",
    "allen_31_ACA__contralateral",
    "allen_477_STR__ipsilateral",
    "allen_477_STR__contralateral",
    "allen_549_TH__ipsilateral",
    "allen_549_TH__contralateral",
    "allen_313_MB__ipsilateral",
    "allen_313_MB__contralateral",
    "allen_771_P__ipsilateral",
    "allen_771_P__contralateral",
    "allen_354_MY__ipsilateral",
    "allen_354_MY__contralateral",
)
LAYERS = ("MOp1", "MOp2/3", "MOp5", "MOp6a")
GROUP_ORDER = (
    "212074",
    "220187",
    "220188",
    "220190",
    "221091",
    "221225",
    "221418",
    "221514",
    "221539",
    "221588",
    "221751",
    "233281",
)
GROUP_SIZES = {
    "212074": 6,
    "220187": 35,
    "220188": 25,
    "220190": 16,
    "221091": 33,
    "221225": 39,
    "221418": 12,
    "221514": 5,
    "221539": 1,
    "221588": 7,
    "221751": 13,
    "233281": 2,
}
FEATURE_NAMES = (
    "mean_soma_target_distance_mm",
    "sd_soma_target_distance_mm",
    "mean_target_pair_distance_mm",
    "cross_side_pair_fraction",
    "homotopic_pair_fraction",
    "mst_mean_edge_length_mm",
)
COVARIATE_NAMES = (
    "intercept",
    "K_legendre_P1",
    "K_legendre_P2",
    "soma_x",
    "soma_y",
    "soma_z",
    "soma_x2",
    "soma_y2",
    "soma_z2",
    "soma_xy",
    "soma_xz",
    "soma_yz",
    "layer_H1",
    "layer_H2",
    "layer_H3",
    "hemisphere_right",
    "segment_unmapped",
)
INTERFACE_FIELDS = (
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "provider_hemisphere",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
    "segment_unmapped",
    "terminal_unmapped_report_only",
    "mapping_overlap_class_report_only",
)

PANEL_SIZE = 16
TARGET_CONTRAST_DIMENSION = 15
DESIGN_DIMENSION = 17
GEOMETRY_DIMENSION = 6
PARAMETER_DIMENSION = 261
JOINT_CELL_DIMENSION = 21
B_PARAMETER_DIMENSION = 255
BATCH_MAXIMUM = 4096
LAMBDAS_DESCENDING = tuple(10.0**power for power in range(3, -9, -1))
GRADIENT_TOLERANCE = 1.0e-7
RELATIVE_TOLERANCE = 1.0e-10
MAXIMUM_ITERATIONS = 2000
ARMIJO_C1 = 1.0e-4
CURVATURE_C2 = 0.9
INITIAL_STEP = 1.0
EXPANSION_FACTOR = 2.0
MAXIMUM_STEP = 1048576.0
LINE_SEARCH_EVALUATIONS = 80
ZOOM_ITERATIONS = 60
POST_WITNESS_CONFIRMATION_ATTEMPTS = 1
LN2 = math.log(2.0)
CPU_CORE_HOUR_CEILING = 1800.0
WALL_CLOCK_HOUR_CEILING = 144.0
PARALLEL_CORE_CEILING = 64
MEMORY_GB_CEILING = 512.0
SCRATCH_GB_CEILING = 1536.0
RECOVERY_LAMBDA = 1.0e-6
POPULATION_KL_RAW_BOUND_NATS = 2.76875e-6
POPULATION_KL_WITNESS_CEILING_NATS = 3.045625e-6
POPULATION_KL_WITNESS_CEILING_BITS = 4.3939080839074494e-6
POPULATION_PINSKER_MARGINAL_BOUND = 0.0012340228928184437

_FROZEN_M0_V7_MODULE: Any | None = None


class M1Error(RuntimeError):
    """A frozen M1 mathematical, input, or sealed-scope rule failed."""


class LineSearchFailure(M1Error):
    def __init__(
        self,
        message: str,
        *,
        evaluations: int,
        expansions: int,
        zoom_iterations: int,
        all_values_finite: bool,
        scalar_context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.evaluations = evaluations
        self.expansions = expansions
        self.zoom_iterations = zoom_iterations
        self.all_values_finite = all_values_finite
        self.scalar_context = dict(scalar_context or {})


class UncertifiedLineSearchFailure(M1Error):
    """A line-search stall whose frozen convergence certificate did not pass."""

    def __init__(self, message: str, scalar_context: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.scalar_context = dict(scalar_context)


@dataclass(frozen=True)
class Cell:
    identifier: str
    group: str
    layer: str
    hemisphere: str
    soma_um: tuple[float, float, float]
    soma_distance_mm: np.ndarray
    k: int
    observed_index: int
    segment_unmapped: int
    sampling_uniform: float = 0.5


@dataclass(frozen=True)
class DesignTransform:
    soma_mean: np.ndarray
    soma_scale: np.ndarray


@dataclass(frozen=True)
class CandidateBasis:
    k: int
    membership: np.ndarray
    contrasts: np.ndarray

    @property
    def count(self) -> int:
        return int(self.membership.shape[0])


@dataclass(frozen=True)
class PairGeometry:
    distance: np.ndarray
    cross_side: np.ndarray
    homotopic: np.ndarray
    ordered_edges: tuple[tuple[float, int, int], ...]


@dataclass(frozen=True)
class UniformCellMoments:
    geometry_mean: np.ndarray
    raw_joint_covariance: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    cells: tuple[Cell, ...]
    x: np.ndarray
    geometry_centers: np.ndarray
    geometry_scales: np.ndarray
    weights: np.ndarray
    design_transform: DesignTransform


@dataclass(frozen=True)
class Evaluation:
    objective: float
    gradient: np.ndarray
    hessian: np.ndarray | None
    log_probability_nats: np.ndarray | None
    marginals: np.ndarray | None
    candidate_visits: int
    raw_hessian_symmetry_error: float | None
    elapsed_seconds: float
    joint_residuals: np.ndarray | None
    line_objective_difference: float | None
    direct_directional_derivative: float | None
    line_small_increment_cells: int
    line_general_cells: int


@dataclass(frozen=True)
class LineSearchResult:
    vector: np.ndarray
    value: float
    gradient: np.ndarray
    objective_difference: float
    directional_derivative: float
    step: float
    evaluations: int
    expansions: int
    zoom_iterations: int
    armijo_slack: float
    curvature_ratio: float
    small_increment_cells: int
    general_cells: int


@dataclass(frozen=True)
class OptimizerResult:
    theta: np.ndarray
    objective: float
    gradient_infinity_norm: float
    iterations: int
    accepted_steps: int
    relative_objective_change: float
    telemetry: Mapping[str, Any]


@dataclass(frozen=True)
class PopulationExpectation:
    joint_mean: np.ndarray
    marginals: np.ndarray
    probability_sum_maximum_absolute_error: float
    candidate_visits: int
    elapsed_seconds: float


@dataclass(frozen=True)
class PopulationLinearCorrection:
    vector: np.ndarray
    observed_aggregate: np.ndarray
    expected_aggregate: np.ndarray


def helmert(rows: int) -> np.ndarray:
    if rows < 2:
        raise ValueError("Helmert coding requires at least two rows")
    result = np.zeros((rows, rows - 1), dtype=float)
    for column in range(rows - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        result[: column + 1, column] = 1.0 / denominator
        result[column + 1, column] = -(column + 1) / denominator
    return result


TARGET_CONTRASTS = helmert(PANEL_SIZE)
LAYER_CONTRASTS = helmert(len(LAYERS))


def _finite(array: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(value)):
        raise M1Error(f"nonfinite {label}")
    return value


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise M1Error(f"YAML is not a mapping: {path}")
    return payload


def validate_contract(
    path: Path = CONTRACT_PATH,
    synthetic_design_addendum_path: Path = SYNTHETIC_DESIGN_ADDENDUM_PATH,
    synthetic_recovery_addendum_path: Path = SYNTHETIC_RECOVERY_ADDENDUM_PATH,
    solver_addendum_path: Path = SOLVER_ADDENDUM_PATH,
) -> Mapping[str, Any]:
    absolute = Path(os.path.abspath(path))
    if (
        absolute != CONTRACT_PATH
        or not absolute.is_file()
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != CONTRACT_PATH.resolve(strict=True)
    ):
        raise M1Error("M1 contract must be the exact unsymlinked stage0 contract")
    contract = _read_yaml(absolute)
    if (
        contract.get("schema_version") != "episode10_v3_m1_model_v1"
        or contract.get("status") != "frozen_before_implementation_and_real_M1_execution"
        or contract.get("sealed_scope", {}).get("role") != "development_only"
        or contract.get("sealed_scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise M1Error("M1 contract identity or sealed scope changed")
    representation = contract.get("model", {}).get("bounded_exact_representation", {})
    optimizer = contract.get("estimation", {}).get("optimizer", {})
    probe = contract.get("pre_real_M1_gates", {}).get(
        "target_free_real_shape_benchmark", {}
    )
    request = probe.get("initial_Slurm_probe_request", {})
    if (
        int(contract.get("model", {}).get("parameterization", {}).get("total_dimension", -1))
        != PARAMETER_DIMENSION
        or int(representation.get("fixed_candidate_batch_maximum", -1))
        != BATCH_MAXIMUM
        or representation.get("candidate_by_261_sufficient_statistic_materialization")
        != "forbidden"
        or optimizer.get("algorithm")
        != "deterministic_exact_Hessian_line_search_Newton"
        or tuple(probe.get("frozen_group_order", ())) != GROUP_ORDER
        or str(request.get("wall_time")) != "08:00:00"
        or int(request.get("cpus_per_task", -1)) != 4
        or str(request.get("memory")) != "16G"
        or int(request.get("BLAS_threads", -1)) != 4
    ):
        raise M1Error("M1 representation, optimizer, or target-free probe drifted")
    addendum_absolute = Path(os.path.abspath(synthetic_design_addendum_path))
    if (
        addendum_absolute != SYNTHETIC_DESIGN_ADDENDUM_PATH
        or not addendum_absolute.is_file()
        or addendum_absolute.is_symlink()
        or addendum_absolute.resolve(strict=True)
        != SYNTHETIC_DESIGN_ADDENDUM_PATH.resolve(strict=True)
    ):
        raise M1Error("M1 synthetic-design addendum must be the exact stage0 file")
    addendum = _read_yaml(addendum_absolute)
    base = addendum.get("base_contract", {})
    sole_change = addendum.get("sole_change", {})
    if (
        addendum.get("schema_version")
        != "episode10_v3_m1_synthetic_design_addendum_v1"
        or addendum.get("status")
        != "frozen_before_v2_target_free_rank_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or base.get("required_schema_version")
        != "episode10_v3_m1_model_v1"
        or sole_change.get("applies_only_to")
        != "deterministic_target_free_synthetic_rows"
        or sole_change.get("schedule_formula")
        != "floor_integer_divide_row_index_by_4_then_modulo_2"
        or sole_change.get("index_0_hemisphere") != "left"
        or sole_change.get("index_1_hemisphere") != "right"
        or sole_change.get("consumes_random_variates") is not False
    ):
        raise M1Error("M1 synthetic-design addendum identity or scope changed")
    recovery_absolute = Path(os.path.abspath(synthetic_recovery_addendum_path))
    if (
        recovery_absolute != SYNTHETIC_RECOVERY_ADDENDUM_PATH
        or not recovery_absolute.is_file()
        or recovery_absolute.is_symlink()
        or recovery_absolute.resolve(strict=True)
        != SYNTHETIC_RECOVERY_ADDENDUM_PATH.resolve(strict=True)
    ):
        raise M1Error("M1 synthetic-recovery addendum must be the exact stage0 file")
    recovery = _read_yaml(recovery_absolute)
    authority = recovery.get("authority", {})
    recovery_base = authority.get("base_contract", {})
    recovery_design = authority.get("synthetic_design_addendum", {})
    recovery_change = recovery.get("sole_change", {})
    population = recovery.get("exact_population_definition", {})
    computation = population.get("computation", {})
    witness = recovery.get("prospective_implementation_consistency_witness", {})
    acceptance = recovery.get("unchanged_population_recovery_acceptance", {})
    execution = recovery.get("execution_order", {})
    implementation = recovery.get("implementation_boundary", {})
    if (
        recovery.get("schema_version")
        != "episode10_v3_m1_synthetic_recovery_addendum_v2"
        or recovery.get("status") != "frozen_before_any_population_recovery_probe"
        or recovery.get("scientific_authority") is not False
        or recovery_base.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m1_model_contract.yaml"
        or recovery_base.get("required_schema_version")
        != "episode10_v3_m1_model_v1"
        or recovery_design.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m1_synthetic_design_addendum_v1.yaml"
        or recovery_design.get("required_schema_version")
        != "episode10_v3_m1_synthetic_design_addendum_v1"
        or recovery_change.get("applies_only_to")
        != "full_544_row_target_free_planted_recovery_hard_acceptance"
        or recovery_change.get("numeric_recovery_thresholds_changed") is not False
        or recovery_change.get("sampled_categorical_generation_changed") is not False
        or recovery_change.get("sampled_fit_retained") is not True
        or recovery_change.get("sampled_fit_role")
        != "descriptive_finite_sample_diagnostic_only"
        or computation.get("candidate_enumeration")
        != "exhaustive_in_frozen_lexicographic_order"
        or int(computation.get("candidate_batch_maximum", -1)) != BATCH_MAXIMUM
        or computation.get("stochastic_or_sampled_expectation") != "forbidden"
        or int(witness.get("frozen_parameter_dimension", -1))
        != PARAMETER_DIMENSION
        or float(witness.get("frozen_lambda", math.nan)) != RECOVERY_LAMBDA
        or float(witness.get("raw_population_KL_upper_bound_nats", math.nan))
        != POPULATION_KL_RAW_BOUND_NATS
        or float(witness.get("implementation_witness_ceiling_nats", math.nan))
        != POPULATION_KL_WITNESS_CEILING_NATS
        or float(witness.get("implementation_witness_ceiling_bits", math.nan))
        != POPULATION_KL_WITNESS_CEILING_BITS
        or float(
            witness.get("Pinsker_report_only_implication", {}).get(
                "with_headroom_mean_inclusion_marginal_MAE_upper_bound", math.nan
            )
        )
        != POPULATION_PINSKER_MARGINAL_BOUND
        or acceptance.get("initialization") != "all_261_parameters_zero"
        or float(acceptance.get("recovery_fit_lambda", math.nan))
        != RECOVERY_LAMBDA
        or float(acceptance.get("mean_conditional_KL_maximum_bits", math.nan))
        != 0.01
        or float(acceptance.get("exact_inclusion_marginal_MAE_maximum", math.nan))
        != 0.02
        or float(
            acceptance.get(
                "planted_vs_recovered_gamma_cosine_similarity_minimum", math.nan
            )
        )
        != 0.90
        or execution.get("narrow_pass_authorizes_only")
        != "complete_target_free_v3_probe"
        or implementation.get("next_probe_output_basename")
        != "v3_m1_synthetic_population_recovery_v2_job_JOBID"
    ):
        raise M1Error("M1 synthetic-recovery addendum identity or scope changed")
    solver_v1_absolute = Path(os.path.abspath(SOLVER_ADDENDUM_V1_PATH))
    if (
        not solver_v1_absolute.is_file()
        or solver_v1_absolute.is_symlink()
        or solver_v1_absolute.resolve(strict=True)
        != SOLVER_ADDENDUM_V1_PATH.resolve(strict=True)
    ):
        raise M1Error("M1 solver addendum v1 must be the exact stage0 file")
    solver_v1 = _read_yaml(solver_v1_absolute)
    solver_v1_authority = solver_v1.get("authority", {})
    solver_base = solver_v1_authority.get("base_contract", {})
    solver_design = solver_v1_authority.get("synthetic_design_addendum", {})
    solver_recovery = solver_v1_authority.get("synthetic_recovery_addendum", {})
    solver_change = solver_v1.get("sole_numerical_change", {})
    stable_identity = solver_v1.get("stable_line_identity", {})
    branch = stable_identity.get("branch_boundary", {})
    solver_v1_boundary = solver_v1.get("implementation_boundary", {})
    if (
        solver_v1.get("schema_version") != "episode10_v3_m1_solver_addendum_v1"
        or solver_v1.get("status")
        != "frozen_before_stable_line_difference_implementation_and_execution"
        or solver_v1.get("scientific_authority") is not False
        or solver_base.get("required_schema_version")
        != "episode10_v3_m1_model_v1"
        or solver_design.get("required_schema_version")
        != "episode10_v3_m1_synthetic_design_addendum_v1"
        or solver_recovery.get("required_schema_version")
        != "episode10_v3_m1_synthetic_recovery_addendum_v2"
        or solver_change.get("changes_model_objective_or_gradient") is not False
        or solver_change.get("changes_Wolfe_constants_or_acceptance_inequalities")
        is not False
        or solver_change.get("changes_convergence_or_certificate_thresholds")
        is not False
        or solver_change.get("adds_acceptance_slack_tie_rule_or_fallback")
        is not False
        or stable_identity.get("arithmetic") != "binary64"
        or branch.get("small_form_includes")
        != "max_A_absolute_r_i_A_less_than_or_equal_to_0.5"
        or branch.get("general_form_begins")
        != "max_A_absolute_r_i_A_greater_than_0.5"
        or solver_v1_boundary.get("next_probe_output_basename")
        != f"{WARM_ZERO_V1_OUTPUT_PREFIX}_job_JOBID"
    ):
        raise M1Error("M1 solver addendum v1 identity or scope changed")

    solver_absolute = Path(os.path.abspath(solver_addendum_path))
    if (
        solver_absolute != SOLVER_ADDENDUM_PATH
        or not solver_absolute.is_file()
        or solver_absolute.is_symlink()
        or solver_absolute.resolve(strict=True)
        != SOLVER_ADDENDUM_PATH.resolve(strict=True)
    ):
        raise M1Error("M1 solver addendum v2 must be the exact stage0 file")
    solver = _read_yaml(solver_absolute)
    solver_authority = solver.get("authority", {})
    solver_base = solver_authority.get("base_contract", {})
    solver_prior = solver_authority.get("prior_solver_addendum", {})
    solver_change = solver.get("sole_numerical_change", {})
    solver_boundary = solver.get("implementation_boundary", {})
    solver_execution = solver.get("execution_order", {})
    if (
        solver.get("schema_version") != "episode10_v3_m1_solver_addendum_v2"
        or solver.get("status")
        != "frozen_before_one_shot_convergence_confirmation_implementation_and_execution"
        or solver.get("scientific_authority") is not False
        or solver_base.get("required_schema_version")
        != "episode10_v3_m1_model_v1"
        or solver_prior.get("required_schema_version")
        != "episode10_v3_m1_solver_addendum_v1"
        or solver_change.get("scope")
        != "every_M1_fit_independent_of_lambda_initialization_or_data_role"
        or int(solver_change.get("fixed_confirmation_attempt_count", -1))
        != POST_WITNESS_CONFIRMATION_ATTEMPTS
        or solver_change.get(
            "accepted_step_exact_binary64_zero_gradient_may_return_without_confirmation"
        )
        is not True
        or solver_change.get("initial_state_repeated_unchanged_witness")
        != "unchanged"
        or solver_change.get("warm_zero_or_other_validation_metric_enters_optimizer_stopping")
        is not False
        or solver_change.get("changes_model_objective_gradient_or_Hessian")
        is not False
        or solver_change.get("changes_Newton_direction_or_factorization")
        is not False
        or solver_change.get("changes_Wolfe_constants_steps_caps_or_inequalities")
        is not False
        or solver_change.get("changes_convergence_or_certificate_thresholds")
        is not False
        or solver_change.get("adds_acceptance_slack_tie_rule_or_fallback")
        is not False
        or solver_execution.get("narrow_pass_authorizes_only")
        != "complete_target_free_v4_probe"
        or solver_boundary.get("next_probe_output_basename")
        != f"{WARM_ZERO_OUTPUT_PREFIX}_job_JOBID"
    ):
        raise M1Error("M1 solver addendum v2 identity or scope changed")
    return contract


class CandidateLibrary:
    """The only panel-wide reusable candidate cache: membership and C sums."""

    def __init__(self) -> None:
        self._bases: dict[int, CandidateBasis] = {}

    def basis(self, k: int) -> CandidateBasis:
        if not 0 <= int(k) <= PANEL_SIZE:
            raise M1Error(f"K={k} is outside 0..16")
        key = int(k)
        if key not in self._bases:
            combinations = tuple(itertools.combinations(range(PANEL_SIZE), key))
            membership = np.zeros((len(combinations), PANEL_SIZE), dtype=np.uint8)
            if key:
                indexes = np.asarray(combinations, dtype=int)
                membership[np.arange(len(combinations))[:, None], indexes] = 1
            contrasts = membership.astype(float) @ TARGET_CONTRASTS
            if key in (0, PANEL_SIZE):
                contrasts[...] = 0.0
            expected = math.comb(PANEL_SIZE, key)
            if membership.shape != (expected, PANEL_SIZE):
                raise M1Error("candidate enumeration count changed")
            self._bases[key] = CandidateBasis(key, membership, contrasts)
        return self._bases[key]

    def all_counts(self) -> dict[int, int]:
        return {k: self.basis(k).count for k in range(PANEL_SIZE + 1)}


class GeometryEngine:
    """Generate exact raw set features; no cell-by-candidate cache is retained."""

    def __init__(self, library: CandidateLibrary, pairs: PairGeometry) -> None:
        self.library = library
        self.pairs = pairs
        # These four columns are cell independent.  They are bounded by the
        # 65,536 panel subsets, never replicated across cells, and are sliced
        # into <=4096 rows whenever used by the likelihood.
        self._shared_features: dict[int, np.ndarray] = {}

    def shared_features(self, k: int) -> np.ndarray:
        key = int(k)
        if key not in self._shared_features:
            basis = self.library.basis(key)
            count = basis.count
            result = np.zeros((count, 4), dtype=float)
            if key >= 2:
                denominator = float(math.comb(key, 2))
                masks = basis.membership
                for row_index, mask in enumerate(masks):
                    chosen = np.flatnonzero(mask)
                    pair_distance = 0.0
                    cross = 0.0
                    homotopic = 0.0
                    for left_offset, left in enumerate(chosen[:-1]):
                        right = chosen[left_offset + 1 :]
                        pair_distance += float(np.sum(self.pairs.distance[left, right]))
                        cross += float(np.sum(self.pairs.cross_side[left, right]))
                        homotopic += float(np.sum(self.pairs.homotopic[left, right]))

                    parent = list(range(PANEL_SIZE))

                    def find(value: int) -> int:
                        while parent[value] != value:
                            parent[value] = parent[parent[value]]
                            value = parent[value]
                        return value

                    selected = set(int(value) for value in chosen)
                    accepted = 0
                    mst_total = 0.0
                    for distance, left, right in self.pairs.ordered_edges:
                        if left not in selected or right not in selected:
                            continue
                        root_left = find(left)
                        root_right = find(right)
                        if root_left == root_right:
                            continue
                        parent[root_right] = root_left
                        mst_total += distance
                        accepted += 1
                        if accepted == key - 1:
                            break
                    if accepted != key - 1:
                        raise M1Error("candidate target graph is disconnected")
                    result[row_index] = (
                        pair_distance / denominator,
                        cross / denominator,
                        homotopic / denominator,
                        mst_total / (key - 1),
                    )
            self._shared_features[key] = _finite(result, "shared set geometry")
        return self._shared_features[key]

    def raw_batch(
        self, cell: Cell, start: int, stop: int
    ) -> np.ndarray:
        basis = self.library.basis(cell.k)
        if not (0 <= start <= stop <= basis.count) or stop - start > BATCH_MAXIMUM:
            raise M1Error("candidate geometry batch exceeds frozen bounds")
        size = stop - start
        if cell.k == 0:
            return np.zeros((size, GEOMETRY_DIMENSION), dtype=float)
        masks = basis.membership[start:stop].astype(float, copy=False)
        distances = _finite(cell.soma_distance_mm, "cell-target distance")
        if distances.shape != (PANEL_SIZE,):
            raise M1Error("cell-target distance vector has the wrong shape")
        mean = (masks @ distances) / float(cell.k)
        mean_square = (masks @ (distances * distances)) / float(cell.k)
        variance = mean_square - mean * mean
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, mean_square)
        if np.any(variance < -tolerance):
            raise M1Error("soma-target population variance became materially negative")
        variance = np.maximum(variance, 0.0)
        raw = np.empty((size, GEOMETRY_DIMENSION), dtype=float)
        raw[:, 0] = mean
        raw[:, 1] = np.sqrt(variance)
        raw[:, 2:] = self.shared_features(cell.k)[start:stop]
        return _finite(raw, "raw candidate geometry")

    def uniform_moments(self, cell: Cell) -> UniformCellMoments:
        basis = self.library.basis(cell.k)
        if cell.k in (0, PANEL_SIZE):
            raw = self.raw_batch(cell, 0, 1)[0]
            return UniformCellMoments(
                geometry_mean=raw,
                raw_joint_covariance=np.zeros(
                    (JOINT_CELL_DIMENSION, JOINT_CELL_DIMENSION), dtype=float
                ),
            )
        total = np.zeros(JOINT_CELL_DIMENSION, dtype=float)
        second = np.zeros((JOINT_CELL_DIMENSION, JOINT_CELL_DIMENSION), dtype=float)
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            raw = self.raw_batch(cell, start, stop)
            joint = np.column_stack((basis.contrasts[start:stop], raw))
            total += np.sum(joint, axis=0, dtype=float)
            second += joint.T @ joint
        mean = total / float(basis.count)
        covariance = second / float(basis.count) - np.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise M1Error("uniform candidate covariance is nonfinite")
        return UniformCellMoments(mean[15:].copy(), covariance)


def equal_group_weights(groups: Sequence[str]) -> np.ndarray:
    if not groups:
        raise M1Error("equal-group weights require rows")
    counts = Counter(str(value) for value in groups)
    number_groups = len(counts)
    weights = np.asarray(
        [1.0 / (number_groups * counts[str(value)]) for value in groups],
        dtype=float,
    )
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise M1Error("equal-group weights do not sum to one")
    return weights


def _soma_monomials(cells: Sequence[Cell]) -> np.ndarray:
    coordinates = np.asarray([cell.soma_um for cell in cells], dtype=float)
    if coordinates.shape != (len(cells), 3) or not np.all(np.isfinite(coordinates)):
        raise M1Error("invalid soma coordinate matrix")
    x, y, z = coordinates.T
    return np.column_stack((x, y, z, x * x, y * y, z * z, x * y, x * z, y * z))


def fit_design_transform(cells: Sequence[Cell]) -> DesignTransform:
    monomials = _soma_monomials(cells)
    mean = np.mean(monomials, axis=0)
    scale = np.sqrt(np.mean((monomials - mean) ** 2, axis=0))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise M1Error("training soma monomial has zero or nonfinite scale")
    return DesignTransform(mean, scale)


def build_design(cells: Sequence[Cell], transform: DesignTransform) -> np.ndarray:
    if not cells:
        return np.empty((0, DESIGN_DIMENSION), dtype=float)
    k = np.asarray([cell.k for cell in cells], dtype=float)
    u = k / 8.0 - 1.0
    soma = (_soma_monomials(cells) - transform.soma_mean) / transform.soma_scale
    layers: list[np.ndarray] = []
    hemisphere: list[float] = []
    segment: list[float] = []
    for cell in cells:
        try:
            layers.append(LAYER_CONTRASTS[LAYERS.index(cell.layer)])
        except ValueError as error:
            raise M1Error(f"unexpected layer {cell.layer!r}") from error
        if cell.hemisphere == "left":
            hemisphere.append(-0.5)
        elif cell.hemisphere == "right":
            hemisphere.append(0.5)
        else:
            raise M1Error(f"unexpected hemisphere {cell.hemisphere!r}")
        if cell.segment_unmapped not in (0, 1):
            raise M1Error("segment_unmapped is not binary")
        segment.append(float(cell.segment_unmapped))
    design = np.column_stack(
        (
            np.ones(len(cells)),
            u,
            0.5 * (3.0 * u * u - 1.0),
            soma,
            np.vstack(layers),
            np.asarray(hemisphere),
            np.asarray(segment),
        )
    )
    if design.shape != (len(cells), DESIGN_DIMENSION) or not np.all(np.isfinite(design)):
        raise M1Error("frozen 17-column design construction failed")
    return design


def apply_fixed_transforms(
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    design_transform: DesignTransform,
    geometry_scales: np.ndarray,
) -> PreparedData:
    """Apply training-only transforms while retaining each cell's own center."""
    subset = tuple(cells)
    subset_moments = tuple(moments)
    if not subset or len(subset) != len(subset_moments):
        raise M1Error("cells and uniform moments are nonempty and aligned")
    scales = np.asarray(geometry_scales, dtype=float).copy()
    if (
        scales.shape != (GEOMETRY_DIMENSION,)
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise M1Error("fixed geometry RMS scales are malformed")
    x = build_design(subset, design_transform)
    weights = equal_group_weights([cell.group for cell in subset])
    centers = np.asarray(
        [moment.geometry_mean for moment in subset_moments], dtype=float
    )
    if centers.shape != (len(subset), GEOMETRY_DIMENSION) or not np.all(
        np.isfinite(centers)
    ):
        raise M1Error("cell-local uniform geometry centers are malformed")
    return PreparedData(
        subset, x, centers, scales, weights, design_transform
    )


def prepare_data(
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    indexes: Sequence[int] | None = None,
) -> PreparedData:
    if len(cells) != len(moments):
        raise M1Error("cells and uniform moments are not aligned")
    selected = tuple(range(len(cells))) if indexes is None else tuple(int(i) for i in indexes)
    subset = tuple(cells[i] for i in selected)
    subset_moments = tuple(moments[i] for i in selected)
    transform = fit_design_transform(subset)
    weights = equal_group_weights([cell.group for cell in subset])
    centered_second = np.asarray(
        [moment.raw_joint_covariance[15:, 15:].diagonal() for moment in subset_moments],
        dtype=float,
    )
    scale_square = weights @ centered_second
    if (
        scale_square.shape != (GEOMETRY_DIMENSION,)
        or not np.all(np.isfinite(scale_square))
        or np.any(scale_square <= 0.0)
    ):
        raise M1Error("fold-local geometry RMS scale is zero or nonfinite")
    scales = np.sqrt(scale_square)
    return apply_fixed_transforms(subset, subset_moments, transform, scales)


def _theta_blocks(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(theta, dtype=float)
    if vector.shape != (PARAMETER_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise M1Error("joint parameter vector is malformed or nonfinite")
    return vector[:B_PARAMETER_DIMENSION].reshape(17, 15), vector[-6:]


def _fsum_dot(left: np.ndarray, right: np.ndarray, label: str) -> float:
    left_array = np.asarray(left, dtype=float).ravel()
    right_array = np.asarray(right, dtype=float).ravel()
    if (
        left_array.shape != right_array.shape
        or not np.all(np.isfinite(left_array))
        or not np.all(np.isfinite(right_array))
    ):
        raise M1Error(f"{label} dot-product inputs are malformed or nonfinite")
    products: list[float] = []
    for left_value, right_value in zip(left_array, right_array, strict=True):
        product = float(left_value) * float(right_value)
        if not math.isfinite(product):
            raise M1Error(f"{label} dot-product product became nonfinite")
        products.append(product)
    try:
        result = math.fsum(products)
    except (OverflowError, ValueError) as error:
        raise M1Error(f"{label} dot product could not be summed finitely") from error
    if not math.isfinite(result):
        raise M1Error(f"{label} dot product became nonfinite")
    return result


def _fsum_batch(values: np.ndarray, label: str) -> float:
    array = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(array)):
        raise M1Error(f"nonfinite {label}")
    try:
        result = math.fsum(array.tolist())
    except (OverflowError, ValueError) as error:
        raise M1Error(f"{label} could not be summed finitely") from error
    if not math.isfinite(result):
        raise M1Error(f"nonfinite {label} sum")
    return result


def _finish_line_cell_difference(
    denominator_parts: Sequence[float],
    numerator_parts: Sequence[float],
    *,
    form: str,
    maximum_shift: float,
) -> float:
    try:
        denominator = math.fsum(float(value) for value in denominator_parts)
        numerator = math.fsum(float(value) for value in numerator_parts)
    except (OverflowError, ValueError) as error:
        raise M1Error("stable line likelihood-ratio sum failed") from error
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise M1Error("stable line current probability denominator is invalid")
    if not math.isfinite(numerator):
        raise M1Error("stable line likelihood-ratio numerator is nonfinite")
    if form == "small_log1p_expm1":
        ratio_minus_one = numerator / denominator
        if not math.isfinite(ratio_minus_one) or ratio_minus_one <= -1.0:
            raise M1Error("stable line log1p domain is invalid")
        result = math.log1p(ratio_minus_one)
    elif form == "general_shifted":
        if not math.isfinite(maximum_shift) or numerator <= 0.0:
            raise M1Error("stable shifted line likelihood ratio is invalid")
        result = maximum_shift + math.log(numerator / denominator)
    else:
        raise M1Error("stable line likelihood-ratio form is unknown")
    if not math.isfinite(result):
        raise M1Error("stable line cell objective difference is nonfinite")
    return result


def _line_difference_form(maximum_absolute_r: float) -> str:
    value = float(maximum_absolute_r)
    if not math.isfinite(value) or value < 0.0:
        raise M1Error("stable line branch magnitude is invalid")
    return "small_log1p_expm1" if value <= 0.5 else "general_shifted"


def _cell_directional_coefficients(
    x: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    direction_b, direction_gamma = _theta_blocks(direction)
    beta_direction = np.asarray(
        [
            math.fsum(
                float(x[column]) * float(direction_b[column, contrast])
                for column in range(DESIGN_DIMENSION)
            )
            for contrast in range(TARGET_CONTRAST_DIMENSION)
        ],
        dtype=float,
    )
    return np.concatenate((beta_direction, direction_gamma))


def _directional_batch_numerator(
    joint: np.ndarray,
    observed_joint: np.ndarray,
    cell_direction: np.ndarray,
    weights: np.ndarray,
) -> float:
    # Center each candidate before taking its directional increment.  Taking
    # E[F] - F_observed first irreversibly rounds away small expectations.
    increments = (joint - observed_joint) @ cell_direction
    return _fsum_batch(weights * increments, "direct directional numerator")


def _finish_directional_cell(
    numerator_parts: list[float], denominator_parts: list[float], weight: float
) -> float:
    numerator = _fsum_batch(np.asarray(numerator_parts), "direct numerator batches")
    denominator = _fsum_batch(
        np.asarray(denominator_parts), "direct normalizer batches"
    )
    if denominator <= 0.0:
        raise M1Error("direct directional normalizer is invalid")
    result = float(weight) * (numerator / denominator)
    if not math.isfinite(result):
        raise M1Error("direct data directional derivative became nonfinite")
    return result


def _finish_directional_derivative(
    data_terms: list[float],
    theta: np.ndarray,
    direction: np.ndarray,
    ridge_lambda: float,
    linear_correction: np.ndarray | None = None,
) -> float:
    if not math.isfinite(float(ridge_lambda)) or ridge_lambda < 0.0:
        raise M1Error("direct directional ridge lambda is invalid")
    # Do not round the total data contribution before combining it with the
    # ridge term: the frozen summation covers cells, ridge, and correction.
    components = list(data_terms)
    ridge = float(ridge_lambda) * _fsum_dot(
        np.asarray(theta, dtype=float),
        np.asarray(direction, dtype=float),
        "ridge directional derivative",
    )
    if not math.isfinite(ridge):
        raise M1Error("ridge directional derivative became nonfinite")
    components.append(ridge)
    if linear_correction is not None:
        components.append(
            _fsum_dot(
                np.asarray(linear_correction, dtype=float),
                np.asarray(direction, dtype=float),
                "linear-correction directional derivative",
            )
        )
    try:
        result = math.fsum(components)
    except (OverflowError, ValueError) as error:
        raise M1Error("direct directional derivative could not be summed") from error
    if not math.isfinite(result):
        raise M1Error("direct directional derivative became nonfinite")
    return result


def _direct_directional_derivative(
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
    theta: np.ndarray,
    direction: np.ndarray,
    ridge_lambda: float,
    linear_correction: np.ndarray | None = None,
) -> tuple[float, int]:
    """Stream the alpha-zero derivative after the Newton direction is known.

    The returned visit count includes both candidate passes.  No rounded
    feature moments or cached derivative from a previous iterate are used.
    """
    b_matrix, gamma = _theta_blocks(theta)
    _theta_blocks(direction)
    data_terms: list[float] = []
    candidate_visits = 0
    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        if not 0 <= cell.observed_index < basis.count:
            raise M1Error("observed candidate index is outside Omega")
        if cell.k in (0, PANEL_SIZE):
            candidate_visits += 2
            continue
        beta = b_matrix.T @ prepared.x[cell_index]
        cell_direction = _cell_directional_coefficients(
            prepared.x[cell_index], direction
        )
        center = prepared.geometry_centers[cell_index]
        scales = prepared.geometry_scales
        maximum = -math.inf
        observed_joint: np.ndarray | None = None
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            z = (geometry.raw_batch(cell, start, stop) - center) / scales
            a = basis.contrasts[start:stop]
            scores = a @ beta + z @ gamma
            if not np.all(np.isfinite(scores)):
                raise M1Error("direct candidate score became nonfinite")
            maximum = max(maximum, float(np.max(scores)))
            if start <= cell.observed_index < stop:
                offset = cell.observed_index - start
                observed_joint = np.concatenate((a[offset], z[offset]))
            candidate_visits += stop - start
        if observed_joint is None or not math.isfinite(maximum):
            raise M1Error("direct candidate maximum or observed feature is invalid")
        numerator_parts: list[float] = []
        denominator_parts: list[float] = []
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            z = (geometry.raw_batch(cell, start, stop) - center) / scales
            a = basis.contrasts[start:stop]
            weights = np.exp(a @ beta + z @ gamma - maximum)
            denominator_parts.append(_fsum_batch(weights, "direct normalizer"))
            numerator_parts.append(
                _directional_batch_numerator(
                    np.column_stack((a, z)), observed_joint, cell_direction, weights
                )
            )
            candidate_visits += stop - start
        data_terms.append(
            _finish_directional_cell(
                numerator_parts, denominator_parts, prepared.weights[cell_index]
            )
        )
    return (
        _finish_directional_derivative(
            data_terms, theta, direction, ridge_lambda, linear_correction
        ),
        candidate_visits,
    )


def evaluate_model(
    theta: np.ndarray,
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
    ridge_lambda: float,
    *,
    need_hessian: bool,
    need_predictions: bool = False,
    linear_correction: np.ndarray | None = None,
    line_origin: np.ndarray | None = None,
    line_direction: np.ndarray | None = None,
    line_step: float | None = None,
) -> Evaluation:
    started = time.perf_counter()
    if not math.isfinite(float(ridge_lambda)) or ridge_lambda < 0.0:
        raise M1Error("ridge lambda must be finite and nonnegative")
    vector = np.asarray(theta, dtype=float)
    b_matrix, gamma = _theta_blocks(vector)
    line_arguments = (line_origin, line_direction, line_step)
    line_mode = any(value is not None for value in line_arguments)
    if line_mode and not all(value is not None for value in line_arguments):
        raise M1Error("stable line evaluation requires origin, direction, and step")
    origin_vector: np.ndarray | None = None
    direction_vector: np.ndarray | None = None
    displacement: np.ndarray | None = None
    origin_b: np.ndarray | None = None
    origin_gamma: np.ndarray | None = None
    displacement_b: np.ndarray | None = None
    displacement_gamma: np.ndarray | None = None
    if line_mode:
        origin_vector = np.asarray(line_origin, dtype=float)
        direction_vector = np.asarray(line_direction, dtype=float)
        step = float(line_step)
        _theta_blocks(origin_vector)
        _theta_blocks(direction_vector)
        if not math.isfinite(step) or step < 0.0:
            raise M1Error("stable line step must be finite and nonnegative")
        expected = origin_vector + step * direction_vector
        if not np.array_equal(vector, expected):
            raise M1Error("stable line candidate is not the frozen trial vector")
        displacement = vector - origin_vector
        origin_b, origin_gamma = _theta_blocks(origin_vector)
        displacement_b, displacement_gamma = _theta_blocks(displacement)
    correction: np.ndarray | None = None
    if linear_correction is not None:
        correction = np.asarray(linear_correction, dtype=float)
        if (
            correction.shape != (PARAMETER_DIMENSION,)
            or not np.all(np.isfinite(correction))
        ):
            raise M1Error("population linear correction is malformed or nonfinite")
    gradient = np.zeros(PARAMETER_DIMENSION, dtype=float)
    hessian = (
        np.zeros((PARAMETER_DIMENSION, PARAMETER_DIMENSION), dtype=float)
        if need_hessian
        else None
    )
    log_probability = (
        np.zeros(len(prepared.cells), dtype=float) if need_predictions else None
    )
    marginals = (
        np.zeros((len(prepared.cells), PANEL_SIZE), dtype=float)
        if need_predictions
        else None
    )
    joint_residuals = (
        np.zeros((len(prepared.cells), JOINT_CELL_DIMENSION), dtype=float)
        if need_hessian or line_mode
        else None
    )
    data_objective = 0.0
    line_data_terms: list[float] = []
    directional_data_terms: list[float] = []
    line_small_increment_cells = 0
    line_general_cells = 0
    candidate_visits = 0
    raw_hessian_symmetry_error: float | None = None

    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        if not 0 <= cell.observed_index < basis.count:
            raise M1Error("observed synthetic candidate index is outside Omega")
        if cell.k in (0, PANEL_SIZE):
            # Preserve the frozen two-pass accounting even though the sole
            # centered sufficient statistic and its data contribution are
            # known exactly to be zero without allocating a batch.
            candidate_visits += 2
            if need_predictions:
                assert log_probability is not None and marginals is not None
                log_probability[cell_index] = 0.0
                marginals[cell_index] = basis.membership[0]
            if line_mode:
                line_small_increment_cells += 1
            continue

        beta = b_matrix.T @ prepared.x[cell_index]
        origin_beta: np.ndarray | None = None
        displacement_beta: np.ndarray | None = None
        cell_direction: np.ndarray | None = None
        direct_observed_joint: np.ndarray | None = None
        if line_mode:
            assert (
                origin_b is not None
                and displacement_b is not None
                and direction_vector is not None
            )
            origin_beta = origin_b.T @ prepared.x[cell_index]
            displacement_beta = displacement_b.T @ prepared.x[cell_index]
            cell_direction = _cell_directional_coefficients(
                prepared.x[cell_index], direction_vector
            )
        center = prepared.geometry_centers[cell_index]
        scales = prepared.geometry_scales
        maximum = -math.inf
        origin_maximum = -math.inf
        combined_maximum = -math.inf
        displacement_minimum = math.inf
        displacement_maximum = -math.inf
        observed_displacement_score = math.nan
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (raw - center) / scales
            scores = basis.contrasts[start:stop] @ beta + z @ gamma
            if not np.all(np.isfinite(scores)):
                raise M1Error("candidate score became nonfinite")
            maximum = max(maximum, float(np.max(scores)))
            if line_mode:
                assert (
                    origin_beta is not None
                    and displacement_beta is not None
                    and origin_gamma is not None
                    and displacement_gamma is not None
                )
                origin_scores = (
                    basis.contrasts[start:stop] @ origin_beta + z @ origin_gamma
                )
                displacement_scores = (
                    basis.contrasts[start:stop] @ displacement_beta
                    + z @ displacement_gamma
                )
                combined_scores = origin_scores + displacement_scores
                if not (
                    np.all(np.isfinite(origin_scores))
                    and np.all(np.isfinite(displacement_scores))
                    and np.all(np.isfinite(combined_scores))
                ):
                    raise M1Error("stable line score became nonfinite")
                origin_maximum = max(
                    origin_maximum, float(np.max(origin_scores))
                )
                combined_maximum = max(
                    combined_maximum, float(np.max(combined_scores))
                )
                displacement_minimum = min(
                    displacement_minimum, float(np.min(displacement_scores))
                )
                displacement_maximum = max(
                    displacement_maximum, float(np.max(displacement_scores))
                )
                if start <= cell.observed_index < stop:
                    offset = cell.observed_index - start
                    observed_displacement_score = float(
                        displacement_scores[offset]
                    )
                    direct_observed_joint = np.concatenate(
                        (basis.contrasts[cell.observed_index], z[offset])
                    )
            candidate_visits += stop - start
        if not math.isfinite(maximum):
            raise M1Error("candidate score maximum is nonfinite")
        line_form: str | None = None
        if line_mode:
            if not all(
                math.isfinite(value)
                for value in (
                    origin_maximum,
                    combined_maximum,
                    displacement_minimum,
                    displacement_maximum,
                    observed_displacement_score,
                )
            ):
                raise M1Error("stable line maxima or observed increment is invalid")
            maximum_absolute_r = max(
                abs(displacement_minimum - observed_displacement_score),
                abs(displacement_maximum - observed_displacement_score),
            )
            if not math.isfinite(maximum_absolute_r):
                raise M1Error("stable line centered increment became nonfinite")
            line_form = _line_difference_form(maximum_absolute_r)

        denominator = 0.0
        first = np.zeros(JOINT_CELL_DIMENSION, dtype=float)
        second = (
            np.zeros((JOINT_CELL_DIMENSION, JOINT_CELL_DIMENSION), dtype=float)
            if need_hessian
            else None
        )
        marginal_numerator = np.zeros(PANEL_SIZE, dtype=float) if need_predictions else None
        observed_joint: np.ndarray | None = None
        observed_score = math.nan
        line_denominator_parts: list[float] = []
        line_numerator_parts: list[float] = []
        directional_denominator_parts: list[float] = []
        directional_numerator_parts: list[float] = []
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (raw - center) / scales
            a = basis.contrasts[start:stop]
            scores = a @ beta + z @ gamma
            weights = np.exp(scores - maximum)
            joint = np.column_stack((a, z))
            denominator += float(np.sum(weights, dtype=float))
            first += weights @ joint
            if need_hessian:
                assert second is not None
                second += joint.T @ (weights[:, None] * joint)
            if need_predictions:
                assert marginal_numerator is not None
                marginal_numerator += weights @ basis.membership[start:stop]
            if start <= cell.observed_index < stop:
                offset = cell.observed_index - start
                observed_joint = joint[offset].copy()
                observed_score = float(scores[offset])
            if line_mode:
                assert (
                    origin_beta is not None
                    and displacement_beta is not None
                    and origin_gamma is not None
                    and displacement_gamma is not None
                    and line_form is not None
                    and cell_direction is not None
                    and direct_observed_joint is not None
                )
                directional_denominator_parts.append(
                    _fsum_batch(weights, "direct normalizer")
                )
                directional_numerator_parts.append(
                    _directional_batch_numerator(
                        joint, direct_observed_joint, cell_direction, weights
                    )
                )
                origin_scores = a @ origin_beta + z @ origin_gamma
                origin_weights = np.exp(origin_scores - origin_maximum)
                line_denominator_parts.append(
                    _fsum_batch(origin_weights, "stable line baseline weights")
                )
                displacement_scores = (
                    a @ displacement_beta + z @ displacement_gamma
                )
                r_values = displacement_scores - observed_displacement_score
                if not np.all(np.isfinite(r_values)):
                    raise M1Error("stable line centered increments became nonfinite")
                if line_form == "small_log1p_expm1":
                    summands = origin_weights * np.expm1(r_values)
                    line_numerator_parts.append(
                        _fsum_batch(
                            summands, "stable small-increment likelihood ratio"
                        )
                    )
                else:
                    combined_scores = origin_scores + displacement_scores
                    combined_weights = np.exp(
                        combined_scores - combined_maximum
                    )
                    line_numerator_parts.append(
                        _fsum_batch(
                            combined_weights,
                            "stable general combined-score weights",
                        )
                    )
            candidate_visits += stop - start
        if (
            not math.isfinite(denominator)
            or denominator <= 0.0
            or observed_joint is None
            or not math.isfinite(observed_score)
        ):
            raise M1Error("exact normalizer or observed candidate is invalid")
        mean = first / denominator
        cell_nll = maximum + math.log(denominator) - observed_score
        weight = float(prepared.weights[cell_index])
        data_objective += weight * cell_nll

        if line_mode:
            assert line_form is not None
            directional_data_terms.append(
                _finish_directional_cell(
                    directional_numerator_parts,
                    directional_denominator_parts,
                    weight,
                )
            )
            if line_form == "small_log1p_expm1":
                line_small_increment_cells += 1
                maximum_shift = 0.0
            else:
                line_general_cells += 1
                maximum_shift = (
                    combined_maximum
                    - origin_maximum
                    - observed_displacement_score
                )
            if not math.isfinite(maximum_shift):
                raise M1Error("stable line normalizer maximum shift is nonfinite")
            cell_difference = _finish_line_cell_difference(
                line_denominator_parts,
                line_numerator_parts,
                form=line_form,
                maximum_shift=maximum_shift,
            )
            weighted_difference = weight * cell_difference
            if not math.isfinite(weighted_difference):
                raise M1Error("stable line weighted objective difference is nonfinite")
            line_data_terms.append(weighted_difference)

        delta_a = mean[:15] - observed_joint[:15]
        delta_z = mean[15:] - observed_joint[15:]
        if joint_residuals is not None:
            joint_residuals[cell_index, :15] = delta_a
            joint_residuals[cell_index, 15:] = delta_z
        gradient[:255] += weight * np.outer(prepared.x[cell_index], delta_a).ravel()
        gradient[255:] += weight * delta_z

        if need_hessian:
            assert second is not None and hessian is not None
            covariance = second / denominator - np.outer(mean, mean)
            covariance = 0.5 * (covariance + covariance.T)
            cov_aa = covariance[:15, :15]
            cov_az = covariance[:15, 15:]
            cov_zz = covariance[15:, 15:]
            x_row = prepared.x[cell_index]
            hessian[:255, :255] += weight * np.einsum(
                "a,b,rs->arbs", x_row, x_row, cov_aa, optimize=True
            ).reshape(255, 255)
            cross = np.einsum("a,rg->arg", x_row, cov_az, optimize=True).reshape(255, 6)
            hessian[:255, 255:] += weight * cross
            hessian[255:, :255] += weight * cross.T
            hessian[255:, 255:] += weight * cov_zz

        if need_predictions:
            assert log_probability is not None and marginals is not None
            assert marginal_numerator is not None
            log_probability[cell_index] = -cell_nll
            marginals[cell_index] = marginal_numerator / denominator

    if correction is not None:
        # The ordinary one-hot objective contains -theta^T F_obs.  Adding
        # theta^T(F_obs-M*) changes only the linear term, yielding the exact
        # planted-population cross entropy while preserving the exhaustive
        # softmax probabilities and analytic Hessian.
        data_objective += float(np.dot(vector, correction))
        gradient += correction
    objective = data_objective + 0.5 * ridge_lambda * float(np.dot(vector, vector))
    gradient += ridge_lambda * vector
    if need_hessian:
        assert hessian is not None
        hessian.flat[:: PARAMETER_DIMENSION + 1] += ridge_lambda
        raw_hessian_symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
        if raw_hessian_symmetry_error > 1.0e-12:
            raise M1Error("analytic Hessian exceeded absolute symmetry tolerance")
        hessian = 0.5 * (hessian + hessian.T)
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise M1Error("objective or gradient became nonfinite")
    line_objective_difference: float | None = None
    direct_directional_derivative: float | None = None
    if line_mode:
        assert (
            origin_vector is not None
            and direction_vector is not None
            and displacement is not None
            and joint_residuals is not None
        )
        ridge_linear = ridge_lambda * _fsum_dot(
            origin_vector, displacement, "stable line ridge linear term"
        )
        ridge_quadratic = 0.5 * ridge_lambda * _fsum_dot(
            displacement, displacement, "stable line ridge quadratic term"
        )
        components = [math.fsum(line_data_terms), ridge_linear, ridge_quadratic]
        if correction is not None:
            components.append(
                _fsum_dot(
                    correction,
                    displacement,
                    "stable line linear-correction difference",
                )
            )
        try:
            line_objective_difference = math.fsum(components)
        except (OverflowError, ValueError) as error:
            raise M1Error("stable line objective difference could not be summed") from error
        if not math.isfinite(line_objective_difference):
            raise M1Error("stable line objective difference became nonfinite")
        direct_directional_derivative = _finish_directional_derivative(
            directional_data_terms,
            vector,
            direction_vector,
            ridge_lambda,
            correction,
        )
    return Evaluation(
        objective=objective,
        gradient=gradient,
        hessian=hessian,
        log_probability_nats=log_probability,
        marginals=marginals,
        candidate_visits=candidate_visits,
        raw_hessian_symmetry_error=raw_hessian_symmetry_error,
        elapsed_seconds=time.perf_counter() - started,
        joint_residuals=joint_residuals,
        line_objective_difference=line_objective_difference,
        direct_directional_derivative=direct_directional_derivative,
        line_small_increment_cells=line_small_increment_cells,
        line_general_cells=line_general_cells,
    )


def _newton_direction(
    hessian: np.ndarray, gradient: np.ndarray, ridge_lambda: float
) -> tuple[np.ndarray, dict[str, float | int]]:
    matrix = np.asarray(hessian, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if (
        matrix.shape != (PARAMETER_DIMENSION, PARAMETER_DIMENSION)
        or gradient.shape != (PARAMETER_DIMENSION,)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(gradient))
        or not math.isfinite(float(ridge_lambda))
        or ridge_lambda <= 0.0
    ):
        raise M1Error("Newton system is malformed or nonfinite")
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_error > 1.0e-12:
        raise M1Error("Newton Hessian is not numerically symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    positive_base = max(float(ridge_lambda), np.finfo(float).eps * scale)
    shifts = [0.0]
    shift = positive_base
    for _ in range(8):
        if shift > 1.0e-4 * scale:
            break
        shifts.append(float(shift))
        shift *= 10.0
    identity = np.eye(PARAMETER_DIMENSION, dtype=float)
    rhs = -gradient
    reason = "no admissible Cholesky shift"
    for attempt, diagonal_shift in enumerate(shifts, start=1):
        system = matrix + diagonal_shift * identity
        try:
            factor = np.linalg.cholesky(system)
            direction = _cholesky_factored_solve(factor, rhs)
        except np.linalg.LinAlgError:
            reason = "Cholesky factorization failed"
            continue
        if not np.all(np.isfinite(direction)):
            reason = "Newton direction became nonfinite"
            continue
        for _ in range(2):
            residual = rhs - system @ direction
            correction = _cholesky_factored_solve(factor, residual)
            if not np.all(np.isfinite(correction)):
                break
            direction = direction + correction
        residual = system @ direction + gradient
        denominator = max(
            np.finfo(float).tiny,
            float(np.linalg.norm(system, ord=np.inf))
            * float(np.linalg.norm(direction, ord=np.inf))
            + float(np.linalg.norm(gradient, ord=np.inf)),
        )
        backward_error = float(np.linalg.norm(residual, ord=np.inf)) / denominator
        directional_derivative = float(np.dot(gradient, direction))
        predicted_decrease = float(
            -directional_derivative - 0.5 * np.dot(direction, matrix @ direction)
        )
        if not math.isfinite(backward_error) or backward_error > 5.0e-11:
            reason = "scaled backward error exceeded 5e-11"
            continue
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            reason = "Newton direction is not a descent direction"
            continue
        if not math.isfinite(predicted_decrease) or predicted_decrease <= 0.0:
            reason = "Newton direction lacks positive predicted decrease"
            continue
        return direction, {
            "factorization_attempts": attempt,
            "diagonal_shift": diagonal_shift,
            "Hessian_scale": scale,
            "scaled_backward_error": backward_error,
            "directional_derivative": directional_derivative,
            "quadratic_predicted_decrease": predicted_decrease,
        }
    raise M1Error("bounded Cholesky solve failed closed: " + reason)


def _strong_wolfe_search(
    vector: np.ndarray,
    value: float,
    gradient: np.ndarray,
    direction: np.ndarray,
    derivative_initial: float,
    evaluate: Callable[[np.ndarray, float], Evaluation],
) -> LineSearchResult:
    if not (
        math.isfinite(float(value))
        and np.all(np.isfinite(vector))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(direction))
        and math.isfinite(float(derivative_initial))
    ):
        raise LineSearchFailure(
            "strong-Wolfe search received a nonfinite initial state",
            evaluations=0,
            expansions=0,
            zoom_iterations=0,
            all_values_finite=False,
            scalar_context={
                "initial_objective_finite": math.isfinite(float(value)),
                "initial_vector_finite": bool(np.all(np.isfinite(vector))),
                "initial_gradient_finite": bool(np.all(np.isfinite(gradient))),
                "direction_finite": bool(np.all(np.isfinite(direction))),
                "initial_directional_derivative_finite": math.isfinite(
                    float(derivative_initial)
                ),
            },
        )
    if not math.isfinite(derivative_initial) or derivative_initial >= 0.0:
        raise LineSearchFailure(
            "strong-Wolfe search requires a descent direction",
            evaluations=0,
            expansions=0,
            zoom_iterations=0,
            all_values_finite=math.isfinite(derivative_initial),
            scalar_context={
                "initial_objective": float(value),
                "initial_directional_derivative": (
                    derivative_initial if math.isfinite(derivative_initial) else None
                ),
            },
        )
    evaluations = 0
    expansions = 0
    zoom_count = 0
    all_finite = True
    last_evaluation: dict[str, float] = {}

    def scalar_context(**extra: float | bool | int | None) -> dict[str, Any]:
        context: dict[str, Any] = {
            "initial_objective": float(value),
            "initial_directional_derivative": derivative_initial,
            **last_evaluation,
        }
        if last_evaluation:
            last_step = last_evaluation["last_evaluated_step"]
            last_difference = last_evaluation[
                "last_candidate_objective_difference"
            ]
            last_derivative = last_evaluation[
                "last_candidate_directional_derivative"
            ]
            armijo_difference_bound = ARMIJO_C1 * last_step * derivative_initial
            context.update(
                {
                    "last_armijo_bound": value + armijo_difference_bound,
                    "last_armijo_difference_bound": armijo_difference_bound,
                    "last_armijo_slack": (
                        armijo_difference_bound - last_difference
                    ),
                    "last_curvature_ratio": (
                        abs(last_derivative) / abs(derivative_initial)
                    ),
                }
            )
        context.update(extra)
        return context

    def at_step(
        step: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float, float, int, int]:
        nonlocal evaluations, all_finite
        if evaluations >= LINE_SEARCH_EVALUATIONS:
            raise LineSearchFailure(
                "strong-Wolfe evaluation ceiling reached",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_count,
                all_values_finite=all_finite,
                scalar_context=scalar_context(requested_step=float(step)),
            )
        candidate = vector + step * direction
        if not np.all(np.isfinite(candidate)):
            all_finite = False
            raise LineSearchFailure(
                "strong-Wolfe candidate vector became nonfinite",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_count,
                all_values_finite=False,
                scalar_context=scalar_context(requested_step=float(step)),
            )
        try:
            result = evaluate(candidate, step)
        except (M1Error, ArithmeticError, ValueError) as error:
            all_finite = False
            raise LineSearchFailure(
                f"strong-Wolfe objective evaluation failed: {error}",
                evaluations=evaluations + 1,
                expansions=expansions,
                zoom_iterations=zoom_count,
                all_values_finite=False,
                scalar_context=scalar_context(requested_step=float(step)),
            ) from error
        candidate_value = result.objective
        candidate_gradient = np.asarray(result.gradient, dtype=float)
        objective_difference = result.line_objective_difference
        derivative = result.direct_directional_derivative
        evaluations += 1
        if not (
            math.isfinite(float(candidate_value))
            and candidate_gradient.shape == gradient.shape
            and np.all(np.isfinite(candidate_gradient))
            and objective_difference is not None
            and math.isfinite(float(objective_difference))
            and derivative is not None
            and math.isfinite(float(derivative))
            and result.line_small_increment_cells >= 0
            and result.line_general_cells >= 0
        ):
            all_finite = False
            raise LineSearchFailure(
                "strong-Wolfe evaluation returned nonfinite values",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_count,
                all_values_finite=False,
                scalar_context=scalar_context(requested_step=float(step)),
            )
        last_evaluation.clear()
        last_evaluation.update(
            {
                "last_evaluated_step": float(step),
                "last_candidate_objective": float(candidate_value),
                "last_candidate_objective_difference": float(
                    objective_difference
                ),
                "last_candidate_directional_derivative": float(derivative),
                "last_small_increment_cells": result.line_small_increment_cells,
                "last_general_cells": result.line_general_cells,
            }
        )
        return (
            candidate,
            float(candidate_value),
            candidate_gradient,
            float(objective_difference),
            float(derivative),
            result.line_small_increment_cells,
            result.line_general_cells,
        )

    def accept(
        step: float,
        candidate: np.ndarray,
        candidate_value: float,
        candidate_gradient: np.ndarray,
        objective_difference: float,
        derivative: float,
        small_increment_cells: int,
        general_cells: int,
    ) -> LineSearchResult:
        armijo_difference_bound = ARMIJO_C1 * step * derivative_initial
        slack = armijo_difference_bound - objective_difference
        ratio = abs(derivative) / abs(derivative_initial)
        if slack < 0.0 or ratio > CURVATURE_C2 or not math.isfinite(slack + ratio):
            raise LineSearchFailure(
                "internal strong-Wolfe acceptance violation",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_count,
                all_values_finite=all_finite,
                scalar_context=scalar_context(requested_step=float(step)),
            )
        return LineSearchResult(
            vector=candidate,
            value=candidate_value,
            gradient=candidate_gradient,
            objective_difference=objective_difference,
            directional_derivative=derivative,
            step=step,
            evaluations=evaluations,
            expansions=expansions,
            zoom_iterations=zoom_count,
            armijo_slack=slack,
            curvature_ratio=ratio,
            small_increment_cells=small_increment_cells,
            general_cells=general_cells,
        )

    def zoom(low: float, high: float, difference_low: float) -> LineSearchResult:
        nonlocal zoom_count
        for _ in range(ZOOM_ITERATIONS):
            zoom_count += 1
            step = 0.5 * (low + high)
            if step == low or step == high:
                raise LineSearchFailure(
                    "strong-Wolfe zoom bracket reached floating-point resolution",
                    evaluations=evaluations,
                    expansions=expansions,
                    zoom_iterations=zoom_count,
                    all_values_finite=all_finite,
                    scalar_context=scalar_context(
                        terminal_bracket_low=float(low),
                        terminal_bracket_high=float(high),
                        terminal_midpoint=float(step),
                        terminal_bracket_absolute_width=abs(float(high - low)),
                        terminal_low_objective=float(value + difference_low),
                        terminal_low_objective_difference=float(difference_low),
                    ),
                )
            (
                candidate,
                candidate_value,
                candidate_gradient,
                objective_difference,
                derivative,
                small_increment_cells,
                general_cells,
            ) = at_step(step)
            armijo_difference_bound = ARMIJO_C1 * step * derivative_initial
            if objective_difference > armijo_difference_bound:
                high = step
                continue
            if abs(derivative) <= CURVATURE_C2 * abs(derivative_initial):
                return accept(
                    step,
                    candidate,
                    candidate_value,
                    candidate_gradient,
                    objective_difference,
                    derivative,
                    small_increment_cells,
                    general_cells,
                )
            if objective_difference >= difference_low:
                high = step
                continue
            if derivative * (high - low) >= 0.0:
                high = low
            low = step
            difference_low = objective_difference
        raise LineSearchFailure(
            "strong-Wolfe zoom iteration ceiling reached",
            evaluations=evaluations,
            expansions=expansions,
            zoom_iterations=zoom_count,
            all_values_finite=all_finite,
            scalar_context=scalar_context(
                terminal_bracket_low=float(low),
                terminal_bracket_high=float(high),
                terminal_midpoint=float(0.5 * (low + high)),
                terminal_bracket_absolute_width=abs(float(high - low)),
                terminal_low_objective=float(value + difference_low),
                terminal_low_objective_difference=float(difference_low),
            ),
        )

    previous_step = 0.0
    previous_difference = 0.0
    step = INITIAL_STEP
    iteration = 0
    while True:
        (
            candidate,
            candidate_value,
            candidate_gradient,
            objective_difference,
            derivative,
            small_increment_cells,
            general_cells,
        ) = at_step(step)
        armijo_difference_bound = ARMIJO_C1 * step * derivative_initial
        if objective_difference > armijo_difference_bound:
            return zoom(previous_step, step, previous_difference)
        if abs(derivative) <= CURVATURE_C2 * abs(derivative_initial):
            return accept(
                step,
                candidate,
                candidate_value,
                candidate_gradient,
                objective_difference,
                derivative,
                small_increment_cells,
                general_cells,
            )
        if iteration > 0 and objective_difference >= previous_difference:
            return zoom(previous_step, step, previous_difference)
        if derivative >= 0.0:
            return zoom(step, previous_step, objective_difference)
        if step >= MAXIMUM_STEP:
            raise LineSearchFailure(
                "strong-Wolfe maximum step reached without a bracket",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_count,
                all_values_finite=all_finite,
                scalar_context=scalar_context(requested_step=float(step)),
            )
        previous_step = step
        previous_difference = objective_difference
        step = min(EXPANSION_FACTOR * step, MAXIMUM_STEP)
        expansions += 1
        iteration += 1


def _upward(value: float) -> float:
    if not math.isfinite(float(value)) or value < 0.0:
        return math.inf
    return math.nextafter(float(value), math.inf)


def _certificate(
    failure: LineSearchFailure,
    objective: float,
    gradient: np.ndarray,
    ridge_lambda: float,
    prior_relative_change: float,
) -> dict[str, Any]:
    eligible = str(failure) in {
        "strong-Wolfe zoom iteration ceiling reached",
        "strong-Wolfe zoom bracket reached floating-point resolution",
    }
    finite = bool(
        failure.all_values_finite
        and math.isfinite(float(objective))
        and gradient.shape == (PARAMETER_DIMENSION,)
        and np.all(np.isfinite(gradient))
        and math.isfinite(float(ridge_lambda))
        and ridge_lambda > 0.0
    )
    gradient_inf = float(np.max(np.abs(gradient))) if finite else math.inf
    scale = max(1.0, abs(float(objective))) if finite else math.inf
    square = _upward(gradient_inf * gradient_inf) if finite else math.inf
    dimension_product = _upward(PARAMETER_DIMENSION * square)
    half = _upward(dimension_product / 2.0)
    absolute_gap = _upward(half / ridge_lambda) if finite else math.inf
    relative_gap = _upward(absolute_gap / scale) if finite else math.inf
    passed = bool(
        eligible
        and finite
        and gradient_inf <= GRADIENT_TOLERANCE
        and relative_gap <= RELATIVE_TOLERANCE
    )
    gradient_pass = bool(finite and gradient_inf <= GRADIENT_TOLERANCE)
    relative_gap_pass = bool(finite and relative_gap <= RELATIVE_TOLERANCE)
    return {
        "attempted": eligible,
        "eligible_failure": eligible,
        "failure_message": str(failure),
        "line_search_all_values_finite": failure.all_values_finite,
        "certificate_inputs_finite": finite,
        "all_values_finite": finite,
        "parameter_dimension": PARAMETER_DIMENSION,
        "objective": float(objective) if math.isfinite(float(objective)) else None,
        "objective_scale": scale if math.isfinite(scale) else None,
        "gradient_infinity_norm": gradient_inf if math.isfinite(gradient_inf) else None,
        "gradient_infinity_tolerance": GRADIENT_TOLERANCE,
        "gradient_predicate_pass": gradient_pass,
        "upward_rounded_absolute_gap_bound": (
            absolute_gap if math.isfinite(absolute_gap) else None
        ),
        "upward_rounded_relative_gap_bound": (
            relative_gap if math.isfinite(relative_gap) else None
        ),
        "relative_gap_tolerance": RELATIVE_TOLERANCE,
        "relative_gap_predicate_pass": relative_gap_pass,
        "prior_relative_objective_change": (
            prior_relative_change if math.isfinite(prior_relative_change) else None
        ),
        "certificate_pass": passed,
    }


def fit_newton(
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
    ridge_lambda: float,
    initial: np.ndarray | None = None,
    *,
    linear_correction: np.ndarray | None = None,
) -> OptimizerResult:
    if ridge_lambda not in LAMBDAS_DESCENDING and ridge_lambda != 1.0e-6:
        raise M1Error("optimizer ridge lambda is outside the frozen grid/recovery value")
    vector = (
        np.zeros(PARAMETER_DIMENSION, dtype=float)
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    if vector.shape != (PARAMETER_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise M1Error("optimizer initial vector is malformed")
    accepted_steps = 0
    relative_change = math.inf
    line_evaluations = 0
    solve_records: list[dict[str, float | int]] = []
    line_records: list[dict[str, float | int]] = []
    steps: list[float] = []
    evaluation_seconds = 0.0
    candidate_visits = 0
    first_ordinary_convergence_iteration: int | None = None
    first_ordinary_convergence_gradient: float | None = None
    first_ordinary_convergence_relative_change: float | None = None
    confirmation_pending = False
    confirmation_attempted = False
    confirmation_accepted_steps = 0
    confirmation_skipped_exact_zero = False

    def confirmation_telemetry() -> dict[str, Any]:
        return {
            "post_witness_confirmation_attempt_limit": (
                POST_WITNESS_CONFIRMATION_ATTEMPTS
            ),
            "first_ordinary_convergence_iteration": (
                first_ordinary_convergence_iteration
            ),
            "first_ordinary_convergence_gradient_infinity_norm": (
                first_ordinary_convergence_gradient
            ),
            "first_ordinary_convergence_relative_objective_change": (
                first_ordinary_convergence_relative_change
            ),
            "post_witness_confirmation_attempted": confirmation_attempted,
            "post_witness_confirmation_accepted_steps": (
                confirmation_accepted_steps
            ),
            "post_witness_confirmation_skipped_exact_zero": (
                confirmation_skipped_exact_zero
            ),
        }

    current = evaluate_model(
        vector,
        prepared,
        library,
        geometry,
        ridge_lambda,
        need_hessian=True,
        linear_correction=linear_correction,
    )
    evaluation_seconds += current.elapsed_seconds
    candidate_visits += current.candidate_visits
    value = current.objective
    gradient = current.gradient
    gradient_norm = float(np.max(np.abs(gradient)))
    if gradient_norm <= GRADIENT_TOLERANCE:
        # Establish the second frozen convergence predicate without inventing
        # a pseudo-step: deterministically re-evaluate the unchanged initial
        # state and require an identical objective and gradient.  The two
        # successive evaluations therefore have relative objective change 0.
        repeated = evaluate_model(
            vector,
            prepared,
            library,
            geometry,
            ridge_lambda,
            need_hessian=False,
            linear_correction=linear_correction,
        )
        evaluation_seconds += repeated.elapsed_seconds
        candidate_visits += repeated.candidate_visits
        relative_change = abs(repeated.objective - value) / max(
            1.0, abs(repeated.objective), abs(value)
        )
        if (
            repeated.objective != value
            or not np.array_equal(repeated.gradient, gradient)
            or relative_change > RELATIVE_TOLERANCE
        ):
            raise M1Error("unchanged initial-state convergence was not deterministic")
        return OptimizerResult(
            vector,
            value,
            gradient_norm,
            0,
            0,
            relative_change,
            {
                "status": "pass",
                "convergence_witness": (
                    "gradient_and_repeated_unchanged_initial_objective"
                ),
                "ridge_lambda": ridge_lambda,
                "iterations": 0,
                "accepted_steps": 0,
                "relative_objective_change": relative_change,
                "successive_parameter_states_identical": True,
                "candidate_visits": candidate_visits,
                "evaluation_seconds": evaluation_seconds,
                **confirmation_telemetry(),
            },
        )

    for iteration in range(1, MAXIMUM_ITERATIONS + 1):
        confirmation_iteration = confirmation_pending
        if confirmation_iteration:
            confirmation_pending = False
            confirmation_attempted = True
        assert current.hessian is not None
        direction, solve = _newton_direction(current.hessian, gradient, ridge_lambda)
        derivative_started = time.perf_counter()
        derivative_initial, derivative_visits = _direct_directional_derivative(
            prepared,
            library,
            geometry,
            vector,
            direction,
            ridge_lambda,
            linear_correction,
        )
        evaluation_seconds += time.perf_counter() - derivative_started
        candidate_visits += derivative_visits
        solve["direct_directional_derivative"] = derivative_initial
        solve_records.append(solve)

        def line_evaluate(candidate: np.ndarray, step: float) -> Evaluation:
            nonlocal evaluation_seconds, candidate_visits
            result = evaluate_model(
                candidate,
                prepared,
                library,
                geometry,
                ridge_lambda,
                need_hessian=False,
                linear_correction=linear_correction,
                line_origin=vector,
                line_direction=direction,
                line_step=step,
            )
            evaluation_seconds += result.elapsed_seconds
            candidate_visits += result.candidate_visits
            return result

        try:
            line = _strong_wolfe_search(
                vector,
                value,
                gradient,
                direction,
                derivative_initial,
                line_evaluate,
            )
        except LineSearchFailure as error:
            certificate = _certificate(
                error, value, gradient, ridge_lambda, relative_change
            )
            if certificate["certificate_pass"]:
                return OptimizerResult(
                    vector,
                    value,
                    gradient_norm,
                    iteration - 1,
                    accepted_steps,
                    relative_change,
                    {
                        "status": "pass",
                        "convergence_witness": "strong_convexity_gap_after_zoom_stall",
                        "ridge_lambda": ridge_lambda,
                        "iterations": iteration - 1,
                        "accepted_steps": accepted_steps,
                        "certificate": certificate,
                        "candidate_visits": candidate_visits,
                        "evaluation_seconds": evaluation_seconds,
                        "solve_records": solve_records,
                        "line_records": line_records,
                        "steps": steps,
                        **confirmation_telemetry(),
                    },
                )
            failure_context: dict[str, Any] = {
                "ridge_lambda": float(ridge_lambda),
                "optimizer_iteration": iteration,
                "completed_iterations": iteration - 1,
                "accepted_steps": accepted_steps,
                "current_objective": float(value),
                "current_gradient_infinity_norm": gradient_norm,
                "prior_relative_objective_change": (
                    relative_change if math.isfinite(relative_change) else None
                ),
                "current_parameter_L2_norm": float(np.linalg.norm(vector)),
                "current_parameter_infinity_norm": float(
                    np.max(np.abs(vector))
                ),
                "successful_line_search_evaluations_before_failure": (
                    line_evaluations
                ),
                "candidate_visits": candidate_visits,
                "evaluation_seconds": evaluation_seconds,
                "last_accepted_step": steps[-1] if steps else None,
                "last_Newton_solve": dict(solve),
                **confirmation_telemetry(),
                "line_search_failure": {
                    "message": str(error),
                    "evaluations": error.evaluations,
                    "expansions": error.expansions,
                    "zoom_iterations": error.zoom_iterations,
                    "all_values_finite": error.all_values_finite,
                    **error.scalar_context,
                },
                "certificate": certificate,
            }
            raise UncertifiedLineSearchFailure(
                "strong-Wolfe search failed without the frozen 261D certificate: "
                + str(error),
                failure_context,
            ) from error
        line_evaluations += line.evaluations
        steps.append(line.step)
        line_records.append(
            {
                "step": line.step,
                "objective_difference": line.objective_difference,
                "initial_directional_derivative": derivative_initial,
                "trial_directional_derivative": line.directional_derivative,
                "armijo_slack": line.armijo_slack,
                "curvature_ratio": line.curvature_ratio,
                "small_increment_cells": line.small_increment_cells,
                "general_cells": line.general_cells,
            }
        )
        new_vector = line.vector
        new_value = line.value
        new_gradient = line.gradient
        relative_change = abs(line.objective_difference) / max(
            1.0, abs(value), abs(new_value)
        )
        vector, value, gradient = new_vector, new_value, new_gradient
        accepted_steps += 1
        if confirmation_iteration:
            confirmation_accepted_steps += 1
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= GRADIENT_TOLERANCE and relative_change <= RELATIVE_TOLERANCE:
            if first_ordinary_convergence_iteration is None:
                first_ordinary_convergence_iteration = iteration
                first_ordinary_convergence_gradient = gradient_norm
                first_ordinary_convergence_relative_change = relative_change
                if gradient_norm == 0.0:
                    confirmation_skipped_exact_zero = True
                else:
                    confirmation_pending = True
            if not confirmation_pending:
                return OptimizerResult(
                    vector,
                    value,
                    gradient_norm,
                    iteration,
                    accepted_steps,
                    relative_change,
                    {
                        "status": "pass",
                        "convergence_witness": (
                            "gradient_and_relative_objective_change"
                        ),
                        "ridge_lambda": ridge_lambda,
                        "iterations": iteration,
                        "accepted_steps": accepted_steps,
                        "line_search_evaluations": line_evaluations,
                        "candidate_visits": candidate_visits,
                        "evaluation_seconds": evaluation_seconds,
                        "solve_records": solve_records,
                        "line_records": line_records,
                        "steps": steps,
                        **confirmation_telemetry(),
                    },
                )
        current = evaluate_model(
            vector,
            prepared,
            library,
            geometry,
            ridge_lambda,
            need_hessian=True,
            linear_correction=linear_correction,
        )
        evaluation_seconds += current.elapsed_seconds
        candidate_visits += current.candidate_visits
        # The Hessian evaluation recomputes objective/gradient at the exact
        # accepted vector; deterministic equality is required fail-closed.
        if current.objective != value or not np.array_equal(current.gradient, gradient):
            raise M1Error("accepted objective/gradient were not deterministic")
    raise M1Error("exact-Hessian Newton reached the 2000-iteration ceiling")


def fit_path(
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
) -> dict[float, OptimizerResult]:
    fitted: dict[float, OptimizerResult] = {}
    initial: np.ndarray | None = None
    for ridge_lambda in LAMBDAS_DESCENDING:
        result = fit_newton(
            prepared, library, geometry, ridge_lambda, initial=initial
        )
        fitted[ridge_lambda] = result
        initial = result.theta
    return fitted


def pair_geometry_from_coordinates(coordinates: np.ndarray) -> PairGeometry:
    coordinates = _finite(coordinates, "target coordinates")
    if coordinates.shape != (PANEL_SIZE, 3):
        raise M1Error("target coordinate matrix must be 16 by 3")
    distance = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=float)
    cross = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=float)
    homotopic = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=float)
    edges: list[tuple[float, int, int]] = []
    for left, right in itertools.combinations(range(PANEL_SIZE), 2):
        value = float(np.linalg.norm(coordinates[right] - coordinates[left]))
        distance[left, right] = distance[right, left] = value
        cross_value = float((left % 2) != (right % 2))
        homotopic_value = float(left // 2 == right // 2 and cross_value == 1.0)
        cross[left, right] = cross[right, left] = cross_value
        homotopic[left, right] = homotopic[right, left] = homotopic_value
        edges.append((value, left, right))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return PairGeometry(distance, cross, homotopic, tuple(edges))


def synthetic_target_coordinates() -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for h in range(8):
        ap = 1.2 + 0.65 * h
        dv = 0.7 + 0.22 * ((3 * h) % 8)
        lateral = 0.8 + 0.17 * h
        rows.append((ap, dv, lateral))
        rows.append((ap, dv, -lateral))
    return np.asarray(rows, dtype=float)


def _synthetic_cell(
    row_index: int,
    group: str,
    k: int,
    normal: np.ndarray,
    uniform: float,
) -> Cell:
    ap = 3.0 + 0.7 * float(normal[0])
    dv = 2.5 + 0.5 * float(normal[1])
    depth = 0.75 + 0.5 * abs(float(normal[2]))
    # Addendum v1 breaks the original exact layer/hemisphere dependency while
    # consuming no random variate: consecutive four-row blocks alternate side.
    hemisphere = "left" if (row_index // 4) % 2 == 0 else "right"
    ml = 5.7 - depth if hemisphere == "left" else 5.7 + depth
    targets = synthetic_target_coordinates()
    soma_canonical = np.asarray((ap, dv, depth), dtype=float)
    distances = np.sqrt(np.sum((targets - soma_canonical) ** 2, axis=1))
    return Cell(
        identifier=f"synthetic_{row_index:05d}",
        group=group,
        layer=LAYERS[row_index % 4],
        hemisphere=hemisphere,
        soma_um=(1000.0 * ap, 1000.0 * dv, 1000.0 * ml),
        soma_distance_mm=distances,
        k=int(k),
        observed_index=0,
        segment_unmapped=int(row_index % 5 == 0),
        sampling_uniform=float(uniform),
    )


def _synthetic_design_rank_gate() -> dict[str, Any]:
    """Show the addendum removes the frozen generator's exact design null."""
    schedules = (
        (16, 34, 20260927, "recovery_544"),
        (4, 17, 20260928, "warm_zero_68"),
    )
    results: list[dict[str, Any]] = []
    dependency_coefficients = np.asarray(
        (-0.7071067811865475, 0.4082482904638631, -0.5773502691896258),
        dtype=float,
    )
    for groups, rows_per_group, seed, label in schedules:
        row_count = groups * rows_per_group
        rng = np.random.Generator(np.random.PCG64(seed))
        normal_block = _draw_synthetic_normal_block(rng, row_count)
        cells: list[Cell] = []
        row_index = 0
        for group_index in range(groups):
            group = f"synthetic_group_{group_index:02d}"
            for local_index in range(rows_per_group):
                cells.append(
                    _synthetic_cell(
                        row_index,
                        group,
                        local_index % 17,
                        normal_block[row_index],
                        0.5,
                    )
                )
                row_index += 1
        transform = fit_design_transform(cells)
        design = build_design(cells, transform)
        singular = np.linalg.svd(design, compute_uv=False)
        tolerance = max(design.shape) * np.finfo(float).eps * float(singular[0])
        rank = int(np.sum(singular > tolerance))
        old_hemisphere = np.where(np.arange(row_count) % 2 == 0, -0.5, 0.5)
        old_layer = np.vstack(
            [LAYER_CONTRASTS[index % 4] for index in range(row_count)]
        )
        dependency_error = float(
            np.max(
                np.abs(
                    old_hemisphere
                    - old_layer @ dependency_coefficients
                )
            )
        )
        if rank != DESIGN_DIMENSION or dependency_error > 2.0e-15:
            raise AssertionError(
                f"synthetic design addendum gate failed for {label}"
            )
        counts = Counter((cell.layer, cell.hemisphere) for cell in cells)
        results.append(
            {
                "schedule": label,
                "rows": row_count,
                "rank": rank,
                "rank_tolerance": tolerance,
                "largest_singular_value": float(singular[0]),
                "smallest_singular_value": float(singular[-1]),
                "condition_number_2": float(singular[0] / singular[-1]),
                "layer_by_hemisphere_counts": {
                    f"{layer}__{hemisphere}": counts[(layer, hemisphere)]
                    for layer in LAYERS
                    for hemisphere in ("left", "right")
                },
                "original_schedule_dependency_max_absolute_error": (
                    dependency_error
                ),
            }
        )
    return {
        "status": "pass",
        "addendum": SYNTHETIC_DESIGN_ADDENDUM_PATH.name,
        "replacement_schedule": "hemisphere_index=(row_index//4)%2",
        "consumes_random_variates": False,
        "schedules": results,
    }


def planted_parameters() -> np.ndarray:
    raw = np.empty((17, 15), dtype=float)
    for a in range(17):
        for r in range(15):
            raw[a, r] = 0.08 * math.sin((a + 1) * (r + 1)) + 0.04 * math.cos(
                (a + 2) * (r + 1)
            )
    raw *= 1.5 / float(np.linalg.norm(raw))
    gamma = np.asarray((0.45, -0.35, 0.30, 0.25, -0.20, 0.40), dtype=float)
    return np.concatenate((raw.ravel(), gamma))


def _draw_synthetic_normal_block(
    rng: np.random.Generator, row_count: int
) -> np.ndarray:
    """Consume exactly one three-normal draw per row, in row order."""
    if row_count <= 0:
        raise M1Error("synthetic normal block requires positive row count")
    normals = np.empty((row_count, 3), dtype=float)
    for row_index in range(row_count):
        normals[row_index] = rng.standard_normal(3)
    return normals


def _candidate_score_batch(
    theta: np.ndarray,
    prepared: PreparedData,
    cell_index: int,
    start: int,
    stop: int,
    library: CandidateLibrary,
    geometry: GeometryEngine,
) -> np.ndarray:
    b_matrix, gamma = _theta_blocks(theta)
    cell = prepared.cells[cell_index]
    basis = library.basis(cell.k)
    raw = geometry.raw_batch(cell, start, stop)
    z = (raw - prepared.geometry_centers[cell_index]) / prepared.geometry_scales
    beta = b_matrix.T @ prepared.x[cell_index]
    return basis.contrasts[start:stop] @ beta + z @ gamma


def sample_candidate_index(
    theta: np.ndarray,
    prepared: PreparedData,
    cell_index: int,
    library: CandidateLibrary,
    geometry: GeometryEngine,
    uniform: float,
) -> int:
    cell = prepared.cells[cell_index]
    basis = library.basis(cell.k)
    if basis.count == 1:
        return 0
    maximum = -math.inf
    for start in range(0, basis.count, BATCH_MAXIMUM):
        stop = min(start + BATCH_MAXIMUM, basis.count)
        scores = _candidate_score_batch(
            theta, prepared, cell_index, start, stop, library, geometry
        )
        maximum = max(maximum, float(np.max(scores)))
    denominator = 0.0
    for start in range(0, basis.count, BATCH_MAXIMUM):
        stop = min(start + BATCH_MAXIMUM, basis.count)
        scores = _candidate_score_batch(
            theta, prepared, cell_index, start, stop, library, geometry
        )
        denominator += float(np.sum(np.exp(scores - maximum), dtype=float))
    threshold = min(max(float(uniform), 0.0), math.nextafter(1.0, 0.0)) * denominator
    cumulative = 0.0
    for start in range(0, basis.count, BATCH_MAXIMUM):
        stop = min(start + BATCH_MAXIMUM, basis.count)
        scores = _candidate_score_batch(
            theta, prepared, cell_index, start, stop, library, geometry
        )
        weights = np.exp(scores - maximum)
        partial = np.cumsum(weights)
        if cumulative + float(partial[-1]) > threshold:
            offset = int(np.searchsorted(partial, threshold - cumulative, side="right"))
            return min(start + offset, basis.count - 1)
        cumulative += float(partial[-1])
    return basis.count - 1


def expected_cell_joint_statistics(
    theta: np.ndarray,
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
) -> PopulationExpectation:
    """Enumerate exact per-cell E[(a(S), ztilde(S))] under ``theta``.

    Only the frozen 21-dimensional streamed moment is retained for each cell;
    no candidate-by-261 representation is formed.  These means are later
    expanded once through each cell's 17-dimensional design row.
    """
    started = time.perf_counter()
    b_matrix, gamma = _theta_blocks(theta)
    joint_means = np.zeros((len(prepared.cells), JOINT_CELL_DIMENSION), dtype=float)
    marginals = np.zeros((len(prepared.cells), PANEL_SIZE), dtype=float)
    probability_error = 0.0
    candidate_visits = 0
    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        if cell.k in (0, PANEL_SIZE):
            # Both target contrasts and centered geometry are exactly zero for
            # the singleton candidate.  Preserve the frozen two-pass count.
            marginals[cell_index] = basis.membership[0]
            candidate_visits += 2
            continue

        beta = b_matrix.T @ prepared.x[cell_index]
        center = prepared.geometry_centers[cell_index]
        scales = prepared.geometry_scales
        maximum = -math.inf
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (raw - center) / scales
            scores = basis.contrasts[start:stop] @ beta + z @ gamma
            if not np.all(np.isfinite(scores)):
                raise M1Error("population candidate score became nonfinite")
            maximum = max(maximum, float(np.max(scores)))
            candidate_visits += stop - start
        if not math.isfinite(maximum):
            raise M1Error("population candidate score maximum is nonfinite")

        denominator = 0.0
        first = np.zeros(JOINT_CELL_DIMENSION, dtype=float)
        marginal_numerator = np.zeros(PANEL_SIZE, dtype=float)
        batch_weight_sums: list[float] = []
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            raw = geometry.raw_batch(cell, start, stop)
            z = (raw - center) / scales
            a = basis.contrasts[start:stop]
            scores = a @ beta + z @ gamma
            weights = np.exp(scores - maximum)
            batch_total = float(np.sum(weights, dtype=float))
            denominator += batch_total
            batch_weight_sums.append(batch_total)
            joint = np.column_stack((a, z))
            first += weights @ joint
            marginal_numerator += weights @ basis.membership[start:stop]
            candidate_visits += stop - start
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise M1Error("population exact normalizer is invalid")
        joint_means[cell_index] = first / denominator
        marginals[cell_index] = marginal_numerator / denominator
        normalized_sum = math.fsum(batch_weight_sums) / denominator
        probability_error = max(probability_error, abs(normalized_sum - 1.0))

    if not (
        np.all(np.isfinite(joint_means))
        and np.all(np.isfinite(marginals))
        and math.isfinite(probability_error)
    ):
        raise M1Error("population expected sufficient statistics are nonfinite")
    return PopulationExpectation(
        joint_mean=joint_means,
        marginals=marginals,
        probability_sum_maximum_absolute_error=probability_error,
        candidate_visits=candidate_visits,
        elapsed_seconds=time.perf_counter() - started,
    )


def observed_cell_joint_statistics(
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
) -> np.ndarray:
    """Return each sampled one-hot candidate's frozen 21D statistic."""
    observed = np.zeros(
        (len(prepared.cells), JOINT_CELL_DIMENSION), dtype=float
    )
    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        if not 0 <= cell.observed_index < basis.count:
            raise M1Error("observed synthetic candidate index is outside Omega")
        if cell.k in (0, PANEL_SIZE):
            continue
        raw = geometry.raw_batch(
            cell, cell.observed_index, cell.observed_index + 1
        )[0]
        z = (
            raw - prepared.geometry_centers[cell_index]
        ) / prepared.geometry_scales
        observed[cell_index, :15] = basis.contrasts[cell.observed_index]
        observed[cell_index, 15:] = z
    if not np.all(np.isfinite(observed)):
        raise M1Error("observed joint sufficient statistics are nonfinite")
    return observed


def aggregate_population_linear_correction(
    prepared: PreparedData,
    expected_joint: np.ndarray,
    observed_joint: np.ndarray,
) -> PopulationLinearCorrection:
    """Compute c = F_obs - M* after equal-group aggregation.

    Adding ``theta.T @ c`` to the sampled objective changes its linear term
    into the exact expected-statistic population objective.  The normalizer,
    gradient expectation, and Hessian remain on the existing evaluator path.
    """
    expected = np.asarray(expected_joint, dtype=float)
    observed = np.asarray(observed_joint, dtype=float)
    required_shape = (len(prepared.cells), JOINT_CELL_DIMENSION)
    if (
        expected.shape != required_shape
        or observed.shape != required_shape
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(observed))
    ):
        raise M1Error("population expected/observed statistics are malformed")
    observed_aggregate = np.zeros(PARAMETER_DIMENSION, dtype=float)
    expected_aggregate = np.zeros(PARAMETER_DIMENSION, dtype=float)
    for cell_index in range(len(prepared.cells)):
        weight = float(prepared.weights[cell_index])
        x_row = prepared.x[cell_index]
        observed_aggregate[:255] += weight * np.outer(
            x_row, observed[cell_index, :15]
        ).ravel()
        observed_aggregate[255:] += weight * observed[cell_index, 15:]
        expected_aggregate[:255] += weight * np.outer(
            x_row, expected[cell_index, :15]
        ).ravel()
        expected_aggregate[255:] += weight * expected[cell_index, 15:]
    vector = observed_aggregate - expected_aggregate
    if not (
        np.all(np.isfinite(observed_aggregate))
        and np.all(np.isfinite(expected_aggregate))
        and np.all(np.isfinite(vector))
    ):
        raise M1Error("population aggregate linear correction is nonfinite")
    return PopulationLinearCorrection(
        vector=vector,
        observed_aggregate=observed_aggregate,
        expected_aggregate=expected_aggregate,
    )


def generate_synthetic_dataset(
    *, groups: int, rows_per_group: int, seed: int
) -> tuple[
    tuple[Cell, ...],
    tuple[UniformCellMoments, ...],
    PreparedData,
    CandidateLibrary,
    GeometryEngine,
    np.ndarray,
]:
    if rows_per_group not in (17, 34):
        raise M1Error("synthetic rows per group must be 17 or 34")
    rng = np.random.Generator(np.random.PCG64(seed))
    normal_block = _draw_synthetic_normal_block(rng, groups * rows_per_group)
    specifications: list[tuple[int, str, int, np.ndarray]] = []
    row_index = 0
    for group_index in range(groups):
        group = f"synthetic_group_{group_index:02d}"
        for local_index in range(rows_per_group):
            normal = normal_block[row_index]
            specifications.append((row_index, group, local_index % 17, normal))
            row_index += 1
    # The frozen RNG order consumes every three-normal row block first.  No
    # categorical uniform is drawn until all transforms/features/scales exist.
    cells = [
        _synthetic_cell(index, group, k, normal, 0.5)
        for index, group, k, normal in specifications
    ]
    library = CandidateLibrary()
    pairs = pair_geometry_from_coordinates(synthetic_target_coordinates())
    geometry = GeometryEngine(library, pairs)
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    provisional = prepare_data(cells, moments)
    theta = planted_parameters()
    sampled: list[Cell] = []
    for index, cell in enumerate(cells):
        uniform = float(rng.random())
        observed = sample_candidate_index(
            theta,
            provisional,
            index,
            library,
            geometry,
            uniform,
        )
        sampled.append(
            Cell(
                identifier=cell.identifier,
                group=cell.group,
                layer=cell.layer,
                hemisphere=cell.hemisphere,
                soma_um=cell.soma_um,
                soma_distance_mm=cell.soma_distance_mm,
                k=cell.k,
                observed_index=observed,
                segment_unmapped=cell.segment_unmapped,
                sampling_uniform=uniform,
            )
        )
    prepared = prepare_data(sampled, moments)
    return tuple(sampled), moments, prepared, library, geometry, theta


def compare_conditional_distributions(
    left: np.ndarray,
    right: np.ndarray,
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
) -> dict[str, float]:
    total_kl = 0.0
    total_marginal_error = 0.0
    maximum_probability_error = 0.0
    maximum_log_probability_error = 0.0
    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        maxima = [-math.inf, -math.inf]
        for parameter_index, theta in enumerate((left, right)):
            for start in range(0, basis.count, BATCH_MAXIMUM):
                stop = min(start + BATCH_MAXIMUM, basis.count)
                scores = _candidate_score_batch(
                    theta, prepared, cell_index, start, stop, library, geometry
                )
                maxima[parameter_index] = max(
                    maxima[parameter_index], float(np.max(scores))
                )
        denominators = [0.0, 0.0]
        for parameter_index, theta in enumerate((left, right)):
            for start in range(0, basis.count, BATCH_MAXIMUM):
                stop = min(start + BATCH_MAXIMUM, basis.count)
                scores = _candidate_score_batch(
                    theta, prepared, cell_index, start, stop, library, geometry
                )
                denominators[parameter_index] += float(
                    np.sum(np.exp(scores - maxima[parameter_index]), dtype=float)
                )
        marginals = [np.zeros(PANEL_SIZE), np.zeros(PANEL_SIZE)]
        cell_kl = 0.0
        for start in range(0, basis.count, BATCH_MAXIMUM):
            stop = min(start + BATCH_MAXIMUM, basis.count)
            probabilities: list[np.ndarray] = []
            logs: list[np.ndarray] = []
            for parameter_index, theta in enumerate((left, right)):
                scores = _candidate_score_batch(
                    theta, prepared, cell_index, start, stop, library, geometry
                )
                logp = scores - maxima[parameter_index] - math.log(
                    denominators[parameter_index]
                )
                logs.append(logp)
                probabilities.append(np.exp(logp))
                marginals[parameter_index] += (
                    probabilities[-1] @ basis.membership[start:stop]
                )
            cell_kl += float(np.sum(probabilities[0] * (logs[0] - logs[1])))
            maximum_probability_error = max(
                maximum_probability_error,
                float(np.max(np.abs(probabilities[0] - probabilities[1]))),
            )
            maximum_log_probability_error = max(
                maximum_log_probability_error,
                float(np.max(np.abs(logs[0] - logs[1]))),
            )
        total_kl += cell_kl / LN2
        total_marginal_error += float(np.mean(np.abs(marginals[0] - marginals[1])))
    n = len(prepared.cells)
    return {
        "mean_conditional_KL_bits": total_kl / n,
        "exact_inclusion_marginal_MAE": total_marginal_error / n,
        "maximum_candidate_probability_absolute_difference": maximum_probability_error,
        "maximum_candidate_log_probability_absolute_difference": maximum_log_probability_error,
    }


def _gram_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    symmetric = 0.5 * (matrix + matrix.T)
    singular = np.linalg.svd(symmetric, compute_uv=False)
    maximum = float(singular[0]) if singular.size else 0.0
    tolerance = max(symmetric.shape) * np.finfo(float).eps * maximum
    rank = int(np.sum(singular > tolerance))
    minimum = float(singular[-1]) if singular.size else 0.0
    condition = maximum / minimum if minimum > 0.0 else math.inf
    return {
        "dimension": symmetric.shape[0],
        "rank": rank,
        "rank_tolerance": tolerance,
        "largest_singular_value": maximum,
        "smallest_singular_value": minimum,
        "condition_number_2": condition if math.isfinite(condition) else None,
        "finite": bool(np.all(np.isfinite(symmetric))),
    }


def fold_gram(
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    indexes: Sequence[int],
) -> tuple[np.ndarray, PreparedData]:
    prepared = prepare_data(cells, moments, indexes)
    gram = np.zeros((PARAMETER_DIMENSION, PARAMETER_DIMENSION), dtype=float)
    for cell_index, cell in enumerate(prepared.cells):
        original_index = int(indexes[cell_index])
        covariance = moments[original_index].raw_joint_covariance
        cov_aa = covariance[:15, :15]
        cov_az = covariance[:15, 15:] / prepared.geometry_scales[None, :]
        cov_zz = (
            covariance[15:, 15:]
            / prepared.geometry_scales[:, None]
            / prepared.geometry_scales[None, :]
        )
        x_row = prepared.x[cell_index]
        weight = float(prepared.weights[cell_index])
        gram[:255, :255] += weight * np.einsum(
            "a,b,rs->arbs", x_row, x_row, cov_aa, optimize=True
        ).reshape(255, 255)
        cross = np.einsum("a,rg->arg", x_row, cov_az, optimize=True).reshape(255, 6)
        gram[:255, 255:] += weight * cross
        gram[255:, :255] += weight * cross.T
        gram[255:, 255:] += weight * cov_zz
    return 0.5 * (gram + gram.T), prepared


def rank_condition_gate(
    cells: Sequence[Cell],
    moments: Sequence[UniformCellMoments],
    folds: Sequence[tuple[str, Sequence[int]]],
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    passed = True
    for label, indexes in folds:
        ordered = tuple(int(value) for value in indexes)
        gram, _ = fold_gram(cells, moments, ordered)
        joint = _gram_diagnostics(gram)
        geometry_block = _gram_diagnostics(gram[255:, 255:])
        b_block = gram[:255, :255]
        b_diagnostics = _gram_diagnostics(b_block)
        residual = gram[255:, 255:] - gram[255:, :255] @ np.linalg.pinv(
            b_block,
            rcond=max(b_block.shape) * np.finfo(float).eps,
            hermitian=True,
        ) @ gram[:255, 255:]
        residual = 0.5 * (residual + residual.T)
        geometry_residual = _gram_diagnostics(residual)
        joint_condition = joint["condition_number_2"]
        fold_pass = bool(
            geometry_block["rank"] == 6
            and geometry_residual["rank"] == 6
            and joint["rank"] == PARAMETER_DIMENSION
            and joint_condition is not None
            and float(joint_condition) <= 1.0e10
        )
        passed = passed and fold_pass
        results.append(
            {
                "fold": label,
                "training_cells": len(ordered),
                "training_groups": len({cells[i].group for i in ordered}),
                "joint": joint,
                "B_block": b_diagnostics,
                "geometry_block": geometry_block,
                "geometry_residual": geometry_residual,
                "status": "pass" if fold_pass else "fail",
            }
        )
    if not passed and raise_on_failure:
        raise M1Error("target-free joint/geometry rank or condition gate failed")
    return {
        "status": "pass" if passed else "fail",
        "unique_training_sets_evaluated": len(results),
        "folds": results,
    }


def _signed_spectrum(matrix: np.ndarray) -> dict[str, Any]:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    absolute = np.abs(eigenvalues)
    largest = float(np.max(absolute)) if absolute.size else 0.0
    tolerance = max(symmetric.shape) * np.finfo(float).eps * largest
    rank = int(np.sum(absolute > tolerance))
    smallest_absolute = float(np.min(absolute)) if absolute.size else 0.0
    condition = largest / smallest_absolute if smallest_absolute > 0.0 else math.inf
    weakest_index = int(np.argmin(absolute)) if absolute.size else 0
    return {
        "dimension": int(symmetric.shape[0]),
        "rank": rank,
        "rank_tolerance": tolerance,
        "smallest_signed_eigenvalue": float(eigenvalues[0]),
        "largest_signed_eigenvalue": float(eigenvalues[-1]),
        "smallest_absolute_eigenvalue": smallest_absolute,
        "largest_absolute_eigenvalue": largest,
        "condition_number_2": condition if math.isfinite(condition) else None,
        "negative_eigenvalues_below_minus_tolerance": int(
            np.sum(eigenvalues < -tolerance)
        ),
        "smallest_12_signed_eigenvalues": [
            float(value) for value in eigenvalues[:12]
        ],
        "smallest_12_absolute_eigenvalues": [
            float(value) for value in np.sort(absolute)[:12]
        ],
        "weakest_eigenvector": [
            float(value) for value in eigenvectors[:, weakest_index]
        ],
    }


def synthetic_full_rank_diagnostics(
    cells: Sequence[Cell], moments: Sequence[UniformCellMoments]
) -> dict[str, Any]:
    indexes = tuple(range(len(cells)))
    gram, _ = fold_gram(cells, moments, indexes)
    b_block = gram[:255, :255]
    cross = gram[:255, 255:]
    geometry_block = gram[255:, 255:]
    b_pseudoinverse = np.linalg.pinv(
        b_block,
        rcond=max(b_block.shape) * np.finfo(float).eps,
        hermitian=True,
    )
    residual = geometry_block - cross.T @ b_pseudoinverse @ cross
    residual = 0.5 * (residual + residual.T)
    cross_norm = float(np.linalg.norm(cross, ord=2))
    range_defect = float(
        np.linalg.norm(cross - b_block @ b_pseudoinverse @ cross, ord=2)
        / max(cross_norm, np.finfo(float).tiny)
    )
    geometry_spectrum = _signed_spectrum(geometry_block)
    residual_spectrum = _signed_spectrum(residual)
    return {
        "joint": _signed_spectrum(gram),
        "B_block": _signed_spectrum(b_block),
        "geometry_block": geometry_spectrum,
        "geometry_beyond_M0_Schur_residual": residual_spectrum,
        "B_to_geometry_cross_range_relative_defect": range_defect,
        "ordered_geometry_features": list(FEATURE_NAMES),
        "geometry_block_matrix": geometry_block.tolist(),
        "geometry_beyond_M0_Schur_residual_matrix": residual.tolist(),
        "geometry_block_weakest_feature_loadings": dict(
            zip(FEATURE_NAMES, geometry_spectrum["weakest_eigenvector"])
        ),
        "geometry_residual_weakest_feature_loadings": dict(
            zip(FEATURE_NAMES, residual_spectrum["weakest_eigenvector"])
        ),
    }


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file() or path.is_symlink():
        raise M1Error(f"required CSV is absent or symlinked: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise M1Error(f"CSV lacks a header: {path}")
        return tuple(reader.fieldnames), list(reader)


def _require_exact_scratch_child(path: Path, basename: str, label: str) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise M1Error("SCRATCH is unset")
    scratch = Path(scratch_value).resolve(strict=True)
    episode = (
        scratch / "br_autoresearch" / "episode10_single_cell_coprojection"
    ).resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    if (
        absolute.name != basename
        or absolute.parent.resolve(strict=True) != episode
        or not absolute.is_dir()
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != absolute
    ):
        raise M1Error(f"{label} must be exact unsymlinked EP10 Scratch child {basename}")
    return absolute


def load_target_free_real_shape(
    interface_root_argument: Path, geometry_root_argument: Path
) -> tuple[tuple[Cell, ...], PairGeometry]:
    interface_root = _require_exact_scratch_child(
        interface_root_argument, "v3_model_interface_preflight_v1", "interface root"
    )
    geometry_root = _require_exact_scratch_child(
        geometry_root_argument, "v3_m1_geometry_v1", "geometry root"
    )
    interface_summary_path = interface_root / "v3_model_interface_summary.json"
    geometry_summary_path = geometry_root / "geometry_summary.json"
    for path in (interface_summary_path, geometry_summary_path):
        if not path.is_file() or path.is_symlink():
            raise M1Error(f"required target-free summary is absent or symlinked: {path}")
    with interface_summary_path.open("r", encoding="utf-8") as handle:
        interface_summary = json.load(handle)
    with geometry_summary_path.open("r", encoding="utf-8") as handle:
        geometry_summary = json.load(handle)
    if not (
        interface_summary.get("status") == "pass"
        and int(interface_summary.get("interface_rows", -1)) == 194
        and interface_summary.get("projection_outcomes_opened_or_emitted") is False
        and interface_summary.get("final_or_unassigned_files_opened") is False
        and geometry_summary.get("status") == "pass"
        and int(geometry_summary.get("interface_rows", -1)) == 194
        and geometry_summary.get("target_calls_or_swcs_opened") is False
        and geometry_summary.get("realized_axon_geometry_opened") is False
        and geometry_summary.get("final_or_unassigned_opened") is False
    ):
        raise M1Error("target-free interface or geometry summary is not passed/sealed")

    interface_fields, interface_rows = _read_csv(
        interface_root / "v3_model_interface.csv"
    )
    if interface_fields != INTERFACE_FIELDS or len(interface_rows) != 194:
        raise M1Error("target-free interface schema or row count changed")
    by_identifier: dict[str, dict[str, str]] = {}
    for row in interface_rows:
        identifier = row["provider_neuron_id"]
        if not identifier or identifier in by_identifier:
            raise M1Error("interface identifiers are empty or duplicated")
        by_identifier[identifier] = row
    observed_groups = dict(
        sorted(Counter(row["fMOST_brain_id"] for row in interface_rows).items())
    )
    if observed_groups != GROUP_SIZES:
        raise M1Error("target-free interface group sizes changed")

    anchor_fields, anchor_rows = _read_csv(geometry_root / "atlas_target_anchors.csv")
    if len(anchor_rows) != 16 or "coordinate_index" not in anchor_fields:
        raise M1Error("atlas target-anchor geometry changed")
    anchor_indexes = sorted(int(row["coordinate_index"]) for row in anchor_rows)
    if anchor_indexes != list(range(16)):
        raise M1Error("atlas target-anchor indexes changed")

    cell_fields, cell_rows = _read_csv(geometry_root / "cell_target_geometry.csv")
    required_cell_fields = {
        "provider_neuron_id",
        "fMOST_brain_id",
        "coordinate_index",
        "coordinate_id",
        "soma_target_distance_mm",
    }
    if len(cell_rows) != 3104 or not required_cell_fields.issubset(cell_fields):
        raise M1Error("cell-target geometry schema or row count changed")
    distances: dict[str, np.ndarray] = {
        identifier: np.full(PANEL_SIZE, np.nan, dtype=float)
        for identifier in by_identifier
    }
    for row in cell_rows:
        identifier = row["provider_neuron_id"]
        if identifier not in distances:
            raise M1Error("cell-target geometry contains an unknown cell")
        interface_row = by_identifier[identifier]
        if row["fMOST_brain_id"] != interface_row["fMOST_brain_id"]:
            raise M1Error("cell-target geometry group mismatch")
        index = int(row["coordinate_index"])
        if not 0 <= index < PANEL_SIZE or row["coordinate_id"] != PANEL[index]:
            raise M1Error("cell-target coordinate order changed")
        if math.isfinite(float(distances[identifier][index])):
            raise M1Error("duplicate cell-target geometry row")
        distances[identifier][index] = float(row["soma_target_distance_mm"])
    if any(not np.all(np.isfinite(value)) for value in distances.values()):
        raise M1Error("cell-target geometry join is incomplete")

    pair_fields, pair_rows = _read_csv(geometry_root / "target_pair_geometry.csv")
    required_pair_fields = {
        "coordinate_index_a",
        "coordinate_index_b",
        "target_pair_distance_mm",
        "cross_side",
        "homotopic",
    }
    if len(pair_rows) != 120 or not required_pair_fields.issubset(pair_fields):
        raise M1Error("target-pair geometry schema or row count changed")
    distance_matrix = np.zeros((16, 16), dtype=float)
    cross_matrix = np.zeros((16, 16), dtype=float)
    homotopic_matrix = np.zeros((16, 16), dtype=float)
    seen_pairs: set[tuple[int, int]] = set()
    edges: list[tuple[float, int, int]] = []
    for row in pair_rows:
        left = int(row["coordinate_index_a"])
        right = int(row["coordinate_index_b"])
        if not 0 <= left < right < 16 or (left, right) in seen_pairs:
            raise M1Error("target-pair indexes are invalid or duplicated")
        seen_pairs.add((left, right))
        distance_value = float(row["target_pair_distance_mm"])
        cross_value = float(int(row["cross_side"]))
        homotopic_value = float(int(row["homotopic"]))
        if not math.isfinite(distance_value) or cross_value not in (0.0, 1.0) or homotopic_value not in (0.0, 1.0):
            raise M1Error("target-pair geometry contains invalid values")
        distance_matrix[left, right] = distance_matrix[right, left] = distance_value
        cross_matrix[left, right] = cross_matrix[right, left] = cross_value
        homotopic_matrix[left, right] = homotopic_matrix[right, left] = homotopic_value
        edges.append((distance_value, left, right))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    pairs = PairGeometry(
        distance_matrix, cross_matrix, homotopic_matrix, tuple(edges)
    )

    ordered_rows: list[dict[str, str]] = []
    for group in GROUP_ORDER:
        group_rows = [row for row in interface_rows if row["fMOST_brain_id"] == group]
        group_rows.sort(key=lambda row: int(row["provider_neuron_id"]))
        ordered_rows.extend(group_rows)
    cells: list[Cell] = []
    global_index = 0
    for group_index, group in enumerate(GROUP_ORDER):
        group_rows = [row for row in ordered_rows if row["fMOST_brain_id"] == group]
        for local_index, row in enumerate(group_rows):
            k = 1 + ((local_index + 7 * group_index) % 15)
            count = math.comb(16, k)
            observed_index = (104729 * global_index) % count
            segment = row["segment_unmapped"]
            if segment not in {"0", "1"}:
                raise M1Error("interface segment_unmapped is not binary")
            cells.append(
                Cell(
                    identifier=row["provider_neuron_id"],
                    group=group,
                    layer=row["source_acronym"],
                    hemisphere=row["provider_hemisphere"],
                    soma_um=(
                        float(row["soma_x_ccfv3_um"]),
                        float(row["soma_y_ccfv3_um"]),
                        float(row["soma_z_ccfv3_um"]),
                    ),
                    soma_distance_mm=distances[row["provider_neuron_id"]],
                    k=k,
                    observed_index=observed_index,
                    segment_unmapped=int(segment),
                )
            )
            global_index += 1
    if len(cells) != 194:
        raise M1Error("target-free real-shape construction did not produce 194 rows")
    return tuple(cells), pairs


def clone_with_k(cells: Sequence[Cell], k: int) -> tuple[Cell, ...]:
    count = math.comb(16, int(k))
    result: list[Cell] = []
    for index, cell in enumerate(cells):
        result.append(
            Cell(
                identifier=cell.identifier,
                group=cell.group,
                layer=cell.layer,
                hemisphere=cell.hemisphere,
                soma_um=cell.soma_um,
                soma_distance_mm=cell.soma_distance_mm,
                k=int(k),
                observed_index=(104729 * index) % count,
                segment_unmapped=cell.segment_unmapped,
            )
        )
    return tuple(result)


def _all_training_folds(cells: Sequence[Cell]) -> list[tuple[str, tuple[int, ...]]]:
    groups = [cell.group for cell in cells]
    folds: list[tuple[str, tuple[int, ...]]] = [
        ("full", tuple(range(len(cells))))
    ]
    for group in GROUP_ORDER:
        folds.append(
            (
                f"leave_one__{group}",
                tuple(i for i, value in enumerate(groups) if value != group),
            )
        )
    for left_index, left in enumerate(GROUP_ORDER[:-1]):
        for right in GROUP_ORDER[left_index + 1 :]:
            folds.append(
                (
                    f"leave_two__{left}__{right}",
                    tuple(
                        i
                        for i, value in enumerate(groups)
                        if value not in {left, right}
                    ),
                )
            )
    if len(folds) != 79:
        raise M1Error("target-free real-shape fold count changed")
    return folds


def _scalar_set_features(
    cell: Cell, selected: Sequence[int], pairs: PairGeometry
) -> np.ndarray:
    selected = tuple(int(value) for value in selected)
    if not selected:
        return np.zeros(6, dtype=float)
    distances = np.asarray([cell.soma_distance_mm[value] for value in selected])
    mean = float(np.mean(distances))
    sd = float(np.sqrt(np.mean((distances - mean) ** 2))) if len(selected) > 1 else 0.0
    if len(selected) == 1:
        return np.asarray((mean, 0.0, 0.0, 0.0, 0.0, 0.0))
    pair_values = [(left, right) for left, right in itertools.combinations(selected, 2)]
    mean_pair = float(np.mean([pairs.distance[left, right] for left, right in pair_values]))
    cross = float(np.mean([pairs.cross_side[left, right] for left, right in pair_values]))
    homotopic = float(
        np.mean([pairs.homotopic[left, right] for left, right in pair_values])
    )
    parent = {value: value for value in selected}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    accepted = 0
    total = 0.0
    for distance, left, right in pairs.ordered_edges:
        if left not in parent or right not in parent:
            continue
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        accepted += 1
        total += distance
        if accepted == len(selected) - 1:
            break
    if accepted != len(selected) - 1:
        raise M1Error("scalar MST reference failed")
    return np.asarray(
        (mean, sd, mean_pair, cross, homotopic, total / (len(selected) - 1)),
        dtype=float,
    )


def _enumeration_and_geometry_gate() -> dict[str, Any]:
    library = CandidateLibrary()
    counts = library.all_counts()
    expected = {k: math.comb(16, k) for k in range(17)}
    if counts != expected or sum(counts.values()) != 65536:
        raise AssertionError("fixed-K candidate counts changed")
    independent: dict[int, set[int]] = {k: set() for k in range(17)}
    for mask in range(1 << 16):
        independent[mask.bit_count()].add(mask)
    for k in range(17):
        basis = library.basis(k)
        observed_order = tuple(
            tuple(int(value) for value in np.flatnonzero(row))
            for row in basis.membership
        )
        expected_order = tuple(itertools.combinations(range(16), k))
        if observed_order != expected_order:
            raise AssertionError(f"candidate order is not lexicographic at K={k}")
        observed = {
            sum(int(value) << index for index, value in enumerate(row))
            for row in basis.membership
        }
        if observed != independent[k]:
            raise AssertionError(f"candidate membership mismatch at K={k}")

    rng = np.random.Generator(np.random.PCG64(1061))
    maximum_logz_error = 0.0
    for k in range(17):
        eta = rng.normal(size=16)
        basis = library.basis(k)
        scores = basis.membership @ eta
        maximum = float(np.max(scores))
        logz = maximum + math.log(float(np.sum(np.exp(scores - maximum))))
        brute_scores = []
        for mask in sorted(independent[k]):
            brute_scores.append(
                sum(eta[index] for index in range(16) if mask & (1 << index))
            )
        brute = np.asarray(brute_scores)
        brute_maximum = float(np.max(brute))
        brute_logz = brute_maximum + math.log(
            float(np.sum(np.exp(brute - brute_maximum)))
        )
        maximum_logz_error = max(maximum_logz_error, abs(logz - brute_logz))
    if maximum_logz_error > 1.0e-12:
        raise AssertionError("stable exhaustive log normalizer disagrees with brute force")

    pairs = pair_geometry_from_coordinates(synthetic_target_coordinates())
    geometry = GeometryEngine(library, pairs)
    normal = np.asarray((0.2, -0.3, 0.5))
    cell = _synthetic_cell(0, "scalar", 1, normal, 0.5)
    maximum_feature_error = 0.0
    for k in (0, 1, 2, 3, 8, 16):
        candidate_cell = Cell(
            cell.identifier,
            cell.group,
            cell.layer,
            cell.hemisphere,
            cell.soma_um,
            cell.soma_distance_mm,
            k,
            0,
            cell.segment_unmapped,
        )
        basis = library.basis(k)
        test_indexes = sorted(set((0, basis.count // 2, basis.count - 1)))
        for index in test_indexes:
            observed = geometry.raw_batch(candidate_cell, index, index + 1)[0]
            selected = tuple(np.flatnonzero(basis.membership[index]))
            expected_feature = _scalar_set_features(candidate_cell, selected, pairs)
            maximum_feature_error = max(
                maximum_feature_error,
                float(np.max(np.abs(observed - expected_feature))),
            )
    if maximum_feature_error > 1.0e-12:
        raise AssertionError("vector set geometry disagrees with scalar materializer formula")
    return {
        "status": "pass",
        "total_panel_subsets": 65536,
        "lexicographic_candidate_order": True,
        "maximum_logz_absolute_error": maximum_logz_error,
        "maximum_scalar_geometry_absolute_error": maximum_feature_error,
    }


def _load_frozen_m0_v7_source() -> Any:
    """Load the frozen M0 V7 source solely as a gamma=0 test oracle.

    This fixed repo-local source path cannot address an M0 fit or prediction
    artifact.  None of its DP, 255D Hessian, solver, or certificate functions
    are called by M1 likelihood or optimization code.
    """
    global _FROZEN_M0_V7_MODULE
    if _FROZEN_M0_V7_MODULE is None:
        path = M0_V7_ENGINE_PATH
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise M1Error("frozen M0 V7 numerical-reference source is unavailable")
        name = "ep10_v3_m0_v7_gamma_zero_reference_for_m1_gate"
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            raise M1Error("cannot load frozen M0 V7 numerical-reference source")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        _FROZEN_M0_V7_MODULE = module
    return _FROZEN_M0_V7_MODULE


def _m0_reference(
    theta_b: np.ndarray,
    prepared: PreparedData,
    library: CandidateLibrary,
    ridge_lambda: float,
) -> dict[str, Any]:
    """Evaluate the paired frozen M0 DP/V6 source as an independent oracle."""
    reference = _load_frozen_m0_v7_source()
    b_vector = np.asarray(theta_b, dtype=float)
    if b_vector.shape != (B_PARAMETER_DIMENSION,):
        raise M1Error("M0 parity vector has the wrong dimension")
    observed = np.vstack(
        [
            library.basis(cell.k).membership[cell.observed_index]
            for cell in prepared.cells
        ]
    ).astype(float)
    k_values = np.asarray([cell.k for cell in prepared.cells], dtype=int)
    groups = [cell.group for cell in prepared.cells]
    objective, gradient = reference.base.objective_and_gradient(
        b_vector,
        prepared.x,
        observed,
        k_values,
        groups,
        TARGET_CONTRASTS,
        ridge_lambda,
    )
    coefficients = b_vector.reshape(DESIGN_DIMENSION, TARGET_CONTRAST_DIMENSION)
    eta = (prepared.x @ coefficients) @ TARGET_CONTRASTS.T
    logz, marginals = reference.base.batch_fixed_k_logz_marginals(eta, k_values)
    observed_score = np.sum(eta * observed, axis=1)
    logs = observed_score - logz
    hessian, hessian_diagnostics = reference.v6._exact_hessian_v6(
        b_vector,
        prepared.x,
        k_values,
        groups,
        TARGET_CONTRASTS,
        ridge_lambda,
    )
    score_error = 0.0
    probabilities: list[np.ndarray] = []
    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        direct_scores = basis.membership @ eta[cell_index]
        contrast_scores = basis.contrasts @ (
            coefficients.T @ prepared.x[cell_index]
        )
        score_error = max(
            score_error,
            float(np.max(np.abs(direct_scores - contrast_scores))),
        )
        probabilities.append(np.exp(direct_scores - logz[cell_index]))
    return {
        "objective": float(objective),
        "gradient": np.asarray(gradient, dtype=float),
        "hessian": np.asarray(hessian, dtype=float),
        "log_probability": np.asarray(logs, dtype=float),
        "marginals": np.asarray(marginals, dtype=float),
        "candidate_probabilities": tuple(probabilities),
        "candidate_score_error": score_error,
        "Hessian_diagnostics": dict(hessian_diagnostics),
        "source": str(M0_V7_ENGINE_PATH.relative_to(STAGE0_DIR.parent.parent)),
    }


def _gamma_zero_and_scaling_gate() -> dict[str, Any]:
    cells, moments, prepared, library, geometry, _ = generate_synthetic_dataset(
        groups=2, rows_per_group=17, seed=20260927
    )
    b = np.linspace(-0.08, 0.07, 255, dtype=float)
    theta = np.concatenate((b, np.zeros(6)))
    ridge = 0.1
    m1 = evaluate_model(
        theta,
        prepared,
        library,
        geometry,
        ridge,
        need_hessian=True,
        need_predictions=True,
    )
    reference = _m0_reference(b, prepared, library, ridge)
    assert m1.hessian is not None
    assert m1.log_probability_nats is not None and m1.marginals is not None
    probability_error = 0.0
    for cell_index, cell in enumerate(prepared.cells):
        basis = library.basis(cell.k)
        scores = np.concatenate(
            [
                _candidate_score_batch(
                    theta,
                    prepared,
                    cell_index,
                    start,
                    min(start + BATCH_MAXIMUM, basis.count),
                    library,
                    geometry,
                )
                for start in range(0, basis.count, BATCH_MAXIMUM)
            ]
        )
        maximum = float(np.max(scores))
        candidate_weights = np.exp(scores - maximum)
        probabilities = candidate_weights / float(np.sum(candidate_weights))
        probability_error = max(
            probability_error,
            float(
                np.max(
                    np.abs(
                        probabilities
                        - reference["candidate_probabilities"][cell_index]
                    )
                )
            ),
        )
    errors = {
        "score": reference["candidate_score_error"],
        "probability": probability_error,
        "NLL": abs(m1.objective - reference["objective"]),
        "B_gradient": float(
            np.max(np.abs(m1.gradient[:255] - reference["gradient"]))
        ),
        "B_Hessian": float(
            np.max(np.abs(m1.hessian[:255, :255] - reference["hessian"]))
        ),
        "log_probability": float(
            np.max(
                np.abs(m1.log_probability_nats - reference["log_probability"])
            )
        ),
        "marginal": float(
            np.max(np.abs(m1.marginals - reference["marginals"]))
        ),
    }
    if not (
        errors["score"] <= 1.0e-12
        and errors["probability"] <= 1.0e-10
        and errors["NLL"] <= 1.0e-10
        and errors["log_probability"] <= 1.0e-10
        and errors["marginal"] <= 1.0e-10
        and errors["B_gradient"] <= 1.0e-9
        and errors["B_Hessian"] <= 1.0e-8
    ):
        raise AssertionError(f"gamma-zero M0 parity failed: {errors}")
    if float(np.max(np.abs(m1.gradient[255:]))) == 0.0:
        raise AssertionError("gamma-gradient test accidentally required zero")

    training = tuple(index for index, cell in enumerate(cells) if cell.group == cells[0].group)
    original = prepare_data(cells, moments, training)
    changed = list(moments)
    validation_index = next(index for index, cell in enumerate(cells) if cell.group != cells[0].group)
    changed[validation_index] = UniformCellMoments(
        geometry_mean=np.full(6, 1.0e12),
        raw_joint_covariance=np.eye(21) * 1.0e24,
    )
    repeated = prepare_data(cells, changed, training)
    if not (
        np.array_equal(original.design_transform.soma_mean, repeated.design_transform.soma_mean)
        and np.array_equal(original.design_transform.soma_scale, repeated.design_transform.soma_scale)
        and np.array_equal(original.geometry_scales, repeated.geometry_scales)
    ):
        raise AssertionError("validation rows leaked into training transforms/scales")
    validation_cells = tuple(
        cells[index] for index in range(len(cells)) if index not in training
    )
    validation_moments = tuple(
        moments[index] for index in range(len(cells)) if index not in training
    )
    validation = apply_fixed_transforms(
        validation_cells,
        validation_moments,
        original.design_transform,
        original.geometry_scales,
    )
    expected_centers = np.asarray(
        [moment.geometry_mean for moment in validation_moments]
    )
    if not (
        np.array_equal(validation.geometry_scales, original.geometry_scales)
        and np.array_equal(validation.geometry_centers, expected_centers)
    ):
        raise AssertionError("validation cell did not retain its own Omega center")
    return {
        "status": "pass",
        "maximum_absolute_errors": errors,
        "paired_frozen_M0_source": reference["source"],
        "M0_prediction_or_fit_artifact_opened": False,
        "validation_cell_own_Omega_center_with_training_scales": True,
    }


def _derivative_and_singleton_gate() -> dict[str, Any]:
    cells, moments, _, library, geometry, _ = generate_synthetic_dataset(
        groups=2, rows_per_group=17, seed=20260929
    )
    indexes = tuple(index for index, cell in enumerate(cells) if cell.k in (2, 3))[:4]
    prepared = prepare_data(cells, moments, indexes)
    theta = 0.025 * np.sin(np.arange(PARAMETER_DIMENSION, dtype=float) + 0.3)
    ridge = 0.1
    analytic = evaluate_model(
        theta, prepared, library, geometry, ridge, need_hessian=True
    )
    assert analytic.hessian is not None
    step = 1.0e-6
    finite_gradient = np.zeros(PARAMETER_DIMENSION)
    for index in range(PARAMETER_DIMENSION):
        offset = np.zeros(PARAMETER_DIMENSION)
        offset[index] = step
        plus = evaluate_model(
            theta + offset, prepared, library, geometry, ridge, need_hessian=False
        ).objective
        minus = evaluate_model(
            theta - offset, prepared, library, geometry, ridge, need_hessian=False
        ).objective
        finite_gradient[index] = (plus - minus) / (2.0 * step)
    gradient_error = np.abs(finite_gradient - analytic.gradient)
    gradient_absolute = float(np.max(gradient_error))
    gradient_relative = float(
        np.max(gradient_error / np.maximum(1.0, np.abs(finite_gradient)))
    )
    rng = np.random.Generator(np.random.PCG64(4451))
    hv_absolute = 0.0
    hv_relative = 0.0
    hv_step = 1.0e-5
    for _ in range(8):
        direction = rng.normal(size=PARAMETER_DIMENSION)
        direction /= np.linalg.norm(direction)
        plus = evaluate_model(
            theta + hv_step * direction,
            prepared,
            library,
            geometry,
            ridge,
            need_hessian=False,
        ).gradient
        minus = evaluate_model(
            theta - hv_step * direction,
            prepared,
            library,
            geometry,
            ridge,
            need_hessian=False,
        ).gradient
        observed = (plus - minus) / (2.0 * hv_step)
        expected = analytic.hessian @ direction
        error = np.abs(observed - expected)
        hv_absolute = max(hv_absolute, float(np.max(error)))
        hv_relative = max(
            hv_relative,
            float(np.max(error / np.maximum(1.0, np.abs(expected)))),
        )
    if analytic.raw_hessian_symmetry_error is None:
        raise AssertionError("analytic Hessian omitted raw symmetry telemetry")
    raw_symmetry = analytic.raw_hessian_symmetry_error
    symmetry = float(np.max(np.abs(analytic.hessian - analytic.hessian.T)))
    eigenvalues = np.linalg.eigvalsh(analytic.hessian)
    spectral = float(np.max(np.abs(eigenvalues)))
    lower_pass = float(eigenvalues[0]) >= ridge - 1.0e-10 * max(1.0, spectral)
    if not (
        gradient_absolute <= 1.0e-6
        or gradient_relative <= 1.0e-5
    ):
        raise AssertionError("all-coordinate analytic gradient check failed")
    if not (hv_absolute <= 1.0e-5 or hv_relative <= 1.0e-4):
        raise AssertionError("analytic Hessian-vector check failed")
    if raw_symmetry > 1.0e-12 or symmetry > 1.0e-12 or not lower_pass:
        raise AssertionError("analytic Hessian symmetry/lower-bound check failed")

    base_cell = cells[0]
    singleton_cells = (
        Cell(
            "k0",
            "singleton0",
            base_cell.layer,
            base_cell.hemisphere,
            base_cell.soma_um,
            base_cell.soma_distance_mm,
            0,
            0,
            base_cell.segment_unmapped,
        ),
        Cell(
            "k16",
            "singleton16",
            cells[1].layer,
            cells[1].hemisphere,
            cells[1].soma_um,
            cells[1].soma_distance_mm,
            16,
            0,
            cells[1].segment_unmapped,
        ),
    )
    singleton_prepared = PreparedData(
        singleton_cells,
        np.vstack((prepared.x[0], prepared.x[1])),
        np.zeros((2, 6)),
        np.ones(6),
        np.asarray((0.5, 0.5)),
        prepared.design_transform,
    )
    singleton_theta = np.linspace(-0.1, 0.1, PARAMETER_DIMENSION)
    singleton = evaluate_model(
        singleton_theta,
        singleton_prepared,
        library,
        geometry,
        ridge,
        need_hessian=True,
        need_predictions=True,
    )
    assert singleton.hessian is not None
    expected_objective = 0.5 * ridge * float(np.dot(singleton_theta, singleton_theta))
    if not (
        singleton.objective == expected_objective
        and np.array_equal(singleton.gradient, ridge * singleton_theta)
        and np.array_equal(singleton.hessian, ridge * np.eye(PARAMETER_DIMENSION))
        and np.array_equal(singleton.log_probability_nats, np.zeros(2))
        and singleton.candidate_visits == 4
    ):
        raise AssertionError("K=0/16 singleton data derivatives are not exact zero")
    return {
        "status": "pass",
        "gradient_max_absolute_error": gradient_absolute,
        "gradient_max_relative_error": gradient_relative,
        "Hessian_vector_max_absolute_error": hv_absolute,
        "Hessian_vector_max_relative_error": hv_relative,
        "raw_Hessian_symmetry_error": raw_symmetry,
        "Hessian_symmetry_error": symmetry,
        "Hessian_minimum_eigenvalue": float(eigenvalues[0]),
    }


def _certificate_gate() -> dict[str, Any]:
    failure = LineSearchFailure(
        "strong-Wolfe zoom iteration ceiling reached",
        evaluations=61,
        expansions=0,
        zoom_iterations=60,
        all_values_finite=True,
    )
    gradient = np.full(PARAMETER_DIMENSION, 1.0e-9)
    gradient_before = gradient.copy()
    passed = _certificate(failure, 1.0, gradient, 1.0e-4, math.inf)
    resolution = _certificate(
        LineSearchFailure(
            "strong-Wolfe zoom bracket reached floating-point resolution",
            evaluations=7,
            expansions=0,
            zoom_iterations=7,
            all_values_finite=True,
        ),
        1.0,
        gradient,
        1.0e-4,
        math.inf,
    )
    ineligible = _certificate(
        LineSearchFailure(
            "strong-Wolfe maximum step reached without a bracket",
            evaluations=4,
            expansions=3,
            zoom_iterations=0,
            all_values_finite=True,
        ),
        1.0,
        gradient,
        1.0e-4,
        math.inf,
    )
    nonfinite = _certificate(
        LineSearchFailure(
            "strong-Wolfe zoom iteration ceiling reached",
            evaluations=61,
            expansions=0,
            zoom_iterations=60,
            all_values_finite=False,
        ),
        1.0,
        np.full(PARAMETER_DIMENSION, math.nan),
        1.0e-4,
        math.inf,
    )
    forbidden_state_keys = {"theta", "vector", "history", "accepted_step"}
    if not (
        passed["certificate_pass"]
        and resolution["certificate_pass"]
        and not ineligible["certificate_pass"]
        and not nonfinite["certificate_pass"]
        and np.array_equal(gradient, gradient_before)
        and forbidden_state_keys.isdisjoint(passed)
    ):
        raise AssertionError("261D finite-precision certificate gate failed")
    return {
        "status": "pass",
        "eligible_iteration_ceiling_case": passed,
        "eligible_resolution_case": resolution,
        "ineligible_pass": False,
        "nonfinite_pass": False,
        "state_or_history_mutated": False,
    }


def _synthetic_rng_order_gate() -> dict[str, Any]:
    expected = {
        (20260927, 544): (
            np.asarray(
                (-2.0542238738174565, -0.4544645348011356, -0.4182984531044877)
            ),
            np.asarray(
                (0.1090497505936872, -1.2297931087905976, 0.6729160032668773)
            ),
            0.6255677395905063,
            0.23424668592609876,
        ),
        (20260928, 68): (
            np.asarray(
                (-0.45297224402656694, -0.7310062017908747, 1.380937885873302)
            ),
            np.asarray(
                (-0.6798267082570127, 0.23883856517522217, 0.5151267347346372)
            ),
            0.7575273456436823,
            0.7631273560567471,
        ),
    }
    rows: list[dict[str, Any]] = []
    for (seed, row_count), frozen in expected.items():
        rng = np.random.Generator(np.random.PCG64(seed))
        normals = _draw_synthetic_normal_block(rng, row_count)
        uniforms = np.asarray([rng.random() for _ in range(row_count)])
        if not (
            np.array_equal(normals[0], frozen[0])
            and np.array_equal(normals[-1], frozen[1])
            and float(uniforms[0]) == frozen[2]
            and float(uniforms[-1]) == frozen[3]
        ):
            raise AssertionError(f"synthetic PCG64 draw order changed for seed {seed}")
        rows.append(
            {
                "seed": seed,
                "rows": row_count,
                "normal_draws_before_first_uniform": 3 * row_count,
                "categorical_uniforms": row_count,
            }
        )
    return {
        "status": "pass",
        "normal_block_precedes_categorical_uniform_block": True,
        "schedules": rows,
    }


def _population_objective_conformance_gate() -> dict[str, Any]:
    """Exercise the exact-population linear correction independently of fit."""
    cells, moments, prepared, library, geometry, planted = generate_synthetic_dataset(
        groups=2, rows_per_group=17, seed=20260929
    )
    expected_all = expected_cell_joint_statistics(
        planted, prepared, library, geometry
    )
    observed_all = observed_cell_joint_statistics(prepared, library, geometry)
    one_hot = aggregate_population_linear_correction(
        prepared, observed_all, observed_all
    )
    if not np.array_equal(
        one_hot.vector, np.zeros(PARAMETER_DIMENSION, dtype=float)
    ):
        raise AssertionError("one-hot population correction is not exact zero")

    trial = 0.02 * np.cos(np.arange(PARAMETER_DIMENSION, dtype=float) + 0.17)
    ridge = 0.1
    sampled = evaluate_model(
        trial, prepared, library, geometry, ridge, need_hessian=True
    )
    one_hot_population = evaluate_model(
        trial,
        prepared,
        library,
        geometry,
        ridge,
        need_hessian=True,
        linear_correction=one_hot.vector,
    )
    assert sampled.hessian is not None and one_hot_population.hessian is not None
    one_hot_errors = {
        "objective": abs(sampled.objective - one_hot_population.objective),
        "gradient": float(
            np.max(np.abs(sampled.gradient - one_hot_population.gradient))
        ),
        "Hessian": float(
            np.max(np.abs(sampled.hessian - one_hot_population.hessian))
        ),
    }
    if any(value != 0.0 for value in one_hot_errors.values()):
        raise AssertionError("one-hot population evaluator did not reproduce sampled math")

    marginal_cardinality_error = float(
        np.max(
            np.abs(
                np.sum(expected_all.marginals, axis=1)
                - np.asarray([cell.k for cell in prepared.cells], dtype=float)
            )
        )
    )
    if (
        sorted({cell.k for cell in prepared.cells}) != list(range(17))
        or expected_all.probability_sum_maximum_absolute_error > 1.0e-12
        or marginal_cardinality_error > 1.0e-10
    ):
        raise AssertionError("population probabilities failed exhaustive K=0..16 checks")

    indexes = tuple(
        index for index, cell in enumerate(cells) if cell.k in (2, 3)
    )[:4]
    derivative_prepared = prepare_data(cells, moments, indexes)
    derivative_expected = expected_cell_joint_statistics(
        planted, derivative_prepared, library, geometry
    )
    derivative_observed = observed_cell_joint_statistics(
        derivative_prepared, library, geometry
    )
    correction = aggregate_population_linear_correction(
        derivative_prepared,
        derivative_expected.joint_mean,
        derivative_observed,
    )
    analytic = evaluate_model(
        trial,
        derivative_prepared,
        library,
        geometry,
        ridge,
        need_hessian=True,
        linear_correction=correction.vector,
    )
    assert analytic.hessian is not None
    step = 1.0e-6
    finite_gradient = np.zeros(PARAMETER_DIMENSION, dtype=float)
    for coordinate in range(PARAMETER_DIMENSION):
        offset = np.zeros(PARAMETER_DIMENSION, dtype=float)
        offset[coordinate] = step
        plus = evaluate_model(
            trial + offset,
            derivative_prepared,
            library,
            geometry,
            ridge,
            need_hessian=False,
            linear_correction=correction.vector,
        ).objective
        minus = evaluate_model(
            trial - offset,
            derivative_prepared,
            library,
            geometry,
            ridge,
            need_hessian=False,
            linear_correction=correction.vector,
        ).objective
        finite_gradient[coordinate] = (plus - minus) / (2.0 * step)
    gradient_error = np.abs(finite_gradient - analytic.gradient)
    gradient_absolute = float(np.max(gradient_error))
    gradient_relative = float(
        np.max(gradient_error / np.maximum(1.0, np.abs(finite_gradient)))
    )

    rng = np.random.Generator(np.random.PCG64(4452))
    hv_absolute = 0.0
    hv_relative = 0.0
    hv_step = 1.0e-5
    for _ in range(8):
        direction = rng.normal(size=PARAMETER_DIMENSION)
        direction /= np.linalg.norm(direction)
        plus = evaluate_model(
            trial + hv_step * direction,
            derivative_prepared,
            library,
            geometry,
            ridge,
            need_hessian=False,
            linear_correction=correction.vector,
        ).gradient
        minus = evaluate_model(
            trial - hv_step * direction,
            derivative_prepared,
            library,
            geometry,
            ridge,
            need_hessian=False,
            linear_correction=correction.vector,
        ).gradient
        observed_hv = (plus - minus) / (2.0 * hv_step)
        expected_hv = analytic.hessian @ direction
        error = np.abs(observed_hv - expected_hv)
        hv_absolute = max(hv_absolute, float(np.max(error)))
        hv_relative = max(
            hv_relative,
            float(np.max(error / np.maximum(1.0, np.abs(expected_hv)))),
        )
    if not (gradient_absolute <= 1.0e-6 or gradient_relative <= 1.0e-5):
        raise AssertionError("population all-coordinate gradient check failed")
    if not (hv_absolute <= 1.0e-5 or hv_relative <= 1.0e-4):
        raise AssertionError("population Hessian-vector check failed")

    planted_unpenalized = evaluate_model(
        planted,
        derivative_prepared,
        library,
        geometry,
        0.0,
        need_hessian=False,
        linear_correction=correction.vector,
    )
    planted_penalized = evaluate_model(
        planted,
        derivative_prepared,
        library,
        geometry,
        RECOVERY_LAMBDA,
        need_hessian=False,
        linear_correction=correction.vector,
    )
    unpenalized_gradient = float(
        np.max(np.abs(planted_unpenalized.gradient))
    )
    ridge_gradient_error = float(
        np.max(
            np.abs(
                planted_penalized.gradient - RECOVERY_LAMBDA * planted
            )
        )
    )
    if (
        unpenalized_gradient > GRADIENT_TOLERANCE
        or ridge_gradient_error > GRADIENT_TOLERANCE
    ):
        raise AssertionError("population planted/ridge gradient conformance failed")
    return {
        "status": "pass",
        "one_hot_sampled_parity_maximum_absolute_errors": one_hot_errors,
        "all_K_0_through_16_normalized": True,
        "probability_sum_maximum_absolute_error": (
            expected_all.probability_sum_maximum_absolute_error
        ),
        "marginal_cardinality_maximum_absolute_error": (
            marginal_cardinality_error
        ),
        "gradient_maximum_absolute_error": gradient_absolute,
        "gradient_maximum_relative_error": gradient_relative,
        "Hessian_vector_maximum_absolute_error": hv_absolute,
        "Hessian_vector_maximum_relative_error": hv_relative,
        "planted_unpenalized_gradient_infinity_norm": unpenalized_gradient,
        "planted_penalized_ridge_gradient_maximum_absolute_error": (
            ridge_gradient_error
        ),
        "candidate_by_261_materialized": False,
    }


def run_unit_gates() -> dict[str, Any]:
    started = time.perf_counter()
    result = {
        "status": "pass",
        "stage": "v3_M1_target_free_unit_gates",
        "enumeration_and_geometry": _enumeration_and_geometry_gate(),
        "synthetic_design_addendum": _synthetic_design_rank_gate(),
        "gamma_zero_and_fold_scaling": _gamma_zero_and_scaling_gate(),
        "derivatives_and_singletons": _derivative_and_singleton_gate(),
        "strong_convexity_certificate": _certificate_gate(),
        "synthetic_RNG_order": _synthetic_rng_order_gate(),
        "exact_population_objective": _population_objective_conformance_gate(),
        "candidate_by_261_materialized": False,
        "M0_DP_or_V6_pair_moments_used_as_M1_math": False,
        "final_or_unassigned_or_SWC_opened": False,
    }
    result["elapsed_wall_seconds"] = time.perf_counter() - started
    return result


def _gamma_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise M1Error("gamma cosine denominator is zero or nonfinite")
    value = float(np.dot(left_vector, right_vector) / denominator)
    if not math.isfinite(value):
        raise M1Error("gamma cosine is nonfinite")
    return value


def _synthetic_population_recovery_gate(
    cells: Sequence[Cell],
    prepared: PreparedData,
    library: CandidateLibrary,
    geometry: GeometryEngine,
    planted: np.ndarray,
) -> dict[str, Any]:
    """Run retained one-hot diagnostics and exact-population hard recovery."""
    unique_states = sum(library.basis(cell.k).count for cell in cells)
    group_counts = Counter(cell.group for cell in cells)
    if (
        len(cells) != 544
        or len(group_counts) != 16
        or set(group_counts.values()) != {34}
        or unique_states != 2097152
    ):
        raise M1Error("544-row recovery shape or candidate-state count changed")

    sampled_started = time.perf_counter()
    sampled_fit = fit_newton(
        prepared, library, geometry, RECOVERY_LAMBDA, initial=None
    )
    sampled_elapsed = time.perf_counter() - sampled_started
    sampled_comparison = compare_conditional_distributions(
        planted, sampled_fit.theta, prepared, library, geometry
    )
    sampled_cosine = _gamma_cosine_similarity(
        planted[-6:], sampled_fit.theta[-6:]
    )
    expected_first_order_kl_bits = (
        PARAMETER_DIMENSION / (2.0 * len(cells) * LN2)
    )
    sampled_payload: dict[str, Any] = {
        "status": "complete_descriptive",
        "role": "descriptive_finite_sample_diagnostic_only",
        "hard_population_thresholds_applied": False,
        "thresholds_not_applied": True,
        "rows": len(cells),
        "groups": len(group_counts),
        "one_hot_draws_per_row": 1,
        "fit_elapsed_wall_seconds": sampled_elapsed,
        **sampled_comparison,
        "expected_first_order_KL_bits": expected_first_order_kl_bits,
        "observed_to_expected_first_order_KL_ratio": (
            sampled_comparison["mean_conditional_KL_bits"]
            / expected_first_order_kl_bits
        ),
        "planted_recovered_gamma_cosine_similarity": sampled_cosine,
        "optimizer": dict(sampled_fit.telemetry),
    }

    try:
        expectation = expected_cell_joint_statistics(
            planted, prepared, library, geometry
        )
        observed = observed_cell_joint_statistics(prepared, library, geometry)
        correction = aggregate_population_linear_correction(
            prepared, expectation.joint_mean, observed
        )
        sampled_score_at_planted = evaluate_model(
            planted,
            prepared,
            library,
            geometry,
            0.0,
            need_hessian=False,
        )
        correction_identity_error = float(
            np.max(
                np.abs(
                    correction.vector + sampled_score_at_planted.gradient
                )
            )
        )
        planted_unpenalized = evaluate_model(
            planted,
            prepared,
            library,
            geometry,
            0.0,
            need_hessian=False,
            linear_correction=correction.vector,
        )
        planted_penalized = evaluate_model(
            planted,
            prepared,
            library,
            geometry,
            RECOVERY_LAMBDA,
            need_hessian=False,
            linear_correction=correction.vector,
        )
        unpenalized_gradient = float(
            np.max(np.abs(planted_unpenalized.gradient))
        )
        ridge_gradient_error = float(
            np.max(
                np.abs(
                    planted_penalized.gradient - RECOVERY_LAMBDA * planted
                )
            )
        )
        marginal_cardinality_error = float(
            np.max(
                np.abs(
                    np.sum(expectation.marginals, axis=1)
                    - np.asarray([cell.k for cell in cells], dtype=float)
                )
            )
        )

        population_started = time.perf_counter()
        population_fit = fit_newton(
            prepared,
            library,
            geometry,
            RECOVERY_LAMBDA,
            initial=None,
            linear_correction=correction.vector,
        )
        population_elapsed = time.perf_counter() - population_started
        population_comparison = compare_conditional_distributions(
            planted, population_fit.theta, prepared, library, geometry
        )
        population_cosine = _gamma_cosine_similarity(
            planted[-6:], population_fit.theta[-6:]
        )
        exact_kl_nats = (
            population_comparison["mean_conditional_KL_bits"] * LN2
        )

        theta_norm_squared = float(np.dot(planted, planted))
        planted_ridge_penalty = (
            0.5 * RECOVERY_LAMBDA * theta_norm_squared
        )
        solver_gap_bound = (
            PARAMETER_DIMENSION * GRADIENT_TOLERANCE**2
            / (2.0 * RECOVERY_LAMBDA)
        )
        raw_bound = planted_ridge_penalty + solver_gap_bound
        derived_ceiling = 1.10 * raw_bound
        derived_pinsker = math.sqrt(
            POPULATION_KL_WITNESS_CEILING_NATS / 2.0
        )
        if not (
            abs(theta_norm_squared - 2.9275) <= 1.0e-14
            and abs(raw_bound - POPULATION_KL_RAW_BOUND_NATS) <= 1.0e-18
            and abs(
                derived_ceiling - POPULATION_KL_WITNESS_CEILING_NATS
            )
            <= 1.0e-18
            and abs(
                derived_pinsker - POPULATION_PINSKER_MARGINAL_BOUND
            )
            <= 1.0e-18
        ):
            raise M1Error("contract-derived population witness arithmetic changed")

        acceptance_checks = {
            "mean_conditional_KL_bits": {
                "observed": population_comparison[
                    "mean_conditional_KL_bits"
                ],
                "maximum": 0.01,
                "pass": population_comparison[
                    "mean_conditional_KL_bits"
                ]
                <= 0.01,
            },
            "exact_inclusion_marginal_MAE": {
                "observed": population_comparison[
                    "exact_inclusion_marginal_MAE"
                ],
                "maximum": 0.02,
                "pass": population_comparison[
                    "exact_inclusion_marginal_MAE"
                ]
                <= 0.02,
            },
            "planted_recovered_gamma_cosine_similarity": {
                "observed": population_cosine,
                "minimum": 0.90,
                "pass": population_cosine >= 0.90,
            },
        }
        implementation_checks = {
            "exact_population_KL_nats": exact_kl_nats,
            "ceiling_nats": POPULATION_KL_WITNESS_CEILING_NATS,
            "pass": exact_kl_nats <= POPULATION_KL_WITNESS_CEILING_NATS,
            "role": "implementation_consistency_not_scientific_threshold",
        }
        conformance_pass = bool(
            sorted({cell.k for cell in cells}) == list(range(17))
            and expectation.probability_sum_maximum_absolute_error <= 1.0e-12
            and marginal_cardinality_error <= 1.0e-10
            and correction_identity_error <= GRADIENT_TOLERANCE
            and unpenalized_gradient <= GRADIENT_TOLERANCE
            and ridge_gradient_error <= GRADIENT_TOLERANCE
        )
        passed = bool(
            conformance_pass
            and population_fit.telemetry.get("status") == "pass"
            and all(check["pass"] for check in acceptance_checks.values())
            and implementation_checks["pass"]
        )
        population_payload = {
            "status": "pass" if passed else "fail",
            "role": "hard_acceptance_exact_planted_population",
            "fit_elapsed_wall_seconds": population_elapsed,
            **population_comparison,
            "exact_population_KL_nats": exact_kl_nats,
            "planted_recovered_gamma_cosine_similarity": population_cosine,
            "acceptance_checks": acceptance_checks,
            "implementation_consistency_witness": {
                **implementation_checks,
                "theta_star_norm_squared": theta_norm_squared,
                "planted_ridge_penalty_nats": planted_ridge_penalty,
                "worst_case_solver_objective_gap_bound_nats": (
                    solver_gap_bound
                ),
                "raw_population_KL_upper_bound_nats": raw_bound,
                "pre_frozen_numerical_headroom_fraction": 0.10,
                "ceiling_bits": POPULATION_KL_WITNESS_CEILING_BITS,
                "Pinsker_with_headroom_mean_inclusion_marginal_MAE_upper_bound": (
                    derived_pinsker
                ),
                "Pinsker_role": "report_only_no_additional_MAE_gate",
            },
            "expected_sufficient_statistics": {
                "cell_joint_dimension": JOINT_CELL_DIMENSION,
                "expanded_parameter_dimension": PARAMETER_DIMENSION,
                "candidate_visits": expectation.candidate_visits,
                "elapsed_wall_seconds": expectation.elapsed_seconds,
                "probability_sum_maximum_absolute_error": (
                    expectation.probability_sum_maximum_absolute_error
                ),
                "marginal_cardinality_maximum_absolute_error": (
                    marginal_cardinality_error
                ),
                "aggregate_correction_L2_norm": float(
                    np.linalg.norm(correction.vector)
                ),
                "aggregate_correction_infinity_norm": float(
                    np.max(np.abs(correction.vector))
                ),
                "correction_equals_negative_sample_score_maximum_absolute_error": (
                    correction_identity_error
                ),
            },
            "planted_gradient_checks": {
                "unpenalized_gradient_infinity_norm": unpenalized_gradient,
                "penalized_gradient_minus_lambda_theta_maximum_absolute_error": (
                    ridge_gradient_error
                ),
                "tolerance": GRADIENT_TOLERANCE,
                "pass": conformance_pass,
            },
            "optimizer": dict(population_fit.telemetry),
        }
        return {
            "status": "pass" if passed else "fail",
            "rows": len(cells),
            "groups": len(group_counts),
            "unique_candidate_states_per_objective": unique_states,
            "score_computations_per_two_pass_objective": 2 * unique_states,
            "sampled_one_hot_recovery": sampled_payload,
            "exact_population_recovery": population_payload,
            "real_target_calls_or_real_K_opened": False,
            "M0_prediction_or_fit_artifact_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        }
    except (M1Error, AssertionError, ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        return {
            "status": "fail",
            "rows": len(cells),
            "groups": len(group_counts),
            "unique_candidate_states_per_objective": unique_states,
            "score_computations_per_two_pass_objective": 2 * unique_states,
            "sampled_one_hot_recovery": sampled_payload,
            "exact_population_recovery": {
                "status": "fail",
                "failure_class": type(error).__name__,
                "failure_message": str(error),
            },
            "real_target_calls_or_real_K_opened": False,
            "M0_prediction_or_fit_artifact_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        }


def run_synthetic_gates() -> dict[str, Any]:
    started = time.perf_counter()
    cells, moments, prepared, library, geometry, planted = generate_synthetic_dataset(
        groups=16, rows_per_group=34, seed=20260927
    )
    unique_states = sum(library.basis(cell.k).count for cell in cells)
    if unique_states != 2097152:
        raise M1Error("544-row recovery unique candidate-state count changed")
    rank = rank_condition_gate(
        cells,
        moments,
        [("full_544_row_recovery_training_set", tuple(range(len(cells))))],
    )
    recovery = _synthetic_population_recovery_gate(
        cells, prepared, library, geometry, planted
    )
    if recovery["status"] != "pass":
        return {
            "status": "fail",
            "stage": "v3_M1_target_free_synthetic_gates_v4",
            "failure_stage": "exact_population_recovery",
            "recovery": recovery,
            "rank_and_condition": rank,
            "real_target_calls_or_real_K_opened": False,
            "M0_prediction_or_fit_artifact_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
            "elapsed_wall_seconds": time.perf_counter() - started,
        }

    warm_cells, warm_moments, warm_prepared, warm_library, warm_geometry, _ = (
        generate_synthetic_dataset(groups=4, rows_per_group=17, seed=20260928)
    )
    warm_unique = sum(warm_library.basis(cell.k).count for cell in warm_cells)
    if warm_unique != 262144:
        raise M1Error("68-row warm/zero unique candidate-state count changed")
    first_path = fit_path(warm_prepared, warm_library, warm_geometry)
    second_path = fit_path(warm_prepared, warm_library, warm_geometry)
    warm_results: list[dict[str, Any]] = []
    for ridge_lambda in LAMBDAS_DESCENDING:
        first = first_path[ridge_lambda]
        second = second_path[ridge_lambda]
        zero = fit_newton(
            warm_prepared,
            warm_library,
            warm_geometry,
            ridge_lambda,
            initial=None,
        )
        repeat_distribution = compare_conditional_distributions(
            first.theta,
            second.theta,
            warm_prepared,
            warm_library,
            warm_geometry,
        )
        zero_distribution = compare_conditional_distributions(
            first.theta,
            zero.theta,
            warm_prepared,
            warm_library,
            warm_geometry,
        )
        repeat_objective = abs(first.objective - second.objective)
        repeat_coefficient = float(np.max(np.abs(first.theta - second.theta)))
        zero_objective = abs(first.objective - zero.objective)
        objective_tolerance = max(
            1.0e-10,
            1.1 * PARAMETER_DIMENSION * GRADIENT_TOLERANCE**2
            / (2.0 * ridge_lambda),
        )
        valid_witnesses = {
            "gradient_and_repeated_unchanged_initial_objective",
            "gradient_and_relative_objective_change",
            "strong_convexity_gap_after_zoom_stall",
        }
        passed = bool(
            repeat_objective <= 1.0e-12
            and repeat_coefficient <= 1.0e-10
            and repeat_distribution[
                "maximum_candidate_probability_absolute_difference"
            ]
            <= 1.0e-10
            and zero_objective <= objective_tolerance
            and zero_distribution[
                "maximum_candidate_probability_absolute_difference"
            ]
            <= 1.0e-5
            and zero_distribution[
                "maximum_candidate_log_probability_absolute_difference"
            ]
            <= 1.0e-4
            and all(
                result.telemetry.get("convergence_witness") in valid_witnesses
                for result in (first, second, zero)
            )
        )
        if not passed:
            raise M1Error(f"warm/zero gate failed at lambda={ridge_lambda:.12g}")
        warm_results.append(
            {
                "lambda": ridge_lambda,
                "status": "pass",
                "repeat_objective_absolute_difference": repeat_objective,
                "repeat_coefficient_max_absolute_difference": repeat_coefficient,
                "repeat_probability_max_absolute_difference": repeat_distribution[
                    "maximum_candidate_probability_absolute_difference"
                ],
                "zero_start_objective_absolute_difference": zero_objective,
                "zero_start_objective_absolute_tolerance": objective_tolerance,
                "zero_start_probability_max_absolute_difference": zero_distribution[
                    "maximum_candidate_probability_absolute_difference"
                ],
                "zero_start_log_probability_max_absolute_difference": zero_distribution[
                    "maximum_candidate_log_probability_absolute_difference"
                ],
                "warm_iterations": first.iterations,
                "zero_iterations": zero.iterations,
                "warm_witness": first.telemetry.get("convergence_witness"),
                "zero_witness": zero.telemetry.get("convergence_witness"),
            }
        )
    return {
        "status": "pass",
        "stage": "v3_M1_target_free_synthetic_gates_v4",
        "recovery": recovery,
        "rank_and_condition": rank,
        "warm_vs_zero": {
            "status": "pass",
            "rows": len(warm_cells),
            "groups": 4,
            "unique_candidate_states_per_objective": warm_unique,
            "score_computations_per_two_pass_objective": 2 * warm_unique,
            "lambdas": warm_results,
        },
        "real_target_calls_or_real_K_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
        "elapsed_wall_seconds": time.perf_counter() - started,
    }


def run_real_shape_benchmark(
    interface_root: Path, geometry_root: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    mixed_cells, pairs = load_target_free_real_shape(interface_root, geometry_root)
    library = CandidateLibrary()
    geometry = GeometryEngine(library, pairs)

    cache_started = time.perf_counter()
    mixed_moments = tuple(geometry.uniform_moments(cell) for cell in mixed_cells)
    mixed_unique_states = sum(library.basis(cell.k).count for cell in mixed_cells)
    feature_cache_seconds = time.perf_counter() - cache_started
    rank_folds = _all_training_folds(mixed_cells)
    mixed_rank = rank_condition_gate(mixed_cells, mixed_moments, rank_folds)
    mixed_rank["fold_role_coverage"] = {
        "unique_training_sets": 79,
        "full_development_unique_sets": 1,
        "leave_one_group_unique_sets": 12,
        "leave_two_groups_unique_sets": 66,
        "ordered_outer_inner_fold_uses_represented": 132,
        "leave_one_full_tuning_fold_uses_represented": 12,
        "leave_one_selected_refit_uses_represented": 12,
        "leave_one_training_set_uses_represented": 24,
        "full_development_training_fold_uses_represented": 1,
        "total_training_fold_uses_represented": 157,
    }

    worst_cells = clone_with_k(mixed_cells, 8)
    worst_cache_started = time.perf_counter()
    worst_moments = tuple(geometry.uniform_moments(cell) for cell in worst_cells)
    worst_prepared = prepare_data(worst_cells, worst_moments)
    worst_feature_seconds = time.perf_counter() - worst_cache_started
    worst_unique_states = sum(library.basis(cell.k).count for cell in worst_cells)
    if worst_unique_states != 2496780:
        raise M1Error("all-K8 benchmark unique candidate-state count changed")

    zero = np.zeros(PARAMETER_DIMENSION, dtype=float)
    ogh_started = time.perf_counter()
    first = evaluate_model(
        zero,
        worst_prepared,
        library,
        geometry,
        1.0e-4,
        need_hessian=True,
    )
    ogh_seconds = time.perf_counter() - ogh_started
    repeat = evaluate_model(
        zero,
        worst_prepared,
        library,
        geometry,
        1.0e-4,
        need_hessian=True,
    )
    if not (
        first.objective == repeat.objective
        and np.array_equal(first.gradient, repeat.gradient)
        and first.hessian is not None
        and repeat.hessian is not None
        and np.array_equal(first.hessian, repeat.hessian)
    ):
        raise M1Error("all-K8 objective/gradient/Hessian is not deterministic")

    factor_started = time.perf_counter()
    assert first.hessian is not None
    direction, solve = _newton_direction(first.hessian, first.gradient, 1.0e-4)
    factor_seconds = time.perf_counter() - factor_started
    if not np.all(np.isfinite(direction)):
        raise M1Error("all-K8 benchmark Newton direction is nonfinite")

    path_started = time.perf_counter()
    path = fit_path(worst_prepared, library, geometry)
    path_seconds = time.perf_counter() - path_started
    if any(result.telemetry.get("status") != "pass" for result in path.values()):
        raise M1Error("all-K8 complete warm path contains a failed fit")

    elapsed = time.perf_counter() - started
    peak_rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    candidate_rate = (
        first.candidate_visits / ogh_seconds if ogh_seconds > 0.0 else math.inf
    )
    # Budget qualification must not assume reuse of the 66 mathematically
    # identical unordered leave-two training sets.  Charge all 132 ordered
    # inner paths, 12 leave-one tuning paths, and each of 13 zero-start refits
    # as a complete path.  The 91-path exact-reuse case is report-only.
    uncached_path_equivalents = 157.0
    exact_reuse_path_equivalents = 91.0
    projected_uncached_wall_hours = (
        path_seconds * uncached_path_equivalents / 3600.0
    )
    projected_uncached_core_hours = 4.0 * projected_uncached_wall_hours
    projected_exact_reuse_wall_hours = (
        path_seconds * exact_reuse_path_equivalents / 3600.0
    )
    projected_exact_reuse_core_hours = 4.0 * projected_exact_reuse_wall_hours
    maximum_parallel_jobs = PARALLEL_CORE_CEILING // 4
    projected_parallel_requested_memory_gb = 16.0 * maximum_parallel_jobs
    # The production implementation writes only the four compact contracted
    # tabular/JSON artifacts and never spills candidates or Hessians to disk.
    # One GiB is a deliberately conservative incremental-output upper bound.
    projected_incremental_scratch_gb_upper_bound = 1.0
    computational_pass = bool(
        elapsed < 0.8 * 8.0 * 3600.0
        and peak_rss_mib < 0.8 * 16.0 * 1024.0
        and math.isfinite(projected_uncached_core_hours)
        and projected_uncached_core_hours <= CPU_CORE_HOUR_CEILING
        and projected_uncached_wall_hours <= WALL_CLOCK_HOUR_CEILING
        and projected_parallel_requested_memory_gb <= MEMORY_GB_CEILING
        and projected_incremental_scratch_gb_upper_bound <= SCRATCH_GB_CEILING
    )
    if not computational_pass:
        raise M1Error(
            "target-free benchmark exceeded frozen time/RSS or 157-path budget projection"
        )
    return {
        "status": (
            "pass_pending_post_job_sacct_MaxRSS_and_remaining_budget_reconciliation"
        ),
        "stage": "v3_M1_target_free_real_shape_benchmark",
        "rows": len(mixed_cells),
        "groups": len(set(cell.group for cell in mixed_cells)),
        "mixed_K_schedule_counts": dict(
            sorted(Counter(cell.k for cell in mixed_cells).items())
        ),
        "mixed_K_unique_candidate_states": mixed_unique_states,
        "mixed_K_feature_moment_build_seconds": feature_cache_seconds,
        "mixed_K_rank_and_condition": mixed_rank,
        "all_K8_unique_candidate_states": worst_unique_states,
        "all_K8_score_computations_per_two_pass_objective": 2 * worst_unique_states,
        "all_K8_feature_moment_build_seconds": worst_feature_seconds,
        "objective_gradient_Hessian_seconds": ogh_seconds,
        "objective_gradient_Hessian_candidate_visits": first.candidate_visits,
        "candidate_sets_scored_per_second": candidate_rate,
        "factorization_seconds": factor_seconds,
        "factorization": solve,
        "complete_12_lambda_warm_path_seconds": path_seconds,
        "complete_path_iterations": {
            format(value, ".12g"): result.iterations for value, result in path.items()
        },
        "uncached_production_path_equivalents_for_pass": uncached_path_equivalents,
        "exact_training_set_reuse_path_equivalents_report_only": (
            exact_reuse_path_equivalents
        ),
        "projected_uncached_production_wall_hours": projected_uncached_wall_hours,
        "projected_uncached_production_core_hours_at_4_CPUs": (
            projected_uncached_core_hours
        ),
        "projected_exact_reuse_wall_hours_report_only": (
            projected_exact_reuse_wall_hours
        ),
        "projected_exact_reuse_core_hours_report_only": (
            projected_exact_reuse_core_hours
        ),
        "projected_maximum_parallel_jobs_at_64_cores": maximum_parallel_jobs,
        "projected_parallel_requested_memory_GB": (
            projected_parallel_requested_memory_gb
        ),
        "projected_incremental_Scratch_GB_upper_bound": (
            projected_incremental_scratch_gb_upper_bound
        ),
        "policy_budget_ceilings": {
            "CPU_core_hours": CPU_CORE_HOUR_CEILING,
            "wall_clock_hours": WALL_CLOCK_HOUR_CEILING,
            "parallel_cores": PARALLEL_CORE_CEILING,
            "memory_GB": MEMORY_GB_CEILING,
            "Scratch_GB": SCRATCH_GB_CEILING,
        },
        "policy_ceiling_projection_pass": True,
        "remaining_budget_reconciliation_required": True,
        "elapsed_wall_seconds": elapsed,
        "process_peak_RSS_MiB_non_authoritative": peak_rss_mib,
        "Slurm_MaxRSS": None,
        "Slurm_MaxRSS_status": "requires_post_job_sacct_batch_step",
        "frozen_time_80_percent_seconds": 0.8 * 8.0 * 3600.0,
        "frozen_memory_80_percent_MiB": 0.8 * 16.0 * 1024.0,
        "real_target_calls_or_real_K_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def _prepare_job_output(path: Path, basename_prefix: str) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise M1Error("target-free probe requires SCRATCH and SLURM_JOB_ID")
    scratch = Path(scratch_value).resolve(strict=True)
    episode = (
        scratch / "br_autoresearch" / "episode10_single_cell_coprojection"
    ).resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    expected_name = f"{basename_prefix}_job_{job_id}"
    if absolute.parent != episode or absolute.name != expected_name or absolute.is_symlink():
        raise M1Error("probe output root is not the frozen job-scoped Scratch child")
    if absolute.exists():
        raise M1Error("probe output root already exists")
    absolute.mkdir(mode=0o770)
    return absolute


def _prepare_probe_output(path: Path) -> Path:
    return _prepare_job_output(path, "v3_m1_target_free_probe_v4")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _scalar_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Copy nested scalar telemetry while rejecting vectors and nonfinite values."""
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            result[str(key)] = _scalar_mapping(value)
        elif value is None or isinstance(value, (str, bool, int)):
            result[str(key)] = value
        elif isinstance(value, (float, np.floating, np.integer)):
            scalar = value.item() if isinstance(value, np.generic) else value
            if isinstance(scalar, float) and not math.isfinite(scalar):
                result[str(key)] = None
            else:
                result[str(key)] = scalar
        else:
            raise M1Error("diagnostic telemetry contains a non-scalar value")
    return result


def _diagnostic_success_record(
    phase: str,
    ridge_lambda: float,
    initialization_source: str,
    initial_from_lambda: float | None,
    phase_fit_ordinal: int,
    fit_ordinal: int,
    result: OptimizerResult,
) -> dict[str, Any]:
    telemetry = result.telemetry
    solves = telemetry.get("solve_records", ())
    last_solve: dict[str, Any] | None = None
    if isinstance(solves, Sequence) and solves:
        candidate = solves[-1]
        if not isinstance(candidate, Mapping):
            raise M1Error("optimizer solve telemetry is malformed")
        last_solve = _scalar_mapping(candidate)
    lines = telemetry.get("line_records", ())
    last_line: dict[str, Any] | None = None
    if isinstance(lines, Sequence) and lines:
        candidate_line = lines[-1]
        if not isinstance(candidate_line, Mapping):
            raise M1Error("optimizer line telemetry is malformed")
        last_line = _scalar_mapping(candidate_line)
    certificate_value = telemetry.get("certificate")
    certificate: dict[str, Any] | None = None
    if certificate_value is not None:
        if not isinstance(certificate_value, Mapping):
            raise M1Error("optimizer certificate telemetry is malformed")
        certificate = _scalar_mapping(certificate_value)
    return {
        "phase": phase,
        "phase_fit_ordinal": phase_fit_ordinal,
        "fit_ordinal": fit_ordinal,
        "lambda": float(ridge_lambda),
        "initialization_source": initialization_source,
        "initial_from_lambda": initial_from_lambda,
        "status": "pass",
        "objective": float(result.objective),
        "gradient_infinity_norm": float(result.gradient_infinity_norm),
        "iterations": int(result.iterations),
        "accepted_steps": int(result.accepted_steps),
        "prior_relative_objective_change": float(
            result.relative_objective_change
        ),
        "convergence_witness": telemetry.get("convergence_witness"),
        "line_search_evaluations": int(
            telemetry.get("line_search_evaluations", 0)
        ),
        "candidate_visits": int(telemetry.get("candidate_visits", 0)),
        "evaluation_seconds": float(telemetry.get("evaluation_seconds", 0.0)),
        "last_Newton_solve": last_solve,
        "last_line_search": last_line,
        "certificate": certificate,
        "first_ordinary_convergence_iteration": telemetry.get(
            "first_ordinary_convergence_iteration"
        ),
        "first_ordinary_convergence_gradient_infinity_norm": telemetry.get(
            "first_ordinary_convergence_gradient_infinity_norm"
        ),
        "first_ordinary_convergence_relative_objective_change": telemetry.get(
            "first_ordinary_convergence_relative_objective_change"
        ),
        "post_witness_confirmation_attempted": bool(
            telemetry.get("post_witness_confirmation_attempted", False)
        ),
        "post_witness_confirmation_accepted_steps": int(
            telemetry.get("post_witness_confirmation_accepted_steps", 0)
        ),
        "post_witness_confirmation_skipped_exact_zero": bool(
            telemetry.get("post_witness_confirmation_skipped_exact_zero", False)
        ),
    }


def _diagnostic_failure_record(
    phase: str,
    ridge_lambda: float,
    initialization_source: str,
    initial_from_lambda: float | None,
    phase_fit_ordinal: int,
    fit_ordinal: int,
    error: M1Error,
) -> dict[str, Any]:
    context: dict[str, Any] | None = None
    if isinstance(error, UncertifiedLineSearchFailure):
        context = _scalar_mapping(error.scalar_context)
    return {
        "phase": phase,
        "phase_fit_ordinal": phase_fit_ordinal,
        "fit_ordinal": fit_ordinal,
        "lambda": float(ridge_lambda),
        "initialization_source": initialization_source,
        "initial_from_lambda": initial_from_lambda,
        "status": "fail",
        "failure_class": type(error).__name__,
        "failure_message": str(error),
        "optimizer_scalar_context": context,
    }


def _synthetic_warm_zero_diagnostic_seals() -> dict[str, bool]:
    return {
        "target_root_argument_supported": False,
        "real_input_root_argument_supported": False,
        "real_target_calls_or_real_K_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def run_synthetic_rank_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_contract(
        args.contract,
        args.synthetic_design_addendum,
        args.synthetic_recovery_addendum,
        args.solver_addendum,
    )
    output_root = _prepare_job_output(
        args.output_root, "v3_m1_synthetic_rank_probe_v1"
    )
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M1_target_free_synthetic_rank_probe",
        "analysis_label": "exploratory",
        "output_root": str(output_root),
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "synthetic_design_addendum": args.synthetic_design_addendum.name,
        "synthetic_recovery_addendum": args.synthetic_recovery_addendum.name,
        "solver_addendum": args.solver_addendum.name,
        "target_root_argument_supported": False,
        "real_target_calls_or_real_K_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }
    summary_path = output_root / "v3_m1_synthetic_rank_probe_summary.json"
    try:
        payload["synthetic_design_rank"] = _synthetic_design_rank_gate()
        cells, moments, _, library, _, _ = generate_synthetic_dataset(
            groups=16, rows_per_group=34, seed=20260927
        )
        unique_states = sum(library.basis(cell.k).count for cell in cells)
        if unique_states != 2097152:
            raise M1Error("544-row rank probe candidate-state count changed")
        rank = rank_condition_gate(
            cells,
            moments,
            [("full_544_row_recovery_training_set", tuple(range(len(cells))))],
            raise_on_failure=False,
        )
        payload["joint_rank_gate"] = rank
        payload["signed_spectrum_diagnostics"] = synthetic_full_rank_diagnostics(
            cells, moments
        )
        payload["unique_candidate_states_per_objective"] = unique_states
        payload["status"] = rank["status"]
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        _atomic_json(summary_path, payload)
        return payload
    except Exception as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(summary_path, payload)
        raise


def run_synthetic_population_recovery_probe(
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_contract(
        args.contract,
        args.synthetic_design_addendum,
        args.synthetic_recovery_addendum,
        args.solver_addendum,
    )
    output_root = _prepare_job_output(
        args.output_root, "v3_m1_synthetic_population_recovery_v2"
    )
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M1_target_free_synthetic_population_recovery_v2",
        "analysis_label": "exploratory",
        "output_root": str(output_root),
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "synthetic_design_addendum": args.synthetic_design_addendum.name,
        "synthetic_recovery_addendum": args.synthetic_recovery_addendum.name,
        "solver_addendum": args.solver_addendum.name,
        "target_root_argument_supported": False,
        "real_input_root_argument_supported": False,
        "real_target_calls_or_real_K_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }
    summary_path = (
        output_root / "v3_m1_synthetic_population_recovery_summary.json"
    )
    try:
        payload["population_conformance"] = (
            _population_objective_conformance_gate()
        )
        payload["synthetic_design_rank"] = _synthetic_design_rank_gate()
        cells, moments, prepared, library, geometry, planted = (
            generate_synthetic_dataset(
                groups=16, rows_per_group=34, seed=20260927
            )
        )
        rank = rank_condition_gate(
            cells,
            moments,
            [("full_544_row_recovery_training_set", tuple(range(len(cells))))],
            raise_on_failure=False,
        )
        payload["joint_rank_gate"] = rank
        if rank["status"] != "pass":
            payload["status"] = "fail"
            payload["failure_stage"] = "unchanged_544_row_rank_gate"
        else:
            recovery = _synthetic_population_recovery_gate(
                cells, prepared, library, geometry, planted
            )
            payload["recovery"] = recovery
            payload["status"] = recovery["status"]
            if recovery["status"] != "pass":
                payload["failure_stage"] = "exact_population_recovery"
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        _atomic_json(summary_path, payload)
        return payload
    except Exception as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(summary_path, payload)
        raise


def run_synthetic_warm_zero_diagnostic(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Replay only the frozen 68-row warm/warm/zero stage with checkpoints."""
    validate_contract(
        args.contract,
        args.synthetic_design_addendum,
        args.synthetic_recovery_addendum,
        args.solver_addendum,
    )
    output_root = _prepare_job_output(args.output_root, WARM_ZERO_OUTPUT_PREFIX)
    summary_path = output_root / "v3_m1_synthetic_warm_zero_diagnostic.json"
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M1_target_free_synthetic_warm_zero_diagnostic_v2",
        "analysis_label": "exploratory",
        "output_root": str(output_root),
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "synthetic_design_addendum": args.synthetic_design_addendum.name,
        "synthetic_recovery_addendum": args.synthetic_recovery_addendum.name,
        "solver_addendum": args.solver_addendum.name,
        "model_or_convergence_threshold_changed": False,
        "Wolfe_or_certificate_threshold_changed": False,
        "stable_line_numerical_implementation_changed_under_solver_addendum_v1": True,
        "one_shot_convergence_confirmation_under_solver_addendum_v2": True,
        "rows": 68,
        "groups": 4,
        "seed": 20260928,
        "phase_order": ["warm_repeat1", "warm_repeat2", "zero"],
        "lambda_order": list(LAMBDAS_DESCENDING),
        "fit_attempts": [],
        "warm_zero_comparisons": [],
        "completed_fit_count": 0,
        "active_fit": None,
        **_synthetic_warm_zero_diagnostic_seals(),
    }

    def checkpoint() -> None:
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        _atomic_json(summary_path, payload)

    checkpoint()
    try:
        cells, _, prepared, library, geometry, _ = generate_synthetic_dataset(
            groups=4, rows_per_group=17, seed=20260928
        )
        group_counts = Counter(cell.group for cell in cells)
        unique_states = sum(library.basis(cell.k).count for cell in cells)
        if (
            len(cells) != 68
            or len(group_counts) != 4
            or set(group_counts.values()) != {17}
            or unique_states != 262144
        ):
            raise M1Error("68-row warm/zero diagnostic shape changed")
        payload["unique_candidate_states_per_objective"] = unique_states
        payload["score_computations_per_two_pass_objective"] = 2 * unique_states
        checkpoint()

        fit_ordinal = 0
        warm_paths: dict[str, dict[float, OptimizerResult]] = {}

        def execute_fit(
            phase: str,
            phase_fit_ordinal: int,
            ridge_lambda: float,
            initial: np.ndarray | None,
            initialization_source: str,
            initial_from_lambda: float | None,
        ) -> OptimizerResult | None:
            nonlocal fit_ordinal
            fit_ordinal += 1
            active = {
                "phase": phase,
                "phase_fit_ordinal": phase_fit_ordinal,
                "fit_ordinal": fit_ordinal,
                "lambda": float(ridge_lambda),
                "initialization_source": initialization_source,
                "initial_from_lambda": initial_from_lambda,
            }
            payload["active_fit"] = active
            checkpoint()
            try:
                result = fit_newton(
                    prepared,
                    library,
                    geometry,
                    ridge_lambda,
                    initial=initial,
                )
            except M1Error as error:
                record = _diagnostic_failure_record(
                    phase,
                    ridge_lambda,
                    initialization_source,
                    initial_from_lambda,
                    phase_fit_ordinal,
                    fit_ordinal,
                    error,
                )
                payload["fit_attempts"].append(record)
                payload["status"] = "fail"
                payload["failure_stage"] = "68_row_warm_zero_fit"
                payload["first_failure"] = record
                payload["active_fit"] = None
                checkpoint()
                return None
            record = _diagnostic_success_record(
                phase,
                ridge_lambda,
                initialization_source,
                initial_from_lambda,
                phase_fit_ordinal,
                fit_ordinal,
                result,
            )
            payload["fit_attempts"].append(record)
            payload["completed_fit_count"] = int(payload["completed_fit_count"]) + 1
            payload["active_fit"] = None
            checkpoint()
            return result

        for phase in ("warm_repeat1", "warm_repeat2"):
            path: dict[float, OptimizerResult] = {}
            initial: np.ndarray | None = None
            previous_lambda: float | None = None
            for phase_fit_ordinal, ridge_lambda in enumerate(
                LAMBDAS_DESCENDING, start=1
            ):
                initialization_source = (
                    "all_261_parameters_zero"
                    if initial is None
                    else "immediately_preceding_converged_joint_B_and_gamma"
                )
                fitted = execute_fit(
                    phase,
                    phase_fit_ordinal,
                    ridge_lambda,
                    initial,
                    initialization_source,
                    previous_lambda,
                )
                if fitted is None:
                    return payload
                path[ridge_lambda] = fitted
                initial = fitted.theta
                previous_lambda = ridge_lambda
            warm_paths[phase] = path

        valid_witnesses = {
            "gradient_and_repeated_unchanged_initial_objective",
            "gradient_and_relative_objective_change",
            "strong_convexity_gap_after_zoom_stall",
        }
        for phase_fit_ordinal, ridge_lambda in enumerate(
            LAMBDAS_DESCENDING, start=1
        ):
            zero = execute_fit(
                "zero",
                phase_fit_ordinal,
                ridge_lambda,
                None,
                "all_261_parameters_zero",
                None,
            )
            if zero is None:
                return payload
            first = warm_paths["warm_repeat1"][ridge_lambda]
            second = warm_paths["warm_repeat2"][ridge_lambda]
            repeat_distribution = compare_conditional_distributions(
                first.theta,
                second.theta,
                prepared,
                library,
                geometry,
            )
            zero_distribution = compare_conditional_distributions(
                first.theta,
                zero.theta,
                prepared,
                library,
                geometry,
            )
            repeat_objective = abs(first.objective - second.objective)
            repeat_coefficient = float(
                np.max(np.abs(first.theta - second.theta))
            )
            zero_objective = abs(first.objective - zero.objective)
            objective_tolerance = max(
                1.0e-10,
                1.1
                * PARAMETER_DIMENSION
                * GRADIENT_TOLERANCE**2
                / (2.0 * ridge_lambda),
            )
            passed = bool(
                repeat_objective <= 1.0e-12
                and repeat_coefficient <= 1.0e-10
                and repeat_distribution[
                    "maximum_candidate_probability_absolute_difference"
                ]
                <= 1.0e-10
                and zero_objective <= objective_tolerance
                and zero_distribution[
                    "maximum_candidate_probability_absolute_difference"
                ]
                <= 1.0e-5
                and zero_distribution[
                    "maximum_candidate_log_probability_absolute_difference"
                ]
                <= 1.0e-4
                and all(
                    result.telemetry.get("convergence_witness")
                    in valid_witnesses
                    for result in (first, second, zero)
                )
            )
            comparison = {
                "lambda": float(ridge_lambda),
                "status": "pass" if passed else "fail",
                "repeat_objective_absolute_difference": repeat_objective,
                "repeat_coefficient_max_absolute_difference": repeat_coefficient,
                "repeat_probability_max_absolute_difference": repeat_distribution[
                    "maximum_candidate_probability_absolute_difference"
                ],
                "zero_start_objective_absolute_difference": zero_objective,
                "zero_start_objective_absolute_tolerance": objective_tolerance,
                "zero_start_probability_max_absolute_difference": zero_distribution[
                    "maximum_candidate_probability_absolute_difference"
                ],
                "zero_start_log_probability_max_absolute_difference": zero_distribution[
                    "maximum_candidate_log_probability_absolute_difference"
                ],
                "warm_repeat1_witness": first.telemetry.get(
                    "convergence_witness"
                ),
                "warm_repeat2_witness": second.telemetry.get(
                    "convergence_witness"
                ),
                "zero_witness": zero.telemetry.get("convergence_witness"),
            }
            payload["warm_zero_comparisons"].append(comparison)
            checkpoint()
            if not passed:
                payload["status"] = "fail"
                payload["failure_stage"] = "68_row_warm_zero_comparison"
                payload["first_failure"] = comparison
                checkpoint()
                return payload

        payload["status"] = "pass"
        payload["failure_stage"] = None
        checkpoint()
        return payload
    except Exception as error:
        payload.update(
            {
                "status": "fail",
                "failure_stage": "68_row_warm_zero_diagnostic_infrastructure",
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "active_fit": None,
            }
        )
        checkpoint()
        raise


def run_full_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_contract(
        args.contract,
        args.synthetic_design_addendum,
        args.synthetic_recovery_addendum,
        args.solver_addendum,
    )
    output_root = _prepare_probe_output(args.output_root)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M1_target_free_pre_real_probe_v4",
        "analysis_label": "exploratory",
        "output_root": str(output_root),
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "candidate_batch_maximum": BATCH_MAXIMUM,
        "joint_parameter_dimension": PARAMETER_DIMENSION,
        "streamed_cell_moment_dimension": JOINT_CELL_DIMENSION,
        "synthetic_design_addendum": args.synthetic_design_addendum.name,
        "synthetic_recovery_addendum": args.synthetic_recovery_addendum.name,
        "solver_addendum": args.solver_addendum.name,
        "candidate_by_261_materialized": False,
        "target_root_argument_supported": False,
        "real_target_calls_or_real_K_opened": False,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }
    summary_path = output_root / "v3_m1_target_free_probe_summary.json"
    try:
        payload["unit_gates"] = run_unit_gates()
        payload["synthetic_gates"] = run_synthetic_gates()
        if payload["synthetic_gates"]["status"] != "pass":
            raise M1Error("target-free v4 synthetic gates failed closed")
        payload["real_shape_benchmark"] = run_real_shape_benchmark(
            args.interface_root, args.geometry_root
        )
        payload["status"] = (
            "pass_pending_post_job_sacct_MaxRSS_and_remaining_budget_reconciliation"
        )
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        _atomic_json(summary_path, payload)
        return payload
    except Exception as error:
        payload.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "elapsed_wall_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(summary_path, payload)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unit-test", action="store_true")
    mode.add_argument("--synthetic-rank-probe", action="store_true")
    mode.add_argument("--synthetic-population-recovery-probe", action="store_true")
    mode.add_argument("--synthetic-warm-zero-diagnostic", action="store_true")
    mode.add_argument("--probe", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--synthetic-design-addendum",
        type=Path,
        default=SYNTHETIC_DESIGN_ADDENDUM_PATH,
    )
    parser.add_argument(
        "--synthetic-recovery-addendum",
        type=Path,
        default=SYNTHETIC_RECOVERY_ADDENDUM_PATH,
    )
    parser.add_argument(
        "--solver-addendum",
        type=Path,
        default=SOLVER_ADDENDUM_PATH,
    )
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.unit_test:
        if any(
            value is not None
            for value in (args.interface_root, args.geometry_root, args.output_root)
        ):
            parser.error("unit test accepts no real input or output paths")
        try:
            validate_contract(
                args.contract,
                args.synthetic_design_addendum,
                args.synthetic_recovery_addendum,
                args.solver_addendum,
            )
            result = run_unit_gates()
        except (M1Error, AssertionError, OSError, ValueError) as error:
            print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.synthetic_rank_probe:
        if args.output_root is None:
            parser.error("synthetic rank probe requires --output-root")
        if args.interface_root is not None or args.geometry_root is not None:
            parser.error("synthetic rank probe accepts no real input roots")
        try:
            result = run_synthetic_rank_probe(args)
        except (M1Error, AssertionError, OSError, ValueError, np.linalg.LinAlgError) as error:
            print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stage": result["stage"],
                    "output_root": result["output_root"],
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "pass" else 2
    if args.synthetic_population_recovery_probe:
        if args.output_root is None:
            parser.error("synthetic population recovery probe requires --output-root")
        if args.interface_root is not None or args.geometry_root is not None:
            parser.error(
                "synthetic population recovery probe accepts no real input roots"
            )
        try:
            result = run_synthetic_population_recovery_probe(args)
        except (M1Error, AssertionError, OSError, ValueError, np.linalg.LinAlgError) as error:
            print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stage": result["stage"],
                    "output_root": result["output_root"],
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "pass" else 2
    if args.synthetic_warm_zero_diagnostic:
        if args.output_root is None:
            parser.error("synthetic warm/zero diagnostic requires --output-root")
        if args.interface_root is not None or args.geometry_root is not None:
            parser.error(
                "synthetic warm/zero diagnostic accepts no real input roots"
            )
        try:
            result = run_synthetic_warm_zero_diagnostic(args)
        except (
            M1Error,
            AssertionError,
            OSError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "stage": result["stage"],
                    "output_root": result["output_root"],
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "pass" else 2
    missing = [
        name
        for name in ("interface_root", "geometry_root", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "probe requires: "
            + ", ".join("--" + value.replace("_", "-") for value in missing)
        )
    try:
        result = run_full_probe(args)
    except (M1Error, AssertionError, OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "output_root": result["output_root"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
