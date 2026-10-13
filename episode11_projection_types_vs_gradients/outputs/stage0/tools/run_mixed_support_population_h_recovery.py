#!/usr/bin/env python3
"""Synthetic-only fitted-H population recovery for EP11.

This runner consumes one passing full-ladder population-C SAA summary and its
validated ``population_saa_C_model_state.npz``.  It rebuilds the identical
R=8192 group-stratified SAA store, replays the frozen C objective, and jointly
optimizes every parameter of H from three fixed start types: the exact fitted-C
embedding, a nonzero truth-based diagnostic start, and a nonzero empirical
start.  The best converged fit receives one fixed-start polish, after which
fit-stream likelihood alone ranks the polished, pre-polish, and second-start
candidates.  Distinct fresh anchor and verification streams then check exact-H
scoring and paired truth/C/H performance.

The program uses only the outcome-blind metadata layout and fully synthetic
56-coordinate outcomes.  It is an optimizer-recovery diagnostic, not amplitude
calibration, registered production-family qualification, Panel-A completion,
or authorization for real-outcome access.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


NUMERICAL_LIBRARY_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def force_single_numerical_library_thread() -> None:
    for name in NUMERICAL_LIBRARY_THREAD_VARIABLES:
        os.environ[name] = "1"


force_single_numerical_library_thread()

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student_t

import run_mixed_support_full_family_calibration as full
import run_mixed_support_population_saa as population


SUMMARY_FILENAME = "population_h_recovery_summary.json"
EMPIRICAL_PERTURBATION_SUMMARY_FILENAME = (
    "population_h_empirical_perturbation_v1_summary.json"
)
LEGACY_START_POLICY = "raw_empirical_v1"
EMPIRICAL_PERTURBATION_POLICY = "empirical_perturbation_v1"
EMPIRICAL_PERTURBATION_SEED = 20269193
EXPECTED_C_SUMMARY_FILENAME = population.SUMMARY_FILENAME
EXPECTED_C_STATE_FILENAME = population.MODEL_STATE_CHECKPOINT_FILENAME
EXPECTED_C_PASS_STATUS = (
    "pass_population_C_SAA_pilot_gates_not_Panel_A_calibration"
)
EXPECTED_C_AMPLITUDE = 1.25
EXPECTED_SAA_LADDER = (128, 512, 2048, 8192)
EXPECTED_FINAL_REPLICATES = 8192
DEFAULT_VERIFICATION_STREAM = 5601
H_EXACT_ANCHOR_STREAM = 5701
H_EXACT_ANCHOR_REPLICATES = 2
H_EXACT_ANCHOR_TOLERANCE = 1.0e-10
H_MAIN_MAX_ITERATIONS = 1200
H_POLISH_MAX_ITERATIONS = 600
H_POLISH_FRESH_GAIN_LIMIT = 0.005
H_POLISH_FRESH_GAIN_MCSE_LIMIT = 0.005
MAX_EXACT_FLOAT_INTEGER = population.MAX_EXACT_FLOAT_INTEGER
H_START_SOURCES = (
    "embedded_fitted_C",
    "truth_based_nonzero_signature",
    "empirical_nonzero_signature",
)
EXPECTED_C_FIRST_LEVEL_FRESH_STARTS = 3
EXPECTED_C_LATER_LEVEL_FRESH_STARTS = 2
EXPECTED_C_FRESH_STARTS_BY_RUNG = (3, 2, 2, 2)
EXPECTED_C_WARM_STARTS_BY_RUNG = (0, 1, 1, 1)
EXPECTED_C_SCHEDULED_STARTS_BY_RUNG = (3, 3, 3, 3)
C_PILOT_GATE_NAMES = (
    "exact_analytic_anchors_pass",
    "all_fit_validation_terminal_streams_are_distinct",
    "SAA_ladder_completed",
    "at_least_two_physical_KKT_converged_starts_at_every_level",
    "best_solution_numerical_boundaries_inactive_at_every_level",
    "best_two_validation_scores_differ_below_threshold_at_every_level",
    "last_ladder_increase_changes_fresh_score_below_threshold_with_precise_pair",
    "fixed_start_polishing_fresh_score_improvement_below_threshold",
    "best_two_terminal_scores_differ_below_threshold",
    "terminal_truth_minus_C_MCSE_at_most_target",
)


@dataclass(frozen=True)
class CRunConfig:
    amplitude: float
    ladder: tuple[int, ...]
    validation_replicates: int
    terminal_replicates: int
    anchor_replicates: int
    streams: population.StreamIDs
    seed: int


@dataclass(frozen=True)
class HStartTask:
    start_index: int
    source: str
    store: population.SAAMemmapStore
    initial_theta: np.ndarray
    replicates: int
    chunk_replicates: int
    preconditioner_replicates: int
    max_iterations: int


@dataclass(frozen=True)
class HStartResult:
    start_index: int
    source: str
    solution: full.OptimizerSolution


@dataclass(frozen=True)
class VerificationTask:
    start: int
    stop: int
    layout: full.core.MetadataLayout
    basis: full.DesignBasis
    indices: np.ndarray
    amplitude: float
    seed: int
    stream: int
    c_theta: np.ndarray
    h_thetas: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VerificationChunk:
    start: int
    stop: int
    truth_log_density: np.ndarray
    c_log_density: np.ndarray
    h_log_density: np.ndarray


def require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def require_boolean(value: object, expected: bool, name: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{name} must be exactly {expected}")


def require_integer(value: object, expected: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{name} must be integer {expected}")


def require_number(value: object, expected: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number) or number != float(expected):
        raise ValueError(f"{name} does not match the fitted-C checkpoint")


def require_positive_number_at_most(
    value: object, maximum: float, name: str
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0 or number > maximum:
        raise ValueError(f"{name} must be positive and at most {maximum}")


def require_string(value: object, expected: str, name: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{name} must be {expected!r}")


def validate_c_start_schedule_contract(value: object) -> Mapping[str, object]:
    schedule = require_mapping(value, "start_schedule")
    require_string(
        schedule.get("policy"),
        "first_rung_fresh_override_then_later_fresh_plus_previous_best_warm_v1",
        "start_schedule.policy",
    )
    require_integer(
        schedule.get("first_level_fresh_starts_requested"),
        EXPECTED_C_FIRST_LEVEL_FRESH_STARTS,
        "start_schedule.first_level_fresh_starts_requested",
    )
    require_integer(
        schedule.get("later_level_fresh_starts_requested"),
        EXPECTED_C_LATER_LEVEL_FRESH_STARTS,
        "start_schedule.later_level_fresh_starts_requested",
    )
    require_integer(
        schedule.get("first_level_scheduled_starts"),
        EXPECTED_C_SCHEDULED_STARTS_BY_RUNG[0],
        "start_schedule.first_level_scheduled_starts",
    )
    require_integer(
        schedule.get("later_level_scheduled_starts"),
        EXPECTED_C_SCHEDULED_STARTS_BY_RUNG[1],
        "start_schedule.later_level_scheduled_starts",
    )
    require_boolean(
        schedule.get("later_levels_add_previous_level_best_as_one_extra_warm_start"),
        True,
        "start_schedule.later_levels_add_previous_level_best_as_one_extra_warm_start",
    )
    return schedule


def reject_nonstandard_json_constant(text: str) -> object:
    raise ValueError(f"Nonstandard JSON numeric constant {text!r} is forbidden")


def load_json_object(path: Path) -> Mapping[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, parse_constant=reject_nonstandard_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read fitted-C summary JSON: {exc}") from exc
    return require_mapping(value, "fitted-C summary")


def exact_checkpoint_integer(array: np.ndarray, name: str) -> int:
    if array.shape != (1,):
        raise ValueError(f"Checkpoint field {name} is not scalar")
    value = float(array[0])
    if not value.is_integer() or abs(value) > MAX_EXACT_FLOAT_INTEGER:
        raise ValueError(f"Checkpoint field {name} is not an exact integer")
    return int(value)


def exact_checkpoint_integer_vector(array: np.ndarray, name: str) -> tuple[int, ...]:
    if array.ndim != 1:
        raise ValueError(f"Checkpoint field {name} is not a vector")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded) or np.any(
        np.abs(array) > MAX_EXACT_FLOAT_INTEGER
    ):
        raise ValueError(f"Checkpoint field {name} is not exact-integer valued")
    return tuple(int(value) for value in rounded.tolist())


def bootstrap_c_checkpoint_config(path: Path) -> CRunConfig:
    """Read only the config needed to reconstruct the authoritative SAA store.

    Full array, cross-link, and objective validation is performed after the
    store is rebuilt by ``load_validate_c_model_state_checkpoint``.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(population.MODEL_STATE_ARRAY_KEYS):
                raise ValueError("checkpoint fields do not match the C-state schema")
            arrays: dict[str, np.ndarray] = {}
            for name in archive.files:
                array = np.asarray(archive[name])
                if array.dtype != np.dtype(np.float64):
                    raise ValueError(f"checkpoint field {name} is not float64")
                if not np.isfinite(array).all():
                    raise ValueError(f"checkpoint field {name} is nonfinite")
                arrays[name] = array
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot bootstrap fitted-C checkpoint: {exc}") from exc

    if exact_checkpoint_integer(arrays["schema_version"], "schema_version") != (
        population.MODEL_STATE_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported fitted-C checkpoint schema")
    if exact_checkpoint_integer(arrays["family_code"], "family_code") != (
        population.MODEL_STATE_FAMILY_CODE_C
    ):
        raise ValueError("The model-state checkpoint is not family C")
    if arrays["amplitude"].shape != (1,):
        raise ValueError("The fitted-C checkpoint amplitude is not scalar")
    amplitude = float(arrays["amplitude"][0])
    if not np.isfinite(amplitude) or amplitude != EXPECTED_C_AMPLITUDE:
        raise ValueError(
            f"The fitted-C checkpoint amplitude must be {EXPECTED_C_AMPLITUDE}"
        )
    requested = exact_checkpoint_integer_vector(
        arrays["requested_saa_ladder"], "requested_saa_ladder"
    )
    executed = exact_checkpoint_integer_vector(
        arrays["executed_rung_replicates"], "executed_rung_replicates"
    )
    if requested != EXPECTED_SAA_LADDER or executed != EXPECTED_SAA_LADDER:
        raise ValueError(
            "Fitted-H recovery requires a completed C ladder of 128,512,2048,8192"
        )
    if exact_checkpoint_integer(
        arrays["frozen_training_replicates"], "frozen_training_replicates"
    ) != EXPECTED_FINAL_REPLICATES:
        raise ValueError("Frozen C was not trained on the R=8192 SAA prefix")
    stream_values = exact_checkpoint_integer_vector(arrays["stream_ids"], "stream_ids")
    if len(stream_values) != len(population.MODEL_STATE_STREAM_NAMES):
        raise ValueError("Fitted-C checkpoint stream vector has the wrong length")
    if stream_values[-1] != population.TRUTH_TEMPLATE_STREAM:
        raise ValueError("Fitted-C checkpoint truth-template stream changed")
    streams = population.StreamIDs(*stream_values[:4])
    streams.validate()
    validation_replicates = exact_checkpoint_integer(
        arrays["validation_replicates"], "validation_replicates"
    )
    terminal_replicates = exact_checkpoint_integer(
        arrays["terminal_replicates"], "terminal_replicates"
    )
    anchor_replicates = exact_checkpoint_integer(
        arrays["anchor_replicates"], "anchor_replicates"
    )
    seed = exact_checkpoint_integer(arrays["seed"], "seed")
    if min(validation_replicates, terminal_replicates, anchor_replicates) < 2:
        raise ValueError("Fitted-C evaluation replicate counts must be at least two")
    if not (0 <= seed <= MAX_EXACT_FLOAT_INTEGER):
        raise ValueError("Fitted-C seed must be a nonnegative exact integer")
    return CRunConfig(
        amplitude=amplitude,
        ladder=requested,
        validation_replicates=validation_replicates,
        terminal_replicates=terminal_replicates,
        anchor_replicates=anchor_replicates,
        streams=streams,
        seed=seed,
    )


def require_all_true_gates(
    value: object, names: Sequence[str], label: str
) -> Mapping[str, object]:
    gates = require_mapping(value, label)
    missing = [name for name in names if name not in gates]
    if missing:
        raise ValueError(f"{label} is missing gates: {', '.join(missing)}")
    for name in names:
        require_boolean(gates[name], True, f"{label}.{name}")
    return gates


def validate_passing_c_summary(
    summary_path: Path,
    state_path: Path,
    config: CRunConfig,
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    expected_preconditioner_replicates: int,
    expected_initialization_replicates: int,
) -> Mapping[str, object]:
    if summary_path.name != EXPECTED_C_SUMMARY_FILENAME:
        raise ValueError(
            f"Fitted-C summary must be named {EXPECTED_C_SUMMARY_FILENAME}"
        )
    if state_path.name != EXPECTED_C_STATE_FILENAME:
        raise ValueError(
            f"Fitted-C state must be named {EXPECTED_C_STATE_FILENAME}"
        )
    try:
        summary_parent = summary_path.resolve(strict=True).parent
        state_parent = state_path.resolve(strict=True).parent
    except OSError as exc:
        raise ValueError(f"A fitted-C input does not exist: {exc}") from exc
    if summary_parent != state_parent:
        raise ValueError("Fitted-C summary and checkpoint must share one result directory")

    summary = load_json_object(summary_path)
    require_string(summary.get("diagnostic"), "EP11_group_stratified_population_C_SAA_pilot", "diagnostic")
    require_string(summary.get("status"), EXPECTED_C_PASS_STATUS, "status")
    require_boolean(summary.get("uses_real_projection_outcomes"), False, "uses_real_projection_outcomes")
    require_boolean(summary.get("outcome_bearing_search_authorized"), False, "outcome_bearing_search_authorized")
    require_boolean(summary.get("Panel_A_population_calibration_complete"), False, "Panel_A_population_calibration_complete")
    require_boolean(summary.get("terminal_population_calibration_pass"), False, "terminal_population_calibration_pass")
    require_boolean(summary.get("D12_to_A8_transport_evaluated"), False, "D12_to_A8_transport_evaluated")
    require_number(summary.get("amplitude"), config.amplitude, "amplitude")
    require_integer(summary.get("A8_units"), len(set(layout.units[indices].tolist())), "A8_units")
    require_integer(summary.get("A8_cells"), len(indices), "A8_cells")
    require_integer(summary.get("P_design_rank"), basis.rank, "P_design_rank")
    require_integer(summary.get("C_parameter_count"), full.ParameterCodec("C", basis.rank).size, "C_parameter_count")
    if summary.get("SAA_ladder") != list(EXPECTED_SAA_LADDER):
        raise ValueError("Fitted-C summary does not report the required full ladder")
    require_boolean(
        summary.get("default_ladder_is_a_pilot_not_terminal_Panel_A_calibration"),
        False,
        "default_ladder_is_a_pilot_not_terminal_Panel_A_calibration",
    )
    require_string(
        summary.get("group_integration"),
        "exact_two_stratum_probability_weighting_per_metadata_row",
        "group_integration",
    )
    require_string(
        summary.get("support_and_value_noise_integration"),
        "nested_sample_average_approximation",
        "support_and_value_noise_integration",
    )
    preconditioner = require_mapping(summary.get("preconditioner"), "preconditioner")
    require_integer(
        preconditioner.get("fit_stream_prefix_replicates"),
        expected_preconditioner_replicates,
        "preconditioner.fit_stream_prefix_replicates",
    )
    start_schedule = validate_c_start_schedule_contract(
        summary.get("start_schedule")
    )
    require_integer(
        start_schedule.get("maximum_iterations_per_main_start"),
        H_MAIN_MAX_ITERATIONS,
        "start_schedule.maximum_iterations_per_main_start",
    )
    require_integer(
        start_schedule.get("maximum_iterations_for_fixed_start_polish"),
        H_POLISH_MAX_ITERATIONS,
        "start_schedule.maximum_iterations_for_fixed_start_polish",
    )
    require_integer(
        start_schedule.get("initialization_prefix_replicates"),
        expected_initialization_replicates,
        "start_schedule.initialization_prefix_replicates",
    )

    thresholds = require_mapping(summary.get("thresholds"), "thresholds")
    for name, maximum in (
        ("validation_start_stability_nats_per_neuron", 0.002),
        ("successive_ladder_validation_change_nats_per_neuron", 0.005),
        ("terminal_start_stability_nats_per_neuron", 0.005),
        ("maximum_fresh_score_gain_from_polishing_nats_per_neuron", 0.005),
        ("terminal_MCSE_nats_per_neuron", 0.005),
    ):
        require_positive_number_at_most(
            thresholds.get(name), maximum, f"thresholds.{name}"
        )
    require_number(
        thresholds.get("analytic_anchor_maximum_absolute_error"),
        population.ANCHOR_TOLERANCE,
        "thresholds.analytic_anchor_maximum_absolute_error",
    )
    require_number(
        thresholds.get("physical_projected_gradient"),
        full.PROJECTED_GRADIENT_TOLERANCE,
        "thresholds.physical_projected_gradient",
    )

    streams = require_mapping(summary.get("streams"), "streams")
    for name, expected in (
        ("fit", config.streams.fit),
        ("validation", config.streams.validation),
        ("terminal", config.streams.terminal),
        ("anchor", config.streams.anchor),
        ("truth_template", population.TRUTH_TEMPLATE_STREAM),
    ):
        require_integer(streams.get(name), expected, f"streams.{name}")
    require_boolean(streams.get("all_distinct"), True, "streams.all_distinct")
    require_boolean(
        streams.get("fit_ladder_uses_nested_prefixes"),
        True,
        "streams.fit_ladder_uses_nested_prefixes",
    )
    require_boolean(
        streams.get("terminal_stream_opened_only_after_final_ladder_fit"),
        True,
        "streams.terminal_stream_opened_only_after_final_ladder_fit",
    )

    anchors = require_mapping(summary.get("anchors"), "anchors")
    for name, family, amplitude in (
        ("C_amplitude_zero", "C", 0.0),
        ("H_requested_amplitude", "H", config.amplitude),
    ):
        anchor = require_mapping(anchors.get(name), f"anchors.{name}")
        require_string(anchor.get("family"), family, f"anchors.{name}.family")
        require_number(anchor.get("amplitude"), amplitude, f"anchors.{name}.amplitude")
        require_integer(anchor.get("replicates"), config.anchor_replicates, f"anchors.{name}.replicates")
        require_boolean(anchor.get("pass"), True, f"anchors.{name}.pass")

    ladder_rows = summary.get("ladder")
    if not isinstance(ladder_rows, list) or len(ladder_rows) != len(EXPECTED_SAA_LADDER):
        raise ValueError("Fitted-C summary ladder rows are incomplete")
    for (
        expected_replicates,
        expected_fresh_starts,
        expected_warm_starts,
        expected_starts,
        raw_row,
    ) in zip(
        EXPECTED_SAA_LADDER,
        EXPECTED_C_FRESH_STARTS_BY_RUNG,
        EXPECTED_C_WARM_STARTS_BY_RUNG,
        EXPECTED_C_SCHEDULED_STARTS_BY_RUNG,
        ladder_rows,
    ):
        row = require_mapping(raw_row, f"ladder.R{expected_replicates}")
        require_integer(row.get("SAA_replicates"), expected_replicates, f"ladder.R{expected_replicates}.SAA_replicates")
        optimizer = require_mapping(
            row.get("optimizer"), f"ladder.R{expected_replicates}.optimizer"
        )
        require_integer(
            row.get("fresh_starts_scheduled"),
            expected_fresh_starts,
            f"ladder.R{expected_replicates}.fresh_starts_scheduled",
        )
        require_integer(
            row.get("previous_level_warm_starts_scheduled"),
            expected_warm_starts,
            f"ladder.R{expected_replicates}.previous_level_warm_starts_scheduled",
        )
        require_integer(
            optimizer.get("scheduled_starts"),
            expected_starts,
            f"ladder.R{expected_replicates}.optimizer.scheduled_starts",
        )
        require_boolean(
            row.get("at_least_two_physical_KKT_converged_starts"),
            True,
            f"ladder.R{expected_replicates}.at_least_two_physical_KKT_converged_starts",
        )
        require_boolean(
            row.get("best_solution_numerical_boundaries_inactive"),
            True,
            f"ladder.R{expected_replicates}.best_solution_numerical_boundaries_inactive",
        )
        validation = row.get("validation")
        if not isinstance(validation, list) or len(validation) < 2:
            raise ValueError(f"R={expected_replicates} lacks two C validation candidates")
        for candidate in validation:
            candidate_map = require_mapping(candidate, "C validation candidate")
            require_integer(
                candidate_map.get("replicates"),
                config.validation_replicates,
                "C validation candidate replicates",
            )

    terminal = require_mapping(summary.get("terminal"), "terminal")
    require_integer(terminal.get("stream"), config.streams.terminal, "terminal.stream")
    require_integer(terminal.get("replicates"), config.terminal_replicates, "terminal.replicates")
    terminal_solutions = terminal.get("solutions")
    if not isinstance(terminal_solutions, list) or len(terminal_solutions) < 2:
        raise ValueError("Passing fitted-C summary lacks two terminal candidates")

    require_all_true_gates(summary.get("pilot_acceptance"), C_PILOT_GATE_NAMES, "pilot_acceptance")
    acceptance = require_all_true_gates(summary.get("acceptance"), C_PILOT_GATE_NAMES, "acceptance")
    require_boolean(
        acceptance.get("recommended_terminal_SAA_ladder_128_512_2048_8192_completed"),
        True,
        "acceptance.recommended_terminal_SAA_ladder_128_512_2048_8192_completed",
    )
    require_boolean(
        acceptance.get("fitted_H_population_optimizer_recovery_completed"),
        False,
        "acceptance.fitted_H_population_optimizer_recovery_completed",
    )
    require_boolean(
        acceptance.get("terminal_population_calibration_pass"),
        False,
        "acceptance.terminal_population_calibration_pass",
    )
    require_boolean(
        acceptance.get("Panel_A_population_calibration_complete"),
        False,
        "acceptance.Panel_A_population_calibration_complete",
    )

    checkpoint = require_mapping(summary.get("model_state_checkpoint"), "model_state_checkpoint")
    require_string(checkpoint.get("filename"), EXPECTED_C_STATE_FILENAME, "model_state_checkpoint.filename")
    require_integer(
        checkpoint.get("schema_version"),
        population.MODEL_STATE_SCHEMA_VERSION,
        "model_state_checkpoint.schema_version",
    )
    require_string(checkpoint.get("family"), "C", "model_state_checkpoint.family")
    require_string(
        checkpoint.get("write_status"),
        "training_objective_replayed_then_atomic_write_completed",
        "model_state_checkpoint.write_status",
    )
    frozen = require_mapping(checkpoint.get("frozen_theta"), "model_state_checkpoint.frozen_theta")
    require_integer(
        frozen.get("zero_based_terminal_candidate_index"),
        0,
        "model_state_checkpoint.frozen_theta.zero_based_terminal_candidate_index",
    )
    require_boolean(
        frozen.get("terminal_scores_used_for_parameter_selection"),
        False,
        "model_state_checkpoint.frozen_theta.terminal_scores_used_for_parameter_selection",
    )
    return summary


def finite_json_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite JSON number")
    return number


def validate_c_summary_checkpoint_numeric_crosslinks(
    summary: Mapping[str, object], state: Mapping[str, np.ndarray]
) -> dict[str, object]:
    """Cross-link the accepted C JSON numerics to the loaded checkpoint."""
    checkpoint = require_mapping(
        summary.get("model_state_checkpoint"), "model_state_checkpoint"
    )
    frozen = require_mapping(
        checkpoint.get("frozen_theta"), "model_state_checkpoint.frozen_theta"
    )
    summary_frozen_objective = finite_json_number(
        frozen.get("fit_stream_training_objective"),
        "model_state_checkpoint.frozen_theta.fit_stream_training_objective",
    )
    state_frozen_objective = float(state["frozen_objective"][0])
    if not np.isclose(
        summary_frozen_objective,
        state_frozen_objective,
        atol=population.MODEL_STATE_OBJECTIVE_ATOL,
        rtol=population.MODEL_STATE_OBJECTIVE_RTOL,
    ):
        raise ValueError(
            "Accepted C summary frozen objective differs from the checkpoint"
        )

    terminal = require_mapping(summary.get("terminal"), "terminal")
    ranking = terminal.get("deduplicated_fit_objective_ranking")
    if not isinstance(ranking, list):
        raise ValueError("Accepted C summary terminal ranking is not a list")
    state_objectives = np.asarray(
        state["terminal_candidate_objectives"], dtype=float
    )
    if len(ranking) != len(state_objectives):
        raise ValueError(
            "Accepted C summary/checkpoint terminal candidate counts differ"
        )
    for index, (raw_row, state_objective) in enumerate(
        zip(ranking, state_objectives)
    ):
        row = require_mapping(raw_row, f"terminal ranking row {index + 1}")
        require_integer(row.get("rank"), index + 1, "terminal ranking rank")
        summary_objective = finite_json_number(
            row.get("fit_stream_training_objective"),
            "terminal ranking fit-stream objective",
        )
        if not np.isclose(
            summary_objective,
            float(state_objective),
            atol=population.MODEL_STATE_OBJECTIVE_ATOL,
            rtol=population.MODEL_STATE_OBJECTIVE_RTOL,
        ):
            raise ValueError(
                "Accepted C summary/checkpoint terminal objective order differs"
            )

    scored = terminal.get("solutions")
    expected_scored_count = min(2, len(state_objectives))
    if not isinstance(scored, list) or len(scored) != expected_scored_count:
        raise ValueError(
            "Accepted C summary/checkpoint terminal scored-candidate counts differ"
        )
    for index, raw_row in enumerate(scored):
        row = require_mapping(raw_row, f"terminal scored solution {index + 1}")
        objective = finite_json_number(
            row.get("training_negative_mean_log_density"),
            "terminal scored solution training objective",
        )
        if not np.isclose(
            objective,
            float(state_objectives[index]),
            atol=population.MODEL_STATE_OBJECTIVE_ATOL,
            rtol=population.MODEL_STATE_OBJECTIVE_RTOL,
        ):
            raise ValueError(
                "Accepted C terminal scored solution differs from checkpoint order"
            )
    return {
        "pass": True,
        "frozen_training_objective_matches": True,
        "ordered_terminal_candidate_count": len(state_objectives),
        "ordered_terminal_candidate_objectives_match": True,
        "terminal_scored_candidate_count": expected_scored_count,
    }


def require_fresh_output_namespace(
    output_dir: Path, summary_filename: str = SUMMARY_FILENAME
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    final_path = output_dir / summary_filename
    if os.path.lexists(final_path):
        raise FileExistsError(f"Refusing to overwrite existing final artifact: {final_path}")


def write_summary(
    output_dir: Path,
    summary: Mapping[str, object],
    summary_filename: str = SUMMARY_FILENAME,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / summary_filename
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{summary_filename}.",
            suffix=".tmp",
            dir=output_dir,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        population.publish_temporary_file_without_overwrite(temporary_path, final_path)
        temporary_path = None
        population.fsync_directory(output_dir)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def h_attempt_settings(start_policy: str) -> tuple[str, int, int]:
    """Preserve the old attempt and freeze fresh streams for its revision."""
    if start_policy == LEGACY_START_POLICY:
        return SUMMARY_FILENAME, DEFAULT_VERIFICATION_STREAM, H_EXACT_ANCHOR_STREAM
    if start_policy == EMPIRICAL_PERTURBATION_POLICY:
        return EMPIRICAL_PERTURBATION_SUMMARY_FILENAME, 5602, 5702
    raise ValueError(f"Unknown H start policy: {start_policy}")


def h_initial_starts(
    store: population.SAAMemmapStore,
    c_theta: np.ndarray,
    amplitude: float,
    initialization_replicates: int,
    start_policy: str = LEGACY_START_POLICY,
) -> tuple[tuple[str, np.ndarray], ...]:
    h_attempt_settings(start_policy)
    codec_h = full.ParameterCodec("H", store.basis.rank)
    embedded = full.embed_c_theta_in_h(c_theta, store.basis.rank)
    lower = np.asarray([bound[0] for bound in codec_h.bounds()], dtype=float)
    upper = np.asarray([bound[1] for bound in codec_h.bounds()], dtype=float)
    if np.any(embedded < lower) or np.any(embedded > upper):
        raise ValueError("The exact fitted-C embedding lies outside H bounds")

    initialization_stop = min(int(initialization_replicates), store.maximum_replicates)
    data = store.score_data(0, initialization_stop, include_outcomes=True)
    pool = full.PopulationPool(
        data=data,
        single_design=store.basis.apply(store.layout.p_design[store.indices]),
        truth=store.truth,
        replicates=initialization_stop,
    )
    if start_policy == EMPIRICAL_PERTURBATION_POLICY:
        _, truth_start, empirical_start = full.make_initial_starts(
            codec_h,
            pool,
            amplitude,
            count=3,
            seed=EMPIRICAL_PERTURBATION_SEED,
            embedded_h_start=embedded,
        )
    else:
        truth_start = full.clip_to_bounds(
            codec_h.pack(full.truth_based_blocks(codec_h, pool, amplitude, "gaussian")),
            codec_h.bounds(),
        )
        empirical_start = full.clip_to_bounds(
            codec_h.pack(full.empirical_blocks(codec_h, pool)), codec_h.bounds()
        )
    if not full.has_nonzero_group_signature(codec_h, truth_start):
        raise AssertionError("Truth-based H start has a zero group signature")
    if not full.has_nonzero_group_signature(codec_h, empirical_start):
        raise AssertionError("Empirical H start has a zero group signature")
    starts = (
        (H_START_SOURCES[0], embedded.copy()),
        (H_START_SOURCES[1], truth_start),
        (H_START_SOURCES[2], empirical_start),
    )
    if len({theta.tobytes() for _, theta in starts}) != len(starts):
        raise AssertionError("The three prescribed H starts are not distinct")
    return starts


def optimize_h_start(task: HStartTask) -> HStartResult:
    force_single_numerical_library_thread()
    if any(
        os.environ.get(name) != "1" for name in NUMERICAL_LIBRARY_THREAD_VARIABLES
    ):
        raise AssertionError("Optimizer worker numerical threads are not limited")
    codec = full.ParameterCodec("H", task.store.basis.rank)
    start = np.asarray(task.initial_theta, dtype=float)
    if start.shape != (codec.size,) or not np.isfinite(start).all():
        raise ValueError("H optimizer received an invalid start")
    preconditioner_data = population.preconditioner_score_data(
        task.store, task.replicates, task.preconditioner_replicates
    )
    optimizer_scale = full.fisher_jacobi_scale(codec, start, preconditioner_data)
    scaled_start = start / optimizer_scale
    scaled_bounds = [
        (lower / scale, upper / scale)
        for (lower, upper), scale in zip(codec.bounds(), optimizer_scale)
    ]

    def scaled_objective(phi: np.ndarray) -> tuple[float, np.ndarray]:
        physical = np.asarray(phi, dtype=float) * optimizer_scale
        objective, gradient = population.chunked_objective_gradient(
            codec,
            physical,
            task.store,
            task.replicates,
            task.chunk_replicates,
        )
        return objective, gradient * optimizer_scale

    result = minimize(
        scaled_objective,
        scaled_start,
        jac=True,
        method="L-BFGS-B",
        bounds=scaled_bounds,
        options={
            "maxiter": int(task.max_iterations),
            "maxls": 40,
            "ftol": 1.0e-14,
            "gtol": 5.0e-7,
            "maxcor": 20,
        },
    )
    theta = np.asarray(result.x, dtype=float) * optimizer_scale
    if theta.shape != (codec.size,) or not np.isfinite(theta).all():
        raise FloatingPointError("H optimizer returned invalid parameters")
    objective, gradient = population.chunked_objective_gradient(
        codec, theta, task.store, task.replicates, task.chunk_replicates
    )
    if not np.isclose(objective, float(result.fun), atol=1.0e-10, rtol=1.0e-10):
        raise AssertionError("H optimizer objective did not replay")
    projected = full.projected_gradient(theta, gradient, codec.bounds())
    scaled_projected = full.projected_gradient(
        np.asarray(result.x, dtype=float),
        gradient * optimizer_scale,
        scaled_bounds,
    )
    projected_norm = float(np.max(np.abs(projected)))
    at_iteration_cap = int(getattr(result, "nit", -1)) >= int(task.max_iterations)
    solution = full.OptimizerSolution(
        theta=theta.copy(),
        objective=float(objective),
        optimizer_success=bool(result.success),
        converged=bool(result.success)
        and not at_iteration_cap
        and projected_norm <= full.PROJECTED_GRADIENT_TOLERANCE,
        iterations=int(getattr(result, "nit", -1)),
        evaluations=int(getattr(result, "nfev", -1)),
        gradient_infinity_norm=float(np.max(np.abs(gradient))),
        projected_gradient_infinity_norm=projected_norm,
        scaled_projected_gradient_infinity_norm=float(
            np.max(np.abs(scaled_projected))
        ),
        projected_gradient_block_infinity_norms={
            name: float(np.max(np.abs(projected[parameter_slice])))
            for name, parameter_slice in codec.slices.items()
        },
        optimizer_scale_minimum=float(optimizer_scale.min()),
        optimizer_scale_maximum=float(optimizer_scale.max()),
        numerical_boundary_diagnostics=full.numerical_boundary_diagnostics(
            codec, theta, preconditioner_data
        ),
        at_iteration_cap=at_iteration_cap,
        message=str(result.message),
    )
    return HStartResult(task.start_index, task.source, solution)


def run_h_start_tasks(
    tasks: Sequence[HStartTask], workers: int
) -> tuple[HStartResult, ...]:
    if not tasks:
        raise ValueError("At least one H optimizer task is required")
    if len({task.start_index for task in tasks}) != len(tasks):
        raise ValueError("H optimizer start indices must be unique")
    if workers < 1:
        raise ValueError("H optimizer worker count must be positive")
    results: list[HStartResult | None] = [None] * len(tasks)

    def record(position: int, result: HStartResult) -> None:
        expected = tasks[position]
        if result.start_index != expected.start_index or result.source != expected.source:
            raise AssertionError("H optimizer returned the wrong start identity")
        results[position] = result
        print(
            f"R={expected.replicates} H start {position + 1}/{len(tasks)} "
            f"({result.source}) completed: objective={result.solution.objective:.12g}, "
            f"KKT_converged={result.solution.converged}",
            file=sys.stderr,
            flush=True,
        )

    effective_workers = min(int(workers), len(tasks))
    if effective_workers == 1:
        for position, task in enumerate(tasks):
            record(position, optimize_h_start(task))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
            initializer=force_single_numerical_library_thread,
        ) as executor:
            pending = {
                executor.submit(optimize_h_start, task): position
                for position, task in enumerate(tasks)
            }
            for future in as_completed(pending):
                position = pending[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"H optimizer start {position + 1}/{len(tasks)} failed"
                    ) from exc
                record(position, result)
    if any(result is None for result in results):
        raise AssertionError("An H optimizer start result was not collected")
    return tuple(result for result in results if result is not None)


def boundary_inactive(solution: full.OptimizerSolution) -> bool:
    diagnostics = solution.numerical_boundary_diagnostics
    return bool(
        diagnostics["active_parameter_bound_count"] == 0
        and not diagnostics["sigma_at_numerical_bound"]
        and not diagnostics["hard_predictor_clipping"]
    )


def optimizer_summary(
    results: Sequence[HStartResult], embedded_c_objective: float
) -> dict[str, object]:
    converged = sorted(
        (result for result in results if result.solution.converged),
        key=lambda result: result.solution.objective,
    )
    training_rank = {
        result.start_index: rank + 1 for rank, result in enumerate(converged)
    }
    starts: list[dict[str, object]] = []
    for result in results:
        solution = result.solution
        starts.append(
            {
                "scheduled_start_index": result.start_index,
                "source": result.source,
                "KKT_converged_training_rank": training_rank.get(result.start_index),
                "training_negative_mean_log_density": solution.objective,
                "optimizer_success": solution.optimizer_success,
                "physical_KKT_converged": solution.converged,
                "iterations": solution.iterations,
                "evaluations": solution.evaluations,
                "at_iteration_cap": solution.at_iteration_cap,
                "physical_projected_gradient_infinity_norm": (
                    solution.projected_gradient_infinity_norm
                ),
                "physical_projected_gradient_block_infinity_norms": (
                    solution.projected_gradient_block_infinity_norms
                ),
                "scaled_projected_gradient_infinity_norm": (
                    solution.scaled_projected_gradient_infinity_norm
                ),
                "optimizer_scale_range": [
                    solution.optimizer_scale_minimum,
                    solution.optimizer_scale_maximum,
                ],
                "boundary_diagnostics": solution.numerical_boundary_diagnostics,
                "message": solution.message,
            }
        )
    return {
        "family": "H",
        "joint_optimization_of_all_H_parameter_blocks": True,
        "parameter_count": len(results[0].solution.theta),
        "scheduled_start_sources": list(H_START_SOURCES),
        "scheduled_starts": len(results),
        "physical_KKT_converged_starts": len(converged),
        "embedded_C_training_negative_mean_log_density": embedded_c_objective,
        "best_H_training_negative_mean_log_density": (
            None if not converged else converged[0].solution.objective
        ),
        "best_H_objective_no_worse_than_embedded_C": (
            False
            if not converged
            else converged[0].solution.objective <= embedded_c_objective
        ),
        "starts": starts,
    }


def single_solution_summary(
    source: str, solution: full.OptimizerSolution | None
) -> dict[str, object] | None:
    if solution is None:
        return None
    return {
        "source": source,
        "training_negative_mean_log_density": solution.objective,
        "optimizer_success": solution.optimizer_success,
        "physical_KKT_converged": solution.converged,
        "iterations": solution.iterations,
        "evaluations": solution.evaluations,
        "at_iteration_cap": solution.at_iteration_cap,
        "physical_projected_gradient_infinity_norm": (
            solution.projected_gradient_infinity_norm
        ),
        "physical_projected_gradient_block_infinity_norms": (
            solution.projected_gradient_block_infinity_norms
        ),
        "scaled_projected_gradient_infinity_norm": (
            solution.scaled_projected_gradient_infinity_norm
        ),
        "optimizer_scale_range": [
            solution.optimizer_scale_minimum,
            solution.optimizer_scale_maximum,
        ],
        "boundary_diagnostics": solution.numerical_boundary_diagnostics,
        "message": solution.message,
    }


def verification_chunk(task: VerificationTask) -> VerificationChunk:
    force_single_numerical_library_thread()
    indices = np.asarray(task.indices, dtype=int)
    truth = population.truth_template(
        task.layout, indices, task.amplitude, task.seed
    )
    cell_weights = full.equal_unit_weights(task.layout, indices)
    codec_c = full.ParameterCodec("C", task.basis.rank)
    codec_h = full.ParameterCodec("H", task.basis.rank)
    replicate_count = task.stop - task.start
    truth_scores = np.zeros(replicate_count, dtype=float)
    c_scores = np.zeros(replicate_count, dtype=float)
    h_scores = np.zeros((len(task.h_thetas), replicate_count), dtype=float)
    for local, replicate in enumerate(range(task.start, task.stop)):
        for group in range(2):
            draw = population.conditional_group_draw(
                truth, group, task.seed, task.stream, replicate
            )
            data = population.score_data_from_draw(
                task.layout, task.basis, indices, draw
            )
            exact = population.truth_scores_from_draw(draw, truth)
            weights = cell_weights * truth.group_probabilities[:, group]
            truth_scores[local] += float(np.sum(weights * exact))
            c_cell_scores = np.asarray(
                full.family_objective_gradient(
                    codec_c, task.c_theta, data, return_scores=True
                ),
                dtype=float,
            )
            c_scores[local] += float(np.sum(weights * c_cell_scores))
            for h_index, theta in enumerate(task.h_thetas):
                h_cell_scores = np.asarray(
                    full.family_objective_gradient(
                        codec_h, theta, data, return_scores=True
                    ),
                    dtype=float,
                )
                h_scores[h_index, local] += float(np.sum(weights * h_cell_scores))
    if not (
        np.isfinite(truth_scores).all()
        and np.isfinite(c_scores).all()
        and np.isfinite(h_scores).all()
    ):
        raise FloatingPointError("Fresh paired verification produced a nonfinite score")
    return VerificationChunk(
        start=task.start,
        stop=task.stop,
        truth_log_density=truth_scores,
        c_log_density=c_scores,
        h_log_density=h_scores,
    )


def contiguous_ranges(total: int, workers: int) -> tuple[tuple[int, int], ...]:
    workers = min(max(int(workers), 1), int(total))
    boundaries = np.linspace(0, total, workers + 1, dtype=int)
    return tuple(
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(workers)
        if boundaries[index] < boundaries[index + 1]
    )


def paired_summary(values: np.ndarray) -> dict[str, object]:
    mean, mcse = population.mean_mcse(np.asarray(values, dtype=float))
    return {"replicates": int(len(values)), "mean": mean, "mcse": mcse}


def run_fresh_paired_verification(
    *,
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    amplitude: float,
    seed: int,
    stream: int,
    replicates: int,
    c_theta: np.ndarray,
    h_solutions: Sequence[full.OptimizerSolution],
    workers: int,
    polish_pair_indices: tuple[int, int] | None = None,
    training_selected_count: int = 2,
) -> dict[str, object]:
    if not h_solutions:
        return {
            "status": "not_run_no_KKT_converged_H_candidate",
            "stream": stream,
            "replicates": 0,
            "contrasts": {},
            "maximum_monitored_contrast_MCSE": None,
            "truth_minus_H_one_sided_95_upper": None,
            "polishing_fresh_score_gain": None,
        }
    selected = tuple(solution.theta.copy() for solution in h_solutions)
    if not (1 <= training_selected_count <= min(2, len(selected))):
        raise ValueError("Training-selected H candidate count is invalid")
    if polish_pair_indices is not None:
        pre_polish_index, polished_index = polish_pair_indices
        if not (
            0 <= pre_polish_index < len(selected)
            and 0 <= polished_index < len(selected)
        ):
            raise ValueError("Polish-pair verification indices are out of range")
    tasks = tuple(
        VerificationTask(
            start=start,
            stop=stop,
            layout=layout,
            basis=basis,
            indices=np.asarray(indices, dtype=int),
            amplitude=float(amplitude),
            seed=int(seed),
            stream=int(stream),
            c_theta=np.asarray(c_theta, dtype=float),
            h_thetas=selected,
        )
        for start, stop in contiguous_ranges(replicates, workers)
    )
    if len(tasks) == 1:
        chunks = (verification_chunk(tasks[0]),)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(tasks),
            mp_context=context,
            initializer=force_single_numerical_library_thread,
        ) as executor:
            chunks = tuple(executor.map(verification_chunk, tasks))
    chunks = tuple(sorted(chunks, key=lambda chunk: chunk.start))
    cursor = 0
    for chunk in chunks:
        if chunk.start != cursor or chunk.stop <= chunk.start:
            raise AssertionError("Fresh verification chunks are not contiguous")
        cursor = chunk.stop
    if cursor != replicates:
        raise AssertionError("Fresh verification did not complete every replicate")
    truth = np.concatenate([chunk.truth_log_density for chunk in chunks])
    c_score = np.concatenate([chunk.c_log_density for chunk in chunks])
    h_score = np.concatenate([chunk.h_log_density for chunk in chunks], axis=1)
    if truth.shape != (replicates,) or c_score.shape != (replicates,):
        raise AssertionError("Fresh verification score vectors have the wrong shape")
    if h_score.shape != (len(selected), replicates):
        raise AssertionError("Fresh verification H score matrix has the wrong shape")

    contrast_values: dict[str, np.ndarray] = {
        "truth_minus_C": truth - c_score,
        "H_best_minus_C": h_score[0] - c_score,
        "truth_minus_H_best": truth - h_score[0],
    }
    if training_selected_count >= 2:
        contrast_values["H_best_minus_H_second"] = h_score[0] - h_score[1]
    if polish_pair_indices is not None:
        pre_polish_index, polished_index = polish_pair_indices
        contrast_values["H_polished_minus_H_pre_polish"] = (
            h_score[polished_index] - h_score[pre_polish_index]
        )
    contrasts = {
        name: paired_summary(values) for name, values in contrast_values.items()
    }
    maximum_mcse = max(float(value["mcse"]) for value in contrasts.values())
    truth_h = contrasts["truth_minus_H_best"]
    truth_h_upper = float(
        float(truth_h["mean"])
        + student_t.ppf(0.95, int(truth_h["replicates"]) - 1)
        * float(truth_h["mcse"])
    )
    polishing_gain: dict[str, object] | None = None
    if "H_polished_minus_H_pre_polish" in contrasts:
        polish_contrast = contrasts["H_polished_minus_H_pre_polish"]
        polishing_gain = {
            "mean_polished_minus_pre_polish": float(polish_contrast["mean"]),
            "positive_fresh_score_gain": max(
                0.0, float(polish_contrast["mean"])
            ),
            "mcse": float(polish_contrast["mcse"]),
            "replicates": int(polish_contrast["replicates"]),
        }
    return {
        "status": "completed",
        "stream": int(stream),
        "replicates": int(replicates),
        "group_integration": "exact_two_stratum_probability_weighting_per_metadata_row",
        "candidate_selection": "fit_stream_training_objective_only",
        "fresh_scores_used_for_parameter_selection": False,
        "paired_on_every_support_and_value_noise_draw": True,
        "H_candidates_scored": len(selected),
        "training_selected_H_candidates_scored": training_selected_count,
        "contrasts": contrasts,
        "maximum_monitored_contrast_MCSE": maximum_mcse,
        "truth_minus_H_one_sided_95_upper": truth_h_upper,
        "polishing_fresh_score_gain": polishing_gain,
    }


def rank_h_terminal_candidates(
    main_ranked: Sequence[full.OptimizerSolution],
    polished_solution: full.OptimizerSolution | None,
) -> tuple[population.RankedTerminalCandidate, ...]:
    candidates: list[tuple[str, full.OptimizerSolution]] = []
    if polished_solution is not None and polished_solution.converged:
        candidates.append(("polished", polished_solution))
    if main_ranked:
        candidates.append(("pre_polish_best", main_ranked[0]))
    if len(main_ranked) >= 2:
        candidates.append(("original_second", main_ranked[1]))
    return population.rank_terminal_candidates(candidates)


def verification_solution_set(
    terminal_candidates: Sequence[population.RankedTerminalCandidate],
    pre_polish: full.OptimizerSolution | None,
    polished: full.OptimizerSolution | None,
) -> tuple[tuple[full.OptimizerSolution, ...], tuple[int, int] | None]:
    solutions: list[full.OptimizerSolution] = [
        candidate.solution for candidate in terminal_candidates[:2]
    ]

    def append_or_find(solution: full.OptimizerSolution) -> int:
        for index, existing in enumerate(solutions):
            if population.parameters_equivalent(solution, existing):
                return index
        solutions.append(solution)
        return len(solutions) - 1

    polish_pair: tuple[int, int] | None = None
    if pre_polish is not None and polished is not None:
        polish_pair = (append_or_find(pre_polish), append_or_find(polished))
    return tuple(solutions), polish_pair


def terminal_candidate_summary(
    candidates: Sequence[population.RankedTerminalCandidate],
) -> list[dict[str, object]]:
    return [
        {
            "fit_objective_rank": rank + 1,
            "sources": list(candidate.sources),
            "training_negative_mean_log_density": candidate.solution.objective,
            "physical_KKT_converged": candidate.solution.converged,
            "at_iteration_cap": candidate.solution.at_iteration_cap,
            "numerical_boundaries_inactive": boundary_inactive(
                candidate.solution
            ),
            "fresh_verification_selected": rank < 2,
        }
        for rank, candidate in enumerate(candidates)
    ]


def assert_h_streams_are_fresh(
    streams: population.StreamIDs,
    verification_stream: int,
    exact_anchor_stream: int = H_EXACT_ANCHOR_STREAM,
) -> None:
    if not (0 <= verification_stream <= MAX_EXACT_FLOAT_INTEGER):
        raise ValueError("Verification stream must be a nonnegative exact integer")
    c_streams = {
        streams.fit,
        streams.validation,
        streams.terminal,
        streams.anchor,
        population.TRUTH_TEMPLATE_STREAM,
    }
    if not (0 <= exact_anchor_stream <= MAX_EXACT_FLOAT_INTEGER):
        raise ValueError("Exact-anchor stream must be a nonnegative exact integer")
    if exact_anchor_stream in c_streams:
        raise ValueError("H exact-anchor stream collides with a fitted-C stream")
    if verification_stream in c_streams:
        raise ValueError("H verification stream must be distinct from every fitted-C stream")
    if verification_stream == exact_anchor_stream:
        raise ValueError("H verification and exact-anchor streams must be distinct")


def self_test(
    layout: full.core.MetadataLayout,
    basis: full.DesignBasis,
    indices: np.ndarray,
    seed: int,
) -> dict[str, object]:
    if os.environ.get("SLURM_JOB_ID") and population.require_slurm_cpus_per_task() < 2:
        raise RuntimeError(
            "The spawned-equivalence self-test requires at least two CPUs"
        )
    if any(
        os.environ.get(name) != "1"
        for name in NUMERICAL_LIBRARY_THREAD_VARIABLES
    ):
        raise AssertionError("Numerical-library thread limits are not enforced")
    roundoff_probe = basis.transform.copy()
    roundoff_probe[0, 0] += 0.5 * population.MODEL_STATE_BASIS_ATOL
    if not population.basis_transform_matches_reconstruction(
        roundoff_probe, basis.transform
    ):
        raise AssertionError("Checkpoint-basis roundoff tolerance is too strict")
    above_tolerance_probe = basis.transform.copy()
    above_tolerance_probe[0, 0] += 2.0 * population.MODEL_STATE_BASIS_ATOL
    if population.basis_transform_matches_reconstruction(
        above_tolerance_probe, basis.transform
    ):
        raise AssertionError("Checkpoint-basis tolerance accepts material changes")
    sign_probe = basis.transform.copy()
    sign_probe[:, 0] *= -1.0
    permutation_probe = basis.transform.copy()
    permutation_probe[:, [0, 1]] = permutation_probe[:, [1, 0]]
    if population.basis_transform_matches_reconstruction(
        sign_probe, basis.transform
    ) or population.basis_transform_matches_reconstruction(
        permutation_probe, basis.transform
    ):
        raise AssertionError("Checkpoint-basis validation accepts a changed basis")
    amplitude = 0.7
    with tempfile.TemporaryDirectory(prefix="ep11-population-h-self-test-") as temporary:
        store = population.SAAMemmapStore.create(
            Path(temporary),
            layout,
            basis,
            indices,
            amplitude=amplitude,
            maximum_replicates=2,
            seed=seed,
            stream=7101,
        )
        data = store.score_data(0, 1, include_outcomes=True)
        pool = full.PopulationPool(
            data=data,
            single_design=basis.apply(layout.p_design[indices]),
            truth=store.truth,
            replicates=1,
        )
        codec_c = full.ParameterCodec("C", basis.rank)
        codec_h = full.ParameterCodec("H", basis.rank)
        c_theta = full.clip_to_bounds(
            codec_c.pack(
                full.truth_based_blocks(codec_c, pool, amplitude, "adjacent_node_scale")
            ),
            codec_c.bounds(),
        )
        starts = h_initial_starts(store, c_theta, amplitude, 1)
        if tuple(source for source, _ in starts) != H_START_SOURCES:
            raise AssertionError("H start ordering changed")

        embedded = starts[0][1]
        arbitrary_gate_blocks = {
            name: value.copy() for name, value in codec_h.unpack(embedded).items()
        }
        arbitrary_gate_blocks["gate_beta"] = np.array([0.4, -0.3, 0.2, -0.1])
        arbitrary_gate = codec_h.pack(arbitrary_gate_blocks)
        c_scores = np.asarray(
            full.family_objective_gradient(
                codec_c, c_theta, data, return_scores=True
            ),
            dtype=float,
        )
        h_scores = np.asarray(
            full.family_objective_gradient(
                codec_h, arbitrary_gate, data, return_scores=True
            ),
            dtype=float,
        )
        np.testing.assert_allclose(c_scores, h_scores, atol=1.0e-10, rtol=0.0)
        _, nested_gradient = full.family_objective_gradient(
            codec_h, arbitrary_gate, data
        )
        np.testing.assert_allclose(
            nested_gradient[codec_h.slices["gate_beta"]],
            0.0,
            atol=1.0e-10,
            rtol=0.0,
        )
        c_objective, _ = population.chunked_objective_gradient(
            codec_c, c_theta, store, 2, 1
        )
        h_objective, _ = population.chunked_objective_gradient(
            codec_h, embedded, store, 2, 1
        )
        np.testing.assert_allclose(c_objective, h_objective, atol=1.0e-12, rtol=0.0)

        theta_h = starts[1][1]
        vectorized, responsibilities = full.family_objective_gradient(
            codec_h,
            theta_h,
            data,
            return_scores=True,
            return_responsibilities=True,
        )
        enumerated, enumerated_responsibilities = full.explicit_h_2_by_7_enumeration(
            codec_h, theta_h, data
        )
        np.testing.assert_allclose(vectorized, enumerated, atol=1.0e-10, rtol=0.0)
        np.testing.assert_allclose(
            responsibilities, enumerated_responsibilities, atol=1.0e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            responsibilities.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0
        )

        objective, gradient = full.family_objective_gradient(codec_h, theta_h, data)
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise AssertionError("H objective/gradient self-test is nonfinite")
        epsilon = 1.0e-6
        block_probe_errors: dict[str, float] = {}
        for name, parameter_slice in codec_h.slices.items():
            index = parameter_slice.start + (parameter_slice.stop - parameter_slice.start) // 2
            plus = theta_h.copy()
            minus = theta_h.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            plus_value = full.family_objective_gradient(codec_h, plus, data)[0]
            minus_value = full.family_objective_gradient(codec_h, minus, data)[0]
            numerical = (plus_value - minus_value) / (2.0 * epsilon)
            np.testing.assert_allclose(
                numerical, gradient[index], atol=3.0e-5, rtol=1.0e-3
            )
            block_probe_errors[name] = abs(float(numerical - gradient[index]))

        # Exercise the actual H memmap/SciPy path for one iteration on a
        # deterministic rank-one, three-row problem.  This is dispatch and
        # replay coverage, not a convergence or recovery fit.
        small_indices = np.asarray(indices[: min(3, len(indices))], dtype=int)
        source_design = np.asarray(layout.p_design[small_indices], dtype=float)
        restricted_design = np.zeros_like(source_design)
        selected_column = int(np.argmax(np.linalg.norm(source_design, axis=0)))
        restricted_design[:, selected_column] = source_design[:, selected_column]
        small_basis = full.DesignBasis.from_support(
            restricted_design,
            full.equal_unit_weights(layout, small_indices),
        )
        small_store = population.SAAMemmapStore.create(
            Path(temporary) / "spawn-equivalence",
            layout,
            small_basis,
            small_indices,
            amplitude=amplitude,
            maximum_replicates=1,
            seed=seed,
            stream=7301,
        )
        small_data = small_store.score_data(0, 1, include_outcomes=True)
        small_pool = full.PopulationPool(
            data=small_data,
            single_design=small_basis.apply(layout.p_design[small_indices]),
            truth=small_store.truth,
            replicates=1,
        )
        small_codec_c = full.ParameterCodec("C", small_basis.rank)
        small_codec_h = full.ParameterCodec("H", small_basis.rank)
        small_c_theta = full.clip_to_bounds(
            small_codec_c.pack(
                full.truth_based_blocks(
                    small_codec_c, small_pool, amplitude, "gaussian"
                )
            ),
            small_codec_c.bounds(),
        )
        small_h_truth = full.clip_to_bounds(
            small_codec_h.pack(
                full.truth_based_blocks(
                    small_codec_h, small_pool, amplitude, "gaussian"
                )
            ),
            small_codec_h.bounds(),
        )
        small_h_embedded = full.embed_c_theta_in_h(
            small_c_theta, small_basis.rank
        )
        equivalence_tasks = tuple(
            HStartTask(
                start_index=start_index,
                source=source,
                store=small_store,
                initial_theta=start.copy(),
                replicates=1,
                chunk_replicates=1,
                preconditioner_replicates=1,
                max_iterations=1,
            )
            for start_index, source, start in (
                (101, "self_test_truth", small_h_truth),
                (202, "self_test_embedded", small_h_embedded),
            )
        )
        initial_snapshots = tuple(
            task.initial_theta.copy() for task in equivalence_tasks
        )
        sequential_results = run_h_start_tasks(equivalence_tasks, workers=1)
        spawned_results = run_h_start_tasks(equivalence_tasks, workers=2)
        for task, snapshot in zip(equivalence_tasks, initial_snapshots):
            np.testing.assert_array_equal(task.initial_theta, snapshot)
        optimizer_equivalence = [
            population.optimizer_solution_equivalence(
                sequential.solution, spawned.solution
            )
            for sequential, spawned in zip(sequential_results, spawned_results)
        ]
        if not all(item["pass"] for item in optimizer_equivalence):
            raise AssertionError(
                "Sequential and spawned H optimizer results differ"
            )
        parallel_start_equivalence = {
            "pass": True,
            "multiprocessing_context": "spawn",
            "starts": len(equivalence_tasks),
            "workers": 2,
            "metadata_rows": len(small_indices),
            "design_rank": small_basis.rank,
            "parameter_count": small_codec_h.size,
            "SAA_replicates": 1,
            "optimizer_iterations": 1,
            "comparisons": optimizer_equivalence,
            "scope_limit": (
                "reduced_deterministic_dispatch_and_replay_probe_not_a_fit"
            ),
        }

        sequential_verification = run_fresh_paired_verification(
            layout=layout,
            basis=small_basis,
            indices=small_indices,
            amplitude=amplitude,
            seed=seed,
            stream=7302,
            replicates=2,
            c_theta=small_c_theta,
            h_solutions=tuple(result.solution for result in sequential_results),
            workers=1,
        )
        spawned_verification = run_fresh_paired_verification(
            layout=layout,
            basis=small_basis,
            indices=small_indices,
            amplitude=amplitude,
            seed=seed,
            stream=7302,
            replicates=2,
            c_theta=small_c_theta,
            h_solutions=tuple(result.solution for result in sequential_results),
            workers=2,
        )
        if not population.nested_diagnostics_match(
            sequential_verification, spawned_verification, tolerance=1.0e-12
        ):
            raise AssertionError(
                "Serial and spawned fresh-verification summaries differ"
            )
        parallel_verification_equivalence = {
            "pass": True,
            "multiprocessing_context": "spawn",
            "workers": 2,
            "replicates": 2,
            "absolute_tolerance": 1.0e-12,
            "scope_limit": "reduced_deterministic_scoring_probe_not_a_fit",
        }

        streams = population.StreamIDs(7201, 7202, 7203, 7204)
        streams.validate()
        assert_h_streams_are_fresh(streams, 7205)
        collision_rejected = False
        try:
            assert_h_streams_are_fresh(streams, streams.validation)
        except ValueError:
            collision_rejected = True
        if not collision_rejected:
            raise AssertionError("A reused verification stream was accepted")
        false_gate_rejected = False
        try:
            require_all_true_gates({"probe": False}, ("probe",), "probe_gates")
        except ValueError:
            false_gate_rejected = True
        if not false_gate_rejected:
            raise AssertionError("A false upstream pass gate was accepted")
        valid_c_start_schedule = {
            "policy": (
                "first_rung_fresh_override_then_later_fresh_plus_previous_best_warm_v1"
            ),
            "first_level_fresh_starts_requested": 3,
            "later_level_fresh_starts_requested": 2,
            "first_level_scheduled_starts": 3,
            "later_level_scheduled_starts": 3,
            "later_levels_add_previous_level_best_as_one_extra_warm_start": True,
        }
        validate_c_start_schedule_contract(valid_c_start_schedule)
        legacy_c_start_schedule_rejected = False
        try:
            validate_c_start_schedule_contract(
                {
                    **valid_c_start_schedule,
                    "first_level_fresh_starts_requested": 2,
                    "first_level_scheduled_starts": 2,
                }
            )
        except ValueError:
            legacy_c_start_schedule_rejected = True
        if not legacy_c_start_schedule_rejected:
            raise AssertionError("The superseded two-start first rung was accepted")

    return {
        "diagnostic": "EP11_population_H_recovery_structural_self_test",
        "status": "pass",
        "uses_real_projection_outcomes": False,
        "C_parameter_count": codec_c.size,
        "H_parameter_count": codec_h.size,
        "maximum_absolute_finite_difference_error_by_H_block": block_probe_errors,
        "parallel_start_equivalence": parallel_start_equivalence,
        "parallel_verification_equivalence": parallel_verification_equivalence,
        "checks": {
            "exact_C_embedding_preserves_cellwise_scores_with_arbitrary_gate": True,
            "zero_signature_embedding_has_zero_gate_gradient": True,
            "chunked_C_and_embedded_H_objectives_match": True,
            "truth_based_H_start_has_nonzero_signature": True,
            "empirical_H_start_has_nonzero_signature": True,
            "vectorized_H_matches_explicit_two_by_seven_enumeration": True,
            "H_state_responsibilities_sum_to_one": True,
            "analytic_gradient_probe_passes_for_every_H_parameter_block": True,
            "fresh_verification_stream_collision_is_rejected": True,
            "sequential_and_spawned_H_optimizer_results_match": True,
            "serial_and_spawned_fresh_verification_match": True,
            "false_upstream_C_pass_gate_is_rejected": True,
            "revised_C_start_schedule_contract_is_enforced": True,
            "superseded_two_start_first_rung_is_rejected": True,
            "group_stratified_store_weights_sum_to_one": True,
            "checkpoint_basis_roundoff_within_tolerance_is_accepted": True,
            "checkpoint_basis_change_above_tolerance_is_rejected": True,
            "checkpoint_basis_sign_or_column_change_is_rejected": True,
        },
    }


def choose_status(acceptance: Mapping[str, bool]) -> str:
    if all(acceptance.values()):
        return "pass_population_H_optimizer_recovery_not_Panel_A_calibration"
    priorities = (
        (
            "fresh_exact_H_anchor_passes_1e_10",
            "technical_failure_fresh_exact_H_anchor",
        ),
        (
            "at_least_two_physical_KKT_converged_H_starts",
            "revise_population_H_insufficient_KKT_convergence",
        ),
        (
            "all_scheduled_H_starts_avoid_iteration_cap",
            "revise_population_H_optimizer_iteration_cap",
        ),
        (
            "fixed_start_H_polish_KKT_converged",
            "revise_population_H_polish_KKT_failure",
        ),
        (
            "fixed_start_H_polish_numerical_boundaries_inactive",
            "revise_population_H_polish_numerical_boundary_active",
        ),
        (
            "top_two_selected_H_candidates_numerical_boundaries_inactive",
            "revise_population_H_selected_candidate_boundary_active",
        ),
        (
            "top_two_selected_H_candidates_not_at_iteration_cap",
            "revise_population_H_selected_candidate_iteration_cap",
        ),
        (
            "best_H_training_objective_no_worse_than_embedded_C",
            "revise_population_H_nested_training_dominance_failed",
        ),
        (
            "fixed_start_polish_fresh_score_gain_MCSE_at_most_0_005",
            "revise_population_H_polish_gain_precision_failed",
        ),
        (
            "fixed_start_polish_fresh_score_gain_at_most_0_005",
            "revise_population_H_polish_fresh_gain_exceeds_threshold",
        ),
        (
            "fresh_paired_verification_maximum_MCSE_at_most_target",
            "revise_population_H_fresh_verification_precision_failed",
        ),
        (
            "fresh_best_two_H_scores_differ_below_threshold",
            "revise_population_H_fresh_start_stability_failed",
        ),
        (
            "fresh_truth_minus_H_one_sided_95_upper_below_threshold",
            "revise_population_H_truth_recovery_failed",
        ),
    )
    for gate, status in priorities:
        if not acceptance.get(gate, False):
            return status
    return "revise_population_H_optimizer_recovery_gates"


def main() -> None:
    episode_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=episode_root / "outputs/stage0/whole_cortex_ssptr_inventory.csv",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=episode_root / "outputs/stage0/allen_target_vocabulary.csv",
    )
    parser.add_argument("--c-summary", type=Path)
    parser.add_argument("--c-model-state", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--chunk-replicates", type=int, default=32)
    parser.add_argument("--preconditioner-replicates", type=int, default=32)
    parser.add_argument("--initialization-replicates", type=int, default=16)
    parser.add_argument("--start-workers", type=int, default=3)
    parser.add_argument("--verification-replicates", type=int, default=512)
    parser.add_argument("--verification-workers", type=int, default=3)
    parser.add_argument(
        "--start-policy",
        choices=(LEGACY_START_POLICY, EMPIRICAL_PERTURBATION_POLICY),
        default=LEGACY_START_POLICY,
    )
    parser.add_argument(
        "--verification-stream", type=int
    )
    parser.add_argument("--target-mcse", type=float, default=0.005)
    parser.add_argument("--start-stability", type=float, default=0.002)
    parser.add_argument("--truth-h-upper", type=float, default=0.010)
    parser.add_argument("--self-test-seed", type=int, default=20261001)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    summary_filename, default_verification_stream, h_exact_anchor_stream = (
        h_attempt_settings(args.start_policy)
    )
    if args.verification_stream is None:
        args.verification_stream = default_verification_stream
    if (
        args.start_policy == EMPIRICAL_PERTURBATION_POLICY
        and args.verification_stream != default_verification_stream
    ):
        raise ValueError("empirical_perturbation_v1 freezes verification stream 5602")
    if not (0 <= args.self_test_seed <= MAX_EXACT_FLOAT_INTEGER):
        raise ValueError("--self-test-seed must be a nonnegative exact integer")

    full.core.validate_vocabulary(args.vocabulary)
    layout = full.core.load_metadata(args.inventory)
    roles = full.panel_b_fixed_roles(layout)
    indices = np.asarray(roles.audit, dtype=int)
    basis = full.DesignBasis.from_support(
        layout.p_design[indices], full.equal_unit_weights(layout, indices)
    )
    if args.self_test:
        print(
            json.dumps(
                self_test(layout, basis, indices, args.self_test_seed),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Population-H recovery must run inside a Slurm allocation")
    slurm_cpus = population.require_slurm_cpus_per_task()
    if args.start_workers < 1 or args.start_workers > slurm_cpus:
        raise ValueError("--start-workers must be within SLURM_CPUS_PER_TASK")
    if args.verification_workers < 1 or args.verification_workers > slurm_cpus:
        raise ValueError("--verification-workers must be within SLURM_CPUS_PER_TASK")
    if args.c_summary is None or args.c_model_state is None or args.output_dir is None:
        raise ValueError("--c-summary, --c-model-state, and --output-dir are required")
    require_fresh_output_namespace(args.output_dir, summary_filename)
    if min(
        args.chunk_replicates,
        args.preconditioner_replicates,
        args.initialization_replicates,
    ) < 1:
        raise ValueError(
            "Chunk, initialization, and preconditioner counts must be positive"
        )
    if args.preconditioner_replicates > EXPECTED_FINAL_REPLICATES:
        raise ValueError("Preconditioner prefix cannot exceed R=8192")
    if args.initialization_replicates > EXPECTED_FINAL_REPLICATES:
        raise ValueError("Initialization prefix cannot exceed R=8192")
    if args.verification_replicates < 2:
        raise ValueError("Fresh paired verification needs at least two replicates")
    thresholds = (args.target_mcse, args.start_stability, args.truth_h_upper)
    if not np.isfinite(thresholds).all() or any(value <= 0.0 for value in thresholds):
        raise ValueError("Verification thresholds must be finite and positive")
    if args.target_mcse > 0.005:
        raise ValueError("--target-mcse may tighten but not exceed 0.005")
    if args.start_stability > 0.002:
        raise ValueError("--start-stability may tighten but not exceed 0.002")
    if args.truth_h_upper > 0.010:
        raise ValueError("--truth-h-upper may tighten but not exceed 0.010")

    config = bootstrap_c_checkpoint_config(args.c_model_state)
    if (
        args.start_policy == EMPIRICAL_PERTURBATION_POLICY
        and config.seed + EXPECTED_FINAL_REPLICATES != EMPIRICAL_PERTURBATION_SEED
    ):
        raise ValueError("empirical_perturbation_v1 requires the frozen C base seed")
    c_summary = validate_passing_c_summary(
        args.c_summary,
        args.c_model_state,
        config,
        layout,
        basis,
        indices,
        args.preconditioner_replicates,
        args.initialization_replicates,
    )
    assert_h_streams_are_fresh(
        config.streams, args.verification_stream, h_exact_anchor_stream
    )
    h_exact_anchor = population.exact_anchor_check(
        "H",
        config.amplitude,
        layout,
        basis,
        indices,
        config.seed,
        h_exact_anchor_stream,
        H_EXACT_ANCHOR_REPLICATES,
    )
    h_exact_anchor["stream"] = h_exact_anchor_stream
    h_exact_anchor["required_maximum_absolute_score_error"] = (
        H_EXACT_ANCHOR_TOLERANCE
    )
    h_exact_anchor_pass = bool(
        h_exact_anchor["pass"]
        and float(h_exact_anchor["maximum_absolute_score_error"])
        <= H_EXACT_ANCHOR_TOLERANCE
    )
    if not h_exact_anchor_pass:
        write_summary(
            args.output_dir,
            {
                "diagnostic": (
                    "EP11_fitted_H_group_stratified_population_optimizer_recovery"
                ),
                "status": "technical_failure_fresh_exact_H_anchor",
                "uses_real_projection_outcomes": False,
                "outcome_bearing_search_authorized": False,
                "fresh_exact_H_anchor": h_exact_anchor,
                "fitted_H_population_optimizer_recovery_completed": False,
                "terminal_population_calibration_pass": False,
                "Panel_A_population_calibration_complete": False,
            },
            summary_filename,
        )
        return
    if args.scratch_dir is None:
        scratch_root = os.environ.get("SCRATCH")
        if not scratch_root:
            raise ValueError("--scratch-dir is required when SCRATCH is unset")
        args.scratch_dir = Path(scratch_root) / (
            "br_autoresearch/episode11_projection_types_vs_gradients/"
            "population_h_recovery"
        )
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="ep11-population-h-", dir=args.scratch_dir
    ) as temporary:
        print(
            "Rebuilding the fitted-C R=8192 group-stratified SAA store",
            file=sys.stderr,
            flush=True,
        )
        store = population.SAAMemmapStore.create(
            Path(temporary),
            layout,
            basis,
            indices,
            amplitude=config.amplitude,
            maximum_replicates=EXPECTED_FINAL_REPLICATES,
            seed=config.seed,
            stream=config.streams.fit,
        )
        state = population.load_validate_c_model_state_checkpoint(
            args.c_model_state,
            layout=layout,
            basis=basis,
            indices=indices,
            amplitude=config.amplitude,
            requested_ladder=config.ladder,
            executed_rung_replicates=config.ladder,
            validation_replicates=config.validation_replicates,
            terminal_replicates=config.terminal_replicates,
            anchor_replicates=config.anchor_replicates,
            streams=config.streams,
            seed=config.seed,
            store=store,
            final_replicates=EXPECTED_FINAL_REPLICATES,
            chunk_replicates=args.chunk_replicates,
        )
        c_summary_checkpoint_crosslinks = (
            validate_c_summary_checkpoint_numeric_crosslinks(c_summary, state)
        )
        c_theta = np.asarray(state["frozen_theta"], dtype=float).copy()
        codec_c = full.ParameterCodec("C", basis.rank)
        codec_h = full.ParameterCodec("H", basis.rank)
        c_objective, _ = population.chunked_objective_gradient(
            codec_c,
            c_theta,
            store,
            EXPECTED_FINAL_REPLICATES,
            args.chunk_replicates,
        )
        embedded_theta = full.embed_c_theta_in_h(c_theta, basis.rank)
        embedded_objective, _ = population.chunked_objective_gradient(
            codec_h,
            embedded_theta,
            store,
            EXPECTED_FINAL_REPLICATES,
            args.chunk_replicates,
        )
        np.testing.assert_allclose(
            embedded_objective, c_objective, atol=1.0e-10, rtol=0.0
        )
        starts = h_initial_starts(
            store, c_theta, config.amplitude, args.initialization_replicates,
            args.start_policy,
        )
        tasks = tuple(
            HStartTask(
                start_index=index,
                source=source,
                store=store,
                initial_theta=theta,
                replicates=EXPECTED_FINAL_REPLICATES,
                chunk_replicates=args.chunk_replicates,
                preconditioner_replicates=args.preconditioner_replicates,
                max_iterations=H_MAIN_MAX_ITERATIONS,
            )
            for index, (source, theta) in enumerate(starts)
        )
        print(
            "Jointly optimizing all H blocks from three prespecified starts",
            file=sys.stderr,
            flush=True,
        )
        results = run_h_start_tasks(tasks, args.start_workers)
        fit = full.FamilyFit(
            family="H",
            amplitude=config.amplitude,
            solutions=tuple(result.solution for result in results),
            parameter_count=codec_h.size,
            design_rank=basis.rank,
            embedded_c_objective=embedded_objective,
        )
        main_ranked = fit.ranked_solutions
        pre_polish_best = main_ranked[0] if main_ranked else None
        polished_solution: full.OptimizerSolution | None = None
        if pre_polish_best is not None:
            print(
                "Polishing the best H solution from one fixed start",
                file=sys.stderr,
                flush=True,
            )
            polished_solution = optimize_h_start(
                HStartTask(
                    start_index=0,
                    source="fixed_start_polish",
                    store=store,
                    initial_theta=pre_polish_best.theta.copy(),
                    replicates=EXPECTED_FINAL_REPLICATES,
                    chunk_replicates=args.chunk_replicates,
                    preconditioner_replicates=args.preconditioner_replicates,
                    max_iterations=H_POLISH_MAX_ITERATIONS,
                )
            ).solution
        terminal_candidates = rank_h_terminal_candidates(
            main_ranked, polished_solution
        )
        verification_solutions, polish_pair_indices = verification_solution_set(
            terminal_candidates, pre_polish_best, polished_solution
        )
        print(
            f"Running fresh paired verification on stream {args.verification_stream}",
            file=sys.stderr,
            flush=True,
        )
        verification = run_fresh_paired_verification(
            layout=layout,
            basis=basis,
            indices=indices,
            amplitude=config.amplitude,
            seed=config.seed,
            stream=args.verification_stream,
            replicates=args.verification_replicates,
            c_theta=c_theta,
            h_solutions=verification_solutions,
            workers=args.verification_workers,
            polish_pair_indices=polish_pair_indices,
            training_selected_count=min(2, len(terminal_candidates)),
        )
        verification["fit_objective_selected_H_candidates"] = [
            {
                "rank": rank + 1,
                "sources": list(candidate.sources),
                "training_negative_mean_log_density": (
                    candidate.solution.objective
                ),
            }
            for rank, candidate in enumerate(terminal_candidates[:2])
        ]

    selected_terminal = terminal_candidates[:2]
    best = selected_terminal[0].solution if selected_terminal else None
    best_two = require_mapping(
        require_mapping(verification.get("contrasts"), "verification.contrasts").get(
            "H_best_minus_H_second", {}
        ),
        "verification H-best-minus-second",
    )
    raw_polishing_gain = verification.get("polishing_fresh_score_gain")
    polishing_gain = (
        raw_polishing_gain if isinstance(raw_polishing_gain, dict) else {}
    )
    maximum_mcse = verification.get("maximum_monitored_contrast_MCSE")
    truth_h_upper = verification.get("truth_minus_H_one_sided_95_upper")
    acceptance: dict[str, bool] = {
        "passing_population_C_summary_strictly_validated": True,
        "fitted_C_checkpoint_strictly_validated_and_training_objective_replayed": True,
        "accepted_C_summary_checkpoint_numeric_crosslinks_pass": bool(
            c_summary_checkpoint_crosslinks["pass"]
        ),
        "identical_R8192_group_stratified_training_store_rebuilt": True,
        "fresh_exact_H_anchor_passes_1e_10": h_exact_anchor_pass,
        "exact_fitted_C_embedding_training_objective_matches_C": bool(
            np.isclose(embedded_objective, c_objective, atol=1.0e-10, rtol=0.0)
        ),
        "three_prespecified_H_start_types_completed": (
            tuple(result.source for result in results) == H_START_SOURCES
        ),
        "truth_and_empirical_starts_have_nonzero_group_signatures": True,
        "all_H_parameters_jointly_optimized": True,
        "main_H_optimizer_budget_is_frozen_at_1200_iterations": (
            len(tasks) == len(H_START_SOURCES)
            and all(
                task.max_iterations == H_MAIN_MAX_ITERATIONS for task in tasks
            )
        ),
        "at_least_two_physical_KKT_converged_H_starts": len(main_ranked) >= 2,
        "all_scheduled_H_starts_avoid_iteration_cap": all(
            not result.solution.at_iteration_cap for result in results
        ),
        "fixed_start_H_polish_KKT_converged": bool(
            polished_solution is not None and polished_solution.converged
        ),
        "fixed_start_H_polish_numerical_boundaries_inactive": bool(
            polished_solution is not None
            and boundary_inactive(polished_solution)
        ),
        "top_two_selected_H_candidates_numerical_boundaries_inactive": (
            len(selected_terminal) >= 2
            and all(
                boundary_inactive(candidate.solution)
                for candidate in selected_terminal
            )
        ),
        "top_two_selected_H_candidates_not_at_iteration_cap": (
            len(selected_terminal) >= 2
            and all(
                not candidate.solution.at_iteration_cap
                for candidate in selected_terminal
            )
        ),
        "best_H_training_objective_no_worse_than_embedded_C": (
            best is not None and best.objective <= embedded_objective
        ),
        "fresh_paired_verification_completed": verification.get("status") == "completed",
        "fresh_paired_verification_maximum_MCSE_at_most_target": (
            isinstance(maximum_mcse, (int, float))
            and not isinstance(maximum_mcse, bool)
            and float(maximum_mcse) <= args.target_mcse
        ),
        "fixed_start_polish_fresh_score_gain_MCSE_at_most_0_005": (
            isinstance(polishing_gain.get("mcse"), (int, float))
            and not isinstance(polishing_gain.get("mcse"), bool)
            and float(polishing_gain["mcse"])
            <= H_POLISH_FRESH_GAIN_MCSE_LIMIT
        ),
        "fixed_start_polish_fresh_score_gain_at_most_0_005": (
            isinstance(
                polishing_gain.get("positive_fresh_score_gain"), (int, float)
            )
            and not isinstance(
                polishing_gain.get("positive_fresh_score_gain"), bool
            )
            and float(polishing_gain["positive_fresh_score_gain"])
            <= H_POLISH_FRESH_GAIN_LIMIT
        ),
        "fresh_best_two_H_scores_differ_below_threshold": (
            isinstance(best_two.get("mean"), (int, float))
            and not isinstance(best_two.get("mean"), bool)
            and abs(float(best_two["mean"])) < args.start_stability
        ),
        "fresh_truth_minus_H_one_sided_95_upper_below_threshold": (
            isinstance(truth_h_upper, (int, float))
            and not isinstance(truth_h_upper, bool)
            and float(truth_h_upper) < args.truth_h_upper
        ),
        "fresh_H_anchor_and_verification_streams_all_distinct": True,
    }
    status = choose_status(acceptance)
    passed = all(acceptance.values())
    summary: dict[str, object] = {
        "diagnostic": "EP11_fitted_H_group_stratified_population_optimizer_recovery",
        "status": status,
        "uses_real_projection_outcomes": False,
        "outcome_bearing_search_authorized": False,
        "scope": "implemented_smoke_family_on_fixed_provisional_A8_metadata_support",
        "D12_to_A8_transport_evaluated": False,
        "amplitude": config.amplitude,
        "amplitude_role": "diagnostic_positive_amplitude_not_population_calibrated_0.100_nat_fixture",
        "A8_units": len(set(layout.units[indices].tolist())),
        "A8_cells": len(indices),
        "P_design_rank": basis.rank,
        "C_parameter_count": codec_c.size,
        "H_parameter_count": codec_h.size,
        "training_SAA_replicates": EXPECTED_FINAL_REPLICATES,
        "training_group_integration": "exact_two_stratum_probability_weighting_per_metadata_row",
        "training_support_and_value_noise_integration": "same_nested_SAA_store_definition_as_validated_C",
        "C_input_validation": {
            "summary_status": EXPECTED_C_PASS_STATUS,
            "checkpoint_schema_version": population.MODEL_STATE_SCHEMA_VERSION,
            "summary_and_checkpoint_share_one_result_directory": True,
            "checkpoint_training_objective_replayed_on_rebuilt_store": True,
            "numeric_crosslinks": c_summary_checkpoint_crosslinks,
        },
        "streams": {
            **config.streams.as_dict(),
            "H_exact_anchor": h_exact_anchor_stream,
            "H_verification": args.verification_stream,
            "H_anchor_and_verification_distinct_from_each_other_and_all_C_and_truth_template_streams": True,
        },
        "fresh_exact_H_anchor": h_exact_anchor,
        "optimizer_contract": {
            "start_policy": args.start_policy,
            "empirical_perturbation_seed": (
                EMPIRICAL_PERTURBATION_SEED
                if args.start_policy == EMPIRICAL_PERTURBATION_POLICY else None
            ),
            "family": "H",
            "objective": "equal_unit_group_stratified_population_SAA_negative_mean_log_density",
            "joint_analytic_gradient_L_BFGS_B": True,
            "hard_group_labels_used_in_fitted_objective": False,
            "point_continuous_latent_estimates_used_in_objective": False,
            "C_coefficients_frozen_inside_H": False,
            "fit_stream_training_objective_alone_selects_H": True,
            "fresh_verification_selects_parameters": False,
            "main_maximum_iterations_per_start_frozen": H_MAIN_MAX_ITERATIONS,
            "fixed_start_polish_maximum_iterations_frozen": (
                H_POLISH_MAX_ITERATIONS
            ),
            "physical_projected_gradient_tolerance": full.PROJECTED_GRADIENT_TOLERANCE,
            "chunk_replicates": args.chunk_replicates,
            "preconditioner_prefix_replicates": args.preconditioner_replicates,
            "initialization_prefix_replicates": args.initialization_replicates,
            "start_workers": args.start_workers,
        },
        "optimizer": optimizer_summary(results, embedded_objective),
        "polishing": {
            "attempted": pre_polish_best is not None,
            "source": "fixed_start_from_pre_polish_best",
            "maximum_iterations_frozen": H_POLISH_MAX_ITERATIONS,
            "optimizer": single_solution_summary(
                "fixed_start_polish", polished_solution
            ),
            "fresh_score_gain": verification.get(
                "polishing_fresh_score_gain"
            ),
        },
        "terminal_candidate_selection": {
            "rule": (
                "deduplicate_polished_pre_polish_best_and_original_second_by_"
                "parameter_equivalence_then_rank_by_fit_stream_training_objective"
            ),
            "fresh_scores_used_for_selection": False,
            "ranking": terminal_candidate_summary(terminal_candidates),
        },
        "fresh_paired_verification": verification,
        "thresholds": {
            "maximum_monitored_verification_MCSE_nats_per_neuron": args.target_mcse,
            "best_two_H_fresh_score_difference_nats_per_neuron": args.start_stability,
            "truth_minus_H_one_sided_95_upper_nats_per_neuron": args.truth_h_upper,
            "embedded_C_training_objective_match_absolute_tolerance": 1.0e-10,
            "fresh_exact_H_anchor_maximum_absolute_error": (
                H_EXACT_ANCHOR_TOLERANCE
            ),
            "fixed_start_polish_fresh_score_gain_nats_per_neuron": (
                H_POLISH_FRESH_GAIN_LIMIT
            ),
            "fixed_start_polish_fresh_score_gain_MCSE_nats_per_neuron": (
                H_POLISH_FRESH_GAIN_MCSE_LIMIT
            ),
        },
        "acceptance": acceptance,
        "failed_gates": [name for name, value in acceptance.items() if not value],
        "fitted_H_population_optimizer_recovery_completed": passed,
        "terminal_population_calibration_pass": False,
        "Panel_A_population_calibration_complete": False,
        "remaining_before_Panel_A_completion": (
            [
                "complete_population_amplitude_calibration_to_the_0.100_nat_fixture",
                "complete_fitted_zero_amplitude_C_and_H_population_recovery",
                "apply_all_remaining_prespecified_Panel_A_acceptance_checks",
            ]
            if passed
            else [
                "revise_and_repeat_fitted_H_population_optimizer_recovery",
                "complete_population_amplitude_calibration_to_the_0.100_nat_fixture",
                "complete_fitted_zero_amplitude_C_and_H_population_recovery",
                "apply_all_remaining_prespecified_Panel_A_acceptance_checks",
            ]
        ),
    }
    write_summary(args.output_dir, summary, summary_filename)


if __name__ == "__main__":
    main()
