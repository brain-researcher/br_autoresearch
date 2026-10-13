"""Generated-only EP12 pilot for paired whole-type interval refinement.

This module is deliberately separate from the frozen qualification entry
point.  It runs the current generated-replicate implementation on a small,
predeclared design and publishes descriptive coverage summaries.  It neither
reads MaleCNS connectivity nor applies a scientific acceptance threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .policy import EpisodePolicy
from .runtime import EPISODE_ROOT
from .scientific_models import ScientificModelError
from .scientific_qualification import (
    QualificationCell,
    exact_binomial_interval,
    run_generated_replicate,
)
from .scientific_synthetic import SCENARIO_TRUTH


SCHEMA_VERSION = "ep12.interval_refinement_pilot.v1"
PILOT_ID = "interval_refinement_v1"
BLOCK_COUNT = 2
REPLICATES_PER_BLOCK = 72
WHOLE_TYPES_PER_REPLICATE = 8
BOOTSTRAP_REPLICATES = 1_999
INTEGRATION_DRAWS = 512
FOLDS = 2
CONFIDENCE_LEVEL = 0.95
FROZEN_MARGIN = 0.02
ENDPOINT_CONTROL_ALPHA = 0.05
SHARD_COUNT = 24
AGGREGATE_NAME = "aggregate_report.json"
DEFAULT_OUTPUT_ROOT = (
    EPISODE_ROOT
    / "outputs"
    / "scientific_qualification"
    / "pilots"
    / PILOT_ID
)


@dataclass(frozen=True)
class PilotConfiguration:
    """One immutable artifact identity for the shared interval-pilot design."""

    key: str
    schema_version: str
    pilot_id: str
    integration_draws: int

    def output_root(self, episode_root: Path = EPISODE_ROOT) -> Path:
        return (
            episode_root.resolve()
            / "outputs"
            / "scientific_qualification"
            / "pilots"
            / self.pilot_id
        )


V1_CONFIGURATION = PilotConfiguration(
    key="v1",
    schema_version=SCHEMA_VERSION,
    pilot_id=PILOT_ID,
    integration_draws=INTEGRATION_DRAWS,
)
V2_CONFIGURATION = PilotConfiguration(
    key="v2",
    schema_version="ep12.interval_refinement_pilot.v2",
    pilot_id="interval_refinement_v2_1024",
    integration_draws=1_024,
)
V3_CONFIGURATION = PilotConfiguration(
    key="v3",
    schema_version="ep12.interval_refinement_pilot.v3",
    pilot_id="interval_refinement_v3_2048",
    integration_draws=2_048,
)
CONFIGURATIONS = {
    V1_CONFIGURATION.key: V1_CONFIGURATION,
    V2_CONFIGURATION.key: V2_CONFIGURATION,
    V3_CONFIGURATION.key: V3_CONFIGURATION,
}
V2_OUTPUT_ROOT = V2_CONFIGURATION.output_root()
V3_OUTPUT_ROOT = V3_CONFIGURATION.output_root()


@dataclass(frozen=True)
class PilotCell:
    """One predeclared generated-data design cell."""

    base_cell_id: str
    scenario: str
    neurons_per_side: int
    count_total: int
    strength: float
    missing_fraction: float


def _cell_id(
    scenario: str,
    neurons_per_side: int,
    count_total: int,
    strength: float,
    missing_fraction: float,
) -> str:
    strength_id = str(strength).replace(".", "p")
    missing_id = str(missing_fraction).replace(".", "p")
    return (
        f"{scenario}__n{neurons_per_side}__N{count_total}"
        f"__s{strength_id}__m{missing_id}"
    )


def _pilot_cell(
    scenario: str,
    neurons_per_side: int,
    count_total: int,
    strength: float,
    missing_fraction: float,
) -> PilotCell:
    if scenario not in SCENARIO_TRUTH:
        raise ValueError(f"unknown generated scenario: {scenario}")
    return PilotCell(
        base_cell_id=_cell_id(
            scenario,
            neurons_per_side,
            count_total,
            strength,
            missing_fraction,
        ),
        scenario=scenario,
        neurons_per_side=neurons_per_side,
        count_total=count_total,
        strength=strength,
        missing_fraction=missing_fraction,
    )


# All nine generator families at the central support point, plus three
# prespecified low-support artifact/confound cells.
PILOT_CELLS = tuple(
    [
        _pilot_cell(scenario, 7, 100, 1.8, 0.1)
        for scenario in SCENARIO_TRUTH
    ]
    + [
        _pilot_cell("side_specific_endpoint_loss", 5, 10, 2.2, 0.0),
        _pilot_cell("anatomy_quality_confound", 5, 10, 2.2, 0.0),
        _pilot_cell("high_strength_or_influential_cells", 5, 10, 2.2, 0.0),
    ]
)

if len(PILOT_CELLS) * BLOCK_COUNT != SHARD_COUNT:  # pragma: no cover
    raise RuntimeError("interval pilot task inventory is internally inconsistent")


def _expected_output_root(
    episode_root: Path,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> Path:
    return configuration.output_root(episode_root)


def _validated_output_root(
    output_root: Path,
    episode_root: Path,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> Path:
    expected = _expected_output_root(episode_root, configuration)
    resolved = output_root.resolve()
    if resolved != expected:
        raise ValueError(
            "interval refinement artifacts must remain at "
            f"{expected}; received {resolved}"
        )
    return resolved


def _publish_json_exclusive(path: Path, value: Any) -> None:
    """Atomically publish JSON and reject every pre-existing destination."""

    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite interval pilot artifact: {path}"
            ) from exc
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def task_definition(task_id: int) -> tuple[int, int, PilotCell]:
    """Map one array index to its block and predeclared cell."""

    if isinstance(task_id, bool) or int(task_id) != task_id:
        raise ValueError("task_id must be an integer")
    task_id = int(task_id)
    if not 0 <= task_id < SHARD_COUNT:
        raise ValueError(f"task_id must lie in [0, {SHARD_COUNT - 1}]")
    block_index, cell_index = divmod(task_id, len(PILOT_CELLS))
    return block_index, cell_index, PILOT_CELLS[cell_index]


def _seed_cell_id(
    block_index: int,
    cell: PilotCell,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> str:
    """Bind all historical seed derivations to pilot and independent block."""

    return (
        f"{configuration.pilot_id}__block_{block_index + 1:02d}__"
        f"{cell.base_cell_id}"
    )


def _policy_margin(policy_path: Path) -> tuple[float, EpisodePolicy]:
    policy = EpisodePolicy.load(policy_path)
    margin = float(policy.meaningful_margin)
    if margin != FROZEN_MARGIN:
        raise ValueError(
            "interval refinement requires SEARCH_POLICY meaningful margin "
            f"{FROZEN_MARGIN}; observed {margin}"
        )
    return margin, policy


def shard_path(output_root: Path, task_id: int) -> Path:
    return output_root / "shards" / f"task_{task_id:02d}.json"


def _run_pilot_replicate(
    cell: QualificationCell,
    *,
    replicate_index: int,
    margin: float,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> dict[str, Any]:
    """Retain an interval-stage numerical failure without aborting its shard."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    try:
        return run_generated_replicate(
            cell,
            replicate_index=replicate_index,
            whole_types=WHOLE_TYPES_PER_REPLICATE,
            margin=margin,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            integration_draws=configuration.integration_draws,
            folds=FOLDS,
        )
    except ScientificModelError as exc:
        return {
            "cell_id": cell.cell_id,
            "scenario": cell.scenario,
            "truth_family": SCENARIO_TRUTH[cell.scenario],
            "replicate_index": replicate_index,
            "terminal": "numerical_failure",
            "numerical_failures": [
                f"replicate-interval:{type(exc).__name__}:{exc}"
            ],
            "type_records": [],
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        }


def run_shard(
    task_id: int,
    *,
    output_root: Path | None = None,
    episode_root: Path = EPISODE_ROOT,
    policy_path: Path | None = None,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> dict[str, Any]:
    """Run and publish one complete 72-replicate generated-data shard."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    requested_output_root = output_root or configuration.output_root(episode_root)
    output_root = _validated_output_root(
        requested_output_root, episode_root, configuration
    )
    block_index, cell_index, pilot_cell = task_definition(task_id)
    destination = shard_path(output_root, int(task_id))
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite interval pilot shard: {destination}"
        )

    resolved_policy_path = (
        policy_path or (episode_root.resolve() / "SEARCH_POLICY.yaml")
    ).resolve()
    margin, policy = _policy_margin(resolved_policy_path)
    seed_cell_id = _seed_cell_id(block_index, pilot_cell, configuration)
    cell = QualificationCell(
        cell_id=seed_cell_id,
        scenario=pilot_cell.scenario,
        neurons_per_side=pilot_cell.neurons_per_side,
        count_total=pilot_cell.count_total,
        strength=pilot_cell.strength,
        missing_fraction=pilot_cell.missing_fraction,
        acceptance_role="interval_refinement_pilot_only",
    )

    records = [
        _run_pilot_replicate(
            cell,
            replicate_index=replicate_index,
            margin=margin,
            configuration=configuration,
        )
        for replicate_index in range(REPLICATES_PER_BLOCK)
    ]
    artifact = {
        "schema_version": configuration.schema_version,
        "pilot_id": configuration.pilot_id,
        "artifact_kind": "generated_replicate_shard",
        "task_id": int(task_id),
        "block_index": block_index,
        "block_id": f"block_{block_index + 1:02d}",
        "cell_index": cell_index,
        "cell": asdict(pilot_cell),
        "seed_namespace_cell_id": seed_cell_id,
        "policy": {
            "path": str(resolved_policy_path),
            "policy_hash": policy.policy_hash,
            "meaningful_margin": margin,
        },
        "configuration": {
            "replicates": REPLICATES_PER_BLOCK,
            "whole_types_per_replicate": WHOLE_TYPES_PER_REPLICATE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "integration_draws": configuration.integration_draws,
            "folds": FOLDS,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "pilot_scope": "generated_data_only",
        "scientific_gate_applied": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "records": records,
        "computationally_complete": True,
        "runtime": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        },
    }
    _publish_json_exclusive(destination, artifact)
    return artifact


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid interval pilot artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"interval pilot artifact is not an object: {path}")
    return value


def _require_equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        raise ValueError(
            f"interval pilot artifact has invalid {field}: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _validate_record(
    record: Mapping[str, Any],
    *,
    replicate_index: int,
    seed_cell_id: str,
    pilot_cell: PilotCell,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> None:
    _require_equal(record.get("cell_id"), seed_cell_id, "record.cell_id")
    _require_equal(
        record.get("scenario"), pilot_cell.scenario, "record.scenario"
    )
    _require_equal(
        record.get("truth_family"),
        SCENARIO_TRUTH[pilot_cell.scenario],
        "record.truth_family",
    )
    _require_equal(
        record.get("replicate_index"), replicate_index, "record.replicate_index"
    )
    terminal = record.get("terminal")
    allowed_terminals = {
        "M_supported_before_full_search_null",
        "single_population_adequate_before_sensitivity_gate",
        "unresolved",
        "numerical_failure",
    }
    if terminal not in allowed_terminals:
        raise ValueError(f"unknown interval pilot terminal: {terminal!r}")
    type_records = record.get("type_records")
    if not isinstance(type_records, list):
        raise ValueError("interval pilot record lacks type_records")
    if terminal == "numerical_failure":
        failures = record.get("numerical_failures")
        if not isinstance(failures, list) or not failures:
            raise ValueError("numerical-failure record lacks failure details")
        if "intervals" in record:
            raise ValueError("numerical-failure record unexpectedly has intervals")
        return

    _require_equal(
        len(type_records),
        WHOLE_TYPES_PER_REPLICATE,
        "record.type_records count",
    )
    intervals = record.get("intervals")
    if not isinstance(intervals, Mapping):
        raise ValueError("estimable interval pilot record lacks intervals")
    _require_equal(
        intervals.get("labels"),
        ["left_to_right", "right_to_left", "aggregate"],
        "record.intervals.labels",
    )
    _require_equal(
        intervals.get("method"),
        "paired_whole_type_studentized_bootstrap_max_t",
        "record.intervals.method",
    )
    _require_equal(
        intervals.get("bootstrap_replicates"),
        BOOTSTRAP_REPLICATES,
        "record.intervals.bootstrap_replicates",
    )
    _require_equal(
        intervals.get("confidence_level"),
        CONFIDENCE_LEVEL,
        "record.intervals.confidence_level",
    )
    _require_equal(
        intervals.get("unit"),
        "whole_provider_type",
        "record.intervals.unit",
    )
    _require_equal(
        intervals.get("directions_resampled_together"),
        True,
        "record.intervals.directions_resampled_together",
    )
    _require_equal(
        intervals.get("margin"), FROZEN_MARGIN, "record.intervals.margin"
    )
    for field in ("estimate", "lower", "upper", "standard_error"):
        values = np.asarray(intervals.get(field), dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(f"record.intervals.{field} must have three finite values")
    lower = np.asarray(intervals["lower"], dtype=float)
    upper = np.asarray(intervals["upper"], dtype=float)
    if np.any(lower > upper):
        raise ValueError("interval pilot lower bound exceeds upper bound")

    indexed_deltas: dict[int, np.ndarray] = {}
    for type_record in type_records:
        if not isinstance(type_record, Mapping):
            raise ValueError("interval pilot type_record must be an object")
        type_index = type_record.get("synthetic_type_index")
        if (
            isinstance(type_index, bool)
            or not isinstance(type_index, int)
            or not 0 <= type_index < WHOLE_TYPES_PER_REPLICATE
            or type_index in indexed_deltas
        ):
            raise ValueError(
                "interval pilot type_record indices must be unique integers "
                f"in [0, {WHOLE_TYPES_PER_REPLICATE - 1}]"
            )
        try:
            deltas = np.asarray(type_record.get("deltas"), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("interval pilot type_record deltas must be numeric") from exc
        if deltas.shape != (2,) or not np.all(np.isfinite(deltas)):
            raise ValueError(
                "interval pilot type_record deltas must contain two finite directions"
            )
        indexed_deltas[type_index] = deltas
    expected_indices = set(range(WHOLE_TYPES_PER_REPLICATE))
    if set(indexed_deltas) != expected_indices:
        raise ValueError(
            "interval pilot record does not contain the complete whole-type index set"
        )
    delta_array = np.asarray(
        [indexed_deltas[index] for index in range(WHOLE_TYPES_PER_REPLICATE)],
        dtype=float,
    )

    controls = record.get("controls")
    if not isinstance(controls, Mapping):
        raise ValueError("estimable interval pilot record lacks controls")
    endpoint_p = controls.get("endpoint_side_permutation_p")
    if isinstance(endpoint_p, bool):
        raise ValueError("endpoint-side permutation p must be numeric")
    try:
        endpoint_p = float(endpoint_p)
    except (TypeError, ValueError) as exc:
        raise ValueError("endpoint-side permutation p must be numeric") from exc
    if not np.isfinite(endpoint_p) or not 0.0 <= endpoint_p <= 1.0:
        raise ValueError("endpoint-side permutation p must lie in [0, 1]")

    for field in (
        "endpoint_mass_control_pass",
        "influence_control_pass",
        "all_pass",
    ):
        if type(controls.get(field)) is not bool:
            raise ValueError(f"record.controls.{field} must be boolean")

    expected_endpoint_pass = endpoint_p > ENDPOINT_CONTROL_ALPHA
    _require_equal(
        controls["endpoint_mass_control_pass"],
        expected_endpoint_pass,
        "record.controls.endpoint_mass_control_pass",
    )
    expected_leave_one_out = np.asarray(
        [
            np.delete(delta_array, index, axis=0).mean(axis=0)
            for index in range(len(delta_array))
        ],
        dtype=float,
    )
    try:
        observed_leave_one_out = np.asarray(
            controls.get("leave_one_type_out_directional_gain"), dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "leave-one-type-out directional gain must be numeric"
        ) from exc
    if (
        observed_leave_one_out.shape != expected_leave_one_out.shape
        or not np.all(np.isfinite(observed_leave_one_out))
        or not np.allclose(
            observed_leave_one_out,
            expected_leave_one_out,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ValueError(
            "record.controls.leave_one_type_out_directional_gain is inconsistent "
            "with stored whole-type deltas"
        )
    direction_means = delta_array.mean(axis=0)
    potential_bilateral_support = bool(np.all(direction_means > FROZEN_MARGIN))
    expected_influence_pass = bool(
        not potential_bilateral_support
        or np.all(expected_leave_one_out > FROZEN_MARGIN)
    )
    _require_equal(
        controls["influence_control_pass"],
        expected_influence_pass,
        "record.controls.influence_control_pass",
    )
    expected_controls_pass = expected_endpoint_pass and expected_influence_pass
    _require_equal(
        controls["all_pass"],
        expected_controls_pass,
        "record.controls.all_pass",
    )
    expected_terminal = (
        "M_supported_before_full_search_null"
        if expected_controls_pass
        and lower[0] > FROZEN_MARGIN
        and lower[1] > FROZEN_MARGIN
        else (
            "single_population_adequate_before_sensitivity_gate"
            if expected_controls_pass
            and upper[0] < FROZEN_MARGIN
            and upper[1] < FROZEN_MARGIN
            else "unresolved"
        )
    )
    _require_equal(
        terminal,
        expected_terminal,
        "record.terminal_from_intervals_and_controls",
    )


def _validate_shard(
    artifact: Mapping[str, Any],
    task_id: int,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> tuple[int, PilotCell, list[Mapping[str, Any]]]:
    block_index, cell_index, pilot_cell = task_definition(task_id)
    seed_cell_id = _seed_cell_id(block_index, pilot_cell, configuration)
    expected_configuration = {
        "replicates": REPLICATES_PER_BLOCK,
        "whole_types_per_replicate": WHOLE_TYPES_PER_REPLICATE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "integration_draws": configuration.integration_draws,
        "folds": FOLDS,
        "confidence_level": CONFIDENCE_LEVEL,
    }
    for field, expected in (
        ("schema_version", configuration.schema_version),
        ("pilot_id", configuration.pilot_id),
        ("artifact_kind", "generated_replicate_shard"),
        ("task_id", task_id),
        ("block_index", block_index),
        ("block_id", f"block_{block_index + 1:02d}"),
        ("cell_index", cell_index),
        ("cell", asdict(pilot_cell)),
        ("seed_namespace_cell_id", seed_cell_id),
        ("configuration", expected_configuration),
        ("pilot_scope", "generated_data_only"),
        ("scientific_gate_applied", False),
        ("real_connectivity_accessed", False),
        ("final_connectivity_accessed", False),
        ("computationally_complete", True),
    ):
        _require_equal(artifact.get(field), expected, field)
    policy = artifact.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("interval pilot shard lacks policy metadata")
    _require_equal(
        policy.get("meaningful_margin"), FROZEN_MARGIN, "policy.meaningful_margin"
    )
    records = artifact.get("records")
    if not isinstance(records, list):
        raise ValueError("interval pilot shard records must be a list")
    _require_equal(len(records), REPLICATES_PER_BLOCK, "records count")
    for replicate_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("interval pilot record must be an object")
        _validate_record(
            record,
            replicate_index=replicate_index,
            seed_cell_id=seed_cell_id,
            pilot_cell=pilot_cell,
            configuration=configuration,
        )
    return block_index, pilot_cell, records


def _distribution_summary(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _coverage_entry(successes: int, total: int) -> dict[str, Any]:
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "exact_95_interval": None,
        }
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total,
        "exact_95_interval": list(exact_binomial_interval(successes, total)),
    }


def _summarize_block(
    task_id: int,
    block_index: int,
    pilot_cell: PilotCell,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminals = (
        "M_supported_before_full_search_null",
        "single_population_adequate_before_sensitivity_gate",
        "unresolved",
        "numerical_failure",
    )
    terminal_counts = {
        terminal: sum(record["terminal"] == terminal for record in records)
        for terminal in terminals
    }
    estimable = [record for record in records if "intervals" in record]
    numerical_failures = [
        {
            "replicate_index": int(record["replicate_index"]),
            "reasons": list(record.get("numerical_failures", [])),
        }
        for record in records
        if record["terminal"] == "numerical_failure"
    ]
    labels = ("left_to_right", "right_to_left", "aggregate")
    coverage_counts = np.zeros(3, dtype=int)
    joint_count = 0
    widths = np.empty((0, 3), dtype=float)
    if len(estimable) >= 2:
        estimates = np.asarray(
            [record["intervals"]["estimate"] for record in estimable],
            dtype=float,
        )
        lower = np.asarray(
            [record["intervals"]["lower"] for record in estimable], dtype=float
        )
        upper = np.asarray(
            [record["intervals"]["upper"] for record in estimable], dtype=float
        )
        centers = (estimates.sum(axis=0) - estimates) / (len(estimable) - 1)
        covered = (lower <= centers) & (centers <= upper)
        coverage_counts = covered.sum(axis=0)
        joint_count = int(np.all(covered, axis=1).sum())
        widths = upper - lower
    elif estimable:
        interval = estimable[0]["intervals"]
        widths = np.asarray(interval["upper"], dtype=float)[None, :] - np.asarray(
            interval["lower"], dtype=float
        )[None, :]

    coverage_total = len(estimable) if len(estimable) >= 2 else 0
    coordinate_coverage = {
        label: _coverage_entry(int(coverage_counts[index]), coverage_total)
        for index, label in enumerate(labels)
    }
    coordinate_widths = {
        label: _distribution_summary(widths[:, index])
        for index, label in enumerate(labels)
    }
    return {
        "task_id": task_id,
        "block_index": block_index,
        "block_id": f"block_{block_index + 1:02d}",
        "cell": asdict(pilot_cell),
        "replicate_records": len(records),
        "estimable_records": len(estimable),
        "terminal_counts": terminal_counts,
        "numerical_failure_count": len(numerical_failures),
        "numerical_failures": numerical_failures,
        "coverage": {
            **coordinate_coverage,
            "joint": _coverage_entry(joint_count, coverage_total),
            "nominal": CONFIDENCE_LEVEL,
            "scope": (
                "leave_one_replicate_out_generated_center; descriptive pilot, "
                "not an oracle parameter"
            ),
            "reference_centers_independent": False,
            "coverage_indicators_independent": False,
            "exact_binomial_acceptance_valid": False,
            "binomial_interval_role": "descriptive_reference_only",
            "scientific_acceptance_interpreted": False,
        },
        "interval_width": coordinate_widths,
        "false_adequacy_interpretation": {
            "status": "not_interpreted",
            "reason": (
                "population M-minus-best-T/U truth remains undefined until the "
                "generated oracle reference is scientifically repaired"
            ),
        },
    }


def aggregate_shards(
    *,
    output_root: Path | None = None,
    episode_root: Path = EPISODE_ROOT,
    configuration: PilotConfiguration = V1_CONFIGURATION,
) -> dict[str, Any]:
    """Verify every shard and publish a descriptive, non-gating pilot report."""

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    requested_output_root = output_root or configuration.output_root(episode_root)
    output_root = _validated_output_root(
        requested_output_root, episode_root, configuration
    )
    aggregate_path = output_root / AGGREGATE_NAME
    if aggregate_path.exists():
        raise FileExistsError(
            f"refusing to overwrite interval pilot aggregate: {aggregate_path}"
        )
    expected_paths = {
        shard_path(output_root, task_id).resolve() for task_id in range(SHARD_COUNT)
    }
    observed_paths = {
        path.resolve() for path in (output_root / "shards").glob("task_*.json")
    }
    missing = sorted(str(path) for path in expected_paths - observed_paths)
    unexpected = sorted(str(path) for path in observed_paths - expected_paths)
    if missing or unexpected:
        raise ValueError(
            "interval pilot shard inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    summaries: list[dict[str, Any]] = []
    total_records = 0
    total_numerical_failures = 0
    total_terminal_counts = {
        "M_supported_before_full_search_null": 0,
        "single_population_adequate_before_sensitivity_gate": 0,
        "unresolved": 0,
        "numerical_failure": 0,
    }
    for task_id in range(SHARD_COUNT):
        artifact = _load_json(shard_path(output_root, task_id))
        block_index, pilot_cell, records = _validate_shard(
            artifact, task_id, configuration
        )
        summary = _summarize_block(
            task_id, block_index, pilot_cell, records
        )
        summaries.append(summary)
        total_records += len(records)
        total_numerical_failures += summary["numerical_failure_count"]
        for terminal, count in summary["terminal_counts"].items():
            total_terminal_counts[terminal] += count

    report = {
        "schema_version": configuration.schema_version,
        "pilot_id": configuration.pilot_id,
        "artifact_kind": "descriptive_pilot_aggregate",
        "pilot_scope": "generated_data_only",
        "scientific_qualification_modified": False,
        "scientific_gate_applied": False,
        "coverage_threshold_applied": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "configuration": {
            "cells": len(PILOT_CELLS),
            "blocks": BLOCK_COUNT,
            "array_shards": SHARD_COUNT,
            "replicates_per_block": REPLICATES_PER_BLOCK,
            "whole_types_per_replicate": WHOLE_TYPES_PER_REPLICATE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "integration_draws": configuration.integration_draws,
            "folds": FOLDS,
            "confidence_level": CONFIDENCE_LEVEL,
            "meaningful_margin": FROZEN_MARGIN,
        },
        "artifact_verification": {
            "expected_shards": SHARD_COUNT,
            "verified_shards": len(summaries),
            "expected_records": SHARD_COUNT * REPLICATES_PER_BLOCK,
            "verified_records": total_records,
            "complete": True,
        },
        "overall_terminal_counts": total_terminal_counts,
        "overall_numerical_failure_count": total_numerical_failures,
        "block_cell_summaries": summaries,
        "coverage_interpretation": (
            "reported descriptively with binomial reference intervals; "
            "overlapping leave-one-replicate-out centers make the indicators "
            "dependent, so no interval is interpreted as exact target-coverage "
            "evidence or promoted to qualification acceptance"
        ),
        "false_adequacy_interpretation": {
            "status": "not_interpreted",
            "reason": (
                "population M-minus-best-T/U truth remains undefined until the "
                "generated oracle reference is scientifically repaired"
            ),
        },
        "computationally_complete": True,
        "runtime": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        },
    }
    _publish_json_exclusive(aggregate_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the generated-only EP12 interval refinement pilot."
    )
    parser.add_argument(
        "--configuration",
        choices=tuple(CONFIGURATIONS),
        default=V1_CONFIGURATION.key,
        help="artifact identity and integration-draw configuration",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard", help="run one array shard")
    shard.add_argument("--task-id", required=True, type=int)
    commands.add_parser("aggregate", help="verify and aggregate all 24 shards")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configuration = CONFIGURATIONS[args.configuration]
    output_root = configuration.output_root()
    if args.command == "shard":
        artifact = run_shard(args.task_id, configuration=configuration)
        print(
            json.dumps(
                {
                    "artifact": str(shard_path(output_root, args.task_id)),
                    "task_id": artifact["task_id"],
                    "records": len(artifact["records"]),
                    "computationally_complete": True,
                    "scientific_gate_applied": False,
                },
                sort_keys=True,
            )
        )
        return 0
    report = aggregate_shards(configuration=configuration)
    print(
        json.dumps(
            {
                "artifact": str(output_root / AGGREGATE_NAME),
                "verified_shards": report["artifact_verification"][
                    "verified_shards"
                ],
                "verified_records": report["artifact_verification"][
                    "verified_records"
                ],
                "computationally_complete": True,
                "scientific_gate_applied": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
