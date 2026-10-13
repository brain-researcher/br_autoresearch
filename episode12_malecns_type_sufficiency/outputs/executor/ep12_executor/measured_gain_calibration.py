"""Generated-only split-block calibration of EP12 fitted measured gain.

This module gives one completed, explicitly selected interval-refinement pilot
a narrow, finite-procedure interpretation.  Block 01 estimates the expected
fitted M-minus-selected-reference gain for each predeclared cell.  Block 02
remains independent and contributes terminal counts only.  The result is
explicitly not population-oracle truth and is not a scientific acceptance
gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import interval_refinement_pilot as pilot
from .runtime import EPISODE_ROOT
from .scientific_oracle import (
    GROUP_GENERATORS,
    classify_directional_margin_interval,
)
from .scientific_synthetic import seed_from
from .simultaneous_inference import (
    bilateral_with_aggregate,
    paired_studentized_max_t,
)


SCHEMA_VERSION = "ep12.finite_procedure_measured_gain_calibration.v1"
V2_SCHEMA_VERSION = "ep12.finite_procedure_measured_gain_calibration.v2"
V3_SCHEMA_VERSION = "ep12.finite_procedure_measured_gain_calibration.v3"
ARTIFACT_KIND = "finite_procedure_measured_gain_calibration"
ARTIFACT_NAME = "finite_procedure_measured_gain_calibration.json"
CALIBRATION_BLOCK_INDEX = 0
EVALUATION_BLOCK_INDEX = 1
CALIBRATION_SEED_NAMESPACE = (
    "ep12.interval_refinement_v1.finite_procedure_measured_gain.v1"
)
V2_CALIBRATION_SEED_NAMESPACE = (
    "ep12.interval_refinement_v2_1024.finite_procedure_measured_gain.v2"
)
V3_CALIBRATION_SEED_NAMESPACE = (
    "ep12.interval_refinement_v3_2048.finite_procedure_measured_gain.v3"
)
SCOPE_NAME = "finite_procedure_model_family_measured_gain"
TERMINALS = (
    "M_supported_before_full_search_null",
    "single_population_adequate_before_sensitivity_gate",
    "unresolved",
    "numerical_failure",
)


def _calibration_identity(
    configuration: pilot.PilotConfiguration,
) -> tuple[str, str]:
    """Return the schema and bootstrap namespace for a supported pilot."""

    if configuration == pilot.V1_CONFIGURATION:
        return SCHEMA_VERSION, CALIBRATION_SEED_NAMESPACE
    if configuration == pilot.V2_CONFIGURATION:
        return V2_SCHEMA_VERSION, V2_CALIBRATION_SEED_NAMESPACE
    if configuration == pilot.V3_CONFIGURATION:
        return V3_SCHEMA_VERSION, V3_CALIBRATION_SEED_NAMESPACE
    raise ValueError(
        "measured-gain calibration supports only declared interval pilot "
        f"configurations; received {configuration!r}"
    )


def _expected_shard_paths(output_root: Path) -> set[Path]:
    return {
        pilot.shard_path(output_root, task_id).resolve()
        for task_id in range(pilot.SHARD_COUNT)
    }


def _verify_inventory(output_root: Path) -> None:
    expected = _expected_shard_paths(output_root)
    observed = {
        path.resolve() for path in (output_root / "shards").glob("task_*.json")
    }
    missing = sorted(str(path) for path in expected - observed)
    unexpected = sorted(str(path) for path in observed - expected)
    if missing or unexpected:
        raise ValueError(
            "interval pilot shard inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validated_calibration_rows(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return exactly 72 x 8 paired block-01 fitted deltas."""

    expected_rows = pilot.REPLICATES_PER_BLOCK * pilot.WHOLE_TYPES_PER_REPLICATE
    rows: list[np.ndarray] = []
    for replicate_index, record in enumerate(records):
        if record.get("terminal") == "numerical_failure":
            raise ValueError(
                "block_01 cannot calibrate measured gain with a numerical-failure "
                f"record (replicate {replicate_index})"
            )
        type_records = record.get("type_records")
        if not isinstance(type_records, list):
            raise ValueError("block_01 calibration record lacks type_records")
        indexed: dict[int, np.ndarray] = {}
        for type_record in type_records:
            if not isinstance(type_record, Mapping):
                raise ValueError("block_01 type_record must be an object")
            type_index = type_record.get("synthetic_type_index")
            if (
                isinstance(type_index, bool)
                or not isinstance(type_index, int)
                or not 0 <= type_index < pilot.WHOLE_TYPES_PER_REPLICATE
                or type_index in indexed
            ):
                raise ValueError(
                    "block_01 type_record indices must be unique integers "
                    f"in [0, {pilot.WHOLE_TYPES_PER_REPLICATE - 1}]"
                )
            if not isinstance(type_record.get("source_fit_metadata"), Mapping):
                raise ValueError("block_01 type_record lacks source_fit_metadata")
            try:
                deltas = np.asarray(type_record.get("deltas"), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("block_01 fitted deltas must be numeric") from exc
            if deltas.shape != (2,) or not np.all(np.isfinite(deltas)):
                raise ValueError(
                    "block_01 fitted deltas must contain two finite directions"
                )
            indexed[type_index] = deltas
        expected_indices = set(range(pilot.WHOLE_TYPES_PER_REPLICATE))
        if set(indexed) != expected_indices:
            raise ValueError(
                "block_01 record does not contain the complete whole-type index set"
            )
        rows.extend(indexed[type_index] for type_index in sorted(indexed))

    array = np.asarray(rows, dtype=float)
    if array.shape != (expected_rows, 2) or not np.all(np.isfinite(array)):
        raise ValueError(
            "block_01 must provide exactly 72 x 8 finite paired fitted deltas"
        )
    return array


def _terminal_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Count block-02 outcomes without consulting any fitted delta record."""

    counts = Counter(record.get("terminal") for record in records)
    unexpected = sorted(value for value in counts if value not in TERMINALS)
    if unexpected:
        raise ValueError(f"block_02 contains unknown terminals: {unexpected}")
    return {terminal: int(counts.get(terminal, 0)) for terminal in TERMINALS}


def _false_adequacy_summary(
    *, scenario: str, margin_classification: str, adequacy_count: int
) -> dict[str, Any]:
    null_fields = {
        "successes": None,
        "total": None,
        "rate": None,
        "clopper_pearson_exact_95_interval": None,
    }
    if scenario not in GROUP_GENERATORS:
        return {
            "status": "not_applicable_non_residual_group_generator",
            "reason": (
                "false adequacy is evaluated only for declared residual-group "
                "generator cells"
            ),
            **null_fields,
        }
    if margin_classification != "above_margin":
        return {
            "status": "unresolved_for_adequacy",
            "reason": (
                "block_01 fitted measured-gain interval is not above the frozen "
                "margin in both directions"
            ),
            **null_fields,
        }

    total = pilot.REPLICATES_PER_BLOCK
    interval = pilot.exact_binomial_interval(adequacy_count, total)
    return {
        "status": "finite_procedure_false_adequacy_estimated",
        "reason": None,
        "successes": int(adequacy_count),
        "total": total,
        "rate": adequacy_count / total,
        "clopper_pearson_exact_95_interval": list(interval),
    }


def _calibration_interval(
    rows: np.ndarray,
    *,
    cell_id: str,
    configuration: pilot.PilotConfiguration = pilot.V1_CONFIGURATION,
) -> tuple[int, dict[str, Any], str]:
    _, seed_namespace = _calibration_identity(configuration)
    interval_seed = seed_from(
        seed_namespace,
        configuration.pilot_id,
        "block_01",
        cell_id,
        "paired-studentized-max-t",
    )
    interval = paired_studentized_max_t(
        bilateral_with_aggregate(rows),
        replicates=pilot.BOOTSTRAP_REPLICATES,
        seed=interval_seed,
        confidence_level=pilot.CONFIDENCE_LEVEL,
    )
    interval.update(
        {
            "labels": ["left_to_right", "right_to_left", "aggregate"],
            "margin": pilot.FROZEN_MARGIN,
            "pooled_whole_type_rows": int(rows.shape[0]),
            "source_block_id": "block_01",
        }
    )
    classification = classify_directional_margin_interval(
        interval["lower"][:2],
        interval["upper"][:2],
        margin=pilot.FROZEN_MARGIN,
    )
    return interval_seed, interval, classification


def publish_measured_gain_calibration(
    *,
    output_root: Path = pilot.DEFAULT_OUTPUT_ROOT,
    episode_root: Path = EPISODE_ROOT,
    configuration: pilot.PilotConfiguration = pilot.V1_CONFIGURATION,
) -> dict[str, Any]:
    """Verify both pilot blocks and atomically publish their split analysis."""

    schema_version, calibration_seed_namespace = _calibration_identity(
        configuration
    )
    output_root = pilot._validated_output_root(
        output_root, episode_root, configuration
    )
    destination = output_root / ARTIFACT_NAME
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite measured-gain calibration: {destination}"
        )
    _verify_inventory(output_root)

    by_block: dict[int, dict[str, tuple[int, list[Mapping[str, Any]], str]]] = {
        CALIBRATION_BLOCK_INDEX: {},
        EVALUATION_BLOCK_INDEX: {},
    }
    namespaces: dict[int, set[str]] = {
        CALIBRATION_BLOCK_INDEX: set(),
        EVALUATION_BLOCK_INDEX: set(),
    }
    record_identities: dict[int, set[tuple[str, int]]] = {
        CALIBRATION_BLOCK_INDEX: set(),
        EVALUATION_BLOCK_INDEX: set(),
    }
    policy_hashes: set[str] = set()
    task_ids: dict[int, list[int]] = {
        CALIBRATION_BLOCK_INDEX: [],
        EVALUATION_BLOCK_INDEX: [],
    }

    for task_id in range(pilot.SHARD_COUNT):
        artifact = pilot._load_json(pilot.shard_path(output_root, task_id))
        block_index, cell, records = pilot._validate_shard(
            artifact, task_id, configuration=configuration
        )
        if block_index not in by_block:
            raise ValueError(f"unexpected pilot block index: {block_index}")
        namespace = artifact.get("seed_namespace_cell_id")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("validated shard lacks a seed namespace")
        policy = artifact.get("policy")
        policy_hash = policy.get("policy_hash") if isinstance(policy, Mapping) else None
        if (
            not isinstance(policy_hash, str)
            or len(policy_hash) != 64
            or any(character not in "0123456789abcdef" for character in policy_hash.lower())
        ):
            raise ValueError("interval pilot shard has an invalid policy hash")
        policy_hashes.add(policy_hash.lower())
        if cell.base_cell_id in by_block[block_index]:
            raise ValueError(
                f"duplicate pilot cell in block_{block_index + 1:02d}: "
                f"{cell.base_cell_id}"
            )
        by_block[block_index][cell.base_cell_id] = (task_id, records, namespace)
        namespaces[block_index].add(namespace)
        task_ids[block_index].append(task_id)
        identities = {
            (namespace, int(record["replicate_index"])) for record in records
        }
        if len(identities) != len(records):
            raise ValueError("pilot shard contains duplicate record identities")
        if record_identities[block_index].intersection(identities):
            raise ValueError("pilot block contains duplicate record identities")
        record_identities[block_index].update(identities)

    if len(policy_hashes) != 1:
        raise ValueError("interval pilot shards do not share one policy hash")
    expected_cells = {cell.base_cell_id for cell in pilot.PILOT_CELLS}
    for block_index in (CALIBRATION_BLOCK_INDEX, EVALUATION_BLOCK_INDEX):
        if set(by_block[block_index]) != expected_cells:
            raise ValueError(
                f"block_{block_index + 1:02d} pilot cell inventory is incomplete"
            )
    if namespaces[CALIBRATION_BLOCK_INDEX] & namespaces[EVALUATION_BLOCK_INDEX]:
        raise ValueError("calibration and evaluation seed namespaces overlap")
    if record_identities[CALIBRATION_BLOCK_INDEX] & record_identities[EVALUATION_BLOCK_INDEX]:
        raise ValueError("calibration and evaluation record identities overlap")

    cell_results: list[dict[str, Any]] = []
    for cell in pilot.PILOT_CELLS:
        calibration_task, calibration_records, _ = by_block[
            CALIBRATION_BLOCK_INDEX
        ][cell.base_cell_id]
        evaluation_task, evaluation_records, _ = by_block[
            EVALUATION_BLOCK_INDEX
        ][cell.base_cell_id]
        rows = _validated_calibration_rows(calibration_records)
        interval_seed, interval, classification = _calibration_interval(
            rows,
            cell_id=cell.base_cell_id,
            configuration=configuration,
        )
        terminal_counts = _terminal_counts(evaluation_records)
        adequacy_count = terminal_counts[
            "single_population_adequate_before_sensitivity_gate"
        ]
        cell_results.append(
            {
                "cell": asdict(cell),
                "calibration": {
                    "task_id": calibration_task,
                    "source_block_id": "block_01",
                    "bootstrap_seed_namespace": calibration_seed_namespace,
                    "bootstrap_seed": interval_seed,
                    "interval": interval,
                    "margin_classification": classification,
                    "scope": SCOPE_NAME,
                },
                "evaluation": {
                    "task_id": evaluation_task,
                    "source_block_id": "block_02",
                    "replicate_records": len(evaluation_records),
                    "terminal_counts": terminal_counts,
                    "adequacy_terminal_count": adequacy_count,
                    "fitted_delta_records_used": False,
                },
                "false_adequacy": _false_adequacy_summary(
                    scenario=cell.scenario,
                    margin_classification=classification,
                    adequacy_count=adequacy_count,
                ),
            }
        )

    calibration_records = len(record_identities[CALIBRATION_BLOCK_INDEX])
    evaluation_records = len(record_identities[EVALUATION_BLOCK_INDEX])
    report = {
        "schema_version": schema_version,
        "pilot_id": configuration.pilot_id,
        "artifact_kind": ARTIFACT_KIND,
        "pilot_scope": "generated_data_only",
        "calibration_scope": SCOPE_NAME,
        "scope_statement": (
            "The block_01 label concerns the expected measured gain of this "
            "finite fitted T/U/M procedure under the declared synthetic design "
            "and model family. It is not population structural or oracle truth."
        ),
        "population_oracle_truth": False,
        "scientific_gate_applied": False,
        "scientific_pass_fail_assigned": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "configuration": {
            "meaningful_margin": pilot.FROZEN_MARGIN,
            "confidence_level": pilot.CONFIDENCE_LEVEL,
            "bootstrap_replicates": pilot.BOOTSTRAP_REPLICATES,
            "calibration_block_id": "block_01",
            "evaluation_block_id": "block_02",
            "whole_type_rows_per_cell": (
                pilot.REPLICATES_PER_BLOCK * pilot.WHOLE_TYPES_PER_REPLICATE
            ),
        },
        "artifact_verification": {
            "expected_shards": pilot.SHARD_COUNT,
            "verified_shards": pilot.SHARD_COUNT,
            "policy_hash": next(iter(policy_hashes)),
            "block_seed_namespaces_disjoint": True,
            "block_record_identities_disjoint": True,
            "evaluation_fitted_delta_records_used": False,
            "complete": True,
        },
        "block_roles": {
            "calibration": {
                "block_id": "block_01",
                "task_ids": sorted(task_ids[CALIBRATION_BLOCK_INDEX]),
                "verified_records": calibration_records,
                "fitted_delta_rows": calibration_records
                * pilot.WHOLE_TYPES_PER_REPLICATE,
                "seed_namespaces": sorted(namespaces[CALIBRATION_BLOCK_INDEX]),
            },
            "evaluation": {
                "block_id": "block_02",
                "task_ids": sorted(task_ids[EVALUATION_BLOCK_INDEX]),
                "verified_records": evaluation_records,
                "field_used": "terminal",
                "fitted_delta_records_used": False,
                "seed_namespaces": sorted(namespaces[EVALUATION_BLOCK_INDEX]),
            },
        },
        "cell_results": cell_results,
        "computationally_complete": True,
    }
    pilot._publish_json_exclusive(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the generated-only EP12 finite-procedure measured-gain "
            "calibration."
        )
    )
    parser.add_argument(
        "--configuration",
        choices=tuple(pilot.CONFIGURATIONS),
        default=pilot.V1_CONFIGURATION.key,
        help="completed interval-pilot artifact identity to calibrate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configuration = pilot.CONFIGURATIONS[args.configuration]
    output_root = configuration.output_root()
    report = publish_measured_gain_calibration(
        output_root=output_root,
        configuration=configuration,
    )
    print(
        json.dumps(
            {
                "artifact": str(output_root / ARTIFACT_NAME),
                "verified_shards": report["artifact_verification"]["verified_shards"],
                "cells": len(report["cell_results"]),
                "computationally_complete": True,
                "scientific_gate_applied": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
