#!/usr/bin/env python3
"""EP09 measurement contract v1.2: finite-cell n_coverage precision simulation."""

from __future__ import annotations

import os

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _variable in THREAD_ENV:
    os.environ[_variable] = "1"

import sys

# Refuse direct execution before importing numerical libraries.
if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise SystemExit("ERROR: SLURM_JOB_ID must be set.")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("ERROR: Python 3.11 is required.")

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import socket
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit, logit, ndtr, ndtri, owens_t


IMPLEMENTATION_VERSION = "ep09-ncoverage-1.0.0"
CONTRACT_VERSION = "1.2"

MASTER_SEED = 9009
REPLICATES = 10_000
CHUNK_SIZE = 1_000
DEVELOPMENT_ANIMALS = 27
TARGETS = 9
FAMILIES = 3
TARGETS_PER_FAMILY = 3
SCENARIOS_PER_N = 108

PREVALENCES = (0.1, 0.5, 0.9)
HETEROGENEITY_SDS = (0.0, 1.0)
TARGET_DEPENDENCES = (0.0, 0.5, 1.0)
ORDERED_PAIRS = (
    (0.2, 0.8),
    (0.8, 0.2),
    (0.35, 0.65),
    (0.65, 0.35),
    (0.45, 0.55),
    (0.55, 0.45),
)

MAX_ERROR_SD = 0.05
MAX_P95_ABSOLUTE_ERROR = 0.10
ANALYTIC_RATIO_MIN = 0.97
ANALYTIC_RATIO_MAX = 1.03

REQUIRED_COLUMNS = (
    "cell", "brain", "line", "area", "layer",
    "pool_key", "pool_size", "ep09_role",
)
EXPECTED_COUNTS = {
    "cells": 1703,
    "animals": 43,
    "development_cells": 1114,
    "development_animals": 27,
    "audit_cells": 589,
    "audit_animals": 16,
}
OUTPUT_NAMES = ("ncoverage_report.json", "ncoverage_scenarios.csv")
SCENARIO_FIELDS = (
    "task_index", "scenario_index", "n", "prevalence",
    "heterogeneity_sd", "target_dependence", "q_first", "q_second",
)
CSV_FIELDS = SCENARIO_FIELDS + (
    "rng_spawn_key",
    "replicates",
    "sided_target_count",
    "primary_count_gate_passes",
    "mean_simulated_contrast",
    "mean_conditional_expected_contrast",
    "mean_error",
    "error_sd",
    "p95_absolute_error",
    "maximum_absolute_error",
    "sd_passes",
    "p95_passes",
    "precision_passes",
    "analytic_check_performed",
    "analytic_error_sd",
    "simulated_to_analytic_sd_ratio",
    "analytic_check_flagged",
)


@dataclass(frozen=True)
class Animal:
    brain: str
    numeric_brain_id: int
    cell_ids: tuple[str, ...]
    pool_keys: tuple[str, ...]
    pool_sizes: tuple[int, ...]
    cell_pool_index: np.ndarray


@dataclass(frozen=True)
class Task:
    task_index: int
    scenario_index: int
    n: int
    prevalence: float
    heterogeneity_sd: float
    target_dependence: float
    q_first: float
    q_second: float
    seed: np.random.SeedSequence


_WORKER_ANIMALS: tuple[Animal, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decimal_integer(value: str, label: str, minimum: int = 0) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{label} must be an unsigned decimal integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    return result


def validate_runtime(workers: int) -> dict[str, int]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 is required.")
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError("SLURM_JOB_ID must be set.")
    if workers < 1:
        raise ValueError("--workers must be >= 1.")

    limits = {}
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name)
        if value is not None:
            limits[name] = decimal_integer(value.strip(), name, minimum=1)
    if hasattr(os, "sched_getaffinity"):
        limits["process_cpu_affinity_count"] = len(os.sched_getaffinity(0))

    if limits and workers > min(limits.values()):
        raise ValueError(
            f"--workers={workers} exceeds an available CPU limit: {limits}"
        )
    return limits


def load_manifest(data: bytes) -> tuple[tuple[Animal, ...], dict]:
    reader = csv.DictReader(
        io.StringIO(data.decode("utf-8-sig"), newline=""),
        strict=True,
    )
    headers = reader.fieldnames
    if headers is None or len(headers) != len(set(headers)):
        raise ValueError("Manifest header is missing or contains duplicates.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(headers))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    by_brain: dict[int, list[dict[str, str]]] = {}
    seen_cells: set[str] = set()
    declared_pool_sizes: dict[str, int] = {}
    observed_pool_sizes: Counter = Counter()
    role_cells: Counter = Counter()

    for record_number, raw in enumerate(reader, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"Malformed manifest record {record_number}.")
        row = {name: raw[name] for name in REQUIRED_COLUMNS}
        if any(not value or value != value.strip() for value in row.values()):
            raise ValueError(
                f"Record {record_number}: required values must be nonempty "
                "and have no surrounding whitespace."
            )

        role = row["ep09_role"]
        if role not in ("development", "audit"):
            raise ValueError(f"Record {record_number}: invalid ep09_role {role!r}.")
        if row["cell"] in seen_cells:
            raise ValueError(f"Duplicate cell ID: {row['cell']!r}")
        seen_cells.add(row["cell"])

        brain_id = decimal_integer(row["brain"], "brain")
        group = by_brain.setdefault(brain_id, [])
        if group and (
            group[0]["brain"] != row["brain"]
            or group[0]["ep09_role"] != role
        ):
            raise ValueError(
                f"Brain {brain_id} has inconsistent ID spelling or role."
            )

        expected_key = "|".join(
            row[name] for name in ("brain", "line", "area", "layer")
        )
        if row["pool_key"] != expected_key:
            raise ValueError(
                f"Record {record_number}: pool_key must equal "
                "brain|line|area|layer."
            )
        size = decimal_integer(row["pool_size"], "pool_size", minimum=1)
        key = row["pool_key"]
        if key in declared_pool_sizes and declared_pool_sizes[key] != size:
            raise ValueError(f"Inconsistent declared pool_size for {key!r}.")
        declared_pool_sizes[key] = size
        observed_pool_sizes[key] += 1
        role_cells[role] += 1
        group.append(row)

    role_animals = Counter(
        rows[0]["ep09_role"] for rows in by_brain.values()
    )
    counts = {
        "cells": len(seen_cells),
        "animals": len(by_brain),
        "development_cells": role_cells["development"],
        "development_animals": role_animals["development"],
        "audit_cells": role_cells["audit"],
        "audit_animals": role_animals["audit"],
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Manifest counts differ from required counts: "
            f"observed={counts}, required={EXPECTED_COUNTS}"
        )
    for key, observed in observed_pool_sizes.items():
        if observed != declared_pool_sizes[key]:
            raise ValueError(
                f"Pool {key!r}: declared {declared_pool_sizes[key]} cells, "
                f"observed {observed}."
            )

    n_max = min(len(rows) for rows in by_brain.values())
    animals = []
    for brain_id in sorted(by_brain):
        rows = by_brain[brain_id]
        if rows[0]["ep09_role"] != "development":
            continue
        pool_keys = tuple(sorted({row["pool_key"] for row in rows}))
        pool_lookup = {key: index for index, key in enumerate(pool_keys)}
        cell_pool_index = np.fromiter(
            (pool_lookup[row["pool_key"]] for row in rows),
            dtype=np.int64,
            count=len(rows),
        )
        cell_pool_index.setflags(write=False)
        animals.append(Animal(
            brain=rows[0]["brain"],
            numeric_brain_id=brain_id,
            cell_ids=tuple(row["cell"] for row in rows),
            pool_keys=pool_keys,
            pool_sizes=tuple(observed_pool_sizes[key] for key in pool_keys),
            cell_pool_index=cell_pool_index,
        ))

    counts.update({
        "n_max": n_max,
        "expected_n_max": 14,
        "n_max_matches_expected": n_max == 14,
    })
    return tuple(animals), counts


def unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")


def load_count_gates(
    data: bytes, n_max: int, candidates: list[int]
) -> dict[int, dict]:
    precheck = json.loads(
        data.decode("utf-8-sig"),
        object_pairs_hook=unique_json_object,
        parse_constant=reject_json_constant,
    )
    if not isinstance(precheck, dict):
        raise ValueError("Precheck JSON must be an object.")
    contract = precheck.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Precheck contract must be an object.")
    supplied_n_max = contract.get("n_max")
    if type(supplied_n_max) is not int or supplied_n_max != n_max:
        raise ValueError(
            f"Manifest n_max={n_max} differs from precheck "
            f"contract.n_max={supplied_n_max!r}."
        )

    entries = precheck.get("V_n")
    if not isinstance(entries, list):
        raise ValueError("Precheck V_n must be a list.")
    gates = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each V_n entry must be an object.")
        n = entry.get("n")
        count = entry.get("sided_target_count")
        gate = entry.get("primary_count_gate_passes")
        if type(n) is not int:
            raise ValueError("Each V_n n must be an integer.")
        if n in gates:
            raise ValueError(f"Duplicate V_n entry for n={n}.")
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid sided_target_count for n={n}.")
        if type(gate) is not bool or gate != (count > 0):
            raise ValueError(
                f"Inconsistent primary count gate for n={n}; "
                "it must equal sided_target_count > 0."
            )
        gates[n] = entry

    missing = [n for n in candidates if n not in gates]
    if missing:
        raise ValueError(f"Precheck V_n is missing allowed candidates: {missing}")
    # Retain all supplied family metadata without interpreting it as a gate.
    return {n: gates[n] for n in candidates}


def make_tasks(candidates: list[int]) -> list[Task]:
    scenarios = [
        (p, s, rho, q1, q2)
        for p in PREVALENCES
        for s in HETEROGENEITY_SDS
        for rho in TARGET_DEPENDENCES
        for q1, q2 in ORDERED_PAIRS
    ]
    if len(scenarios) != SCENARIOS_PER_N:
        raise RuntimeError("Frozen scenario count is inconsistent.")

    total_tasks = len(candidates) * SCENARIOS_PER_N
    children = np.random.SeedSequence(MASTER_SEED).spawn(total_tasks)
    tasks = []
    for n in candidates:
        for scenario_index, (p, s, rho, q1, q2) in enumerate(scenarios):
            task_index = len(tasks)
            tasks.append(Task(
                task_index=task_index,
                scenario_index=scenario_index,
                n=n,
                prevalence=p,
                heterogeneity_sd=s,
                target_dependence=rho,
                q_first=q1,
                q_second=q2,
                seed=children[task_index],
            ))
    return tasks


def initialize_worker(animals: tuple[Animal, ...]) -> None:
    global _WORKER_ANIMALS
    if len(animals) != DEVELOPMENT_ANIMALS:
        raise ValueError("Workers must receive exactly 27 development animals.")
    for animal in animals:
        animal.cell_pool_index.setflags(write=False)
    _WORKER_ANIMALS = animals


def analytic_error_sd(task: Task) -> float:
    p = task.prevalence
    rho = task.target_dependence
    if rho == 0.0:
        covariance = 0.0
    elif rho == 1.0:
        covariance = p * (1.0 - p)
    else:
        z = float(ndtri(p))
        # Exact identity for equal thresholds, evaluated deterministically.
        # Phi2(z,z;rho) = Phi(z) - 2*T(z,sqrt((1-rho)/(1+rho))).
        phi2 = float(ndtr(z)) - 2.0 * float(
            owens_t(z, math.sqrt((1.0 - rho) / (1.0 + rho)))
        )
        covariance = phi2 - p * p

    cell_variance = (p * (1.0 - p) + 8.0 * covariance) / TARGETS
    population_variance = cell_variance / (task.n * DEVELOPMENT_ANIMALS)
    return 2.0 * abs(task.q_first - task.q_second) * math.sqrt(
        population_variance
    )


def simulate_task(task: Task) -> dict:
    if len(_WORKER_ANIMALS) != DEVELOPMENT_ANIMALS:
        raise RuntimeError("Worker development-animal data are not initialized.")

    rng = np.random.Generator(np.random.PCG64(task.seed))
    simulated = np.empty(REPLICATES, dtype=np.float64)
    expected = np.empty(REPLICATES, dtype=np.float64)
    baseline_logit = float(logit(task.prevalence))
    shared_scale = math.sqrt(task.target_dependence)
    independent_scale = math.sqrt(1.0 - task.target_dependence)
    q1, q2 = task.q_first, task.q_second

    for start in range(0, REPLICATES, CHUNK_SIZE):
        stop = start + CHUNK_SIZE
        simulated_sum = np.zeros(CHUNK_SIZE, dtype=np.float64)
        expected_sum = np.zeros(CHUNK_SIZE, dtype=np.float64)

        for animal in _WORKER_ANIMALS:
            cell_count = len(animal.cell_ids)
            pool_count = len(animal.pool_keys)

            # Frozen RNG calls 1--5. No endpoint shortcuts are permitted.
            keys = rng.random(
                (CHUNK_SIZE, cell_count), dtype=np.float64
            )
            sampled_indices = np.argsort(
                keys, axis=1, kind="stable"
            )[:, :task.n]

            animal_effect = rng.standard_normal(
                (CHUNK_SIZE,), dtype=np.float64
            ) * task.heterogeneity_sd
            pool_effect = rng.standard_normal(
                (CHUNK_SIZE, pool_count), dtype=np.float64
            ) * task.heterogeneity_sd
            z_cell = rng.standard_normal(
                (CHUNK_SIZE, task.n), dtype=np.float64
            )
            z_target = rng.standard_normal(
                (CHUNK_SIZE, task.n, TARGETS), dtype=np.float64
            )

            pool_probabilities = expit(
                baseline_logit + animal_effect[:, None] + pool_effect
            )
            sampled_pools = animal.cell_pool_index[sampled_indices]
            probabilities = np.take_along_axis(
                pool_probabilities, sampled_pools, axis=1
            )
            latent = (
                shared_scale * z_cell[:, :, None]
                + independent_scale * z_target
            )
            outcomes = (
                latent < ndtri(probabilities)[:, :, None]
            ).astype(np.float64)

            # Registered order: cells, then targets, then animals.
            loss_difference = (
                np.square(outcomes - q1) - np.square(outcomes - q2)
            )
            target_contrasts = loss_difference.mean(
                axis=1, dtype=np.float64
            )
            simulated_sum += target_contrasts.mean(
                axis=1, dtype=np.float64
            )

            # All nine targets have the same conditional probability.
            # Their nine expected target contrasts are therefore identical.
            expected_cell_difference = (
                np.square(probabilities - q1)
                - np.square(probabilities - q2)
            )
            expected_sum += expected_cell_difference.mean(
                axis=1, dtype=np.float64
            )

        simulated[start:stop] = simulated_sum / DEVELOPMENT_ANIMALS
        expected[start:stop] = expected_sum / DEVELOPMENT_ANIMALS

    errors = simulated - expected
    absolute_errors = np.abs(errors)
    metrics = {
        "mean_simulated_contrast": float(
            simulated.mean(dtype=np.float64)
        ),
        "mean_conditional_expected_contrast": float(
            expected.mean(dtype=np.float64)
        ),
        "mean_error": float(errors.mean(dtype=np.float64)),
        "error_sd": float(errors.std(ddof=1, dtype=np.float64)),
        "p95_absolute_error": float(np.quantile(absolute_errors, 0.95)),
        "maximum_absolute_error": float(absolute_errors.max()),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite metric in task {task.task_index}.")

    sd_passes = metrics["error_sd"] <= MAX_ERROR_SD
    p95_passes = metrics["p95_absolute_error"] <= MAX_P95_ABSOLUTE_ERROR
    analytic_sd = None
    ratio = None
    analytic_flagged = False
    if task.heterogeneity_sd == 0.0:
        analytic_sd = analytic_error_sd(task)
        ratio = metrics["error_sd"] / analytic_sd
        if not math.isfinite(ratio):
            raise RuntimeError(
                f"Nonfinite analytic SD ratio in task {task.task_index}."
            )
        analytic_flagged = not (
            ANALYTIC_RATIO_MIN <= ratio <= ANALYTIC_RATIO_MAX
        )

    return {
        "task_index": task.task_index,
        "scenario_index": task.scenario_index,
        "n": task.n,
        "prevalence": task.prevalence,
        "heterogeneity_sd": task.heterogeneity_sd,
        "target_dependence": task.target_dependence,
        "q_first": q1,
        "q_second": q2,
        "rng_spawn_key": json.dumps(
            list(task.seed.spawn_key), separators=(",", ":")
        ),
        "replicates": REPLICATES,
        **metrics,
        "sd_passes": sd_passes,
        "p95_passes": p95_passes,
        "precision_passes": sd_passes and p95_passes,
        "analytic_check_performed": task.heterogeneity_sd == 0.0,
        "analytic_error_sd": analytic_sd,
        "simulated_to_analytic_sd_ratio": ratio,
        "analytic_check_flagged": analytic_flagged,
    }


def scenario_identity(row: dict) -> dict:
    return {name: row[name] for name in SCENARIO_FIELDS}


def summarize_n(n: int, rows: list[dict], gate: dict) -> dict:
    if len(rows) != SCENARIOS_PER_N or any(row["n"] != n for row in rows):
        raise RuntimeError(f"Incomplete or mixed scenario results for n={n}.")

    worst_sd = max(rows, key=lambda row: row["error_sd"])
    worst_p95 = max(rows, key=lambda row: row["p95_absolute_error"])
    failing = sum(not row["precision_passes"] for row in rows)
    return {
        "n": n,
        "V_n": gate,
        "scenario_count": len(rows),
        "failing_scenario_count": failing,
        "simulation_passes": failing == 0,
        "passes": gate["sided_target_count"] > 0 and failing == 0,
        "worst_error_sd": {
            "value": worst_sd["error_sd"],
            "scenario": scenario_identity(worst_sd),
        },
        "worst_p95_absolute_error": {
            "value": worst_p95["p95_absolute_error"],
            "scenario": scenario_identity(worst_p95),
        },
    }


def summarize_analytic_checks(rows: list[dict]) -> dict:
    checked = [row for row in rows if row["analytic_check_performed"]]
    flagged = [row for row in checked if row["analytic_check_flagged"]]
    ratios = [row["simulated_to_analytic_sd_ratio"] for row in checked]
    return {
        "checked_scenario_count": len(checked),
        "flagged_scenario_count": len(flagged),
        "passes": not flagged,
        "allowed_ratio_interval_inclusive": [
            ANALYTIC_RATIO_MIN, ANALYTIC_RATIO_MAX
        ],
        "minimum_ratio": min(ratios) if ratios else None,
        "maximum_ratio": max(ratios) if ratios else None,
        "flagged_scenarios": [
            {
                **scenario_identity(row),
                "error_sd": row["error_sd"],
                "analytic_error_sd": row["analytic_error_sd"],
                "simulated_to_analytic_sd_ratio":
                    row["simulated_to_analytic_sd_ratio"],
            }
            for row in flagged
        ],
    }


def frozen_settings() -> dict:
    return {
        "master_seed": MASTER_SEED,
        "bit_generator": "PCG64",
        "seed_assignment": (
            "SeedSequence(9009).spawn(total_tasks); task k uses child k; "
            "indices are zero-based"
        ),
        "replicates_per_scenario": REPLICATES,
        "chunk_size": CHUNK_SIZE,
        "chunks_per_task": REPLICATES // CHUNK_SIZE,
        "synthetic_targets": TARGETS,
        "synthetic_families": FAMILIES,
        "targets_per_family": TARGETS_PER_FAMILY,
        "virtual_family_target_indices": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "development_animal_count": DEVELOPMENT_ANIMALS,
        "prevalence_grid": list(PREVALENCES),
        "heterogeneity_sd_grid": list(HETEROGENEITY_SDS),
        "target_dependence_grid": list(TARGET_DEPENDENCES),
        "ordered_prediction_pairs": [list(pair) for pair in ORDERED_PAIRS],
        "scenario_order": [
            "n ascending", "prevalence", "heterogeneity_sd",
            "target_dependence", "ordered_prediction_pair",
        ],
        "scenarios_per_n": SCENARIOS_PER_N,
        "animal_order": "ascending numeric brain ID",
        "cell_order": "manifest order within each animal",
        "pool_order": "sorted pool_key within each animal",
        "sampling": (
            "rng.random((1000,N_g), dtype=float64); stable argsort along "
            "cells; take first n, in increasing-key order; ties retain "
            "manifest order; same selected IDs for all nine targets"
        ),
        "rng_call_order_per_chunk_per_animal": [
            "random((1000,N_g), dtype=float64)",
            "standard_normal((1000,), dtype=float64) * s",
            "standard_normal((1000,K_g), dtype=float64) * s",
            "standard_normal((1000,n), dtype=float64)",
            "standard_normal((1000,n,9), dtype=float64)",
        ],
        "endpoint_draws": "All five calls occur even at s=0 or rho in {0,1}.",
        "probability_model": "expit(logit(p) + a_g + b_gk)",
        "outcome_rule": (
            "sqrt(rho)*Z_i + sqrt(1-rho)*Z_it < ndtri(p_gk)"
        ),
        "arithmetic": "float64 for numerical simulation and reductions",
        "contrast": "Brier(first) - Brier(second)",
        "weighting": "equal mean over cells, then 9 targets, then 27 animals",
        "conditional_expectation": (
            "Replace each outcome by its sampled cell's realized p_gk "
            "in the squared-loss difference"
        ),
        "error_definition": "simulated contrast - conditional expectation",
        "sd_ddof": 1,
        "absolute_error_quantile": 0.95,
        "quantile_method": "np.quantile default linear",
        "maximum_error_standard_deviation": MAX_ERROR_SD,
        "maximum_95th_percentile_absolute_error": MAX_P95_ABSOLUTE_ERROR,
        "precision_thresholds_are_inclusive": True,
        "analytic_check": {
            "applies_when": "s == 0",
            "phi2_method": (
                "Phi2(z,z;rho) = ndtr(z) - "
                "2*owens_t(z,sqrt((1-rho)/(1+rho)))"
            ),
            "endpoint_covariances": {"rho=0": "0", "rho=1": "p*(1-p)"},
            "cell_target_mean_variance": "(p*(1-p) + 8*c)/9",
            "population_mean_variance": "cell_target_mean_variance/(n*27)",
            "error_sd": "2*abs(q_first-q_second)*sqrt(population_mean_variance)",
            "ratio_interval_inclusive": [
                ANALYTIC_RATIO_MIN, ANALYTIC_RATIO_MAX
            ],
            "finite_population_correction": False,
        },
        "selection": (
            "Smallest allowed n with sided_target_count > 0 and all "
            "108 scenarios passing both precision thresholds"
        ),
        "family_metadata": (
            "Preserved verbatim in per_n[].V_n; informational only"
        ),
        "worst_scenario_tie_break": "first in frozen scenario order",
    }


def scenarios_csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=CSV_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_exclusive(path: Path, data: bytes, created: list[Path]) -> None:
    stream = path.open("xb")
    created.append(path)
    with stream:
        stream.write(data)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())


def rollback(created: list[Path]) -> None:
    for path in reversed(created):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            print(
                f"ERROR: Could not roll back {path}: {error}",
                file=sys.stderr,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_manifest_csv", type=Path)
    parser.add_argument("precheck_summary_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    created: list[Path] = []
    try:
        started_at = utc_now()
        cpu_limits = validate_runtime(args.workers)
        manifest_path = args.cell_manifest_csv.expanduser().resolve(strict=True)
        precheck_path = args.precheck_summary_json.expanduser().resolve(strict=True)
        script_path = Path(__file__).resolve(strict=True)
        out_dir = args.out_dir.expanduser().resolve()

        # Parse exactly the bytes whose hashes are recorded.
        manifest_bytes = manifest_path.read_bytes()
        precheck_bytes = precheck_path.read_bytes()
        script_bytes = script_path.read_bytes()

        animals, manifest_summary = load_manifest(manifest_bytes)
        n_max = manifest_summary["n_max"]
        candidates = list(range(3, n_max + 1))
        gates = load_count_gates(precheck_bytes, n_max, candidates)
        tasks = make_tasks(candidates)

        out_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUT_NAMES:
            if os.path.lexists(out_dir / name):
                raise FileExistsError(f"Output already exists: {out_dir / name}")

        rows = []
        per_n = []
        workers_used = min(args.workers, len(tasks))
        if tasks:
            executor = ProcessPoolExecutor(
                max_workers=workers_used,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(animals,),
            )
            try:
                # map yields input order, independently of completion order.
                for row in executor.map(simulate_task, tasks, chunksize=1):
                    gate = gates[row["n"]]
                    row["sided_target_count"] = gate["sided_target_count"]
                    row["primary_count_gate_passes"] = (
                        gate["primary_count_gate_passes"]
                    )
                    rows.append(row)
                    if len(rows) % SCENARIOS_PER_N == 0:
                        summary = summarize_n(
                            row["n"], rows[-SCENARIOS_PER_N:], gate
                        )
                        per_n.append(summary)
                        print(
                            f"n={summary['n']:2d} "
                            f"worst_SD={summary['worst_error_sd']['value']:.6f} "
                            "worst_P95="
                            f"{summary['worst_p95_absolute_error']['value']:.6f} "
                            f"pass={summary['passes']}",
                            flush=True,
                        )
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        if len(rows) != len(tasks) or len(per_n) != len(candidates):
            raise RuntimeError("Simulation result set is incomplete.")

        selected = next(
            (summary["n"] for summary in per_n if summary["passes"]), None
        )
        analytic_summary = summarize_analytic_checks(rows)
        guard_failed = not analytic_summary["passes"]
        status = (
            "implementation_error"
            if guard_failed
            else (
                "primary_coverage_precision_infeasible"
                if selected is None else "feasible"
            )
        )
        csv_bytes = scenarios_csv_bytes(rows)
        selected_bytes = canonical_json_bytes(selected) + b"\n"

        report = {
            "implementation_version": IMPLEMENTATION_VERSION,
            "contract_version": CONTRACT_VERSION,
            "purpose": (
                "Bound finite-cell measurement noise in an animal-averaged "
                "paired Brier contrast; not a power calculation or an "
                "estimate of model benefit."
            ),
            "status": status,
            "provenance": {
                "SLURM_JOB_ID": os.environ["SLURM_JOB_ID"],
                "hostname": socket.gethostname(),
                "started_at_utc": started_at,
                "simulation_completed_at_utc": utc_now(),
                "python_version": sys.version,
                "python_executable": sys.executable,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "cell_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256(manifest_bytes),
                },
                "precheck_summary": {
                    "path": str(precheck_path),
                    "sha256": sha256(precheck_bytes),
                },
                "script": {
                    "path": str(script_path),
                    "sha256": sha256(script_bytes),
                },
            },
            "execution": {
                "workers_requested": args.workers,
                "workers_used": workers_used,
                "multiprocessing_start_method": "spawn",
                "executor_map_chunksize": 1,
                "cpu_limits": cpu_limits,
                "thread_environment": {
                    name: os.environ[name] for name in THREAD_ENV
                },
            },
            "frozen_settings": frozen_settings(),
            "manifest_validation": manifest_summary,
            "allowed_candidates": candidates,
            "total_scenario_tasks": len(tasks),
            "development_animals": [
                {
                    "brain": animal.brain,
                    "numeric_brain_id": animal.numeric_brain_id,
                    "cell_count": len(animal.cell_ids),
                    "pools": [
                        {"pool_key": key, "pool_size": size}
                        for key, size in zip(
                            animal.pool_keys, animal.pool_sizes, strict=True
                        )
                    ],
                }
                for animal in animals
            ],
            "per_n": per_n,
            "selected_n_coverage": selected,
            "coverage_precision_feasible_by_thresholds": selected is not None,
            "selection_usable": selected is not None and not guard_failed,
            "analytic_check_summary": analytic_summary,
            "hashing": {
                "selected_value_encoding": (
                    "UTF-8 canonical JSON scalar followed by one LF; "
                    "null denotes no selected value"
                ),
                "selected_n_coverage_sha256": sha256(selected_bytes),
                "ncoverage_scenarios_csv_sha256": sha256(csv_bytes),
                "report_payload_definition": (
                    "Entire report excluding top-level report_payload_sha256; "
                    "json.dumps(sort_keys=True, ensure_ascii=True, "
                    "separators=(',', ':'), allow_nan=False); "
                    "UTF-8 without trailing newline"
                ),
                "complete_report_file_sha256_location": "console output",
            },
            "completed_run_exit_codes": {
                "0": "feasible and analytic guard passed",
                "2": "primary coverage/precision infeasible",
                "3": "analytic implementation-error guard failed",
            },
        }
        report["report_payload_sha256"] = sha256(canonical_json_bytes(report))
        report_bytes = (
            json.dumps(
                report,
                sort_keys=True,
                ensure_ascii=True,
                indent=2,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")

        write_exclusive(out_dir / OUTPUT_NAMES[0], report_bytes, created)
        write_exclusive(out_dir / OUTPUT_NAMES[1], csv_bytes, created)

        print(
            f"ncoverage_report.json SHA-256: {sha256(report_bytes)}",
            flush=True,
        )
        if selected is None:
            print(
                "selected n_coverage: none "
                "(primary coverage/precision infeasible by thresholds)",
                flush=True,
            )
        else:
            suffix = " (UNVALIDATED: analytic guard failed)" if guard_failed else ""
            print(f"selected n_coverage: {selected}{suffix}", flush=True)

        # Completed diagnostic outcomes deliberately retain both output files.
        if guard_failed:
            print(
                "ERROR: Analytic SD guard flagged "
                f"{analytic_summary['flagged_scenario_count']} scenarios; "
                "outputs retained for diagnosis.",
                file=sys.stderr,
                flush=True,
            )
            return 3
        return 2 if selected is None else 0

    except BaseException:
        # Validation, computation, writing, or console failures are operational
        # failures. Never remove a pre-existing output.
        rollback(created)
        raise


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("ERROR: Interrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)
