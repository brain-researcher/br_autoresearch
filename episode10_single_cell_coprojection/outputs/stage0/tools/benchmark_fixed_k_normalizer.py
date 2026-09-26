#!/usr/bin/env python3
"""Outcome-blind exact-normalization benchmark for EP10.

The benchmark exhaustively evaluates a deterministic synthetic binary pairwise
log score over every state with a fixed number of active coordinates.  It does
not read morphology files, target calls, or projection outcomes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


JSON_SAFE_INTEGER = (1 << 53) - 1
PRIMARY_COORDINATE_IDS = (
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


def synthetic_coefficients(
    dimension: int, seed: int
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return deterministic, bounded coefficients without consulting data."""
    if dimension < 1:
        raise ValueError("dimension must be positive")

    seed_term = seed % 10_007
    fields = tuple(
        (((seed_term + 37 * (i + 1) + 13 * (i + 1) ** 2) % 211) - 105)
        / 300.0
        for i in range(dimension)
    )
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i + 1, dimension):
            numerator = (
                seed_term
                + 19 * (i + 1)
                + 41 * (j + 1)
                + 11 * (i + 1) * (j + 1)
            ) % 199 - 99
            value = numerator / 800.0
            matrix[i][j] = value
            matrix[j][i] = value
    return fields, tuple(tuple(row) for row in matrix)


def log_score(
    active: Sequence[int],
    fields: Sequence[float],
    interactions: Sequence[Sequence[float]],
) -> float:
    score = 0.0
    active_count = len(active)
    for left_position in range(active_count):
        left = active[left_position]
        score += fields[left]
        row = interactions[left]
        for right_position in range(left_position + 1, active_count):
            score += row[active[right_position]]
    return score


class StreamingLogSumExp:
    """Stable constant-memory log-sum-exp accumulator."""

    def __init__(self) -> None:
        self.maximum = -math.inf
        self.scaled_sum = 0.0
        self.count = 0

    def add(self, value: float) -> None:
        if value > self.maximum:
            if self.count:
                self.scaled_sum = self.scaled_sum * math.exp(self.maximum - value) + 1.0
            else:
                self.scaled_sum = 1.0
            self.maximum = value
        else:
            self.scaled_sum += math.exp(value - self.maximum)
        self.count += 1

    def value(self) -> float:
        if not self.count:
            raise ValueError("log-sum-exp is undefined for an empty stream")
        return self.maximum + math.log(self.scaled_sum)


def deep_size_bytes(value: object, seen: set[int] | None = None) -> int:
    """Estimate owned Python payload, avoiding double-counted references."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(deep_size_bytes(item, seen) for item in value)
    elif isinstance(value, dict):
        size += sum(
            deep_size_bytes(key, seen) + deep_size_bytes(item, seen)
            for key, item in value.items()
        )
    elif hasattr(value, "__dict__"):
        size += deep_size_bytes(vars(value), seen)
    return size


def working_set_estimate(
    dimension: int,
    k: int,
    configuration_count: int,
    fields: Sequence[float],
    interactions: Sequence[Sequence[float]],
) -> dict[str, int | str]:
    model_bytes = deep_size_bytes((fields, interactions))
    combination_tuple_bytes = sys.getsizeof(tuple(range(k)))
    iterator_bytes = sys.getsizeof(itertools.combinations(range(dimension), k))
    iterator_pool_tuple_bytes = sys.getsizeof(tuple(range(dimension)))
    accumulator_bytes = deep_size_bytes(StreamingLogSumExp())
    streaming_bytes = (
        model_bytes
        + combination_tuple_bytes
        + iterator_bytes
        + iterator_pool_tuple_bytes
        + accumulator_bytes
    )
    return {
        "method": (
            "sys.getsizeof recursive model payload plus one combination tuple, "
            "the iterator's coordinate pool, one itertools iterator, and one "
            "accumulator; this steady-state estimate excludes interpreter, allocator "
            "overhead, and coefficient-construction temporaries"
        ),
        "coefficient_model_bytes": model_bytes,
        "combination_tuple_bytes": combination_tuple_bytes,
        "iterator_bytes": iterator_bytes,
        "iterator_pool_tuple_bytes": iterator_pool_tuple_bytes,
        "accumulator_bytes": accumulator_bytes,
        "estimated_peak_steady_state_streaming_payload_bytes": streaming_bytes,
        "naive_dense_binary_uint8_materialization_bytes": configuration_count
        * dimension,
        "naive_log_weight_float64_materialization_bytes": configuration_count * 8,
    }


def coefficient_construction_estimate_bytes(dimension: int) -> int:
    """Approximate payload while list and tuple coefficient matrices coexist."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    list_row_bytes = sys.getsizeof([None] * dimension)
    tuple_row_bytes = sys.getsizeof(tuple([None] * dimension))
    list_matrix_bytes = sys.getsizeof([None] * dimension) + dimension * list_row_bytes
    tuple_matrix_bytes = (
        sys.getsizeof(tuple([None] * dimension)) + dimension * tuple_row_bytes
    )
    unique_float_count = dimension + math.comb(dimension, 2) + 1
    float_bytes = unique_float_count * sys.getsizeof(0.0)
    field_tuple_bytes = tuple_row_bytes
    return list_matrix_bytes + tuple_matrix_bytes + float_bytes + field_tuple_bytes


def max_rss_bytes() -> int:
    # Sherlock runs Linux, where ru_maxrss is reported in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def benchmark_fixed_k(
    dimension: int,
    k: int,
    fields: Sequence[float],
    interactions: Sequence[Sequence[float]],
) -> dict[str, object]:
    if not 0 <= k <= dimension:
        raise ValueError(f"K={k} is outside 0..{dimension}")

    expected_count = math.comb(dimension, k)
    rss_before = max_rss_bytes()
    accumulator = StreamingLogSumExp()
    started = time.perf_counter()
    for active in itertools.combinations(range(dimension), k):
        accumulator.add(log_score(active, fields, interactions))
    elapsed = time.perf_counter() - started
    rss_after = max_rss_bytes()

    if accumulator.count != expected_count:
        raise RuntimeError(
            f"enumerated {accumulator.count} states, expected {expected_count}"
        )

    return {
        "k": k,
        "configuration_count": expected_count,
        "enumerated_configuration_count": accumulator.count,
        "exact_log_partition_nats": accumulator.value(),
        "wall_seconds": elapsed,
        "configurations_per_second": accumulator.count / elapsed,
        "process_max_rss_before_bytes": rss_before,
        "process_max_rss_after_bytes": rss_after,
        "working_set_estimate": working_set_estimate(
            dimension, k, expected_count, fields, interactions
        ),
    }


def combinatorial_counts(dimension: int) -> list[dict[str, object]]:
    rows = []
    for k in range(dimension + 1):
        count = math.comb(dimension, k)
        rows.append(
            {
                "k": k,
                "configuration_count_decimal": str(count),
                "fits_json_safe_integer": count <= JSON_SAFE_INTEGER,
            }
        )
    return rows


def benchmark_suite(
    primary_dimension: int,
    wide_dimension: int,
    wide_k: int,
    wide_state_cap: int,
    primary_state_cap: int,
    maximum_dimension: int,
    coefficient_construction_cap_bytes: int,
    seed: int,
) -> dict[str, object]:
    if min(wide_state_cap, primary_state_cap, coefficient_construction_cap_bytes) < 0:
        raise ValueError("state and byte caps must be non-negative")
    if maximum_dimension < 1:
        raise ValueError("maximum dimension must be positive")
    if primary_dimension < 1 or wide_dimension < 1:
        raise ValueError("dimensions must be positive")
    if not 0 <= wide_k <= wide_dimension:
        raise ValueError("wide K must be within 0..wide dimension")
    if primary_dimension > maximum_dimension or wide_dimension > maximum_dimension:
        raise ValueError(
            f"requested dimension exceeds declared maximum {maximum_dimension}"
        )

    primary_total_expected = 1 << primary_dimension
    if primary_total_expected > primary_state_cap:
        raise ValueError(
            f"primary panel requires {primary_total_expected} states, exceeding "
            f"declared cap {primary_state_cap}"
        )
    primary_coefficient_estimate = coefficient_construction_estimate_bytes(
        primary_dimension
    )
    primary_coefficient_guarded = 2 * primary_coefficient_estimate
    if primary_coefficient_guarded > coefficient_construction_cap_bytes:
        raise ValueError(
            "guarded primary coefficient construction estimate exceeds declared byte cap"
        )

    primary_fields, primary_interactions = synthetic_coefficients(
        primary_dimension, seed
    )
    primary_started = time.perf_counter()
    primary_rows = [
        benchmark_fixed_k(
            primary_dimension, k, primary_fields, primary_interactions
        )
        for k in range(primary_dimension + 1)
    ]
    primary_elapsed = time.perf_counter() - primary_started
    primary_total_observed = sum(
        int(row["enumerated_configuration_count"]) for row in primary_rows
    )
    if primary_total_observed != primary_total_expected:
        raise RuntimeError(
            f"primary enumeration total {primary_total_observed} != "
            f"2^{primary_dimension}={primary_total_expected}"
        )

    wide_count = math.comb(wide_dimension, wide_k)
    wide_coefficient_estimate = coefficient_construction_estimate_bytes(wide_dimension)
    wide_coefficient_guarded = 2 * wide_coefficient_estimate
    wide_result: dict[str, object]
    if wide_count > wide_state_cap:
        wide_result = {
            "status": "skipped_state_cap",
            "dimension": wide_dimension,
            "k": wide_k,
            "configuration_count": wide_count,
            "declared_state_cap": wide_state_cap,
            "reason": "configuration count exceeds the predeclared enumeration cap",
        }
    else:
        if wide_coefficient_guarded > coefficient_construction_cap_bytes:
            raise ValueError(
                "guarded wide coefficient construction estimate exceeds declared byte cap"
            )
        wide_fields, wide_interactions = synthetic_coefficients(wide_dimension, seed)
        wide_result = {
            "status": "completed",
            "dimension": wide_dimension,
            "declared_state_cap": wide_state_cap,
            **benchmark_fixed_k(
                wide_dimension, wide_k, wide_fields, wide_interactions
            ),
        }

    payload = {
        "benchmark": "EP10 synthetic fixed-K binary pairwise exact normalizer",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_blind": {
            "synthetic_coefficients_only": True,
            "morphology_files_read": False,
            "projection_outcomes_read": False,
        },
        "normalization_semantics": {
            "state": "binary vector x with sum(x)=K",
            "log_weight": "sum_i h_i*x_i + sum_{i<j} J_ij*x_i*x_j",
            "log_partition": "log(sum_{x: sum(x)=K} exp(log_weight(x)))",
            "exactness": (
                "every fixed-K state is exhaustively enumerated; reported log "
                "partitions are float64 values, not symbolic values"
            ),
            "enumeration_order": "itertools.combinations lexicographic order",
            "memory_strategy": "stream one combination at a time",
        },
        "synthetic_coefficient_specification": {
            "seed": seed,
            "field_denominator": 300,
            "pair_denominator": 800,
            "generator": "documented integer modular formulas in this script",
        },
        "preflight_limits": {
            "maximum_dimension": maximum_dimension,
            "primary_total_state_cap": primary_state_cap,
            "wide_fixed_k_state_cap": wide_state_cap,
            "coefficient_construction_cap_bytes": coefficient_construction_cap_bytes,
            "coefficient_estimate_guard_factor": 2,
            "coefficient_estimate_scope": (
                "approximate Python payload for coexisting list and tuple matrices; "
                "the byte-cap gate uses the reported 2x guarded value"
            ),
            "primary_coefficient_construction_estimate_bytes": primary_coefficient_estimate,
            "primary_guarded_coefficient_bytes": primary_coefficient_guarded,
            "wide_coefficient_construction_estimate_bytes": wide_coefficient_estimate,
            "wide_guarded_coefficient_bytes": wide_coefficient_guarded,
        },
        "runtime": {
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "primary_panel": {
            "dimension": primary_dimension,
            "coordinate_ids": (
                list(PRIMARY_COORDINATE_IDS)
                if primary_dimension == len(PRIMARY_COORDINATE_IDS)
                else None
            ),
            "coordinate_binding": (
                "frozen order from outputs/stage0/projection_observation_contract.yaml"
                if primary_dimension == len(PRIMARY_COORDINATE_IDS)
                else "dimension-only synthetic diagnostic"
            ),
            "k_range": [0, primary_dimension],
            "expected_total_configurations": primary_total_expected,
            "enumerated_total_configurations": primary_total_observed,
            "wall_seconds_all_k": primary_elapsed,
            "by_k": primary_rows,
        },
        "wide_panel": {
            "dimension": wide_dimension,
            "all_k_combinatorial_counts": combinatorial_counts(wide_dimension),
            "bounded_streaming_case": wide_result,
        },
    }
    payload["runtime"]["process_peak_rss_bytes_after_payload_assembly"] = max_rss_bytes()
    return payload


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        os.fchmod(handle.fileno(), 0o660)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def run_self_check() -> None:
    dimension = 6
    fields, interactions = synthetic_coefficients(dimension, 17)
    observed_total = 0
    for k in range(dimension + 1):
        result = benchmark_fixed_k(dimension, k, fields, interactions)
        expected_count = math.comb(dimension, k)
        if result["enumerated_configuration_count"] != expected_count:
            raise AssertionError("fixed-K state count mismatch")
        direct_weights = [
            math.exp(log_score(active, fields, interactions))
            for active in itertools.combinations(range(dimension), k)
        ]
        direct_log_partition = math.log(math.fsum(direct_weights))
        if not math.isclose(
            float(result["exact_log_partition_nats"]),
            direct_log_partition,
            rel_tol=1e-13,
            abs_tol=1e-13,
        ):
            raise AssertionError("streaming and direct log partitions differ")
        observed_total += expected_count
    if observed_total != 1 << dimension:
        raise AssertionError("sum_K comb(D,K) != 2^D")
    if math.comb(56, 5) != 3_819_816:
        raise AssertionError("unexpected 56-choose-5 count")
    if len(PRIMARY_COORDINATE_IDS) != 16:
        raise AssertionError("primary coordinate binding is not 16-dimensional")
    if max(math.comb(16, k) for k in range(17)) != 12_870:
        raise AssertionError("unexpected maximum 16-coordinate subset count")
    print("self-check passed: exhaustive counts and log partitions agree for D=6")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--primary-dimension", type=int, default=16)
    parser.add_argument("--wide-dimension", type=int, default=56)
    parser.add_argument("--wide-k", type=int, default=5)
    parser.add_argument("--wide-state-cap", type=int, default=4_000_000)
    parser.add_argument("--primary-state-cap", type=int, default=1_000_000)
    parser.add_argument("--maximum-dimension", type=int, default=256)
    parser.add_argument(
        "--coefficient-construction-cap-bytes", type=int, default=64 * 1024 * 1024
    )
    parser.add_argument("--seed", type=int, default=20_260_926)
    parser.add_argument(
        "--timing-repetitions",
        type=int,
        default=1,
        help=(
            "repeat the complete synthetic suite with consecutive seeds; 99 "
            "measures exact-normalization throughput at the null-count scale"
        ),
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        run_self_check()
        return 0
    if args.output_json is None:
        raise SystemExit("--output-json is required unless --self-check is used")
    if args.primary_dimension < 1 or args.wide_dimension < 1:
        raise SystemExit("dimensions must be positive")
    if not 0 <= args.wide_k <= args.wide_dimension:
        raise SystemExit("--wide-k must be within 0..--wide-dimension")
    if not 1 <= args.timing_repetitions <= 200:
        raise SystemExit("--timing-repetitions must be within 1..200")

    payload = None
    repetition_rows: list[dict[str, object]] = []
    repeated_started = time.perf_counter()
    for repetition in range(args.timing_repetitions):
        repetition_started = time.perf_counter()
        observed = benchmark_suite(
            primary_dimension=args.primary_dimension,
            wide_dimension=args.wide_dimension,
            wide_k=args.wide_k,
            wide_state_cap=args.wide_state_cap,
            primary_state_cap=args.primary_state_cap,
            maximum_dimension=args.maximum_dimension,
            coefficient_construction_cap_bytes=args.coefficient_construction_cap_bytes,
            seed=args.seed + repetition,
        )
        if payload is None:
            payload = observed
        wide_result = observed["wide_panel"]["bounded_streaming_case"]
        repetition_rows.append(
            {
                "repetition": repetition + 1,
                "seed": args.seed + repetition,
                "primary_all_k_wall_seconds": observed["primary_panel"][
                    "wall_seconds_all_k"
                ],
                "wide_fixed_k_wall_seconds": wide_result.get("wall_seconds"),
                "suite_wall_seconds": time.perf_counter() - repetition_started,
            }
        )
    assert payload is not None
    suite_seconds = [float(row["suite_wall_seconds"]) for row in repetition_rows]
    payload["replicated_timing"] = {
        "repetitions": args.timing_repetitions,
        "seed_first": args.seed,
        "seed_last": args.seed + args.timing_repetitions - 1,
        "purpose": (
            "repeat the outcome-blind exact-normalization workload to measure "
            "throughput at the 99-null planning scale; this is not a benchmark "
            "of a complete null-search rerun"
        ),
        "total_wall_seconds": time.perf_counter() - repeated_started,
        "suite_wall_seconds_min": min(suite_seconds),
        "suite_wall_seconds_median": statistics.median(suite_seconds),
        "suite_wall_seconds_max": max(suite_seconds),
        "runs": repetition_rows,
    }
    atomic_write_json(args.output_json, payload)
    wide = payload["wide_panel"]["bounded_streaming_case"]
    print(f"wrote {args.output_json}")
    print(
        "primary D={} enumerated={} states in {:.6f}s".format(
            args.primary_dimension,
            payload["primary_panel"]["enumerated_total_configurations"],
            payload["primary_panel"]["wall_seconds_all_k"],
        )
    )
    print(
        "wide D={} K={} count={} status={}".format(
            args.wide_dimension,
            args.wide_k,
            math.comb(args.wide_dimension, args.wide_k),
            wide["status"],
        )
    )
    print(
        "timing repetitions={} total={:.6f}s median={:.6f}s".format(
            args.timing_repetitions,
            payload["replicated_timing"]["total_wall_seconds"],
            payload["replicated_timing"]["suite_wall_seconds_median"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
