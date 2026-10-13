#!/usr/bin/env python3
"""Focused, login-node-safe tests for the sealed M2 scale benchmark."""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_v3_m2_search_scale_v1 as bench  # noqa: E402


class _Raises:
    def __init__(self, error: type[BaseException], match: str | None = None) -> None:
        self.error = error
        self.match = match

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> bool:
        if kind is None:
            raise AssertionError(f"expected {self.error.__name__} was not raised")
        if not issubclass(kind, self.error):
            return False
        if self.match is not None and re.search(self.match, str(value)) is None:
            raise AssertionError(f"exception did not match {self.match!r}: {value}")
        return True


def _raises(error: type[BaseException], match: str | None = None) -> _Raises:
    return _Raises(error, match)


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)


def _unit(value: float = 60.0) -> dict[str, object]:
    return {
        "status": "pass",
        "wall_seconds": value,
        "cpu_seconds": value / 2.0,
        "peak_rss_gib": 2.0,
        "scratch_gib": 0.01,
        "real_membership_opened": False,
        "existing_M0_artifact_or_telemetry_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def _units(value: float = 60.0) -> dict[str, dict[str, object]]:
    return {name: _unit(value) for name in bench.REQUIRED_UNITS}


def test_exact_52_and_80_trial_accounting_and_99_replay_projection() -> None:
    counts_52 = bench.trial_unit_counts(52)
    counts_80 = bench.trial_unit_counts(80)
    assert counts_52[bench.UNIT_REFERENCE] == 1
    assert counts_52[bench.UNIT_CORE] == 15
    assert counts_52[bench.UNIT_LOW_RANK] == 8
    assert counts_52[bench.UNIT_HYBRID] == 4
    assert counts_52["adaptive_worst_trial"] == 24
    assert counts_80["adaptive_worst_trial"] == 52

    projections = bench.project_52_and_80(_units())
    projected_52 = projections["52_trial_fully_triggered_branch"]
    projected_80 = projections["80_trial_cap"]
    # Shared setup + 52 trial units + diagnostic + refit-pair + report = 56 minutes.
    assert _close(projected_52["observed_controller_wall_hours"], 56 / 60)
    # A replay adds synthetic generation plus fresh M0 and M1: 59 minutes.
    assert _close(projected_52["one_fresh_replay_wall_hours"], 59 / 60)
    assert projected_52["replay_waves"] == 4
    assert projected_52["timing_safety_factor"] == 2.0
    # Shared setup + 80 trial units + three other fixed units = 84; replay = 87.
    assert _close(projected_80["observed_controller_wall_hours"], 84 / 60)
    assert _close(projected_80["one_fresh_replay_wall_hours"], 87 / 60)
    assert _close(
        projected_80["observed_plus_99_replays_allocated_cpu_core_hours"],
        2.0 * (84 + 99 * 87) * 2 / 60,
    )
    assert _close(
        projected_80["observed_plus_four_replay_waves_wall_clock_hours"],
        2.0 * (84 + 4 * 87) / 60,
    )
    assert projected_80["aggregate_memory_GiB"] == 256.0
    assert projected_80["requested_CPUs_per_task"] == 1
    assert projected_80["effective_numeric_threads_per_task"] == 1
    assert projected_80["scheduler_allocated_CPUs_per_task"] == 2
    assert projected_80["parallel_cores"] == 64
    assert _close(
        projected_80["projected_scratch_GiB_no_deletion_credit"],
        (84 + 99 * 87) * 0.01,
    )
    assert projected_80["all_ceiling_gates_pass"] is True


def test_resource_ceiling_failure_is_not_a_pass() -> None:
    units = _units()
    units[bench.UNIT_HYBRID] = _unit(100_000.0)
    projection = bench.project_search_resources(units, 80)
    assert projection["ceiling_gates"]["cpu_core_hours"] is False
    assert projection["ceiling_gates"]["wall_clock_hours"] is False
    assert projection["all_ceiling_gates_pass"] is False


def test_measured_unit_inventory_is_exact_and_nonpassing_units_are_rejected() -> None:
    units = _units()
    units.pop(bench.UNIT_HYBRID)
    with _raises(bench.BenchmarkError, match="inventory changed"):
        bench.normalize_measured_units(units)
    units = _units()
    units[bench.UNIT_CORE]["status"] = "unavailable"
    with _raises(bench.BenchmarkError, match="nonpassing"):
        bench.normalize_measured_units(units)


@dataclass(frozen=True)
class _Cell:
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


class _Basis:
    def __init__(self, k: int) -> None:
        rows = []
        for chosen in itertools.combinations(range(bench.EXPECTED_PANEL_SIZE), k):
            row = np.zeros(bench.EXPECTED_PANEL_SIZE, dtype=np.uint8)
            row[list(chosen)] = 1
            rows.append(row)
        self.membership = np.stack(rows)

    @property
    def count(self) -> int:
        return int(self.membership.shape[0])


class _Library:
    def __init__(self) -> None:
        self.cache: dict[int, _Basis] = {}

    def basis(self, k: int) -> _Basis:
        if k not in self.cache:
            self.cache[k] = _Basis(k)
        return self.cache[k]


class _Geometry:
    def __init__(self, library: _Library, pair_geometry: object) -> None:
        self.library = library

    def uniform_moments(self, cell: _Cell) -> object:
        return object()


class _FakeM1:
    CandidateLibrary = _Library
    GeometryEngine = _Geometry

    @staticmethod
    def prepare_data(cells: object, moments: object) -> object:
        return SimpleNamespace(cells=tuple(cells))

    @staticmethod
    def planted_parameters() -> np.ndarray:
        return np.zeros(1)

    @staticmethod
    def sample_candidate_index(
        theta: object,
        prepared: object,
        cell_index: int,
        library: _Library,
        geometry: object,
        uniform: float,
    ) -> int:
        count = library.basis(prepared.cells[cell_index].k).count
        return min(int(uniform * count), count - 1)


def _poisoned_cells() -> tuple[_Cell, ...]:
    cells = []
    for index in range(bench.EXPECTED_ROWS):
        k = index % (bench.EXPECTED_PANEL_SIZE + 1)
        cells.append(
            _Cell(
                identifier=str(1000 + index),
                group=str(index % bench.EXPECTED_GROUPS),
                layer="L2",
                hemisphere="left",
                soma_um=(1.0, 2.0, 3.0),
                soma_distance_mm=np.zeros(bench.EXPECTED_PANEL_SIZE),
                k=k,
                observed_index=math.comb(bench.EXPECTED_PANEL_SIZE, k),
                segment_unmapped=index % 2,
            )
        )
    return tuple(cells)


def test_deterministic_synthetic_memberships_preserve_every_actual_k() -> None:
    cells = _poisoned_cells()
    first = bench.construct_synthetic_memberships(
        cells, object(), seed=12345, m1_module=_FakeM1
    )
    second = bench.construct_synthetic_memberships(
        cells, object(), seed=12345, m1_module=_FakeM1
    )
    assert np.array_equal(first.membership_matrix, second.membership_matrix)
    assert first.membership_matrix.shape == (bench.EXPECTED_ROWS, 16)
    assert not first.membership_matrix.flags.writeable
    assert np.array_equal(
        first.membership_matrix.sum(axis=1), np.asarray([cell.k for cell in cells])
    )
    assert [cell.observed_index for cell in first.cells] == [
        cell.observed_index for cell in second.cells
    ]
    assert all(
        0 <= synthetic.observed_index < math.comb(16, original.k)
        for original, synthetic in zip(cells, first.cells, strict=True)
    )
    # Inputs retain the one-past-end poison and therefore never name a real set.
    assert all(cell.observed_index == math.comb(16, cell.k) for cell in cells)
    assert first.witness["cell_ids_candidate_indexes_K_values_or_memberships_emitted"] is False


def test_synthetic_constructor_rejects_an_input_membership_identity() -> None:
    cells = list(_poisoned_cells())
    cells[0] = _Cell(**{**cells[0].__dict__, "observed_index": 0})
    with _raises(bench.BenchmarkError, match="not membership-blind"):
        bench.construct_synthetic_memberships(
            cells, object(), seed=1, m1_module=_FakeM1
        )


def test_prerequisite_payload_predicates_are_fail_closed() -> None:
    core = {
        "status": "pass",
        "overall_gate_status": "pass",
        "mode": "full",
        "stage": "v3_M2_target_free_frozen_core_contract_qualification",
        "execution": {"slurm_job_id": "11"},
        "gate_inventory": {
            "all_required_gates_present_and_passed": True,
            "required_gate_count": 8,
        },
        "real_target_membership_or_M2_effects_opened": False,
        "actual_K_opened": False,
        "M0_full_refit_or_telemetry_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
        "controller_replay_qualification_included": False,
    }
    bench._validate_core_payload(core, "11")
    core["actual_K_opened"] = True
    with _raises(bench.BenchmarkError):
        bench._validate_core_payload(core, "11")

    actual = {
        "status": "pass",
        "stage": "v3_M2_actual_K_only_residual_rank_gate_v1",
        "SLURM_job_id": "12",
        "qualified_core_job_id": "11",
        "rows": 194,
        "groups": 12,
        "actual_K_opened": True,
        "cell_ids_groups_segment_flags_or_K_values_emitted": False,
        "realized_positive_set_membership_opened": False,
        "candidate_scores_or_M2_fit_computed": False,
        "M0_full_refit_or_telemetry_opened": False,
        "SWC_opened": False,
        "final_or_unassigned_opened": False,
        "rank_gate": {
            "status": "pass",
            "training_folds_evaluated": 12,
            "training_folds_passed": 12,
            "expected_rank": 104,
            "computed_rank_at_tau_over_10_unique": [104],
            "computed_rank_at_tau_unique": [104],
            "computed_rank_at_ten_tau_unique": [104],
            "rank_stable_in_every_fold": True,
            "rank_computed_not_hard_coded": True,
            "fold_level_rows_emitted": False,
            "cell_ids_groups_or_K_values_emitted": False,
        },
    }
    assert bench._validate_actual_k_payload(actual, "12") == "11"
    actual["rank_gate"]["computed_rank_at_tau_unique"] = [103]
    with _raises(bench.BenchmarkError):
        bench._validate_actual_k_payload(actual, "12")

    optional = {
        "status": "pass",
        "overall_gate_status": "pass",
        "stage": "v3_M2_target_free_optional_family_qualification_v1",
        "execution": {"slurm_job_id": "13"},
        "gate_inventory": {
            "required": [
                "optional_trigger_diagnostic",
                "low_rank_population_recovery",
                "sparse_plus_low_rank_population_recovery",
            ],
            "passing": [
                "optional_trigger_diagnostic",
                "low_rank_population_recovery",
                "sparse_plus_low_rank_population_recovery",
            ],
            "required_gate_count": 3,
            "all_required_gates_present_and_passed": True,
        },
        "gates": {
            "optional_trigger_diagnostic": {"status": "pass"},
            "low_rank_population_recovery": {"status": "pass"},
            "sparse_plus_low_rank_population_recovery": {"status": "pass"},
        },
        "real_target_membership_or_M2_effects_opened": False,
        "actual_K_opened": False,
        "M0_full_refit_or_telemetry_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }
    bench._validate_optional_payload(optional, "13")
    optional["gates"]["low_rank_population_recovery"]["status"] = "fail"
    with _raises(bench.BenchmarkError):
        bench._validate_optional_payload(optional, "13")


def test_aggregate_report_seal_and_atomic_output(tmp_path: Path) -> None:
    safe = {
        "status": "blocked_missing_execution_units",
        "rows": 194,
        "actual_K_opened": False,
    }
    bench.validate_aggregate_report(safe)
    path = tmp_path / "summary.json"
    bench._atomic_json(path, safe)
    assert json.loads(path.read_text(encoding="utf-8")) == safe
    with _raises(bench.BenchmarkError, match="prohibited key"):
        bench.validate_aggregate_report({"status": "pass", "cell_key": "secret"})


def test_parser_requires_the_exact_sealed_cli_and_disallows_abbreviation() -> None:
    parser = bench.build_parser()
    names = (
        "contract",
        "addendum",
        "hybrid-clarification",
        "hybrid-rank-clarification",
        "inventory",
        "interface-root",
        "geometry-root",
        "target-root",
        "core-report",
        "actual-k-report",
        "adequacy-report",
        "optional-report",
        "output-root",
    )
    argv = []
    for name in names:
        argv.extend((f"--{name}", f"/{name}"))
    parsed = parser.parse_args(argv)
    assert parsed.actual_k_report == Path("/actual-k-report")
    with contextlib.redirect_stderr(io.StringIO()):
        with _raises(SystemExit):
            parser.parse_args(["--contr", "/contract"])


def run() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        if test.__name__ == "test_aggregate_report_seal_and_atomic_output":
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
        else:
            test()
    print(f"v3 M2 scale benchmark tests: PASS ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    run()
