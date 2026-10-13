#!/usr/bin/env python3
"""Light mocked tests for the central M2 scale-benchmark execution bridge."""

from __future__ import annotations

import copy
import inspect
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_v3_m2_search_scale_v1 as scaffold  # noqa: E402
import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import v3_m2_benchmark_replay_v1 as replay  # noqa: E402
import v3_m2_scale_benchmark_units_v1 as bridge  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


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
    sampling_uniform: float


def _cells() -> tuple[_Cell, ...]:
    return tuple(
        _Cell(
            identifier=f"synthetic_{index:03d}",
            group=m1.GROUP_ORDER[index % len(m1.GROUP_ORDER)],
            layer="L2",
            hemisphere="left" if index % 2 == 0 else "right",
            soma_um=(float(index), 1.0, 2.0),
            soma_distance_mm=np.linspace(0.0, 1.0, m1.PANEL_SIZE),
            k=0,
            observed_index=0,
            segment_unmapped=index % 2,
            sampling_uniform=(index + 0.5) / bridge.EXPECTED_ROWS,
        )
        for index in range(bridge.EXPECTED_ROWS)
    )


def _pair_geometry() -> m1.PairGeometry:
    dimension = m1.PANEL_SIZE
    return m1.PairGeometry(
        distance=np.arange(dimension * dimension, dtype=float).reshape(
            dimension, dimension
        ),
        cross_side=np.zeros((dimension, dimension), dtype=float),
        homotopic=np.eye(dimension, dtype=float),
        ordered_edges=((0.0, 0, 1),),
    )


def _outcome() -> scaffold.SyntheticOutcome:
    membership = np.zeros(
        (bridge.EXPECTED_ROWS, m1.PANEL_SIZE), dtype=np.uint8
    )
    membership.setflags(write=False)
    return scaffold.SyntheticOutcome(
        cells=_cells(),
        membership_matrix=membership,
        witness={"status": "pass"},
    )


def _context() -> scaffold.BenchmarkContext:
    outcome = _outcome()
    pair_geometry = _pair_geometry()
    session = bridge.issue_scale_benchmark_session()
    returned, timing = bridge.time_synthetic_constructor(
        session,
        pair_geometry,
        lambda: outcome,
    )
    context = scaffold.BenchmarkContext(
        cells=returned.cells,
        membership_matrix=returned.membership_matrix,
        pair_geometry=pair_geometry,
        synthetic_seed=scaffold.SYNTHETIC_SEED,
        synthetic_generation_timing=timing,
        scale_benchmark_session=session,
    )
    bridge.bind_scale_benchmark_context(context)
    return context


def _fake_run(context: object) -> bridge.ScaleBenchmarkRun:
    measurements = {
        name: bridge._issue_measurement(1.0, 0.5, 0.25, 0.0)
        for name in bridge.REQUIRED_UNITS
    }
    return bridge._issue_run(
        context,
        measurements,
        {"status": "pass"},
        (),
    )


class ScaleBenchmarkBridgeTests(unittest.TestCase):
    def test_scaffold_factory_owns_constructor_timing_and_exact_outcome(self) -> None:
        expected = _outcome()
        geometry = _pair_geometry()
        calls = 0
        original_constructor = scaffold.construct_synthetic_memberships
        original_loader = scaffold._require_local_module

        def constructor(cells: object, pair_geometry: object) -> object:
            nonlocal calls
            calls += 1
            self.assertIs(pair_geometry, geometry)
            return expected

        scaffold.construct_synthetic_memberships = constructor
        scaffold._require_local_module = lambda name: bridge
        try:
            returned, context = scaffold.construct_benchmark_context(
                expected.cells,
                geometry,
            )
        finally:
            scaffold.construct_synthetic_memberships = original_constructor
            scaffold._require_local_module = original_loader
        self.assertEqual(calls, 1)
        self.assertIs(returned, expected)
        self.assertIs(context.cells, expected.cells)
        self.assertIs(context.membership_matrix, expected.membership_matrix)
        bridge._require_bound_session(context)

    def test_constructor_timing_cannot_be_cross_spliced(self) -> None:
        first = _outcome()
        second = _outcome()
        geometry = _pair_geometry()
        session = bridge.issue_scale_benchmark_session()
        _, timing = bridge.time_synthetic_constructor(
            session,
            geometry,
            lambda: first,
        )
        forged_context = scaffold.BenchmarkContext(
            cells=second.cells,
            membership_matrix=second.membership_matrix,
            pair_geometry=geometry,
            synthetic_seed=scaffold.SYNTHETIC_SEED,
            synthetic_generation_timing=timing,
            scale_benchmark_session=session,
        )
        with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "another constructor"):
            bridge.bind_scale_benchmark_context(forged_context)

    def test_one_central_run_is_shared_by_both_thin_hook_views(self) -> None:
        context = _context()
        calls = 0
        original = bridge._execute_scale_benchmark_run

        def execute(bound_context: object) -> bridge.ScaleBenchmarkRun:
            nonlocal calls
            calls += 1
            self.assertIs(bound_context, context)
            return _fake_run(bound_context)

        bridge._execute_scale_benchmark_run = execute
        try:
            executor_view = executor.benchmark_scale_units(context)
            optional_view = optional.benchmark_scale_units(context)
            repeat_view = executor.benchmark_scale_units(context)
        finally:
            bridge._execute_scale_benchmark_run = original
        self.assertEqual(calls, 1)
        self.assertEqual(set(executor_view), set(bridge.EXECUTOR_UNITS))
        self.assertEqual(set(optional_view), set(bridge.OPTIONAL_UNITS))
        self.assertEqual(executor_view, repeat_view)
        self.assertTrue(
            all(
                item["peak_rss_semantics"]
                == "process_lifetime_high_water_conservative"
                for item in (*executor_view.values(), *optional_view.values())
            )
        )

    def test_context_session_measurement_and_run_forgery_fail_closed(self) -> None:
        context = _context()
        with self.assertRaises(bridge.ScaleBenchmarkError):
            bridge._require_bound_session(replace(context))

        forged_measurement = bridge.AggregateMeasurement(
            1.0,
            1.0,
            1.0,
            _seal=bridge._MEASUREMENT_SEAL,
        )
        with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "not issued"):
            bridge._require_measurement(forged_measurement)

        measurements = {
            name: bridge._issue_measurement(1.0, 1.0, 1.0)
            for name in bridge.REQUIRED_UNITS
        }
        forged_run = bridge.ScaleBenchmarkRun(
            context,
            measurements,
            {"status": "pass"},
            (),
            _seal=bridge._RUN_SEAL,
        )
        with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "not issued"):
            bridge._require_run(forged_run)

        session = context.scale_benchmark_session
        with self.assertRaises(bridge.ScaleBenchmarkError):
            copy.copy(session)
        session._failed = True
        with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "changed"):
            bridge._require_bound_session(context)

    def test_context_mutation_is_rejected_before_cached_view(self) -> None:
        context = _context()
        original = bridge._execute_scale_benchmark_run
        bridge._execute_scale_benchmark_run = _fake_run
        try:
            bridge.executor_scale_units(context)
            context.membership_matrix.setflags(write=True)
            with self.assertRaises(bridge.ScaleBenchmarkError):
                bridge.optional_scale_units(context)
        finally:
            context.membership_matrix.setflags(write=False)
            bridge._execute_scale_benchmark_run = original

    def test_componentwise_max_and_refit_serial_composition(self) -> None:
        left = bridge._issue_measurement(10.0, 1.0, 2.0, 4.0)
        right = bridge._issue_measurement(1.0, 9.0, 8.0, 2.0)
        rank_neighbor = bridge._issue_measurement(3.0, 4.0, 5.0, 1.0)
        fourth_family = bridge._issue_measurement(2.0, 2.0, 3.0, 3.0)
        maximum = bridge._componentwise_max(
            (left, right, rank_neighbor, fourth_family)
        )
        self.assertEqual(
            (
                maximum.wall_seconds,
                maximum.cpu_seconds,
                maximum.peak_rss_gib,
                maximum.scratch_gib,
            ),
            (10.0, 9.0, 8.0, 4.0),
        )
        total = bridge._serial_sum((left, right))
        self.assertEqual(
            (
                total.wall_seconds,
                total.cpu_seconds,
                total.peak_rss_gib,
                total.scratch_gib,
            ),
            (11.0, 10.0, 8.0, 6.0),
        )

    def test_exact_unit_and_anchor_inventories(self) -> None:
        self.assertEqual(len(bridge.REQUIRED_UNITS), 11)
        self.assertEqual(len(set(bridge.REQUIRED_UNITS)), 11)
        self.assertEqual(len(controller.coverage_trials()[1:]), 15)
        self.assertEqual(len(controller.low_rank_anchor_configurations()), 8)
        configurations = bridge._hybrid_configurations(
            2,
            tuple(m2.PAIR_INDEX[:16]),
        )
        self.assertEqual(len(configurations), 4)
        self.assertEqual(
            tuple((value.rho, value.sparse_rho) for value in configurations),
            ((1.0, 1.0), (1.0, 0.001), (0.001, 1.0), (0.001, 0.001)),
        )
        target, precursor = bridge._rank_neighbor_full_refit_configurations(
            configurations[-1]
        )
        self.assertEqual(bridge.RANK_NEIGHBOR_SOURCE_RANK, 6)
        self.assertEqual(bridge.RANK_NEIGHBOR_TARGET_RANK, 8)
        self.assertEqual(
            (target.rank, target.rho, target.sparse_rho, target.sparse_edges),
            (
                8,
                configurations[-1].rho,
                configurations[-1].sparse_rho,
                configurations[-1].sparse_edges,
            ),
        )
        self.assertEqual(
            (precursor.family, precursor.rank, precursor.rho),
            ("low_rank", 8, 0.1),
        )

    def test_repeat_validation_fails_each_frozen_tolerance(self) -> None:
        cold = SimpleNamespace(objective=1.0, beta=np.zeros(m2.PAIR_DIMENSION))
        warm = SimpleNamespace(objective=1.0, beta=np.zeros(m2.PAIR_DIMENSION))
        original_compare = m2.compare_conditional_probabilities
        original_contrast = bridge._equal_group_pair_contrast
        workload = SimpleNamespace(
            candidates=object(), geometry=object(), pairs=object()
        )
        probability_error = [0.0]
        contrast_values = [
            np.zeros(m2.PAIR_DIMENSION),
            np.zeros(m2.PAIR_DIMENSION),
        ]

        def compare(*args: object, **kwargs: object) -> dict[str, float]:
            return {
                "maximum_candidate_probability_absolute_error": probability_error[0]
            }

        def contrast(*args: object, **kwargs: object) -> np.ndarray:
            return contrast_values.pop(0)

        m2.compare_conditional_probabilities = compare
        bridge._equal_group_pair_contrast = contrast
        try:
            passing = bridge._validate_repeat(
                cold,
                warm,
                np.zeros(m1.PARAMETER_DIMENSION),
                object(),
                workload,
            )
            self.assertEqual(
                passing["candidate_probability_maximum_absolute_difference"], 0.0
            )

            warm.objective = 1.0 + 2.0e-12
            contrast_values[:] = [
                np.zeros(m2.PAIR_DIMENSION),
                np.zeros(m2.PAIR_DIMENSION),
            ]
            with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "tolerance"):
                bridge._validate_repeat(cold, warm, np.zeros(1), object(), workload)

            warm.objective = 1.0
            probability_error[0] = 2.0e-10
            contrast_values[:] = [
                np.zeros(m2.PAIR_DIMENSION),
                np.zeros(m2.PAIR_DIMENSION),
            ]
            with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "tolerance"):
                bridge._validate_repeat(cold, warm, np.zeros(1), object(), workload)

            probability_error[0] = 0.0
            changed = np.zeros(m2.PAIR_DIMENSION)
            changed[0] = 2.0e-10
            contrast_values[:] = [np.zeros(m2.PAIR_DIMENSION), changed]
            with self.assertRaisesRegex(bridge.ScaleBenchmarkError, "tolerance"):
                bridge._validate_repeat(cold, warm, np.zeros(1), object(), workload)
        finally:
            m2.compare_conditional_probabilities = original_compare
            bridge._equal_group_pair_contrast = original_contrast

    def test_diagnostic_binds_genuine_core_evidence_from_score_winner(self) -> None:
        trials = controller.coverage_trials()
        reference = SimpleNamespace(
            result=controller.TrialResult(trials[0], 0.0, 0),
            equal_group_pair_contrasts=(0.0,) * m2.PAIR_DIMENSION,
        )
        winner_contrasts = tuple(
            (0.4, 0.3, 0.2, 0.1)[index]
            if index < 4
            else 0.0
            for index in range(m2.PAIR_DIMENSION)
        )
        executions = []
        for index, trial in enumerate(trials[1:], start=1):
            contrasts = (
                winner_contrasts
                if index == 1
                else (9.0,) * m2.PAIR_DIMENSION
                if index == 15
                else (0.0,) * m2.PAIR_DIMENSION
            )
            executions.append(
                SimpleNamespace(
                    result=controller.TrialResult(
                        trial,
                        1.0 if index == 1 else 0.0,
                        index,
                    ),
                    equal_group_pair_contrasts=contrasts,
                )
            )

        cells = tuple(
            SimpleNamespace(
                identifier=f"row_{index}",
                group=group,
                k=0,
            )
            for index, group in enumerate(m1.GROUP_ORDER)
        )
        membership = np.zeros((len(cells), m1.PANEL_SIZE), dtype=np.uint8)
        folds = tuple(
            SimpleNamespace(
                spec=SimpleNamespace(validation_indexes=(index,)),
                baseline_pair_probabilities=np.zeros((1, m2.PAIR_DIMENSION)),
            )
            for index in range(len(cells))
        )
        context = SimpleNamespace(cells=cells, membership_matrix=membership)
        workload = SimpleNamespace(cache=SimpleNamespace(folds=folds))
        result = bridge._diagnose_optional_trigger(
            context,
            workload,
            reference,
            tuple(executions),
        )
        self.assertEqual(result.evidence.equal_group_pair_contrasts, winner_contrasts)
        self.assertIs(
            result.evidence.core_results[-1].coverage_binding,
            result.evidence.binding,
        )
        self.assertTrue(result.witness["coverage_evidence_bound"])
        self.assertEqual(
            result.witness["core_winner_configuration"],
            trials[1].configuration.key,
        )

    def test_explicit_offset_adapter_and_synthetic_optional_paths(self) -> None:
        transform = m1.DesignTransform(np.zeros(9), np.ones(9))
        offsets = {
            group: replay.FoldBoundM1Offset(
                held_out_group=group,
                training_indexes=(index,),
                theta=np.zeros(m1.PARAMETER_DIMENSION),
                design_transform=transform,
                geometry_scales=np.ones(m1.GEOMETRY_DIMENSION),
                fit_identity=f"fit_{index}",
            )
            for index, group in enumerate(m1.GROUP_ORDER)
        }
        receipt = replay.SyntheticGenerationReceipt(
            "synthetic_seed_2026092901",
            "numpy_SeedSequence_PCG64_planted_M1",
            replay.EXPECTED_SYNTHETIC_SEED,
            replay.EXPECTED_ROWS,
            "inventory",
            "uniforms",
            "indexes",
            "membership",
        )
        full = replay.FullDevelopmentM1Fit(
            tuple(range(replay.EXPECTED_ROWS)),
            1.0,
            np.zeros(m1.PARAMETER_DIMENSION),
            transform,
            np.ones(m1.GEOMETRY_DIMENSION),
            "full",
            object(),
        )
        result = replay.M1ReplayResult(
            MappingProxyType(offsets),
            full,
            replay.EXPECTED_ROWS,
            replay.EXPECTED_FOLDS,
            0,
            replay.EXPECTED_M1_WARM_PATHS,
            replay.EXPECTED_M1_ZERO_REFITS,
            receipt,
        )
        adapted = bridge._adapt_fixed_offsets(result)
        self.assertEqual(set(adapted), set(m1.GROUP_ORDER))
        self.assertTrue(
            all(type(value) is executor.FixedM1Offset for value in adapted.values())
        )
        hybrid_source = inspect.getsource(bridge._run_hybrid_trial)
        full_source = inspect.getsource(bridge._hybrid_full_pair)
        self.assertIn("fit_sparse_plus_low_rank_synthetic_qualification", hybrid_source)
        self.assertIn("refit_sparse_plus_low_rank_from_converged_synthetic", full_source)
        self.assertNotIn("authorize_production_hybrid", hybrid_source + full_source)
        report_source = inspect.getsource(bridge._run_report_unit)
        callback_source = inspect.getsource(bridge._WorstCaseObservedLedgerCallback)
        execution_source = inspect.getsource(bridge._execute_scale_benchmark_run)
        self.assertIn("MAXIMUM_VALID_TRIALS", report_source)
        self.assertIn("run_next_synthetic_qualification", report_source)
        self.assertIn("run_target_free_replay_conformance", report_source)
        self.assertNotIn("fit_", callback_source)
        self.assertIn("rank_neighbor_hybrid_pair", execution_source)
        self.assertIn("rank_neighbor_hybrid_pair.measurement", execution_source)
        self.assertNotIn("hashlib", inspect.getsource(bridge))


if __name__ == "__main__":
    unittest.main(verbosity=2)
