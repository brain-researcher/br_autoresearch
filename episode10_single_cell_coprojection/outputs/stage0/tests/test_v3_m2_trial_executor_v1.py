#!/usr/bin/env python3
"""Focused target-free tests for the shared EP10 v3 M2 core executor."""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


def _structural_cells() -> tuple[m1.Cell, ...]:
    cells = []
    for group_index, group in enumerate(m1.GROUP_ORDER):
        for local_index in range(2):
            value = float(2 * group_index + local_index + 1)
            cells.append(
                m1.Cell(
                    identifier=f"cell_{group_index:02d}_{local_index}",
                    group=group,
                    layer=m1.LAYERS[(group_index + local_index) % len(m1.LAYERS)],
                    hemisphere="left" if local_index == 0 else "right",
                    soma_um=(value, value * value, value * value * value),
                    soma_distance_mm=np.linspace(0.1, 1.6, m1.PANEL_SIZE),
                    k=2,
                    observed_index=(group_index + local_index) % math.comb(16, 2),
                    segment_unmapped=local_index,
                )
            )
    return tuple(cells)


def _optimizer(beta: np.ndarray) -> m2.M2OptimizerResult:
    return m2.M2OptimizerResult(
        beta=np.asarray(beta, dtype=float).copy(),
        objective=0.0,
        smooth_objective=0.0,
        penalty_value=0.0,
        iterations=0,
        lipschitz=1.0,
        kkt_residual=0.0,
        relative_objective_change=0.0,
        range_residual=0.0,
        status="pass",
        telemetry={"test_fixture": True},
    )


def _coverage_binding() -> controller.CoverageBinding:
    return controller.CoverageBinding(
        tuple((index,) for index in range(controller.COVERAGE_TRIALS)),
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0", 0),
        tuple("0x0.0p+0" for _ in range(m2.PAIR_DIMENSION)),
    )


class FoldAndPenaltyTests(unittest.TestCase):
    def test_logo_specs_are_exact_ordered_training_only_partitions(self) -> None:
        cells = _structural_cells()
        folds = executor.logo_fold_specs(cells)
        self.assertEqual(len(folds), 12)
        self.assertEqual(tuple(fold.ordinal for fold in folds), tuple(range(1, 13)))
        self.assertEqual(tuple(fold.held_out_group for fold in folds), m1.GROUP_ORDER)
        for fold in folds:
            self.assertEqual(len(fold.validation_indexes), 2)
            self.assertEqual(len(fold.training_indexes), 22)
            self.assertFalse(set(fold.training_indexes) & set(fold.validation_indexes))
            self.assertNotIn(
                fold.held_out_group,
                {cells[index].group for index in fold.training_indexes},
            )

    def test_core_configuration_maps_to_engine_penalty(self) -> None:
        maximum = m2.LambdaMaximumResult(
            4.0, np.nextafter(4.0, 0.0), 4.0, 0.0, 10
        )
        cases = (
            (controller.Configuration("lasso", 0.1), "lasso", 1.0, 0.4),
            (
                controller.Configuration("elastic_net", 0.03, alpha=0.25),
                "elastic_net",
                0.25,
                0.12,
            ),
            (
                controller.Configuration("group_lasso", 0.001),
                "group_lasso",
                1.0,
                0.004,
            ),
        )
        for configuration, kind, ratio, strength in cases:
            observed = executor.penalty_from_configuration(configuration, maximum)
            self.assertEqual(observed.penalty_kind, kind)
            self.assertEqual(observed.l1_ratio, ratio)
            self.assertAlmostEqual(observed.specification.strength, strength)
            self.assertEqual(observed.specification.kind, kind)

        exact_zero = m2.LambdaMaximumResult(0.0, 0.0, 0.0, 0.0, 0)
        zero_mapping = executor.penalty_from_configuration(
            controller.Configuration("lasso", 0.1), exact_zero
        )
        self.assertEqual(zero_mapping.specification.strength, 0.0)


class ProductionBoundaryTests(unittest.TestCase):
    def test_production_requires_exact_inventory_and_rank(self) -> None:
        cells = _structural_cells()
        moments = (None,) * len(cells)
        inventory = executor.FrozenExecutionInventory.from_tsv()
        self.assertEqual(len(inventory.entries), 194)
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.FrozenExecutionInventory(
                inventory.source_path, inventory.entries
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.prepare_trial_cache(
                cells, moments, None, None, None, {}, expected_rank=104
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.prepare_trial_cache(
                cells,
                moments,
                None,
                None,
                None,
                {},
                execution_inventory=inventory,
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor._require_canonical_inventory(
                dataclasses.replace(
                    inventory,
                    source_path=Path("/tmp/forged_v3_m2_inventory.tsv"),
                )
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor._require_canonical_inventory(
                dataclasses.replace(
                    inventory,
                    entries=tuple(
                        (f"forged_{index:03d}", m1.GROUP_ORDER[index % 12])
                        for index in range(executor.EXPECTED_PRODUCTION_ROWS)
                    ),
                )
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.prepare_trial_cache(
                cells,
                moments,
                None,
                None,
                None,
                {},
                execution_inventory=inventory,
                expected_rank=104,
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.prepare_trial_cache(
                cells,
                moments,
                None,
                None,
                None,
                {},
                mode=executor.SYNTHETIC_TEST_MODE,
                execution_inventory=inventory,
            )

    def test_offsets_are_fold_bound_and_have_unique_fit_identities(self) -> None:
        specs = executor.logo_fold_specs(_structural_cells())
        transform = m1.DesignTransform(np.zeros(9), np.ones(9))
        offsets = {
            spec.held_out_group: executor.FixedM1Offset(
                held_out_group=spec.held_out_group,
                training_indexes=spec.training_indexes,
                theta=np.zeros(m1.PARAMETER_DIMENSION),
                design_transform=transform,
                geometry_scales=np.ones(m1.GEOMETRY_DIMENSION),
                fit_identity=f"fold_fit_{spec.ordinal}",
            )
            for spec in specs
        }
        executor._validate_offset_inventory(offsets, specs)

        first = specs[0].held_out_group
        wrong_fold = dict(offsets)
        wrong_fold[first] = dataclasses.replace(
            wrong_fold[first], training_indexes=specs[1].training_indexes
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor._validate_offset_inventory(wrong_fold, specs)

        empty_identity = dict(offsets)
        empty_identity[first] = dataclasses.replace(
            empty_identity[first], fit_identity=""
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor._validate_offset_inventory(empty_identity, specs)

        duplicate_identity = dict(offsets)
        second = specs[1].held_out_group
        duplicate_identity[second] = dataclasses.replace(
            duplicate_identity[second], fit_identity=offsets[first].fit_identity
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor._validate_offset_inventory(duplicate_identity, specs)

    def test_cache_constructor_is_sealed(self) -> None:
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.PreparedTrialCache(
                None,
                None,
                None,
                (),
                executor.SYNTHETIC_TEST_MODE,
                None,
                (),
                None,
            )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor(
                None,
                maximum_iterations=m2.FISTA_MAXIMUM_ITERATIONS + 1,
            )

    def test_lambda_and_fit_diagnostics_fail_closed(self) -> None:
        valid_maximum = m2.LambdaMaximumResult(
            2.0, np.nextafter(2.0, 0.0), 2.0, 0.0, 7
        )
        configuration = controller.Configuration("lasso", 1.0)
        executor.penalty_from_configuration(configuration, valid_maximum)
        malformed_maxima = (
            dataclasses.replace(valid_maximum, value=math.nan),
            dataclasses.replace(valid_maximum, lower_bracket=-1.0),
            dataclasses.replace(valid_maximum, lower_bracket=1.0),
            dataclasses.replace(
                valid_maximum, zero_kkt_residual=2.0 * m2.PROX_TOLERANCE
            ),
            dataclasses.replace(valid_maximum, bisections=-1),
        )
        for maximum in malformed_maxima:
            with self.subTest(maximum=maximum):
                with self.assertRaises(executor.CoreTrialExecutorError):
                    executor.penalty_from_configuration(configuration, maximum)

        valid_fit = _optimizer(np.zeros(m2.PAIR_DIMENSION))
        executor._validate_optimizer_result(valid_fit, 10)
        malformed_fits = (
            dataclasses.replace(valid_fit, kkt_residual=math.nan),
            dataclasses.replace(valid_fit, relative_objective_change=-1.0),
            dataclasses.replace(valid_fit, objective=-1.0),
            dataclasses.replace(valid_fit, penalty_value=math.inf),
            dataclasses.replace(valid_fit, lipschitz=0.0),
            dataclasses.replace(
                valid_fit, range_residual=2.0 * m2.PROX_TOLERANCE
            ),
            dataclasses.replace(valid_fit, iterations=11),
        )
        for fit in malformed_fits:
            with self.subTest(fit=fit):
                with self.assertRaises(executor.CoreTrialExecutorError):
                    executor._validate_optimizer_result(fit, 10)

    def test_trials_are_bound_to_controller_identity(self) -> None:
        core = controller.coverage_trials()
        self.assertTrue(executor.CoreTrialExecutor._validate_trial(core[0]))
        self.assertFalse(executor.CoreTrialExecutor._validate_trial(core[1]))

        wrong_reference = controller.Trial(
            2, "reference", "reference_anchor", "coverage", False
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor._validate_trial(wrong_reference)

        wrong_anchor = controller.Trial(
            2,
            "association",
            "anchor",
            "coverage",
            False,
            controller.Configuration("lasso", 0.1),
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor._validate_trial(wrong_anchor)

        early_successor = controller.Trial(
            2,
            "association",
            "successor",
            "postcoverage",
            False,
            controller.Configuration("lasso", 0.3),
            cycle=1,
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor._validate_trial(early_successor)

        controller_successor = controller.Trial(
            17,
            "association",
            "successor",
            "postcoverage",
            False,
            controller.Configuration("lasso", 0.3),
            cycle=1,
            coverage_binding=_coverage_binding(),
        )
        self.assertFalse(
            executor.CoreTrialExecutor._validate_trial(controller_successor)
        )
        over_cap = controller.Trial(
            81,
            "association",
            "successor",
            "patience",
            False,
            controller.Configuration("lasso", 0.3),
            cycle=3,
            coverage_binding=_coverage_binding(),
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor._validate_trial(over_cap)

    def test_public_forged_results_cannot_authorize_a_successor(self) -> None:
        core = controller.coverage_trials()
        forged_results = tuple(
            controller.TrialResult(trial, float(index), index)
            for index, trial in enumerate(core)
        )
        evidence = controller.bind_coverage_evidence(
            forged_results,
            controller.OptionalDiagnostics(0.0, 1.0, 0.0, 0),
            (0.0,) * m2.PAIR_DIMENSION,
        )
        forged_executions = tuple(
            executor.TrialExecution(trial, (), (), (0.0,) * 120, result)
            for trial, result in zip(core, forged_results, strict=True)
        )
        successor = controller.Trial(
            17,
            "association",
            "successor",
            "postcoverage",
            False,
            controller.Configuration("lasso", 0.3),
            cycle=1,
            coverage_binding=evidence.binding,
        )
        # No executor instance is needed to establish that the public result
        # dataclasses themselves do not carry module authority.
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor._lookup_execution_identity(forged_executions[0])


class SharedExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            source_cells,
            _,
            source_prepared,
            cls.candidates,
            cls.geometry,
            cls.pairs,
            cls.theta,
        ) = m2.synthetic_k2_problem(rows=24)
        cls.cells = tuple(
            dataclasses.replace(
                cell,
                group=m1.GROUP_ORDER[index // 2],
                identifier=f"shared_executor_{index:03d}",
            )
            for index, cell in enumerate(source_cells)
        )
        cls.moments = tuple(cls.geometry.uniform_moments(cell) for cell in cls.cells)
        cls.base_transform = source_prepared.transform

        offsets = {}
        cls.training_calls = []
        for spec in executor.logo_fold_specs(cls.cells):
            prepared = m1.prepare_data(cls.cells, cls.moments, spec.training_indexes)
            offsets[spec.held_out_group] = executor.FixedM1Offset(
                held_out_group=spec.held_out_group,
                training_indexes=spec.training_indexes,
                theta=cls.theta.copy(),
                design_transform=prepared.design_transform,
                geometry_scales=prepared.geometry_scales,
                fit_identity=f"synthetic_{spec.held_out_group}",
            )

        def outcome_free_transform(
            cells,
            moments,
            candidates,
            geometry,
            pairs,
            indexes=None,
            *,
            expected_rank=None,
        ):
            del candidates, geometry, pairs, expected_rank
            assert indexes is not None
            ordered = tuple(indexes)
            cls.training_calls.append(ordered)
            prepared = m1.prepare_data(cells, moments, ordered)
            transform = dataclasses.replace(
                cls.base_transform,
                design_transform=prepared.design_transform,
                geometry_scales=prepared.geometry_scales.copy(),
            )
            return transform, m2.PreparedM2(prepared, transform)

        with mock.patch.object(
            m2, "fit_residual_transform", side_effect=outcome_free_transform
        ):
            cls.cache = executor.prepare_trial_cache(
                cls.cells,
                cls.moments,
                cls.candidates,
                cls.geometry,
                cls.pairs,
                offsets,
                mode=executor.SYNTHETIC_TEST_MODE,
            )

    def test_preparation_uses_exact_12_training_sets(self) -> None:
        expected = tuple(
            spec.training_indexes for spec in executor.logo_fold_specs(self.cells)
        )
        self.assertEqual(tuple(self.training_calls), expected)
        self.assertEqual(len(self.cache.folds), 12)
        for fold in self.cache.folds:
            self.assertNotIn(
                fold.spec.held_out_group,
                {cell.group for cell in fold.training.cells},
            )
            self.assertEqual(
                {cell.group for cell in fold.validation.cells},
                {fold.spec.held_out_group},
            )

    def test_executor_owned_iteration_cap_boundaries(self) -> None:
        for value in (0, -1, True, 1.0, 20_000.0, 20_001):
            with self.subTest(value=value):
                with self.assertRaises(executor.CoreTrialExecutorError):
                    executor.CoreTrialExecutor(
                        self.cache,
                        maximum_iterations=value,
                    )
        self.assertEqual(
            executor.CoreTrialExecutor(self.cache, maximum_iterations=1).maximum_iterations,
            1,
        )
        self.assertEqual(
            executor.CoreTrialExecutor(
                self.cache,
                maximum_iterations=executor.FROZEN_MAXIMUM_ITERATIONS,
            ).maximum_iterations,
            20_000,
        )

        production_like = dataclasses.replace(
            self.cache,
            mode=executor.SEALED_PRODUCTION_MODE,
            expected_rank=executor.EXPECTED_PRODUCTION_RANK,
            execution_inventory=executor.FrozenExecutionInventory.from_tsv(),
        )
        executor._register_prepared_cache(production_like)
        with self.assertRaisesRegex(
            executor.CoreTrialExecutorError,
            "exactly 20000",
        ):
            executor.CoreTrialExecutor(production_like, maximum_iterations=1)

    def test_replaced_or_mutated_authentic_cache_is_rejected(self) -> None:
        forged = dataclasses.replace(self.cache, folds=self.cache.folds)
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor(forged)

        baseline = self.cache.folds[0].baseline_log_probability_nats
        original = float(baseline[0])
        try:
            baseline.setflags(write=True)
            baseline[0] = original + 1.0
            with self.assertRaises(executor.CoreTrialExecutorError):
                executor.CoreTrialExecutor(self.cache)
        finally:
            baseline[0] = original
            baseline.setflags(write=False)
        executor.CoreTrialExecutor(self.cache)

        membership = self.cache.candidates.basis(2).membership
        original_membership = int(membership[0, 0])
        try:
            membership[0, 0] = 1 - original_membership
            with self.assertRaises(executor.CoreTrialExecutorError):
                executor.CoreTrialExecutor(self.cache)
        finally:
            membership[0, 0] = original_membership
        executor.CoreTrialExecutor(self.cache)

    def test_successor_without_authenticated_history_fails_before_fit(self) -> None:
        successor = controller.Trial(
            17,
            "association",
            "successor",
            "postcoverage",
            False,
            controller.Configuration("lasso", 0.3),
            cycle=1,
            coverage_binding=_coverage_binding(),
        )
        with mock.patch.object(m2, "fit_m2") as fit:
            with self.assertRaises(executor.CoreTrialExecutorError):
                executor.CoreTrialExecutor(self.cache).execute(successor)
        fit.assert_not_called()

    def test_forged_public_result_prefix_cannot_authorize_successor(self) -> None:
        core = controller.coverage_trials()
        forged_results = tuple(
            controller.TrialResult(trial, float(index), index)
            for index, trial in enumerate(core)
        )
        evidence = controller.bind_coverage_evidence(
            forged_results,
            controller.OptionalDiagnostics(0.0, 1.0, 0.0, 0),
            (0.0,) * m2.PAIR_DIMENSION,
        )
        forged_executions = tuple(
            executor.TrialExecution(trial, (), (), (0.0,) * 120, result)
            for trial, result in zip(core, forged_results, strict=True)
        )
        successor = controller.Trial(
            17,
            "association",
            "successor",
            "postcoverage",
            False,
            controller.Configuration("lasso", 0.3),
            cycle=1,
            coverage_binding=evidence.binding,
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.authorize_successor(
                executor.CoreTrialExecutor(self.cache),
                successor,
                forged_executions,
                evidence,
            )

    def test_validator_bound_external_receipts_enable_canonical_trial_21(self) -> None:
        owner = executor.CoreTrialExecutor(self.cache)
        maximum = m2.LambdaMaximumResult(
            2.0, np.nextafter(2.0, 0.0), 2.0, 0.0, 7
        )
        core = controller.coverage_trials()
        executions = []

        def marked_pair_probabilities(
            beta,
            fixed_m1_theta,
            prepared,
            candidates,
            geometry,
            pairs,
        ):
            del beta, fixed_m1_theta, candidates, geometry, pairs
            values = np.full(
                (len(prepared.cells), m2.PAIR_DIMENSION),
                0.25,
                dtype=float,
            )
            values.setflags(write=False)
            return values

        with mock.patch.object(
            m2,
            "fold_lambda_maximum",
            return_value=maximum,
        ), mock.patch.object(
            m2,
            "fit_m2",
            return_value=_optimizer(np.zeros(m2.PAIR_DIMENSION)),
        ):
            executions.append(owner.execute(core[0]))
            for trial in core[1:]:
                if trial.valid_index == 2:
                    with mock.patch.object(
                        executor,
                        "_pair_probabilities",
                        side_effect=marked_pair_probabilities,
                    ):
                        executions.append(owner.execute(trial))
                else:
                    executions.append(owner.execute(trial))

            self.assertNotEqual(
                executions[1].equal_group_pair_contrasts,
                executions[-1].equal_group_pair_contrasts,
            )
            evidence = controller.bind_coverage_evidence(
                tuple(execution.result for execution in executions),
                controller.OptionalDiagnostics(0.0, 1.0, 0.0, 0),
                executions[1].equal_group_pair_contrasts,
            )
            cycle_one = controller.successor_configurations(
                evidence.core_results,
                coverage_evidence=evidence,
            )
            cycle_trials = controller._minimum_cycle_trials(
                cycle_one,
                cycle=1,
                start_valid_index=17,
                coverage_binding=evidence.binding,
            )
            noise_results = tuple(
                controller.TrialResult(trial, 0.0, 0)
                for trial in cycle_trials[:4]
            )

            class Producer:
                def __init__(self, results):
                    self.results = {
                        result.trial.valid_index: result for result in results
                    }

            class Validator:
                def __call__(self, producer, result):
                    return producer.results.get(result.trial.valid_index) is result

            producer = Producer(noise_results)
            validator = Validator()
            receipts = tuple(
                executor.authenticate_external_execution(
                    owner,
                    producer,
                    validator,
                    result,
                )
                for result in noise_results
            )
            successor = cycle_trials[4]
            history = (*executions, *receipts)
            authorization = executor.authorize_successor(
                owner,
                successor,
                history,
                evidence,
            )
            observed = owner.execute(successor, authorization=authorization)
            self.assertEqual(observed.trial.valid_index, 21)

            with self.assertRaises(executor.CoreTrialExecutorError):
                executor.authorize_successor(
                    owner,
                    successor,
                    (*executions, *noise_results),
                    evidence,
                )
            replaced = list(receipts)
            replaced[0] = dataclasses.replace(receipts[0])
            with self.assertRaises(executor.CoreTrialExecutorError):
                executor.authorize_successor(
                    owner,
                    successor,
                    (*executions, *replaced),
                    evidence,
                )
            with self.assertRaises(executor.CoreTrialExecutorError):
                executor.ExternalExecutionReceipt(noise_results[0])

    def test_beta_zero_reference_runs_unmocked_scoring(self) -> None:
        trial = controller.coverage_trials()[0]
        result = executor.CoreTrialExecutor(self.cache).execute(trial)
        self.assertEqual(len(result.folds), 12)
        self.assertEqual(result.result.gain_bits_per_cell, 0.0)
        self.assertEqual(result.result.active_pair_degrees, 0)
        self.assertEqual(len(result.pair_contrast_rows), 24)
        self.assertEqual(len({row.cell_key for row in result.pair_contrast_rows}), 24)
        self.assertEqual(result.equal_group_pair_contrasts, (0.0,) * 120)
        for fold in result.folds:
            self.assertEqual(fold.gain_bits_per_cell, 0.0)
            self.assertIsNone(fold.penalty)
            self.assertIsNone(fold.fit)

    def test_core_trial_delegates_lambda_fit_and_exact_scoring(self) -> None:
        maximum = m2.LambdaMaximumResult(
            2.0, np.nextafter(2.0, 0.0), 2.0, 0.0, 7
        )
        coverage = controller.coverage_trials()
        families = (
            (coverage[1], "lasso", 1.0),
            (coverage[5], "elastic_net", 0.25),
            (coverage[12], "group_lasso", 1.0),
        )
        for trial, expected_kind, expected_ratio in families:
            fit_calls = []

            def fit_zero(*args, **kwargs):
                fit_calls.append((args, kwargs))
                return _optimizer(np.zeros(m2.PAIR_DIMENSION))

            with mock.patch.object(
                m2, "fold_lambda_maximum", return_value=maximum
            ) as lambda_maximum, mock.patch.object(
                m2, "fit_m2", side_effect=fit_zero
            ):
                observed = executor.CoreTrialExecutor(self.cache).execute(trial)
            self.assertEqual(lambda_maximum.call_count, 12)
            self.assertEqual(len(fit_calls), 12)
            self.assertEqual(observed.result.gain_bits_per_cell, 0.0)
            self.assertEqual(observed.equal_group_pair_contrasts, (0.0,) * 120)
            for fold in observed.folds:
                assert fold.penalty is not None
                self.assertEqual(fold.penalty.penalty_kind, expected_kind)
                self.assertEqual(fold.penalty.l1_ratio, expected_ratio)
                self.assertEqual(fold.penalty.specification.strength, 2.0)

    def test_k0_and_k16_have_exact_zero_gain_and_contrasts(self) -> None:
        boundary_candidates = m1.CandidateLibrary()
        boundary_geometry = m1.GeometryEngine(
            boundary_candidates,
            self.geometry.pairs,
        )
        boundary_pairs = m2.PairFeatureLibrary(boundary_candidates)
        boundary_cells = tuple(
            dataclasses.replace(
                cell,
                k=0 if index % 2 == 0 else m1.PANEL_SIZE,
                observed_index=0,
                identifier=f"boundary_{index:03d}",
            )
            for index, cell in enumerate(self.cells)
        )
        boundary_moments = tuple(
            boundary_geometry.uniform_moments(cell) for cell in boundary_cells
        )
        offsets = {}
        source_folds = {
            fold.spec.held_out_group: fold for fold in self.cache.folds
        }
        for spec in executor.logo_fold_specs(boundary_cells):
            source = source_folds[spec.held_out_group]
            offsets[spec.held_out_group] = executor.FixedM1Offset(
                held_out_group=spec.held_out_group,
                training_indexes=spec.training_indexes,
                theta=self.theta.copy(),
                design_transform=source.training.transform.design_transform,
                geometry_scales=source.training.transform.geometry_scales,
                fit_identity=f"boundary_{spec.ordinal}",
            )

        def boundary_transform(
            cells,
            moments,
            candidates,
            geometry,
            pairs,
            indexes=None,
            *,
            expected_rank=None,
        ):
            del candidates, geometry, pairs, expected_rank
            assert indexes is not None
            ordered = tuple(indexes)
            held_out = next(
                group
                for group in m1.GROUP_ORDER
                if group not in {cells[index].group for index in ordered}
            )
            transform = source_folds[held_out].training.transform
            prepared = m1.apply_fixed_transforms(
                tuple(cells[index] for index in ordered),
                tuple(moments[index] for index in ordered),
                transform.design_transform,
                transform.geometry_scales,
            )
            return transform, m2.PreparedM2(prepared, transform)

        with mock.patch.object(
            m2,
            "fit_residual_transform",
            side_effect=boundary_transform,
        ):
            cache = executor.prepare_trial_cache(
                boundary_cells,
                boundary_moments,
                boundary_candidates,
                boundary_geometry,
                boundary_pairs,
                offsets,
                mode=executor.SYNTHETIC_TEST_MODE,
            )
        maximum = m2.LambdaMaximumResult(
            2.0, np.nextafter(2.0, 0.0), 2.0, 0.0, 7
        )
        nonzero = np.full(m2.PAIR_DIMENSION, 0.01)
        with mock.patch.object(
            m2,
            "fold_lambda_maximum",
            return_value=maximum,
        ), mock.patch.object(m2, "fit_m2", return_value=_optimizer(nonzero)):
            observed = executor.CoreTrialExecutor(cache).execute(
                controller.coverage_trials()[2]
            )
        self.assertEqual(observed.result.gain_bits_per_cell, 0.0)
        self.assertEqual(observed.equal_group_pair_contrasts, (0.0,) * 120)
        for fold in observed.folds:
            self.assertEqual(fold.gain_bits_per_cell, 0.0)
            for cell in fold.cells:
                self.assertEqual(
                    cell.m1_log_probability_nats,
                    cell.m2_log_probability_nats,
                )
                self.assertEqual(cell.pair_probability_contrasts, (0.0,) * 120)

    def test_falsifier_and_optional_trials_fail_closed(self) -> None:
        optional = controller.Trial(
            17,
            "association",
            "anchor",
            "extension_coverage",
            False,
            controller.Configuration("low_rank", 0.1, rank=1),
            coverage_binding=_coverage_binding(),
        )
        with self.assertRaises(executor.CoreTrialExecutorError):
            executor.CoreTrialExecutor(self.cache).execute(optional)


if __name__ == "__main__":
    unittest.main()
