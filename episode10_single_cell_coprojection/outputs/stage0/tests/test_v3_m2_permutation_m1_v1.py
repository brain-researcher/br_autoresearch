#!/usr/bin/env python3
"""Target-free tests for fresh outer-only M1 on whole-set permutations."""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m1_real_v1 as real_m1  # noqa: E402
import v3_m2_permutation_adapter_v1 as permutation  # noqa: E402
import v3_m2_permutation_m1_v1 as permutation_m1  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


def _coverage_binding() -> controller.CoverageBinding:
    return controller.CoverageBinding(
        tuple((index,) for index in range(controller.COVERAGE_TRIALS)),
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0", 0),
        tuple("0x0.0p+0" for _ in range(controller.PAIR_COUNT)),
    )


def _trial() -> controller.Trial:
    return controller.Trial(
        valid_index=28,
        role="association",
        kind="permutation",
        phase="postcoverage",
        falsifier=True,
        configuration=controller.Configuration("lasso", 0.1),
        seed=2026092811,
        cycle=1,
        coverage_binding=_coverage_binding(),
    )


def _cells() -> tuple[m1.Cell, ...]:
    group_ordinal = {group: index for index, group in enumerate(m1.GROUP_ORDER)}
    local_by_group = {group: 0 for group in m1.GROUP_ORDER}
    execution_order = tuple(
        entry
        for group in m1.GROUP_ORDER
        for entry in sorted(
            (
                entry
                for entry in controller.FROZEN_CELL_INVENTORY.entries
                if entry[1] == group
            ),
            key=lambda entry: int(entry[0]),
        )
    )
    rows: list[m1.Cell] = []
    for row_index, (identifier, group) in enumerate(execution_order):
        local = local_by_group[group]
        local_by_group[group] += 1
        k = (5 * local + group_ordinal[group]) % (m1.PANEL_SIZE + 1)
        distance = np.linspace(0.1, 1.6, m1.PANEL_SIZE) + row_index * 1.0e-4
        distance.setflags(write=False)
        rows.append(
            m1.Cell(
                identifier=identifier,
                group=group,
                layer=m1.LAYERS[(row_index + local) % len(m1.LAYERS)],
                hemisphere="left" if row_index % 2 == 0 else "right",
                soma_um=(float(row_index), float(local), float(row_index + local)),
                soma_distance_mm=distance,
                k=k,
                observed_index=(37 * row_index + 11 * local)
                % math.comb(m1.PANEL_SIZE, k),
                segment_unmapped=(local // 3) % 2,
                sampling_uniform=(row_index + 0.5) / 194.0,
            )
        )
    assert local_by_group == m1.GROUP_SIZES
    return tuple(rows)


def _prepared_stub(cells: tuple[m1.Cell, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        cells=cells,
        design_transform=m1.DesignTransform(np.zeros(9), np.ones(9)),
        geometry_scales=np.ones(m1.GEOMETRY_DIMENSION),
    )


def _result(ridge_lambda: float) -> SimpleNamespace:
    return SimpleNamespace(
        theta=np.full(m1.PARAMETER_DIMENSION, ridge_lambda * 1.0e-7),
        ridge_lambda=float(ridge_lambda),
        objective=1.0,
        gradient_infinity_norm=0.0,
        iterations=1,
        accepted_steps=1,
        relative_objective_change=0.0,
        telemetry={"status": "pass", "ridge_lambda": float(ridge_lambda)},
    )


def _geometry(library: m1.CandidateLibrary) -> m1.GeometryEngine:
    indexes = np.arange(m1.PANEL_SIZE, dtype=float)
    distance = np.abs(indexes[:, None] - indexes[None, :])
    cross_side = np.zeros_like(distance)
    homotopic = np.zeros_like(distance)
    ordered_edges = tuple(
        sorted(
            (float(distance[left, right]), left, right)
            for left in range(m1.PANEL_SIZE)
            for right in range(left + 1, m1.PANEL_SIZE)
        )
    )
    engine = m1.GeometryEngine(
        library,
        m1.PairGeometry(distance, cross_side, homotopic, ordered_edges),
    )
    # The orchestration tests do not exercise the geometry algorithm.  Seed
    # its bounded cache with exact-shape arrays so the production type and
    # state-binding path remain real without recomputing MST features.
    engine._shared_features.update(
        {
            k: np.zeros((math.comb(m1.PANEL_SIZE, k), 4), dtype=float)
            for k in range(m1.PANEL_SIZE + 1)
        }
    )
    return engine


class PermutationM1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cells = _cells()
        self.assignment = permutation.build_observed_permutation_assignment(
            self.cells, _trial()
        )
        self.candidates = m1.CandidateLibrary()
        self.geometry = _geometry(self.candidates)
        self.warm_calls: list[tuple[str, tuple[int, ...]]] = []
        self.zero_calls: list[tuple[str, tuple[int, ...], float]] = []
        self.preferred_ridge = 1.0

    def _warm(
        self,
        key: str,
        indexes: tuple[int, ...],
        cells: tuple[m1.Cell, ...],
        moments: tuple[object, ...],
        library: m1.CandidateLibrary,
        geometry: object,
    ) -> real_m1.TrainingBundle:
        del moments, library, geometry
        selected = tuple(indexes)
        self.warm_calls.append((key, selected))
        return real_m1.TrainingBundle(
            key,
            selected,
            _prepared_stub(tuple(cells[index] for index in selected)),
            {ridge: _result(ridge) for ridge in m1.LAMBDAS_DESCENDING},
            False,
        )

    def _zero(
        self,
        key: str,
        indexes: tuple[int, ...],
        ridge_lambda: float,
        cells: tuple[m1.Cell, ...],
        moments: tuple[object, ...],
        library: m1.CandidateLibrary,
        geometry: object,
    ) -> real_m1.TrainingBundle:
        del moments, library, geometry
        selected = tuple(indexes)
        ridge = float(ridge_lambda)
        self.zero_calls.append((key, selected, ridge))
        return real_m1.TrainingBundle(
            key,
            selected,
            _prepared_stub(tuple(cells[index] for index in selected)),
            {ridge: _result(ridge)},
            False,
        )

    def _predict(
        self,
        bundle: real_m1.TrainingBundle,
        result: SimpleNamespace,
        validation_indexes: tuple[int, ...],
        cells: tuple[m1.Cell, ...],
        moments: tuple[object, ...],
        library: m1.CandidateLibrary,
        geometry: object,
    ) -> real_m1.Prediction:
        del bundle, moments, library, geometry
        held = tuple(cells[index] for index in validation_indexes)
        ridge = float(result.ridge_lambda)
        nll = 0.1 if ridge == self.preferred_ridge else 1.0
        log_probability = np.asarray(
            [0.0 if cell.k in {0, m1.PANEL_SIZE} else -nll for cell in held],
            dtype=float,
        )
        marginals = np.asarray(
            [np.full(m1.PANEL_SIZE, cell.k / m1.PANEL_SIZE) for cell in held],
            dtype=float,
        )
        return real_m1.Prediction(log_probability, marginals)

    def _fit(self) -> permutation_m1.PermutationM1Execution:
        with (
            mock.patch.object(permutation_m1, "_fit_warm_path", self._warm),
            mock.patch.object(permutation_m1, "_fit_zero_outer", self._zero),
            mock.patch.object(
                permutation_m1.real_m1,
                "_production_predictor",
                self._predict,
            ),
        ):
            return permutation_m1.fit_permuted_outer_m1(
                self.cells,
                self.assignment,
                self.candidates,
                self.geometry,
                identity_prefix="qualification.permutation.trial28.attempt1",
            )

    def test_exact_66_warm_12_zero_outer_only_and_own_baseline(self) -> None:
        execution = self._fit()
        self.assertEqual(len(self.warm_calls), 66)
        self.assertEqual(len({key for key, _ in self.warm_calls}), 66)
        self.assertEqual(len(self.zero_calls), 12)
        self.assertEqual(len({key for key, _, _ in self.zero_calls}), 12)
        self.assertFalse(any("leave_one" in key for key, _ in self.warm_calls))
        self.assertFalse(any("full" in key for key, _, _ in self.zero_calls))
        self.assertFalse(execution.full_development_refit_run)
        self.assertEqual(execution.warm_path_count, 66)
        self.assertEqual(execution.zero_refit_count, 12)
        self.assertEqual(execution.boundary_count, 0)
        self.assertEqual(set(execution.selected_lambdas), set(m1.GROUP_ORDER))
        self.assertEqual(set(execution.selected_lambdas.values()), {1.0})
        self.assertFalse(execution.baseline_log_probability_nats.flags.writeable)
        self.assertFalse(execution.baseline_marginals.flags.writeable)
        with self.assertRaises(ValueError):
            execution.baseline_log_probability_nats.setflags(write=True)
        with self.assertRaises(ValueError):
            execution.uniform_moments[0].raw_joint_covariance.setflags(write=True)
        with self.assertRaises(ValueError):
            execution.permuted_cells[0].soma_distance_mm.setflags(write=True)
        self.assertEqual(len(execution.permuted_cells), 194)
        self.assertEqual(
            tuple(cell.observed_index for cell in execution.permuted_cells),
            tuple(
                self.cells[donor].observed_index
                for donor in self.assignment.recipient_to_donor
            ),
        )

        for group, offset in execution.fixed_outer_offsets.items():
            expected = tuple(
                index
                for index, cell in enumerate(execution.permuted_cells)
                if cell.group != group
            )
            self.assertEqual(offset.training_indexes, expected)
            self.assertEqual(offset.held_out_group, group)
            self.assertFalse(offset.theta.flags.writeable)
            with self.assertRaises(ValueError):
                offset.theta.setflags(write=True)
            self.assertIn(group, offset.fit_identity)
        for key, indexes in self.warm_calls:
            excluded = key.removeprefix("leave_two__").split("__")
            self.assertEqual(
                {self.cells[index].group for index in indexes},
                set(m1.GROUP_ORDER).difference(excluded),
            )
        for key, indexes, ridge in self.zero_calls:
            held = key.removeprefix("outer_refit__")
            self.assertEqual(ridge, 1.0)
            self.assertNotIn(held, {self.cells[index].group for index in indexes})

    def test_no_generic_factory_api_and_fail_closed_source_or_construction(self) -> None:
        parameters = inspect.signature(
            permutation_m1.fit_permuted_outer_m1
        ).parameters
        self.assertNotIn("warm_factory", parameters)
        self.assertNotIn("zero_factory", parameters)
        self.assertNotIn("predictor", parameters)
        with self.assertRaisesRegex(
            permutation_m1.PermutationM1Error, "issued by fit_permuted_outer_m1"
        ):
            permutation_m1.PermutationM1Execution(
                assignment=self.assignment,
                candidates=self.candidates,
                geometry=self.geometry,
                permuted_cells=self.cells,
                uniform_moments=(),
                selected_lambdas={},
                fixed_outer_offsets={},
                baseline_log_probability_nats=np.zeros(194),
                baseline_marginals=np.zeros((194, 16)),
                boundary_count=0,
                identity_prefix="forged",
            )
        forged = permutation_m1.PermutationM1Execution(
            assignment=self.assignment,
            candidates=self.candidates,
            geometry=self.geometry,
            permuted_cells=self.cells,
            uniform_moments=(),
            selected_lambdas={},
            fixed_outer_offsets={},
            baseline_log_probability_nats=np.zeros(194),
            baseline_marginals=np.zeros((194, 16)),
            boundary_count=0,
            identity_prefix="forged_with_imported_seal",
            _seal=permutation_m1._EXECUTION_SEAL,
        )
        with self.assertRaisesRegex(
            permutation_m1.PermutationM1Error, "lacks live producer authority"
        ):
            permutation_m1._require_execution(forged)
        changed = list(self.cells)
        index = next(i for i, cell in enumerate(changed) if math.comb(16, cell.k) > 1)
        changed[index] = dataclasses.replace(
            changed[index],
            observed_index=(changed[index].observed_index + 1)
            % math.comb(16, changed[index].k),
        )
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "different source view"
        ):
            permutation_m1.fit_permuted_outer_m1(
                changed,
                self.assignment,
                self.candidates,
                self.geometry,
                identity_prefix="qualification.permutation.trial28.attempt1",
            )

    def test_six_or_more_boundary_folds_stop_before_zero_refits(self) -> None:
        self.preferred_ridge = m1.LAMBDAS_DESCENDING[0]
        with self.assertRaises(real_m1.BoundaryReview):
            self._fit()
        self.assertEqual(len(self.warm_calls), 66)
        self.assertEqual(len(self.zero_calls), 0)

    def test_m2_cache_must_use_own_outer_cross_fitted_baseline(self) -> None:
        execution = self._fit()
        specs = executor.logo_fold_specs(execution.permuted_cells)

        def cache_with_delta(delta: float) -> SimpleNamespace:
            folds = tuple(
                SimpleNamespace(
                    spec=spec,
                    baseline_log_probability_nats=(
                        execution.baseline_log_probability_nats[
                            list(spec.validation_indexes)
                        ]
                        + delta
                    ),
                )
                for spec in specs
            )
            return SimpleNamespace(folds=folds)

        with mock.patch.object(
            permutation_m1.executor,
            "prepare_trial_cache",
            return_value=cache_with_delta(0.0),
        ) as prepare:
            cache = permutation_m1.prepare_permuted_m2_cache(
                execution,
                self.candidates,
                self.geometry,
                mode=executor.SYNTHETIC_TEST_MODE,
                execution_inventory=None,
                expected_rank=None,
            )
        self.assertEqual(len(cache.folds), 12)
        prepare.assert_called_once()
        self.assertIsInstance(prepare.call_args.args[4], permutation_m1.m2.PairFeatureLibrary)
        self.assertIs(prepare.call_args.args[4].candidates, self.candidates)

        pair_class = permutation_m1.m2.PairFeatureLibrary

        class ReboundPairFeatureLibrary(pair_class):
            pass

        with (
            mock.patch.object(
                permutation_m1.m2,
                "PairFeatureLibrary",
                ReboundPairFeatureLibrary,
            ),
            self.assertRaisesRegex(
                permutation_m1.PermutationM1Error,
                "pair-feature class or methods changed",
            ),
        ):
            permutation_m1.prepare_permuted_m2_cache(
                execution,
                self.candidates,
                self.geometry,
                mode=executor.SYNTHETIC_TEST_MODE,
                execution_inventory=None,
                expected_rank=None,
            )

        original_basis_method = pair_class.basis

        def rebound_basis(
            instance: permutation_m1.m2.PairFeatureLibrary, k: int
        ) -> permutation_m1.m2.PairBasis:
            return original_basis_method(instance, k)

        with (
            mock.patch.object(pair_class, "basis", rebound_basis),
            self.assertRaisesRegex(
                permutation_m1.PermutationM1Error,
                "pair-feature class or methods changed",
            ),
        ):
            permutation_m1.prepare_permuted_m2_cache(
                execution,
                self.candidates,
                self.geometry,
                mode=executor.SYNTHETIC_TEST_MODE,
                execution_inventory=None,
                expected_rank=None,
            )

        foreign_candidates = m1.CandidateLibrary()
        foreign_geometry = SimpleNamespace(library=foreign_candidates)
        with self.assertRaisesRegex(
            permutation_m1.PermutationM1Error, "cache authority changed"
        ):
            permutation_m1.prepare_permuted_m2_cache(
                execution,
                foreign_candidates,
                foreign_geometry,
                mode=executor.SYNTHETIC_TEST_MODE,
                execution_inventory=None,
                expected_rank=None,
            )

        with (
            mock.patch.object(
                permutation_m1.executor,
                "prepare_trial_cache",
                return_value=cache_with_delta(1.0e-12),
            ),
            self.assertRaisesRegex(
                permutation_m1.PermutationM1Error, "own outer-cross-fitted"
            ),
        ):
            permutation_m1.prepare_permuted_m2_cache(
                execution,
                self.candidates,
                self.geometry,
                mode=executor.SYNTHETIC_TEST_MODE,
                execution_inventory=None,
                expected_rank=None,
            )

        basis = self.candidates.basis(2)
        original_membership = int(basis.membership[0, 0])
        basis.membership[0, 0] = 1 - original_membership
        try:
            with self.assertRaisesRegex(
                permutation_m1.PermutationM1Error,
                "candidate membership K=2 changed",
            ):
                permutation_m1._require_execution(execution)
        finally:
            basis.membership[0, 0] = original_membership

        zero_index = tuple(np.argwhere(basis.contrasts == 0.0)[0])
        original_zero = float(basis.contrasts[zero_index])
        basis.contrasts[zero_index] = -original_zero
        self.assertNotEqual(
            np.asarray(original_zero).tobytes(),
            np.asarray(basis.contrasts[zero_index]).tobytes(),
        )
        try:
            with self.assertRaisesRegex(
                permutation_m1.PermutationM1Error,
                "candidate contrasts K=2 changed",
            ):
                permutation_m1._require_execution(execution)
        finally:
            basis.contrasts[zero_index] = original_zero

        shared = self.geometry.shared_features(2)
        original_shared = float(shared[0, 0])
        shared[0, 0] = original_shared + 1.0
        try:
            with self.assertRaisesRegex(
                permutation_m1.PermutationM1Error,
                "shared geometry K=2 changed",
            ):
                permutation_m1._require_execution(execution)
        finally:
            shared[0, 0] = original_shared


if __name__ == "__main__":
    unittest.main()
