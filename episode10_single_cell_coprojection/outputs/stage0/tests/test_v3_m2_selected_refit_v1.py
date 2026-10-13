#!/usr/bin/env python3
"""Target-free tests for the terminal selected full-development refit."""

from __future__ import annotations

import dataclasses
import copy
import gc
import math
import os
import pickle
import stat
import sys
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import load_v3_m2_observed_inputs_v1 as input_loader  # noqa: E402
import v3_m2_selected_refit_v1 as selected  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


class _Marker:
    pass


class SelectedRefitTests(unittest.TestCase):
    @staticmethod
    def _outer_fixture(
        group_gains: tuple[float, ...],
        group_pair_contrasts: tuple[float, ...] | None = None,
    ) -> SimpleNamespace:
        if len(group_gains) != len(m1.GROUP_ORDER):
            raise AssertionError("fixture requires one gain per group")
        if group_pair_contrasts is None:
            group_pair_contrasts = (0.25,) * len(m1.GROUP_ORDER)
        if len(group_pair_contrasts) != len(m1.GROUP_ORDER):
            raise AssertionError("fixture requires one contrast per group")
        inventory = controller.EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES
        cells: list[SimpleNamespace] = []
        rows: list[controller.PairContrastRow] = []
        folds: list[executor.FoldTrialResult] = []
        next_index = 0
        for ordinal, (group, requested_gain, requested_contrast) in enumerate(
            zip(
                m1.GROUP_ORDER,
                group_gains,
                group_pair_contrasts,
                strict=True,
            ),
            start=1,
        ):
            pair_values = (float(requested_contrast),) + (0.0,) * (
                controller.PAIR_COUNT - 1
            )
            group_entries = tuple(key for key, value in inventory if value == group)
            validation_indexes: list[int] = []
            results: list[executor.CellTrialResult] = []
            for key in group_entries:
                cell = SimpleNamespace(
                    identifier=key,
                    group=group,
                    k=2,
                    soma_um=(0.0, 0.0, 0.0),
                )
                cells.append(cell)
                validation_indexes.append(next_index)
                result = executor.CellTrialResult(
                    global_index=next_index,
                    cell_key=key,
                    group=group,
                    k=2,
                    m1_log_probability_nats=0.0,
                    m2_log_probability_nats=float(requested_gain) * m1.LN2,
                    pair_probability_contrasts=pair_values,
                )
                results.append(result)
                rows.append(
                    controller.make_pair_contrast_row(
                        key,
                        group,
                        2,
                        pair_values,
                    )
                )
                next_index += 1
            exact_gains = tuple(
                (value.m2_log_probability_nats - value.m1_log_probability_nats)
                / m1.LN2
                for value in results
            )
            fold_gain = math.fsum(exact_gains) / len(exact_gains)
            folds.append(
                executor.FoldTrialResult(
                    spec=executor.FoldSpec(
                        ordinal=ordinal,
                        held_out_group=group,
                        training_indexes=(),
                        validation_indexes=tuple(validation_indexes),
                    ),
                    penalty=None,
                    fit=None,
                    gain_bits_per_cell=fold_gain,
                    active_pair_degrees=0,
                    cells=tuple(results),
                )
            )
        if next_index != executor.EXPECTED_PRODUCTION_ROWS:
            raise AssertionError("fixture did not cover all frozen rows")
        equal_pairs = executor._aggregate_pair_contrasts(tuple(rows))
        execution = SimpleNamespace(
            folds=tuple(folds),
            pair_contrast_rows=tuple(rows),
            equal_group_pair_contrasts=equal_pairs,
        )
        prepared = SimpleNamespace(cells=tuple(cells))
        full = SimpleNamespace(prepared=prepared)
        strata = input_loader._freeze_position_strata(tuple(cells))
        winner_gain = math.fsum(
            fold.gain_bits_per_cell for fold in folds
        ) / len(folds)
        winner = SimpleNamespace(gain_bits_per_cell=winner_gain)
        candidate_selection = controller.select_pair_contrast(tuple(rows))
        return SimpleNamespace(
            owner=_Marker(),
            terminal=_Marker(),
            execution=execution,
            full=full,
            strata=strata,
            winner=winner,
            candidate_selection=candidate_selection,
        )

    @staticmethod
    def _audit(
        fixture: SimpleNamespace,
        strata: object | None = None,
        candidate_selection: object | None = None,
    ):
        with mock.patch.object(
            selected.observed,
            "_selected_refit_outer_execution",
            return_value=fixture.execution,
        ):
            return selected._outer_crossfit_audit(
                fixture.owner,
                fixture.terminal,
                fixture.winner,
                (
                    fixture.candidate_selection
                    if candidate_selection is None
                    else candidate_selection
                ),
                fixture.full,
                fixture.strata if strata is None else strata,
            )

    @staticmethod
    def _fresh_m1_fixture(*, reused: bool = False) -> SimpleNamespace:
        cells: list[_Marker] = []
        for key, group in controller.EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES:
            cell = _Marker()
            cell.identifier = key
            cell.group = group
            cells.append(cell)
        cells_tuple = tuple(cells)
        moments = tuple(_Marker() for _ in cells_tuple)
        candidates = _Marker()
        geometry = _Marker()
        pairs = _Marker()

        x = np.arange(len(cells_tuple) * 2, dtype=float).reshape(len(cells_tuple), 2)
        geometry_centers = np.arange(36, dtype=float).reshape(12, 3)
        geometry_scales = np.ones((12, 3), dtype=float)
        weights = np.ones(len(cells_tuple), dtype=float)
        soma_mean = np.asarray((1.0, 2.0, 3.0))
        soma_scale = np.asarray((4.0, 5.0, 6.0))
        design_transform = SimpleNamespace(
            soma_mean=soma_mean,
            soma_scale=soma_scale,
        )
        m1_prepared = SimpleNamespace(
            x=x,
            geometry_centers=geometry_centers,
            geometry_scales=geometry_scales,
            weights=weights,
            design_transform=design_transform,
        )

        warm_cache: dict[str, SimpleNamespace] = {}
        for left_index, left in enumerate(m1.GROUP_ORDER[:-1]):
            for right in m1.GROUP_ORDER[left_index + 1 :]:
                key = f"leave_two__{left}__{right}"
                indexes = tuple(
                    index
                    for index, cell in enumerate(cells_tuple)
                    if cell.group not in {left, right}
                )
                warm_cache[key] = SimpleNamespace(
                    key=key,
                    indexes=indexes,
                    checkpoint_reused=False,
                    fits={},
                    prepared=m1_prepared,
                )
        for held_out in m1.GROUP_ORDER:
            key = f"leave_one__{held_out}"
            indexes = tuple(
                index
                for index, cell in enumerate(cells_tuple)
                if cell.group != held_out
            )
            warm_cache[key] = SimpleNamespace(
                key=key,
                indexes=indexes,
                checkpoint_reused=False,
                fits={},
                prepared=m1_prepared,
            )
        if reused:
            next(iter(warm_cache.values())).checkpoint_reused = True

        selected_lambda = float(m1.LAMBDAS_DESCENDING[0])
        theta = np.zeros(m1.PARAMETER_DIMENSION, dtype=float)
        full_bundle = SimpleNamespace(
            key="full_refit",
            indexes=tuple(range(executor.EXPECTED_PRODUCTION_ROWS)),
            checkpoint_reused=False,
            fits={selected_lambda: SimpleNamespace(theta=theta)},
            prepared=m1_prepared,
        )
        full_result = SimpleNamespace(
            warm_cache=warm_cache,
            full_refit=full_bundle,
            selected_lambda=selected_lambda,
        )
        selection = SimpleNamespace(
            warm_cache={
                key: value
                for key, value in warm_cache.items()
                if key.startswith("leave_two__")
            },
            boundary_count=0,
        )

        m2_prepared = SimpleNamespace(
            cells=cells_tuple,
            m1_data=SimpleNamespace(
                cells=cells_tuple,
                x=x,
                geometry_centers=geometry_centers,
                geometry_scales=geometry_scales,
                weights=weights,
            ),
            transform=SimpleNamespace(
                rank=executor.EXPECTED_PRODUCTION_RANK,
                design_transform=design_transform,
                geometry_scales=geometry_scales,
                projection_f_to_h=np.zeros((2, 2), dtype=float),
                pair_residual_scales=np.ones(controller.PAIR_COUNT, dtype=float),
                range_basis=np.zeros((2, 2), dtype=float),
                range_projector=np.zeros((2, 2), dtype=float),
                residual_gram=np.zeros((2, 2), dtype=float),
                scaled_residual_gram=np.zeros((2, 2), dtype=float),
                diagnostics={
                    "rank_at_tau_over_10_tau_ten_tau": {
                        "tau_over_10": executor.EXPECTED_PRODUCTION_RANK,
                        "tau": executor.EXPECTED_PRODUCTION_RANK,
                        "ten_tau": executor.EXPECTED_PRODUCTION_RANK,
                    },
                    "rank_tolerance": 1.0e-12,
                    "rank_stable": True,
                    "rank": executor.EXPECTED_PRODUCTION_RANK,
                    "expected_rank": executor.EXPECTED_PRODUCTION_RANK,
                    "expected_rank_match": True,
                },
            ),
        )
        return SimpleNamespace(
            owner=_Marker(),
            terminal=_Marker(),
            authority=_Marker(),
            cells=cells_tuple,
            moments=moments,
            candidates=candidates,
            geometry=geometry,
            pairs=pairs,
            selection=selection,
            full_result=full_result,
            m2_prepared=m2_prepared,
            theta=theta,
        )

    def _run_fresh_m1_fixture(
        self,
        fixture: SimpleNamespace,
        output_root: Path,
    ) -> tuple[object, mock.Mock, mock.Mock]:
        def complete(
            owner: object,
            terminal: object,
            authority: object,
            full: object,
        ) -> None:
            theta, prepared, audit, digest = (
                selected._full_objective_authority_record(
                    full,
                    authority,
                    owner,
                    terminal,
                )
            )
            self.assertIs(theta, fixture.theta)
            self.assertIs(prepared, fixture.m2_prepared)
            self.assertEqual(audit["leave_two_path_count"], 66)
            self.assertEqual(audit["leave_one_path_count"], 12)
            self.assertIsInstance(digest, bytes)

        with (
            mock.patch.object(
                selected.observed,
                "_begin_selected_refit_full_m1_request",
                return_value=b"begun",
            ),
            mock.patch.object(
                selected.observed,
                "_selected_refit_source_request",
                return_value=(
                    fixture.cells,
                    fixture.moments,
                    fixture.candidates,
                    fixture.geometry,
                    fixture.pairs,
                ),
            ),
            mock.patch.object(
                selected.real_m1,
                "select_outer_lambdas",
                return_value=fixture.selection,
            ),
            mock.patch.object(
                selected.real_m1,
                "tune_and_refit_full_development",
                return_value=fixture.full_result,
            ),
            mock.patch.object(
                selected.m2,
                "fit_residual_transform",
                return_value=(None, fixture.m2_prepared),
            ),
            mock.patch.object(
                selected.executor,
                "_deep_integrity_digest",
                return_value=b"integrity",
            ),
            mock.patch.object(
                selected.observed,
                "_complete_selected_refit_full_m1_request",
                side_effect=complete,
            ) as complete_mock,
            mock.patch.object(
                selected.observed,
                "_burn_selected_refit_procedure",
            ) as burn,
        ):
            result = selected._fresh_full_m1(
                fixture.owner,
                fixture.terminal,
                fixture.authority,
                output_root,
            )
        return result, complete_mock, burn

    def test_fresh_full_m1_builds_and_registers_exact_66_12_1_objective(
        self,
    ) -> None:
        fixture = self._fresh_m1_fixture()
        with tempfile.TemporaryDirectory() as raw:
            full, complete, burn = self._run_fresh_m1_fixture(
                fixture, Path(raw)
            )
        complete.assert_called_once()
        burn.assert_not_called()
        self.assertEqual(full.audit_evidence["leave_two_path_count"], 66)
        self.assertEqual(full.audit_evidence["leave_one_path_count"], 12)
        self.assertEqual(full.audit_evidence["zero_full_refit_count"], 1)
        self.assertEqual(len(full.audit_evidence["training_paths"]), 78)
        self.assertTrue(full.audit_evidence["all_79_bundle_identities_unique"])
        self.assertTrue(full.audit_evidence["all_checkpoints_fresh"])
        self.assertEqual(full.prepared.transform.rank, 104)
        self.assertTrue(
            all(
                full.audit_evidence["M1_M2_full_transform_exact_matches"].values()
            )
        )
        for operation in (
            lambda: copy.copy(full),
            lambda: copy.deepcopy(full),
            lambda: pickle.dumps(full),
        ):
            with self.assertRaises(selected.SelectedRefitError):
                operation()
        selected_lambda = full.selected_lambda
        full.selected_lambda = float(m1.LAMBDAS_DESCENDING[-1])
        with self.assertRaisesRegex(
            selected.SelectedRefitError, "changed after issuance"
        ):
            selected._full_objective_authority_record(
                full,
                fixture.authority,
                fixture.owner,
                fixture.terminal,
            )
        full.selected_lambda = selected_lambda
        with self.assertRaises(selected.SelectedRefitError):
            selected._FullObjective(
                fixture.theta,
                fixture.m2_prepared,
                float(m1.LAMBDAS_DESCENDING[0]),
                {},
            )

    def test_fresh_full_m1_reused_path_rejects_and_burns(self) -> None:
        fixture = self._fresh_m1_fixture(reused=True)
        with tempfile.TemporaryDirectory() as raw:
            with (
                mock.patch.object(
                    selected.observed,
                    "_begin_selected_refit_full_m1_request",
                    return_value=b"begun",
                ),
                mock.patch.object(
                    selected.observed,
                    "_selected_refit_source_request",
                    return_value=(
                        fixture.cells,
                        fixture.moments,
                        fixture.candidates,
                        fixture.geometry,
                        fixture.pairs,
                    ),
                ),
                mock.patch.object(
                    selected.real_m1,
                    "select_outer_lambdas",
                    return_value=fixture.selection,
                ),
                mock.patch.object(
                    selected.real_m1,
                    "tune_and_refit_full_development",
                    return_value=fixture.full_result,
                ),
                mock.patch.object(
                    selected.m2,
                    "fit_residual_transform",
                    return_value=(None, fixture.m2_prepared),
                ),
                mock.patch.object(
                    selected.observed,
                    "_complete_selected_refit_full_m1_request",
                ) as complete,
                mock.patch.object(
                    selected.observed,
                    "_burn_selected_refit_procedure",
                ) as burn,
            ):
                with self.assertRaisesRegex(
                    selected.SelectedRefitError,
                    "fresh full-M1 refit binding is incomplete",
                ):
                    selected._fresh_full_m1(
                        fixture.owner,
                        fixture.terminal,
                        fixture.authority,
                        Path(raw),
                    )
        complete.assert_not_called()
        burn.assert_called_once_with(fixture.authority)

    def test_octant_diagnostic_is_complete_equal_group_and_strictly_bounded(
        self,
    ) -> None:
        # The equal-group mean is exactly -0.01, while unequal group sizes make
        # the pooled-cell descriptive mean different.
        boundary = self._outer_fixture((-0.021,) + (-0.009,) * 11)
        boundary_audit = self._audit(boundary)
        diagnostic = boundary_audit.octant_diagnostic
        rows = diagnostic["group_by_octant_all_96"]
        summaries = diagnostic["octant_summaries"]
        self.assertEqual(len(rows), 96)
        self.assertEqual(len(summaries), 8)
        self.assertEqual(len(boundary_audit.raw_cells), 194)
        self.assertEqual(sum(value["count"] for value in rows), 194)
        self.assertEqual(
            [value["octant"] for value in summaries],
            list(selected.OCTANT_LABELS),
        )
        empty_rows = [value for value in rows if value["octant"] != "111"]
        self.assertEqual(len(empty_rows), 84)
        self.assertTrue(
            all(
                value["count"] == 0
                and value["mean_outer_CV_M2_minus_M1_gain_bits_per_cell"]
                is None
                for value in empty_rows
            )
        )
        empty_summaries = [
            value for value in summaries if value["octant"] != "111"
        ]
        self.assertTrue(
            all(
                value["primary_equal_nonempty_group_gain_bits_per_cell"] is None
                and value["descriptive_pooled_cell_gain_bits_per_cell"] is None
                for value in empty_summaries
            )
        )
        occupied = next(value for value in summaries if value["octant"] == "111")
        self.assertEqual(occupied["nonempty_group_count"], 12)
        self.assertEqual(
            occupied["primary_equal_nonempty_group_gain_bits_per_cell"],
            selected.MOP_GAIN_FLOOR_BITS_PER_CELL,
        )
        self.assertNotEqual(
            occupied["primary_equal_nonempty_group_gain_bits_per_cell"],
            occupied["descriptive_pooled_cell_gain_bits_per_cell"],
        )
        self.assertFalse(diagnostic["scope_limitation"])
        self.assertEqual(diagnostic["material_octants_below_boundary"], [])

        below = self._outer_fixture((-0.0100000001,) * 12)
        below_diagnostic = self._audit(below).octant_diagnostic
        self.assertTrue(below_diagnostic["scope_limitation"])
        self.assertEqual(
            below_diagnostic["material_octants_below_boundary"], ["111"]
        )
        self.assertTrue(
            below_diagnostic[
                "diagnostic_only_no_filter_reselection_or_pass_gate"
            ]
        )

    def test_callback_side_octant_reauthentication_rejects_relabeling(
        self,
    ) -> None:
        fixture = self._outer_fixture((0.0,) * 12)
        relabeled = dataclasses.replace(
            fixture.strata,
            octants=("000",) + fixture.strata.octants[1:],
        )
        with self.assertRaisesRegex(
            selected.SelectedRefitError, "lost source alignment"
        ):
            self._audit(fixture, relabeled)

    def test_outer_crossfit_group_leave_one_dominance_and_direction_math(
        self,
    ) -> None:
        requested_gains = tuple(0.001 * (index + 1) for index in range(12))
        nine_stable_contrasts = (1.0,) * 3 + (-0.25,) * 9
        fixture = self._outer_fixture(
            requested_gains,
            nine_stable_contrasts,
        )
        audit = self._audit(fixture)
        expected_gains = {
            group: math.fsum(
                [
                    (requested_gains[index] * m1.LN2) / m1.LN2
                    for _ in range(controller.GROUP_SIZES[group])
                ]
            )
            / controller.GROUP_SIZES[group]
            for index, group in enumerate(m1.GROUP_ORDER)
        }
        expected_leave_one_gains = {
            group: math.fsum(
                expected_gains[other]
                for other in m1.GROUP_ORDER
                if other != group
            )
            / 11.0
            for group in m1.GROUP_ORDER
        }
        expected_contributions = {
            group: math.fsum(
                [nine_stable_contrasts[index]]
                * controller.GROUP_SIZES[group]
            )
            / controller.GROUP_SIZES[group]
            for index, group in enumerate(m1.GROUP_ORDER)
        }
        expected_leave_one_contrasts = {
            group: math.fsum(
                expected_contributions[other]
                for other in m1.GROUP_ORDER
                if other != group
            )
            / 11.0
            for group in m1.GROUP_ORDER
        }
        expected_absolute_sum = math.fsum(
            abs(value) for value in expected_gains.values()
        )
        self.assertEqual(dict(audit.group_gains), expected_gains)
        self.assertEqual(dict(audit.leave_one_gains), expected_leave_one_gains)
        self.assertEqual(
            audit.dominance_ratio,
            max(abs(value) for value in expected_gains.values())
            / expected_absolute_sum,
        )
        self.assertEqual(
            dict(audit.selected_group_contributions), expected_contributions
        )
        self.assertEqual(
            dict(audit.leave_one_contrasts), expected_leave_one_contrasts
        )
        self.assertEqual(audit.stable_count, 9)
        self.assertGreaterEqual(
            audit.stable_count, selected.DIRECTION_STABILITY_MINIMUM
        )

        eight_stable = self._audit(
            self._outer_fixture(
                requested_gains,
                (1.0,) * 4 + (-0.4,) * 8,
            )
        )
        self.assertEqual(eight_stable.stable_count, 8)
        self.assertLess(
            eight_stable.stable_count, selected.DIRECTION_STABILITY_MINIMUM
        )

        changed_selection = dict(fixture.candidate_selection)
        changed_selection["direction"] = "negative"
        with self.assertRaisesRegex(
            selected.SelectedRefitError,
            "selection is not reproducible",
        ):
            self._audit(
                fixture,
                candidate_selection=changed_selection,
            )

        row = fixture.execution.pair_contrast_rows[0]
        object.__setattr__(
            row,
            "values",
            (float(row.values[0]) + 0.1,) + row.values[1:],
        )
        with self.assertRaisesRegex(
            selected.SelectedRefitError,
            "score rows and pair rows are not aligned",
        ):
            self._audit(fixture)

    def test_all_120_pair_maximum_not_only_selected_pair_controls_gate(
        self,
    ) -> None:
        class _CandidateFixture:
            def __init__(self) -> None:
                self._basis = SimpleNamespace(
                    count=2,
                    membership=np.asarray(
                        ((True, False), (False, True)), dtype=bool
                    ),
                )

            def basis(self, k: int) -> object:
                self.asserted_k = k
                return self._basis

        candidates = _CandidateFixture()
        cell = SimpleNamespace(identifier="cell", group=m1.GROUP_ORDER[0], k=2)
        full = SimpleNamespace(
            prepared=SimpleNamespace(cells=(cell,)),
            theta=np.zeros(1, dtype=float),
        )
        fit_pair = selected._FitPair(
            cold=SimpleNamespace(beta=np.zeros(1), objective=0.0),
            warm=SimpleNamespace(beta=np.ones(1), objective=0.0),
            specification=None,
            initializer_configuration=None,
        )
        cold_matrix = np.zeros((12, controller.PAIR_COUNT), dtype=float)
        warm_matrix = np.zeros_like(cold_matrix)
        cold_pairs = np.zeros(controller.PAIR_COUNT, dtype=float)
        warm_pairs = np.zeros(controller.PAIR_COUNT, dtype=float)
        warm_pairs[1] = 2.0 * selected.REPEAT_PAIR_CONTRAST_TOLERANCE
        distribution = SimpleNamespace(probabilities=np.asarray((0.5, 0.5)))
        with (
            mock.patch.object(
                selected.m2,
                "candidate_distribution",
                return_value=distribution,
            ),
            mock.patch.object(
                selected,
                "_group_pair_contrasts",
                side_effect=(
                    (cold_matrix, cold_pairs),
                    (warm_matrix, warm_pairs),
                ),
            ),
        ):
            probability_error, pair_error, evidence = selected._repeat_evidence(
                full,
                fit_pair,
                candidates,
                _Marker(),
                _Marker(),
                0,
            )
        self.assertEqual(probability_error, 0.0)
        self.assertEqual(
            evidence["outer_selected_pair_repeat"]["absolute_difference"], 0.0
        )
        self.assertGreater(pair_error, selected.REPEAT_PAIR_CONTRAST_TOLERANCE)

        boundary = self._outer_fixture((-0.021,) + (-0.009,) * 11)
        outer = self._audit(boundary)
        below = self._outer_fixture((-0.0100000001,) * 12)
        scope_only = dataclasses.replace(
            outer,
            octant_diagnostic=self._audit(below).octant_diagnostic,
        )
        report_full = SimpleNamespace(
            selected_lambda=1.0,
            audit_evidence={},
            prepared=SimpleNamespace(
                cells=boundary.full.prepared.cells,
                transform=SimpleNamespace(rank=executor.EXPECTED_PRODUCTION_RANK),
            ),
        )
        report_owner = SimpleNamespace(
            prepared=SimpleNamespace(
                candidates=_Marker(), geometry=_Marker(), pairs=_Marker()
            )
        )
        report_winner = SimpleNamespace(
            trial=SimpleNamespace(valid_index=7)
        )
        configuration = SimpleNamespace(key="selected", family="lasso")
        selection = {
            "pair_index": 0,
            "pair": controller.PAIR_NAMES[0],
        }
        with (
            mock.patch.object(
                selected, "_outer_crossfit_audit", return_value=scope_only
            ),
            mock.patch.object(
                selected,
                "_repeat_evidence",
                return_value=(probability_error, pair_error, evidence),
            ),
        ):
            payload = selected._completion_payload(
                report_owner,
                _Marker(),
                report_winner,
                selection,
                configuration,
                report_full,
                fit_pair,
                boundary.strata,
            )
        self.assertEqual(payload["status"], "failed_completion_gate")
        self.assertFalse(
            payload["completion_gates"]["repeat_identified_pair_contrast"]
        )
        self.assertTrue(
            payload["target_blind_position_octant_diagnostic"][
                "scope_limitation"
            ]
        )
        self.assertNotIn(
            "target_blind_position_octant_diagnostic",
            payload["completion_gates"],
        )
        self.assertFalse(payload["selection_reopened"])

    def test_octant_scope_limitation_never_changes_a_passing_completion(
        self,
    ) -> None:
        boundary = self._outer_fixture((-0.021,) + (-0.009,) * 11)
        outer = self._audit(boundary)
        below = self._outer_fixture((-0.0100000001,) * 12)
        scope_only = dataclasses.replace(
            outer,
            octant_diagnostic=self._audit(below).octant_diagnostic,
        )
        full = SimpleNamespace(
            selected_lambda=1.0,
            audit_evidence={},
            prepared=SimpleNamespace(
                cells=boundary.full.prepared.cells,
                transform=SimpleNamespace(rank=executor.EXPECTED_PRODUCTION_RANK),
            ),
        )
        owner = SimpleNamespace(
            prepared=SimpleNamespace(
                candidates=_Marker(), geometry=_Marker(), pairs=_Marker()
            )
        )
        fit_pair = selected._FitPair(
            cold=SimpleNamespace(objective=0.0),
            warm=SimpleNamespace(objective=0.0),
            specification=None,
            initializer_configuration=None,
        )
        winner = SimpleNamespace(trial=SimpleNamespace(valid_index=7))
        configuration = SimpleNamespace(key="selected", family="lasso")
        with (
            mock.patch.object(
                selected, "_outer_crossfit_audit", return_value=scope_only
            ),
            mock.patch.object(
                selected,
                "_repeat_evidence",
                return_value=(
                    0.0,
                    0.0,
                    {
                        "outer_selected_pair_repeat": {
                            "absolute_difference": 0.0
                        }
                    },
                ),
            ),
        ):
            payload = selected._completion_payload(
                owner,
                _Marker(),
                winner,
                {"pair_index": 0, "pair": controller.PAIR_NAMES[0]},
                configuration,
                full,
                fit_pair,
                boundary.strata,
            )
        self.assertEqual(payload["status"], "pass")
        self.assertTrue(all(payload["completion_gates"].values()))
        self.assertTrue(
            payload["target_blind_position_octant_diagnostic"][
                "scope_limitation"
            ]
        )
        self.assertEqual(
            set(payload["completion_gates"]),
            {
                "repeat_objective",
                "repeat_candidate_probability",
                "repeat_identified_pair_contrast",
                "no_single_group_dominance",
                "selected_pair_direction_stability",
                "MOp_source_family",
            },
        )

    def test_hybrid_precursor_remains_strongly_live_through_cold_and_warm(
        self,
    ) -> None:
        owner = SimpleNamespace(
            prepared=SimpleNamespace(
                candidates=_Marker(), geometry=_Marker(), pairs=_Marker()
            ),
            maximum_iterations=1,
        )
        configuration = controller.Configuration(
            "sparse_plus_low_rank",
            0.001,
            rank=2,
            sparse_rho=0.001,
            sparse_edges=(0,),
        )
        specification = _Marker()
        initializer = configuration
        binding = SimpleNamespace(
            target_specification_reference=lambda: specification,
            initializer_configuration_reference=lambda: initializer,
            phase="issued_for_precursor",
            selected_anchor_reference=None,
            cold_result_reference=None,
            warm_result_reference=None,
        )
        precursor_reference: list[weakref.ReferenceType[_Marker] | None] = [None]
        cold_reference: list[weakref.ReferenceType[_Marker] | None] = [None]
        receipt_reference: list[weakref.ReferenceType[_Marker] | None] = [None]

        def authorize(*args: object, **kwargs: object) -> _Marker:
            del args, kwargs
            result = _Marker()
            receipt_reference[0] = weakref.ref(result)
            return result

        def precursor(*args: object, **kwargs: object) -> _Marker:
            del args, kwargs
            result = _Marker()
            precursor_reference[0] = weakref.ref(result)
            binding.selected_anchor_reference = precursor_reference[0]
            binding.phase = "precursor_ready"
            return result

        def cold(*args: object, **kwargs: object) -> _Marker:
            del args, kwargs
            gc.collect()
            self.assertIsNotNone(precursor_reference[0])
            assert precursor_reference[0] is not None
            self.assertIsNotNone(precursor_reference[0]())
            result = _Marker()
            cold_reference[0] = weakref.ref(result)
            binding.cold_result_reference = cold_reference[0]
            binding.phase = "cold_ready"
            return result

        def warm(*args: object, **kwargs: object) -> _Marker:
            del args, kwargs
            gc.collect()
            assert precursor_reference[0] is not None
            assert cold_reference[0] is not None
            self.assertIsNotNone(precursor_reference[0]())
            self.assertIsNotNone(cold_reference[0]())
            result = _Marker()
            binding.warm_result_reference = weakref.ref(result)
            binding.phase = "warm_ready"
            return result

        def registered_result(result: object, mode: str) -> object:
            del result
            self.assertEqual(mode, "production")
            assert precursor_reference[0] is not None
            assert receipt_reference[0] is not None
            return SimpleNamespace(
                selected_anchor_reference=precursor_reference[0],
                production_receipt_reference=receipt_reference[0],
            )

        with (
            mock.patch.object(
                selected.optional,
                "authorize_terminal_full_development_hybrid",
                side_effect=authorize,
            ),
            mock.patch.object(
                selected.optional,
                "_require_terminal_full_development_hybrid_receipt",
                return_value=binding,
            ),
            mock.patch.object(
                selected.optional,
                "fit_terminal_full_development_hybrid_precursor",
                side_effect=precursor,
            ),
            mock.patch.object(
                selected.optional,
                "fit_terminal_full_development_hybrid_cold",
                side_effect=cold,
            ),
            mock.patch.object(
                selected.optional,
                "fit_terminal_full_development_hybrid_warm",
                side_effect=warm,
            ),
            mock.patch.object(
                selected.optional,
                "_require_registered_hybrid_result",
                side_effect=registered_result,
            ),
            mock.patch.object(
                selected.observed, "_burn_selected_refit_procedure"
            ) as burn,
        ):
            result = selected._fit_selected_pair(
                owner,
                _Marker(),
                _Marker(),
                configuration,
                _Marker(),
            )
            gc.collect()
            selected._require_hybrid_fit_pair_authority(result)
        burn.assert_not_called()
        self.assertIsNotNone(result.cold)
        self.assertIsNotNone(result.warm)
        assert receipt_reference[0] is not None
        assert precursor_reference[0] is not None
        self.assertIsNotNone(receipt_reference[0]())
        self.assertIsNotNone(precursor_reference[0]())
        self.assertEqual(len(result.authority_chain), 2)

    def test_terminal_hybrid_authorization_strongly_owns_its_minted_spec(
        self,
    ) -> None:
        optional = selected.optional
        owner = _Marker()
        owner.prepared = SimpleNamespace(
            candidates=_Marker(), geometry=_Marker(), pairs=_Marker()
        )
        terminal = _Marker()
        authority = _Marker()
        full = _Marker()
        winner = _Marker()
        coverage = _Marker()
        anchors = (_Marker(),)
        target = controller.Configuration(
            "sparse_plus_low_rank",
            0.001,
            rank=2,
            sparse_rho=0.001,
            sparse_edges=(0,),
        )
        initializer = controller.Configuration("low_rank", 0.001, rank=2)
        theta = np.zeros(1, dtype=float)
        prepared = _Marker()
        authorization_digest = b"terminal-hybrid"

        def observed_request(name: str, *args: object) -> object:
            del args
            if name == "_terminal_full_refit_hybrid_inputs_request":
                return winner, coverage, anchors, target, initializer
            if name in {
                "_reserve_terminal_full_refit_hybrid_request",
                "_reauthenticate_terminal_full_refit_hybrid_request",
            }:
                return authorization_digest
            if name == "_bind_selected_refit_hybrid_receipt":
                return None
            raise AssertionError(name)

        def objective_binding(
            fixed: np.ndarray,
            bound_prepared: object,
            candidates: object,
            geometry: object,
            pairs: object,
            specification_signature: tuple[object, ...],
            correction: object,
        ) -> object:
            self.assertIsNone(correction)
            return optional._ObjectiveBinding(
                prepared_reference=weakref.ref(bound_prepared),
                candidates_reference=weakref.ref(candidates),
                geometry_reference=weakref.ref(geometry),
                pairs_reference=weakref.ref(pairs),
                fixed_m1_reference=weakref.ref(fixed),
                fixed_m1_signature=(),
                prepared_signature=(),
                specification_signature=specification_signature,
                linear_correction_signature=None,
            )

        with (
            mock.patch.object(
                optional,
                "_observed_session_request",
                side_effect=observed_request,
            ),
            mock.patch.object(
                optional,
                "_selected_refit_request",
                return_value=(theta, prepared, {}, b"objective"),
            ),
            mock.patch.object(optional, "_require_full_development_prepared"),
            mock.patch.object(
                optional,
                "fold_low_rank_scales",
                return_value=SimpleNamespace(lambda_reference=2.0),
            ),
            mock.patch.object(
                optional.core,
                "fold_lambda_maximum",
                return_value=SimpleNamespace(value=3.0),
            ),
            mock.patch.object(
                executor,
                "_validate_lambda_maximum",
                side_effect=lambda value: value,
            ),
            mock.patch.object(
                optional,
                "_require_hybrid_target_matches_specification",
            ),
            mock.patch.object(
                optional,
                "_coverage_evidence_signature",
                return_value=("coverage",),
            ),
            mock.patch.object(
                optional,
                "_controller_result_signature",
                return_value=("result",),
            ),
            mock.patch.object(
                optional,
                "_controller_configuration_signature",
                return_value=("configuration",),
            ),
            mock.patch.object(
                optional,
                "_hybrid_specification_signature",
                return_value=("specification",),
            ),
            mock.patch.object(
                optional,
                "_make_objective_binding",
                side_effect=objective_binding,
            ),
            mock.patch.object(optional, "_require_same_objective"),
        ):
            receipt = optional._authorize_terminal_full_development_hybrid_impl(
                owner,
                terminal,
                authority,
                full,
            )
            gc.collect()
            binding = optional._require_terminal_full_development_hybrid_receipt(
                receipt
            )
        self.assertIs(
            binding.target_specification_reference(),
            binding.target_specification,
        )

    def test_hybrid_post_warm_receipt_mutation_rejects_and_burns(self) -> None:
        owner = SimpleNamespace(
            prepared=SimpleNamespace(
                candidates=_Marker(), geometry=_Marker(), pairs=_Marker()
            ),
            maximum_iterations=1,
        )
        configuration = controller.Configuration(
            "sparse_plus_low_rank",
            0.001,
            rank=2,
            sparse_rho=0.001,
            sparse_edges=(0,),
        )
        receipt = _Marker()
        binding = SimpleNamespace(
            target_specification_reference=lambda: _Marker(),
            initializer_configuration_reference=lambda: configuration,
        )
        with (
            mock.patch.object(
                selected.optional,
                "authorize_terminal_full_development_hybrid",
                return_value=receipt,
            ),
            mock.patch.object(
                selected.optional,
                "_require_terminal_full_development_hybrid_receipt",
                side_effect=(
                    binding,
                    selected.optional.LowRankError("post-warm spec mutation"),
                ),
            ),
            mock.patch.object(
                selected.optional,
                "fit_terminal_full_development_hybrid_precursor",
                return_value=_Marker(),
            ),
            mock.patch.object(
                selected.optional,
                "fit_terminal_full_development_hybrid_cold",
                return_value=_Marker(),
            ),
            mock.patch.object(
                selected.optional,
                "fit_terminal_full_development_hybrid_warm",
                return_value=_Marker(),
            ),
            mock.patch.object(
                selected.observed,
                "_burn_selected_refit_procedure",
            ) as burn,
        ):
            with self.assertRaisesRegex(
                selected.SelectedRefitError, "final reauthentication"
            ):
                selected._fit_selected_pair(
                    owner,
                    _Marker(),
                    _Marker(),
                    configuration,
                    _Marker(),
                )
        burn.assert_called_once()

    def test_hybrid_child_burn_during_solver_cannot_be_resurrected(self) -> None:
        for stage in ("cold", "warm"):
            with self.subTest(stage=stage):
                receipt = _Marker()
                anchor = _Marker()
                cold = _Marker()
                result = _Marker()
                binding = SimpleNamespace(
                    lock=threading.RLock(),
                    phase=("precursor_ready" if stage == "cold" else "cold_ready"),
                    selected_anchor_reference=weakref.ref(anchor),
                    cold_result_reference=(
                        None if stage == "cold" else weakref.ref(cold)
                    ),
                    warm_result_reference=None,
                )

                def solver(*args: object, **kwargs: object) -> object:
                    del args, kwargs
                    with binding.lock:
                        binding.phase = "burned"
                    return result

                fit = (
                    selected.optional.fit_terminal_full_development_hybrid_cold
                    if stage == "cold"
                    else selected.optional.fit_terminal_full_development_hybrid_warm
                )
                with (
                    mock.patch.object(
                        selected.optional,
                        "_terminal_receipt_minimal",
                        return_value=binding,
                    ),
                    mock.patch.object(
                        selected.optional,
                        "_terminal_stage_context",
                        return_value=(_Marker(), _Marker(), _Marker(), _Marker()),
                    ),
                    mock.patch.object(
                        selected.optional,
                        "_observed_session_request",
                    ),
                    mock.patch.object(
                        selected.optional,
                        "_require_terminal_full_development_hybrid_receipt",
                        return_value=binding,
                    ),
                    mock.patch.object(
                        selected.optional,
                        "_terminal_objective",
                        return_value=(
                            _Marker(),
                            _Marker(),
                            _Marker(),
                            _Marker(),
                            _Marker(),
                            _Marker(),
                        ),
                    ),
                    mock.patch.object(
                        selected.optional,
                        "_fit_sparse_plus_low_rank_impl",
                        side_effect=solver,
                    ),
                ):
                    with self.assertRaisesRegex(
                        selected.optional.LowRankError,
                        f"{stage} publication lost its in-flight authority",
                    ):
                        if stage == "cold":
                            fit(receipt, maximum_sweeps=1)
                        else:
                            fit(receipt, maximum_sweeps=1)
                self.assertEqual(binding.phase, "burned")
                if stage == "cold":
                    self.assertIsNone(binding.cold_result_reference)
                else:
                    self.assertIsNone(binding.warm_result_reference)

    def test_all_family_final_reauth_failure_burns_without_report(self) -> None:
        owner = _Marker()
        terminal = _Marker()
        authority = _Marker()
        full = _Marker()
        fit_pair = selected._FitPair(
            cold=_Marker(),
            warm=_Marker(),
            specification=_Marker(),
            initializer_configuration=None,
        )
        configuration = SimpleNamespace(family="lasso")
        with tempfile.TemporaryDirectory() as raw:
            output_root = Path(raw)
            with (
                mock.patch.object(
                    selected.observed,
                    "_require_selected_refit_full_objective",
                    return_value=b"full",
                ),
                mock.patch.object(
                    selected.observed,
                    "_require_selected_refit_fit_pair",
                    side_effect=selected.observed.ObservedSessionError(
                        "post-warm mutation"
                    ),
                ),
                mock.patch.object(
                    selected.observed,
                    "_burn_selected_refit_procedure",
                ) as burn,
            ):
                with self.assertRaisesRegex(
                    selected.observed.ObservedSessionError,
                    "post-warm mutation",
                ):
                    selected._commit_completion_report(
                        output_root,
                        owner,
                        terminal,
                        authority,
                        configuration,
                        full,
                        fit_pair,
                        {"status": "pass"},
                    )
            self.assertFalse((output_root / selected.REPORT_FILENAME).exists())
        burn.assert_called_once_with(authority)

    def test_atomic_report_is_mode_0600_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / selected.REPORT_FILENAME
            selected._atomic_json(path, {"status": "pass"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(
                selected.SelectedRefitError, "already exists"
            ):
                selected._atomic_json(path, {"status": "pass"})


if __name__ == "__main__":
    unittest.main()
