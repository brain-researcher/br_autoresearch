#!/usr/bin/env python3
"""Focused, mocked tests for the in-memory M0/M1 benchmark replay helper."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import v3_m2_benchmark_replay_v1 as replay  # noqa: E402


def _context() -> SimpleNamespace:
    cells = []
    row_index = 0
    uniforms = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(replay.EXPECTED_SYNTHETIC_SEED))
    ).random(replay.EXPECTED_ROWS)
    for group_index, group in enumerate(replay.m1_real.m1.GROUP_ORDER):
        identifiers = replay._FROZEN_CELL_IDS_BY_GROUP[group]
        for local_index, identifier in enumerate(identifiers):
            cells.append(
                replay.m1_real.m1.Cell(
                    identifier=identifier,
                    group=group,
                    layer=replay.m1_real.m1.LAYERS[
                        (group_index + local_index)
                        % len(replay.m1_real.m1.LAYERS)
                    ],
                    hemisphere="left" if local_index % 2 == 0 else "right",
                    soma_um=(
                        float(row_index + 1),
                        float(row_index + 2),
                        float(row_index + 3),
                    ),
                    soma_distance_mm=np.linspace(
                        0.1, 1.6, replay.EXPECTED_PANEL_SIZE
                    ),
                    k=0,
                    observed_index=0,
                    segment_unmapped=local_index % 2,
                    sampling_uniform=float(uniforms[row_index]),
                )
            )
            row_index += 1
    membership = np.zeros(
        (replay.EXPECTED_ROWS, replay.EXPECTED_PANEL_SIZE), dtype=np.uint8
    )
    membership.setflags(write=False)
    return SimpleNamespace(
        cells=tuple(cells),
        membership_matrix=membership,
        pair_geometry=object(),
        synthetic_seed=replay.EXPECTED_SYNTHETIC_SEED,
        expected_folds=replay.EXPECTED_FOLDS,
        expected_residual_rank=replay.EXPECTED_RESIDUAL_RANK,
    )


class _NoFilesystem:
    def __fspath__(self) -> str:
        raise AssertionError("checkpoint path was inspected")

    def exists(self) -> bool:
        raise AssertionError("checkpoint path was inspected")


class _Basis:
    count = 1
    membership = np.zeros((1, replay.EXPECTED_PANEL_SIZE), dtype=np.uint8)


class _Library:
    def basis(self, k: int) -> _Basis:
        if int(k) != 0:
            raise AssertionError("test library supports only K=0")
        return _Basis()


class _Geometry:
    def uniform_moments(self, cell: object) -> object:
        del cell
        return object()


def _prepared() -> SimpleNamespace:
    return SimpleNamespace(
        design_transform=replay.m1_real.m1.DesignTransform(
            np.arange(9, dtype=float), np.arange(1, 10, dtype=float)
        ),
        geometry_scales=np.arange(1, 7, dtype=float),
    )


def _m1_result(value: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        theta=np.full(replay.m1_real.m1.PARAMETER_DIMENSION, value, dtype=float)
    )


class FreshReplayTests(unittest.TestCase):
    def test_context_conversion_builds_exact_m0_records_and_v7_is_installed(self) -> None:
        context = _context()
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
            receipt = replay.validate_synthetic_generation(context)
        records = replay.context_to_m0_records(context)
        self.assertEqual(len(records), replay.EXPECTED_ROWS)
        self.assertEqual(receipt.seed, replay.EXPECTED_SYNTHETIC_SEED)
        self.assertIn(str(replay.EXPECTED_SYNTHETIC_SEED), receipt.replay_identity)
        self.assertEqual(len(receipt.membership_sha256), 64)
        self.assertIs(replay.m0_v7.base.fit_lbfgs, replay.m0_v7.fit_lbfgs_v7)
        self.assertEqual(records[0].provider_neuron_id, context.cells[0].identifier)
        self.assertEqual(records[0].group, context.cells[0].group)
        self.assertEqual(records[0].source_acronym, context.cells[0].layer)
        self.assertEqual(records[0].outcome, tuple([0] * 16))
        self.assertEqual(records[0].k, 0)
        self.assertEqual(records[0].terminal_unmapped_report_only, 0)

    def test_context_conversion_rejects_mutable_or_misaligned_membership(self) -> None:
        context = _context()
        mutable = np.asarray(context.membership_matrix).copy()
        with self.assertRaisesRegex(replay.FreshReplayError, "read-only"):
            replay.context_to_m0_records(
                SimpleNamespace(**{**vars(context), "membership_matrix": mutable})
            )
        mutable[0, 0] = 1
        mutable.setflags(write=False)
        with self.assertRaisesRegex(replay.FreshReplayError, "observed fixed-K"):
            replay.context_to_m0_records(
                SimpleNamespace(**{**vars(context), "membership_matrix": mutable})
            )

    def test_context_rejects_wrong_inventory_seed_or_generation_uniforms(self) -> None:
        context = _context()
        wrong_inventory_cells = list(context.cells)
        wrong_inventory_cells[0] = replace(
            wrong_inventory_cells[0], identifier="not_in_frozen_inventory"
        )
        with self.assertRaisesRegex(replay.FreshReplayError, "frozen ID/group"):
            replay.validate_synthetic_generation(
                SimpleNamespace(
                    **{**vars(context), "cells": tuple(wrong_inventory_cells)}
                )
            )

        with self.assertRaisesRegex(replay.FreshReplayError, "synthetic seed"):
            replay.validate_synthetic_generation(
                SimpleNamespace(
                    **{
                        **vars(context),
                        "synthetic_seed": replay.EXPECTED_SYNTHETIC_SEED + 1,
                    }
                )
            )

        wrong_uniform_cells = list(context.cells)
        wrong_uniform_cells[0] = replace(wrong_uniform_cells[0], sampling_uniform=0.5)
        with self.assertRaisesRegex(replay.FreshReplayError, "PCG64 receipt"):
            replay.validate_synthetic_generation(
                SimpleNamespace(
                    **{**vars(context), "cells": tuple(wrong_uniform_cells)}
                )
            )

    def test_context_rejects_candidate_consistent_but_non_generated_membership(self) -> None:
        context = _context()
        library = replay.m1_real.m1.CandidateLibrary()
        basis = library.basis(1)
        cells = list(context.cells)
        cells[0] = replace(cells[0], k=1, observed_index=1)
        membership = np.asarray(context.membership_matrix).copy()
        membership[0] = basis.membership[1]
        membership.setflags(write=False)
        changed = SimpleNamespace(
            **{
                **vars(context),
                "cells": tuple(cells),
                "membership_matrix": membership,
            }
        )
        with (
            mock.patch.object(
                replay.m1_real.m1, "GeometryEngine", return_value=_Geometry()
            ),
            mock.patch.object(
                replay.m1_real.m1,
                "prepare_data",
                return_value=SimpleNamespace(cells=tuple(cells)),
            ),
            mock.patch.object(
                replay.m1_real.m1, "sample_candidate_index", return_value=0
            ),
        ):
            with self.assertRaisesRegex(replay.FreshReplayError, "planted-M1 receipt"):
                replay.validate_synthetic_generation(changed)

    def test_complete_m0_replay_calls_nested_tuning_and_full_refit(self) -> None:
        context = _context()
        block = replay.m0_v7.base.PredictionBlock(
            eta=np.zeros((194, 16), dtype=float),
            log_probability_nats=np.zeros(194, dtype=float),
            marginals=np.zeros((194, 16), dtype=float),
        )
        selected = 1.0
        details = [
            {
                "held_out_group": group,
                "selected_lambda": selected,
                "selected_lambda_is_grid_boundary": False,
            }
            for group in replay.m0_v7.base.GROUP_SIZES
        ]
        transform = replay.m0_v7.base.DesignTransform(
            np.zeros(3, dtype=float), np.ones(3, dtype=float)
        )
        optimizer = SimpleNamespace(coefficients=np.zeros((17, 15), dtype=float))
        replay.m0_v7.FIT_TELEMETRY[:] = [{"stale": True}]
        self.addCleanup(replay.m0_v7.FIT_TELEMETRY.clear)
        with (
            mock.patch.object(
                replay.m0_v7.base,
                "nested_outer_logo",
                return_value=(block, details, 0),
            ) as nested,
            mock.patch.object(
                replay.m0_v7.base,
                "tune_lambda_logo",
                return_value=(selected, {"selected": selected}),
            ) as tune,
            mock.patch.object(
                replay.m0_v7.base,
                "_fit_single",
                return_value=(transform, optimizer),
            ) as refit,
        ):
            result = replay.run_fresh_m0_replay(context)
        nested.assert_called_once()
        tune.assert_called_once()
        refit.assert_called_once()
        self.assertEqual(result.cross_fit_cells, 194)
        self.assertEqual(result.outer_fold_count, 12)
        self.assertEqual(result.outer_boundary_count, 0)
        self.assertEqual(result.outer_selected_lambdas, (selected,) * 12)
        self.assertEqual(result.full_selected_lambda, selected)
        self.assertIs(result.full_optimizer_result, optimizer)
        self.assertEqual(replay.m0_v7.FIT_TELEMETRY, [])
        self.assertEqual(
            result.generation_receipt.seed, replay.EXPECTED_SYNTHETIC_SEED
        )

    def test_m0_replay_preserves_six_boundary_stop_before_full_fit(self) -> None:
        context = _context()
        block = replay.m0_v7.base.PredictionBlock(
            eta=np.zeros((194, 16), dtype=float),
            log_probability_nats=np.zeros(194, dtype=float),
            marginals=np.zeros((194, 16), dtype=float),
        )
        details = []
        for ordinal, group in enumerate(replay.m0_v7.base.GROUP_SIZES):
            boundary = ordinal < 6
            details.append(
                {
                    "held_out_group": group,
                    "selected_lambda": (
                        replay.m0_v7.base.LAMBDAS[0] if boundary else 1.0
                    ),
                    "selected_lambda_is_grid_boundary": boundary,
                }
            )
        with (
            mock.patch.object(
                replay.m0_v7.base,
                "nested_outer_logo",
                return_value=(block, details, 6),
            ),
            mock.patch.object(replay.m0_v7.base, "tune_lambda_logo") as tune,
        ):
            with self.assertRaises(replay.m0_v7.base.BoundaryReview):
                replay.run_fresh_m0_replay(context)
        tune.assert_not_called()

    def test_in_memory_m1_factories_never_inspect_checkpoint_argument(self) -> None:
        checkpoint = _NoFilesystem()
        prepared = _prepared()
        warm_fits = {
            ridge: _m1_result(float(index))
            for index, ridge in enumerate(replay.m1_real.m1.LAMBDAS_DESCENDING)
        }
        with (
            mock.patch.object(
                replay.m1_real, "_prepare_training", return_value=prepared
            ) as prepare,
            mock.patch.object(
                replay.m1_real.m1, "fit_path", return_value=warm_fits
            ) as fit_path,
            mock.patch.object(replay.m1_real, "_validate_optimizer_result") as validate,
        ):
            bundle = replay.in_memory_m1_warm_factory(
                "leave_two__212074__220187",
                (2, 4),
                (object(),) * 5,
                (object(),) * 5,
                object(),
                object(),
                checkpoint,
            )
        prepare.assert_called_once()
        fit_path.assert_called_once()
        self.assertEqual(validate.call_count, len(warm_fits))
        self.assertEqual(bundle.indexes, (2, 4))
        self.assertFalse(bundle.checkpoint_reused)
        self.assertIs(bundle.fits, warm_fits)

        zero_result = _m1_result(7.0)
        ridge = 1.0
        with (
            mock.patch.object(
                replay.m1_real, "_prepare_training", return_value=prepared
            ),
            mock.patch.object(
                replay.m1_real.m1, "fit_newton", return_value=zero_result
            ) as fit_zero,
            mock.patch.object(replay.m1_real, "_validate_optimizer_result") as validate,
        ):
            bundle = replay.in_memory_m1_zero_factory(
                "outer_refit__212074",
                (1, 3),
                ridge,
                (object(),) * 5,
                (object(),) * 5,
                object(),
                object(),
                checkpoint,
            )
        self.assertEqual(
            fit_zero.call_args.args[3:], (ridge,)
        )
        self.assertIsNone(fit_zero.call_args.kwargs["initial"])
        validate.assert_called_once_with(zero_result, ridge)
        self.assertEqual(bundle.fits, {ridge: zero_result})
        self.assertFalse(bundle.checkpoint_reused)

    def test_m1_replay_returns_fold_bound_offsets_and_full_fit(self) -> None:
        context = _context()
        selected = {group: 1.0 for group in replay.m1_real.m1.GROUP_ORDER}
        warm_cache = {key: object() for key in replay._EXPECTED_LEAVE_TWO_KEYS}
        selection = replay.m1_real.OuterSelection(
            selected_lambdas=selected,
            details=tuple(),
            boundary_count=0,
            warm_cache=warm_cache,
        )
        prepared = _prepared()
        outer_refits = {}
        for ordinal, group in enumerate(replay.m1_real.m1.GROUP_ORDER, start=1):
            indexes = tuple(
                index for index, cell in enumerate(context.cells) if cell.group != group
            )
            result = _m1_result(float(ordinal))
            outer_refits[group] = replay.m1_real.TrainingBundle(
                f"outer_refit__{group}",
                indexes,
                prepared,
                {1.0: result},
                False,
            )
        cross_fit = replay.m1_real.CrossFitResult(
            prediction=replay.m1_real.Prediction(
                np.zeros(194, dtype=float), np.zeros((194, 16), dtype=float)
            ),
            outer_details=tuple(),
            boundary_count=0,
            warm_cache=warm_cache,
            outer_refits=outer_refits,
        )
        full_optimizer = _m1_result(99.0)
        full_bundle = replay.m1_real.TrainingBundle(
            "full_refit",
            tuple(range(194)),
            prepared,
            {1.0: full_optimizer},
            False,
        )
        full_warm_cache = {key: object() for key in replay._EXPECTED_WARM_KEYS}
        full = replay.m1_real.FullFitResult(
            selected_lambda=1.0,
            selection={"selected": 1.0},
            warm_cache=full_warm_cache,
            full_refit=full_bundle,
        )

        def select_side_effect(*args: object, **kwargs: object) -> object:
            warm_factory = kwargs["warm_factory"]
            for key in replay._EXPECTED_LEAVE_TWO_KEYS:
                warm_factory(key, (), *args[:4], args[4])
            return selection

        def refit_side_effect(*args: object, **kwargs: object) -> object:
            zero_factory = kwargs["zero_factory"]
            for key in replay._EXPECTED_OUTER_REFIT_KEYS:
                zero_factory(key, (), 1.0, *args[:4], args[4])
            return cross_fit

        def full_side_effect(*args: object, **kwargs: object) -> object:
            warm_factory = kwargs["warm_factory"]
            zero_factory = kwargs["zero_factory"]
            for key in replay._EXPECTED_LEAVE_ONE_KEYS:
                warm_factory(key, (), *args[:4], args[4])
            zero_factory("full_refit", tuple(range(194)), 1.0, *args[:4], args[4])
            return full

        with (
            mock.patch.object(
                replay.m1_real.m1, "CandidateLibrary", return_value=_Library()
            ),
            mock.patch.object(
                replay.m1_real.m1, "GeometryEngine", return_value=_Geometry()
            ),
            mock.patch.object(
                replay.m1_real,
                "select_outer_lambdas",
                side_effect=select_side_effect,
            ) as select,
            mock.patch.object(
                replay.m1_real,
                "refit_outer_and_predict",
                side_effect=refit_side_effect,
            ) as refit,
            mock.patch.object(
                replay.m1_real,
                "tune_and_refit_full_development",
                side_effect=full_side_effect,
            ) as full_refit,
            mock.patch.object(
                replay, "in_memory_m1_warm_factory", return_value=object()
            ) as warm_factory,
            mock.patch.object(
                replay, "in_memory_m1_zero_factory", return_value=object()
            ) as zero_factory,
        ):
            result = replay.run_fresh_m1_replay(context)

        self.assertIs(select.call_args.args[4], replay._NO_CHECKPOINT_IO)
        self.assertIs(refit.call_args.args[4], replay._NO_CHECKPOINT_IO)
        self.assertIs(full_refit.call_args.args[4], replay._NO_CHECKPOINT_IO)
        self.assertTrue(callable(select.call_args.kwargs["warm_factory"]))
        self.assertTrue(callable(refit.call_args.kwargs["zero_factory"]))
        self.assertEqual(warm_factory.call_count, replay.EXPECTED_M1_WARM_PATHS)
        self.assertEqual(zero_factory.call_count, replay.EXPECTED_M1_ZERO_REFITS)
        self.assertTrue(
            all(call.args[-1] is replay._NO_CHECKPOINT_IO for call in warm_factory.call_args_list)
        )
        self.assertTrue(
            all(call.args[-1] is replay._NO_CHECKPOINT_IO for call in zero_factory.call_args_list)
        )
        self.assertEqual(tuple(result.outer_offsets), replay.m1_real.m1.GROUP_ORDER)
        self.assertEqual(result.outer_fold_count, 12)
        self.assertEqual(result.warm_path_count, 78)
        self.assertEqual(result.zero_refit_count, 13)
        first_group = replay.m1_real.m1.GROUP_ORDER[0]
        first = result.outer_offsets[first_group]
        self.assertEqual(first.held_out_group, first_group)
        self.assertFalse(first.theta.flags.writeable)
        self.assertFalse(first.geometry_scales.flags.writeable)
        self.assertTrue(first.fit_identity.endswith("01"))
        self.assertTrue(first.fit_identity.startswith(result.generation_receipt.replay_identity))
        self.assertEqual(result.full_fit.training_indexes, tuple(range(194)))
        self.assertEqual(result.full_fit.ridge_lambda, 1.0)
        self.assertFalse(result.full_fit.theta.flags.writeable)
        self.assertIs(result.full_fit.optimizer_result, full_optimizer)
        self.assertTrue(
            result.full_fit.fit_identity.startswith(
                result.generation_receipt.replay_identity
            )
        )

    def test_m1_factory_accounting_rejects_missing_warm_or_zero_call(self) -> None:
        accounting = replay._M1FactoryAccounting()
        for key in replay._EXPECTED_WARM_KEYS[:-1]:
            accounting.record_warm(key, replay._NO_CHECKPOINT_IO)
        for key in replay._EXPECTED_ZERO_KEYS:
            accounting.record_zero(key, replay._NO_CHECKPOINT_IO)
        with self.assertRaisesRegex(replay.FreshReplayError, "exactly 78"):
            accounting.validate_complete()

        accounting = replay._M1FactoryAccounting()
        for key in replay._EXPECTED_WARM_KEYS:
            accounting.record_warm(key, replay._NO_CHECKPOINT_IO)
        for key in replay._EXPECTED_ZERO_KEYS[:-1]:
            accounting.record_zero(key, replay._NO_CHECKPOINT_IO)
        with self.assertRaisesRegex(replay.FreshReplayError, "exactly 13"):
            accounting.validate_complete()

    def test_m1_replay_preserves_boundary_stop_before_zero_refits(self) -> None:
        context = _context()
        warm_cache = {key: object() for key in replay._EXPECTED_LEAVE_TWO_KEYS}
        selection = replay.m1_real.OuterSelection(
            selected_lambdas={
                group: 1.0 for group in replay.m1_real.m1.GROUP_ORDER
            },
            details=tuple(),
            boundary_count=6,
            warm_cache=warm_cache,
        )

        def select_side_effect(*args: object, **kwargs: object) -> object:
            warm_factory = kwargs["warm_factory"]
            for key in replay._EXPECTED_LEAVE_TWO_KEYS:
                warm_factory(key, (), *args[:4], args[4])
            return selection

        with (
            mock.patch.object(
                replay.m1_real.m1, "CandidateLibrary", return_value=_Library()
            ),
            mock.patch.object(
                replay.m1_real.m1, "GeometryEngine", return_value=_Geometry()
            ),
            mock.patch.object(
                replay.m1_real,
                "select_outer_lambdas",
                side_effect=select_side_effect,
            ),
            mock.patch.object(
                replay, "in_memory_m1_warm_factory", return_value=object()
            ) as warm_factory,
            mock.patch.object(
                replay.m1_real, "tune_and_refit_full_development"
            ) as full_refit,
        ):
            with self.assertRaises(replay.m1_real.BoundaryReview):
                replay.run_fresh_m1_replay(context)
        self.assertEqual(warm_factory.call_count, len(replay._EXPECTED_LEAVE_TWO_KEYS))
        full_refit.assert_not_called()

    def test_timing_and_report_api_exposes_aggregate_units_only(self) -> None:
        m0_result = object()
        m1_result = object()
        with (
            mock.patch.object(
                replay, "run_fresh_m0_replay", return_value=m0_result
            ),
            mock.patch.object(
                replay, "run_fresh_m1_replay", return_value=m1_result
            ),
            mock.patch.object(
                replay.time,
                "perf_counter",
                side_effect=(10.0, 12.5, 20.0, 24.0),
            ),
            mock.patch.object(
                replay.time,
                "process_time",
                side_effect=(4.0, 5.5, 8.0, 10.0),
            ),
            mock.patch.object(replay, "_peak_rss_gib", side_effect=(1.25, 1.5)),
        ):
            measured = replay.run_fresh_replay_benchmark(object())
        self.assertIs(measured.m0.result, m0_result)
        self.assertIs(measured.m1.result, m1_result)
        self.assertEqual(measured.m0.timing.wall_seconds, 2.5)
        self.assertEqual(measured.m1.timing.wall_seconds, 4.0)
        units = replay.aggregate_benchmark_units(measured)
        self.assertEqual(set(units), {replay.M0_UNIT_NAME, replay.M1_UNIT_NAME})
        self.assertEqual(units[replay.M0_UNIT_NAME]["cpu_seconds"], 1.5)
        self.assertEqual(units[replay.M1_UNIT_NAME]["peak_rss_gib"], 1.5)
        for unit in units.values():
            self.assertEqual(unit["status"], "pass")
            self.assertEqual(
                unit["peak_rss_semantics"],
                "process_lifetime_high_water_conservative",
            )
            self.assertEqual(unit["scratch_gib"], 0.0)
            self.assertFalse(unit["real_membership_opened"])
            self.assertNotIn("identifier", unit)
            self.assertNotIn("membership", unit)
            self.assertNotIn("prediction", unit)


if __name__ == "__main__":
    unittest.main()
