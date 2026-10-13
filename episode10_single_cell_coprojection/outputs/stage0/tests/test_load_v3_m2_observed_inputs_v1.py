#!/usr/bin/env python3
"""Target-free contract tests for the sealed EP10 v3 M2 input loader."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(TOOLS_DIR))

import load_v3_m2_observed_inputs_v1 as loader  # noqa: E402


def _pair_geometry() -> Any:
    size = loader.m2.m1.PANEL_SIZE
    distance = np.zeros((size, size), dtype=float)
    cross = np.zeros((size, size), dtype=float)
    homotopic = np.zeros((size, size), dtype=float)
    edges: list[tuple[float, int, int]] = []
    for left in range(size - 1):
        for right in range(left + 1, size):
            value = float(right - left)
            distance[left, right] = distance[right, left] = value
            edges.append((value, left, right))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return loader.m2.m1.PairGeometry(distance, cross, homotopic, tuple(edges))


def _target_blind_cells() -> tuple[Any, ...]:
    cells: list[Any] = []
    ordinal = 0
    for group_index, group in enumerate(loader.m2.m1.GROUP_ORDER):
        for local_index in range(loader.m2.m1.GROUP_SIZES[group]):
            cells.append(
                loader.m2.Cell(
                    identifier=f"synthetic_{ordinal:03d}",
                    group=group,
                    layer=loader.m2.m1.LAYERS[
                        (group_index + local_index) % len(loader.m2.m1.LAYERS)
                    ],
                    hemisphere="left" if local_index % 2 == 0 else "right",
                    soma_um=(
                        float(ordinal + 1),
                        float((ordinal + 1) ** 2),
                        float((ordinal + 1) ** 3),
                    ),
                    soma_distance_mm=np.linspace(
                        0.1,
                        1.6,
                        loader.m2.m1.PANEL_SIZE,
                    ),
                    k=1,
                    observed_index=ordinal % loader.m2.m1.PANEL_SIZE,
                    segment_unmapped=local_index % 2,
                )
            )
            ordinal += 1
    if len(cells) != loader.EXPECTED_ROWS:
        raise AssertionError("synthetic 194-cell inventory changed")
    return tuple(cells)


def _membership(cells: tuple[Any, ...]) -> dict[str, tuple[str, int, tuple[int, ...]]]:
    zero = (0,) * loader.m2.m1.PANEL_SIZE
    membership = {cell.identifier: (cell.group, 0, zero) for cell in cells}
    positive = [0] * loader.m2.m1.PANEL_SIZE
    positive[0] = 1
    positive[5] = 1
    membership[cells[0].identifier] = (cells[0].group, 2, tuple(positive))
    return membership


@contextmanager
def _sealed_roots() -> Iterator[tuple[Path, Path, Path, Path]]:
    with tempfile.TemporaryDirectory() as raw:
        scratch = Path(raw) / "scratch"
        episode = scratch.joinpath(*loader.EPISODE_SCRATCH_COMPONENTS)
        interface = episode / loader.INTERFACE_ROOT_BASENAME
        geometry = episode / loader.GEOMETRY_ROOT_BASENAME
        target = episode / loader.TARGET_ROOT_BASENAME
        checkpoint = (
            episode
            / loader.outer_import.PRODUCER_ROOT_BASENAME
            / loader.outer_import.CHECKPOINT_DIRECTORY
        )
        for path in (interface, geometry, target, checkpoint):
            path.mkdir(parents=True, exist_ok=True)
        with mock.patch.dict(os.environ, {"SCRATCH": str(scratch)}):
            yield interface, geometry, target, checkpoint


class ObservedInputLoaderTests(unittest.TestCase):
    def _load(self) -> tuple[loader.ObservedInputBundle, tuple[Any, ...]]:
        cells = _target_blind_cells()
        call_order: list[str] = []
        freeze_position_strata = loader._freeze_position_strata

        def freeze(supplied: tuple[Any, ...]) -> loader.PositionStrata:
            call_order.append("freeze_position_strata")
            return freeze_position_strata(supplied)

        def read_membership(*args: Any) -> tuple[Any, Any]:
            del args
            call_order.append("read_membership")
            return _membership(cells), counts

        counts = loader.real_m1.MembershipScanCounts(
            loader.real_m1.EXPECTED_TARGET_ROWS,
            loader.EXPECTED_ROWS,
            loader.real_m1.EXPECTED_EXCLUDED_ROWS,
        )
        with _sealed_roots() as (interface, geometry, target, _), mock.patch.object(
            loader.m2.m1,
            "load_target_free_real_shape",
            return_value=(cells, _pair_geometry()),
        ) as target_blind, mock.patch.object(
            loader,
            "_freeze_position_strata",
            side_effect=freeze,
        ), mock.patch.object(
            loader.real_m1,
            "_read_membership_inputs",
            side_effect=read_membership,
        ) as membership_reader:
            bundle = loader.load_observed_inputs(interface, geometry, target)
        target_blind.assert_called_once_with(interface, geometry)
        membership_reader.assert_called_once_with(
            target, {cell.identifier for cell in cells}
        )
        self.assertEqual(
            call_order,
            ["freeze_position_strata", "read_membership"],
        )
        return bundle, cells

    def test_loads_exact_live_objects_with_positive_only_alignment(self) -> None:
        bundle, target_blind = self._load()
        self.assertEqual(len(bundle.cells), 194)
        self.assertEqual(len(bundle.moments), 194)
        self.assertIs(type(bundle.candidates), loader.m2.CandidateLibrary)
        self.assertIs(type(bundle.geometry), loader.m2.GeometryEngine)
        self.assertIs(type(bundle.pairs), loader.m2.PairFeatureLibrary)
        self.assertIs(bundle.geometry.library, bundle.candidates)
        self.assertIs(bundle.pairs.candidates, bundle.candidates)
        self.assertTrue(all(type(cell) is loader.m2.Cell for cell in bundle.cells))
        self.assertTrue(
            all(type(moment) is loader.m2.UniformCellMoments for moment in bundle.moments)
        )
        self.assertEqual(bundle.cells[0].k, 2)
        np.testing.assert_array_equal(
            bundle.candidates.basis(2).membership[bundle.cells[0].observed_index],
            np.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
        self.assertTrue(
            all(cell.k == 0 and cell.observed_index == 0 for cell in bundle.cells[1:])
        )
        self.assertEqual(
            tuple(cell.segment_unmapped for cell in bundle.cells),
            tuple(cell.segment_unmapped for cell in target_blind),
        )
        self.assertEqual(sum(cell.group == "221588" for cell in bundle.cells), 7)
        kwargs = bundle.runner_kwargs()
        self.assertEqual(
            set(kwargs),
            {
                "cells",
                "moments",
                "candidates",
                "geometry",
                "pairs",
                "position_strata",
            },
        )
        self.assertIs(kwargs["cells"], bundle.cells)
        self.assertIs(kwargs["position_strata"], bundle.position_strata)
        with self.assertRaises(TypeError):
            kwargs["cells"] = ()  # type: ignore[index]

    def test_position_strata_xyz_order_and_median_ties_are_frozen_before_targets(self) -> None:
        cells = tuple(
            replace(
                cell,
                soma_um=(
                    0.0,
                    float(index - 96),
                    float(96 - index),
                ),
            )
            for index, cell in enumerate(_target_blind_cells())
        )
        strata = loader._freeze_position_strata(cells)
        coordinates = np.asarray([cell.soma_um for cell in cells], dtype=float)
        expected_medians = tuple(float(value) for value in np.median(coordinates, axis=0))
        self.assertEqual(strata.medians_xyz, expected_medians)
        expected = tuple(
            "".join(
                "1" if coordinate >= median else "0"
                for coordinate, median in zip(row, expected_medians, strict=True)
            )
            for row in coordinates
        )
        self.assertEqual(strata.octants, expected)
        self.assertTrue(all(octant[0] == "1" for octant in strata.octants))
        self.assertEqual(strata.octants[0], "101")
        self.assertEqual(strata.octants[-1], "110")

    def test_position_freeze_failure_prevents_target_membership_read(self) -> None:
        cells = _target_blind_cells()
        with _sealed_roots() as (interface, geometry, target, _), mock.patch.object(
            loader.m2.m1,
            "load_target_free_real_shape",
            return_value=(cells, _pair_geometry()),
        ), mock.patch.object(
            loader,
            "_freeze_position_strata",
            side_effect=loader.ObservedInputError("freeze failed"),
        ), mock.patch.object(
            loader.real_m1,
            "_read_membership_inputs",
        ) as membership_reader:
            with self.assertRaisesRegex(loader.ObservedInputError, "freeze failed"):
                loader.load_observed_inputs(interface, geometry, target)
        membership_reader.assert_not_called()

    def test_rejects_nonexact_or_symlinked_root_before_payload_reader(self) -> None:
        cells = _target_blind_cells()
        with _sealed_roots() as (interface, geometry, target, _):
            forbidden = target.parent / "final"
            forbidden.mkdir()
            with mock.patch.object(
                loader.m2.m1, "load_target_free_real_shape"
            ) as target_blind:
                with self.assertRaisesRegex(loader.ObservedInputError, "exact sealed"):
                    loader.load_observed_inputs(interface, geometry, forbidden)
                target_blind.assert_not_called()

        with tempfile.TemporaryDirectory() as raw:
            scratch = Path(raw) / "scratch"
            episode = scratch.joinpath(*loader.EPISODE_SCRATCH_COMPONENTS)
            interface = episode / loader.INTERFACE_ROOT_BASENAME
            geometry = episode / loader.GEOMETRY_ROOT_BASENAME
            target_destination = episode / "ordinary_development_target"
            interface.mkdir(parents=True)
            geometry.mkdir()
            target_destination.mkdir()
            target = episode / loader.TARGET_ROOT_BASENAME
            target.symlink_to(target_destination, target_is_directory=True)
            with mock.patch.dict(os.environ, {"SCRATCH": str(scratch)}), mock.patch.object(
                loader.m2.m1, "load_target_free_real_shape"
            ) as target_blind:
                with self.assertRaisesRegex(loader.ObservedInputError, "symlinked"):
                    loader.load_observed_inputs(interface, geometry, target)
                target_blind.assert_not_called()

    def test_rejects_wrong_alignment_and_projection_counts(self) -> None:
        cells = _target_blind_cells()
        bad_counts = loader.real_m1.MembershipScanCounts(199, 194, 5)
        with _sealed_roots() as (interface, geometry, target, _), mock.patch.object(
            loader.m2.m1,
            "load_target_free_real_shape",
            return_value=(cells, _pair_geometry()),
        ), mock.patch.object(
            loader.real_m1,
            "_read_membership_inputs",
            return_value=(_membership(cells), bad_counts),
        ):
            with self.assertRaisesRegex(loader.ObservedInputError, "counts changed"):
                loader.load_observed_inputs(interface, geometry, target)

        counts = loader.real_m1.MembershipScanCounts(200, 194, 6)
        with _sealed_roots() as (interface, geometry, target, _), mock.patch.object(
            loader.m2.m1,
            "load_target_free_real_shape",
            return_value=(cells, _pair_geometry()),
        ), mock.patch.object(
            loader.real_m1,
            "_read_membership_inputs",
            return_value=(_membership(cells), counts),
        ), mock.patch.object(
            loader.real_m1,
            "_align_membership",
            return_value=cells[:-1],
        ):
            with self.assertRaisesRegex(loader.ObservedInputError, "exactly 194"):
                loader.load_observed_inputs(interface, geometry, target)

    def test_importer_receives_same_cells_and_moments_only(self) -> None:
        bundle, _ = self._load()
        sentinel_inventory = object()
        sentinel_import = object()
        with _sealed_roots() as (_, _, _, checkpoint), mock.patch.object(
            loader.outer_import,
            "import_outer_m1_offsets",
            return_value=sentinel_import,
        ) as importer:
            result = loader.import_outer_m1_for_observed(
                bundle,
                checkpoint,
                execution_inventory=sentinel_inventory,
            )
        self.assertIs(result, sentinel_import)
        importer.assert_called_once_with(
            checkpoint,
            bundle.cells,
            bundle.moments,
            mode=loader.outer_import.executor.SEALED_PRODUCTION_MODE,
            execution_inventory=sentinel_inventory,
            expected_rank=loader.outer_import.executor.EXPECTED_PRODUCTION_RANK,
        )

    def test_parser_exposes_no_protected_data_route(self) -> None:
        parser = loader.build_parser()
        args = parser.parse_args(
            [
                "--interface-root",
                "/synthetic/interface",
                "--geometry-root",
                "/synthetic/geometry",
                "--target-root",
                "/synthetic/targets",
                "--m1-checkpoint-root",
                "/synthetic/checkpoints",
            ]
        )
        self.assertEqual(
            set(vars(args)),
            {"interface_root", "geometry_root", "target_root", "m1_checkpoint_root"},
        )
        for forbidden in (
            "--final-root",
            "--unassigned-root",
            "--swc-root",
            "--m0-full-root",
            "--m1-full-refit",
        ):
            with self.subTest(forbidden=forbidden), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--interface-root",
                        "/synthetic/interface",
                        "--geometry-root",
                        "/synthetic/geometry",
                        "--target-root",
                        "/synthetic/targets",
                        "--m1-checkpoint-root",
                        "/synthetic/checkpoints",
                        forbidden,
                        "/forbidden",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
