#!/usr/bin/env python3
"""Target-free tests for the EP10 v3 M1 outer-refit importer."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1_real_v1 as real_m1  # noqa: E402
import v3_m2_m1_outer_import_v1 as outer_import  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402


def _cells_and_moments() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    cells: list[Any] = []
    moments: list[Any] = []
    for group_index, group in enumerate(real_m1.m1.GROUP_ORDER):
        for local_index in range(2):
            ordinal = 2 * group_index + local_index + 1
            value = float(ordinal)
            cells.append(
                executor.m1.Cell(
                    identifier=f"synthetic_{group_index:02d}_{local_index}",
                    group=group,
                    layer=real_m1.m1.LAYERS[
                        (group_index + local_index) % len(real_m1.m1.LAYERS)
                    ],
                    hemisphere="left" if local_index == 0 else "right",
                    soma_um=(
                        value + 0.1 * (ordinal % 3),
                        value * value + 0.2 * (ordinal % 5),
                        value * value * value + 0.3 * (ordinal % 7),
                    ),
                    soma_distance_mm=np.linspace(
                        0.1 + 0.001 * ordinal,
                        1.6 + 0.001 * ordinal,
                        real_m1.m1.PANEL_SIZE,
                    ),
                    k=2,
                    observed_index=ordinal % math.comb(real_m1.m1.PANEL_SIZE, 2),
                    segment_unmapped=local_index,
                )
            )
            covariance = np.eye(real_m1.m1.JOINT_CELL_DIMENSION, dtype=float)
            moments.append(
                executor.m1.UniformCellMoments(
                    geometry_mean=np.linspace(
                        0.01,
                        0.06,
                        real_m1.m1.GEOMETRY_DIMENSION,
                    ),
                    raw_joint_covariance=covariance,
                )
            )
    return tuple(cells), tuple(moments)


def _optimizer(ridge_lambda: float, ordinal: int) -> Any:
    return real_m1.m1.OptimizerResult(
        theta=np.full(
            real_m1.m1.PARAMETER_DIMENSION,
            0.001 * float(ordinal),
            dtype=float,
        ),
        objective=1.0 + 0.01 * ordinal,
        gradient_infinity_norm=1.0e-9,
        iterations=ordinal,
        accepted_steps=ordinal,
        relative_objective_change=1.0e-13,
        telemetry={
            "status": "pass",
            "ridge_lambda": ridge_lambda,
            "convergence_witness": "gradient_and_relative_objective_change",
        },
    )


@contextmanager
def _producer_fixture() -> Iterator[tuple[Path, tuple[Any, ...], tuple[Any, ...]]]:
    cells, moments = _cells_and_moments()
    with tempfile.TemporaryDirectory() as raw:
        checkpoint_root = (
            Path(raw)
            / outer_import.PRODUCER_ROOT_BASENAME
            / outer_import.CHECKPOINT_DIRECTORY
        )
        checkpoint_root.mkdir(parents=True)
        previous = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = outer_import.PRODUCER_SLURM_JOB_ID
        try:
            for ordinal, group in enumerate(real_m1.m1.GROUP_ORDER, start=1):
                indexes = tuple(
                    index for index, cell in enumerate(cells) if cell.group != group
                )
                training_groups = tuple(
                    candidate
                    for candidate in real_m1.m1.GROUP_ORDER
                    if candidate != group
                )
                ridge_lambda = real_m1.m1.LAMBDAS_DESCENDING[ordinal - 1]
                real_m1._write_fit_checkpoint(
                    checkpoint_root / f"zero__outer_refit__{group}.npz",
                    kind="zero",
                    key=f"outer_refit__{group}",
                    results={ridge_lambda: _optimizer(ridge_lambda, ordinal)},
                    training_cells=len(indexes),
                    training_groups=training_groups,
                )
        finally:
            if previous is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = previous
        yield checkpoint_root, cells, moments


def _replace_checkpoint(path: Path, **updates: Any) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload.update(updates)
    np.savez(path, **payload)


class OuterImportTests(unittest.TestCase):
    def test_imports_exact_twelve_offsets_when_consumer_job_differs(self) -> None:
        with _producer_fixture() as (root, cells, moments):
            previous = os.environ.get("SLURM_JOB_ID")
            os.environ["SLURM_JOB_ID"] = "99999999"
            try:
                imported = outer_import.import_outer_m1_offsets(
                    root,
                    cells,
                    moments,
                    mode=executor.SYNTHETIC_TEST_MODE,
                    expected_rank=104,
                )
            finally:
                if previous is None:
                    os.environ.pop("SLURM_JOB_ID", None)
                else:
                    os.environ["SLURM_JOB_ID"] = previous
        self.assertEqual(
            imported.producer_job_id, outer_import.PRODUCER_SLURM_JOB_ID
        )
        self.assertEqual(tuple(imported.offsets), real_m1.m1.GROUP_ORDER)
        self.assertEqual(tuple(imported.selected_lambdas), real_m1.m1.GROUP_ORDER)
        self.assertEqual(len(imported.cell_inventory), len(cells))
        self.assertEqual(imported.expected_rank, 104)
        fit_identities = set()
        for ordinal, group in enumerate(real_m1.m1.GROUP_ORDER, start=1):
            offset = imported.offsets[group]
            expected_indexes = tuple(
                index for index, cell in enumerate(cells) if cell.group != group
            )
            self.assertIsInstance(offset, executor.FixedM1Offset)
            self.assertIsInstance(offset.design_transform, executor.m1.DesignTransform)
            self.assertEqual(offset.held_out_group, group)
            self.assertEqual(offset.training_indexes, expected_indexes)
            self.assertTrue(offset.fit_identity.endswith(f"outer_refit__{group}"))
            self.assertIn(outer_import.PRODUCER_SLURM_JOB_ID, offset.fit_identity)
            self.assertFalse(offset.theta.flags.writeable)
            self.assertFalse(offset.design_transform.soma_mean.flags.writeable)
            self.assertFalse(offset.design_transform.soma_scale.flags.writeable)
            self.assertFalse(offset.geometry_scales.flags.writeable)
            self.assertTrue(
                np.array_equal(
                    offset.theta,
                    np.full(real_m1.m1.PARAMETER_DIMENSION, 0.001 * ordinal),
                )
            )
            fit_identities.add(offset.fit_identity)
        self.assertEqual(len(fit_identities), outer_import.EXPECTED_OUTER_FOLDS)
        with self.assertRaises(TypeError):
            imported.offsets[real_m1.m1.GROUP_ORDER[0]] = imported.offsets[  # type: ignore[index]
                real_m1.m1.GROUP_ORDER[0]
            ]
        with self.assertRaises(ValueError):
            imported.offsets[real_m1.m1.GROUP_ORDER[0]].theta[0] = 1.0

    def test_rejects_wrong_producer_group_inventory_and_extra_content(self) -> None:
        mutations = (
            (
                "producer_slurm_job_id",
                {"producer_slurm_job_id": np.asarray("123456")},
                "identity",
            ),
            ("key", {"key": np.asarray("outer_refit__forged")}, "identity"),
            (
                "training_groups",
                {"training_groups": np.asarray(("forged",), dtype=np.str_)},
                "identity",
            ),
            ("extra", {"unneeded_extra": np.asarray(1)}, "fields"),
        )
        for label, update, message in mutations:
            with self.subTest(label=label), _producer_fixture() as (
                root,
                cells,
                moments,
            ):
                group = real_m1.m1.GROUP_ORDER[0]
                path = root / f"zero__outer_refit__{group}.npz"
                _replace_checkpoint(path, **update)
                with self.assertRaisesRegex(
                    outer_import.M1OuterImportError, message
                ):
                    outer_import.import_outer_m1_offsets(
                        root,
                        cells,
                        moments,
                        mode=executor.SYNTHETIC_TEST_MODE,
                    )

    def test_reuses_frozen_convergence_and_parameter_shape_checks(self) -> None:
        mutations = (
            (
                "theta_dimension",
                {"theta": np.zeros((1, real_m1.m1.PARAMETER_DIMENSION - 1))},
                "arrays",
            ),
            (
                "failed_witness",
                {
                    "telemetry_json": np.asarray(
                        (
                            '{"status":"pass","ridge_lambda":1000.0,'
                            '"convergence_witness":"forged"}',
                        ),
                        dtype=np.str_,
                    )
                },
                "convergence witness",
            ),
            (
                "multiple_lambdas",
                {"lambdas": np.asarray((1000.0, 100.0), dtype=float)},
                "one frozen-grid lambda",
            ),
        )
        for label, update, message in mutations:
            with self.subTest(label=label), _producer_fixture() as (
                root,
                cells,
                moments,
            ):
                group = real_m1.m1.GROUP_ORDER[0]
                path = root / f"zero__outer_refit__{group}.npz"
                _replace_checkpoint(path, **update)
                with self.assertRaisesRegex(
                    outer_import.M1OuterImportError, message
                ):
                    outer_import.import_outer_m1_offsets(
                        root,
                        cells,
                        moments,
                        mode=executor.SYNTHETIC_TEST_MODE,
                    )

    def test_rejects_symlinked_checkpoint_and_wrong_producer_root(self) -> None:
        with _producer_fixture() as (root, cells, moments):
            first, second = real_m1.m1.GROUP_ORDER[:2]
            first_path = root / f"zero__outer_refit__{first}.npz"
            first_path.unlink()
            first_path.symlink_to(root / f"zero__outer_refit__{second}.npz")
            with self.assertRaisesRegex(
                outer_import.M1OuterImportError, "absent or symlinked"
            ):
                outer_import.import_outer_m1_offsets(
                    root,
                    cells,
                    moments,
                    mode=executor.SYNTHETIC_TEST_MODE,
                )

        with _producer_fixture() as (root, cells, moments):
            wrong = root.parent.parent / "v3_m1_real_fit_single_thread_v3_job_1"
            wrong.mkdir()
            wrong_root = wrong / outer_import.CHECKPOINT_DIRECTORY
            wrong_root.mkdir()
            with self.assertRaisesRegex(
                outer_import.M1OuterImportError, "exact producer checkpoint root"
            ):
                outer_import.import_outer_m1_offsets(
                    wrong_root,
                    cells,
                    moments,
                    mode=executor.SYNTHETIC_TEST_MODE,
                )

    def test_has_no_full_refit_or_telemetry_path(self) -> None:
        with _producer_fixture() as (root, cells, moments):
            full_refit = root / "zero__full_refit.npz"
            full_refit.write_bytes(b"not opened")
            telemetry = root.parent / real_m1.TELEMETRY_FILENAME
            telemetry.write_bytes(b"not opened")
            real_open = os.open

            def reject_forbidden(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                supplied = Path(os.fspath(path))
                if supplied in {full_refit, telemetry}:
                    raise AssertionError("forbidden producer artifact was opened")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(outer_import.os, "open", side_effect=reject_forbidden):
                imported = outer_import.import_outer_m1_offsets(
                    root,
                    cells,
                    moments,
                    mode=executor.SYNTHETIC_TEST_MODE,
                )
            self.assertEqual(len(imported.offsets), 12)
            for forbidden_path in (full_refit, telemetry):
                with self.assertRaisesRegex(
                    outer_import.M1OuterImportError,
                    "exact producer checkpoint root",
                ):
                    outer_import.import_outer_m1_offsets(
                        forbidden_path,
                        cells,
                        moments,
                        mode=executor.SYNTHETIC_TEST_MODE,
                    )

    def test_rejects_invalid_inventory_and_rank_before_checkpoint_access(self) -> None:
        cells, moments = _cells_and_moments()
        with self.assertRaisesRegex(
            outer_import.M1OuterImportError, "rank 104"
        ), mock.patch.object(
            outer_import, "_require_checkpoint_root", side_effect=AssertionError
        ):
            outer_import.import_outer_m1_offsets(
                "/not/inspected",
                cells,
                moments,
                mode=executor.SEALED_PRODUCTION_MODE,
                execution_inventory=None,
                expected_rank=103,
            )

        duplicated = list(cells)
        duplicated[1] = executor.m1.Cell(
            identifier=duplicated[0].identifier,
            group=duplicated[1].group,
            layer=duplicated[1].layer,
            hemisphere=duplicated[1].hemisphere,
            soma_um=duplicated[1].soma_um,
            soma_distance_mm=duplicated[1].soma_distance_mm,
            k=duplicated[1].k,
            observed_index=duplicated[1].observed_index,
            segment_unmapped=duplicated[1].segment_unmapped,
        )
        with self.assertRaisesRegex(
            outer_import.M1OuterImportError, "inventory or rank"
        ), mock.patch.object(
            outer_import, "_require_checkpoint_root", side_effect=AssertionError
        ):
            outer_import.import_outer_m1_offsets(
                "/not/inspected",
                duplicated,
                moments,
                mode=executor.SYNTHETIC_TEST_MODE,
                expected_rank=104,
            )

        with self.assertRaisesRegex(
            outer_import.M1OuterImportError, "inventory or rank"
        ), mock.patch.object(
            outer_import, "_require_checkpoint_root", side_effect=AssertionError
        ):
            outer_import.import_outer_m1_offsets(
                "/not/inspected",
                cells,
                moments,
                mode=executor.SEALED_PRODUCTION_MODE,
                execution_inventory=object(),  # type: ignore[arg-type]
                expected_rank=104,
            )

    def test_expected_checkpoint_root_is_pure_and_exact(self) -> None:
        observed = outer_import.expected_checkpoint_root("/scratch/example")
        self.assertEqual(
            observed,
            Path("/scratch/example")
            / "br_autoresearch"
            / "episode10_single_cell_coprojection"
            / outer_import.PRODUCER_ROOT_BASENAME
            / outer_import.CHECKPOINT_DIRECTORY,
        )

    def test_production_root_is_canonical_and_has_no_symlinked_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            scratch = Path(raw) / "scratch"
            expected = outer_import.expected_checkpoint_root(scratch)
            expected.mkdir(parents=True)
            decoy = (
                Path(raw)
                / "decoy"
                / outer_import.PRODUCER_ROOT_BASENAME
                / outer_import.CHECKPOINT_DIRECTORY
            )
            decoy.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"SCRATCH": str(scratch)}):
                self.assertEqual(
                    outer_import._require_checkpoint_root(
                        expected,
                        mode=executor.SEALED_PRODUCTION_MODE,
                    ),
                    expected,
                )
                with self.assertRaisesRegex(
                    outer_import.M1OuterImportError, "canonical Scratch path"
                ):
                    outer_import._require_checkpoint_root(
                        decoy,
                        mode=executor.SEALED_PRODUCTION_MODE,
                    )

            real_parent = Path(raw) / "real_parent"
            synthetic_root = (
                real_parent
                / outer_import.PRODUCER_ROOT_BASENAME
                / outer_import.CHECKPOINT_DIRECTORY
            )
            synthetic_root.mkdir(parents=True)
            alias = Path(raw) / "alias_parent"
            alias.symlink_to(real_parent, target_is_directory=True)
            alias_root = (
                alias
                / outer_import.PRODUCER_ROOT_BASENAME
                / outer_import.CHECKPOINT_DIRECTORY
            )
            with self.assertRaisesRegex(
                outer_import.M1OuterImportError, "symlinked component"
            ):
                outer_import._require_checkpoint_root(
                    alias_root,
                    mode=executor.SYNTHETIC_TEST_MODE,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
