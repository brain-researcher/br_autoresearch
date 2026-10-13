#!/usr/bin/env python3
"""Read the successful EP10 v3 M1 outer refits for a later M2 job.

The producer deliberately scoped checkpoint *resume* to its own Slurm job.
M2 needs a narrower operation: read the twelve completed outer-LOGO refits
from that producer without pretending to resume it.  This module therefore
uses the checkpoint validation rules from :mod:`fit_v3_m1_real_v1`, while
binding ``producer_slurm_job_id`` to the completed producer rather than the
current consumer job.

The importer has no CLI and performs no writes.  It accepts cells and uniform
moments that an authorized caller has already loaded, reconstructs only their
training-fold transforms, and opens only
``zero__outer_refit__<held-out-group>.npz``.  It has no route to the producer's
full refit, solver-telemetry artifact, predictions, membership inputs, M0,
final/unassigned data, or morphology.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1_real_v1 as real_m1  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402


PRODUCER_SLURM_JOB_ID = "45739355"
PRODUCER_ROOT_BASENAME = (
    f"{real_m1.OUTPUT_PREFIX}_job_{PRODUCER_SLURM_JOB_ID}"
)
CHECKPOINT_DIRECTORY = real_m1.CHECKPOINT_DIRECTORY
EXPECTED_OUTER_FOLDS = 12

_CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "key",
        "actual_k_job_id",
        "qualification_job_id",
        "prior_membership_opening_job_id",
        "numerical_execution_schema_version",
        "producer_slurm_job_id",
        "training_cells",
        "training_groups",
        "lambdas",
        "theta",
        "objective",
        "gradient",
        "iterations",
        "accepted_steps",
        "relative_change",
        "telemetry_json",
    }
)


class M1OuterImportError(real_m1.RealM1Error):
    """A producer path, checkpoint, fold binding, or scope check failed."""


@dataclass(frozen=True)
class ImportedOuterM1:
    """The twelve immutable, fold-bound M1 offsets needed by real M2."""

    producer_job_id: str
    selected_lambdas: Mapping[str, float]
    offsets: Mapping[str, executor.FixedM1Offset]
    cell_inventory: tuple[tuple[str, str], ...]
    mode: str
    expected_rank: int | None


def expected_checkpoint_root(scratch_root: str | os.PathLike[str]) -> Path:
    """Return the producer checkpoint path without inspecting the filesystem."""

    return (
        Path(os.path.abspath(os.fspath(scratch_root)))
        / "br_autoresearch"
        / "episode10_single_cell_coprojection"
        / PRODUCER_ROOT_BASENAME
        / CHECKPOINT_DIRECTORY
    )


def _require_checkpoint_root(
    path: str | os.PathLike[str],
    *,
    mode: str,
) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    if mode == executor.SEALED_PRODUCTION_MODE:
        scratch = os.environ.get("SCRATCH")
        if not scratch or root != expected_checkpoint_root(scratch):
            raise M1OuterImportError(
                "production M1 outer-refit source is not the canonical Scratch path"
            )
    if (
        root.name != CHECKPOINT_DIRECTORY
        or root.parent.name != PRODUCER_ROOT_BASENAME
        or root.is_symlink()
        or root.parent.is_symlink()
    ):
        raise M1OuterImportError(
            "M1 outer-refit source is not the exact producer checkpoint root"
        )
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise M1OuterImportError(
            "M1 outer-refit producer checkpoint root is unavailable or symlinked"
        ) from error
    if resolved != root:
        raise M1OuterImportError(
            "M1 outer-refit producer checkpoint root contains a symlinked component"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise M1OuterImportError(
            "M1 outer-refit producer checkpoint root is unavailable or symlinked"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise M1OuterImportError(
                "M1 outer-refit producer checkpoint root is not a directory"
            )
    finally:
        os.close(descriptor)
    return root


def _checkpoint_path(root: Path, group: str) -> Path:
    if group not in real_m1.m1.GROUP_ORDER:
        raise M1OuterImportError("M1 outer-refit group is outside the frozen order")
    key = f"outer_refit__{group}"
    path = root / f"zero__{key}.npz"
    if path.parent != root:
        raise M1OuterImportError("M1 outer-refit checkpoint path escaped its root")
    return path


def _read_checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read one no-follow regular NPZ into detached in-memory arrays."""

    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise M1OuterImportError(
            "M1 outer-refit checkpoint is absent or symlinked"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise M1OuterImportError("M1 outer-refit checkpoint is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            with np.load(handle, allow_pickle=False) as archive:
                if set(archive.files) != _CHECKPOINT_FIELDS:
                    raise M1OuterImportError("M1 outer-refit checkpoint fields changed")
                return {name: np.array(archive[name], copy=True) for name in archive.files}
    except M1OuterImportError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise M1OuterImportError("M1 outer-refit checkpoint is unreadable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _scalar_text(array: np.ndarray, label: str) -> str:
    if array.shape != ():
        raise M1OuterImportError(f"M1 outer-refit {label} is not scalar")
    try:
        return str(array.item())
    except ValueError as error:
        raise M1OuterImportError(f"M1 outer-refit {label} is malformed") from error


def _scalar_integer(array: np.ndarray, label: str) -> int:
    if array.shape != () or array.dtype.kind not in "iu":
        raise M1OuterImportError(f"M1 outer-refit {label} is not an integer scalar")
    value = array.item()
    if isinstance(value, (bool, np.bool_)):
        raise M1OuterImportError(f"M1 outer-refit {label} is not an integer scalar")
    return int(value)


def _load_outer_checkpoint(
    path: Path,
    *,
    group: str,
    training_cells: int,
    training_groups: tuple[str, ...],
) -> tuple[float, np.ndarray]:
    """Load one outer refit with the producer runner's numerical semantics."""

    key = f"outer_refit__{group}"
    data = _read_checkpoint_arrays(path)
    lambdas = np.asarray(data["lambdas"], dtype=float)
    if (
        lambdas.shape != (1,)
        or not np.all(np.isfinite(lambdas))
        or float(lambdas[0]) not in real_m1.m1.LAMBDAS_DESCENDING
    ):
        raise M1OuterImportError(
            "M1 outer-refit checkpoint does not contain one frozen-grid lambda"
        )
    ridge_lambda = float(lambdas[0])
    observed_training_groups = np.asarray(data["training_groups"])
    try:
        checkpoint_groups = tuple(
            str(value) for value in observed_training_groups.tolist()
        )
    except (TypeError, ValueError) as error:
        raise M1OuterImportError(
            "M1 outer-refit training-group inventory is malformed"
        ) from error
    if not (
        _scalar_text(data["schema"], "schema") == real_m1.CHECKPOINT_SCHEMA
        and _scalar_text(data["kind"], "kind") == "zero"
        and _scalar_text(data["key"], "key") == key
        and _scalar_text(data["actual_k_job_id"], "actual-K job")
        == real_m1.ACTUAL_K_JOB_ID
        and _scalar_text(data["qualification_job_id"], "qualification job")
        == real_m1.QUALIFICATION_JOB_ID
        and _scalar_text(
            data["prior_membership_opening_job_id"], "membership-opening job"
        )
        == real_m1.PRIOR_MEMBERSHIP_OPENING_JOB_ID
        and _scalar_text(
            data["numerical_execution_schema_version"], "execution policy"
        )
        == real_m1.EXECUTION_SCHEMA_VERSION
        and _scalar_text(data["producer_slurm_job_id"], "producer Slurm job")
        == PRODUCER_SLURM_JOB_ID
        and _scalar_integer(data["training_cells"], "training-cell count")
        == training_cells
        and observed_training_groups.ndim == 1
        and checkpoint_groups == training_groups
    ):
        raise M1OuterImportError("M1 outer-refit checkpoint identity changed")

    theta = np.asarray(data["theta"], dtype=float)
    objective = np.asarray(data["objective"], dtype=float)
    gradient = np.asarray(data["gradient"], dtype=float)
    iterations = np.asarray(data["iterations"])
    accepted = np.asarray(data["accepted_steps"])
    relative = np.asarray(data["relative_change"], dtype=float)
    telemetry_values = np.asarray(data["telemetry_json"])
    if not (
        theta.shape == (1, real_m1.m1.PARAMETER_DIMENSION)
        and objective.shape == gradient.shape == relative.shape == (1,)
        and iterations.shape == accepted.shape == (1,)
        and iterations.dtype.kind in "iu"
        and accepted.dtype.kind in "iu"
        and telemetry_values.shape == (1,)
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(objective))
        and np.all(np.isfinite(gradient))
    ):
        raise M1OuterImportError("M1 outer-refit checkpoint arrays are malformed")
    try:
        telemetry = json.loads(str(telemetry_values[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise M1OuterImportError(
            "M1 outer-refit optimizer telemetry is unreadable"
        ) from error
    if not isinstance(telemetry, Mapping):
        raise M1OuterImportError(
            "M1 outer-refit optimizer telemetry is not a mapping"
        )
    result = real_m1.m1.OptimizerResult(
        theta=theta[0].copy(),
        objective=float(objective[0]),
        gradient_infinity_norm=float(gradient[0]),
        iterations=int(iterations[0]),
        accepted_steps=int(accepted[0]),
        relative_objective_change=float(relative[0]),
        telemetry=telemetry,
    )
    try:
        real_m1._validate_optimizer_result(result, ridge_lambda)
    except (real_m1.RealM1Error, TypeError, ValueError) as error:
        raise M1OuterImportError(
            "M1 outer-refit lacks the frozen convergence witness"
        ) from error
    imported_theta = np.asarray(result.theta, dtype=float).copy()
    imported_theta.setflags(write=False)
    return ridge_lambda, imported_theta


def _readonly_vector(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise M1OuterImportError(f"{label} is malformed or nonfinite")
    result = array.copy()
    result.setflags(write=False)
    return result


def import_outer_m1_offsets(
    checkpoint_root: str | os.PathLike[str],
    cells: Sequence[Any],
    moments: Sequence[Any],
    *,
    mode: str = executor.SEALED_PRODUCTION_MODE,
    execution_inventory: executor.FrozenExecutionInventory | None = None,
    expected_rank: int | None = executor.EXPECTED_PRODUCTION_RANK,
) -> ImportedOuterM1:
    """Import and bind the twelve completed outer-LOGO M1 offsets.

    Production callers must supply the same authenticated target-blind
    ``FrozenExecutionInventory`` and expected residual rank that the M2
    executor requires.  ``synthetic_test`` mode exists only for target-free
    contract tests and never accepts a production inventory object.
    """

    rows = tuple(cells)
    uniform_moments = tuple(moments)
    if len(rows) != len(uniform_moments):
        raise M1OuterImportError("M1 import cells and moments are not aligned")
    if mode == executor.SEALED_PRODUCTION_MODE and (
        isinstance(expected_rank, bool)
        or not isinstance(expected_rank, int)
        or expected_rank != executor.EXPECTED_PRODUCTION_RANK
    ):
        raise M1OuterImportError(
            "production M1 import requires expected residual rank 104"
        )
    try:
        cell_inventory = executor._validate_execution_scope(
            rows, mode, execution_inventory, expected_rank
        )
        specs = executor.logo_fold_specs(rows)
    except executor.CoreTrialExecutorError as error:
        raise M1OuterImportError("M1 import execution inventory or rank is invalid") from error
    if (
        len(specs) != EXPECTED_OUTER_FOLDS
        or tuple(spec.held_out_group for spec in specs) != real_m1.m1.GROUP_ORDER
    ):
        raise M1OuterImportError("M1 import did not produce the exact 12 LOGO folds")
    root = _require_checkpoint_root(checkpoint_root, mode=mode)

    offsets: dict[str, executor.FixedM1Offset] = {}
    selected_lambdas: dict[str, float] = {}
    for spec in specs:
        group = spec.held_out_group
        expected_indexes = tuple(
            index for index, cell in enumerate(rows) if str(cell.group) != group
        )
        if spec.training_indexes != expected_indexes:
            raise M1OuterImportError("M1 outer-refit LOGO training indexes changed")
        training_groups = tuple(
            candidate for candidate in real_m1.m1.GROUP_ORDER if candidate != group
        )
        try:
            prepared = real_m1._prepare_training(
                rows, uniform_moments, spec.training_indexes
            )
        except real_m1.m1.M1Error as error:
            raise M1OuterImportError(
                "M1 outer-refit training transform cannot be reconstructed"
            ) from error
        if (
            tuple(str(cell.identifier) for cell in prepared.cells)
            != tuple(str(rows[index].identifier) for index in expected_indexes)
            or tuple(
                candidate
                for candidate in real_m1.m1.GROUP_ORDER
                if any(str(cell.group) == candidate for cell in prepared.cells)
            )
            != training_groups
        ):
            raise M1OuterImportError(
                "M1 outer-refit reconstructed training inventory changed"
            )
        ridge_lambda, theta = _load_outer_checkpoint(
            _checkpoint_path(root, group),
            group=group,
            training_cells=len(expected_indexes),
            training_groups=training_groups,
        )
        soma_mean = _readonly_vector(
            prepared.design_transform.soma_mean,
            (9,),
            "M1 outer-refit soma mean",
        )
        soma_scale = _readonly_vector(
            prepared.design_transform.soma_scale,
            (9,),
            "M1 outer-refit soma scale",
        )
        geometry_scales = _readonly_vector(
            prepared.geometry_scales,
            (real_m1.m1.GEOMETRY_DIMENSION,),
            "M1 outer-refit geometry scales",
        )
        if np.any(soma_scale <= 0.0) or np.any(geometry_scales <= 0.0):
            raise M1OuterImportError(
                "M1 outer-refit reconstructed training scales are not positive"
            )
        offsets[group] = executor.FixedM1Offset(
            held_out_group=group,
            training_indexes=spec.training_indexes,
            theta=theta,
            design_transform=executor.m1.DesignTransform(soma_mean, soma_scale),
            geometry_scales=geometry_scales,
            fit_identity=(
                f"{PRODUCER_ROOT_BASENAME}__zero__outer_refit__{group}"
            ),
        )
        selected_lambdas[group] = ridge_lambda

    try:
        executor._validate_offset_inventory(offsets, specs)
    except executor.CoreTrialExecutorError as error:
        raise M1OuterImportError(
            "imported M1 outer-refit inventory is not exactly fold-bound"
        ) from error
    return ImportedOuterM1(
        producer_job_id=PRODUCER_SLURM_JOB_ID,
        selected_lambdas=MappingProxyType(dict(selected_lambdas)),
        offsets=MappingProxyType(dict(offsets)),
        cell_inventory=tuple(cell_inventory),
        mode=mode,
        expected_rank=expected_rank,
    )


__all__ = [
    "CHECKPOINT_DIRECTORY",
    "EXPECTED_OUTER_FOLDS",
    "ImportedOuterM1",
    "M1OuterImportError",
    "PRODUCER_ROOT_BASENAME",
    "PRODUCER_SLURM_JOB_ID",
    "expected_checkpoint_root",
    "import_outer_m1_offsets",
]
