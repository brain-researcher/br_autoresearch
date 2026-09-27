"""Lossless authenticated artifacts for EP12 fitted-null runtime state.

Numeric state is stored as ``allow_pickle=False`` NPY payloads. Every file,
metadata object, reconstructed design, and fitted generator is authenticated
before use. There is deliberately no connectivity loader in this module.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import secrets
from typing import Any, Mapping

import numpy as np

from .fitted_null import (
    FittedNullDesign,
    fitted_generator_manifest,
    fitted_generator_sha256,
)
from .policy import digest_bytes, digest_object
from .scientific_models import FittedPredictiveModel


STATE_SCHEMA = "ep12.fitted_null_model_state.v1"
DESIGN_SCHEMA = "ep12.fitted_null_design_artifact.v1"
MODEL_SCHEMA = "ep12.fitted_predictive_model_artifact.v1"
ATOMIC_TEMP_DIRECTORY = ".ep12_atomic_tmp"

MODEL_ARRAYS = (
    "basis",
    "nuisance_coefficients",
    "latent_offsets",
    "log_weights",
    "prior_means",
    "prior_covariances",
    "prior_weights",
)


class ModelArtifactError(RuntimeError):
    """A serialized fitted model/design is incomplete or has changed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ModelArtifactError("artifact metadata is not canonical JSON") from error


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value).decode("utf-8"))


def _sha256(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ModelArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Publish one fsync'd file by rename, refusing an existing target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ModelArtifactError(f"refusing to overwrite artifact: {path}")
    temporary_root = path.parent / ATOMIC_TEMP_DIRECTORY
    temporary_root.mkdir(mode=0o770, exist_ok=True)
    if temporary_root.is_symlink() or not temporary_root.is_dir():
        raise ModelArtifactError("artifact temporary namespace is unsafe")
    namespace = digest_bytes(
        "\0".join((
            os.uname().nodename,
            os.environ.get("SLURM_JOB_ID", "no-job"),
            os.environ.get("SLURM_ARRAY_TASK_ID", "no-task"),
            str(os.getpid()),
        )).encode("utf-8")
    )[:16]
    basename = digest_bytes(path.name.encode("utf-8"))[:16]
    descriptor: int | None = None
    temporary: Path | None = None
    for _ in range(8):
        candidate = temporary_root / (
            f"{basename}.{namespace}.{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(
                str(candidate), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660
            )
        except FileExistsError:
            continue
        temporary = candidate
        break
    if descriptor is None or temporary is None:
        raise ModelArtifactError(f"cannot allocate artifact staging file for {path}")
    try:
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ModelArtifactError(f"short write while creating {path}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            if temporary is not None:
                temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def atomic_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    _atomic_bytes(path, buffer.getvalue())


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ModelArtifactError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ModelArtifactError(f"malformed {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise ModelArtifactError(f"{label} must be a JSON object")
    return value


def _load_npy(path: Path, *, dtype: np.dtype[Any], label: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ModelArtifactError(f"cannot load {label}: {path}") from error
    if value.dtype != np.dtype(dtype):
        raise ModelArtifactError(
            f"{label} dtype changed: expected {np.dtype(dtype)}, got {value.dtype}"
        )
    return np.ascontiguousarray(value)


def _record_files(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {
        name: digest_bytes((directory / name).read_bytes())
        for name in names
    }


def _verify_files(
    directory: Path, files: Mapping[str, Any], expected: set[str]
) -> None:
    if set(map(str, files)) != expected:
        raise ModelArtifactError("artifact file inventory differs from schema")
    for relative, expected_hash in files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ModelArtifactError("artifact manifest contains an unsafe path")
        path = directory / relative_path
        if not path.is_file() or path.is_symlink():
            raise ModelArtifactError(f"artifact file is missing or unsafe: {relative}")
        if digest_bytes(path.read_bytes()) != _sha256(expected_hash, str(relative)):
            raise ModelArtifactError(f"artifact file digest mismatch: {relative}")


def _write_design(directory: Path, design: FittedNullDesign) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        "covariates.npy": np.asarray(design.covariates, dtype=np.float64),
        "raw_counts.npy": np.asarray(design.raw_counts, dtype=np.int64),
        "raw_to_collapsed.npy": np.asarray(design.raw_to_collapsed, dtype=np.int64),
    }
    for name, value in arrays.items():
        atomic_npy(directory / name, value)
    core = {
        "schema_version": DESIGN_SCHEMA,
        "design_sha256": design.design_sha256,
        "neuron_ids": list(design.neuron_ids),
        "focal_types": list(design.focal_types),
        "sides": list(design.sides),
        "raw_vocabulary": list(design.raw_vocabulary),
        "collapsed_vocabulary": list(design.collapsed_vocabulary),
        "collapsed_typed_bins": [
            name
            for name, typed in zip(
                design.collapsed_vocabulary, design.collapsed_typed_mask
            )
            if bool(typed)
        ],
        "files": _record_files(directory, tuple(arrays)),
    }
    metadata = {**core, "artifact_sha256": digest_object(core)}
    atomic_json(directory / "design.json", metadata)
    return metadata


def _read_design(directory: Path) -> FittedNullDesign:
    metadata = _read_json(directory / "design.json", "design metadata")
    required = {
        "schema_version",
        "design_sha256",
        "neuron_ids",
        "focal_types",
        "sides",
        "raw_vocabulary",
        "collapsed_vocabulary",
        "collapsed_typed_bins",
        "files",
        "artifact_sha256",
    }
    if set(metadata) != required or metadata.get("schema_version") != DESIGN_SCHEMA:
        raise ModelArtifactError("design metadata schema mismatch")
    core = {key: value for key, value in metadata.items() if key != "artifact_sha256"}
    if metadata["artifact_sha256"] != digest_object(core):
        raise ModelArtifactError("design metadata digest mismatch")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ModelArtifactError("design file inventory is missing")
    _verify_files(
        directory,
        files,
        {"covariates.npy", "raw_counts.npy", "raw_to_collapsed.npy"},
    )
    design = FittedNullDesign.create(
        neuron_ids=list(metadata["neuron_ids"]),
        focal_types=list(metadata["focal_types"]),
        sides=list(metadata["sides"]),
        covariates=_load_npy(
            directory / "covariates.npy", dtype=np.float64, label="design covariates"
        ),
        raw_counts=_load_npy(
            directory / "raw_counts.npy", dtype=np.int64, label="design raw counts"
        ),
        raw_vocabulary=list(metadata["raw_vocabulary"]),
        collapsed_vocabulary=list(metadata["collapsed_vocabulary"]),
        raw_to_collapsed=_load_npy(
            directory / "raw_to_collapsed.npy",
            dtype=np.int64,
            label="raw-to-collapsed mapping",
        ),
        collapsed_typed_bins=list(metadata["collapsed_typed_bins"]),
    )
    if design.design_sha256 != _sha256(metadata["design_sha256"], "design"):
        raise ModelArtifactError("reconstructed design digest mismatch")
    return design


def _write_model(
    directory: Path,
    *,
    provider_type: str,
    side: str,
    model: FittedPredictiveModel,
    design: FittedNullDesign,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        f"{name}.npy": np.asarray(getattr(model, name), dtype=np.float64)
        for name in MODEL_ARRAYS
    }
    for name, value in arrays.items():
        atomic_npy(directory / name, value)
    core = {
        "schema_version": MODEL_SCHEMA,
        "provider_type": provider_type,
        "side": side,
        "model_id": model.model_id,
        "design_sha256": design.design_sha256,
        "generator_sha256": fitted_generator_sha256(model, design),
        "integration_draws": int(model.integration_draws),
        "integration_seed": int(model.integration_seed),
        "parameter_count": int(model.parameter_count),
        "pseudocount": float(model.pseudocount),
        "diagnostics": _json_copy(model.diagnostics),
        "files": _record_files(directory, tuple(arrays)),
    }
    metadata = {**core, "artifact_sha256": digest_object(core)}
    atomic_json(directory / "model.json", metadata)
    return metadata


def _read_model(
    directory: Path,
    *,
    provider_type: str,
    side: str,
    design: FittedNullDesign,
) -> FittedPredictiveModel:
    metadata = _read_json(directory / "model.json", "model metadata")
    required = {
        "schema_version",
        "provider_type",
        "side",
        "model_id",
        "design_sha256",
        "generator_sha256",
        "integration_draws",
        "integration_seed",
        "parameter_count",
        "pseudocount",
        "diagnostics",
        "files",
        "artifact_sha256",
    }
    if set(metadata) != required or metadata.get("schema_version") != MODEL_SCHEMA:
        raise ModelArtifactError("model metadata schema mismatch")
    core = {key: value for key, value in metadata.items() if key != "artifact_sha256"}
    if metadata["artifact_sha256"] != digest_object(core):
        raise ModelArtifactError("model metadata digest mismatch")
    if (
        metadata["provider_type"] != provider_type
        or metadata["side"] != side
        or metadata["design_sha256"] != design.design_sha256
    ):
        raise ModelArtifactError("model identity differs from its manifest row")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ModelArtifactError("model file inventory is missing")
    _verify_files(directory, files, {f"{name}.npy" for name in MODEL_ARRAYS})
    arrays = {
        name: _load_npy(
            directory / f"{name}.npy", dtype=np.float64, label=f"model {name}"
        )
        for name in MODEL_ARRAYS
    }
    model = FittedPredictiveModel(
        model_id=str(metadata["model_id"]),
        basis=arrays["basis"],
        nuisance_coefficients=arrays["nuisance_coefficients"],
        latent_offsets=arrays["latent_offsets"],
        log_weights=arrays["log_weights"],
        prior_means=arrays["prior_means"],
        prior_covariances=arrays["prior_covariances"],
        prior_weights=arrays["prior_weights"],
        integration_draws=int(metadata["integration_draws"]),
        integration_seed=int(metadata["integration_seed"]),
        parameter_count=int(metadata["parameter_count"]),
        diagnostics=_json_copy(metadata["diagnostics"]),
        pseudocount=float(metadata["pseudocount"]),
    )
    observed = fitted_generator_sha256(model, design)
    if observed != _sha256(metadata["generator_sha256"], "generator"):
        raise ModelArtifactError("reconstructed generator digest mismatch")
    return model


def write_model_state(
    directory: Path,
    *,
    design: FittedNullDesign,
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
) -> dict[str, Any]:
    """Write a complete design/generator state into a new directory."""

    directory = directory.resolve()
    if directory.exists():
        raise ModelArtifactError(f"model-state directory already exists: {directory}")
    directory.mkdir(parents=True)
    design_metadata = _write_design(directory / "design", design)
    generator_rows, generator_manifest_sha256 = fitted_generator_manifest(
        generators, design
    )
    serialized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(generator_rows):
        provider_type = str(row["focal_type"])
        side = str(row["side"])
        relative = f"generators/generator_{index:04d}"
        metadata = _write_model(
            directory / relative,
            provider_type=provider_type,
            side=side,
            model=generators[(provider_type, side)],
            design=design,
        )
        serialized_rows.append(
            {
                **dict(row),
                "artifact_directory": relative,
                "artifact_sha256": metadata["artifact_sha256"],
            }
        )
    core = {
        "schema_version": STATE_SCHEMA,
        "design_sha256": design.design_sha256,
        "design_artifact_sha256": design_metadata["artifact_sha256"],
        "generator_manifest_sha256": generator_manifest_sha256,
        "generator_count": len(serialized_rows),
        "generators": serialized_rows,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    manifest = {**core, "state_sha256": digest_object(core)}
    atomic_json(directory / "model_state.json", manifest)
    return manifest


def load_model_state(
    directory: Path,
) -> tuple[
    FittedNullDesign,
    dict[tuple[str, str], FittedPredictiveModel],
    dict[str, Any],
]:
    """Authenticate and reconstruct a complete fitted-null model state."""

    directory = directory.resolve()
    manifest = _read_json(directory / "model_state.json", "model-state manifest")
    required = {
        "schema_version",
        "design_sha256",
        "design_artifact_sha256",
        "generator_manifest_sha256",
        "generator_count",
        "generators",
        "development_only",
        "final_connectivity_accessed",
        "final_connectivity_access_authorized",
        "state_sha256",
    }
    if set(manifest) != required or manifest.get("schema_version") != STATE_SCHEMA:
        raise ModelArtifactError("model-state manifest schema mismatch")
    core = {key: value for key, value in manifest.items() if key != "state_sha256"}
    if manifest["state_sha256"] != digest_object(core):
        raise ModelArtifactError("model-state manifest digest mismatch")
    if not (
        manifest["development_only"] is True
        and manifest["final_connectivity_accessed"] is False
        and manifest["final_connectivity_access_authorized"] is False
    ):
        raise ModelArtifactError("model state crosses the final-connectivity boundary")
    design = _read_design(directory / "design")
    design_metadata = _read_json(
        directory / "design" / "design.json", "design metadata"
    )
    if (
        design.design_sha256 != manifest["design_sha256"]
        or design_metadata["artifact_sha256"]
        != manifest["design_artifact_sha256"]
    ):
        raise ModelArtifactError("model-state design identity mismatch")
    rows = manifest.get("generators")
    if not isinstance(rows, list) or len(rows) != int(manifest["generator_count"]):
        raise ModelArtifactError("model-state generator table is incomplete")
    generators: dict[tuple[str, str], FittedPredictiveModel] = {}
    expected_row_fields = {
        "focal_type",
        "side",
        "model_id",
        "generator_sha256",
        "artifact_directory",
        "artifact_sha256",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ModelArtifactError("model-state generator row is malformed")
        provider_type = str(row["focal_type"])
        side = str(row["side"])
        key = (provider_type, side)
        if key in generators or side not in {"L", "R"}:
            raise ModelArtifactError("model-state generator key is duplicated or invalid")
        relative = Path(str(row["artifact_directory"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelArtifactError("generator artifact directory is unsafe")
        model = _read_model(
            directory / relative,
            provider_type=provider_type,
            side=side,
            design=design,
        )
        metadata = _read_json(
            directory / relative / "model.json", "model metadata"
        )
        if (
            model.model_id != row["model_id"]
            or fitted_generator_sha256(model, design) != row["generator_sha256"]
            or metadata["artifact_sha256"] != row["artifact_sha256"]
        ):
            raise ModelArtifactError("generator row differs from serialized model")
        generators[key] = model
    reconstructed_rows, reconstructed_sha = fitted_generator_manifest(
        generators, design
    )
    stripped = [
        {
            key: row[key]
            for key in (
                "focal_type",
                "side",
                "model_id",
                "generator_sha256",
            )
        }
        for row in rows
    ]
    if list(reconstructed_rows) != stripped:
        raise ModelArtifactError("generator manifest rows changed during reconstruction")
    if reconstructed_sha != _sha256(
        manifest["generator_manifest_sha256"], "generator manifest"
    ):
        raise ModelArtifactError("generator manifest digest mismatch")
    return design, generators, _json_copy(manifest)
