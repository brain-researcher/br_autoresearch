#!/usr/bin/env python3
"""Outcome-blind synthetic replay for the EP12 post-36 contract transition.

This file is intentionally standalone: invoke it by file path, not with
``python -m ep12_executor``.  The scientific package imported for a replay is
selected exclusively by ``--source-executor-root``.  No command accepts a
connectivity, annotation, or other data path.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import platform
import stat
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_SCHEMA = "ep12.contract_transition_invariance_probe.v1"
COMPARE_SCHEMA = "ep12.contract_transition_invariance_comparison.v1"
SPEC_SCHEMA = "ep12.contract_transition_probe_spec.v1"
INPUT_SCHEMA = "ep12.contract_transition_probe_inputs.v1"
SCIENTIFIC_SCHEMA = "ep12.contract_transition_probe_scientific_payload.v1"
DIAGNOSTICS_SCHEMA = "ep12.contract_transition_probe_diagnostics.v1"

THIS_FILE = Path(__file__).absolute()
EPISODE_ROOT = THIS_FILE.parents[3]
EXECUTOR_ROOT = THIS_FILE.parents[1]
DEFAULT_SPEC = EXECUTOR_ROOT / "CONTRACT_TRANSITION_PROBE_SPEC.json"
EPISODE_OUTPUTS = EPISODE_ROOT / "outputs"
EPISODE_SCRATCH = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
)

EXPECTED_MODULES = (
    "ep12_executor/__init__.py",
    "ep12_executor/scientific_models.py",
    "ep12_executor/scientific_synthetic.py",
)
EXPECTED_REQUIRED_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "LANG": "C",
    "LC_ALL": "C",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}
EXPECTED_GENERATOR = {
    "left_neurons": 19,
    "right_neurons": 19,
    "count_total": 100,
    "strength": 1.8,
    "missing_fraction": 0.1,
}
EXPECTED_FIT = {
    "integration_draws": 256,
    "folds": 2,
    "ridge": 0.1,
    "pseudocount": 0.5,
    "tie_tolerance": 1e-6,
    "component_candidates": [2, 3],
}
EXPECTED_CASES = (
    {
        "case_id": "adequate_template_count_aware",
        "scenario": "adequate_template",
        "seed": 1201,
        "representation": "count_aware_log_ratio",
        "rank": 16,
        "covariance": "diagonal",
        "covariance_rank": 8,
    },
    {
        "case_id": "curved_continuum_low_rank_composition",
        "scenario": "curved_continuum_dense",
        "seed": 1202,
        "representation": "low_rank_composition",
        "rank": 2,
        "covariance": "shrinkage",
        "covariance_rank": 4,
    },
    {
        "case_id": "residual_groups_low_rank_count_model",
        "scenario": "residual_groups_within_continuum",
        "seed": 1201,
        "representation": "low_rank_count_model",
        "rank": 4,
        "covariance": "low_rank",
        "covariance_rank": 8,
    },
)
EXPECTED_PHASE_CONTRACTS = {
    "legacy": "SCIENTIFIC_CONTRACT.yaml",
    "post36": "SCIENTIFIC_CONTRACT_POST36.yaml",
}
EXPECTED_DIRECTIONS = ("left_to_right", "right_to_left")
MODEL_FAMILIES = ("T", "U", "M")


class ProbeError(RuntimeError):
    """The outcome-blind transition probe failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProbeError("payload is not canonical finite JSON") from exc
    return encoded.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_payload(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _has_forbidden_data_component(path: Path) -> bool:
    for component in path.parts:
        lowered = component.lower()
        if lowered == "inputs":
            return True
        if any(token in lowered for token in ("connectome", "connectivity")):
            return True
        if lowered in {
            "weights",
            "body-statistics",
            "body_statistics",
            "synapses",
            "annotations",
        }:
            return True
    return False


def _reject_symlink_components(path: Path, *, allow_missing_leaf: bool) -> None:
    path = _lexical_absolute(path)
    current = Path(path.anchor)
    for index, component in enumerate(path.parts[1:]):
        current = current / component
        is_leaf = index == len(path.parts[1:]) - 1
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and is_leaf:
                return
            raise ProbeError(f"path component does not exist: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise ProbeError(f"symlinked path components are forbidden: {current}")


def _validate_nondataset_path(
    path: Path | str,
    *,
    label: str,
    must_exist: bool,
    expect_directory: bool = False,
    output: bool = False,
) -> Path:
    candidate = _lexical_absolute(path)
    if _has_forbidden_data_component(candidate):
        raise ProbeError(f"{label} resembles a forbidden data path")
    allowed_roots = (EPISODE_OUTPUTS, EPISODE_SCRATCH, Path("/tmp"))
    if not any(_is_within(candidate, root) for root in allowed_roots):
        raise ProbeError(
            f"{label} must remain under EP12 outputs, EP12 scratch, or /tmp"
        )
    _reject_symlink_components(candidate, allow_missing_leaf=output and not must_exist)
    if must_exist:
        if not candidate.exists():
            raise ProbeError(f"{label} does not exist: {candidate}")
        if expect_directory and not candidate.is_dir():
            raise ProbeError(f"{label} must be a directory: {candidate}")
        if not expect_directory and not candidate.is_file():
            raise ProbeError(f"{label} must be a regular file: {candidate}")
    elif candidate.exists() and candidate.is_symlink():
        raise ProbeError(f"{label} cannot be a symlink: {candidate}")
    return candidate


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read valid JSON for {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"{label} must be a JSON object")
    return value


def _validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ProbeError("unexpected contract-transition probe spec schema")
    scope = spec.get("data_scope")
    if not isinstance(scope, dict):
        raise ProbeError("probe spec lacks data_scope")
    if scope.get("generated_examples_only") is not True:
        raise ProbeError("probe spec must be generated-data-only")
    if scope.get("real_connectivity_access") != "forbidden":
        raise ProbeError("probe spec does not forbid real connectivity")
    if scope.get("connectivity_path_arguments") != "forbidden":
        raise ProbeError("probe spec does not forbid connectivity path arguments")
    if tuple(spec.get("source_modules", ())) != EXPECTED_MODULES:
        raise ProbeError("probe source-module list differs from the fixed list")
    if spec.get("required_environment") != EXPECTED_REQUIRED_ENVIRONMENT:
        raise ProbeError("probe thread/runtime environment differs from the fixed values")
    if spec.get("generator") != EXPECTED_GENERATOR:
        raise ProbeError("probe generator settings differ from the fixed values")
    if spec.get("fit") != EXPECTED_FIT:
        raise ProbeError("probe fit settings differ from the fixed values")
    if tuple(spec.get("cases", ())) != EXPECTED_CASES:
        raise ProbeError("probe cases differ from the three fixed cases")
    if tuple(spec.get("directions", ())) != EXPECTED_DIRECTIONS:
        raise ProbeError("probe directions differ from reciprocal transfer")
    phases = spec.get("phases")
    if not isinstance(phases, dict):
        raise ProbeError("probe phases are missing")
    observed_phase_contracts = {
        phase: value.get("contract_relative_path")
        for phase, value in phases.items()
        if isinstance(value, dict)
    }
    if observed_phase_contracts != EXPECTED_PHASE_CONTRACTS:
        raise ProbeError("probe phase contracts differ from the fixed transition")


def _load_spec(path: Path | str) -> tuple[Path, Mapping[str, Any]]:
    spec_path = _validate_nondataset_path(
        path, label="probe spec", must_exist=True, expect_directory=False
    )
    if spec_path.name != "CONTRACT_TRANSITION_PROBE_SPEC.json":
        raise ProbeError("probe spec must use the fixed filename")
    spec = _read_json(spec_path, label="probe spec")
    _validate_spec(spec)
    return spec_path, spec


def _require_environment(spec: Mapping[str, Any]) -> None:
    required = spec["required_environment"]
    mismatches = {
        key: {"required": value, "observed": os.environ.get(key)}
        for key, value in required.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise ProbeError(
            "deterministic single-thread environment is not fixed: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _source_identity(
    source_root: Path,
    contract_relative_path: str,
    spec_path: Path,
) -> dict[str, Any]:
    if contract_relative_path not in EXPECTED_PHASE_CONTRACTS.values():
        raise ProbeError("unrecognized scientific contract filename")
    contract = source_root / contract_relative_path
    _reject_symlink_components(contract, allow_missing_leaf=False)
    if not contract.is_file():
        raise ProbeError(f"scientific contract is missing: {contract}")
    module_hashes: dict[str, str] = {}
    for relative in EXPECTED_MODULES:
        module_path = source_root / relative
        _reject_symlink_components(module_path, allow_missing_leaf=False)
        if not module_path.is_file():
            raise ProbeError(f"required source module is missing: {module_path}")
        module_hashes[relative] = _sha256_file(module_path)
    source_payload = {
        "contract_relative_path": contract_relative_path,
        "contract_sha256": _sha256_file(contract),
        "module_sha256": module_hashes,
    }
    return {
        "source_executor_root": str(source_root),
        "source_payload_sha256": _sha256_payload(source_payload),
        "module_sha256": module_hashes,
        "contract_relative_path": contract_relative_path,
        "contract_sha256": source_payload["contract_sha256"],
        "spec_path": str(spec_path),
        "spec_sha256": _sha256_file(spec_path),
        "probe_program_path": str(THIS_FILE),
        "probe_program_sha256": _sha256_file(THIS_FILE),
    }


def _module_file_record(module: Any) -> dict[str, str]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise ProbeError(f"module lacks a source file: {module!r}")
    path = Path(raw_path).resolve(strict=True)
    return {"path": str(path), "sha256": _sha256_file(path)}


def _import_scientific_stack(source_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    namespace = "_ep12_contract_probe_source"
    already_loaded = sorted(
        name
        for name in sys.modules
        if name == namespace or name.startswith(namespace + ".")
    )
    if already_loaded:
        raise ProbeError(
            "the private scientific-source namespace is already loaded"
        )

    package_directory = source_root / "ep12_executor"
    package = types.ModuleType(namespace)
    package.__file__ = str(package_directory / "__init__.py")
    package.__package__ = namespace
    package.__path__ = [str(package_directory)]
    package_spec = importlib.machinery.ModuleSpec(
        namespace, loader=None, is_package=True
    )
    package_spec.submodule_search_locations = [str(package_directory)]
    package.__spec__ = package_spec
    sys.modules[namespace] = package

    def load_source_module(module_name: str) -> Any:
        qualified_name = f"{namespace}.{module_name}"
        module_path = package_directory / f"{module_name}.py"
        specification = importlib.util.spec_from_file_location(
            qualified_name, module_path
        )
        if specification is None or specification.loader is None:
            raise ProbeError(f"cannot construct a loader for {module_name}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[qualified_name] = module
        specification.loader.exec_module(module)
        return module

    try:
        # Do not execute the archived package __init__.py here.  It imports the
        # controller plane, which is intentionally outside this generated-data
        # probe's minimal authenticated scientific surface.
        models = load_source_module("scientific_models")
        synthetic = load_source_module("scientific_synthetic")
        numpy = importlib.import_module("numpy")
        scipy = importlib.import_module("scipy")
        sklearn = importlib.import_module("sklearn")
    except Exception as exc:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        raise ProbeError("cannot import the explicit scientific source root") from exc
    expected_models = (source_root / "ep12_executor/scientific_models.py").resolve()
    expected_synthetic = (source_root / "ep12_executor/scientific_synthetic.py").resolve()
    if Path(models.__file__).resolve() != expected_models:
        raise ProbeError("scientific_models imported from outside the explicit source root")
    if Path(synthetic.__file__).resolve() != expected_synthetic:
        raise ProbeError("scientific_synthetic imported from outside the explicit source root")
    return models, synthetic, numpy, scipy, sklearn


def _environment_payload(
    numpy: Any,
    scipy: Any,
    sklearn: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    dependency_modules = {
        "numpy": _module_file_record(numpy),
        "numpy_core": _module_file_record(
            importlib.import_module("numpy.core._multiarray_umath")
        ),
        "scipy": _module_file_record(scipy),
        "scipy_special": _module_file_record(
            importlib.import_module("scipy.special._ufuncs")
        ),
        "sklearn": _module_file_record(sklearn),
        "sklearn_mixture": _module_file_record(
            importlib.import_module("sklearn.mixture._gaussian_mixture")
        ),
    }
    payload = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "byteorder": sys.byteorder,
            "executable_path": str(executable),
            "executable_sha256": _sha256_file(executable),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "numpy_version": str(numpy.__version__),
            "scipy_version": str(scipy.__version__),
            "sklearn_version": str(sklearn.__version__),
            "module_files": dependency_modules,
        },
        "required_environment": {
            key: os.environ[key] for key in spec["required_environment"]
        },
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
    }
    return payload


def _float_hex(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ProbeError("non-finite scientific scalar")
    return number.hex()


def _json_numeric_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_numeric_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numeric_tree(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return {"float_hex": _float_hex(value)}
    if hasattr(value, "item"):
        return _json_numeric_tree(value.item())
    raise ProbeError(f"unsupported diagnostic value: {type(value).__name__}")


def _array_record(value: Any, *, include_bytes: bool) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - guarded by scientific import
        raise ProbeError("numpy is unavailable") from exc
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ProbeError("object arrays cannot enter a probe payload")
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    record: dict[str, Any] = {
        "dtype": contiguous.dtype.str,
        "shape": [int(size) for size in contiguous.shape],
        "c_contiguous": True,
        "byte_count": len(raw),
        "c_order_bytes_sha256": _sha256_bytes(raw),
    }
    if include_bytes:
        record["c_order_bytes_base64"] = base64.b64encode(raw).decode("ascii")
    return record


def _selection_payload(selection: Any) -> dict[str, Any]:
    return {
        "model_id": str(selection.model_id),
        "mean_score": _float_hex(selection.mean_score),
        "standard_error": _float_hex(selection.standard_error),
        "fold_scores": _array_record(selection.fold_scores, include_bytes=True),
        "parameter_count": int(selection.parameter_count),
    }


def _model_scientific_payload(model: Any) -> dict[str, Any]:
    tensors = {
        "basis": _array_record(model.basis, include_bytes=True),
        "nuisance_coefficients": _array_record(
            model.nuisance_coefficients, include_bytes=True
        ),
        "latent_offsets": _array_record(model.latent_offsets, include_bytes=True),
        "log_weights": _array_record(model.log_weights, include_bytes=True),
        "prior_means": _array_record(model.prior_means, include_bytes=True),
        "prior_covariances": _array_record(
            model.prior_covariances, include_bytes=True
        ),
        "prior_weights": _array_record(model.prior_weights, include_bytes=True),
    }
    payload = {
        "model_id": str(model.model_id),
        "integration_draws": int(model.integration_draws),
        "integration_seed": int(model.integration_seed),
        "parameter_count": int(model.parameter_count),
        "pseudocount": _float_hex(model.pseudocount),
        "tensors": tensors,
    }
    payload["fit_payload_sha256"] = _sha256_payload(payload)
    return payload


def _score_payload(scored: Mapping[str, Any]) -> dict[str, Any]:
    vectors = {
        name: _array_record(scored[name], include_bytes=True)
        for name in ("T", "U", "M", "reference", "delta_M_minus_reference")
    }
    payload = {
        "vectors": vectors,
        "mean_scores": {
            name: _float_hex(scored["mean_scores"][name])
            for name in ("T", "U", "M")
        },
        "mean_delta_M_minus_reference": _float_hex(
            scored["mean_delta_M_minus_reference"]
        ),
        "reference_model_id": str(scored["reference_model_id"]),
        "null_generator_model_id": str(scored["null_generator_model_id"]),
    }
    payload["score_payload_sha256"] = _sha256_payload(payload)
    return payload


def _fit_direction(
    *,
    models: Any,
    source: Any,
    target: Any,
    case: Mapping[str, Any],
    direction: str,
    fit_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_label = "left-source" if direction == "left_to_right" else "right-source"
    fit_seed = int(models.seed_from(case["seed"], seed_label))
    fitted = models.fit_triplet(
        source.counts,
        source.covariates,
        seed=fit_seed,
        integration_draws=int(fit_spec["integration_draws"]),
        latent_rank=int(case["rank"]),
        ridge=float(fit_spec["ridge"]),
        folds=int(fit_spec["folds"]),
        tie_tolerance=float(fit_spec["tie_tolerance"]),
        pseudocount=float(fit_spec["pseudocount"]),
        representation=str(case["representation"]),
        covariance_mode=str(case["covariance"]),
        covariance_rank=int(case["covariance_rank"]),
        component_candidates=tuple(int(x) for x in fit_spec["component_candidates"]),
    )
    scored = fitted.score_target(target.counts, target.covariates)
    family_models = {"T": fitted.T, "U": fitted.U, "M": fitted.M}
    scientific = {
        "fit_seed": fit_seed,
        "selected_model_ids": {
            family: str(model.model_id) for family, model in family_models.items()
        },
        "reference_model_id": str(fitted.reference_model_id),
        "null_generator_model_id": str(fitted.null_generator_model_id),
        "cross_validation": {
            "T": _selection_payload(fitted.T_cv),
            "U": _selection_payload(fitted.U_cv),
            "M": _selection_payload(fitted.M_cv),
        },
        "models": {
            family: _model_scientific_payload(model)
            for family, model in family_models.items()
        },
        "target_scores": _score_payload(scored),
    }
    scientific["direction_payload_sha256"] = _sha256_payload(scientific)
    diagnostics = {
        "models": {
            family: {
                "model_id": str(model.model_id),
                "diagnostics": _json_numeric_tree(model.diagnostics),
            }
            for family, model in family_models.items()
        }
    }
    return scientific, diagnostics


def _build_payloads(
    models: Any,
    synthetic: Any,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    generator = spec["generator"]
    fit_spec = spec["fit"]
    input_cases: dict[str, Any] = {}
    scientific_cases: dict[str, Any] = {}
    diagnostic_cases: dict[str, Any] = {}
    for case in spec["cases"]:
        case_id = str(case["case_id"])
        fixture = synthetic.generate_synthetic_type(
            str(case["scenario"]),
            left_neurons=int(generator["left_neurons"]),
            right_neurons=int(generator["right_neurons"]),
            count_total=int(generator["count_total"]),
            strength=float(generator["strength"]),
            missing_fraction=float(generator["missing_fraction"]),
            seed=int(case["seed"]),
        )
        input_case = {
            "scenario": str(fixture.scenario),
            "truth_family": str(fixture.truth_family),
            "partner_names": [str(name) for name in fixture.partner_names],
            "generator_parameters": _json_numeric_tree(
                fixture.generator_parameters
            ),
            "fit_branch": _json_numeric_tree(dict(case)),
            "sides": {
                "left": {
                    "counts": _array_record(fixture.left.counts, include_bytes=False),
                    "covariates": _array_record(
                        fixture.left.covariates, include_bytes=False
                    ),
                },
                "right": {
                    "counts": _array_record(fixture.right.counts, include_bytes=False),
                    "covariates": _array_record(
                        fixture.right.covariates, include_bytes=False
                    ),
                },
            },
        }
        input_case["case_input_payload_sha256"] = _sha256_payload(input_case)
        input_cases[case_id] = input_case

        scientific_directions: dict[str, Any] = {}
        diagnostic_directions: dict[str, Any] = {}
        for direction in EXPECTED_DIRECTIONS:
            if direction == "left_to_right":
                source, target = fixture.left, fixture.right
            else:
                source, target = fixture.right, fixture.left
            scientific_direction, diagnostic_direction = _fit_direction(
                models=models,
                source=source,
                target=target,
                case=case,
                direction=direction,
                fit_spec=fit_spec,
            )
            scientific_directions[direction] = scientific_direction
            diagnostic_directions[direction] = diagnostic_direction
        scientific_cases[case_id] = {"directions": scientific_directions}
        diagnostic_cases[case_id] = {"directions": diagnostic_directions}

    input_payload = {
        "schema_version": INPUT_SCHEMA,
        "generator": _json_numeric_tree(generator),
        "fit": _json_numeric_tree(fit_spec),
        "cases": input_cases,
    }
    scientific_payload = {
        "schema_version": SCIENTIFIC_SCHEMA,
        "cases": scientific_cases,
    }
    diagnostics_payload = {
        "schema_version": DIAGNOSTICS_SCHEMA,
        "cases": diagnostic_cases,
    }
    return input_payload, scientific_payload, diagnostics_payload


def _artifact_digest(record: Mapping[str, Any]) -> str:
    without_digest = dict(record)
    without_digest.pop("artifact_sha256", None)
    return _sha256_payload(without_digest)


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    parent = path.parent
    _reject_symlink_components(parent, allow_missing_leaf=False)
    if not parent.is_dir():
        raise ProbeError(f"output parent is not a directory: {parent}")
    if path.exists() or path.is_symlink():
        raise ProbeError(f"refusing to overwrite probe evidence: {path}")
    payload = json.dumps(
        record,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_embedded_arrays(value: Any) -> None:
    if isinstance(value, Mapping):
        if "c_order_bytes_base64" in value:
            try:
                raw = base64.b64decode(
                    value["c_order_bytes_base64"], validate=True
                )
            except Exception as exc:
                raise ProbeError("invalid base64 tensor payload") from exc
            if len(raw) != value.get("byte_count"):
                raise ProbeError("tensor byte count does not match embedded bytes")
            if _sha256_bytes(raw) != value.get("c_order_bytes_sha256"):
                raise ProbeError("tensor byte hash does not match embedded bytes")
        for child in value.values():
            _validate_embedded_arrays(child)
    elif isinstance(value, list):
        for child in value:
            _validate_embedded_arrays(child)


def _verify_run_record(record: Mapping[str, Any], spec_sha256: str) -> None:
    if record.get("schema_version") != RUN_SCHEMA:
        raise ProbeError("not an EP12 contract-transition run receipt")
    if record.get("phase") not in EXPECTED_PHASE_CONTRACTS:
        raise ProbeError("invalid probe phase")
    if record.get("artifact_sha256") != _artifact_digest(record):
        raise ProbeError("probe artifact self-hash mismatch")
    firewall = record.get("outcome_firewall")
    if not isinstance(firewall, dict):
        raise ProbeError("probe outcome firewall is missing")
    required_firewall = {
        "generated_examples_only": True,
        "real_connectivity_accessed": False,
        "connectivity_paths_accepted": False,
        "annotation_or_outcome_files_accessed": False,
    }
    if firewall != required_firewall:
        raise ProbeError("probe outcome firewall is not exact")
    identity = record.get("byte_identity")
    if not isinstance(identity, dict) or identity.get("start") != identity.get("end"):
        raise ProbeError("source/module/contract/spec bytes changed during replay")
    start = identity["start"]
    if not isinstance(start, dict) or start.get("spec_sha256") != spec_sha256:
        raise ProbeError("probe was not run against the supplied fixed spec")
    expected_hashes = {
        "input_payload_sha256": _sha256_payload(record.get("input_payload")),
        "environment_payload_sha256": _sha256_payload(
            record.get("environment_payload")
        ),
        "scientific_payload_sha256": _sha256_payload(
            record.get("scientific_payload")
        ),
        "diagnostics_payload_sha256": _sha256_payload(
            record.get("diagnostics_payload")
        ),
    }
    for field, expected in expected_hashes.items():
        if record.get(field) != expected:
            raise ProbeError(f"probe payload hash mismatch: {field}")
    input_payload = record.get("input_payload")
    scientific_payload = record.get("scientific_payload")
    diagnostics_payload = record.get("diagnostics_payload")
    if not isinstance(input_payload, dict) or input_payload.get("schema_version") != INPUT_SCHEMA:
        raise ProbeError("invalid probe input payload")
    if not isinstance(scientific_payload, dict) or scientific_payload.get("schema_version") != SCIENTIFIC_SCHEMA:
        raise ProbeError("invalid probe scientific payload")
    if not isinstance(diagnostics_payload, dict) or diagnostics_payload.get("schema_version") != DIAGNOSTICS_SCHEMA:
        raise ProbeError("invalid probe diagnostics payload")
    expected_cases = {case["case_id"] for case in EXPECTED_CASES}
    for payload in (input_payload, scientific_payload, diagnostics_payload):
        cases = payload.get("cases")
        if not isinstance(cases, dict) or set(cases) != expected_cases:
            raise ProbeError("probe payload does not contain exactly the three fixed cases")
    _validate_embedded_arrays(scientific_payload)


def _allowed_added_rank_path(path: tuple[str, ...], value: Any) -> bool:
    if len(path) != 8:
        return False
    expected_prefix = (
        "cases",
        "adequate_template_count_aware",
        "directions",
    )
    if path[:3] != expected_prefix:
        return False
    if path[3] not in EXPECTED_DIRECTIONS:
        return False
    if path[4] != "models" or path[5] not in MODEL_FAMILIES:
        return False
    if path[6:] != ("diagnostics", "representation_rank"):
        return False
    return isinstance(value, int) and not isinstance(value, bool) and value == 10


def _diagnostic_additions(
    legacy: Any,
    post36: Any,
    path: tuple[str, ...] = (),
) -> list[str]:
    additions: list[str] = []
    if isinstance(legacy, Mapping) and isinstance(post36, Mapping):
        missing = set(legacy) - set(post36)
        if missing:
            raise ProbeError(
                "post36 diagnostics removed fields at " + ".".join(path or ("<root>",))
            )
        for key in sorted(set(post36) - set(legacy)):
            addition_path = path + (str(key),)
            if not _allowed_added_rank_path(addition_path, post36[key]):
                raise ProbeError(
                    "unapproved post36 diagnostic addition: "
                    + ".".join(addition_path)
                )
            additions.append(".".join(addition_path))
        for key in sorted(legacy):
            additions.extend(
                _diagnostic_additions(legacy[key], post36[key], path + (str(key),))
            )
        return additions
    if isinstance(legacy, list) and isinstance(post36, list):
        if len(legacy) != len(post36):
            raise ProbeError("diagnostic list length changed at " + ".".join(path))
        for index, (before, after) in enumerate(zip(legacy, post36)):
            additions.extend(
                _diagnostic_additions(before, after, path + (str(index),))
            )
        return additions
    if legacy != post36:
        raise ProbeError("diagnostic value changed at " + ".".join(path))
    return additions


def _expected_rank_additions() -> set[str]:
    return {
        ".".join(
            (
                "cases",
                "adequate_template_count_aware",
                "directions",
                direction,
                "models",
                family,
                "diagnostics",
                "representation_rank",
            )
        )
        for direction in EXPECTED_DIRECTIONS
        for family in MODEL_FAMILIES
    }


def _compare_probe_records(
    legacy: Mapping[str, Any], post36: Mapping[str, Any]
) -> dict[str, Any]:
    if legacy.get("phase") != "legacy" or post36.get("phase") != "post36":
        raise ProbeError("comparison requires one legacy and one post36 receipt")
    exact_fields = (
        "input_payload_sha256",
        "environment_payload_sha256",
        "scientific_payload_sha256",
    )
    for field in exact_fields:
        if legacy.get(field) != post36.get(field):
            raise ProbeError(f"legacy/post36 exact comparison failed: {field}")
    additions = _diagnostic_additions(
        legacy.get("diagnostics_payload"), post36.get("diagnostics_payload")
    )
    if set(additions) != _expected_rank_additions():
        missing = sorted(_expected_rank_additions() - set(additions))
        unexpected = sorted(set(additions) - _expected_rank_additions())
        raise ProbeError(
            "representation-rank diagnostic delta is incomplete: "
            + json.dumps({"missing": missing, "unexpected": unexpected})
        )
    return {
        "input_payload_byte_identical": True,
        "environment_byte_identical": True,
        "scientific_payload_byte_identical": True,
        "diagnostic_changes_limited_to_added_representation_rank": True,
        "diagnostic_additions": sorted(additions),
    }


def run_probe(
    *,
    phase: str,
    source_executor_root: Path | str,
    spec_path: Path | str,
    output_path: Path | str,
) -> Mapping[str, Any]:
    if phase not in EXPECTED_PHASE_CONTRACTS:
        raise ProbeError("phase must be legacy or post36")
    source_root = _validate_nondataset_path(
        source_executor_root,
        label="source executor root",
        must_exist=True,
        expect_directory=True,
    )
    if source_root.name != "executor":
        raise ProbeError("source executor root must be the explicit executor directory")
    spec_file, spec = _load_spec(spec_path)
    output = _validate_nondataset_path(
        output_path, label="probe output", must_exist=False, output=True
    )
    if output.suffix != ".json":
        raise ProbeError("probe output must be a JSON file")
    _require_environment(spec)
    contract_relative = EXPECTED_PHASE_CONTRACTS[phase]
    start_identity = _source_identity(source_root, contract_relative, spec_file)
    models, synthetic, numpy, scipy, sklearn = _import_scientific_stack(source_root)
    environment = _environment_payload(numpy, scipy, sklearn, spec)
    input_payload, scientific_payload, diagnostics_payload = _build_payloads(
        models, synthetic, spec
    )
    end_identity = _source_identity(source_root, contract_relative, spec_file)
    if start_identity != end_identity:
        raise ProbeError("source/module/contract/spec bytes changed during replay")
    record: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "phase": phase,
        "outcome_firewall": {
            "generated_examples_only": True,
            "real_connectivity_accessed": False,
            "connectivity_paths_accepted": False,
            "annotation_or_outcome_files_accessed": False,
        },
        "provenance": {
            "source_executor_root": str(source_root),
            "contract_relative_path": contract_relative,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "byte_identity": {"start": start_identity, "end": end_identity},
        "environment_payload": environment,
        "environment_payload_sha256": _sha256_payload(environment),
        "input_payload": input_payload,
        "input_payload_sha256": _sha256_payload(input_payload),
        "scientific_payload": scientific_payload,
        "scientific_payload_sha256": _sha256_payload(scientific_payload),
        "diagnostics_payload": diagnostics_payload,
        "diagnostics_payload_sha256": _sha256_payload(diagnostics_payload),
    }
    record["artifact_sha256"] = _artifact_digest(record)
    _verify_run_record(record, _sha256_file(spec_file))
    _atomic_json(output, record)
    return record


def verify_artifact(
    *, artifact_path: Path | str, spec_path: Path | str
) -> Mapping[str, Any]:
    artifact = _validate_nondataset_path(
        artifact_path, label="probe artifact", must_exist=True
    )
    spec_file, _ = _load_spec(spec_path)
    record = _read_json(artifact, label="probe artifact")
    schema = record.get("schema_version")
    if schema == RUN_SCHEMA:
        _verify_run_record(record, _sha256_file(spec_file))
    elif schema == COMPARE_SCHEMA:
        if record.get("artifact_sha256") != _artifact_digest(record):
            raise ProbeError("comparison artifact self-hash mismatch")
        if record.get("spec_sha256") != _sha256_file(spec_file):
            raise ProbeError("comparison artifact uses another spec")
        if record.get("status") != "pass":
            raise ProbeError("comparison artifact did not pass")
        checks = record.get("checks")
        if not isinstance(checks, dict) or any(
            checks.get(name) is not True
            for name in (
                "input_payload_byte_identical",
                "environment_byte_identical",
                "scientific_payload_byte_identical",
                "diagnostic_changes_limited_to_added_representation_rank",
            )
        ):
            raise ProbeError("comparison artifact checks are incomplete")
        if set(checks.get("diagnostic_additions", ())) != _expected_rank_additions():
            raise ProbeError("comparison artifact rank additions are incomplete")
    else:
        raise ProbeError("unrecognized contract-transition artifact schema")
    return record


def compare_probes(
    *,
    legacy_path: Path | str,
    post36_path: Path | str,
    spec_path: Path | str,
    output_path: Path | str,
) -> Mapping[str, Any]:
    legacy_file = _validate_nondataset_path(
        legacy_path, label="legacy probe", must_exist=True
    )
    post36_file = _validate_nondataset_path(
        post36_path, label="post36 probe", must_exist=True
    )
    spec_file, _ = _load_spec(spec_path)
    output = _validate_nondataset_path(
        output_path, label="comparison output", must_exist=False, output=True
    )
    legacy = _read_json(legacy_file, label="legacy probe")
    post36 = _read_json(post36_file, label="post36 probe")
    spec_sha = _sha256_file(spec_file)
    _verify_run_record(legacy, spec_sha)
    _verify_run_record(post36, spec_sha)
    checks = _compare_probe_records(legacy, post36)
    comparison: dict[str, Any] = {
        "schema_version": COMPARE_SCHEMA,
        "status": "pass",
        "outcome_firewall": {
            "generated_examples_only": True,
            "real_connectivity_accessed": False,
            "connectivity_paths_accepted": False,
            "annotation_or_outcome_files_accessed": False,
        },
        "spec_path": str(spec_file),
        "spec_sha256": spec_sha,
        "legacy": {
            "path": str(legacy_file),
            "file_sha256": _sha256_file(legacy_file),
            "artifact_sha256": legacy["artifact_sha256"],
            "source_payload_sha256": legacy["byte_identity"]["start"][
                "source_payload_sha256"
            ],
            "contract_sha256": legacy["byte_identity"]["start"][
                "contract_sha256"
            ],
        },
        "post36": {
            "path": str(post36_file),
            "file_sha256": _sha256_file(post36_file),
            "artifact_sha256": post36["artifact_sha256"],
            "source_payload_sha256": post36["byte_identity"]["start"][
                "source_payload_sha256"
            ],
            "contract_sha256": post36["byte_identity"]["start"][
                "contract_sha256"
            ],
        },
        "checks": checks,
    }
    comparison["artifact_sha256"] = _artifact_digest(comparison)
    _atomic_json(output, comparison)
    verify_artifact(artifact_path=output, spec_path=spec_file)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EP12 generated-only contract-transition invariance probe"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one fixed synthetic replay")
    run.add_argument("--phase", required=True, choices=tuple(EXPECTED_PHASE_CONTRACTS))
    run.add_argument("--source-executor-root", required=True, type=Path)
    run.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    run.add_argument("--output", required=True, type=Path)

    verify = subparsers.add_parser("verify", help="verify a run or comparison artifact")
    verify.add_argument("--artifact", required=True, type=Path)
    verify.add_argument("--spec", type=Path, default=DEFAULT_SPEC)

    compare = subparsers.add_parser(
        "compare", help="compare exact legacy and post36 synthetic replays"
    )
    compare.add_argument("--legacy", required=True, type=Path)
    compare.add_argument("--post36", required=True, type=Path)
    compare.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            record = run_probe(
                phase=arguments.phase,
                source_executor_root=arguments.source_executor_root,
                spec_path=arguments.spec,
                output_path=arguments.output,
            )
            print(record["artifact_sha256"])
        elif arguments.command == "verify":
            record = verify_artifact(
                artifact_path=arguments.artifact, spec_path=arguments.spec
            )
            print(record["artifact_sha256"])
        elif arguments.command == "compare":
            record = compare_probes(
                legacy_path=arguments.legacy,
                post36_path=arguments.post36,
                spec_path=arguments.spec,
                output_path=arguments.output,
            )
            print(record["artifact_sha256"])
        else:  # pragma: no cover - argparse enforces this
            parser.error("unknown command")
    except ProbeError as exc:
        print(f"contract invariance probe failed closed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
