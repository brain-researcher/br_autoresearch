"""Adaptive fitted-T/U full-search null execution for EP12.

Each replicate is sampled by fitted_null.sample_fitted_null, every trial in an
authenticated realized controller schedule is executed through a lock-bound
fit/score callback, and one atomic checkpoint is written.  This module has no
MaleCNS loader and no final-connectivity path.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .fitted_null import (
    NULL_REPLICATE_COUNT,
    FittedNullDesign,
    FittedNullSample,
    TypePooledExpansionPlan,
    fitted_generator_manifest,
    null_seed_binding,
    sample_fitted_null,
    verify_null_seed_manifest,
)
from .null_controller import (
    NullControllerError,
    disposition_for_result,
    freeze_null_controller_program,
    freeze_realized_null_schedule,
    normalize_objective,
    propose_next_trial,
    records_from_steps,
    stop_status,
    verify_null_controller_program,
)
from .policy import EpisodePolicy
from .scientific_models import FittedPredictiveModel


PROCEDURE_LOCK_SCHEMA = "ep12.full_search_null_procedure_lock.v2"
CALLBACK_IDENTITY_SCHEMA = "ep12.full_search_callback_identity.v2"
CALLBACK_PROTOCOL = "ep12.full_search_fit_score_callback.v3"
FIT_RESULT_SCHEMA = "ep12.full_search_fit_score_result.v4"
FINAL_RESULT_SCHEMA = "ep12.full_search_callback_final.v3"
FINAL_ARTIFACT_SCHEMA = "ep12.full_search_selection_artifact.v1"
RUN_MANIFEST_SCHEMA = "ep12.full_search_null_run_manifest.v2"
STEP_CHECKPOINT_SCHEMA = "ep12.full_search_null_step_checkpoint.v1"
CHECKPOINT_SCHEMA = "ep12.full_search_null_replicate_checkpoint.v2"
WALL_BUDGET_EXHAUSTION_SCHEMA = (
    "ep12.full_search_null_wall_budget_exhaustion.v1"
)
AGGREGATE_SCHEMA = "ep12.executed_full_search_null.v3"
AGGREGATE_MANIFEST_SCHEMA = "ep12.executed_full_search_null_manifest.v3"
REQUEUE_EXIT_CODE = 75
RESTART_SENTINEL_BYTES = b"ep12-safe-requeue-request-v1\n"
MODEL_TRIPLET = (
    "T_type_template",
    "U_flexible_unimodal",
    "M_residual_modes",
)
PRODUCTION_CALLBACK_ID = "ep12.production_full_search_callback.v2"
ATOMIC_TEMP_DIRECTORY = ".ep12_atomic_tmp"
_ATOMIC_TEMP_FILE_RE = re.compile(
    r"^[0-9a-f]{16}\.[0-9a-f]{16}\.[0-9a-f]{16}\.tmp$"
)
_TRIAL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EPISODE_OUTPUTS_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency/outputs"
)
_EPISODE_SCRATCH_ROOT = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)


class FullSearchNullError(RuntimeError):
    """The full-search null is incomplete, unauthenticated, or unsafe."""


class FullSearchNullRequeueRequested(RuntimeError):
    """A safe scheduler restart was requested at a trial boundary."""

    def __init__(self, *, replicate_index: int, completed_trial_count: int) -> None:
        super().__init__("safe scheduler restart requested at a trial boundary")
        self.replicate_index = int(replicate_index)
        self.completed_trial_count = int(completed_trial_count)


class FullSearchNullTrialTimeout(FullSearchNullError):
    """One isolated callback reached its authenticated timeout."""


def _trial_id(value: Any, label: str = "trial ID") -> str:
    result = str(value)
    if _TRIAL_ID_RE.fullmatch(result) is None:
        raise FullSearchNullError(f"{label} is not a safe canonical identifier")
    return result


def _canonical_output_path(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise FullSearchNullError(f"{label} must be absolute and traversal-free")
    lexical = Path(os.path.abspath(os.fspath(raw)))
    resolved = raw.resolve(strict=False)
    if lexical != resolved:
        raise FullSearchNullError(f"{label} may not traverse a symbolic link")
    return resolved


def _restart_sentinel_path(
    path: Path,
    *,
    output_dir: Path,
    production: bool,
) -> Path:
    """Validate a restart request path without admitting it to result artifacts."""

    resolved = _canonical_output_path(path, label="restart sentinel path")
    allowed_root = (
        _EPISODE_SCRATCH_ROOT.resolve(strict=False)
        if production
        else output_dir
    )
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as error:
        raise FullSearchNullError(
            "restart sentinel must be inside the dedicated EP12 scratch root"
            if production
            else "synthetic restart sentinel must be inside its output root"
        ) from error
    if not relative.parts:
        raise FullSearchNullError("restart sentinel may not equal its allowed root")
    if production:
        try:
            resolved.relative_to(output_dir)
        except ValueError:
            pass
        else:
            raise FullSearchNullError(
                "production restart sentinel must be outside the artifact tree"
            )
    return resolved


def _restart_requested(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise FullSearchNullError("restart sentinel is unreadable") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FullSearchNullError("restart sentinel is not an ordinary file")
        encoded = os.read(descriptor, len(RESTART_SENTINEL_BYTES) + 1)
    finally:
        os.close(descriptor)
    if encoded != RESTART_SENTINEL_BYTES:
        raise FullSearchNullError("restart sentinel payload is invalid")
    return True


def _production_output_path(
    runtime: "FullSearchNullRuntime",
    path: Path,
    *,
    aggregate: bool,
) -> Path:
    """Enforce the scratch execution and durable aggregate path roles."""

    resolved = _canonical_output_path(path, label="full-search output path")
    identity = runtime.callback.identity()
    if identity.get("callback_id") != PRODUCTION_CALLBACK_ID:
        return resolved
    inputs = getattr(runtime.callback, "inputs", None)
    work_root_value = getattr(inputs, "artifact_root", None)
    if work_root_value is None:
        raise FullSearchNullError("production callback lacks its scratch artifact root")
    work_root = _canonical_output_path(
        Path(work_root_value), label="production scratch artifact root"
    )
    if resolved == work_root:
        return resolved
    if not aggregate:
        raise FullSearchNullError(
            "production replicate output must equal the locked scratch work root"
        )
    outputs_root = _EPISODE_OUTPUTS_ROOT.resolve(strict=False)
    try:
        relative = resolved.relative_to(outputs_root)
    except ValueError as error:
        raise FullSearchNullError(
            "production aggregate output must be scratch or EP12 durable output"
        ) from error
    if not relative.parts:
        raise FullSearchNullError(
            "production aggregate output may not equal the EP12 outputs root"
        )
    return resolved


def _fit_score_child(
    callback: Any,
    sample: FittedNullSample,
    trial: Mapping[str, Any],
    context: Mapping[str, Any],
    sender: Any,
) -> None:
    try:
        sender.send(("ok", callback.fit_score(
            sample=sample, trial=trial, context=context
        )))
    except BaseException as error:
        sender.send(("error", type(error).__name__, str(error)))
    finally:
        sender.close()


def _invoke_fit_score(
    callback: Any,
    *,
    sample: FittedNullSample,
    trial: Mapping[str, Any],
    context: Mapping[str, Any],
    timeout_seconds: float,
    isolate: bool,
) -> tuple[Mapping[str, Any], float]:
    started = time.monotonic()
    if not isolate:
        result = callback.fit_score(sample=sample, trial=trial, context=context)
    else:
        receiver, sender = multiprocessing.get_context("fork").Pipe(duplex=False)
        process = multiprocessing.get_context("fork").Process(
            target=_fit_score_child,
            args=(callback, sample, trial, context, sender),
        )
        process.start()
        sender.close()
        if not receiver.poll(timeout_seconds):
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()
            raise FullSearchNullTrialTimeout(
                "trial exceeded the frozen wall-time limit"
            )
        try:
            message = receiver.recv()
        except EOFError as error:
            raise FullSearchNullError("isolated trial exited without a result") from error
        finally:
            receiver.close()
            process.join()
        if not message or message[0] != "ok":
            detail = ": ".join(map(str, message[1:])) if message else "unknown error"
            raise FullSearchNullError(f"isolated trial failed: {detail}")
        result = message[1]
    elapsed = time.monotonic() - started
    if not math.isfinite(elapsed) or elapsed < 0.0 or elapsed > timeout_seconds:
        raise FullSearchNullTrialTimeout(
            "trial exceeded the frozen wall-time limit"
        )
    return _mapping(result, "fit/score callback result"), elapsed


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FullSearchNullError("metadata is not canonical JSON") from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise FullSearchNullError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullSearchNullError(f"{label} must be a mapping")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return _digest({key: value for key, value in payload.items() if key != field})


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(_canonical_bytes(list(array.shape)) + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _open_safe_artifact_parent(*, root: Path, path: Path) -> tuple[int, str]:
    """Create/open ``path.parent`` beneath ``root`` without following links."""

    lexical_root = Path(os.path.abspath(os.fspath(root)))
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise FullSearchNullError("artifact target is outside its locked root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise FullSearchNullError("artifact target is not a safe root-relative path")
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(os.fspath(lexical_root), directory_flags)
    except OSError as error:
        raise FullSearchNullError("locked artifact root is missing or unsafe") from error
    current_descriptor = root_descriptor
    try:
        for component in relative.parts[:-1]:
            try:
                os.mkdir(component, mode=0o770, dir_fd=current_descriptor)
                os.fsync(current_descriptor)
            except FileExistsError:
                pass
            except OSError as error:
                raise FullSearchNullError(
                    "cannot create an artifact directory beneath the locked root"
                ) from error
            try:
                next_descriptor = os.open(
                    component, directory_flags, dir_fd=current_descriptor
                )
            except OSError as error:
                raise FullSearchNullError(
                    "artifact directory component is a symlink or not a directory"
                ) from error
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            current_descriptor = next_descriptor
        if current_descriptor != root_descriptor:
            os.close(root_descriptor)
        return current_descriptor, relative.name
    except BaseException:
        if current_descriptor != root_descriptor:
            os.close(current_descriptor)
        os.close(root_descriptor)
        raise


def _read_regular_file_at(
    parent_descriptor: int, name: str, *, label: str
) -> bytes | None:
    flags = (
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise FullSearchNullError(f"{label} is missing or unsafe") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FullSearchNullError(f"{label} is not an ordinary file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_once_bytes(path: Path, encoded: bytes, *, root: Path) -> None:
    """Atomically create durable bytes beneath a no-follow locked root."""

    parent_descriptor, target_name = _open_safe_artifact_parent(
        root=root, path=path
    )
    try:
        existing = _read_regular_file_at(
            parent_descriptor, target_name, label="artifact target"
        )
        if existing is not None:
            if existing != encoded:
                raise FullSearchNullError(f"artifact already differs: {path}")
            return
        namespace = hashlib.sha256(
            "\0".join((
                os.uname().nodename,
                os.environ.get("SLURM_JOB_ID", "no-job"),
                os.environ.get("SLURM_ARRAY_TASK_ID", "no-task"),
                str(os.getpid()),
            )).encode("utf-8")
        ).hexdigest()[:16]
        try:
            os.mkdir(
                ATOMIC_TEMP_DIRECTORY, mode=0o770,
                dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except FileExistsError:
            pass
        except OSError as error:
            raise FullSearchNullError(
                "cannot create the atomic temporary namespace"
            ) from error
        directory_flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            temporary_descriptor = os.open(
                ATOMIC_TEMP_DIRECTORY,
                directory_flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise FullSearchNullError(
                "atomic temporary namespace is a symlink or not a directory"
            ) from error
        basename = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
        descriptor: int | None = None
        temporary_name: str | None = None
        try:
            for _ in range(8):
                candidate = (
                    f"{basename}.{namespace}.{secrets.token_hex(8)}.tmp"
                )
                try:
                    descriptor = os.open(
                        candidate,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                        | getattr(os, "O_CLOEXEC", 0),
                        0o660,
                        dir_fd=temporary_descriptor,
                    )
                except FileExistsError:
                    continue
                temporary_name = candidate
                break
            if descriptor is None or temporary_name is None:
                raise FullSearchNullError(
                    f"cannot allocate a unique temporary artifact for {path}"
                )
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise FullSearchNullError(
                            f"short write while creating {path}"
                        )
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
                descriptor = None
            try:
                os.link(
                    temporary_name,
                    target_name,
                    src_dir_fd=temporary_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise
                concurrent = _read_regular_file_at(
                    parent_descriptor,
                    target_name,
                    label="concurrent artifact target",
                )
                if concurrent != encoded:
                    raise FullSearchNullError(
                        f"concurrent artifact differs: {path}"
                    ) from error
            os.fsync(parent_descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=temporary_descriptor)
                except FileNotFoundError:
                    pass
            os.close(temporary_descriptor)
    finally:
        os.close(parent_descriptor)


def _write_once_json(
    path: Path, value: Mapping[str, Any], *, root: Path
) -> None:
    """Atomically create a durable JSON file; reconcile identical retries."""

    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    _write_once_bytes(path, encoded, root=root)




def _verify_callback_identity(
    identity: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    controller_program: Mapping[str, Any],
    selection_sha256: str,
) -> dict[str, Any]:
    identity = _mapping(identity, "callback identity")
    fields = {
        "schema_version", "protocol", "callback_id", "callback_version",
        "implementation_sha256", "selection_program_sha256",
        "supported_model_triplet", "supported_falsifier_ids",
        "development_only", "final_connectivity_accessed",
        "nested_null_searches_supported",
    }
    if set(identity) != fields:
        raise FullSearchNullError("callback identity fields differ from protocol")
    if identity["schema_version"] != CALLBACK_IDENTITY_SCHEMA or identity["protocol"] != CALLBACK_PROTOCOL:
        raise FullSearchNullError("callback protocol is unsupported")
    if not isinstance(identity["callback_id"], str) or not identity["callback_id"]:
        raise FullSearchNullError("callback ID is missing")
    if not isinstance(identity["callback_version"], str) or not identity["callback_version"]:
        raise FullSearchNullError("callback version is missing")
    _sha(identity["implementation_sha256"], "callback implementation")
    if identity["selection_program_sha256"] != _sha(selection_sha256, "selection program"):
        raise FullSearchNullError("callback selection program differs from procedure")
    if tuple(identity["supported_model_triplet"]) != policy.model_triplet:
        raise FullSearchNullError("callback does not support the frozen T/U/M triplet")
    scheduled_falsifiers = set(map(
        str, controller_program["countable_falsifier_ids"]
    ))
    if not scheduled_falsifiers <= set(map(str, identity["supported_falsifier_ids"])):
        raise FullSearchNullError("callback does not support every scheduled falsifier")
    if not (
        identity["development_only"] is True
        and identity["final_connectivity_accessed"] is False
        and identity["nested_null_searches_supported"] is False
    ):
        raise FullSearchNullError("callback identity violates development-only execution")
    return _json_copy(identity)


@runtime_checkable
class FullSearchFitScoreCallback(Protocol):
    """Interface the real EP12 fitter/selector adapter must implement."""

    def identity(self) -> Mapping[str, Any]:
        ...

    def fit_score(
        self,
        *,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...

    def finalize_search(
        self,
        *,
        sample: FittedNullSample,
        schedule: Mapping[str, Any],
        trial_results: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


def freeze_procedure_lock(
    *,
    policy: EpisodePolicy,
    controller_program: Mapping[str, Any],
    design: FittedNullDesign,
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
    expansion_plan: TypePooledExpansionPlan,
    seed_manifest: Mapping[str, Any],
    selection_sha256: str,
    development_role_manifest_sha256: str,
    callback_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind all inputs required before the first null search can execute."""

    program = verify_null_controller_program(controller_program, policy=policy)
    selection_sha256 = _sha(selection_sha256, "selection program")
    role_hash = _sha(development_role_manifest_sha256, "development role manifest")
    binding = null_seed_binding(
        design, generators, expansion_plan, selection_sha256=selection_sha256
    )
    verify_null_seed_manifest(seed_manifest, expected_binding=binding)
    _, generator_hash = fitted_generator_manifest(generators, design)
    identity = _verify_callback_identity(
        callback_identity,
        policy=policy,
        controller_program=program,
        selection_sha256=selection_sha256,
    )
    core = {
        "schema_version": PROCEDURE_LOCK_SCHEMA,
        "state": "complete",
        "policy_sha256": policy.policy_hash,
        "development_role_manifest_sha256": role_hash,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
        "full_search_null_required": True,
        "null_replicates": NULL_REPLICATE_COUNT,
        "initial_ledger_each_replicate": "empty",
        "nested_null_searches": 0,
        "controller_program_sha256": program["program_sha256"],
        "minimum_trial_count": policy.minimum_valid_trials,
        "maximum_trial_count": policy.maximum_valid_trials,
        "design_sha256": design.design_sha256,
        "generator_manifest_sha256": generator_hash,
        "expansion_plan_sha256": binding["expansion_plan_sha256"],
        "seed_manifest_sha256": _digest(seed_manifest),
        "selection_sha256": selection_sha256,
        "callback_identity": identity,
        "callback_identity_sha256": _digest(identity),
    }
    return {**core, "procedure_lock_sha256": _digest(core)}


@dataclass(frozen=True)
class FullSearchNullRuntime:
    """Inputs returned by a trusted, development-only runtime provider."""

    design: FittedNullDesign
    generators: Mapping[tuple[str, str], FittedPredictiveModel]
    expansion_plan: TypePooledExpansionPlan
    seed_manifest: Mapping[str, Any]
    selection_sha256: str
    development_role_manifest_sha256: str
    controller_program: Mapping[str, Any]
    procedure_lock: Mapping[str, Any]
    callback: FullSearchFitScoreCallback
    durable_output_root: Path | None = None


def _verify_runtime(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
) -> tuple[tuple[int, ...], dict[str, Any], dict[str, Any]]:
    if not isinstance(runtime, FullSearchNullRuntime):
        raise FullSearchNullError("provider did not return FullSearchNullRuntime")
    try:
        program = verify_null_controller_program(
            runtime.controller_program, policy=policy
        )
    except NullControllerError as error:
        raise FullSearchNullError(str(error)) from error
    lock = _mapping(runtime.procedure_lock, "procedure lock")
    fields = {
        "schema_version", "state", "policy_sha256",
        "development_role_manifest_sha256", "development_only",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "full_search_null_required", "null_replicates",
        "initial_ledger_each_replicate", "nested_null_searches",
        "controller_program_sha256",
        "minimum_trial_count", "maximum_trial_count", "design_sha256",
        "generator_manifest_sha256", "expansion_plan_sha256",
        "seed_manifest_sha256",
        "selection_sha256", "callback_identity",
        "callback_identity_sha256", "procedure_lock_sha256",
    }
    if not fields.issubset(lock):
        raise FullSearchNullError("procedure lock is incomplete")
    if lock["schema_version"] != PROCEDURE_LOCK_SCHEMA or lock["state"] != "complete":
        raise FullSearchNullError("procedure lock is incomplete")
    if lock["procedure_lock_sha256"] != _self_hash(lock, "procedure_lock_sha256"):
        raise FullSearchNullError("procedure lock self-hash mismatch")
    if lock["policy_sha256"] != policy.policy_hash:
        raise FullSearchNullError("procedure lock policy hash mismatch")
    if not (
        lock["development_only"] is True
        and lock["final_connectivity_accessed"] is False
        and lock["final_connectivity_access_authorized"] is False
        and lock["full_search_null_required"] is True
        and int(lock["null_replicates"]) == NULL_REPLICATE_COUNT
        and lock["initial_ledger_each_replicate"] == "empty"
        and int(lock["nested_null_searches"]) == 0
    ):
        raise FullSearchNullError("procedure lock does not authorize this null")
    if lock["development_role_manifest_sha256"] != _sha(
        runtime.development_role_manifest_sha256, "development role manifest"
    ):
        raise FullSearchNullError("runtime role manifest differs from lock")
    if not (
        lock["controller_program_sha256"] == program["program_sha256"]
        and int(lock["minimum_trial_count"]) == policy.minimum_valid_trials
        and int(lock["maximum_trial_count"]) == policy.maximum_valid_trials
        and lock["design_sha256"] == runtime.design.design_sha256
        and lock["seed_manifest_sha256"] == _digest(runtime.seed_manifest)
        and lock["selection_sha256"] == runtime.selection_sha256
    ):
        raise FullSearchNullError("runtime program, design, seed, or selection differs from lock")
    if runtime.expansion_plan.design_sha256 != runtime.design.design_sha256:
        raise FullSearchNullError("expansion plan belongs to another design")
    binding = null_seed_binding(
        runtime.design,
        runtime.generators,
        runtime.expansion_plan,
        selection_sha256=runtime.selection_sha256,
    )
    seeds = verify_null_seed_manifest(runtime.seed_manifest, expected_binding=binding)
    _, generator_hash = fitted_generator_manifest(runtime.generators, runtime.design)
    if generator_hash != lock["generator_manifest_sha256"]:
        raise FullSearchNullError("fitted generators differ from procedure lock")
    if binding["expansion_plan_sha256"] != lock["expansion_plan_sha256"]:
        raise FullSearchNullError("raw expansion plan differs from procedure lock")
    if not isinstance(runtime.callback, FullSearchFitScoreCallback):
        raise FullSearchNullError("callback does not implement the required interface")
    identity = _verify_callback_identity(
        runtime.callback.identity(),
        policy=policy,
        controller_program=program,
        selection_sha256=runtime.selection_sha256,
    )
    if identity != lock["callback_identity"] or _digest(identity) != lock["callback_identity_sha256"]:
        raise FullSearchNullError("runtime callback differs from procedure lock")
    if identity["callback_id"] == PRODUCTION_CALLBACK_ID:
        if runtime.durable_output_root is None:
            raise FullSearchNullError(
                "production runtime lacks its validated durable output root"
            )
        durable_root = _canonical_output_path(
            runtime.durable_output_root,
            label="full-search durable output root",
        )
        try:
            relative = durable_root.relative_to(
                _EPISODE_OUTPUTS_ROOT.resolve(strict=False)
            )
        except ValueError as error:
            raise FullSearchNullError(
                "production durable output root is outside EP12 Outputs"
            ) from error
        if not relative.parts:
            raise FullSearchNullError(
                "production durable output root may not equal EP12 Outputs"
            )
    return seeds, program, identity


def freeze_fit_score_result(
    *,
    replicate_index: int,
    replicate_id: str,
    trial: Mapping[str, Any],
    objective: Mapping[str, Any],
    executed_complete_T_U_M_fit_count: int,
    fit_score_artifact_path: str,
    fit_score_artifact_sha256: str,
) -> dict[str, Any]:
    normalized_objective = normalize_objective(objective)
    statistic = float(normalized_objective["bilateral"])
    if not math.isfinite(statistic):
        raise FullSearchNullError("fit/score statistic is nonfinite")
    complete_fit_count = int(executed_complete_T_U_M_fit_count)
    if complete_fit_count < 1:
        raise FullSearchNullError("executed complete T/U/M fit count must be positive")
    core = {
        "schema_version": FIT_RESULT_SCHEMA,
        "callback_protocol": CALLBACK_PROTOCOL,
        "replicate_index": int(replicate_index),
        "replicate_id": str(replicate_id),
        "valid_trial_ordinal": int(trial["valid_trial_ordinal"]),
        "trial_id": _trial_id(trial["trial_id"]),
        "configuration_sha256": str(trial["configuration_sha256"]),
        "status": "completed",
        "valid_complete_T_U_M_triplet": True,
        "model_triplet": list(MODEL_TRIPLET),
        "executed_complete_T_U_M_fit_count": complete_fit_count,
        "executed_model_family_fit_count": complete_fit_count * len(MODEL_TRIPLET),
        "objective": normalized_objective,
        "selection_statistic": statistic,
        "fit_score_artifact_path": str(fit_score_artifact_path),
        "fit_score_artifact_sha256": _sha(
            fit_score_artifact_sha256, "fit/score artifact"
        ),
        "falsifier_id": trial["falsifier_id"],
        "promotion_eligible": trial["promotion_eligible"],
        "development_only": True,
        "final_connectivity_accessed": False,
        "nested_null_searches": 0,
    }
    return {**core, "result_sha256": _digest(core)}


def freeze_final_result(
    *,
    replicate_index: int,
    replicate_id: str,
    executed_trial_count: int,
    selected_trial_id: str,
    selected_configuration_sha256: str,
    selected_statistic: float,
    selection_program_sha256: str,
    final_selection_artifact_path: str,
    final_selection_artifact_sha256: str,
) -> dict[str, Any]:
    statistic = float(selected_statistic)
    if not math.isfinite(statistic):
        raise FullSearchNullError("final selected statistic is nonfinite")
    core = {
        "schema_version": FINAL_RESULT_SCHEMA,
        "callback_protocol": CALLBACK_PROTOCOL,
        "replicate_index": int(replicate_index),
        "replicate_id": str(replicate_id),
        "status": "completed",
        "executed_trial_count": int(executed_trial_count),
        "selected_trial_id": _trial_id(selected_trial_id, "selected trial ID"),
        "selected_configuration_sha256": _sha(
            selected_configuration_sha256, "selected configuration"
        ),
        "selected_statistic": statistic,
        "selection_program_sha256": _sha(
            selection_program_sha256, "selection program"
        ),
        "final_selection_artifact_path": str(final_selection_artifact_path),
        "final_selection_artifact_sha256": _sha(
            final_selection_artifact_sha256, "final selection artifact"
        ),
        "complete_controller_rerun": True,
        "initial_ledger_empty": True,
        "development_only": True,
        "final_connectivity_accessed": False,
        "nested_null_searches": 0,
    }
    return {**core, "result_sha256": _digest(core)}


def _verify_fit_result(
    result: Mapping[str, Any],
    *,
    replicate_index: int,
    replicate_id: str,
    trial: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    result = _mapping(result, "fit/score callback result")
    fields = {
        "schema_version", "callback_protocol", "replicate_index",
        "replicate_id", "valid_trial_ordinal", "trial_id",
        "configuration_sha256", "status", "valid_complete_T_U_M_triplet",
        "model_triplet", "objective", "selection_statistic",
        "executed_complete_T_U_M_fit_count", "executed_model_family_fit_count",
        "fit_score_artifact_path", "fit_score_artifact_sha256",
        "falsifier_id", "promotion_eligible", "development_only",
        "final_connectivity_accessed", "nested_null_searches", "result_sha256",
    }
    if set(result) != fields or result["result_sha256"] != _self_hash(result, "result_sha256"):
        raise FullSearchNullError("fit/score result schema or hash mismatch")
    checks = (
        result["schema_version"] == FIT_RESULT_SCHEMA,
        result["callback_protocol"] == CALLBACK_PROTOCOL,
        int(result["replicate_index"]) == replicate_index,
        result["replicate_id"] == replicate_id,
        int(result["valid_trial_ordinal"]) == int(trial["valid_trial_ordinal"]),
        result["trial_id"] == trial["trial_id"],
        result["configuration_sha256"] == trial["configuration_sha256"],
        result["status"] == "completed",
        result["valid_complete_T_U_M_triplet"] is True,
        tuple(result["model_triplet"]) == MODEL_TRIPLET,
        int(result["executed_complete_T_U_M_fit_count"]) >= 1,
        int(result["executed_model_family_fit_count"])
        == int(result["executed_complete_T_U_M_fit_count"]) * len(MODEL_TRIPLET),
        result["falsifier_id"] == trial["falsifier_id"],
        result["promotion_eligible"] is trial["promotion_eligible"],
        result["development_only"] is True,
        result["final_connectivity_accessed"] is False,
        int(result["nested_null_searches"]) == 0,
    )
    objective = normalize_objective(result["objective"])
    if (
        not all(checks)
        or not math.isfinite(float(result["selection_statistic"]))
        or float(result["selection_statistic"]) != float(objective["bilateral"])
    ):
        raise FullSearchNullError("fit/score result does not match its invocation")
    expected_hash = _sha(result["fit_score_artifact_sha256"], "fit/score artifact")
    artifact_relative = Path(str(result["fit_score_artifact_path"]))
    root = artifact_root.resolve()
    candidate = root / artifact_relative
    resolved = candidate.resolve()
    if (
        artifact_relative.is_absolute()
        or not artifact_relative.parts
        or ".." in artifact_relative.parts
        or candidate.absolute() != resolved
        or root not in resolved.parents
        or not resolved.is_file()
        or hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_hash
    ):
        raise FullSearchNullError("fit/score callback artifact is missing or changed")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullSearchNullError("fit/score callback artifact is unreadable") from error
    artifact_core = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    if not (
        artifact.get("artifact_sha256") == _digest(artifact_core)
        and artifact.get("replicate_index") == replicate_index
        and artifact.get("replicate_id") == replicate_id
        and artifact.get("trial_id") == trial["trial_id"]
        and artifact.get("configuration_sha256") == trial["configuration_sha256"]
        and artifact.get("objective") == objective
        and artifact.get("complete_T_U_M_result") is True
        and int(artifact.get("executed_complete_T_U_M_fit_count", 0))
        == int(result["executed_complete_T_U_M_fit_count"])
        and int(artifact.get("executed_model_family_fit_count", 0))
        == int(result["executed_model_family_fit_count"])
        and artifact.get("development_only") is True
        and artifact.get("final_connectivity_accessed") is False
    ):
        raise FullSearchNullError("fit/score callback artifact does not authenticate its receipt")
    return _json_copy(result)


def _verify_final_result(
    result: Mapping[str, Any],
    *,
    replicate_index: int,
    replicate_id: str,
    schedule: Mapping[str, Any],
    trial_results: Sequence[Mapping[str, Any]],
    selection_sha256: str,
    artifact_root: Path,
) -> dict[str, Any]:
    result = _mapping(result, "callback final result")
    fields = {
        "schema_version", "callback_protocol", "replicate_index",
        "replicate_id", "status", "executed_trial_count",
        "selected_trial_id", "selected_configuration_sha256",
        "selected_statistic", "selection_program_sha256",
        "final_selection_artifact_path", "final_selection_artifact_sha256",
        "complete_controller_rerun", "initial_ledger_empty",
        "development_only", "final_connectivity_accessed",
        "nested_null_searches", "result_sha256",
    }
    if set(result) != fields or result["result_sha256"] != _self_hash(result, "result_sha256"):
        raise FullSearchNullError("callback final result schema or hash mismatch")
    checks = (
        result["schema_version"] == FINAL_RESULT_SCHEMA,
        result["callback_protocol"] == CALLBACK_PROTOCOL,
        int(result["replicate_index"]) == replicate_index,
        result["replicate_id"] == replicate_id,
        result["status"] == "completed",
        int(result["executed_trial_count"]) == len(schedule["trials"]),
        result["selection_program_sha256"] == selection_sha256,
        result["complete_controller_rerun"] is True,
        result["initial_ledger_empty"] is True,
        result["development_only"] is True,
        result["final_connectivity_accessed"] is False,
        int(result["nested_null_searches"]) == 0,
    )
    if not all(checks):
        raise FullSearchNullError("callback final result does not describe a complete rerun")
    by_id = {
        _trial_id(trial["trial_id"], "realized trial ID"): (trial, receipt)
        for trial, receipt in zip(schedule["trials"], trial_results)
    }
    selected = by_id.get(_trial_id(result["selected_trial_id"], "selected trial ID"))
    if selected is None or selected[0]["promotion_eligible"] is not True:
        raise FullSearchNullError("callback selected an ineligible or unknown trial")
    if (
        result["selected_configuration_sha256"] != selected[0]["configuration_sha256"]
        or float(result["selected_statistic"]) != float(selected[1]["selection_statistic"])
    ):
        raise FullSearchNullError("callback selection is inconsistent with trial results")
    expected_hash = _sha(
        result["final_selection_artifact_sha256"], "final selection artifact"
    )
    artifact_relative = Path(str(result["final_selection_artifact_path"]))
    root = artifact_root.resolve()
    candidate = root / artifact_relative
    resolved = candidate.resolve()
    if (
        artifact_relative.is_absolute()
        or not artifact_relative.parts
        or ".." in artifact_relative.parts
        or candidate.absolute() != resolved
        or root not in resolved.parents
        or not resolved.is_file()
        or hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_hash
    ):
        raise FullSearchNullError("final selection artifact is missing or changed")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullSearchNullError("final selection artifact is unreadable") from error
    artifact_core = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    if not (
        artifact.get("schema_version") == FINAL_ARTIFACT_SCHEMA
        and artifact.get("artifact_sha256") == _digest(artifact_core)
        and int(artifact.get("replicate_index", -1)) == replicate_index
        and artifact.get("replicate_id") == replicate_id
        and int(artifact.get("executed_trial_count", -1)) == len(schedule["trials"])
        and artifact.get("realized_schedule_sha256") == schedule["schedule_sha256"]
        and artifact.get("selected_trial_id") == result["selected_trial_id"]
        and artifact.get("selected_configuration_sha256")
        == result["selected_configuration_sha256"]
        and float(artifact.get("selected_statistic", float("nan")))
        == float(result["selected_statistic"])
        and artifact.get("selection_program_sha256") == selection_sha256
        and artifact.get("complete_controller_rerun") is True
        and artifact.get("initial_ledger_empty") is True
        and artifact.get("development_only") is True
        and artifact.get("final_connectivity_accessed") is False
    ):
        raise FullSearchNullError(
            "final selection artifact does not authenticate its receipt"
        )
    return _json_copy(result)




def _sample_receipt(sample: FittedNullSample) -> dict[str, Any]:
    core = {
        "replicate_index": sample.replicate_index,
        "replicate_id": sample.replicate_id,
        "replicate_seed": sample.replicate_seed,
        "design_sha256": sample.design_sha256,
        "generator_manifest_sha256": sample.generator_manifest_sha256,
        "raw_counts_sha256": _array_sha(sample.raw_counts),
        "collapsed_counts_sha256": _array_sha(sample.collapsed_counts),
        "covariates_sha256": _array_sha(sample.covariates),
        "fallback_allocations_with_positive_mass": sample.fallback_allocations_with_positive_mass,
    }
    return {**core, "sample_receipt_sha256": _digest(core)}




def _adaptive_run_manifest(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "policy_sha256": policy.policy_hash,
        "procedure_lock_sha256": runtime.procedure_lock["procedure_lock_sha256"],
        "controller_program_sha256": runtime.controller_program["program_sha256"],
        "minimum_trial_count": policy.minimum_valid_trials,
        "maximum_trial_count": policy.maximum_valid_trials,
        "design_sha256": runtime.design.design_sha256,
        "generator_manifest_sha256": runtime.procedure_lock[
            "generator_manifest_sha256"
        ],
        "seed_manifest_sha256": _digest(runtime.seed_manifest),
        "callback_identity_sha256": _digest(identity),
        "replicate_count": NULL_REPLICATE_COUNT,
        "initial_ledger_each_replicate": "empty",
        "observed_schedule_replayed": False,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "run_manifest_sha256": _digest(core)}


def _replicate_dir(output_dir: Path, replicate_index: int) -> Path:
    return output_dir / "replicates" / f"null_{replicate_index + 1:03d}"


def _step_path(output_dir: Path, replicate_index: int, ordinal: int) -> Path:
    return _replicate_dir(output_dir, replicate_index) / "steps" / f"step_{ordinal:03d}.json"


def _adaptive_checkpoint_path(output_dir: Path, replicate_index: int) -> Path:
    return _replicate_dir(output_dir, replicate_index) / "checkpoint.json"


def _wall_budget_exhaustion_path(
    output_dir: Path, replicate_index: int
) -> Path:
    return _replicate_dir(output_dir, replicate_index) / (
        "wall_budget_exhaustion.json"
    )


def _durable_wall_budget_exhaustion_path(
    runtime: FullSearchNullRuntime, replicate_index: int
) -> Path | None:
    if runtime.durable_output_root is None:
        return None
    durable_root = _canonical_output_path(
        runtime.durable_output_root,
        label="full-search durable output root",
    )
    outputs_root = _EPISODE_OUTPUTS_ROOT.resolve(strict=False)
    try:
        relative = durable_root.relative_to(outputs_root)
    except ValueError as error:
        raise FullSearchNullError(
            "durable wall-exhaustion receipt is outside EP12 Outputs"
        ) from error
    if not relative.parts:
        raise FullSearchNullError(
            "durable wall-exhaustion receipt may not equal EP12 Outputs"
        )
    return durable_root.with_name(
        f"{durable_root.name}_incomplete_searches"
    ) / f"null_{replicate_index + 1:03d}" / "wall_budget_exhaustion.json"


def _overall_wall_seconds(policy: EpisodePolicy) -> float:
    seconds = float(policy.raw["budgets"]["wall_clock_hour_ceiling"]) * 3600.0
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise FullSearchNullError("overall wall-clock budget is invalid")
    return seconds


def _persisted_trial_wall_seconds(
    steps: Sequence[Mapping[str, Any]],
) -> float:
    values = [float(step["trial_wall_seconds"]) for step in steps]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise FullSearchNullError("persisted trial wall time is invalid")
    return math.fsum(values)


def _verify_wall_budget_exhaustion_receipt(
    raw: Mapping[str, Any],
    *,
    runtime: FullSearchNullRuntime,
    policy: EpisodePolicy,
    replicate_index: int,
    replicate_id: str,
    replicate_seed: int,
    steps: Sequence[Mapping[str, Any]],
    sample_receipt_sha256: str,
) -> dict[str, Any]:
    receipt = _mapping(raw, "wall-budget exhaustion receipt")
    fields = {
        "schema_version", "terminal_status", "exhaustion_reason", "trigger",
        "replicate_index", "replicate_id", "replicate_seed",
        "procedure_lock_sha256", "controller_program_sha256",
        "sample_receipt_sha256", "executed_trial_count",
        "step_checkpoint_sha256", "controller_status_at_boundary",
        "cumulative_persisted_trial_wall_seconds",
        "overall_wall_seconds_limit", "per_trial_wall_seconds_limit",
        "remaining_cumulative_wall_seconds_at_boundary",
        "callback_timeout_seconds", "next_proposal_sha256",
        "partial_callback_result_persisted", "logical_search_complete",
        "finalization_performed", "aggregate_eligible", "development_only",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "receipt_sha256",
    }
    if set(receipt) != fields or receipt["receipt_sha256"] != _self_hash(
        receipt, "receipt_sha256"
    ):
        raise FullSearchNullError(
            "wall-budget exhaustion receipt schema or hash mismatch"
        )
    cumulative = _persisted_trial_wall_seconds(steps)
    overall_limit = _overall_wall_seconds(policy)
    per_trial_limit = policy.wall_hours_per_trial_maximum * 3600.0
    remaining = max(0.0, overall_limit - cumulative)
    status = stop_status(
        program=runtime.controller_program, policy=policy, steps=steps
    )
    trigger = receipt["trigger"]
    timeout_seconds = float(receipt["callback_timeout_seconds"])
    proposal_sha256: str | None = None
    trigger_valid = False
    if trigger == "authenticated_boundary_before_next_trial":
        trigger_valid = (
            remaining == 0.0
            and status["reason"] != "valid_trials_max_reached"
            and timeout_seconds == 0.0
            and receipt["next_proposal_sha256"] is None
        )
    elif trigger == "callback_timeout_at_cumulative_limit":
        if status["action"] == "continue" and 0.0 < remaining <= per_trial_limit:
            proposal = propose_next_trial(
                program=runtime.controller_program,
                policy=policy,
                prior_steps=steps,
            )
            proposal_sha256 = str(proposal["proposal_sha256"])
            trigger_valid = (
                math.isclose(timeout_seconds, remaining)
                and receipt["next_proposal_sha256"] == proposal_sha256
            )
    if not (
        receipt["schema_version"] == WALL_BUDGET_EXHAUSTION_SCHEMA
        and receipt["terminal_status"] == "incomplete_search"
        and receipt["exhaustion_reason"] == "wall_clock_budget_exhausted"
        and trigger_valid
        and int(receipt["replicate_index"]) == replicate_index
        and receipt["replicate_id"] == replicate_id
        and int(receipt["replicate_seed"]) == int(replicate_seed)
        and receipt["procedure_lock_sha256"]
        == runtime.procedure_lock["procedure_lock_sha256"]
        and receipt["controller_program_sha256"]
        == runtime.controller_program["program_sha256"]
        and receipt["sample_receipt_sha256"] == sample_receipt_sha256
        and int(receipt["executed_trial_count"]) == len(steps)
        and receipt["step_checkpoint_sha256"]
        == [step["step_sha256"] for step in steps]
        and receipt["controller_status_at_boundary"] == status
        and math.isclose(
            float(receipt["cumulative_persisted_trial_wall_seconds"]),
            cumulative,
        )
        and math.isclose(
            float(receipt["overall_wall_seconds_limit"]), overall_limit
        )
        and math.isclose(
            float(receipt["per_trial_wall_seconds_limit"]), per_trial_limit
        )
        and math.isclose(
            float(receipt["remaining_cumulative_wall_seconds_at_boundary"]),
            remaining,
        )
        and receipt["partial_callback_result_persisted"] is False
        and receipt["logical_search_complete"] is False
        and receipt["finalization_performed"] is False
        and receipt["aggregate_eligible"] is False
        and receipt["development_only"] is True
        and receipt["final_connectivity_accessed"] is False
        and receipt["final_connectivity_access_authorized"] is False
    ):
        raise FullSearchNullError(
            "wall-budget exhaustion receipt differs from authenticated steps"
        )
    durable_path = _durable_wall_budget_exhaustion_path(
        runtime, replicate_index
    )
    if durable_path is not None:
        if durable_path.is_symlink() or not durable_path.is_file():
            raise FullSearchNullError(
                "durable wall-budget exhaustion receipt is missing or unsafe"
            )
        try:
            durable_receipt = json.loads(
                durable_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise FullSearchNullError(
                "durable wall-budget exhaustion receipt is unreadable"
            ) from error
        if durable_receipt != receipt:
            raise FullSearchNullError(
                "durable wall-budget exhaustion receipt differs from Scratch"
            )
    return _json_copy(receipt)


def _record_wall_budget_exhaustion(
    *,
    runtime: FullSearchNullRuntime,
    policy: EpisodePolicy,
    output_dir: Path,
    replicate_index: int,
    replicate_id: str,
    replicate_seed: int,
    steps: Sequence[Mapping[str, Any]],
    sample_receipt_sha256: str,
    controller_status: Mapping[str, Any],
    trigger: str,
    callback_timeout_seconds: float,
    next_proposal_sha256: str | None,
) -> dict[str, Any]:
    cumulative = _persisted_trial_wall_seconds(steps)
    overall_limit = _overall_wall_seconds(policy)
    core = {
        "schema_version": WALL_BUDGET_EXHAUSTION_SCHEMA,
        "terminal_status": "incomplete_search",
        "exhaustion_reason": "wall_clock_budget_exhausted",
        "trigger": trigger,
        "replicate_index": replicate_index,
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "procedure_lock_sha256": runtime.procedure_lock[
            "procedure_lock_sha256"
        ],
        "controller_program_sha256": runtime.controller_program[
            "program_sha256"
        ],
        "sample_receipt_sha256": sample_receipt_sha256,
        "executed_trial_count": len(steps),
        "step_checkpoint_sha256": [step["step_sha256"] for step in steps],
        "controller_status_at_boundary": _json_copy(controller_status),
        "cumulative_persisted_trial_wall_seconds": cumulative,
        "overall_wall_seconds_limit": overall_limit,
        "per_trial_wall_seconds_limit": (
            policy.wall_hours_per_trial_maximum * 3600.0
        ),
        "remaining_cumulative_wall_seconds_at_boundary": max(
            0.0, overall_limit - cumulative
        ),
        "callback_timeout_seconds": float(callback_timeout_seconds),
        "next_proposal_sha256": next_proposal_sha256,
        "partial_callback_result_persisted": False,
        "logical_search_complete": False,
        "finalization_performed": False,
        "aggregate_eligible": False,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    receipt = {**core, "receipt_sha256": _digest(core)}
    path = _wall_budget_exhaustion_path(output_dir, replicate_index)
    _write_once_json(path, receipt, root=output_dir)
    durable_path = _durable_wall_budget_exhaustion_path(
        runtime, replicate_index
    )
    if durable_path is not None:
        _write_once_json(
            durable_path,
            receipt,
            root=_EPISODE_OUTPUTS_ROOT.resolve(strict=False),
        )
    return _verify_wall_budget_exhaustion_receipt(
        receipt,
        runtime=runtime,
        policy=policy,
        replicate_index=replicate_index,
        replicate_id=replicate_id,
        replicate_seed=replicate_seed,
        steps=steps,
        sample_receipt_sha256=sample_receipt_sha256,
    )


def _relative_inventory_path(value: Any, *, label: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FullSearchNullError(f"{label} is not a safe relative artifact path")
    normalized = path.as_posix()
    if normalized.startswith("./") or normalized == ".":
        raise FullSearchNullError(f"{label} is not canonical")
    return normalized


def _ordinary_file_inventory(root: Path) -> dict[str, str]:
    """Hash ordinary files while validating the isolated temp namespace."""

    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise FullSearchNullError("aggregate artifact tree may not contain symlinks")
        relative_path = path.relative_to(root)
        if ATOMIC_TEMP_DIRECTORY in relative_path.parts:
            index = relative_path.parts.index(ATOMIC_TEMP_DIRECTORY)
            remainder = relative_path.parts[index + 1:]
            if not remainder:
                if not path.is_dir():
                    raise FullSearchNullError("atomic temporary namespace is not a directory")
                continue
            if (
                len(remainder) != 1
                or not path.is_file()
                or _ATOMIC_TEMP_FILE_RE.fullmatch(path.name) is None
            ):
                raise FullSearchNullError("atomic temporary namespace contains unsafe content")
            continue
        if path.is_file():
            actual[relative_path.as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return dict(sorted(actual.items()))


def _aggregate_file_inventory(
    output_dir: Path,
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Inventory the exact authenticated aggregate closure, excluding itself."""

    expected: set[str] = {"run_manifest.json", "full_search_null_report.json"}

    def add(relative: str, *, label: str) -> None:
        normalized = _relative_inventory_path(relative, label=label)
        if normalized in expected:
            raise FullSearchNullError(f"duplicate aggregate artifact path: {normalized}")
        expected.add(normalized)

    for index, checkpoint in enumerate(checkpoints):
        prefix = f"replicates/null_{index + 1:03d}"
        add(f"{prefix}/sample_receipt.json", label="sample receipt path")
        add(f"{prefix}/checkpoint.json", label="replicate checkpoint path")
        results = checkpoint.get("trial_results")
        if not isinstance(results, list) or len(results) != int(
            checkpoint.get("executed_trial_count", -1)
        ):
            raise FullSearchNullError("checkpoint trial inventory is incomplete")
        for ordinal, result in enumerate(results, start=1):
            add(
                f"{prefix}/steps/step_{ordinal:03d}.json",
                label="step checkpoint path",
            )
            add(
                str(result["fit_score_artifact_path"]),
                label="fit/score callback artifact path",
            )
        final = _mapping(checkpoint.get("final_result"), "final result")
        add(
            str(final["final_selection_artifact_path"]),
            label="final selection artifact path",
        )

    actual = _ordinary_file_inventory(output_dir)
    actual.pop("manifest.json", None)
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        unexpected = sorted(set(actual) - expected)
        raise FullSearchNullError(
            "aggregate artifact inventory mismatch; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return dict(sorted(actual.items()))


def _copy_authenticated_aggregate_tree(
    *, source: Path, destination: Path, manifest: Mapping[str, Any]
) -> None:
    """Reconcile only manifest-listed bytes into a resumable staging tree."""

    claimed = manifest.get("manifest_sha256")
    core = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    files = manifest.get("files")
    if not (
        manifest.get("schema_version") == AGGREGATE_MANIFEST_SCHEMA
        and isinstance(claimed, str)
        and claimed == _digest(core)
        and isinstance(files, Mapping)
    ):
        raise FullSearchNullError("aggregate manifest is invalid")
    expected: dict[str, str] = {}
    for relative, expected_sha256 in files.items():
        normalized = _relative_inventory_path(
            relative, label="aggregate manifest path"
        )
        expected[normalized] = _sha(expected_sha256, normalized)
    source_inventory = _ordinary_file_inventory(source)
    manifest_file_sha256 = hashlib.sha256(
        (source / "manifest.json").read_bytes()
    ).hexdigest()
    expected_source = {**expected, "manifest.json": manifest_file_sha256}
    if source_inventory != dict(sorted(expected_source.items())):
        raise FullSearchNullError("source aggregate differs from its manifest")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise FullSearchNullError("aggregate staging target is unsafe")
    else:
        destination.mkdir(parents=False)
    destination_inventory = _ordinary_file_inventory(destination)
    unexpected = set(destination_inventory) - set(expected_source)
    if unexpected:
        raise FullSearchNullError(
            f"aggregate staging target contains unexpected files: {sorted(unexpected)[:3]}"
        )
    for relative in [*sorted(expected), "manifest.json"]:
        source_path = source / relative
        if source_path.is_symlink() or not source_path.is_file():
            raise FullSearchNullError("manifest-listed source artifact is unsafe")
        encoded = source_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != expected_source[relative]:
            raise FullSearchNullError("manifest-listed source artifact changed")
        _write_once_bytes(destination / relative, encoded, root=destination)


def _callback_context(
    *,
    runtime: FullSearchNullRuntime,
    sample: FittedNullSample,
) -> Mapping[str, Any]:
    return MappingProxyType({
        "callback_protocol": CALLBACK_PROTOCOL,
        "procedure_lock_sha256": runtime.procedure_lock["procedure_lock_sha256"],
        "controller_program_sha256": runtime.controller_program["program_sha256"],
        "replicate_index": sample.replicate_index,
        "replicate_id": sample.replicate_id,
        "replicate_seed": sample.replicate_seed,
        "sample_receipt_sha256": _sample_receipt(sample)[
            "sample_receipt_sha256"
        ],
        "initial_ledger_empty": True,
        "nested_null_searches": 0,
        "development_only": True,
        "final_connectivity_accessed": False,
    })


def _verify_adaptive_step(
    raw: Mapping[str, Any],
    *,
    runtime: FullSearchNullRuntime,
    policy: EpisodePolicy,
    replicate_index: int,
    replicate_id: str,
    replicate_seed: int,
    prior_steps: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    sample_receipt_sha256: str,
) -> dict[str, Any]:
    step = _mapping(raw, "adaptive null step checkpoint")
    fields = {
        "schema_version", "complete", "replicate_index", "replicate_id",
        "replicate_seed", "procedure_lock_sha256", "controller_program_sha256",
        "valid_trial_ordinal", "previous_step_sha256", "proposal", "fit_result",
        "disposition", "state_after", "sample_receipt_sha256",
        "trial_wall_seconds", "development_only",
        "final_connectivity_accessed", "step_sha256",
    }
    if set(step) != fields or step["step_sha256"] != _self_hash(step, "step_sha256"):
        raise FullSearchNullError("adaptive step schema or self-hash mismatch")
    ordinal = len(prior_steps) + 1
    previous = prior_steps[-1]["step_sha256"] if prior_steps else None
    if not (
        step["schema_version"] == STEP_CHECKPOINT_SCHEMA
        and step["complete"] is True
        and int(step["replicate_index"]) == replicate_index
        and step["replicate_id"] == replicate_id
        and int(step["replicate_seed"]) == int(replicate_seed)
        and step["procedure_lock_sha256"]
        == runtime.procedure_lock["procedure_lock_sha256"]
        and step["controller_program_sha256"]
        == runtime.controller_program["program_sha256"]
        and step["sample_receipt_sha256"]
        == sample_receipt_sha256
        and math.isfinite(float(step["trial_wall_seconds"]))
        and 0.0 <= float(step["trial_wall_seconds"])
        <= policy.wall_hours_per_trial_maximum * 3600.0
        and int(step["valid_trial_ordinal"]) == ordinal
        and step["previous_step_sha256"] == previous
        and step["development_only"] is True
        and step["final_connectivity_accessed"] is False
    ):
        raise FullSearchNullError("adaptive step differs from locked replicate")
    try:
        proposal = propose_next_trial(
            program=runtime.controller_program,
            policy=policy,
            prior_steps=prior_steps,
        )
    except NullControllerError as error:
        raise FullSearchNullError(str(error)) from error
    if step["proposal"] != proposal:
        raise FullSearchNullError("adaptive proposal does not replay from prior null outcomes")
    _trial_id(proposal["trial_id"], "adaptive proposal trial ID")
    _sha(proposal["configuration_sha256"], "adaptive proposal configuration")
    fit = _verify_fit_result(
        step["fit_result"],
        replicate_index=replicate_index,
        replicate_id=replicate_id,
        trial=proposal,
        artifact_root=artifact_root,
    )
    disposition = disposition_for_result(
        proposal=proposal,
        objective=fit["objective"],
        prior_steps=prior_steps,
    )
    if step["disposition"] != disposition:
        raise FullSearchNullError("adaptive promotion does not replay")
    temporary = {
        **{key: value for key, value in step.items() if key != "state_after"},
        "state_after": {},
    }
    expected_state = stop_status(
        program=runtime.controller_program,
        policy=policy,
        steps=[*prior_steps, temporary],
    )
    if step["state_after"] != expected_state:
        raise FullSearchNullError("adaptive stop arithmetic does not replay")
    return _json_copy(step)


def _load_adaptive_steps(
    *,
    runtime: FullSearchNullRuntime,
    policy: EpisodePolicy,
    output_dir: Path,
    replicate_index: int,
    replicate_id: str,
    replicate_seed: int,
    sample_receipt_sha256: str,
) -> list[dict[str, Any]]:
    directory = _replicate_dir(output_dir, replicate_index) / "steps"
    if not directory.exists():
        return []
    paths = sorted(directory.glob("step_*.json"))
    steps: list[dict[str, Any]] = []
    for ordinal, path in enumerate(paths, start=1):
        if path.name != f"step_{ordinal:03d}.json" or path.is_symlink():
            raise FullSearchNullError("adaptive step checkpoint set is noncontiguous")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FullSearchNullError("adaptive step checkpoint is unreadable") from error
        steps.append(_verify_adaptive_step(
            raw,
            runtime=runtime,
            policy=policy,
            replicate_index=replicate_index,
            replicate_id=replicate_id,
            replicate_seed=replicate_seed,
            prior_steps=steps,
            artifact_root=output_dir,
            sample_receipt_sha256=sample_receipt_sha256,
        ))
    return steps


def _verify_adaptive_checkpoint(
    raw: Mapping[str, Any],
    *,
    runtime: FullSearchNullRuntime,
    policy: EpisodePolicy,
    output_dir: Path,
    replicate_index: int,
    seeds: Sequence[int],
    sample_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _mapping(raw, "adaptive replicate checkpoint")
    fields = {
        "schema_version", "complete", "replicate_index", "replicate_id",
        "replicate_seed", "procedure_lock_sha256", "controller_program_sha256",
        "executed_trial_count", "step_checkpoint_sha256", "realized_schedule",
        "realized_schedule_sha256", "stop_reason", "initial_ledger_empty",
        "observed_schedule_replayed", "nested_null_searches", "sample",
        "trial_results", "final_result", "development_only",
        "final_connectivity_accessed", "checkpoint_sha256",
    }
    if set(checkpoint) != fields or checkpoint["checkpoint_sha256"] != _self_hash(
        checkpoint, "checkpoint_sha256"
    ):
        raise FullSearchNullError("adaptive replicate checkpoint schema or hash mismatch")
    replicate_id = f"null_{replicate_index + 1:03d}"
    seed = int(seeds[replicate_index])
    steps = _load_adaptive_steps(
        runtime=runtime,
        policy=policy,
        output_dir=output_dir,
        replicate_index=replicate_index,
        replicate_id=replicate_id,
        replicate_seed=seed,
        sample_receipt_sha256=str(sample_receipt["sample_receipt_sha256"]),
    )
    schedule = freeze_realized_null_schedule(
        program=runtime.controller_program,
        policy=policy,
        steps=steps,
    )
    if not (
        checkpoint["schema_version"] == CHECKPOINT_SCHEMA
        and checkpoint["complete"] is True
        and int(checkpoint["replicate_index"]) == replicate_index
        and checkpoint["replicate_id"] == replicate_id
        and int(checkpoint["replicate_seed"]) == seed
        and checkpoint["procedure_lock_sha256"]
        == runtime.procedure_lock["procedure_lock_sha256"]
        and checkpoint["controller_program_sha256"]
        == runtime.controller_program["program_sha256"]
        and int(checkpoint["executed_trial_count"]) == len(steps)
        and checkpoint["step_checkpoint_sha256"]
        == [step["step_sha256"] for step in steps]
        and checkpoint["realized_schedule"] == schedule
        and checkpoint["realized_schedule_sha256"] == schedule["schedule_sha256"]
        and checkpoint["stop_reason"] == schedule["stop"]["reason"]
        and checkpoint["initial_ledger_empty"] is True
        and checkpoint["observed_schedule_replayed"] is False
        and int(checkpoint["nested_null_searches"]) == 0
        and checkpoint["development_only"] is True
        and checkpoint["final_connectivity_accessed"] is False
    ):
        raise FullSearchNullError("adaptive replicate checkpoint differs from execution")
    results = [step["fit_result"] for step in steps]
    if checkpoint["trial_results"] != results:
        raise FullSearchNullError("adaptive checkpoint trial results changed")
    _verify_final_result(
        checkpoint["final_result"],
        replicate_index=replicate_index,
        replicate_id=replicate_id,
        schedule=schedule,
        trial_results=results,
        selection_sha256=runtime.selection_sha256,
        artifact_root=output_dir,
    )
    sample = _mapping(checkpoint["sample"], "sample receipt")
    if sample != sample_receipt:
        raise FullSearchNullError("sample receipt differs from locked generator")
    return _json_copy(checkpoint)


def authenticate_null_replicate_checkpoint(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
    replicate_index: int,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Read-only authentication for scheduler recovery checkpoint decisions."""

    if isinstance(replicate_index, bool) or not isinstance(replicate_index, int):
        raise FullSearchNullError("replicate index must be an integer")
    if not 0 <= replicate_index < NULL_REPLICATE_COUNT:
        raise FullSearchNullError("replicate index is outside the frozen 99 searches")
    seeds, _, identity = _verify_runtime(runtime, policy=policy)
    output_dir = _production_output_path(runtime, output_dir, aggregate=True)
    expected_run = _adaptive_run_manifest(runtime, policy=policy, identity=identity)
    run_path = output_dir / "run_manifest.json"
    if (
        not run_path.is_file()
        or run_path.is_symlink()
        or json.loads(run_path.read_text(encoding="utf-8")) != expected_run
    ):
        raise FullSearchNullError("run manifest is missing or mismatched")
    checkpoint_path = _adaptive_checkpoint_path(output_dir, replicate_index)
    if _wall_budget_exhaustion_path(output_dir, replicate_index).exists():
        raise FullSearchNullError(
            "wall-budget-exhausted replicate is incomplete and has no "
            "authenticatable completed checkpoint"
        )
    if not checkpoint_path.exists():
        return None
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise FullSearchNullError("replicate checkpoint path is unsafe")
    sample = sample_fitted_null(
        runtime.design,
        runtime.generators,
        runtime.expansion_plan,
        seed_manifest=runtime.seed_manifest,
        replicate_index=replicate_index,
        selection_sha256=runtime.selection_sha256,
    )
    sample_receipt = _sample_receipt(sample)
    sample_path = _replicate_dir(output_dir, replicate_index) / "sample_receipt.json"
    if (
        not sample_path.is_file()
        or sample_path.is_symlink()
        or json.loads(sample_path.read_text(encoding="utf-8")) != sample_receipt
    ):
        raise FullSearchNullError("sample receipt differs from locked generator")
    try:
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullSearchNullError("replicate checkpoint is unreadable") from error
    return _verify_adaptive_checkpoint(
        raw,
        runtime=runtime,
        policy=policy,
        output_dir=output_dir,
        replicate_index=replicate_index,
        seeds=seeds,
        sample_receipt=sample_receipt,
    )


def run_null_replicate(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
    replicate_index: int,
    output_dir: Path,
    restart_sentinel: Path | None = None,
) -> dict[str, Any]:
    """Execute/resume one null-outcome-driven controller from an empty ledger."""

    if isinstance(replicate_index, bool) or not isinstance(replicate_index, int):
        raise FullSearchNullError("replicate index must be an integer")
    if not 0 <= replicate_index < NULL_REPLICATE_COUNT:
        raise FullSearchNullError("replicate index is outside the frozen 99 searches")
    seeds, program, identity = _verify_runtime(runtime, policy=policy)
    output_dir = _production_output_path(runtime, output_dir, aggregate=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    restart_sentinel_path = (
        _restart_sentinel_path(
            restart_sentinel,
            output_dir=output_dir,
            production=identity["callback_id"] == PRODUCTION_CALLBACK_ID,
        )
        if restart_sentinel is not None
        else None
    )
    _write_once_json(
        output_dir / "run_manifest.json",
        _adaptive_run_manifest(runtime, policy=policy, identity=identity),
        root=output_dir,
    )
    sample = sample_fitted_null(
        runtime.design,
        runtime.generators,
        runtime.expansion_plan,
        seed_manifest=runtime.seed_manifest,
        replicate_index=replicate_index,
        selection_sha256=runtime.selection_sha256,
    )
    sample_receipt = _sample_receipt(sample)
    _write_once_json(
        _replicate_dir(output_dir, replicate_index) / "sample_receipt.json",
        sample_receipt,
        root=output_dir,
    )
    checkpoint_path = _adaptive_checkpoint_path(output_dir, replicate_index)
    exhaustion_path = _wall_budget_exhaustion_path(
        output_dir, replicate_index
    )
    if checkpoint_path.is_file():
        if exhaustion_path.exists():
            raise FullSearchNullError(
                "replicate has both a completed checkpoint and a wall-budget "
                "exhaustion receipt"
            )
        return _verify_adaptive_checkpoint(
            json.loads(checkpoint_path.read_text(encoding="utf-8")),
            runtime=runtime,
            policy=policy,
            output_dir=output_dir,
            replicate_index=replicate_index,
            seeds=seeds,
            sample_receipt=sample_receipt,
        )
    context = _callback_context(runtime=runtime, sample=sample)
    steps = _load_adaptive_steps(
        runtime=runtime,
        policy=policy,
        output_dir=output_dir,
        replicate_index=replicate_index,
        replicate_id=sample.replicate_id,
        replicate_seed=sample.replicate_seed,
        sample_receipt_sha256=str(sample_receipt["sample_receipt_sha256"]),
    )
    if exhaustion_path.exists():
        if exhaustion_path.is_symlink() or not exhaustion_path.is_file():
            raise FullSearchNullError(
                "wall-budget exhaustion receipt path is unsafe"
            )
        try:
            exhaustion = json.loads(
                exhaustion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise FullSearchNullError(
                "wall-budget exhaustion receipt is unreadable"
            ) from error
        _verify_wall_budget_exhaustion_receipt(
            exhaustion,
            runtime=runtime,
            policy=policy,
            replicate_index=replicate_index,
            replicate_id=sample.replicate_id,
            replicate_seed=sample.replicate_seed,
            steps=steps,
            sample_receipt_sha256=str(
                sample_receipt["sample_receipt_sha256"]
            ),
        )
        raise FullSearchNullError(
            "null replicate is an incomplete search after cumulative wall-budget "
            "exhaustion"
        )
    while True:
        status = stop_status(program=program, policy=policy, steps=steps)
        if (
            status["action"] == "stop"
            and status["reason"] == "valid_trials_max_reached"
        ):
            break
        cumulative_wall_seconds = _persisted_trial_wall_seconds(steps)
        overall_wall_seconds = _overall_wall_seconds(policy)
        remaining_wall_seconds = max(
            0.0, overall_wall_seconds - cumulative_wall_seconds
        )
        if remaining_wall_seconds == 0.0:
            _record_wall_budget_exhaustion(
                runtime=runtime,
                policy=policy,
                output_dir=output_dir,
                replicate_index=replicate_index,
                replicate_id=sample.replicate_id,
                replicate_seed=sample.replicate_seed,
                steps=steps,
                sample_receipt_sha256=str(
                    sample_receipt["sample_receipt_sha256"]
                ),
                controller_status=status,
                trigger="authenticated_boundary_before_next_trial",
                callback_timeout_seconds=0.0,
                next_proposal_sha256=None,
            )
            raise FullSearchNullError(
                "null replicate exhausted the cumulative wall budget before "
                "completing its search"
            )
        if status["action"] == "stop":
            break
        if _restart_requested(restart_sentinel_path):
            raise FullSearchNullRequeueRequested(
                replicate_index=replicate_index,
                completed_trial_count=len(steps),
            )
        proposal = propose_next_trial(
            program=program, policy=policy, prior_steps=steps
        )
        _trial_id(proposal["trial_id"], "adaptive proposal trial ID")
        _sha(proposal["configuration_sha256"], "adaptive proposal configuration")
        per_trial_wall_seconds = policy.wall_hours_per_trial_maximum * 3600.0
        callback_timeout_seconds = min(
            per_trial_wall_seconds, remaining_wall_seconds
        )
        try:
            raw_fit, trial_wall_seconds = _invoke_fit_score(
                runtime.callback,
                sample=sample,
                trial=proposal,
                context=context,
                timeout_seconds=callback_timeout_seconds,
                isolate=identity["callback_id"] == PRODUCTION_CALLBACK_ID,
            )
        except FullSearchNullTrialTimeout as error:
            if remaining_wall_seconds <= per_trial_wall_seconds:
                _record_wall_budget_exhaustion(
                    runtime=runtime,
                    policy=policy,
                    output_dir=output_dir,
                    replicate_index=replicate_index,
                    replicate_id=sample.replicate_id,
                    replicate_seed=sample.replicate_seed,
                    steps=steps,
                    sample_receipt_sha256=str(
                        sample_receipt["sample_receipt_sha256"]
                    ),
                    controller_status=status,
                    trigger="callback_timeout_at_cumulative_limit",
                    callback_timeout_seconds=callback_timeout_seconds,
                    next_proposal_sha256=str(proposal["proposal_sha256"]),
                )
                raise FullSearchNullError(
                    "null replicate exhausted the cumulative wall budget during "
                    "its next logical trial"
                ) from error
            raise
        fit = _verify_fit_result(
            raw_fit,
            replicate_index=replicate_index,
            replicate_id=sample.replicate_id,
            trial=proposal,
            artifact_root=output_dir,
        )
        disposition = disposition_for_result(
            proposal=proposal,
            objective=fit["objective"],
            prior_steps=steps,
        )
        partial = {
            "schema_version": STEP_CHECKPOINT_SCHEMA,
            "complete": True,
            "replicate_index": replicate_index,
            "replicate_id": sample.replicate_id,
            "replicate_seed": sample.replicate_seed,
            "procedure_lock_sha256": runtime.procedure_lock[
                "procedure_lock_sha256"
            ],
            "controller_program_sha256": program["program_sha256"],
            "valid_trial_ordinal": int(proposal["valid_trial_ordinal"]),
            "previous_step_sha256": (
                steps[-1]["step_sha256"] if steps else None
            ),
            "proposal": proposal,
            "fit_result": fit,
            "disposition": disposition,
            "state_after": {},
            "sample_receipt_sha256": sample_receipt[
                "sample_receipt_sha256"
            ],
            "trial_wall_seconds": trial_wall_seconds,
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        state_after = stop_status(
            program=program, policy=policy, steps=[*steps, partial]
        )
        core = {**partial, "state_after": state_after}
        step = {**core, "step_sha256": _digest(core)}
        _write_once_json(
            _step_path(output_dir, replicate_index, len(steps) + 1),
            step,
            root=output_dir,
        )
        steps.append(step)
    schedule = freeze_realized_null_schedule(
        program=program, policy=policy, steps=steps
    )
    results = [step["fit_result"] for step in steps]
    final = _verify_final_result(
        runtime.callback.finalize_search(
            sample=sample,
            schedule=schedule,
            trial_results=tuple(results),
            context=context,
        ),
        replicate_index=replicate_index,
        replicate_id=sample.replicate_id,
        schedule=schedule,
        trial_results=results,
        selection_sha256=runtime.selection_sha256,
        artifact_root=output_dir,
    )
    core = {
        "schema_version": CHECKPOINT_SCHEMA,
        "complete": True,
        "replicate_index": replicate_index,
        "replicate_id": sample.replicate_id,
        "replicate_seed": sample.replicate_seed,
        "procedure_lock_sha256": runtime.procedure_lock["procedure_lock_sha256"],
        "controller_program_sha256": program["program_sha256"],
        "executed_trial_count": len(steps),
        "step_checkpoint_sha256": [step["step_sha256"] for step in steps],
        "realized_schedule": schedule,
        "realized_schedule_sha256": schedule["schedule_sha256"],
        "stop_reason": schedule["stop"]["reason"],
        "initial_ledger_empty": True,
        "observed_schedule_replayed": False,
        "nested_null_searches": 0,
        "sample": sample_receipt,
        "trial_results": results,
        "final_result": final,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    checkpoint = {**core, "checkpoint_sha256": _digest(core)}
    _write_once_json(checkpoint_path, checkpoint, root=output_dir)
    return checkpoint


def aggregate_full_search_null(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate 99 authenticated, independently realized adaptive searches."""

    seeds, _, identity = _verify_runtime(runtime, policy=policy)
    output_dir = _production_output_path(runtime, output_dir, aggregate=True)
    expected_run = _adaptive_run_manifest(runtime, policy=policy, identity=identity)
    run_path = output_dir / "run_manifest.json"
    if not run_path.is_file() or json.loads(run_path.read_text(encoding="utf-8")) != expected_run:
        raise FullSearchNullError("run manifest is missing or mismatched")
    checkpoints: list[dict[str, Any]] = []
    for index in range(NULL_REPLICATE_COUNT):
        if _wall_budget_exhaustion_path(output_dir, index).exists():
            raise FullSearchNullError(
                f"cannot aggregate: replicate {index} exhausted its cumulative "
                "wall budget and is incomplete"
            )
        sample = sample_fitted_null(
            runtime.design,
            runtime.generators,
            runtime.expansion_plan,
            seed_manifest=runtime.seed_manifest,
            replicate_index=index,
            selection_sha256=runtime.selection_sha256,
        )
        sample_receipt = _sample_receipt(sample)
        sample_receipt_path = _replicate_dir(
            output_dir, index
        ) / "sample_receipt.json"
        if (
            not sample_receipt_path.is_file()
            or json.loads(sample_receipt_path.read_text(encoding="utf-8"))
            != sample_receipt
        ):
            raise FullSearchNullError(
                f"cannot aggregate: replicate {index} sample receipt changed"
            )
        path = _adaptive_checkpoint_path(output_dir, index)
        if not path.is_file():
            raise FullSearchNullError(f"cannot aggregate: replicate {index} is incomplete")
        checkpoints.append(_verify_adaptive_checkpoint(
            json.loads(path.read_text(encoding="utf-8")),
            runtime=runtime,
            policy=policy,
            output_dir=output_dir,
            replicate_index=index,
            seeds=seeds,
            sample_receipt=sample_receipt,
        ))
    statistics = [float(value["final_result"]["selected_statistic"]) for value in checkpoints]
    counts = [int(value["executed_trial_count"]) for value in checkpoints]
    complete_fit_count = sum(
        int(result["executed_complete_T_U_M_fit_count"])
        for checkpoint in checkpoints
        for result in checkpoint["trial_results"]
    )
    model_family_fit_count = sum(
        int(result["executed_model_family_fit_count"])
        for checkpoint in checkpoints
        for result in checkpoint["trial_results"]
    )
    trial_wall_seconds = [
        [float(step["trial_wall_seconds"]) for step in _load_adaptive_steps(
            runtime=runtime,
            policy=policy,
            output_dir=output_dir,
            replicate_index=index,
            replicate_id=value["replicate_id"],
            replicate_seed=int(value["replicate_seed"]),
            sample_receipt_sha256=str(value["sample"]["sample_receipt_sha256"]),
        )]
        for index, value in enumerate(checkpoints)
    ]
    if not all(math.isfinite(value) for value in statistics):
        raise FullSearchNullError("the complete null statistic vector is unavailable")
    core = {
        "schema_version": AGGREGATE_SCHEMA,
        "procedure_lock_sha256": runtime.procedure_lock["procedure_lock_sha256"],
        "controller_program_sha256": runtime.controller_program["program_sha256"],
        "realized_trial_count_per_replicate": counts,
        "minimum_realized_trial_count": min(counts),
        "maximum_realized_trial_count": max(counts),
        "replicate_count": NULL_REPLICATE_COUNT,
        "replicate_ids": [value["replicate_id"] for value in checkpoints],
        "replicate_checkpoint_sha256": [value["checkpoint_sha256"] for value in checkpoints],
        "realized_schedule_sha256": [value["realized_schedule_sha256"] for value in checkpoints],
        "null_statistics": statistics,
        "executed_controller_trial_count": sum(counts),
        "executed_complete_T_U_M_fit_count": complete_fit_count,
        "executed_model_family_fit_count": model_family_fit_count,
        "control_branch_fits_included_in_executed_counts": True,
        "trial_wall_seconds_per_replicate": trial_wall_seconds,
        "maximum_trial_wall_seconds": max(
            value for replicate in trial_wall_seconds for value in replicate
        ),
        "per_trial_wall_seconds_limit": policy.wall_hours_per_trial_maximum * 3600.0,
        "all_replicates_initial_ledger_empty": True,
        "all_replicates_complete_controller_rerun": True,
        "all_replicates_recomputed_adaptive_schedule": True,
        "observed_schedule_replayed": False,
        "nested_null_searches": 0,
        "executed_full_search_null_completed": True,
        "monte_carlo_p_computed": False,
        "monte_carlo_p_requires_locked_observed_statistic": True,
        "legacy_synthetic_receipt_controller_commit_used": False,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    report = {**core, "report_sha256": _digest(core)}
    report_path = output_dir / "full_search_null_report.json"
    _write_once_json(report_path, report, root=output_dir)
    manifest_core = {
        "schema_version": AGGREGATE_MANIFEST_SCHEMA,
        "procedure_lock_sha256": runtime.procedure_lock["procedure_lock_sha256"],
        "controller_program_sha256": runtime.controller_program["program_sha256"],
        "report_sha256": report["report_sha256"],
        "files": _aggregate_file_inventory(output_dir, checkpoints),
        "replicate_count": NULL_REPLICATE_COUNT,
        "executed_full_search_null_completed": True,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": _digest(manifest_core),
    }
    _write_once_json(output_dir / "manifest.json", manifest, root=output_dir)
    return report


def run_full_search_null(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
    output_dir: Path,
) -> dict[str, Any]:
    """Sequential reference execution for qualification and tests."""

    for index in range(NULL_REPLICATE_COUNT):
        run_null_replicate(
            runtime, policy=policy, replicate_index=index, output_dir=output_dir
        )
    return aggregate_full_search_null(runtime, policy=policy, output_dir=output_dir)


def stage_authenticated_aggregate(
    runtime: FullSearchNullRuntime,
    *,
    policy: EpisodePolicy,
    source_dir: Path,
    stage_dir: Path,
) -> dict[str, Any]:
    """Copy only authenticated aggregate bytes and verify the staged tree."""

    source = _production_output_path(runtime, source_dir, aggregate=True)
    stage = _production_output_path(runtime, stage_dir, aggregate=True)
    if source == stage or source in stage.parents or stage in source.parents:
        raise FullSearchNullError("aggregate source and staging paths overlap")
    report = aggregate_full_search_null(runtime, policy=policy, output_dir=source)
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FullSearchNullError("source aggregate manifest is missing or unsafe")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullSearchNullError("source aggregate manifest is unreadable") from error
    _copy_authenticated_aggregate_tree(
        source=source, destination=stage, manifest=_mapping(manifest, "manifest")
    )
    staged_report = aggregate_full_search_null(
        runtime, policy=policy, output_dir=stage
    )
    if (
        staged_report != report
        or (stage / "manifest.json").read_bytes() != manifest_path.read_bytes()
    ):
        raise FullSearchNullError("staged aggregate differs from authenticated source")
    return staged_report


def _load_runtime(specification: str) -> FullSearchNullRuntime:
    if specification.count(":") != 1:
        raise FullSearchNullError("runtime provider must use module:attribute")
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise FullSearchNullError("runtime provider import is incomplete")
    provider = getattr(importlib.import_module(module_name), attribute_name)
    runtime = provider() if callable(provider) else provider
    if not isinstance(runtime, FullSearchNullRuntime):
        raise FullSearchNullError(
            "provider must return FullSearchNullRuntime; no fallback loader exists"
        )
    return runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-full-search-null")
    parser.add_argument("--runtime-provider", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restart-sentinel", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replicate-index", type=int)
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--publish-stage", type=Path)
    mode.add_argument("--run-all", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    policy = EpisodePolicy.load(arguments.policy)
    runtime = _load_runtime(arguments.runtime_provider)
    if arguments.restart_sentinel is not None and arguments.replicate_index is None:
        raise FullSearchNullError(
            "restart sentinel is valid only for one replicate array task"
        )
    if arguments.replicate_index is not None:
        try:
            result = run_null_replicate(
                runtime,
                policy=policy,
                replicate_index=arguments.replicate_index,
                output_dir=arguments.output_dir,
                restart_sentinel=arguments.restart_sentinel,
            )
        except FullSearchNullRequeueRequested as request:
            print(json.dumps({
                "mode": "replicate_checkpointed_for_requeue",
                "replicate_index": request.replicate_index,
                "completed_trial_count": request.completed_trial_count,
                "complete": False,
                "finalized": False,
                "final_connectivity_accessed": False,
            }, sort_keys=True))
            return REQUEUE_EXIT_CODE
        summary = {
            "mode": "replicate",
            "replicate_id": result["replicate_id"],
            "complete": result["complete"],
            "executed_trial_count": result["executed_trial_count"],
            "final_connectivity_accessed": False,
        }
    elif arguments.aggregate_only:
        result = aggregate_full_search_null(
            runtime, policy=policy, output_dir=arguments.output_dir
        )
        summary = {
            "mode": "aggregate",
            "replicate_count": result["replicate_count"],
            "executed_full_search_null_completed": True,
            "final_connectivity_accessed": False,
        }
    elif arguments.publish_stage is not None:
        result = stage_authenticated_aggregate(
            runtime,
            policy=policy,
            source_dir=arguments.output_dir,
            stage_dir=arguments.publish_stage,
        )
        summary = {
            "mode": "authenticated_stage",
            "replicate_count": result["replicate_count"],
            "executed_full_search_null_completed": True,
            "final_connectivity_accessed": False,
        }
    else:
        result = run_full_search_null(
            runtime, policy=policy, output_dir=arguments.output_dir
        )
        summary = {
            "mode": "sequential_all",
            "replicate_count": result["replicate_count"],
            "executed_full_search_null_completed": True,
            "final_connectivity_accessed": False,
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
