#!/usr/bin/env python3
"""Shared in-memory 12-LOGO executor for the EP10 v3 M2 core families.

This module is deliberately an adapter, not another numerical implementation.
It delegates all set enumeration, residualization, exact normalization,
lambda-max certification, and optimization to :mod:`fit_v3_m1` and
:mod:`fit_v3_m2`.  It accepts already constructed cells and already fitted,
fold-specific M1 offsets.  Its sole filesystem reader accepts only the frozen,
target-blind identity/group TSV used to seal production execution.  It has no
command-line entry point and cannot open observed, final, unassigned, or
morphology data.

Only the beta-zero reference and non-falsifier lasso, elastic-net, and
bilateral-group-lasso trials are in scope.  Optional families, Rademacher
references, permutations, null replays, persistence, and controller scheduling
remain responsibilities of later adapters.
"""

from __future__ import annotations

import math
import os
import sys
import csv
import hashlib
import threading
import weakref
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


EXPECTED_FOLDS = 12
EXPECTED_PRODUCTION_ROWS = 194
EXPECTED_PRODUCTION_RANK = 104
FROZEN_MAXIMUM_ITERATIONS = 20_000
SEALED_PRODUCTION_MODE = "sealed_production"
SYNTHETIC_TEST_MODE = "synthetic_test"
LN2 = math.log(2.0)
_FROZEN_INVENTORY_SEAL = object()
_PREPARED_CACHE_SEAL = object()
_SUCCESSOR_AUTHORIZATION_SEAL = object()
_EXTERNAL_EXECUTION_RECEIPT_SEAL = object()
_PRODUCTION_CACHE_CLAIM_SEAL = object()
_PRODUCTION_EXECUTION_CAPABILITY_SEAL = object()


class CoreTrialExecutorError(m2.M2Error):
    """A shared core-executor invariant failed closed."""


class ProductionExecutionAuthorityError(CoreTrialExecutorError):
    """A production cache claim or one-shot execution capability was invalid."""


def _make_weak_identity_registry(
    label: str,
) -> tuple[Any, Any]:
    """Return closure-private weak identity registration and lookup functions."""
    records: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}

    def register(instance: Any, metadata: Any) -> None:
        identity = id(instance)

        def remove(reference: weakref.ReferenceType[Any]) -> None:
            current = records.get(identity)
            if current is not None and current[0] is reference:
                records.pop(identity, None)

        records[identity] = (weakref.ref(instance, remove), metadata)

    def lookup(instance: Any) -> Any:
        record = records.get(id(instance))
        if record is None or record[0]() is not instance:
            raise CoreTrialExecutorError(f"{label} was not issued by this executor module")
        return record[1]

    return register, lookup


_register_inventory_identity, _lookup_inventory_identity = (
    _make_weak_identity_registry("frozen execution inventory")
)
_register_cache_identity, _lookup_cache_identity = _make_weak_identity_registry(
    "prepared trial cache"
)
_register_execution_identity, _lookup_execution_identity = (
    _make_weak_identity_registry("trial execution")
)
_register_authorization_identity, _lookup_authorization_identity = (
    _make_weak_identity_registry("successor authorization")
)
_register_external_receipt_identity, _lookup_external_receipt_identity = (
    _make_weak_identity_registry("external execution receipt")
)
_register_production_claim_identity, _lookup_production_claim_identity = (
    _make_weak_identity_registry("production cache claim")
)
_register_production_capability_identity, _lookup_production_capability_identity = (
    _make_weak_identity_registry("production execution capability")
)


@dataclass(frozen=True)
class FoldSpec:
    """One frozen leave-one-conservative-group-out split."""

    ordinal: int
    held_out_group: str
    training_indexes: tuple[int, ...]
    validation_indexes: tuple[int, ...]


@dataclass(frozen=True, init=False)
class FrozenExecutionInventory:
    """Exact episode-local target-blind identity/group production seal."""

    source_path: Path
    entries: tuple[tuple[str, str], ...]
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        source_path: Path,
        entries: tuple[tuple[str, str], ...],
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _FROZEN_INVENTORY_SEAL:
            raise CoreTrialExecutorError(
                "frozen execution inventory must be built from the exact TSV"
            )
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_seal", _seal)

    @classmethod
    def from_tsv(
        cls,
        path: str | os.PathLike[str] = controller.FROZEN_CELL_INVENTORY_PATH,
    ) -> FrozenExecutionInventory:
        """Read only the controller-owned target-blind production TSV."""
        supplied = Path(os.path.abspath(os.fspath(path)))
        expected = Path(os.path.abspath(controller.FROZEN_CELL_INVENTORY_PATH))
        if supplied != expected or not supplied.is_file() or supplied.is_symlink():
            raise CoreTrialExecutorError(
                "production inventory is not the exact episode-local regular TSV"
            )
        try:
            with supplied.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if tuple(reader.fieldnames or ()) != (
                    "cell_key",
                    "conservative_group",
                ):
                    raise CoreTrialExecutorError(
                        "frozen execution inventory header changed"
                    )
                entries = tuple(
                    sorted(
                        (str(row["cell_key"]), str(row["conservative_group"]))
                        for row in reader
                    )
                )
        except (csv.Error, KeyError, OSError, UnicodeError) as error:
            raise CoreTrialExecutorError(
                "frozen execution inventory is unreadable"
            ) from error
        if (
            len(entries) != EXPECTED_PRODUCTION_ROWS
            or entries != controller.FROZEN_CELL_INVENTORY.entries
        ):
            raise CoreTrialExecutorError(
                "frozen execution inventory does not match the controller seal"
            )
        inventory = cls(supplied, entries, _seal=_FROZEN_INVENTORY_SEAL)
        _register_inventory_identity(inventory, (supplied, entries))
        return inventory


def _require_canonical_inventory(
    inventory: FrozenExecutionInventory,
) -> FrozenExecutionInventory:
    """Reauthenticate identity, path, and entries on every production use."""
    if type(inventory) is not FrozenExecutionInventory:
        raise CoreTrialExecutorError(
            "sealed production requires a FrozenExecutionInventory"
        )
    issued_path, issued_entries = _lookup_inventory_identity(inventory)
    expected = Path(os.path.abspath(controller.FROZEN_CELL_INVENTORY_PATH))
    source = Path(os.path.abspath(os.fspath(inventory.source_path)))
    try:
        exact_regular_file = (
            source == expected
            and issued_path == expected
            and source.is_file()
            and not source.is_symlink()
            and source.resolve(strict=True) == expected.resolve(strict=True)
        )
    except OSError as error:
        raise CoreTrialExecutorError(
            "production inventory path cannot be revalidated"
        ) from error
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != (
                "cell_key",
                "conservative_group",
            ):
                raise CoreTrialExecutorError(
                    "production inventory header changed after issuance"
                )
            on_disk_entries = tuple(
                sorted(
                    (str(row["cell_key"]), str(row["conservative_group"]))
                    for row in reader
                )
            )
    except (csv.Error, KeyError, OSError, UnicodeError) as error:
        raise CoreTrialExecutorError(
            "production inventory entries cannot be revalidated"
        ) from error
    canonical_entries = controller.FROZEN_CELL_INVENTORY.entries
    if (
        not exact_regular_file
        or inventory._seal is not _FROZEN_INVENTORY_SEAL
        or inventory.entries != issued_entries
        or inventory.entries != canonical_entries
        or on_disk_entries != canonical_entries
        or len(inventory.entries) != EXPECTED_PRODUCTION_ROWS
    ):
        raise CoreTrialExecutorError(
            "production inventory no longer matches its canonical issuance"
        )
    return inventory


@dataclass(frozen=True)
class FixedM1Offset:
    """An M1 coefficient vector and the training transforms in its coordinates."""

    held_out_group: str
    training_indexes: tuple[int, ...]
    theta: np.ndarray
    design_transform: m1.DesignTransform
    geometry_scales: np.ndarray
    fit_identity: str


@dataclass(frozen=True)
class PreparedFold:
    """Training/validation M2 views sharing one immutable training transform."""

    spec: FoldSpec
    fixed_m1: FixedM1Offset
    training: m2.PreparedM2
    validation: m2.PreparedM2
    baseline_log_probability_nats: np.ndarray
    baseline_pair_probabilities: np.ndarray


@dataclass(frozen=True, init=False)
class PreparedTrialCache:
    """Reusable core inputs shared by every configuration in one outcome view."""

    candidates: m1.CandidateLibrary
    geometry: m1.GeometryEngine
    pairs: m2.PairFeatureLibrary
    folds: tuple[PreparedFold, ...]
    mode: str
    expected_rank: int | None
    cell_inventory: tuple[tuple[str, str], ...]
    execution_inventory: FrozenExecutionInventory | None
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        candidates: m1.CandidateLibrary,
        geometry: m1.GeometryEngine,
        pairs: m2.PairFeatureLibrary,
        folds: tuple[PreparedFold, ...],
        mode: str,
        expected_rank: int | None,
        cell_inventory: tuple[tuple[str, str], ...],
        execution_inventory: FrozenExecutionInventory | None,
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _PREPARED_CACHE_SEAL:
            raise CoreTrialExecutorError(
                "prepared trial caches must be built by prepare_trial_cache"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "expected_rank", expected_rank)
        object.__setattr__(self, "cell_inventory", cell_inventory)
        object.__setattr__(self, "execution_inventory", execution_inventory)
        object.__setattr__(self, "_seal", _seal)


@dataclass(frozen=True)
class PenaltyMapping:
    """Exact controller-to-engine mapping for one fold."""

    family: str
    penalty_kind: str
    l1_ratio: float
    rho: float
    lambda_maximum: m2.LambdaMaximumResult
    specification: m2.PenaltySpec


@dataclass(frozen=True)
class CellTrialResult:
    """One unique outer-cross-fitted held-out prediction."""

    global_index: int
    cell_key: str
    group: str
    k: int
    m1_log_probability_nats: float
    m2_log_probability_nats: float
    pair_probability_contrasts: tuple[float, ...]


@dataclass(frozen=True)
class FoldTrialResult:
    """One completed held-out fold for one controller trial."""

    spec: FoldSpec
    penalty: PenaltyMapping | None
    fit: m2.M2OptimizerResult | None
    gain_bits_per_cell: float
    active_pair_degrees: int
    cells: tuple[CellTrialResult, ...]


@dataclass(frozen=True)
class TrialExecution:
    """The complete 12-fold execution and controller-facing result."""

    trial: controller.Trial
    folds: tuple[FoldTrialResult, ...]
    pair_contrast_rows: tuple[controller.PairContrastRow, ...]
    equal_group_pair_contrasts: tuple[float, ...]
    result: controller.TrialResult


@dataclass(frozen=True, init=False)
class ExternalExecutionReceipt:
    """A validator-issued receipt for a trial executed by another adapter.

    ``execution_detail`` is optional for legacy/synthetic adapters.  Named
    production adapters use it to keep their complete immutable execution in
    the authenticated controller history; a bare ``TrialResult`` is not a
    sufficient production record for those adapters.
    """

    result: controller.TrialResult
    execution_detail: object | None
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        result: controller.TrialResult,
        *,
        execution_detail: object | None = None,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _EXTERNAL_EXECUTION_RECEIPT_SEAL:
            raise CoreTrialExecutorError(
                "external receipts must be issued by authenticate_external_execution"
        )
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "execution_detail", execution_detail)
        object.__setattr__(self, "_seal", _seal)


@dataclass(frozen=True, init=False)
class SuccessorAuthorization:
    """An executor-owned, exact authenticated prefix for one successor."""

    trial: controller.Trial
    prior_executions: tuple[TrialExecution | ExternalExecutionReceipt, ...]
    coverage_evidence: controller.CoverageEvidence
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        trial: controller.Trial,
        prior_executions: tuple[TrialExecution | ExternalExecutionReceipt, ...],
        coverage_evidence: controller.CoverageEvidence,
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _SUCCESSOR_AUTHORIZATION_SEAL:
            raise CoreTrialExecutorError(
                "successor authorizations must be built by authorize_successor"
            )
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "prior_executions", prior_executions)
        object.__setattr__(self, "coverage_evidence", coverage_evidence)
        object.__setattr__(self, "_seal", _seal)


class ProductionCacheClaim:
    """Opaque permanent claim over one exact sealed-production cache."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _PRODUCTION_CACHE_CLAIM_SEAL:
            raise ProductionExecutionAuthorityError(
                "production cache claims require observed-session authority"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise ProductionExecutionAuthorityError("production claims cannot be copied")

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise ProductionExecutionAuthorityError("production claims cannot be copied")

    def __reduce__(self) -> None:
        raise ProductionExecutionAuthorityError(
            "production claims cannot be serialized"
        )


class ProductionExecutionCapability:
    """Opaque one-shot authority for one exact production trial attempt."""

    __slots__ = ("_seal", "__weakref__")

    def __init__(self, *, _seal: object | None = None) -> None:
        if _seal is not _PRODUCTION_EXECUTION_CAPABILITY_SEAL:
            raise ProductionExecutionAuthorityError(
                "production execution capabilities require a claimed session"
            )
        object.__setattr__(self, "_seal", _seal)

    def __copy__(self) -> None:
        raise ProductionExecutionAuthorityError(
            "production execution capabilities cannot be copied"
        )

    def __deepcopy__(self, memo: Any) -> None:
        del memo
        raise ProductionExecutionAuthorityError(
            "production execution capabilities cannot be copied"
        )

    def __reduce__(self) -> None:
        raise ProductionExecutionAuthorityError(
            "production execution capabilities cannot be serialized"
        )


@dataclass
class _ProductionCacheControl:
    cache_reference: weakref.ReferenceType[PreparedTrialCache]
    lock: threading.RLock
    raw_attempts_before_claim: int = 0
    claimed: bool = False
    claim_identity: int | None = None
    claim_reference: weakref.ReferenceType[ProductionCacheClaim] | None = None
    session_reference: weakref.ReferenceType[Any] | None = None
    acknowledged_attempts: int = 0
    current_capability: ProductionExecutionCapability | None = None


@dataclass(frozen=True)
class _ProductionClaimRecord:
    cache_reference: weakref.ReferenceType[PreparedTrialCache]
    owner_reference: weakref.ReferenceType[Any]
    session_reference: weakref.ReferenceType[Any]


@dataclass
class _ProductionCapabilityRecord:
    cache_reference: weakref.ReferenceType[PreparedTrialCache]
    owner_reference: weakref.ReferenceType[Any]
    session_reference: weakref.ReferenceType[Any]
    claim_reference: weakref.ReferenceType[ProductionCacheClaim]
    trial_reference: weakref.ReferenceType[controller.Trial]
    trial_digest: bytes
    authorization_reference: weakref.ReferenceType[SuccessorAuthorization] | None
    authorization_digest: bytes | None
    session_attempt_ordinal: int
    session_version: int
    production_attempt_ordinal: int
    binding_digest: bytes
    phase: str = "issued"
    completion: str | None = None
    execution_reference: weakref.ReferenceType[TrialExecution] | None = None
    execution_digest: bytes | None = None
    error_type: str | None = None
    attempt_reference: weakref.ReferenceType[Any] | None = None
    attempt_digest: bytes | None = None


def _readonly_copy(array: np.ndarray, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise CoreTrialExecutorError(f"{label} is malformed or nonfinite")
    copied = value.copy()
    copied.setflags(write=False)
    return copied


def _deep_integrity_digest(value: Any) -> bytes:
    """Hash a numerical object graph, including array identity, flags, and bytes."""
    digest = hashlib.sha256()
    seen: dict[int, int] = {}

    def visit(item: Any) -> None:
        if item is None:
            digest.update(b"none;")
            return
        if isinstance(item, (bool, int, str, bytes)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b";")
            return
        if isinstance(item, (float, np.floating)):
            number = float(item)
            digest.update(b"float:")
            digest.update(number.hex().encode("ascii"))
            digest.update(b";")
            return
        if isinstance(item, np.integer):
            visit(int(item))
            return
        if isinstance(item, Path):
            digest.update(b"path:")
            digest.update(os.fspath(item).encode("utf-8"))
            digest.update(b";")
            return

        identity = id(item)
        if identity in seen:
            digest.update(f"ref:{seen[identity]};".encode("ascii"))
            return
        ordinal = len(seen)
        seen[identity] = ordinal
        digest.update(f"obj:{ordinal}:{identity}:".encode("ascii"))

        if isinstance(item, np.ndarray):
            array = np.asarray(item)
            digest.update(b"ndarray:")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(b":w" if array.flags.writeable else b":r")
            digest.update(np.ascontiguousarray(array).view(np.uint8))
            digest.update(b";")
            return
        if isinstance(item, Mapping):
            digest.update(type(item).__qualname__.encode("utf-8"))
            ordered = sorted(
                item.items(),
                key=lambda pair: (type(pair[0]).__qualname__, repr(pair[0])),
            )
            for key, nested in ordered:
                visit(key)
                visit(nested)
            digest.update(b";")
            return
        if isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(f":{len(item)}:".encode("ascii"))
            for nested in item:
                visit(nested)
            digest.update(b";")
            return
        if isinstance(item, (set, frozenset)):
            digest.update(type(item).__name__.encode("ascii"))
            for nested in sorted(item, key=lambda nested: repr(nested)):
                visit(nested)
            digest.update(b";")
            return
        if is_dataclass(item) and not isinstance(item, type):
            digest.update(type(item).__module__.encode("utf-8"))
            digest.update(b".")
            digest.update(type(item).__qualname__.encode("utf-8"))
            digest.update(b":")
            for descriptor in fields(item):
                digest.update(descriptor.name.encode("utf-8"))
                digest.update(b"=")
                visit(getattr(item, descriptor.name))
            digest.update(b";")
            return

        state = getattr(item, "__dict__", None)
        if isinstance(state, dict):
            digest.update(type(item).__module__.encode("utf-8"))
            digest.update(b".")
            digest.update(type(item).__qualname__.encode("utf-8"))
            digest.update(b":state=")
            visit(state)
            digest.update(b";")
            return

        # Stateless opaque sentinels are bound by exact identity.
        digest.update(type(item).__module__.encode("utf-8"))
        digest.update(b".")
        digest.update(type(item).__qualname__.encode("utf-8"))
        digest.update(b":")
        digest.update(str(identity).encode("ascii"))
        digest.update(b";")

    visit(value)
    return digest.digest()


def _authority_binding_digest(*components: object) -> bytes:
    """Hash an authority payload without binding transient container identities.

    ``_deep_integrity_digest`` intentionally includes the identity of every
    traversed object.  That is appropriate for live numerical object graphs,
    but not for a temporary tuple assembled while recomputing a capability
    seal.  Authority callers must first reduce live objects to explicit
    identities and integrity digests, then pass only the primitive immutable
    values accepted here.
    """

    digest = hashlib.sha256()

    def visit(component: object) -> None:
        if component is None:
            digest.update(b"n;")
            return
        if type(component) is bool:
            digest.update(b"b:1;" if component else b"b:0;")
            return
        if type(component) is int:
            encoded = str(component).encode("ascii")
            digest.update(b"i:")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)
            digest.update(b";")
            return
        if type(component) is str:
            encoded = component.encode("utf-8")
            digest.update(b"s:")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)
            digest.update(b";")
            return
        if type(component) is bytes:
            digest.update(b"y:")
            digest.update(str(len(component)).encode("ascii"))
            digest.update(b":")
            digest.update(component)
            digest.update(b";")
            return
        if type(component) is tuple:
            digest.update(b"t:")
            digest.update(str(len(component)).encode("ascii"))
            digest.update(b":[")
            for nested in component:
                visit(nested)
            digest.update(b"];")
            return
        raise ProductionExecutionAuthorityError(
            "authority binding contains a nonprimitive component"
        )

    visit(tuple(components))
    return digest.digest()


def _observed_session_request(name: str, *arguments: Any) -> Any:
    module = sys.modules.get("v3_m2_observed_session_v1")
    expected_path = TOOLS_DIR / "v3_m2_observed_session_v1.py"
    if (
        module is None
        or Path(os.path.abspath(os.fspath(getattr(module, "SCRIPT_PATH", ""))))
        != expected_path
    ):
        raise ProductionExecutionAuthorityError(
            "production execution requires the exact observed-session adapter"
        )
    callback = getattr(module, name, None)
    if not callable(callback):
        raise ProductionExecutionAuthorityError(
            "observed-session capability handshake is unavailable"
        )
    try:
        return callback(*arguments)
    except ProductionExecutionAuthorityError:
        raise
    except Exception as error:
        raise ProductionExecutionAuthorityError(
            "observed-session capability handshake failed"
        ) from error


def _make_production_execution_control() -> tuple[Any, ...]:
    controls: dict[int, _ProductionCacheControl] = {}
    controls_lock = threading.RLock()

    def lookup_claim_authority(claim: Any) -> _ProductionClaimRecord:
        if (
            type(claim) is not ProductionCacheClaim
            or getattr(claim, "_seal", None) is not _PRODUCTION_CACHE_CLAIM_SEAL
        ):
            raise ProductionExecutionAuthorityError(
                "production cache claim lacks its exact opaque type or seal"
            )
        try:
            record = _lookup_production_claim_identity(claim)
        except Exception as error:
            raise ProductionExecutionAuthorityError(
                "production cache claim was not issued by this executor"
            ) from error
        if type(record) is not _ProductionClaimRecord:
            raise ProductionExecutionAuthorityError(
                "production cache claim registry metadata is malformed"
            )
        return record

    def lookup_capability_authority(
        capability: Any,
    ) -> _ProductionCapabilityRecord:
        if (
            type(capability) is not ProductionExecutionCapability
            or getattr(capability, "_seal", None)
            is not _PRODUCTION_EXECUTION_CAPABILITY_SEAL
        ):
            raise ProductionExecutionAuthorityError(
                "production capability lacks its exact opaque type or seal"
            )
        try:
            record = _lookup_production_capability_identity(capability)
        except Exception as error:
            raise ProductionExecutionAuthorityError(
                "production capability was not issued by this executor"
            ) from error
        if type(record) is not _ProductionCapabilityRecord:
            raise ProductionExecutionAuthorityError(
                "production capability registry metadata is malformed"
            )
        return record

    def control_for(cache: PreparedTrialCache) -> _ProductionCacheControl:
        identity = id(cache)
        with controls_lock:
            current = controls.get(identity)
            if current is not None and current.cache_reference() is cache:
                return current

            def remove(reference: weakref.ReferenceType[Any]) -> None:
                with controls_lock:
                    observed = controls.get(identity)
                    if observed is not None and observed.cache_reference is reference:
                        controls.pop(identity, None)

            created = _ProductionCacheControl(
                cache_reference=weakref.ref(cache, remove),
                lock=threading.RLock(),
            )
            controls[identity] = created
            return created

    def require_claim_record(
        owner: CoreTrialExecutor,
        session: Any,
        claim: ProductionCacheClaim,
        control: _ProductionCacheControl,
    ) -> _ProductionClaimRecord:
        if (
            type(claim) is not ProductionCacheClaim
            or getattr(claim, "_seal", None) is not _PRODUCTION_CACHE_CLAIM_SEAL
        ):
            raise ProductionExecutionAuthorityError(
                "production execution lacks an exact cache claim"
            )
        record = lookup_claim_authority(claim)
        if (
            type(record) is not _ProductionClaimRecord
            or record.cache_reference() is not owner.prepared
            or record.owner_reference() is not owner
            or record.session_reference() is not session
            or not control.claimed
            or control.claim_identity != id(claim)
            or control.claim_reference is None
            or control.claim_reference() is not claim
            or control.session_reference is None
            or control.session_reference() is not session
        ):
            raise ProductionExecutionAuthorityError(
                "production cache claim owner/session binding changed"
            )
        return record

    def claim_cache(
        owner: CoreTrialExecutor,
        session: Any,
    ) -> ProductionCacheClaim:
        if type(owner) is not CoreTrialExecutor:
            raise ProductionExecutionAuthorityError(
                "production claim owner has the wrong exact type"
            )
        cache = _require_prepared_cache(owner.prepared)
        if cache.mode != SEALED_PRODUCTION_MODE:
            raise ProductionExecutionAuthorityError(
                "only a sealed-production cache can be claimed"
            )
        accepted = _observed_session_request(
            "_executor_claim_request",
            session,
            owner,
        )
        if accepted is not True:
            raise ProductionExecutionAuthorityError(
                "observed session rejected the production cache claim"
            )
        control = control_for(cache)
        with control.lock:
            if control.claimed:
                raise ProductionExecutionAuthorityError(
                    "this production cache was already permanently claimed"
                )
            if control.raw_attempts_before_claim != 0:
                raise ProductionExecutionAuthorityError(
                    "production cache had an execute attempt before session claim"
                )
            claim = ProductionCacheClaim(_seal=_PRODUCTION_CACHE_CLAIM_SEAL)
            record = _ProductionClaimRecord(
                cache_reference=weakref.ref(cache),
                owner_reference=weakref.ref(owner),
                session_reference=weakref.ref(session),
            )
            _register_production_claim_identity(claim, record)
            control.claimed = True
            control.claim_identity = id(claim)
            control.claim_reference = weakref.ref(claim)
            control.session_reference = weakref.ref(session)
            return claim

    def capability_binding(record: _ProductionCapabilityRecord) -> bytes:
        trial = record.trial_reference()
        authorization = (
            None
            if record.authorization_reference is None
            else record.authorization_reference()
        )
        return _authority_binding_digest(
            id(record.cache_reference()),
            id(record.owner_reference()),
            id(record.session_reference()),
            id(record.claim_reference()),
            id(trial),
            None if trial is None else _deep_integrity_digest(trial),
            None if authorization is None else id(authorization),
            (
                None
                if authorization is None
                else _deep_integrity_digest(authorization)
            ),
            record.session_attempt_ordinal,
            record.session_version,
            record.production_attempt_ordinal,
        )

    def issue_capability(
        owner: CoreTrialExecutor,
        session: Any,
        claim: ProductionCacheClaim,
        trial: controller.Trial,
        authorization: SuccessorAuthorization | None,
    ) -> ProductionExecutionCapability:
        if type(owner) is not CoreTrialExecutor:
            raise ProductionExecutionAuthorityError(
                "production capability owner has the wrong exact type"
            )
        cache = _require_prepared_cache(owner.prepared)
        if cache.mode != SEALED_PRODUCTION_MODE or type(trial) is not controller.Trial:
            raise ProductionExecutionAuthorityError(
                "production capability cache or trial is invalid"
            )
        request = _observed_session_request(
            "_executor_capability_request",
            session,
            owner,
            trial,
        )
        if (
            type(request) is not tuple
            or len(request) != 2
            or type(request[0]) is not int
            or type(request[1]) is not int
        ):
            raise ProductionExecutionAuthorityError(
                "observed session returned a malformed capability request"
            )
        session_attempt_ordinal, session_version = request
        control = control_for(cache)
        with control.lock:
            require_claim_record(owner, session, claim, control)
            current = control.current_capability
            if current is not None:
                current_record = lookup_capability_authority(current)
                if current_record.phase != "ledger_acked":
                    raise ProductionExecutionAuthorityError(
                        "the prior production capability is not ledger-acknowledged"
                    )
            if authorization is not None and type(authorization) is not SuccessorAuthorization:
                raise ProductionExecutionAuthorityError(
                    "production capability successor authority has the wrong type"
                )
            capability = ProductionExecutionCapability(
                _seal=_PRODUCTION_EXECUTION_CAPABILITY_SEAL
            )
            record = _ProductionCapabilityRecord(
                cache_reference=weakref.ref(cache),
                owner_reference=weakref.ref(owner),
                session_reference=weakref.ref(session),
                claim_reference=weakref.ref(claim),
                trial_reference=weakref.ref(trial),
                trial_digest=_deep_integrity_digest(trial),
                authorization_reference=(
                    None if authorization is None else weakref.ref(authorization)
                ),
                authorization_digest=(
                    None
                    if authorization is None
                    else _deep_integrity_digest(authorization)
                ),
                session_attempt_ordinal=session_attempt_ordinal,
                session_version=session_version,
                production_attempt_ordinal=control.acknowledged_attempts + 1,
                binding_digest=b"",
            )
            record.binding_digest = capability_binding(record)
            _register_production_capability_identity(capability, record)
            control.current_capability = capability
            return capability

    def begin_attempt(
        owner: CoreTrialExecutor,
        trial: controller.Trial,
        authorization: SuccessorAuthorization | None,
        capability: ProductionExecutionCapability | None,
    ) -> _ProductionCapabilityRecord | None:
        cache = owner.prepared
        if cache.mode != SEALED_PRODUCTION_MODE:
            if capability is not None:
                raise ProductionExecutionAuthorityError(
                    "synthetic execution rejects production capabilities"
                )
            return None
        control = control_for(cache)
        with control.lock:
            if not control.claimed:
                control.raw_attempts_before_claim += 1
                raise ProductionExecutionAuthorityError(
                    "raw production execute is forbidden before session claim"
                )
            if capability is None:
                raise ProductionExecutionAuthorityError(
                    "raw production execute is forbidden on a claimed cache"
                )
            if (
                type(capability) is not ProductionExecutionCapability
                or getattr(capability, "_seal", None)
                is not _PRODUCTION_EXECUTION_CAPABILITY_SEAL
            ):
                raise ProductionExecutionAuthorityError(
                    "production execute capability has the wrong exact type"
                )
            record = lookup_capability_authority(capability)
            expected_authorization = (
                None
                if record.authorization_reference is None
                else record.authorization_reference()
            )
            if (
                control.current_capability is not capability
                or record.phase != "issued"
                or record.cache_reference() is not cache
                or record.owner_reference() is not owner
                or record.trial_reference() is not trial
                or _deep_integrity_digest(trial) != record.trial_digest
                or expected_authorization is not authorization
                or (
                    authorization is not None
                    and _deep_integrity_digest(authorization)
                    != record.authorization_digest
                )
                or capability_binding(record) != record.binding_digest
                or record.production_attempt_ordinal
                != control.acknowledged_attempts + 1
            ):
                raise ProductionExecutionAuthorityError(
                    "production capability binding changed before consumption"
                )
            record.phase = "in_flight"
            return record

    def complete_attempt(
        capability: ProductionExecutionCapability,
        record: _ProductionCapabilityRecord,
        *,
        execution: TrialExecution | None,
        error: BaseException | None,
    ) -> None:
        cache = record.cache_reference()
        if cache is None:
            raise ProductionExecutionAuthorityError(
                "production cache died during an execution attempt"
            )
        control = control_for(cache)
        with control.lock:
            if (
                control.current_capability is not capability
                or lookup_capability_authority(capability) is not record
                or record.phase != "in_flight"
                or capability_binding(record) != record.binding_digest
                or (execution is None) == (error is None)
            ):
                raise ProductionExecutionAuthorityError(
                    "production capability completion is not atomic"
                )
            if execution is not None:
                record.completion = "success"
                record.execution_reference = weakref.ref(execution)
                record.execution_digest = _deep_integrity_digest(execution)
            else:
                assert error is not None
                record.completion = "error"
                record.error_type = f"{type(error).__module__}.{type(error).__qualname__}"
            record.phase = "completed_unacked"

    def acknowledge_attempt(
        owner: CoreTrialExecutor,
        session: Any,
        claim: ProductionCacheClaim,
        capability: ProductionExecutionCapability,
        attempt: Any,
    ) -> tuple[int, bytes]:
        record = lookup_capability_authority(capability)
        cache = record.cache_reference()
        if cache is None:
            raise ProductionExecutionAuthorityError(
                "production cache died before attempt acknowledgment"
            )
        control = control_for(cache)
        with control.lock:
            require_claim_record(owner, session, claim, control)
            accepted = _observed_session_request(
                "_executor_ack_request",
                session,
                owner,
                capability,
                attempt,
                record.binding_digest,
                record.session_attempt_ordinal,
                record.session_version,
            )
            expected_outcomes = (
                {"valid"} if record.completion == "success" else {"invalid", "resource"}
            )
            if (
                accepted is not True
                or control.current_capability is not capability
                or record.phase != "completed_unacked"
                or capability_binding(record) != record.binding_digest
                or record.session_reference() is not session
                or record.claim_reference() is not claim
                or getattr(attempt, "outcome", None) not in expected_outcomes
                or getattr(attempt, "trial", None) is not record.trial_reference()
                or getattr(attempt, "execution_capability", None) is not capability
                or getattr(attempt, "execution_capability_digest", None)
                != record.binding_digest
                or getattr(attempt, "production_attempt_ordinal", None)
                != record.production_attempt_ordinal
            ):
                raise ProductionExecutionAuthorityError(
                    "session ledger did not exactly acknowledge the capability"
                )
            record.attempt_reference = weakref.ref(attempt)
            record.attempt_digest = _deep_integrity_digest(attempt)
            record.phase = "ledger_acked"
            control.acknowledged_attempts += 1
            return record.production_attempt_ordinal, record.binding_digest

    def completed_binding(
        owner: CoreTrialExecutor,
        session: Any,
        claim: ProductionCacheClaim,
        capability: ProductionExecutionCapability,
    ) -> tuple[int, bytes, str, str | None]:
        record = lookup_capability_authority(capability)
        cache = record.cache_reference()
        if cache is None:
            raise ProductionExecutionAuthorityError(
                "production cache died before session ledger commit"
            )
        control = control_for(cache)
        with control.lock:
            require_claim_record(owner, session, claim, control)
            if (
                control.current_capability is not capability
                or record.phase != "completed_unacked"
                or record.completion not in {"success", "error"}
                or capability_binding(record) != record.binding_digest
            ):
                raise ProductionExecutionAuthorityError(
                    "production capability is not ready for ledger acknowledgment"
                )
            return (
                record.production_attempt_ordinal,
                record.binding_digest,
                record.completion,
                record.error_type,
            )

    def require_claim(
        owner: CoreTrialExecutor,
        session: Any,
        claim: ProductionCacheClaim,
        acknowledged_attempts: int,
        *,
        allow_pending: bool,
    ) -> None:
        cache = _require_prepared_cache(owner.prepared)
        control = control_for(cache)
        with control.lock:
            require_claim_record(owner, session, claim, control)
            if control.acknowledged_attempts != acknowledged_attempts:
                raise ProductionExecutionAuthorityError(
                    "production attempt count differs from the session ledger"
                )
            current = control.current_capability
            if current is None:
                return
            record = lookup_capability_authority(current)
            if capability_binding(record) != record.binding_digest:
                raise ProductionExecutionAuthorityError(
                    "production capability binding changed after issuance"
                )
            if record.phase == "ledger_acked":
                return
            if not allow_pending or record.phase not in {
                "issued",
                "in_flight",
                "completed_unacked",
            }:
                raise ProductionExecutionAuthorityError(
                    "production capability remains outside the acknowledged ledger"
                )

    def require_acknowledged(
        owner: CoreTrialExecutor,
        session: Any,
        claim: ProductionCacheClaim,
        capability: ProductionExecutionCapability,
        attempt: Any,
    ) -> tuple[int, bytes]:
        record = lookup_capability_authority(capability)
        cache = record.cache_reference()
        if cache is None:
            raise ProductionExecutionAuthorityError(
                "production capability cache no longer exists"
            )
        control = control_for(cache)
        with control.lock:
            require_claim_record(owner, session, claim, control)
            if (
                record.phase != "ledger_acked"
                or capability_binding(record) != record.binding_digest
                or record.attempt_reference is None
                or record.attempt_reference() is not attempt
                or record.attempt_digest != _deep_integrity_digest(attempt)
                or record.binding_digest
                != getattr(attempt, "execution_capability_digest", None)
            ):
                raise ProductionExecutionAuthorityError(
                    "production capability acknowledgment changed"
                )
            return record.production_attempt_ordinal, record.binding_digest

    return (
        claim_cache,
        issue_capability,
        begin_attempt,
        complete_attempt,
        completed_binding,
        acknowledge_attempt,
        require_claim,
        require_acknowledged,
    )


(
    _claim_production_cache,
    _issue_production_execution_capability,
    _begin_production_execution_attempt,
    _complete_production_execution_attempt,
    _completed_production_execution_binding,
    _acknowledge_production_execution_attempt,
    _require_production_cache_claim,
    _require_acknowledged_production_execution,
) = _make_production_execution_control()


def _register_prepared_cache(cache: PreparedTrialCache) -> PreparedTrialCache:
    _register_cache_identity(cache, _deep_integrity_digest(cache))
    return cache


def _require_prepared_cache(cache: PreparedTrialCache) -> PreparedTrialCache:
    if type(cache) is not PreparedTrialCache or cache._seal is not _PREPARED_CACHE_SEAL:
        raise CoreTrialExecutorError("prepared cache lacks its construction authority")
    expected = _lookup_cache_identity(cache)
    if _deep_integrity_digest(cache) != expected:
        raise CoreTrialExecutorError("prepared cache changed after authenticated preparation")
    if cache.mode == SEALED_PRODUCTION_MODE:
        assert cache.execution_inventory is not None
        _require_canonical_inventory(cache.execution_inventory)
    return cache


def logo_fold_specs(cells: Sequence[m1.Cell]) -> tuple[FoldSpec, ...]:
    """Return the exact 12 LOGO splits in the frozen conservative-group order."""
    rows = tuple(cells)
    if not rows:
        raise CoreTrialExecutorError("12-LOGO preparation requires cells")
    identifiers = tuple(str(cell.identifier) for cell in rows)
    if any(not value for value in identifiers) or len(set(identifiers)) != len(rows):
        raise CoreTrialExecutorError("cell identifiers must be nonempty and unique")
    observed_groups = {str(cell.group) for cell in rows}
    if observed_groups != set(m1.GROUP_ORDER) or len(m1.GROUP_ORDER) != EXPECTED_FOLDS:
        raise CoreTrialExecutorError("cell groups do not match the frozen 12-group order")

    folds: list[FoldSpec] = []
    for ordinal, held_out_group in enumerate(m1.GROUP_ORDER, start=1):
        validation = tuple(
            index for index, cell in enumerate(rows) if cell.group == held_out_group
        )
        training = tuple(
            index for index, cell in enumerate(rows) if cell.group != held_out_group
        )
        if not validation or not training:
            raise CoreTrialExecutorError("a LOGO fold has an empty training or validation set")
        if set(training).intersection(validation) or len(training) + len(validation) != len(rows):
            raise CoreTrialExecutorError("a LOGO fold does not partition the supplied cells")
        folds.append(FoldSpec(ordinal, held_out_group, training, validation))
    if len({fold.training_indexes for fold in folds}) != EXPECTED_FOLDS:
        raise CoreTrialExecutorError("LOGO training sets are not unique")
    return tuple(folds)


def _validate_execution_scope(
    rows: tuple[m1.Cell, ...],
    mode: str,
    execution_inventory: FrozenExecutionInventory | None,
    expected_rank: int | None,
) -> tuple[tuple[str, str], ...]:
    cell_inventory = tuple((str(cell.identifier), str(cell.group)) for cell in rows)
    if mode == SEALED_PRODUCTION_MODE:
        if execution_inventory is None:
            raise CoreTrialExecutorError(
                "sealed production requires a FrozenExecutionInventory"
            )
        _require_canonical_inventory(execution_inventory)
        if (
            isinstance(expected_rank, bool)
            or not isinstance(expected_rank, int)
            or expected_rank != EXPECTED_PRODUCTION_RANK
        ):
            raise CoreTrialExecutorError(
                "sealed production requires expected residual rank 104"
            )
        if (
            len(cell_inventory) != EXPECTED_PRODUCTION_ROWS
            or tuple(sorted(cell_inventory)) != execution_inventory.entries
        ):
            raise CoreTrialExecutorError(
                "production cells do not match the exact frozen identity/group vector"
            )
    elif mode == SYNTHETIC_TEST_MODE:
        if execution_inventory is not None:
            raise CoreTrialExecutorError(
                "synthetic-test execution cannot carry a production inventory seal"
            )
        if expected_rank is not None and (
            isinstance(expected_rank, bool)
            or not isinstance(expected_rank, int)
            or expected_rank <= 0
        ):
            raise CoreTrialExecutorError(
                "synthetic-test expected rank must be a positive integer or None"
            )
    else:
        raise CoreTrialExecutorError("unknown execution mode")
    return cell_inventory


def _validate_offset_binding(offset: FixedM1Offset, spec: FoldSpec) -> None:
    if not isinstance(offset, FixedM1Offset):
        raise CoreTrialExecutorError("fixed M1 offset has the wrong type")
    if offset.held_out_group != spec.held_out_group:
        raise CoreTrialExecutorError("fixed M1 offset is assigned to the wrong fold")
    if (
        not isinstance(offset.training_indexes, tuple)
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in offset.training_indexes
        )
        or offset.training_indexes != spec.training_indexes
    ):
        raise CoreTrialExecutorError(
            "fixed M1 offset is not bound to the exact LOGO training indexes"
        )
    if not isinstance(offset.fit_identity, str) or not offset.fit_identity.strip():
        raise CoreTrialExecutorError("fixed M1 fit identity must be nonempty")


def _validate_offset_inventory(
    offsets: Mapping[str, FixedM1Offset], specs: tuple[FoldSpec, ...]
) -> None:
    if set(offsets) != set(m1.GROUP_ORDER):
        raise CoreTrialExecutorError("fixed M1 offsets do not cover exactly 12 groups")
    fit_identities: list[str] = []
    for spec in specs:
        offset = offsets[spec.held_out_group]
        _validate_offset_binding(offset, spec)
        fit_identities.append(offset.fit_identity)
    if len(set(fit_identities)) != EXPECTED_FOLDS:
        raise CoreTrialExecutorError("fixed M1 fit identities must be unique by fold")


def _validate_fixed_m1_offset(
    offset: FixedM1Offset,
    spec: FoldSpec,
    transform: m2.ResidualTransform,
) -> FixedM1Offset:
    _validate_offset_binding(offset, spec)
    theta = _readonly_copy(
        offset.theta,
        shape=(m1.PARAMETER_DIMENSION,),
        label="fixed M1 theta",
    )
    geometry_scales = _readonly_copy(
        offset.geometry_scales,
        shape=(m1.GEOMETRY_DIMENSION,),
        label="fixed M1 geometry scales",
    )
    design_mean = np.asarray(offset.design_transform.soma_mean, dtype=float)
    design_scale = np.asarray(offset.design_transform.soma_scale, dtype=float)
    if (
        design_mean.shape != (9,)
        or design_scale.shape != (9,)
        or not np.all(np.isfinite(design_mean))
        or not np.all(np.isfinite(design_scale))
        or np.any(design_scale <= 0.0)
    ):
        raise CoreTrialExecutorError("fixed M1 design transform is malformed")
    if not (
        np.array_equal(design_mean, transform.design_transform.soma_mean)
        and np.array_equal(design_scale, transform.design_transform.soma_scale)
        and np.array_equal(geometry_scales, transform.geometry_scales)
    ):
        raise CoreTrialExecutorError(
            "fixed M1 offset coordinates differ from the M2 training transform"
        )
    return FixedM1Offset(
        held_out_group=offset.held_out_group,
        training_indexes=offset.training_indexes,
        theta=theta,
        design_transform=m1.DesignTransform(
            _readonly_copy(design_mean, shape=(9,), label="M1 soma mean"),
            _readonly_copy(design_scale, shape=(9,), label="M1 soma scale"),
        ),
        geometry_scales=geometry_scales,
        fit_identity=offset.fit_identity,
    )


def _pair_probabilities(
    beta: np.ndarray,
    fixed_m1_theta: np.ndarray,
    prepared: m2.PreparedM2,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
) -> np.ndarray:
    result = np.empty((len(prepared.cells), m2.PAIR_DIMENSION), dtype=float)
    for cell_index, cell in enumerate(prepared.cells):
        if cell.k < 2:
            result[cell_index] = 0.0
            continue
        if cell.k == m1.PANEL_SIZE:
            result[cell_index] = 1.0
            continue
        distribution = m2.candidate_distribution(
            beta,
            fixed_m1_theta,
            prepared,
            cell_index,
            candidates,
            geometry,
            pairs,
        )
        probabilities = np.asarray(distribution.probabilities, dtype=float)
        if not math.isclose(
            float(np.sum(probabilities, dtype=float)),
            1.0,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        ):
            raise CoreTrialExecutorError("exact candidate probabilities do not sum to one")
        result[cell_index] = probabilities @ pairs.basis(cell.k).indicators
    if not (
        np.all(np.isfinite(result))
        and np.all(result >= -1.0e-12)
        and np.all(result <= 1.0 + 1.0e-12)
    ):
        raise CoreTrialExecutorError("pair-event probabilities are malformed")
    result.setflags(write=False)
    return result


def prepare_trial_cache(
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
    fixed_m1_offsets: Mapping[str, FixedM1Offset],
    *,
    mode: str = SEALED_PRODUCTION_MODE,
    execution_inventory: FrozenExecutionInventory | None = None,
    expected_rank: int | None = None,
) -> PreparedTrialCache:
    """Fit training-only residual transforms and cache exact M1 held-out offsets."""
    rows = tuple(cells)
    uniform_moments = tuple(moments)
    if len(rows) != len(uniform_moments):
        raise CoreTrialExecutorError("cells and uniform moments are not aligned")
    cell_inventory = _validate_execution_scope(
        rows, mode, execution_inventory, expected_rank
    )
    if geometry.library is not candidates or pairs.candidates is not candidates:
        raise CoreTrialExecutorError("candidate, geometry, and pair libraries are not shared")
    specs = logo_fold_specs(rows)
    _validate_offset_inventory(fixed_m1_offsets, specs)

    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    prepared_folds: list[PreparedFold] = []
    for spec in specs:
        transform, training = m2.fit_residual_transform(
            rows,
            uniform_moments,
            candidates,
            geometry,
            pairs,
            spec.training_indexes,
            expected_rank=expected_rank,
        )
        expected_training_ids = tuple(rows[index].identifier for index in spec.training_indexes)
        if (
            training.transform is not transform
            or tuple(cell.identifier for cell in training.cells) != expected_training_ids
        ):
            raise CoreTrialExecutorError(
                "residual transform did not return the exact LOGO training rows"
            )
        fixed_m1 = _validate_fixed_m1_offset(
            fixed_m1_offsets[spec.held_out_group], spec, transform
        )
        validation_cells = tuple(rows[index] for index in spec.validation_indexes)
        validation_moments = tuple(
            uniform_moments[index] for index in spec.validation_indexes
        )
        validation = m2.apply_residual_transform(
            validation_cells, validation_moments, transform
        )
        if (
            validation.transform is not transform
            or tuple(cell.identifier for cell in validation.cells)
            != tuple(cell.identifier for cell in validation_cells)
        ):
            raise CoreTrialExecutorError(
                "held-out rows were not transformed with the frozen training transform"
            )
        baseline = m2.evaluate_m2(
            zero,
            fixed_m1.theta,
            validation,
            candidates,
            geometry,
            pairs,
            need_predictions=True,
        )
        direct_m1 = m1.evaluate_model(
            fixed_m1.theta,
            validation.m1_data,
            candidates,
            geometry,
            0.0,
            need_hessian=False,
            need_predictions=True,
        )
        if baseline.log_probability_nats is None or direct_m1.log_probability_nats is None:
            raise CoreTrialExecutorError("beta-zero prediction arrays were omitted")
        if not np.array_equal(
            baseline.log_probability_nats, direct_m1.log_probability_nats
        ):
            raise CoreTrialExecutorError("beta zero did not exactly reproduce fixed M1")
        baseline_log_probability = _readonly_copy(
            baseline.log_probability_nats,
            shape=(len(validation_cells),),
            label="baseline held-out log probability",
        )
        baseline_pairs = _pair_probabilities(
            zero,
            fixed_m1.theta,
            validation,
            candidates,
            geometry,
            pairs,
        )
        prepared_folds.append(
            PreparedFold(
                spec=spec,
                fixed_m1=fixed_m1,
                training=training,
                validation=validation,
                baseline_log_probability_nats=baseline_log_probability,
                baseline_pair_probabilities=baseline_pairs,
            )
        )
    return _register_prepared_cache(
        PreparedTrialCache(
        candidates,
        geometry,
        pairs,
        tuple(prepared_folds),
        mode,
        expected_rank,
        cell_inventory,
        execution_inventory,
        _seal=_PREPARED_CACHE_SEAL,
        )
    )


def _validate_lambda_maximum(
    result: m2.LambdaMaximumResult,
) -> m2.LambdaMaximumResult:
    if not isinstance(result, m2.LambdaMaximumResult):
        raise CoreTrialExecutorError("lambda-max result has the wrong type")
    values = (
        float(result.value),
        float(result.lower_bracket),
        float(result.upper_bracket),
        float(result.zero_kkt_residual),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise CoreTrialExecutorError("lambda-max diagnostics are nonfinite or negative")
    value, lower, upper, residual = values
    if (
        isinstance(result.bisections, bool)
        or not isinstance(result.bisections, int)
        or not 0 <= result.bisections <= m2.LAMBDA_MAXIMUM_BISECTIONS
    ):
        raise CoreTrialExecutorError("lambda-max bisection count is invalid")
    if lower > upper or value != upper:
        raise CoreTrialExecutorError("lambda-max bracket is inconsistent")
    if residual > m2.PROX_TOLERANCE:
        raise CoreTrialExecutorError("lambda-max zero-KKT residual exceeds tolerance")
    if value == 0.0:
        if values != (0.0, 0.0, 0.0, 0.0) or result.bisections != 0:
            raise CoreTrialExecutorError("zero lambda-max is not the canonical zero tuple")
    elif upper - lower > m2.LAMBDA_MAXIMUM_RELATIVE_BRACKET_TOLERANCE * upper:
        raise CoreTrialExecutorError("lambda-max bracket exceeds its relative tolerance")
    return result


def _validate_optimizer_result(
    result: m2.M2OptimizerResult,
    maximum_iterations: int,
) -> m2.M2OptimizerResult:
    if not isinstance(result, m2.M2OptimizerResult):
        raise CoreTrialExecutorError("M2 optimizer result has the wrong type")
    beta = np.asarray(result.beta, dtype=float)
    if beta.shape != (m2.PAIR_DIMENSION,) or not np.all(np.isfinite(beta)):
        raise CoreTrialExecutorError("M2 fitted beta is malformed or nonfinite")
    diagnostics = (
        float(result.objective),
        float(result.smooth_objective),
        float(result.penalty_value),
        float(result.kkt_residual),
        float(result.relative_objective_change),
        float(result.range_residual),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in diagnostics):
        raise CoreTrialExecutorError("M2 fit diagnostics are nonfinite or negative")
    if not math.isfinite(float(result.lipschitz)) or result.lipschitz <= 0.0:
        raise CoreTrialExecutorError("M2 fit Lipschitz diagnostic is invalid")
    if (
        isinstance(result.iterations, bool)
        or not isinstance(result.iterations, int)
        or not 0 <= result.iterations <= maximum_iterations
    ):
        raise CoreTrialExecutorError("M2 fit iteration count is invalid")
    if result.status != "pass":
        raise CoreTrialExecutorError("M2 fit did not report pass")
    if result.kkt_residual > m2.KKT_TOLERANCE:
        raise CoreTrialExecutorError("M2 fit KKT residual exceeds tolerance")
    if result.relative_objective_change > m2.RELATIVE_OBJECTIVE_TOLERANCE:
        raise CoreTrialExecutorError(
            "M2 fit relative-objective change exceeds tolerance"
        )
    if result.range_residual > m2.PROX_TOLERANCE:
        raise CoreTrialExecutorError("M2 fit range residual exceeds tolerance")
    return result


def penalty_from_configuration(
    configuration: controller.Configuration,
    lambda_maximum: m2.LambdaMaximumResult,
) -> PenaltyMapping:
    """Map one frozen core controller configuration to an engine penalty."""
    _validate_lambda_maximum(lambda_maximum)
    if configuration.family == "lasso":
        kind = "lasso"
        l1_ratio = 1.0
    elif configuration.family == "elastic_net":
        if configuration.alpha is None:
            raise CoreTrialExecutorError("elastic-net configuration lacks alpha")
        kind = "elastic_net"
        l1_ratio = float(configuration.alpha)
    elif configuration.family == "group_lasso":
        kind = "group_lasso"
        l1_ratio = 1.0
    else:
        raise CoreTrialExecutorError("the core executor does not implement this family")
    value = float(lambda_maximum.value)
    strength = float(configuration.rho) * value
    specification = m2.PenaltySpec(kind, strength, l1_ratio)
    return PenaltyMapping(
        family=configuration.family,
        penalty_kind=kind,
        l1_ratio=l1_ratio,
        rho=float(configuration.rho),
        lambda_maximum=lambda_maximum,
        specification=specification,
    )


def _aggregate_pair_contrasts(
    rows: Sequence[controller.PairContrastRow],
) -> tuple[float, ...]:
    """Apply the controller's equal-group/equal-cell 120-pair aggregation."""
    by_group: dict[str, list[controller.PairContrastRow]] = {
        group: [] for group in m1.GROUP_ORDER
    }
    for row in rows:
        if row.group not in by_group:
            raise CoreTrialExecutorError("pair-contrast row has an unexpected group")
        by_group[row.group].append(row)
    if any(not by_group[group] for group in m1.GROUP_ORDER):
        raise CoreTrialExecutorError("pair contrasts do not cover all 12 groups")
    group_means = tuple(
        tuple(
            math.fsum(float(row.values[pair]) for row in by_group[group])
            / len(by_group[group])
            for pair in range(m2.PAIR_DIMENSION)
        )
        for group in m1.GROUP_ORDER
    )
    result = tuple(
        math.fsum(group[pair] for group in group_means) / EXPECTED_FOLDS
        for pair in range(m2.PAIR_DIMENSION)
    )
    if not all(math.isfinite(value) for value in result):
        raise CoreTrialExecutorError("equal-group pair contrasts are nonfinite")
    return result


class CoreTrialExecutor:
    """Execute reusable beta-zero and core-family trials over prepared folds."""

    def __init__(
        self,
        prepared: PreparedTrialCache,
        *,
        maximum_iterations: int = FROZEN_MAXIMUM_ITERATIONS,
    ) -> None:
        if m2.FISTA_MAXIMUM_ITERATIONS != FROZEN_MAXIMUM_ITERATIONS:
            raise CoreTrialExecutorError(
                "engine iteration cap differs from the executor-owned frozen cap"
            )
        if (
            isinstance(maximum_iterations, bool)
            or not isinstance(maximum_iterations, int)
            or maximum_iterations <= 0
            or maximum_iterations > FROZEN_MAXIMUM_ITERATIONS
        ):
            raise CoreTrialExecutorError(
                "maximum iterations must be an integer in the frozen 1..20000 range"
            )
        _require_prepared_cache(prepared)
        if (
            prepared.mode == SEALED_PRODUCTION_MODE
            and maximum_iterations != FROZEN_MAXIMUM_ITERATIONS
        ):
            raise CoreTrialExecutorError(
                "sealed production requires exactly 20000 maximum iterations"
            )
        if prepared.geometry.library is not prepared.candidates or (
            prepared.pairs.candidates is not prepared.candidates
        ):
            raise CoreTrialExecutorError("prepared cache libraries are not shared")
        if (
            len(prepared.folds) != EXPECTED_FOLDS
            or tuple(fold.spec.ordinal for fold in prepared.folds)
            != tuple(range(1, EXPECTED_FOLDS + 1))
            or tuple(fold.spec.held_out_group for fold in prepared.folds)
            != m1.GROUP_ORDER
        ):
            raise CoreTrialExecutorError("prepared cache is not the frozen 12-LOGO order")
        if prepared.mode == SEALED_PRODUCTION_MODE:
            _validate_execution_scope(
                tuple(
                    cell
                    for fold in prepared.folds
                    for cell in fold.validation.cells
                ),
                prepared.mode,
                prepared.execution_inventory,
                prepared.expected_rank,
            )
        elif prepared.mode != SYNTHETIC_TEST_MODE or prepared.execution_inventory is not None:
            raise CoreTrialExecutorError("prepared cache execution mode is invalid")

        expected_indexes = set(range(len(prepared.cell_inventory)))
        seen_validation: set[int] = set()
        for fold in prepared.folds:
            spec = fold.spec
            expected_validation = tuple(
                index
                for index, (_, group) in enumerate(prepared.cell_inventory)
                if group == spec.held_out_group
            )
            expected_training = tuple(
                index
                for index, (_, group) in enumerate(prepared.cell_inventory)
                if group != spec.held_out_group
            )
            if (
                spec.validation_indexes != expected_validation
                or spec.training_indexes != expected_training
                or fold.fixed_m1.training_indexes != expected_training
                or fold.training.transform is not fold.validation.transform
                or (
                    prepared.mode == SEALED_PRODUCTION_MODE
                    and fold.training.transform.rank != EXPECTED_PRODUCTION_RANK
                )
                or tuple(cell.identifier for cell in fold.training.cells)
                != tuple(prepared.cell_inventory[index][0] for index in expected_training)
                or tuple(cell.identifier for cell in fold.validation.cells)
                != tuple(prepared.cell_inventory[index][0] for index in expected_validation)
                or fold.baseline_log_probability_nats.shape
                != (len(expected_validation),)
                or fold.baseline_pair_probabilities.shape
                != (len(expected_validation), m2.PAIR_DIMENSION)
                or not np.all(np.isfinite(fold.baseline_log_probability_nats))
                or not np.all(np.isfinite(fold.baseline_pair_probabilities))
            ):
                raise CoreTrialExecutorError("prepared fold cache is internally inconsistent")
            seen_validation.update(expected_validation)
        if seen_validation != expected_indexes:
            raise CoreTrialExecutorError("prepared folds do not cover every cached cell")
        self.prepared = prepared
        self.maximum_iterations = int(maximum_iterations)
        self._lambda_maxima: dict[
            tuple[int, str, float], m2.LambdaMaximumResult
        ] = {}

    def _lambda_maximum(
        self,
        fold: PreparedFold,
        configuration: controller.Configuration,
    ) -> m2.LambdaMaximumResult:
        if configuration.family == "elastic_net":
            assert configuration.alpha is not None
            ratio = float(configuration.alpha)
        else:
            ratio = 1.0
        kind = configuration.family
        key = (fold.spec.ordinal, kind, ratio)
        if key not in self._lambda_maxima:
            self._lambda_maxima[key] = _validate_lambda_maximum(
                m2.fold_lambda_maximum(
                    fold.fixed_m1.theta,
                    fold.training,
                    self.prepared.candidates,
                    self.prepared.geometry,
                    self.prepared.pairs,
                    kind,
                    l1_ratio=ratio,
                )
            )
        return self._lambda_maxima[key]

    @staticmethod
    def _validate_trial(trial: controller.Trial) -> bool:
        if type(trial) is not controller.Trial:
            raise CoreTrialExecutorError("trial was not created by the frozen controller")
        core = controller.coverage_trials()
        if trial.kind == "reference_anchor":
            if trial != core[0]:
                raise CoreTrialExecutorError(
                    "beta-zero reference is not canonical coverage trial 1"
                )
            return True
        if trial.kind == "anchor":
            if (
                not 2 <= trial.valid_index <= controller.COVERAGE_TRIALS
                or trial != core[trial.valid_index - 1]
            ):
                raise CoreTrialExecutorError(
                    "association anchor does not match its canonical coverage slot"
                )
            return False
        if (
            trial.kind != "successor"
            or trial.valid_index <= controller.COVERAGE_TRIALS
            or trial.valid_index > controller.MAXIMUM_VALID_TRIALS
        ):
            raise CoreTrialExecutorError(
                "core successor must be a post-coverage trial within the 80-trial cap"
            )
        if (
            trial.falsifier
            or type(trial.configuration) is not controller.Configuration
            or trial.configuration.family not in controller.CORE_FAMILIES
        ):
            raise CoreTrialExecutorError(
                "core successor configuration is not controller-valid"
            )
        return False

    def execute(
        self,
        trial: controller.Trial,
        *,
        authorization: SuccessorAuthorization | None = None,
        _session_capability: ProductionExecutionCapability | None = None,
    ) -> TrialExecution:
        """Run one controller trial, enforcing claimed production chronology."""
        capability_record = _begin_production_execution_attempt(
            self,
            trial,
            authorization,
            _session_capability,
        )
        try:
            execution = self._execute_trial(trial, authorization=authorization)
        except BaseException as error:
            if capability_record is not None:
                try:
                    assert _session_capability is not None
                    _complete_production_execution_attempt(
                        _session_capability,
                        capability_record,
                        execution=None,
                        error=error,
                    )
                except Exception as completion_error:
                    raise ProductionExecutionAuthorityError(
                        "production attempt could not be burned after failure"
                    ) from completion_error
            raise
        if capability_record is not None:
            assert _session_capability is not None
            _complete_production_execution_attempt(
                _session_capability,
                capability_record,
                execution=execution,
                error=None,
            )
        return execution

    def _execute_trial(
        self,
        trial: controller.Trial,
        *,
        authorization: SuccessorAuthorization | None = None,
    ) -> TrialExecution:
        """Numerical body shared unchanged by synthetic and claimed production."""
        _require_prepared_cache(self.prepared)
        reference = self._validate_trial(trial)
        if trial.kind == "successor":
            _require_successor_authorization(self, trial, authorization)
        elif authorization is not None:
            raise CoreTrialExecutorError(
                "successor authorization cannot be attached to an anchor"
            )
        zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
        fold_results: list[FoldTrialResult] = []
        rows: list[controller.PairContrastRow] = []
        seen_indexes: set[int] = set()

        for fold in self.prepared.folds:
            if reference:
                penalty = None
                fit = None
                beta = zero
                log_probability = fold.baseline_log_probability_nats
                pair_contrasts = np.zeros_like(fold.baseline_pair_probabilities)
                active_degrees = 0
            else:
                assert trial.configuration is not None
                maximum = self._lambda_maximum(fold, trial.configuration)
                penalty = penalty_from_configuration(trial.configuration, maximum)
                fit = _validate_optimizer_result(
                    m2.fit_m2(
                        fold.fixed_m1.theta,
                        fold.training,
                        self.prepared.candidates,
                        self.prepared.geometry,
                        self.prepared.pairs,
                        penalty.specification,
                        maximum_iterations=self.maximum_iterations,
                    ),
                    self.maximum_iterations,
                )
                if trial.configuration.rho == 1.0 and np.count_nonzero(fit.beta) != 0:
                    raise CoreTrialExecutorError("rho=1 did not retain the beta-zero solution")
                beta = fit.beta
                prediction = m2.evaluate_m2(
                    beta,
                    fold.fixed_m1.theta,
                    fold.validation,
                    self.prepared.candidates,
                    self.prepared.geometry,
                    self.prepared.pairs,
                    need_predictions=True,
                )
                if prediction.log_probability_nats is None:
                    raise CoreTrialExecutorError("held-out M2 log probabilities were omitted")
                log_probability = _readonly_copy(
                    prediction.log_probability_nats,
                    shape=(len(fold.validation.cells),),
                    label="held-out M2 log probability",
                )
                fitted_pair_probability = _pair_probabilities(
                    beta,
                    fold.fixed_m1.theta,
                    fold.validation,
                    self.prepared.candidates,
                    self.prepared.geometry,
                    self.prepared.pairs,
                )
                pair_contrasts = (
                    fitted_pair_probability - fold.baseline_pair_probabilities
                )
                active_degrees = int(np.count_nonzero(np.asarray(beta, dtype=float)))

            for local_index, cell in enumerate(fold.validation.cells):
                if cell.k in {0, m1.PANEL_SIZE} and float(
                    log_probability[local_index]
                ) != float(fold.baseline_log_probability_nats[local_index]):
                    raise CoreTrialExecutorError(
                        "K in {0,16} must have exact-zero M2-minus-M1 gain"
                    )
                if cell.k in {0, 1, m1.PANEL_SIZE} and np.any(
                    np.asarray(pair_contrasts[local_index], dtype=float) != 0.0
                ):
                    raise CoreTrialExecutorError(
                        "K in {0,1,16} must have exact-zero pair contrasts"
                    )

            cell_gains = (
                np.asarray(log_probability, dtype=float)
                - fold.baseline_log_probability_nats
            ) / LN2
            if not np.all(np.isfinite(cell_gains)):
                raise CoreTrialExecutorError("held-out score gains are nonfinite")
            fold_gain = math.fsum(float(value) for value in cell_gains) / len(cell_gains)
            cell_results: list[CellTrialResult] = []
            for local_index, (global_index, cell) in enumerate(
                zip(
                    fold.spec.validation_indexes,
                    fold.validation.cells,
                    strict=True,
                )
            ):
                if global_index in seen_indexes:
                    raise CoreTrialExecutorError("a cell received two held-out predictions")
                seen_indexes.add(global_index)
                contrast = np.asarray(pair_contrasts[local_index], dtype=float)
                if cell.k in {0, 1, m1.PANEL_SIZE}:
                    contrast = np.zeros(m2.PAIR_DIMENSION, dtype=float)
                values = tuple(float(value) for value in contrast)
                row = controller.make_pair_contrast_row(
                    cell_key=str(cell.identifier),
                    group=str(cell.group),
                    k=int(cell.k),
                    values=values,
                )
                rows.append(row)
                cell_results.append(
                    CellTrialResult(
                        global_index=int(global_index),
                        cell_key=str(cell.identifier),
                        group=str(cell.group),
                        k=int(cell.k),
                        m1_log_probability_nats=float(
                            fold.baseline_log_probability_nats[local_index]
                        ),
                        m2_log_probability_nats=float(log_probability[local_index]),
                        pair_probability_contrasts=values,
                    )
                )
            fold_results.append(
                FoldTrialResult(
                    spec=fold.spec,
                    penalty=penalty,
                    fit=fit,
                    gain_bits_per_cell=float(fold_gain),
                    active_pair_degrees=active_degrees,
                    cells=tuple(cell_results),
                )
            )

        expected_indexes = {
            index
            for fold in self.prepared.folds
            for index in fold.spec.validation_indexes
        }
        if seen_indexes != expected_indexes or len(seen_indexes) != sum(
            len(fold.spec.validation_indexes) for fold in self.prepared.folds
        ):
            raise CoreTrialExecutorError("outer cross-fit did not predict every cell once")
        rows.sort(key=lambda row: row.cell_key)
        contrasts = _aggregate_pair_contrasts(rows)
        gain = math.fsum(fold.gain_bits_per_cell for fold in fold_results) / EXPECTED_FOLDS
        active_pair_degrees = max(fold.active_pair_degrees for fold in fold_results)
        result = controller.TrialResult(
            trial=trial,
            gain_bits_per_cell=float(gain),
            active_pair_degrees=active_pair_degrees,
        )
        execution = TrialExecution(
            trial=trial,
            folds=tuple(fold_results),
            pair_contrast_rows=tuple(rows),
            equal_group_pair_contrasts=contrasts,
            result=result,
        )
        _register_owned_execution(self, execution)
        return execution


def _register_owned_execution(
    owner: CoreTrialExecutor,
    execution: TrialExecution,
) -> None:
    _register_execution_identity(
        execution,
        (
            weakref.ref(owner),
            weakref.ref(owner.prepared),
            _deep_integrity_digest(execution),
        ),
    )


def _require_owned_execution(
    owner: CoreTrialExecutor,
    execution: TrialExecution,
) -> controller.TrialResult:
    if type(execution) is not TrialExecution:
        raise CoreTrialExecutorError(
            "successor history contains a non-executor TrialExecution"
        )
    owner_reference, cache_reference, expected_digest = (
        _lookup_execution_identity(execution)
    )
    if owner_reference() is not owner or cache_reference() is not owner.prepared:
        raise CoreTrialExecutorError(
            "successor history was produced by a different executor/cache"
        )
    _require_prepared_cache(owner.prepared)
    if _deep_integrity_digest(execution) != expected_digest:
        raise CoreTrialExecutorError(
            "an executor-owned result changed after execution"
        )
    return execution.result


ExternalExecutionValidator = Callable[[Any, controller.TrialResult], bool]


def _requires_external_execution(trial: controller.Trial) -> bool:
    """Return whether this trial belongs to an adapter outside the core executor."""
    if type(trial) is not controller.Trial:
        return False
    if trial.kind in {"permutation", "rademacher_noise"}:
        return True
    configuration = trial.configuration
    return (
        type(configuration) is controller.Configuration
        and configuration.family not in controller.CORE_FAMILIES
    )


def _run_external_validator(
    producer: Any,
    validator: ExternalExecutionValidator,
    result: controller.TrialResult,
    *,
    execution_detail: object | None = None,
    sealed_production: bool = False,
    owner: CoreTrialExecutor | None = None,
) -> None:
    if sealed_production:
        if execution_detail is None:
            raise CoreTrialExecutorError(
                "sealed-production external execution requires immutable detail"
            )
        if result.trial.kind == "permutation":
            module_name = "v3_m2_permutation_production_adapter_v1"
            validator_name = "_PermutationExecutionValidator"
            producer_name = "WholeSetPermutationAdapter"
        elif result.trial.kind == "rademacher_noise":
            module_name = "v3_m2_rademacher_production_adapter_v1"
            validator_name = "_RademacherExecutionValidator"
            producer_name = "RademacherReferenceAdapter"
        else:
            module_name = "v3_m2_optional_adapter_v1"
            validator_name = "_OptionalExecutionValidator"
            producer_name = "OptionalFamilyAdapter"
        expected_path = TOOLS_DIR / f"{module_name}.py"
        module = sys.modules.get(module_name)
        module_path = getattr(module, "SCRIPT_PATH", None)
        module_file = getattr(module, "__file__", None)
        module_file_path = (
            None
            if module_file is None
            else Path(os.path.abspath(os.fspath(module_file)))
        )
        if (
            module is None
            or getattr(module, "__name__", None) != module_name
            or module_path is None
            or Path(os.path.abspath(os.fspath(module_path))) != expected_path
            or not expected_path.is_file()
            or expected_path.is_symlink()
            or module_file_path != expected_path
            or module_file_path.resolve(strict=True)
            != expected_path.resolve(strict=True)
        ):
            raise CoreTrialExecutorError(
                "sealed-production external execution lacks its exact named adapter"
            )
        validator_type = getattr(module, validator_name, None)
        producer_type = getattr(module, producer_name, None)
        callback = getattr(module, "_validate_named_external_execution_detail", None)
        code = getattr(callback, "__code__", None)
        if (
            not isinstance(validator_type, type)
            or not isinstance(producer_type, type)
            or type(producer) is not producer_type
            or type(validator) is not validator_type
            or not callable(callback)
            or getattr(callback, "__module__", None) != module_name
            or code is None
            or Path(os.path.abspath(code.co_filename)) != expected_path
        ):
            raise CoreTrialExecutorError(
                "sealed-production external execution requires the exact named validator"
            )
        if module_name in {
            "v3_m2_permutation_production_adapter_v1",
            "v3_m2_rademacher_production_adapter_v1",
        }:
            owner_binding = getattr(module, "_adapter_owner_and_cache", None)
            binding_code = getattr(owner_binding, "__code__", None)
            if (
                type(owner) is not CoreTrialExecutor
                or not callable(owner_binding)
                or getattr(owner_binding, "__module__", None) != module_name
                or binding_code is None
                or Path(os.path.abspath(binding_code.co_filename)) != expected_path
            ):
                raise CoreTrialExecutorError(
                    "sealed falsifier execution lacks its exact owner binding"
                )
            try:
                bound_owner, bound_cache = owner_binding(producer)
            except Exception as error:
                raise CoreTrialExecutorError(
                    "named falsifier owner binding could not be authenticated"
                ) from error
            if bound_owner is not owner or bound_cache is not owner.prepared:
                raise CoreTrialExecutorError(
                    "named falsifier execution belongs to another owner or cache"
                )
        try:
            accepted = callback(producer, validator, result, execution_detail)
        except Exception as error:
            raise CoreTrialExecutorError(
                "named external execution validator raised while authenticating detail"
            ) from error
        if accepted is not True:
            raise CoreTrialExecutorError(
                "named external execution validator rejected immutable detail"
            )
        return

    try:
        accepted = validator(producer, result)
    except Exception as error:
        raise CoreTrialExecutorError(
            "external execution validator raised while authenticating its producer"
        ) from error
    if accepted is not True:
        raise CoreTrialExecutorError(
            "external execution validator did not authenticate the producer result"
        )
    detail_validator = getattr(validator, "validate_execution_detail", None)
    if detail_validator is None:
        return
    if not callable(detail_validator):
        raise CoreTrialExecutorError(
            "external execution detail validator is not callable"
        )
    try:
        detail_accepted = detail_validator(producer, result, execution_detail)
    except Exception as error:
        raise CoreTrialExecutorError(
            "external execution detail validator raised while authenticating its record"
        ) from error
    if detail_accepted is not True:
        raise CoreTrialExecutorError(
            "external execution detail validator rejected the producer record"
        )


def authenticate_external_execution(
    owner: CoreTrialExecutor,
    producer: Any,
    validator: ExternalExecutionValidator,
    result: controller.TrialResult,
    *,
    execution_detail: object | None = None,
) -> ExternalExecutionReceipt:
    """Bind one non-core result to its producer, validator, owner, and cache."""
    if type(owner) is not CoreTrialExecutor:
        raise CoreTrialExecutorError("external receipt owner has the wrong executor type")
    _require_prepared_cache(owner.prepared)
    if type(result) is not controller.TrialResult or not result.valid:
        raise CoreTrialExecutorError("external receipt requires an exact valid TrialResult")
    if not _requires_external_execution(result.trial):
        raise CoreTrialExecutorError(
            "core reference, anchor, and successor results cannot use external receipts"
        )
    if not callable(validator):
        raise CoreTrialExecutorError("external execution validator is not callable")
    try:
        producer_reference = weakref.ref(producer)
        validator_reference = weakref.ref(validator)
    except TypeError as error:
        raise CoreTrialExecutorError(
            "external producer and validator must have stable weak identities"
        ) from error
    _run_external_validator(
        producer,
        validator,
        result,
        execution_detail=execution_detail,
        sealed_production=owner.prepared.mode == SEALED_PRODUCTION_MODE,
        owner=owner,
    )
    receipt = ExternalExecutionReceipt(
        result,
        execution_detail=execution_detail,
        _seal=_EXTERNAL_EXECUTION_RECEIPT_SEAL,
    )
    _register_external_receipt_identity(
        receipt,
        (
            weakref.ref(owner),
            weakref.ref(owner.prepared),
            producer_reference,
            validator_reference,
            _deep_integrity_digest(receipt),
        ),
    )
    return receipt


def _require_external_execution_receipt(
    owner: CoreTrialExecutor,
    receipt: ExternalExecutionReceipt,
) -> controller.TrialResult:
    if type(receipt) is not ExternalExecutionReceipt:
        raise CoreTrialExecutorError(
            "successor history contains neither an owned execution nor a receipt"
        )
    (
        owner_reference,
        cache_reference,
        producer_reference,
        validator_reference,
        expected_digest,
    ) = _lookup_external_receipt_identity(receipt)
    if owner_reference() is not owner or cache_reference() is not owner.prepared:
        raise CoreTrialExecutorError(
            "external execution receipt belongs to another executor/cache"
        )
    if (
        receipt._seal is not _EXTERNAL_EXECUTION_RECEIPT_SEAL
        or _deep_integrity_digest(receipt) != expected_digest
    ):
        raise CoreTrialExecutorError("external execution receipt changed after issuance")
    producer = producer_reference()
    validator = validator_reference()
    if producer is None or validator is None:
        raise CoreTrialExecutorError(
            "external execution producer or validator no longer exists"
        )
    _run_external_validator(
        producer,
        validator,
        receipt.result,
        execution_detail=receipt.execution_detail,
        sealed_production=owner.prepared.mode == SEALED_PRODUCTION_MODE,
        owner=owner,
    )
    if _deep_integrity_digest(receipt) != expected_digest:
        raise CoreTrialExecutorError(
            "external execution receipt changed during revalidation"
        )
    return receipt.result


def _authenticated_history_result(
    owner: CoreTrialExecutor,
    execution: TrialExecution | ExternalExecutionReceipt,
) -> controller.TrialResult:
    if type(execution) is TrialExecution:
        return _require_owned_execution(owner, execution)
    if type(execution) is ExternalExecutionReceipt:
        return _require_external_execution_receipt(owner, execution)
    raise CoreTrialExecutorError(
        "successor history contains an unauthenticated raw result"
    )


def _authenticated_result_prefix(
    owner: CoreTrialExecutor,
    trial: controller.Trial,
    prior_executions: tuple[TrialExecution | ExternalExecutionReceipt, ...],
    coverage_evidence: controller.CoverageEvidence,
) -> tuple[controller.TrialResult, ...]:
    if type(coverage_evidence) is not controller.CoverageEvidence:
        raise CoreTrialExecutorError("successor authorization lacks exact coverage evidence")
    if not isinstance(prior_executions, tuple):
        raise CoreTrialExecutorError("successor execution history must be immutable")
    results = tuple(
        _authenticated_history_result(owner, execution)
        for execution in prior_executions
    )
    if len(results) < controller.COVERAGE_TRIALS:
        raise CoreTrialExecutorError("successor history has incomplete core coverage")
    if tuple(result.trial.valid_index for result in results) != tuple(
        range(1, len(results) + 1)
    ):
        raise CoreTrialExecutorError("successor history is not an exact contiguous prefix")
    if trial.valid_index != len(results) + 1:
        raise CoreTrialExecutorError("successor is not the next contiguous trial")
    if trial.valid_index > controller.MAXIMUM_VALID_TRIALS:
        raise CoreTrialExecutorError("successor exceeds the frozen 80-trial cap")
    if trial.coverage_binding != coverage_evidence.binding:
        raise CoreTrialExecutorError("successor and coverage evidence bindings differ")

    bound_sixteenth = replace(
        results[controller.COVERAGE_TRIALS - 1],
        coverage_binding=coverage_evidence.binding,
    )
    normalized = (
        *results[: controller.COVERAGE_TRIALS - 1],
        bound_sixteenth,
        *results[controller.COVERAGE_TRIALS :],
    )
    if normalized[: controller.COVERAGE_TRIALS] != coverage_evidence.core_results:
        raise CoreTrialExecutorError(
            "coverage evidence is not the exact executor-owned core prefix"
        )
    try:
        core_incumbent = controller.choose_incumbent(
            normalized[: controller.COVERAGE_TRIALS]
        )
    except controller.ControllerError as error:
        raise CoreTrialExecutorError(
            "coverage evidence has no selectable core sparse incumbent"
        ) from error
    incumbent_execution = prior_executions[core_incumbent.trial.valid_index - 1]
    if type(incumbent_execution) is not TrialExecution:
        raise CoreTrialExecutorError(
            "core sparse incumbent was not executed by this core executor"
        )
    if (
        incumbent_execution.equal_group_pair_contrasts
        != coverage_evidence.equal_group_pair_contrasts
    ):
        raise CoreTrialExecutorError(
            "coverage evidence contrasts differ from the executor-owned core incumbent"
        )
    try:
        dummy = controller.TrialResult(
            trial=trial,
            gain_bits_per_cell=0.0,
            active_pair_degrees=0,
        )
        controller.validate_available_result_prefix(
            (*normalized, dummy),
            coverage_evidence,
        )
    except (controller.ControllerError, ValueError, TypeError) as error:
        raise CoreTrialExecutorError(
            "proposed successor is not the canonical next controller trial"
        ) from error
    return normalized


def authorize_successor(
    owner: CoreTrialExecutor,
    trial: controller.Trial,
    prior_executions: Sequence[TrialExecution | ExternalExecutionReceipt],
    coverage_evidence: controller.CoverageEvidence,
) -> SuccessorAuthorization:
    """Issue a one-trial capability from an exact executor-owned result prefix."""
    if type(owner) is not CoreTrialExecutor:
        raise CoreTrialExecutorError("successor owner has the wrong executor type")
    _require_prepared_cache(owner.prepared)
    if type(trial) is not controller.Trial or trial.kind != "successor":
        raise CoreTrialExecutorError("authorization target is not a controller successor")
    immutable_history = tuple(prior_executions)
    _authenticated_result_prefix(
        owner,
        trial,
        immutable_history,
        coverage_evidence,
    )
    authorization = SuccessorAuthorization(
        trial,
        immutable_history,
        coverage_evidence,
        _seal=_SUCCESSOR_AUTHORIZATION_SEAL,
    )
    _register_authorization_identity(
        authorization,
        (weakref.ref(owner), _deep_integrity_digest(authorization)),
    )
    return authorization


def _require_successor_authorization(
    owner: CoreTrialExecutor,
    trial: controller.Trial,
    authorization: SuccessorAuthorization | None,
) -> None:
    if type(authorization) is not SuccessorAuthorization:
        raise CoreTrialExecutorError(
            "post-coverage successors require executor-owned authorization"
        )
    owner_reference, expected_digest = _lookup_authorization_identity(authorization)
    if owner_reference() is not owner:
        raise CoreTrialExecutorError("successor authorization belongs to another executor")
    if (
        authorization._seal is not _SUCCESSOR_AUTHORIZATION_SEAL
        or authorization.trial != trial
        or _deep_integrity_digest(authorization) != expected_digest
    ):
        raise CoreTrialExecutorError("successor authorization changed after issuance")
    _authenticated_result_prefix(
        owner,
        trial,
        authorization.prior_executions,
        authorization.coverage_evidence,
    )


def benchmark_scale_units(context: Any) -> Mapping[str, Mapping[str, Any]]:
    """Expose the executor view of one context-bound central benchmark run."""

    import v3_m2_scale_benchmark_units_v1 as bridge

    return bridge.executor_scale_units(context)


__all__ = (
    "CellTrialResult",
    "CoreTrialExecutor",
    "CoreTrialExecutorError",
    "EXPECTED_PRODUCTION_RANK",
    "EXPECTED_PRODUCTION_ROWS",
    "ExternalExecutionReceipt",
    "ExternalExecutionValidator",
    "FROZEN_MAXIMUM_ITERATIONS",
    "FixedM1Offset",
    "FoldSpec",
    "FoldTrialResult",
    "FrozenExecutionInventory",
    "PenaltyMapping",
    "PreparedFold",
    "PreparedTrialCache",
    "ProductionExecutionAuthorityError",
    "SEALED_PRODUCTION_MODE",
    "SYNTHETIC_TEST_MODE",
    "SuccessorAuthorization",
    "TrialExecution",
    "authenticate_external_execution",
    "authorize_successor",
    "benchmark_scale_units",
    "logo_fold_specs",
    "penalty_from_configuration",
    "prepare_trial_cache",
)
