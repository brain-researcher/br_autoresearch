#!/usr/bin/env python3
"""Fail-closed, payload-free execution planning guards for EP21.

Only a conspicuously synthetic fixture context is available today.  The
empirical entrypoint stays unavailable until a real protocol lock, a later
execution authorization, locked module settings, and authenticated structural
metadata exist.  This module performs no file I/O, payload access, model or
array import, device creation, executor creation, optimization, or inference.

Every public binding and plan is a distinct frozen type holding only canonical
JSON bytes plus a domain-separated integrity seal.  Constructors and every
public consumer revalidate the canonical record, its invariants, and its seal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SYNTHETIC_SCHEMA = "ep21.synthetic_adapter_fixture.v1"
EPISODE_ID = "ep21"
MODULE_ID = "pipeline_conformance_adapter"
REPETITION_MODES = {
    "image_average_before_fit",
    "authenticated_cross_subject_pairing",
}


class AdapterValidationError(ValueError):
    """A metadata invariant failed before any payload access."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise AdapterValidationError(code, message)


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        _fail("E_IDENTIFIER", f"{label} is not a canonical identifier")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        _fail("E_SHA256", f"{label} is not a canonical SHA256")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("E_INTEGER", f"{label} must be a positive integer")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail("E_INTEGER", f"{label} must be a nonnegative integer")
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AdapterValidationError(
            "E_CANONICAL_JSON", "metadata is not canonical JSON"
        ) from error


def metadata_sha256(value: Any) -> str:
    """Digest metadata using the adapter's canonical JSON encoding."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(namespace: str, canonical_record: bytes) -> str:
    return hashlib.sha256(namespace.encode("ascii") + b"\0" + canonical_record).hexdigest()


def _decode(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        _fail("E_IMMUTABILITY", "sealed record must be immutable bytes")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterValidationError("E_CANONICAL_JSON", "sealed record is not JSON") from error
    if not isinstance(value, dict) or _canonical(value) != raw:
        _fail("E_CANONICAL_JSON", "sealed record is not a canonical object")
    return value


def _fields(record: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(record) != expected:
        _fail("E_FIELDS", f"{label} fields differ")


def _validated(
    obj: Any,
    cls: type,
    namespace: str,
    validator: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if type(obj) is not cls:
        _fail("E_TYPE", f"expected {cls.__name__}")
    record = _decode(obj.canonical_record_json)
    validator(record)
    if obj.seal != _seal(namespace, obj.canonical_record_json):
        _fail("E_SEAL", f"{namespace} seal differs")
    return record


def _build(cls: type, namespace: str, record: Mapping[str, Any]) -> Any:
    raw = _canonical(record)
    return cls(canonical_record_json=raw, seal=_seal(namespace, raw))


ALPHA_DECLARATION_FIELDS = {
    "policy_id",
    "policy_sha256",
    "selection_scope",
    "allowed_fit_role",
    "fixed_before_tau_test",
    "tau_test_or_ood_used",
    "resolved_value",
    "selection_receipt_sha256",
}


def _validate_alpha_declaration(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != ALPHA_DECLARATION_FIELDS:
        _fail("E_ALPHA_FIELDS", "alpha declaration fields differ")
    policy_id = _id(record["policy_id"], "alpha policy ID")
    _sha(record["policy_sha256"], "alpha policy digest")
    if record["selection_scope"] not in {"fixed_predeclared", "tau_train_only"}:
        _fail("E_ALPHA_SCOPE", "alpha selection scope is unsupported")
    if record["allowed_fit_role"] != "tau_train":
        _fail("E_ALPHA_ROLE", "alpha policy may use tau_train only")
    if (
        record["fixed_before_tau_test"] is not True
        or record["tau_test_or_ood_used"] is not False
    ):
        _fail("E_ALPHA_LEAKAGE", "alpha declaration does not exclude tau-test/OOD use")
    policy_words = re.sub(r"[^a-z0-9]+", "_", policy_id.lower()).strip("_")
    if "tau_test" in policy_words or "ood" in policy_words:
        _fail("E_ALPHA_LEAKAGE", "alpha policy ID names a forbidden role")
    value = record["resolved_value"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        _fail("E_ALPHA_VALUE", "resolved alpha must be finite and positive")
    receipt = record["selection_receipt_sha256"]
    if record["selection_scope"] == "tau_train_only":
        _sha(receipt, "alpha selection receipt")
    elif receipt is not None:
        _fail("E_ALPHA_RECEIPT", "fixed alpha unexpectedly has a selection receipt")


def _settings(record: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    settings = record.get("settings")
    if not isinstance(settings, dict) or set(settings) != {
        "content_feature_layers",
        "brain_loss_repetition_mode",
        "brain_loss_alpha_bindings",
    }:
        _fail("E_CONTEXT_SETTINGS", "synthetic module-setting fields differ")
    layers = settings["content_feature_layers"]
    if not isinstance(layers, list) or not layers:
        _fail("E_CONTEXT_SETTINGS", "content layers must be a non-empty list")
    normalized = tuple(_id(value, "content layer") for value in layers)
    if len(set(normalized)) != len(normalized):
        _fail("E_CONTEXT_SETTINGS", "content layers contain duplicates")
    mode = settings["brain_loss_repetition_mode"]
    if mode not in REPETITION_MODES:
        _fail("E_REPETITION_MODE", "brain-loss repetition mode is unsupported")
    alpha_bindings = settings["brain_loss_alpha_bindings"]
    if not isinstance(alpha_bindings, dict) or not alpha_bindings:
        _fail("E_CONTEXT_SETTINGS", "alpha bindings must be a non-empty mapping")
    for pair_id, declaration in alpha_bindings.items():
        _id(pair_id, "alpha-binding pair ID")
        _validate_alpha_declaration(declaration)
    if record.get("settings_sha256") != metadata_sha256(settings):
        _fail("E_CONTEXT_SETTINGS", "module-setting digest differs")
    return normalized, mode


def _validate_context(record: dict[str, Any]) -> None:
    _fields(
        record,
        {
            "schema",
            "context_kind",
            "episode_id",
            "module_id",
            "fixture_id",
            "pair_table_sha256",
            "pair_table_records",
            "settings",
            "settings_sha256",
            "empirical_capable",
        },
        "synthetic context",
    )
    if (
        record["schema"] != SYNTHETIC_SCHEMA
        or record["context_kind"] != "synthetic_fixture"
        or record["episode_id"] != EPISODE_ID
        or record["module_id"] != MODULE_ID
        or record["empirical_capable"] is not False
    ):
        _fail("E_CONTEXT_CAPABILITY", "context is not the exact synthetic schema")
    _id(record["fixture_id"], "fixture ID")
    _sha(record["pair_table_sha256"], "pair-table digest")
    pair_records = record["pair_table_records"]
    if not isinstance(pair_records, list) or not pair_records:
        _fail("E_PAIR_TABLE", "synthetic pair table must be a non-empty list")
    pair_ids: list[str] = []
    directions: list[tuple[str, str]] = []
    ordinals: list[int] = []
    for pair in pair_records:
        if not isinstance(pair, dict) or set(pair) != {
            "pair_id", "source_id", "target_id", "ordinal"
        }:
            _fail("E_PAIR_TABLE", "synthetic pair-table row fields differ")
        pair_ids.append(_id(pair["pair_id"], "pair-table pair ID"))
        source = _id(pair["source_id"], "pair-table source ID")
        target = _id(pair["target_id"], "pair-table target ID")
        if source == target:
            _fail("E_PAIR_DIRECTION", "pair-table source and target must differ")
        directions.append((source, target))
        ordinals.append(_nonnegative(pair["ordinal"], "pair-table ordinal"))
    if (
        len(set(pair_ids)) != len(pair_ids)
        or len(set(directions)) != len(directions)
        or len(set(ordinals)) != len(ordinals)
    ):
        _fail("E_PAIR_TABLE", "synthetic pair table contains duplicate keys")
    if record["pair_table_sha256"] != metadata_sha256(pair_records):
        _fail("E_PAIR_TABLE", "synthetic pair-table digest differs")
    _settings(record)
    if set(record["settings"]["brain_loss_alpha_bindings"]) != set(pair_ids):
        _fail("E_CONTEXT_SETTINGS", "alpha bindings differ from the sealed pair table")


@dataclass(frozen=True)
class SyntheticFixtureContext:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, SyntheticFixtureContext, "SyntheticFixtureContext", _validate_context)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, SyntheticFixtureContext, "SyntheticFixtureContext", _validate_context)

    @property
    def content_layers(self) -> tuple[str, ...]:
        return _settings(self.record)[0]

    @property
    def repetition_mode(self) -> str:
        return _settings(self.record)[1]


def make_synthetic_context(
    *,
    fixture_id: str,
    pair_table_records: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> SyntheticFixtureContext:
    _id(fixture_id, "fixture ID")
    if not isinstance(pair_table_records, Sequence) or isinstance(
        pair_table_records, (str, bytes)
    ):
        _fail("E_PAIR_TABLE", "pair-table records must be a sequence")
    if not isinstance(settings, Mapping):
        _fail("E_CONTEXT_SETTINGS", "settings must be a mapping")
    copied_pairs = json.loads(_canonical(list(pair_table_records)).decode("utf-8"))
    copied = json.loads(_canonical(settings).decode("utf-8"))
    record = {
        "schema": SYNTHETIC_SCHEMA,
        "context_kind": "synthetic_fixture",
        "episode_id": EPISODE_ID,
        "module_id": MODULE_ID,
        "fixture_id": fixture_id,
        "pair_table_sha256": metadata_sha256(copied_pairs),
        "pair_table_records": copied_pairs,
        "settings": copied,
        "settings_sha256": metadata_sha256(copied),
        "empirical_capable": False,
    }
    _validate_context(record)
    return _build(SyntheticFixtureContext, "SyntheticFixtureContext", record)


def _context_record(context: Any) -> dict[str, Any]:
    return _validated(
        context,
        SyntheticFixtureContext,
        "SyntheticFixtureContext",
        _validate_context,
    )


def bind_empirical_context(*, verified_lock_receipt: Any, authorization_receipt: Any) -> None:
    """Stay unavailable until the real lock and authorization resolver exists."""
    del verified_lock_receipt, authorization_receipt
    _fail(
        "E_EMPIRICAL_CONTEXT_UNAVAILABLE",
        "the protocol lock, later authorization, and empirical resolver are absent",
    )


def _validate_pair(record: dict[str, Any]) -> None:
    _fields(
        record,
        {"context_seal", "pair_id", "source_id", "target_id", "pair_table_sha256", "ordinal"},
        "pair binding",
    )
    _sha(record["context_seal"], "pair context seal")
    _id(record["pair_id"], "locked pair ID")
    _id(record["source_id"], "locked source ID")
    _id(record["target_id"], "locked target ID")
    if record["source_id"] == record["target_id"]:
        _fail("E_PAIR_DIRECTION", "source and target must differ")
    _sha(record["pair_table_sha256"], "pair-table digest")
    _nonnegative(record["ordinal"], "pair ordinal")


@dataclass(frozen=True)
class PairBinding:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, PairBinding, "PairBinding", _validate_pair)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, PairBinding, "PairBinding", _validate_pair)


def bind_pair(
    context: SyntheticFixtureContext, locked_pair_record: Mapping[str, Any]
) -> PairBinding:
    context_record = _context_record(context)
    if not isinstance(locked_pair_record, Mapping):
        _fail("E_PAIR_RECORD", "pair record must be a mapping")
    copied = json.loads(_canonical(locked_pair_record).decode("utf-8"))
    _fields(
        copied,
        {"pair_id", "source_id", "target_id", "pair_table_sha256", "ordinal"},
        "locked pair record",
    )
    if copied["pair_table_sha256"] != context_record["pair_table_sha256"]:
        _fail("E_PAIR_TABLE", "pair record is not bound to the fixture pair table")
    table_row = {
        field: copied[field]
        for field in ("pair_id", "source_id", "target_id", "ordinal")
    }
    if table_row not in context_record["pair_table_records"]:
        _fail("E_PAIR_MEMBERSHIP", "pair record is absent from the sealed pair table")
    record = {"context_seal": context.seal, **copied}
    _validate_pair(record)
    return _build(PairBinding, "PairBinding", record)


def _pair_for(context: SyntheticFixtureContext, pair: PairBinding) -> dict[str, Any]:
    context_record = _context_record(context)
    pair_record = _validated(pair, PairBinding, "PairBinding", _validate_pair)
    if (
        pair_record["context_seal"] != context.seal
        or pair_record["pair_table_sha256"] != context_record["pair_table_sha256"]
    ):
        _fail("E_PAIR_BINDING", "pair is not bound to this context")
    table_row = {
        field: pair_record[field]
        for field in ("pair_id", "source_id", "target_id", "ordinal")
    }
    if table_row not in context_record["pair_table_records"]:
        _fail("E_PAIR_MEMBERSHIP", "pair binding is absent from the sealed pair table")
    return pair_record


def _validate_worker(record: dict[str, Any]) -> None:
    _fields(record, {"context_seal", "max_workers", "explicit_worker_limit"}, "worker")
    _sha(record["context_seal"], "worker context seal")
    _positive(record["max_workers"], "max workers")
    _positive(record["explicit_worker_limit"], "explicit worker limit")
    if record["max_workers"] != record["explicit_worker_limit"]:
        _fail("E_WORKER_ALLOCATION", "max workers must equal the explicit limit")


@dataclass(frozen=True)
class WorkerBinding:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, WorkerBinding, "WorkerBinding", _validate_worker)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, WorkerBinding, "WorkerBinding", _validate_worker)


def bind_worker(
    context: SyntheticFixtureContext, *, max_workers: int, explicit_worker_limit: int
) -> WorkerBinding:
    """Bind a synthetic worker count to an explicit foreground-process limit."""
    _context_record(context)
    record = {
        "context_seal": context.seal,
        "max_workers": max_workers,
        "explicit_worker_limit": explicit_worker_limit,
    }
    _validate_worker(record)
    return _build(WorkerBinding, "WorkerBinding", record)


def _worker_for(context: SyntheticFixtureContext, worker: WorkerBinding) -> dict[str, Any]:
    _context_record(context)
    record = _validated(worker, WorkerBinding, "WorkerBinding", _validate_worker)
    if record["context_seal"] != context.seal:
        _fail("E_WORKER_BINDING", "worker is not bound to this context")
    return record


ROLE_ROW_FIELDS = {"row_id", "image_id"}
PRESENTATION_FIELDS = {
    "row_id",
    "image_id",
    "local_repetition_id",
    "cross_subject_alignment_key",
}


def _validate_role_rows(rows: Any, label: str) -> tuple[list[str], list[str]]:
    if not isinstance(rows, list) or not rows:
        _fail("E_STRUCTURAL_TYPE", f"{label} must be a non-empty list")
    row_ids: list[str] = []
    image_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROLE_ROW_FIELDS:
            _fail("E_STRUCTURAL_TYPE", f"{label} row fields differ")
        row_ids.append(_id(row["row_id"], f"{label} row ID"))
        image_ids.append(_id(row["image_id"], f"{label} image ID"))
    if len(set(row_ids)) != len(row_ids) or len(set(image_ids)) != len(image_ids):
        _fail("E_ROLE_DUPLICATE", f"{label} contains duplicate IDs")
    return row_ids, image_ids


def _validate_presentations(rows: Any, label: str, tau_set: set[str]) -> None:
    if not isinstance(rows, list) or not rows:
        _fail("E_STRUCTURAL_TYPE", f"{label} must be a non-empty list")
    row_ids: list[str] = []
    local_keys: list[tuple[str, str]] = []
    images: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != PRESENTATION_FIELDS:
            _fail("E_STRUCTURAL_TYPE", f"{label} presentation fields differ")
        row_ids.append(_id(row["row_id"], f"{label} row ID"))
        image_id = _id(row["image_id"], f"{label} image ID")
        repetition = _id(row["local_repetition_id"], f"{label} repetition ID")
        alignment = row["cross_subject_alignment_key"]
        if alignment is not None:
            _id(alignment, f"{label} alignment key")
        images.add(image_id)
        local_keys.append((image_id, repetition))
    if len(set(row_ids)) != len(row_ids) or len(set(local_keys)) != len(local_keys):
        _fail("E_PRESENTATION_DUPLICATE", f"{label} contains duplicate presentations")
    if images != tau_set:
        _fail("E_PRESENTATION_MEMBERSHIP", f"{label} does not cover exact tau_train")


def _validate_roles(record: dict[str, Any]) -> None:
    _fields(
        record,
        {
            "context_seal",
            "pair_seal",
            "manifest_sha256",
            "regular_only_exclusion_receipt_sha256",
            "cross_subject_alignment_receipt_sha256",
            "source_unique_rows",
            "target_unique_rows",
            "source_feature_rows_by_layer",
            "tau_train_ordered_ids",
            "tau_test_ordered_ids",
            "source_tau_presentations",
            "target_tau_presentations",
        },
        "structural binding",
    )
    _sha(record["context_seal"], "structural context seal")
    _sha(record["pair_seal"], "structural pair seal")
    _sha(record["manifest_sha256"], "structural manifest digest")
    _sha(record["regular_only_exclusion_receipt_sha256"], "regular-only receipt")
    if record["cross_subject_alignment_receipt_sha256"] is not None:
        _sha(record["cross_subject_alignment_receipt_sha256"], "alignment receipt")
    _, source_images = _validate_role_rows(record["source_unique_rows"], "source unique")
    _, target_images = _validate_role_rows(record["target_unique_rows"], "target unique")
    feature_rows = record["source_feature_rows_by_layer"]
    if not isinstance(feature_rows, dict) or not feature_rows:
        _fail("E_STRUCTURAL_TYPE", "source feature-row bindings must be a non-empty mapping")
    for layer, rows in feature_rows.items():
        _id(layer, "structural feature layer")
        _, feature_images = _validate_role_rows(rows, f"feature {layer}")
        if feature_images != source_images:
            _fail("E_CONTENT_FEATURE_ALIGNMENT", f"structural feature order differs for {layer}")
    tau_train = record["tau_train_ordered_ids"]
    tau_test = record["tau_test_ordered_ids"]
    if not isinstance(tau_train, list) or not tau_train or not isinstance(tau_test, list) or not tau_test:
        _fail("E_STRUCTURAL_TYPE", "tau role IDs must be non-empty lists")
    for value in tau_train + tau_test:
        _id(value, "tau image ID")
    role_sets = [set(source_images), set(target_images), set(tau_train), set(tau_test)]
    if any(len(values) != len(source) for values, source in zip(role_sets, [source_images, target_images, tau_train, tau_test])):
        _fail("E_ROLE_DUPLICATE", "role image IDs contain duplicates")
    for left in range(len(role_sets)):
        for right in range(left + 1, len(role_sets)):
            if role_sets[left] & role_sets[right]:
                _fail("E_ROLE_OVERLAP", "structural role image IDs overlap")
    _validate_presentations(record["source_tau_presentations"], "source tau", set(tau_train))
    _validate_presentations(record["target_tau_presentations"], "target tau", set(tau_train))


def _alignment_contract(context_record: Mapping[str, Any], roles_record: Mapping[str, Any]) -> None:
    mode = _settings(context_record)[1]
    layers = _settings(context_record)[0]
    if set(roles_record["source_feature_rows_by_layer"]) != set(layers):
        _fail("E_CONTENT_LAYER_SET", "structural feature layers differ from context")
    if mode != "authenticated_cross_subject_pairing":
        return
    if roles_record["cross_subject_alignment_receipt_sha256"] is None:
        _fail("E_ALIGNMENT_RECEIPT", "paired mode lacks an authenticated receipt")
    key_sets: list[set[tuple[str, str]]] = []
    for side in ("source", "target"):
        rows = roles_record[f"{side}_tau_presentations"]
        keys: list[tuple[str, str]] = []
        for row in rows:
            key = row["cross_subject_alignment_key"]
            if key is None:
                _fail("E_ALIGNMENT_KEY", f"{side} presentation lacks an alignment key")
            keys.append((row["image_id"], key))
        if len(set(keys)) != len(keys):
            _fail("E_ALIGNMENT_KEY", f"{side} alignment keys contain duplicates")
        key_sets.append(set(keys))
    if key_sets[0] != key_sets[1]:
        _fail("E_ALIGNMENT_MISMATCH", "source and target alignment keys differ")


@dataclass(frozen=True)
class StructuralRoleBinding:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, StructuralRoleBinding, "StructuralRoleBinding", _validate_roles)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, StructuralRoleBinding, "StructuralRoleBinding", _validate_roles)


def bind_structural_roles(
    context: SyntheticFixtureContext,
    pair: PairBinding,
    *,
    manifest_sha256: str,
    regular_only_exclusion_receipt_sha256: str,
    cross_subject_alignment_receipt_sha256: str | None,
    source_unique_rows: Sequence[Mapping[str, Any]],
    target_unique_rows: Sequence[Mapping[str, Any]],
    source_feature_rows_by_layer: Mapping[str, Sequence[Mapping[str, Any]]],
    tau_train_ordered_ids: Sequence[str],
    tau_test_ordered_ids: Sequence[str],
    source_tau_presentations: Sequence[Mapping[str, Any]],
    target_tau_presentations: Sequence[Mapping[str, Any]],
) -> StructuralRoleBinding:
    context_record = _context_record(context)
    _pair_for(context, pair)
    record = json.loads(
        _canonical(
            {
                "context_seal": context.seal,
                "pair_seal": pair.seal,
                "manifest_sha256": manifest_sha256,
                "regular_only_exclusion_receipt_sha256": regular_only_exclusion_receipt_sha256,
                "cross_subject_alignment_receipt_sha256": cross_subject_alignment_receipt_sha256,
                "source_unique_rows": list(source_unique_rows),
                "target_unique_rows": list(target_unique_rows),
                "source_feature_rows_by_layer": {
                    layer: list(rows)
                    for layer, rows in source_feature_rows_by_layer.items()
                },
                "tau_train_ordered_ids": list(tau_train_ordered_ids),
                "tau_test_ordered_ids": list(tau_test_ordered_ids),
                "source_tau_presentations": list(source_tau_presentations),
                "target_tau_presentations": list(target_tau_presentations),
            }
        ).decode("utf-8")
    )
    _validate_roles(record)
    _alignment_contract(context_record, record)
    return _build(StructuralRoleBinding, "StructuralRoleBinding", record)


def _roles_for(
    context: SyntheticFixtureContext, pair: PairBinding, roles: StructuralRoleBinding
) -> dict[str, Any]:
    context_record = _context_record(context)
    _pair_for(context, pair)
    record = _validated(
        roles, StructuralRoleBinding, "StructuralRoleBinding", _validate_roles
    )
    if record["context_seal"] != context.seal or record["pair_seal"] != pair.seal:
        _fail("E_STRUCTURAL_BINDING", "structural roles are bound elsewhere")
    _alignment_contract(context_record, record)
    return record


def _validate_rows(record: dict[str, Any]) -> None:
    _fields(
        record,
        {"structural_seal", "participant_id", "role", "row_ids", "image_ids", "shape"},
        "ordered rows",
    )
    _sha(record["structural_seal"], "row structural seal")
    _id(record["participant_id"], "row participant ID")
    if record["role"] not in {"source_unique_train", "tau_train"}:
        _fail("E_ROWS_ROLE", "ordered-row role is unsupported")
    row_ids, image_ids, shape = record["row_ids"], record["image_ids"], record["shape"]
    if not isinstance(row_ids, list) or not row_ids or not isinstance(image_ids, list) or len(row_ids) != len(image_ids):
        _fail("E_ROWS", "ordered row and image counts differ")
    for value in row_ids:
        _id(value, "ordered row ID")
    for value in image_ids:
        _id(value, "ordered image ID")
    if len(set(row_ids)) != len(row_ids):
        _fail("E_ROWS", "ordered row IDs contain duplicates")
    if not isinstance(shape, list) or not shape or shape[0] != len(row_ids):
        _fail("E_SHAPE", "ordered-row shape differs")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
        _fail("E_SHAPE", "ordered-row shape contains an invalid dimension")


@dataclass(frozen=True)
class OrderedRows:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, OrderedRows, "OrderedRows", _validate_rows)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, OrderedRows, "OrderedRows", _validate_rows)


def make_ordered_rows(
    roles: StructuralRoleBinding,
    *,
    participant_id: str,
    role: str,
    row_ids: Sequence[str],
    image_ids: Sequence[str],
    shape: Sequence[int],
) -> OrderedRows:
    _validated(roles, StructuralRoleBinding, "StructuralRoleBinding", _validate_roles)
    record = {
        "structural_seal": roles.seal,
        "participant_id": participant_id,
        "role": role,
        "row_ids": list(row_ids),
        "image_ids": list(image_ids),
        "shape": list(shape),
    }
    _validate_rows(record)
    return _build(OrderedRows, "OrderedRows", record)


def _validate_features(record: dict[str, Any]) -> None:
    _fields(record, {"structural_seal", "layer_id", "row_ids", "image_ids", "shape"}, "feature rows")
    _sha(record["structural_seal"], "feature structural seal")
    _id(record["layer_id"], "feature layer ID")
    proxy = {
        "structural_seal": record["structural_seal"],
        "participant_id": "feature-provider",
        "role": "source_unique_train",
        "row_ids": record["row_ids"],
        "image_ids": record["image_ids"],
        "shape": record["shape"],
    }
    _validate_rows(proxy)


@dataclass(frozen=True)
class FeatureRows:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, FeatureRows, "FeatureRows", _validate_features)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, FeatureRows, "FeatureRows", _validate_features)


def make_feature_rows(
    roles: StructuralRoleBinding,
    *,
    layer_id: str,
    row_ids: Sequence[str],
    image_ids: Sequence[str],
    shape: Sequence[int],
) -> FeatureRows:
    _validated(roles, StructuralRoleBinding, "StructuralRoleBinding", _validate_roles)
    record = {
        "structural_seal": roles.seal,
        "layer_id": layer_id,
        "row_ids": list(row_ids),
        "image_ids": list(image_ids),
        "shape": list(shape),
    }
    _validate_features(record)
    return _build(FeatureRows, "FeatureRows", record)


def _validate_alpha(record: dict[str, Any]) -> None:
    _fields(
        record,
        {
            "context_seal",
            "pair_seal",
            "pair_id",
            "policy_id",
            "policy_sha256",
            "selection_scope",
            "allowed_fit_role",
            "fixed_before_tau_test",
            "tau_test_or_ood_used",
            "resolved_value",
            "selection_receipt_sha256",
        },
        "alpha binding",
    )
    _sha(record["context_seal"], "alpha context seal")
    _sha(record["pair_seal"], "alpha pair seal")
    _id(record["pair_id"], "alpha pair ID")
    _validate_alpha_declaration(
        {field: record[field] for field in ALPHA_DECLARATION_FIELDS}
    )


@dataclass(frozen=True)
class AlphaBinding:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, AlphaBinding, "AlphaBinding", _validate_alpha)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, AlphaBinding, "AlphaBinding", _validate_alpha)


def bind_alpha(
    context: SyntheticFixtureContext,
    pair: PairBinding,
    *,
    policy_id: str,
    policy_sha256: str,
    selection_scope: str,
    allowed_fit_role: str,
    fixed_before_tau_test: bool,
    tau_test_or_ood_used: bool,
    resolved_value: float,
    selection_receipt_sha256: str | None,
) -> AlphaBinding:
    pair_record = _pair_for(context, pair)
    record = {
        "context_seal": context.seal,
        "pair_seal": pair.seal,
        "pair_id": pair_record["pair_id"],
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "selection_scope": selection_scope,
        "allowed_fit_role": allowed_fit_role,
        "fixed_before_tau_test": fixed_before_tau_test,
        "tau_test_or_ood_used": tau_test_or_ood_used,
        "resolved_value": resolved_value,
        "selection_receipt_sha256": selection_receipt_sha256,
    }
    _validate_alpha(record)
    declared = _context_record(context)["settings"]["brain_loss_alpha_bindings"].get(
        pair_record["pair_id"]
    )
    observed_declaration = {
        field: record[field] for field in ALPHA_DECLARATION_FIELDS
    }
    if _canonical(declared) != _canonical(observed_declaration):
        _fail("E_ALPHA_BINDING", "alpha differs from the sealed context declaration")
    return _build(AlphaBinding, "AlphaBinding", record)


def _alpha_for(context: SyntheticFixtureContext, pair: PairBinding, alpha: AlphaBinding) -> dict[str, Any]:
    pair_record = _pair_for(context, pair)
    record = _validated(alpha, AlphaBinding, "AlphaBinding", _validate_alpha)
    if (
        record["context_seal"] != context.seal
        or record["pair_seal"] != pair.seal
        or record["pair_id"] != pair_record["pair_id"]
    ):
        _fail("E_ALPHA_BINDING", "alpha is bound to a different context or pair")
    declared = _context_record(context)["settings"]["brain_loss_alpha_bindings"].get(
        pair_record["pair_id"]
    )
    observed_declaration = {
        field: record[field] for field in ALPHA_DECLARATION_FIELDS
    }
    if _canonical(declared) != _canonical(observed_declaration):
        _fail("E_ALPHA_BINDING", "alpha differs from the sealed context declaration")
    return record


def _validate_content_plan(record: dict[str, Any]) -> None:
    _fields(
        record,
        {
            "schema",
            "context_seal",
            "pair_seal",
            "roles_seal",
            "worker_seal",
            "pair_id",
            "source_row_count",
            "response_rows_seal",
            "response_shape",
            "response_ordered_row_sha256",
            "response_ordered_image_sha256",
            "feature_rows",
            "source_ordered_image_sha256",
        },
        "content plan",
    )
    if record["schema"] != "ep21.synthetic_content_plan.v1":
        _fail("E_CONTENT_PLAN", "content-plan schema differs")
    for field in (
        "context_seal", "pair_seal", "roles_seal", "worker_seal",
        "response_rows_seal", "response_ordered_row_sha256",
        "response_ordered_image_sha256", "source_ordered_image_sha256",
    ):
        _sha(record[field], field)
    _id(record["pair_id"], "content pair ID")
    count = _positive(record["source_row_count"], "content source-row count")
    shape = record["response_shape"]
    if not isinstance(shape, list) or not shape or shape[0] != count:
        _fail("E_CONTENT_PLAN", "content response shape differs")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in shape
    ):
        _fail("E_CONTENT_PLAN", "content response shape has an invalid dimension")
    features = record["feature_rows"]
    if not isinstance(features, list) or not features:
        _fail("E_CONTENT_PLAN", "content feature rows are absent")
    layers: list[str] = []
    for item in features:
        if not isinstance(item, dict) or set(item) != {
            "layer_id", "rows_seal", "dynamic_buffer_shape",
            "ordered_row_sha256", "ordered_image_sha256",
        }:
            _fail("E_CONTENT_PLAN", "content feature-plan fields differ")
        layers.append(_id(item["layer_id"], "content layer"))
        _sha(item["rows_seal"], "feature rows seal")
        _sha(item["ordered_row_sha256"], "feature ordered-row digest")
        _sha(item["ordered_image_sha256"], "feature ordered-image digest")
        feature_shape = item["dynamic_buffer_shape"]
        if not isinstance(feature_shape, list) or not feature_shape or feature_shape[0] != count:
            _fail("E_CONTENT_PLAN", "dynamic content shape differs")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in feature_shape
        ):
            _fail("E_CONTENT_PLAN", "dynamic content shape has an invalid dimension")
    if len(set(layers)) != len(layers):
        _fail("E_CONTENT_PLAN", "content-plan layers contain duplicates")


@dataclass(frozen=True)
class ContentPlan:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, ContentPlan, "ContentPlan", _validate_content_plan)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, ContentPlan, "ContentPlan", _validate_content_plan)


def plan_content(
    context: SyntheticFixtureContext,
    pair: PairBinding,
    roles: StructuralRoleBinding,
    worker: WorkerBinding,
    *,
    response_rows: OrderedRows,
    feature_rows_by_layer: Mapping[str, FeatureRows],
) -> ContentPlan:
    context_record = _context_record(context)
    pair_record = _pair_for(context, pair)
    roles_record = _roles_for(context, pair, roles)
    worker_record = _worker_for(context, worker)
    del worker_record
    response = _validated(response_rows, OrderedRows, "OrderedRows", _validate_rows)
    source_rows = roles_record["source_unique_rows"]
    expected_row_ids = [row["row_id"] for row in source_rows]
    expected_image_ids = [row["image_id"] for row in source_rows]
    if (
        response["structural_seal"] != roles.seal
        or response["participant_id"] != pair_record["source_id"]
        or response["role"] != "source_unique_train"
        or response["row_ids"] != expected_row_ids
        or response["image_ids"] != expected_image_ids
    ):
        _fail("E_CONTENT_RESPONSE_ALIGNMENT", "response rows differ from exact source order")
    locked_layers = list(_settings(context_record)[0])
    if not isinstance(feature_rows_by_layer, Mapping) or set(feature_rows_by_layer) != set(locked_layers):
        _fail("E_CONTENT_LAYER_SET", "content feature layer set differs")
    features: list[dict[str, Any]] = []
    for layer in locked_layers:
        feature_object = feature_rows_by_layer[layer]
        feature = _validated(
            feature_object, FeatureRows, "FeatureRows", _validate_features
        )
        expected_feature_rows = roles_record["source_feature_rows_by_layer"][layer]
        if (
            feature["structural_seal"] != roles.seal
            or feature["layer_id"] != layer
            or feature["row_ids"]
            != [row["row_id"] for row in expected_feature_rows]
            or feature["image_ids"]
            != [row["image_id"] for row in expected_feature_rows]
        ):
            _fail("E_CONTENT_FEATURE_ALIGNMENT", f"feature row order differs for {layer}")
        features.append(
            {
                "layer_id": layer,
                "rows_seal": feature_object.seal,
                "dynamic_buffer_shape": feature["shape"],
                "ordered_row_sha256": metadata_sha256(feature["row_ids"]),
                "ordered_image_sha256": metadata_sha256(feature["image_ids"]),
            }
        )
    record = {
        "schema": "ep21.synthetic_content_plan.v1",
        "context_seal": context.seal,
        "pair_seal": pair.seal,
        "roles_seal": roles.seal,
        "worker_seal": worker.seal,
        "pair_id": pair_record["pair_id"],
        "source_row_count": len(expected_image_ids),
        "response_rows_seal": response_rows.seal,
        "response_shape": response["shape"],
        "response_ordered_row_sha256": metadata_sha256(response["row_ids"]),
        "response_ordered_image_sha256": metadata_sha256(response["image_ids"]),
        "feature_rows": features,
        "source_ordered_image_sha256": metadata_sha256(expected_image_ids),
    }
    _validate_content_plan(record)
    return _build(ContentPlan, "ContentPlan", record)


def _validate_brain_plan(record: dict[str, Any]) -> None:
    _fields(
        record,
        {
            "schema",
            "context_seal",
            "pair_seal",
            "roles_seal",
            "worker_seal",
            "alpha_seal",
            "pair_id",
            "repetition_mode",
            "source_row_count",
            "target_row_count",
            "fit_unit_keys",
            "source_group_indices",
            "target_group_indices",
            "source_aligned_indices",
            "target_aligned_indices",
            "source_rows_seal",
            "target_rows_seal",
            "source_ordered_row_sha256",
            "source_ordered_image_sha256",
            "target_ordered_row_sha256",
            "target_ordered_image_sha256",
            "alpha_binding_record",
        },
        "brain plan",
    )
    if record["schema"] != "ep21.synthetic_brain_plan.v1":
        _fail("E_BRAIN_PLAN", "brain-plan schema differs")
    for field in (
        "context_seal", "pair_seal", "roles_seal", "worker_seal",
        "alpha_seal", "source_rows_seal", "target_rows_seal",
        "source_ordered_row_sha256", "source_ordered_image_sha256",
        "target_ordered_row_sha256", "target_ordered_image_sha256",
    ):
        _sha(record[field], field)
    _id(record["pair_id"], "brain pair ID")
    source_count = _positive(record["source_row_count"], "brain source count")
    target_count = _positive(record["target_row_count"], "brain target count")
    mode = record["repetition_mode"]
    if mode not in REPETITION_MODES:
        _fail("E_REPETITION_MODE", "brain-plan repetition mode differs")
    keys = record["fit_unit_keys"]
    if not isinstance(keys, list) or not keys:
        _fail("E_BRAIN_PLAN", "brain fit-unit keys are absent")
    normalized_keys: list[tuple[str, str]] = []
    for key in keys:
        if not isinstance(key, list) or len(key) != 2:
            _fail("E_BRAIN_PLAN", "brain fit-unit key differs")
        normalized_keys.append((_id(key[0], "fit image ID"), _id(key[1], "fit alignment label")))
    if len(set(normalized_keys)) != len(normalized_keys):
        _fail("E_BRAIN_PLAN", "brain fit-unit keys contain duplicates")
    source_groups, target_groups = record["source_group_indices"], record["target_group_indices"]
    source_aligned, target_aligned = record["source_aligned_indices"], record["target_aligned_indices"]

    def require_indices(values: Any, label: str) -> list[int]:
        if not isinstance(values, list) or any(type(index) is not int for index in values):
            _fail("E_BRAIN_PLAN", f"{label} indices are not strict integers")
        return values

    for values, label in (
        (source_groups, "source groups"),
        (target_groups, "target groups"),
        (source_aligned, "source aligned"),
        (target_aligned, "target aligned"),
    ):
        if not isinstance(values, list):
            _fail("E_BRAIN_PLAN", f"{label} container is not a list")
    if mode == "image_average_before_fit":
        if len(source_groups) != len(keys) or len(target_groups) != len(keys) or source_aligned or target_aligned:
            _fail("E_BRAIN_PLAN", "image-average index layout differs")
        if any(not isinstance(group, list) or not group for group in source_groups + target_groups):
            _fail("E_BRAIN_PLAN", "image-average group is invalid")
        for index, group in enumerate(source_groups):
            require_indices(group, f"source group {index}")
        for index, group in enumerate(target_groups):
            require_indices(group, f"target group {index}")
        if sorted(index for group in source_groups for index in group) != list(range(source_count)):
            _fail("E_BRAIN_PLAN", "source averaging groups are not exhaustive")
        if sorted(index for group in target_groups for index in group) != list(range(target_count)):
            _fail("E_BRAIN_PLAN", "target averaging groups are not exhaustive")
    else:
        if source_groups or target_groups:
            _fail("E_BRAIN_PLAN", "paired plan contains averaging groups")
        require_indices(source_aligned, "source aligned")
        require_indices(target_aligned, "target aligned")
        if sorted(source_aligned) != list(range(source_count)):
            _fail("E_BRAIN_PLAN", "source paired indices are not exhaustive")
        if sorted(target_aligned) != list(range(target_count)) or len(keys) != source_count or len(keys) != target_count:
            _fail("E_BRAIN_PLAN", "target paired indices are not exhaustive")
    alpha_record = record["alpha_binding_record"]
    if not isinstance(alpha_record, dict):
        _fail("E_BRAIN_PLAN", "alpha binding record is absent")
    _validate_alpha(alpha_record)
    if alpha_record["pair_id"] != record["pair_id"]:
        _fail("E_BRAIN_PLAN", "alpha pair differs from brain plan")


@dataclass(frozen=True)
class BrainPlan:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, BrainPlan, "BrainPlan", _validate_brain_plan)

    @property
    def record(self) -> dict[str, Any]:
        return _validated(self, BrainPlan, "BrainPlan", _validate_brain_plan)


def _expected_brain_layout(
    context_record: Mapping[str, Any], roles_record: Mapping[str, Any]
) -> dict[str, Any]:
    mode = _settings(context_record)[1]
    tau_order = roles_record["tau_train_ordered_ids"]
    if mode == "image_average_before_fit":
        return {
            "fit_unit_keys": [[image_id, "post_average"] for image_id in tau_order],
            "source_group_indices": [
                [
                    index
                    for index, row in enumerate(roles_record["source_tau_presentations"])
                    if row["image_id"] == image_id
                ]
                for image_id in tau_order
            ],
            "target_group_indices": [
                [
                    index
                    for index, row in enumerate(roles_record["target_tau_presentations"])
                    if row["image_id"] == image_id
                ]
                for image_id in tau_order
            ],
            "source_aligned_indices": [],
            "target_aligned_indices": [],
        }
    _alignment_contract(context_record, roles_record)
    source_index = {
        (row["image_id"], row["cross_subject_alignment_key"]): index
        for index, row in enumerate(roles_record["source_tau_presentations"])
    }
    target_index = {
        (row["image_id"], row["cross_subject_alignment_key"]): index
        for index, row in enumerate(roles_record["target_tau_presentations"])
    }
    keys = [
        [image_id, alignment]
        for image_id in tau_order
        for alignment in sorted(
            key[1] for key in source_index if key[0] == image_id
        )
    ]
    return {
        "fit_unit_keys": keys,
        "source_group_indices": [],
        "target_group_indices": [],
        "source_aligned_indices": [
            source_index[(key[0], key[1])] for key in keys
        ],
        "target_aligned_indices": [
            target_index[(key[0], key[1])] for key in keys
        ],
    }


def plan_brain_loss(
    context: SyntheticFixtureContext,
    pair: PairBinding,
    roles: StructuralRoleBinding,
    worker: WorkerBinding,
    *,
    source_rows: OrderedRows,
    target_rows: OrderedRows,
    alpha: AlphaBinding,
) -> BrainPlan:
    context_record = _context_record(context)
    pair_record = _pair_for(context, pair)
    roles_record = _roles_for(context, pair, roles)
    _worker_for(context, worker)
    alpha_record = _alpha_for(context, pair, alpha)
    source = _validated(source_rows, OrderedRows, "OrderedRows", _validate_rows)
    target = _validated(target_rows, OrderedRows, "OrderedRows", _validate_rows)
    for side, observed, participant in (
        ("source", source, pair_record["source_id"]),
        ("target", target, pair_record["target_id"]),
    ):
        presentations = roles_record[f"{side}_tau_presentations"]
        if (
            observed["structural_seal"] != roles.seal
            or observed["participant_id"] != participant
            or observed["role"] != "tau_train"
            or observed["row_ids"] != [row["row_id"] for row in presentations]
            or observed["image_ids"] != [row["image_id"] for row in presentations]
        ):
            _fail("E_BRAIN_PRESENTATION_ALIGNMENT", f"{side} rows differ from authenticated presentations")
    mode = _settings(context_record)[1]
    layout = _expected_brain_layout(context_record, roles_record)
    record = {
        "schema": "ep21.synthetic_brain_plan.v1",
        "context_seal": context.seal,
        "pair_seal": pair.seal,
        "roles_seal": roles.seal,
        "worker_seal": worker.seal,
        "alpha_seal": alpha.seal,
        "pair_id": pair_record["pair_id"],
        "repetition_mode": mode,
        "source_row_count": len(source["row_ids"]),
        "target_row_count": len(target["row_ids"]),
        **layout,
        "source_rows_seal": source_rows.seal,
        "target_rows_seal": target_rows.seal,
        "source_ordered_row_sha256": metadata_sha256(source["row_ids"]),
        "source_ordered_image_sha256": metadata_sha256(source["image_ids"]),
        "target_ordered_row_sha256": metadata_sha256(target["row_ids"]),
        "target_ordered_image_sha256": metadata_sha256(target["image_ids"]),
        "alpha_binding_record": alpha_record,
    }
    _validate_brain_plan(record)
    return _build(BrainPlan, "BrainPlan", record)


def _validate_summary(record: dict[str, Any]) -> None:
    _fields(
        record,
        {
            "schema", "context_schema", "context_seal", "settings_sha256", "pair_id",
            "pair_seal", "roles_seal", "worker_seal", "content_plan_seal",
            "brain_plan_seal", "content_row_count", "brain_fit_unit_count",
            "max_workers", "explicit_worker_limit", "repetition_mode",
            "synthetic_only", "empirical_capable",
        },
        "plan summary",
    )
    if record["schema"] != "ep21.synthetic_adapter_summary.v1" or record["context_schema"] != SYNTHETIC_SCHEMA:
        _fail("E_SUMMARY", "summary schema differs")
    for field in ("context_seal", "settings_sha256", "pair_seal", "roles_seal", "worker_seal", "content_plan_seal", "brain_plan_seal"):
        _sha(record[field], field)
    _id(record["pair_id"], "summary pair ID")
    _positive(record["content_row_count"], "summary content rows")
    _positive(record["brain_fit_unit_count"], "summary brain fit units")
    _positive(record["max_workers"], "summary max workers")
    _positive(record["explicit_worker_limit"], "summary explicit worker limit")
    if record["max_workers"] != record["explicit_worker_limit"]:
        _fail("E_SUMMARY", "summary workers differ from the explicit limit")
    if record["repetition_mode"] not in REPETITION_MODES:
        _fail("E_SUMMARY", "summary repetition mode differs")
    if record["synthetic_only"] is not True or record["empirical_capable"] is not False:
        _fail("E_SUMMARY_CAPABILITY", "summary is not synthetic-only")


@dataclass(frozen=True)
class PlanSummary:
    canonical_record_json: bytes
    seal: str

    def __post_init__(self) -> None:
        _validated(self, PlanSummary, "PlanSummary", _validate_summary)


def summarize_plans(
    context: SyntheticFixtureContext,
    pair: PairBinding,
    roles: StructuralRoleBinding,
    worker: WorkerBinding,
    content_plan: ContentPlan,
    brain_plan: BrainPlan,
    alpha: AlphaBinding,
    *,
    response_rows: OrderedRows,
    feature_rows_by_layer: Mapping[str, FeatureRows],
    source_rows: OrderedRows,
    target_rows: OrderedRows,
) -> PlanSummary:
    context_record = _context_record(context)
    pair_record = _pair_for(context, pair)
    roles_record = _roles_for(context, pair, roles)
    worker_record = _worker_for(context, worker)
    alpha_record = _alpha_for(context, pair, alpha)
    content = _validated(
        content_plan, ContentPlan, "ContentPlan", _validate_content_plan
    )
    brain = _validated(brain_plan, BrainPlan, "BrainPlan", _validate_brain_plan)
    regenerated_content = plan_content(
        context,
        pair,
        roles,
        worker,
        response_rows=response_rows,
        feature_rows_by_layer=feature_rows_by_layer,
    )
    regenerated_brain = plan_brain_loss(
        context,
        pair,
        roles,
        worker,
        source_rows=source_rows,
        target_rows=target_rows,
        alpha=alpha,
    )
    if (
        content_plan.canonical_record_json != regenerated_content.canonical_record_json
        or content_plan.seal != regenerated_content.seal
    ):
        _fail("E_CONTENT_PLAN_BINDING", "content plan differs from regenerated row inputs")
    if (
        brain_plan.canonical_record_json != regenerated_brain.canonical_record_json
        or brain_plan.seal != regenerated_brain.seal
    ):
        _fail("E_BRAIN_PLAN_BINDING", "brain plan differs from regenerated row inputs")
    expected = (context.seal, pair.seal, roles.seal, worker.seal, pair_record["pair_id"])
    if tuple(content[field] for field in ("context_seal", "pair_seal", "roles_seal", "worker_seal", "pair_id")) != expected:
        _fail("E_PLAN_BINDING", "content-plan bindings differ")
    if tuple(brain[field] for field in ("context_seal", "pair_seal", "roles_seal", "worker_seal", "pair_id")) != expected:
        _fail("E_PLAN_BINDING", "brain-plan bindings differ")
    if brain["repetition_mode"] != _settings(context_record)[1]:
        _fail("E_REPETITION_MODE", "brain-plan mode differs from context")
    source_rows = roles_record["source_unique_rows"]
    source_row_ids = [row["row_id"] for row in source_rows]
    source_image_ids = [row["image_id"] for row in source_rows]
    if (
        content["source_row_count"] != len(source_rows)
        or content["response_ordered_row_sha256"]
        != metadata_sha256(source_row_ids)
        or content["response_ordered_image_sha256"]
        != metadata_sha256(source_image_ids)
        or content["source_ordered_image_sha256"]
        != metadata_sha256(source_image_ids)
    ):
        _fail("E_CONTENT_PLAN_BINDING", "content rows differ from structural roles")
    locked_layers = list(_settings(context_record)[0])
    if [item["layer_id"] for item in content["feature_rows"]] != locked_layers:
        _fail("E_CONTENT_PLAN_BINDING", "content layer order differs from context")
    for item in content["feature_rows"]:
        expected_feature_rows = roles_record["source_feature_rows_by_layer"][
            item["layer_id"]
        ]
        if (
            item["ordered_row_sha256"]
            != metadata_sha256([row["row_id"] for row in expected_feature_rows])
            or item["ordered_image_sha256"]
            != metadata_sha256([row["image_id"] for row in expected_feature_rows])
            or item["dynamic_buffer_shape"][0] != len(source_rows)
        ):
            _fail("E_CONTENT_PLAN_BINDING", "content feature rows differ from structural roles")
    source_presentations = roles_record["source_tau_presentations"]
    target_presentations = roles_record["target_tau_presentations"]
    expected_layout = _expected_brain_layout(context_record, roles_record)
    if (
        brain["source_row_count"] != len(source_presentations)
        or brain["target_row_count"] != len(target_presentations)
        or brain["source_ordered_row_sha256"]
        != metadata_sha256([row["row_id"] for row in source_presentations])
        or brain["source_ordered_image_sha256"]
        != metadata_sha256([row["image_id"] for row in source_presentations])
        or brain["target_ordered_row_sha256"]
        != metadata_sha256([row["row_id"] for row in target_presentations])
        or brain["target_ordered_image_sha256"]
        != metadata_sha256([row["image_id"] for row in target_presentations])
        or any(brain[field] != value for field, value in expected_layout.items())
    ):
        _fail("E_BRAIN_PLAN_BINDING", "brain plan differs from structural roles")
    if brain["alpha_seal"] != alpha.seal or brain["alpha_binding_record"] != alpha_record:
        _fail("E_BRAIN_PLAN_BINDING", "brain plan alpha binding differs")
    record = {
        "schema": "ep21.synthetic_adapter_summary.v1",
        "context_schema": SYNTHETIC_SCHEMA,
        "context_seal": context.seal,
        "settings_sha256": context_record["settings_sha256"],
        "pair_id": pair_record["pair_id"],
        "pair_seal": pair.seal,
        "roles_seal": roles.seal,
        "worker_seal": worker.seal,
        "content_plan_seal": content_plan.seal,
        "brain_plan_seal": brain_plan.seal,
        "content_row_count": content["source_row_count"],
        "brain_fit_unit_count": len(brain["fit_unit_keys"]),
        "max_workers": worker_record["max_workers"],
        "explicit_worker_limit": worker_record["explicit_worker_limit"],
        "repetition_mode": brain["repetition_mode"],
        "synthetic_only": True,
        "empirical_capable": False,
    }
    _validate_summary(record)
    return _build(PlanSummary, "PlanSummary", record)


def export_summary(
    summary: PlanSummary,
    context: SyntheticFixtureContext,
    pair: PairBinding,
    roles: StructuralRoleBinding,
    worker: WorkerBinding,
    content_plan: ContentPlan,
    brain_plan: BrainPlan,
    alpha: AlphaBinding,
    *,
    response_rows: OrderedRows,
    feature_rows_by_layer: Mapping[str, FeatureRows],
    source_rows: OrderedRows,
    target_rows: OrderedRows,
) -> dict[str, Any]:
    """Export only after regenerating the summary from every sealed input."""
    observed = _validated(summary, PlanSummary, "PlanSummary", _validate_summary)
    regenerated = summarize_plans(
        context,
        pair,
        roles,
        worker,
        content_plan,
        brain_plan,
        alpha,
        response_rows=response_rows,
        feature_rows_by_layer=feature_rows_by_layer,
        source_rows=source_rows,
        target_rows=target_rows,
    )
    if (
        summary.canonical_record_json != regenerated.canonical_record_json
        or summary.seal != regenerated.seal
    ):
        _fail("E_SUMMARY_BINDING", "summary differs from regenerated plan inputs")
    return observed
