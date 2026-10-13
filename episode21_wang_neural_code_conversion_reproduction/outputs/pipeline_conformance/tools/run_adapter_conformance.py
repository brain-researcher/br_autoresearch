#!/usr/bin/env python3
"""Run EP21's payload-free synthetic execution-adapter conformance audit.

The runner binds frozen protocol-neutral evidence and exercises only synthetic
identifier, role, shape, and presentation metadata.  It does not read a model,
array, neural value, image, empirical/OOD handoff, EP04 object, or legacy
Scratch object.  It never creates a device, executor, optimizer, or protocol
lock.  The real empirical adapter entrypoint is required to remain unavailable.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping


EPISODE_DIR = Path(__file__).resolve().parents[3]
ADAPTER_PATH = EPISODE_DIR / "outputs/pipeline_conformance/tools/execution_adapter.py"
PUBLICATION_HELPER_PATH = (
    EPISODE_DIR / "outputs/source_qualification/tools/audit_public_method_semantics.py"
)
NEUTRAL_REPORT = (
    EPISODE_DIR / "outputs/pipeline_conformance/attempt-005/neutral_conformance_report.json"
)
METHOD_REPORT = (
    EPISODE_DIR / "outputs/source_qualification/attempt-019/public_method_semantics_audit.json"
)
PROTOCOL_TABLE = EPISODE_DIR / "PROTOCOL_TABLE.yaml"
OUTPUT_DIR = EPISODE_DIR / "outputs/pipeline_conformance/attempt-006"
OUTPUT_NAME = "adapter_conformance_report.json"
ATTENDED_LOGIN_CONTEXT = "attended-login"

ADAPTER_SHA256 = "2e269d5fe53933523871a613e84ee621c6058b05a769085c943166cae70b8ed8"
PUBLICATION_HELPER_SHA256 = "122f44866ba95a6eddf24a79d7a093dd4d748ed82b5f9fe09dec443aadc5ee46"
NEUTRAL_REPORT_SHA256 = "54fb25d6ffeb8197fde920f631c10258e1d2c82709518d4c2e958d8da2e4a187"
METHOD_REPORT_SHA256 = "3840e33fe22076607ff4c4f8e91bc30b781284f8cd2e804df4055137d2ee1b74"
PROTOCOL_TABLE_SHA256 = "c12b12b172c5e9916c4843d1b287b16c5831a29736c97a0f1cca2ad66bb68f82"
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_PROTOCOL_BYTES = 2 * 1024 * 1024
MAX_CODE_BYTES = 2 * 1024 * 1024

GATE_PATHS = {
    "protocol_lock": EPISODE_DIR / "outputs/protocol_lock.yaml",
    "laion_regular_feature_handoff": EPISODE_DIR / "inputs/laion_regular_feature_handoff",
    "laion_regular_scoring_handoff": EPISODE_DIR / "inputs/laion_regular_scoring_handoff",
    "external_site_feature_handoff": EPISODE_DIR / "inputs/external_site_feature_handoff",
    "external_site_scoring_handoff": EPISODE_DIR / "inputs/external_site_scoring_handoff",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def validate_attended_execution_context(
    execution_context: str, max_workers: int
) -> dict[str, Any]:
    if execution_context != ATTENDED_LOGIN_CONTEXT:
        raise ValueError("attempt 6 requires the attended-login execution context")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers != 1:
        raise ValueError("attempt 6 requires exactly one explicit worker")
    if os.environ.get("SLURM_JOB_ID"):
        raise ValueError("attended-login execution rejects an active Slurm allocation")
    hostname = os.uname().nodename
    short_hostname = hostname.split(".", 1)[0]
    if "-ln" not in short_hostname:
        raise ValueError("attended-login execution requires a Sherlock login node")
    return {
        "kind": ATTENDED_LOGIN_CONTEXT,
        "hostname": hostname,
        "max_workers": max_workers,
        "slurm_allocation_present": False,
        "synchronous_foreground_required": True,
    }


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def open_nofollow(path: Path, flags: int) -> int:
    """Open an absolute path through held no-follow directory descriptors."""
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError(f"path is not a fixed absolute path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open("/", directory_flags)
    try:
        for component in path.parts[1:-1]:
            if component in {"", ".", ".."}:
                raise ValueError(f"unsafe path component in {path}")
            following = os.open(component, directory_flags, dir_fd=current)
            os.close(current)
            current = following
        return os.open(
            path.name,
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
    finally:
        os.close(current)


def read_stable_bytes(
    path: Path, *, label: str, maximum_bytes: int, expected_sha256: str
) -> tuple[bytes, dict[str, Any]]:
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"{label} expected SHA256 is not canonical")
    descriptor = open_nofollow(
        path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} is not a single-link regular file")
        if before.st_size <= 0 or before.st_size > maximum_bytes:
            raise ValueError(f"{label} has an invalid size")
        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > maximum_bytes:
                raise ValueError(f"{label} exceeded its size bound")
        after = os.fstat(descriptor)
        if _identity(after) != _identity(before) or count != before.st_size:
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    return payload, {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": observed,
        "regular_file": True,
        "single_link": True,
    }


def load_verified_module(
    raw: bytes, *, path: Path, module_name: str
) -> types.ModuleType:
    """Execute only already-verified local code bytes, never a path re-read."""
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(raw, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def adapter_static_audit(raw: bytes) -> dict[str, Any]:
    source = raw.decode("utf-8", errors="strict")
    tree = ast.parse(source)
    imports: set[str] = set()
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            called = None
            direct_name = False
            if isinstance(node.func, ast.Name):
                called = node.func.id
                direct_name = True
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called in {
                "open", "exec", "eval", "__import__", "system",
                "Popen", "run", "ProcessPoolExecutor", "ThreadPoolExecutor",
            } or (direct_name and called == "compile"):
                forbidden_calls.append(str(called))
    allowed_imports = {
        "__future__", "hashlib", "json", "math", "re", "dataclasses", "typing"
    }
    if imports != allowed_imports:
        raise ValueError(f"adapter import boundary differs: {sorted(imports)}")
    if forbidden_calls:
        raise ValueError(f"adapter contains forbidden calls: {sorted(forbidden_calls)}")
    for token in ("torch", "numpy", "payload_path", "device=", "executor="):
        if token in source:
            raise ValueError(f"adapter contains forbidden token {token!r}")
    return {
        "standard_library_import_roots": sorted(imports),
        "forbidden_call_count": 0,
        "model_or_array_import_present": False,
        "file_or_network_io_call_present": False,
        "device_or_executor_constructor_present": False,
    }


def validate_upstream_bindings(
    neutral: Mapping[str, Any], method: Mapping[str, Any], protocol_text: str
) -> None:
    if (
        neutral.get("schema_version") != "ep21.neutral_conformance_report.v1"
        or neutral.get("attempt") != 5
        or neutral.get("outcome")
        != "passed_protocol_neutral_checks_with_protocol_and_access_blockers"
        or neutral.get("synthetic_only") is not True
        or neutral.get("protocol_lock_emitted") is not False
        or neutral.get("empirical_readiness") != "blocked"
        or neutral.get("payload_accessed") is not False
        or neutral.get("empirical_or_ood_payload_accessed") is not False
    ):
        raise ValueError("attempt-5 neutral-conformance semantics changed")
    if neutral.get("protocol_table", {}).get("sha256") != PROTOCOL_TABLE_SHA256:
        raise ValueError("attempt-5 protocol-table binding changed")
    if (
        method.get("schema_version") != "ep21.public_method_semantics_audit.v1"
        or method.get("attempt") != 19
        or method.get("outcome")
        != "completed_tagged_source_method_facts_with_choices_and_dependencies_unresolved"
    ):
        raise ValueError("attempt-19 method-report semantics changed")
    boundary = method.get("boundary")
    if not isinstance(boundary, Mapping) or not boundary or any(value is not False for value in boundary.values()):
        raise ValueError("attempt-19 closed-boundary predicates changed")
    effect = method.get("protocol_effect", {})
    required_false = (
        "scientific_choice_adopted",
        "dependency_semantics_fully_authenticated",
        "protocol_lock_emitted",
        "empirical_access_authorized",
        "empirical_readiness",
    )
    if any(effect.get(field) is not False for field in required_false):
        raise ValueError("attempt-19 retained blockers changed")
    required_protocol_fragments = (
        "empirical_execution_authorized: false",
        "pair_id_format: unresolved_deterministic_source_to_target_id_required_before_lock",
        "repetition_aggregation_order: unresolved_required_before_lock",
        "protocol_lock_exists: false",
        "neural_access_or_execution_authorized: false",
    )
    for fragment in required_protocol_fragments:
        if protocol_text.count(fragment) != 1:
            raise ValueError(f"protocol-table binding changed for {fragment!r}")


def final_entry_absent(path: Path) -> bool:
    """Check final-entry metadata only; never open a protected handoff."""
    parent_descriptor = open_nofollow(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    finally:
        os.close(parent_descriptor)


def assert_current_gate_closed() -> dict[str, str]:
    state: dict[str, str] = {}
    for label, path in GATE_PATHS.items():
        if not final_entry_absent(path):
            raise ValueError(f"current gate unexpectedly present: {label}")
        state[label] = "absent"
    return state


def fixture_bundle(
    adapter: types.ModuleType, mode: str, content_count: int, worker_limit: int
) -> dict[str, Any]:
    pair_core = {
        "pair_id": "opaque-fixture-pair-7",
        "source_id": "synthetic-participant-A",
        "target_id": "synthetic-participant-B",
        "ordinal": 7,
    }
    pair_record = {
        **pair_core,
        "pair_table_sha256": adapter.metadata_sha256([pair_core]),
    }
    selected_on_tau_train = mode == "authenticated_cross_subject_pairing"
    alpha_declaration = {
        "policy_id": (
            "synthetic-tau-train-selection"
            if selected_on_tau_train
            else "synthetic-fixed-alpha"
        ),
        "policy_sha256": "4" * 64,
        "selection_scope": (
            "tau_train_only" if selected_on_tau_train else "fixed_predeclared"
        ),
        "allowed_fit_role": "tau_train",
        "fixed_before_tau_test": True,
        "tau_test_or_ood_used": False,
        "resolved_value": 1.0,
        "selection_receipt_sha256": "5" * 64 if selected_on_tau_train else None,
    }
    context = adapter.make_synthetic_context(
        fixture_id=f"attempt6-{mode}-{content_count}",
        pair_table_records=[pair_core],
        settings={
            "content_feature_layers": ["synthetic-layer-conv", "synthetic-layer-fc"],
            "brain_loss_repetition_mode": mode,
            "brain_loss_alpha_bindings": {
                pair_core["pair_id"]: alpha_declaration
            },
        },
    )
    pair = adapter.bind_pair(context, pair_record)
    source_unique = [
        {"row_id": f"source-row-{index}", "image_id": f"source-image-{index}"}
        for index in range(content_count)
    ]
    target_unique = [
        {"row_id": f"target-row-{index}", "image_id": f"target-image-{index}"}
        for index in range(content_count + 1)
    ]
    tau_train = ["tau-image-0", "tau-image-1", "tau-image-2"]
    tau_test = ["tau-test-image-0", "tau-test-image-1"]
    source_tau: list[dict[str, Any]] = []
    target_tau: list[dict[str, Any]] = []
    for image_index, image_id in enumerate(tau_train):
        source_repetitions = 2
        target_repetitions = 2 if mode == "authenticated_cross_subject_pairing" else image_index + 1
        for repetition in range(source_repetitions):
            source_tau.append(
                {
                    "row_id": f"source-tau-{image_index}-{repetition}",
                    "image_id": image_id,
                    "local_repetition_id": f"source-local-{repetition}",
                    "cross_subject_alignment_key": (
                        f"alignment-{repetition}"
                        if mode == "authenticated_cross_subject_pairing"
                        else None
                    ),
                }
            )
        for repetition in reversed(range(target_repetitions)):
            target_tau.append(
                {
                    "row_id": f"target-tau-{image_index}-{repetition}",
                    "image_id": image_id,
                    "local_repetition_id": f"target-local-{repetition}",
                    "cross_subject_alignment_key": (
                        f"alignment-{repetition}"
                        if mode == "authenticated_cross_subject_pairing"
                        else None
                    ),
                }
            )
    roles = adapter.bind_structural_roles(
        context,
        pair,
        manifest_sha256="1" * 64,
        regular_only_exclusion_receipt_sha256="2" * 64,
        cross_subject_alignment_receipt_sha256=(
            "3" * 64 if mode == "authenticated_cross_subject_pairing" else None
        ),
        source_unique_rows=source_unique,
        target_unique_rows=target_unique,
        source_feature_rows_by_layer={
            "synthetic-layer-conv": [
                {"row_id": f"conv-row-{index}", "image_id": row["image_id"]}
                for index, row in enumerate(source_unique)
            ],
            "synthetic-layer-fc": [
                {"row_id": f"fc-row-{index}", "image_id": row["image_id"]}
                for index, row in enumerate(source_unique)
            ],
        },
        tau_train_ordered_ids=tau_train,
        tau_test_ordered_ids=tau_test,
        source_tau_presentations=source_tau,
        target_tau_presentations=target_tau,
    )
    worker = adapter.bind_worker(
        context, max_workers=1, explicit_worker_limit=worker_limit
    )
    response = adapter.make_ordered_rows(
        roles,
        participant_id=pair_record["source_id"],
        role="source_unique_train",
        row_ids=[row["row_id"] for row in source_unique],
        image_ids=[row["image_id"] for row in source_unique],
        shape=[content_count, 11],
    )
    features = {
        "synthetic-layer-conv": adapter.make_feature_rows(
            roles,
            layer_id="synthetic-layer-conv",
            row_ids=[f"conv-row-{index}" for index in range(content_count)],
            image_ids=[row["image_id"] for row in source_unique],
            shape=[content_count, 8, 4, 4],
        ),
        "synthetic-layer-fc": adapter.make_feature_rows(
            roles,
            layer_id="synthetic-layer-fc",
            row_ids=[f"fc-row-{index}" for index in range(content_count)],
            image_ids=[row["image_id"] for row in source_unique],
            shape=[content_count, 16],
        ),
    }
    source_rows = adapter.make_ordered_rows(
        roles,
        participant_id=pair_record["source_id"],
        role="tau_train",
        row_ids=[row["row_id"] for row in source_tau],
        image_ids=[row["image_id"] for row in source_tau],
        shape=[len(source_tau), 20],
    )
    target_rows = adapter.make_ordered_rows(
        roles,
        participant_id=pair_record["target_id"],
        role="tau_train",
        row_ids=[row["row_id"] for row in target_tau],
        image_ids=[row["image_id"] for row in target_tau],
        shape=[len(target_tau), 24],
    )
    alpha = adapter.bind_alpha(
        context,
        pair,
        policy_id=(
            "synthetic-tau-train-selection"
            if selected_on_tau_train
            else "synthetic-fixed-alpha"
        ),
        policy_sha256="4" * 64,
        selection_scope=("tau_train_only" if selected_on_tau_train else "fixed_predeclared"),
        allowed_fit_role="tau_train",
        fixed_before_tau_test=True,
        tau_test_or_ood_used=False,
        resolved_value=1.0,
        selection_receipt_sha256=("5" * 64 if selected_on_tau_train else None),
    )
    content = adapter.plan_content(
        context,
        pair,
        roles,
        worker,
        response_rows=response,
        feature_rows_by_layer=features,
    )
    brain = adapter.plan_brain_loss(
        context,
        pair,
        roles,
        worker,
        source_rows=source_rows,
        target_rows=target_rows,
        alpha=alpha,
    )
    summary = adapter.summarize_plans(
        context,
        pair,
        roles,
        worker,
        content,
        brain,
        alpha,
        response_rows=response,
        feature_rows_by_layer=features,
        source_rows=source_rows,
        target_rows=target_rows,
    )
    return {
        "adapter": adapter,
        "context": context,
        "pair": pair,
        "roles": roles,
        "worker": worker,
        "response": response,
        "features": features,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "alpha": alpha,
        "content": content,
        "brain": brain,
        "summary": summary,
        "source_unique": source_unique,
        "target_unique": target_unique,
        "source_tau": source_tau,
        "target_tau": target_tau,
    }


def fixture_report(bundle: Mapping[str, Any]) -> dict[str, Any]:
    summary = bundle["adapter"].export_summary(
        bundle["summary"],
        bundle["context"],
        bundle["pair"],
        bundle["roles"],
        bundle["worker"],
        bundle["content"],
        bundle["brain"],
        bundle["alpha"],
        response_rows=bundle["response"],
        feature_rows_by_layer=bundle["features"],
        source_rows=bundle["source_rows"],
        target_rows=bundle["target_rows"],
    )
    content = bundle["content"].record
    brain = bundle["brain"].record
    return {
        "summary": summary,
        "content_dynamic_buffer_shapes": {
            item["layer_id"]: item["dynamic_buffer_shape"]
            for item in content["feature_rows"]
        },
        "source_presentation_count": brain["source_row_count"],
        "target_presentation_count": brain["target_row_count"],
        "fit_unit_count": len(brain["fit_unit_keys"]),
        "source_group_sizes": [len(group) for group in brain["source_group_indices"]],
        "target_group_sizes": [len(group) for group in brain["target_group_indices"]],
        "paired_source_order": brain["source_aligned_indices"],
        "paired_target_order": brain["target_aligned_indices"],
        "alpha_value_omitted_from_report": True,
        "alpha_binding_kind": brain["alpha_binding_record"]["selection_scope"],
        "synthetic_identifiers_only": True,
    }


def expect_adapter_error(
    adapter: types.ModuleType,
    label: str,
    expected_code: str,
    operation: Callable[[], Any],
) -> dict[str, str]:
    try:
        operation()
    except adapter.AdapterValidationError as error:
        if error.code != expected_code:
            raise ValueError(
                f"negative fixture {label} returned {error.code}, expected {expected_code}"
            ) from error
        return {"label": label, "result": "rejected", "error_code": error.code}
    raise ValueError(f"negative fixture {label} was accepted")


def summarize_bundle(
    adapter: types.ModuleType,
    bundle: Mapping[str, Any],
    *,
    content: Any | None = None,
    brain: Any | None = None,
    alpha: Any | None = None,
) -> Any:
    return adapter.summarize_plans(
        bundle["context"],
        bundle["pair"],
        bundle["roles"],
        bundle["worker"],
        content if content is not None else bundle["content"],
        brain if brain is not None else bundle["brain"],
        alpha if alpha is not None else bundle["alpha"],
        response_rows=bundle["response"],
        feature_rows_by_layer=bundle["features"],
        source_rows=bundle["source_rows"],
        target_rows=bundle["target_rows"],
    )


def export_bundle(
    adapter: types.ModuleType,
    bundle: Mapping[str, Any],
    *,
    summary: Any | None = None,
    content: Any | None = None,
    brain: Any | None = None,
    alpha: Any | None = None,
) -> dict[str, Any]:
    return adapter.export_summary(
        summary if summary is not None else bundle["summary"],
        bundle["context"],
        bundle["pair"],
        bundle["roles"],
        bundle["worker"],
        content if content is not None else bundle["content"],
        brain if brain is not None else bundle["brain"],
        alpha if alpha is not None else bundle["alpha"],
        response_rows=bundle["response"],
        feature_rows_by_layer=bundle["features"],
        source_rows=bundle["source_rows"],
        target_rows=bundle["target_rows"],
    )


def negative_fixtures(
    adapter: types.ModuleType, worker_limit: int
) -> list[dict[str, str]]:
    base = fixture_bundle(adapter, "image_average_before_fit", 3, worker_limit)
    paired = fixture_bundle(
        adapter, "authenticated_cross_subject_pairing", 7, worker_limit
    )
    checks: list[dict[str, str]] = []
    add = checks.append
    add(expect_adapter_error(adapter, "absent_empirical_lock_and_authorization", "E_EMPIRICAL_CONTEXT_UNAVAILABLE", lambda: adapter.bind_empirical_context(verified_lock_receipt=None, authorization_receipt=None)))
    add(expect_adapter_error(adapter, "synthetic_receipt_as_empirical", "E_EMPIRICAL_CONTEXT_UNAVAILABLE", lambda: adapter.bind_empirical_context(verified_lock_receipt={"synthetic": True}, authorization_receipt={"synthetic": True})))
    wrong_pair = dict(base["pair"].record)
    wrong_pair.pop("context_seal")
    wrong_pair["pair_table_sha256"] = "9" * 64
    add(expect_adapter_error(adapter, "wrong_pair_table", "E_PAIR_TABLE", lambda: adapter.bind_pair(base["context"], wrong_pair)))
    unlisted_pair = dict(
        wrong_pair,
        pair_id="unlisted-pair",
        ordinal=99,
        pair_table_sha256=base["context"].record["pair_table_sha256"],
    )
    add(expect_adapter_error(adapter, "unlisted_pair_membership", "E_PAIR_MEMBERSHIP", lambda: adapter.bind_pair(base["context"], unlisted_pair)))
    same_pair_core = {
        "pair_id": "invalid-same-participant",
        "source_id": "same-participant",
        "target_id": "same-participant",
        "ordinal": 0,
    }
    add(expect_adapter_error(
        adapter,
        "same_source_target",
        "E_PAIR_DIRECTION",
        lambda: adapter.make_synthetic_context(
            fixture_id="invalid-same-participant",
            pair_table_records=[same_pair_core],
            settings={
                "content_feature_layers": ["synthetic-layer-conv"],
                "brain_loss_repetition_mode": "image_average_before_fit",
                "brain_loss_alpha_bindings": {
                    "invalid-same-participant": {
                        "policy_id": "synthetic-fixed-alpha",
                        "policy_sha256": "4" * 64,
                        "selection_scope": "fixed_predeclared",
                        "allowed_fit_role": "tau_train",
                        "fixed_before_tau_test": True,
                        "tau_test_or_ood_used": False,
                        "resolved_value": 1.0,
                        "selection_receipt_sha256": None,
                    }
                },
            },
        ),
    ))
    for label, workers, allocation, code in (
        ("worker_bool", True, worker_limit, "E_INTEGER"),
        ("worker_string", "1", worker_limit, "E_INTEGER"),
        ("worker_zero", 0, worker_limit, "E_INTEGER"),
        ("worker_negative", -1, worker_limit, "E_INTEGER"),
        (
            "worker_oversubscription",
            worker_limit + 1,
            worker_limit,
            "E_WORKER_ALLOCATION",
        ),
        ("worker_underallocation", 1, 2, "E_WORKER_ALLOCATION"),
    ):
        add(expect_adapter_error(adapter, label, code, lambda workers=workers, allocation=allocation: adapter.bind_worker(base["context"], max_workers=workers, explicit_worker_limit=allocation)))
    role_record = base["roles"].record
    overlapping_source = json.loads(json.dumps(role_record["source_unique_rows"]))
    overlapping_source[0]["image_id"] = role_record["tau_test_ordered_ids"][0]
    add(expect_adapter_error(adapter, "role_overlap", "E_ROLE_OVERLAP", lambda: adapter.bind_structural_roles(
        base["context"], base["pair"], manifest_sha256="1" * 64,
        regular_only_exclusion_receipt_sha256="2" * 64,
        cross_subject_alignment_receipt_sha256=None,
        source_unique_rows=overlapping_source,
        target_unique_rows=role_record["target_unique_rows"],
        source_feature_rows_by_layer={
            layer: [
                {**row, "image_id": overlapping_source[index]["image_id"]}
                for index, row in enumerate(rows)
            ]
            for layer, rows in role_record["source_feature_rows_by_layer"].items()
        },
        tau_train_ordered_ids=role_record["tau_train_ordered_ids"],
        tau_test_ordered_ids=role_record["tau_test_ordered_ids"],
        source_tau_presentations=role_record["source_tau_presentations"],
        target_tau_presentations=role_record["target_tau_presentations"],
    )))
    response_record = base["response"].record
    wrong_response = adapter.make_ordered_rows(
        base["roles"], participant_id=base["pair"].record["source_id"], role="source_unique_train",
        row_ids=[*response_record["row_ids"][:-1], "substituted-row"],
        image_ids=response_record["image_ids"], shape=response_record["shape"],
    )
    add(expect_adapter_error(adapter, "same_count_substituted_response_row", "E_CONTENT_RESPONSE_ALIGNMENT", lambda: adapter.plan_content(base["context"], base["pair"], base["roles"], base["worker"], response_rows=wrong_response, feature_rows_by_layer=base["features"])))
    permuted_response = adapter.make_ordered_rows(
        base["roles"], participant_id=base["pair"].record["source_id"], role="source_unique_train",
        row_ids=list(reversed(response_record["row_ids"])), image_ids=list(reversed(response_record["image_ids"])), shape=response_record["shape"],
    )
    add(expect_adapter_error(adapter, "permuted_response_rows", "E_CONTENT_RESPONSE_ALIGNMENT", lambda: adapter.plan_content(base["context"], base["pair"], base["roles"], base["worker"], response_rows=permuted_response, feature_rows_by_layer=base["features"])))
    feature_record = base["features"]["synthetic-layer-fc"].record
    bad_features = dict(base["features"])
    bad_features["synthetic-layer-fc"] = adapter.make_feature_rows(
        base["roles"], layer_id="synthetic-layer-fc", row_ids=feature_record["row_ids"],
        image_ids=list(reversed(feature_record["image_ids"])), shape=feature_record["shape"],
    )
    add(expect_adapter_error(adapter, "permuted_feature_images", "E_CONTENT_FEATURE_ALIGNMENT", lambda: adapter.plan_content(base["context"], base["pair"], base["roles"], base["worker"], response_rows=base["response"], feature_rows_by_layer=bad_features)))
    substituted_feature_rows = dict(base["features"])
    substituted_feature_rows["synthetic-layer-fc"] = adapter.make_feature_rows(
        base["roles"], layer_id="synthetic-layer-fc",
        row_ids=[*feature_record["row_ids"][:-1], "substituted-feature-row"],
        image_ids=feature_record["image_ids"], shape=feature_record["shape"],
    )
    add(expect_adapter_error(adapter, "same_count_substituted_feature_row", "E_CONTENT_FEATURE_ALIGNMENT", lambda: adapter.plan_content(base["context"], base["pair"], base["roles"], base["worker"], response_rows=base["response"], feature_rows_by_layer=substituted_feature_rows)))
    add(expect_adapter_error(adapter, "missing_content_layer", "E_CONTENT_LAYER_SET", lambda: adapter.plan_content(base["context"], base["pair"], base["roles"], base["worker"], response_rows=base["response"], feature_rows_by_layer={"synthetic-layer-conv": base["features"]["synthetic-layer-conv"]})))
    source_rows_record = base["source_rows"].record
    missing_repeat = adapter.make_ordered_rows(
        base["roles"], participant_id=base["pair"].record["source_id"], role="tau_train",
        row_ids=source_rows_record["row_ids"][:-1], image_ids=source_rows_record["image_ids"][:-1], shape=[len(source_rows_record["row_ids"]) - 1, 20],
    )
    add(expect_adapter_error(adapter, "missing_authenticated_repeat", "E_BRAIN_PRESENTATION_ALIGNMENT", lambda: adapter.plan_brain_loss(base["context"], base["pair"], base["roles"], base["worker"], source_rows=missing_repeat, target_rows=base["target_rows"], alpha=base["alpha"])))
    extra_repeat = adapter.make_ordered_rows(
        base["roles"], participant_id=base["pair"].record["source_id"], role="tau_train",
        row_ids=[*source_rows_record["row_ids"], "extra-row"], image_ids=[*source_rows_record["image_ids"], source_rows_record["image_ids"][0]], shape=[len(source_rows_record["row_ids"]) + 1, 20],
    )
    add(expect_adapter_error(adapter, "extra_unauthenticated_repeat", "E_BRAIN_PRESENTATION_ALIGNMENT", lambda: adapter.plan_brain_loss(base["context"], base["pair"], base["roles"], base["worker"], source_rows=extra_repeat, target_rows=base["target_rows"], alpha=base["alpha"])))
    paired_roles = paired["roles"].record
    bad_target_presentations = json.loads(json.dumps(paired_roles["target_tau_presentations"]))
    bad_target_presentations[0]["local_repetition_id"] = paired_roles["source_tau_presentations"][0]["local_repetition_id"]
    bad_target_presentations[0]["cross_subject_alignment_key"] = "wrong-alignment"
    add(expect_adapter_error(adapter, "local_repetition_cannot_replace_alignment_key", "E_ALIGNMENT_MISMATCH", lambda: adapter.bind_structural_roles(
        paired["context"], paired["pair"], manifest_sha256="1" * 64,
        regular_only_exclusion_receipt_sha256="2" * 64,
        cross_subject_alignment_receipt_sha256="3" * 64,
        source_unique_rows=paired_roles["source_unique_rows"], target_unique_rows=paired_roles["target_unique_rows"],
        source_feature_rows_by_layer=paired_roles["source_feature_rows_by_layer"],
        tau_train_ordered_ids=paired_roles["tau_train_ordered_ids"], tau_test_ordered_ids=paired_roles["tau_test_ordered_ids"],
        source_tau_presentations=paired_roles["source_tau_presentations"], target_tau_presentations=bad_target_presentations,
    )))
    alpha_args = dict(
        policy_sha256="4" * 64, selection_scope="fixed_predeclared",
        allowed_fit_role="tau_train", fixed_before_tau_test=True,
        tau_test_or_ood_used=False, resolved_value=1.0, selection_receipt_sha256=None,
    )
    add(expect_adapter_error(adapter, "alpha_policy_names_tau_test", "E_ALPHA_LEAKAGE", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="select-on-tau-test", **alpha_args)))
    add(expect_adapter_error(adapter, "alpha_dotted_policy_names_tau_test", "E_ALPHA_LEAKAGE", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="select.on.tau.test", **alpha_args)))
    alpha_test_args = dict(alpha_args, allowed_fit_role="tau_test")
    add(expect_adapter_error(adapter, "alpha_fit_role_tau_test", "E_ALPHA_ROLE", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="synthetic-policy", **alpha_test_args)))
    alpha_leak_args = dict(alpha_args, tau_test_or_ood_used=True)
    add(expect_adapter_error(adapter, "alpha_declares_tau_test_or_ood_use", "E_ALPHA_LEAKAGE", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="synthetic-policy", **alpha_leak_args)))
    alpha_missing_receipt = dict(alpha_args, selection_scope="tau_train_only")
    add(expect_adapter_error(adapter, "selected_alpha_missing_receipt", "E_SHA256", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="synthetic-policy", **alpha_missing_receipt)))
    alpha_value_override = dict(alpha_args, resolved_value=2.0)
    add(expect_adapter_error(adapter, "alpha_value_differs_from_context_declaration", "E_ALPHA_BINDING", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="synthetic-fixed-alpha", **alpha_value_override)))
    alpha_numeric_representation_override = dict(alpha_args, resolved_value=1)
    add(expect_adapter_error(adapter, "alpha_numeric_representation_differs_from_context_declaration", "E_ALPHA_BINDING", lambda: adapter.bind_alpha(base["context"], base["pair"], policy_id="synthetic-fixed-alpha", **alpha_numeric_representation_override)))
    add(expect_adapter_error(adapter, "direct_context_bad_seal", "E_SEAL", lambda: adapter.SyntheticFixtureContext(base["context"].canonical_record_json, "0" * 64)))
    tampered_content_record = base["content"].record
    tampered_content_record["source_row_count"] = 999
    tampered_content = adapter.ContentPlan.__new__(adapter.ContentPlan)
    object.__setattr__(tampered_content, "canonical_record_json", json.dumps(tampered_content_record, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    object.__setattr__(tampered_content, "seal", base["content"].seal)
    add(expect_adapter_error(adapter, "tampered_content_plan", "E_CONTENT_PLAN", lambda: summarize_bundle(adapter, base, content=tampered_content)))
    self_consistent_content_record = base["content"].record
    self_consistent_content_record["source_row_count"] = 999
    self_consistent_content_record["response_shape"][0] = 999
    for item in self_consistent_content_record["feature_rows"]:
        item["dynamic_buffer_shape"][0] = 999
    self_consistent_content_raw = json.dumps(self_consistent_content_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    self_consistent_content = adapter.ContentPlan(
        self_consistent_content_raw,
        adapter._seal("ContentPlan", self_consistent_content_raw),
    )
    add(expect_adapter_error(adapter, "resealed_self_consistent_content_plan", "E_CONTENT_PLAN_BINDING", lambda: summarize_bundle(adapter, base, content=self_consistent_content)))
    invalid_response_shape_record = base["content"].record
    invalid_response_shape_record["response_shape"][-1] = -1
    invalid_response_shape_raw = json.dumps(invalid_response_shape_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    add(expect_adapter_error(adapter, "content_plan_negative_tail_dimension", "E_CONTENT_PLAN", lambda: adapter.ContentPlan(invalid_response_shape_raw, adapter._seal("ContentPlan", invalid_response_shape_raw))))
    invalid_feature_shape_record = base["content"].record
    invalid_feature_shape_record["feature_rows"][0]["dynamic_buffer_shape"][-1] = True
    invalid_feature_shape_raw = json.dumps(invalid_feature_shape_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    add(expect_adapter_error(adapter, "content_plan_bool_feature_dimension", "E_CONTENT_PLAN", lambda: adapter.ContentPlan(invalid_feature_shape_raw, adapter._seal("ContentPlan", invalid_feature_shape_raw))))
    altered_response_seal_record = base["content"].record
    altered_response_seal_record["response_rows_seal"] = "9" * 64
    altered_response_seal_raw = json.dumps(altered_response_seal_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_response_seal = adapter.ContentPlan(
        altered_response_seal_raw,
        adapter._seal("ContentPlan", altered_response_seal_raw),
    )
    add(expect_adapter_error(adapter, "resealed_arbitrary_response_rows_seal", "E_CONTENT_PLAN_BINDING", lambda: summarize_bundle(adapter, base, content=altered_response_seal)))
    altered_feature_seal_record = base["content"].record
    altered_feature_seal_record["feature_rows"][0]["rows_seal"] = "9" * 64
    altered_feature_seal_raw = json.dumps(altered_feature_seal_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_feature_seal = adapter.ContentPlan(
        altered_feature_seal_raw,
        adapter._seal("ContentPlan", altered_feature_seal_raw),
    )
    add(expect_adapter_error(adapter, "resealed_arbitrary_feature_rows_seal", "E_CONTENT_PLAN_BINDING", lambda: summarize_bundle(adapter, base, content=altered_feature_seal)))
    altered_brain_record = base["brain"].record
    altered_brain_record["source_group_indices"][0], altered_brain_record["source_group_indices"][1] = (
        altered_brain_record["source_group_indices"][1],
        altered_brain_record["source_group_indices"][0],
    )
    altered_brain_raw = json.dumps(altered_brain_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_brain = adapter.BrainPlan(
        altered_brain_raw, adapter._seal("BrainPlan", altered_brain_raw)
    )
    add(expect_adapter_error(adapter, "resealed_brain_group_reorder", "E_BRAIN_PLAN_BINDING", lambda: summarize_bundle(adapter, base, brain=altered_brain)))
    invalid_bool_index_record = base["brain"].record
    invalid_bool_index_record["source_group_indices"][0][0] = True
    invalid_bool_index_raw = json.dumps(invalid_bool_index_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    add(expect_adapter_error(adapter, "brain_plan_bool_index", "E_BRAIN_PLAN", lambda: adapter.BrainPlan(invalid_bool_index_raw, adapter._seal("BrainPlan", invalid_bool_index_raw))))
    invalid_float_index_record = base["brain"].record
    invalid_float_index_record["target_group_indices"][0][0] = 0.0
    invalid_float_index_raw = json.dumps(invalid_float_index_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    add(expect_adapter_error(adapter, "brain_plan_float_index", "E_BRAIN_PLAN", lambda: adapter.BrainPlan(invalid_float_index_raw, adapter._seal("BrainPlan", invalid_float_index_raw))))
    invalid_group_container_record = base["brain"].record
    invalid_group_container_record["source_group_indices"] = None
    invalid_group_container_raw = json.dumps(invalid_group_container_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    add(expect_adapter_error(adapter, "brain_plan_nonlist_group_container", "E_BRAIN_PLAN", lambda: adapter.BrainPlan(invalid_group_container_raw, adapter._seal("BrainPlan", invalid_group_container_raw))))
    invalid_aligned_container_record = base["brain"].record
    invalid_aligned_container_record["target_aligned_indices"] = False
    invalid_aligned_container_raw = json.dumps(invalid_aligned_container_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    add(expect_adapter_error(adapter, "brain_plan_falsey_nonlist_aligned_container", "E_BRAIN_PLAN", lambda: adapter.BrainPlan(invalid_aligned_container_raw, adapter._seal("BrainPlan", invalid_aligned_container_raw))))
    altered_source_rows_seal_record = base["brain"].record
    altered_source_rows_seal_record["source_rows_seal"] = "9" * 64
    altered_source_rows_seal_raw = json.dumps(altered_source_rows_seal_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_source_rows_seal = adapter.BrainPlan(
        altered_source_rows_seal_raw,
        adapter._seal("BrainPlan", altered_source_rows_seal_raw),
    )
    add(expect_adapter_error(adapter, "resealed_arbitrary_source_rows_seal", "E_BRAIN_PLAN_BINDING", lambda: summarize_bundle(adapter, base, brain=altered_source_rows_seal)))
    altered_target_rows_seal_record = base["brain"].record
    altered_target_rows_seal_record["target_rows_seal"] = "9" * 64
    altered_target_rows_seal_raw = json.dumps(altered_target_rows_seal_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_target_rows_seal = adapter.BrainPlan(
        altered_target_rows_seal_raw,
        adapter._seal("BrainPlan", altered_target_rows_seal_raw),
    )
    add(expect_adapter_error(adapter, "resealed_arbitrary_target_rows_seal", "E_BRAIN_PLAN_BINDING", lambda: summarize_bundle(adapter, base, brain=altered_target_rows_seal)))
    summary_record = json.loads(base["summary"].canonical_record_json.decode("utf-8"))
    summary_record["content_row_count"] = 999
    summary_record["brain_fit_unit_count"] = 999
    altered_summary_raw = json.dumps(summary_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_summary = adapter.PlanSummary(
        altered_summary_raw,
        adapter._seal("PlanSummary", altered_summary_raw),
    )
    add(expect_adapter_error(adapter, "resealed_self_consistent_summary", "E_SUMMARY_BINDING", lambda: export_bundle(adapter, base, summary=altered_summary)))
    altered_summary_plan_seal_record = json.loads(base["summary"].canonical_record_json.decode("utf-8"))
    altered_summary_plan_seal_record["content_plan_seal"] = "9" * 64
    altered_summary_plan_seal_raw = json.dumps(altered_summary_plan_seal_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    altered_summary_plan_seal = adapter.PlanSummary(
        altered_summary_plan_seal_raw,
        adapter._seal("PlanSummary", altered_summary_plan_seal_raw),
    )
    add(expect_adapter_error(adapter, "resealed_summary_plan_authority_seal", "E_SUMMARY_BINDING", lambda: export_bundle(adapter, base, summary=altered_summary_plan_seal)))
    response_shape_record = base["response"].record
    add(expect_adapter_error(adapter, "ordered_rows_zero_tail_dimension", "E_SHAPE", lambda: adapter.make_ordered_rows(
        base["roles"], participant_id=response_shape_record["participant_id"],
        role=response_shape_record["role"], row_ids=response_shape_record["row_ids"],
        image_ids=response_shape_record["image_ids"],
        shape=[len(response_shape_record["row_ids"]), 0],
    )))
    feature_shape_record = base["features"]["synthetic-layer-fc"].record
    add(expect_adapter_error(adapter, "feature_rows_bool_tail_dimension", "E_SHAPE", lambda: adapter.make_feature_rows(
        base["roles"], layer_id=feature_shape_record["layer_id"],
        row_ids=feature_shape_record["row_ids"], image_ids=feature_shape_record["image_ids"],
        shape=[len(feature_shape_record["row_ids"]), True],
    )))
    wrong_participant_record = base["response"].record
    wrong_participant = adapter.make_ordered_rows(
        base["roles"], participant_id=base["pair"].record["target_id"],
        role="source_unique_train", row_ids=wrong_participant_record["row_ids"],
        image_ids=wrong_participant_record["image_ids"], shape=wrong_participant_record["shape"],
    )
    add(expect_adapter_error(adapter, "wrong_content_participant", "E_CONTENT_RESPONSE_ALIGNMENT", lambda: adapter.plan_content(base["context"], base["pair"], base["roles"], base["worker"], response_rows=wrong_participant, feature_rows_by_layer=base["features"])))
    fake = types.SimpleNamespace(record={"synthetic_only": False}, seal="a" * 64)
    add(expect_adapter_error(adapter, "duck_typed_alpha", "E_TYPE", lambda: adapter.plan_brain_loss(base["context"], base["pair"], base["roles"], base["worker"], source_rows=base["source_rows"], target_rows=base["target_rows"], alpha=fake)))
    add(expect_adapter_error(adapter, "duck_typed_content_plan", "E_TYPE", lambda: summarize_bundle(adapter, base, content=fake)))
    class ContentPlanSubclass(adapter.ContentPlan):
        pass
    subclass_content = ContentPlanSubclass.__new__(ContentPlanSubclass)
    object.__setattr__(subclass_content, "canonical_record_json", base["content"].canonical_record_json)
    object.__setattr__(subclass_content, "seal", base["content"].seal)
    add(expect_adapter_error(adapter, "subclassed_content_plan", "E_TYPE", lambda: summarize_bundle(adapter, base, content=subclass_content)))
    return checks


def build_report(
    attempt: int, max_workers: int, execution_context: str
) -> dict[str, Any]:
    if attempt != 6:
        raise ValueError("this frozen runner is valid only for attempt 6")
    execution = validate_attended_execution_context(execution_context, max_workers)
    started_at = utc_now()
    adapter_raw, adapter_identity = read_stable_bytes(
        ADAPTER_PATH, label="execution adapter", maximum_bytes=MAX_CODE_BYTES,
        expected_sha256=ADAPTER_SHA256,
    )
    helper_raw, helper_identity = read_stable_bytes(
        PUBLICATION_HELPER_PATH, label="publication helper", maximum_bytes=MAX_CODE_BYTES,
        expected_sha256=PUBLICATION_HELPER_SHA256,
    )
    neutral_raw, neutral_identity = read_stable_bytes(
        NEUTRAL_REPORT, label="attempt-5 neutral report", maximum_bytes=MAX_JSON_BYTES,
        expected_sha256=NEUTRAL_REPORT_SHA256,
    )
    method_raw, method_identity = read_stable_bytes(
        METHOD_REPORT, label="attempt-19 method report", maximum_bytes=MAX_JSON_BYTES,
        expected_sha256=METHOD_REPORT_SHA256,
    )
    protocol_raw, protocol_identity = read_stable_bytes(
        PROTOCOL_TABLE, label="protocol table", maximum_bytes=MAX_PROTOCOL_BYTES,
        expected_sha256=PROTOCOL_TABLE_SHA256,
    )
    neutral = json.loads(neutral_raw.decode("utf-8", errors="strict"))
    method = json.loads(method_raw.decode("utf-8", errors="strict"))
    protocol_text = protocol_raw.decode("utf-8", errors="strict")
    validate_upstream_bindings(neutral, method, protocol_text)
    gate_state = assert_current_gate_closed()
    code_audit = adapter_static_audit(adapter_raw)
    adapter = load_verified_module(
        adapter_raw, path=ADAPTER_PATH,
        module_name="ep21_execution_adapter_attempt6_verified",
    )
    fixtures = {
        "dynamic_rows_3_image_average": fixture_report(
            fixture_bundle(adapter, "image_average_before_fit", 3, max_workers)
        ),
        "dynamic_rows_7_authenticated_pairing": fixture_report(
            fixture_bundle(
                adapter, "authenticated_cross_subject_pairing", 7, max_workers
            )
        ),
    }
    negative = negative_fixtures(adapter, max_workers)
    if not negative or any(item["result"] != "rejected" for item in negative):
        raise ValueError("negative fixture suite is incomplete")
    return {
        "schema_version": "ep21.adapter_conformance_report.v1",
        "astra_analysis_id": "pipeline_conformance",
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": os.getpid(),
        "execution": execution,
        "scope": "synthetic_metadata_only_execution_adapter_conformance",
        "outcome": "passed_synthetic_adapter_checks_with_protocol_and_access_blockers",
        "bindings": {
            "execution_adapter": adapter_identity,
            "atomic_publication_helper": helper_identity,
            "neutral_conformance_attempt_5": neutral_identity,
            "public_method_semantics_attempt_19": method_identity,
            "protocol_table": protocol_identity,
        },
        "adapter_static_audit": code_audit,
        "contexts_exercised": {
            "synthetic_only": True,
            "empirical_context_accepted": False,
            "real_protocol_lock_present": False,
            "later_execution_authorization_present": False,
            "synthetic_receipts_are_authenticated_handoff_evidence": False,
            "fixture_factor_cooccurrence_defines_an_allowed_protocol_combination": False,
            "synthetic_worker_count_matches_explicit_runner_limit": True,
            "empirical_scheduler_binding_authenticated": False,
            "real_ood_identifier_inventory_authenticated": False,
        },
        "synthetic_fixtures": fixtures,
        "negative_fixture_results": negative,
        "negative_fixture_count": len(negative),
        "current_gate": gate_state,
        "retained_blockers": [
            "scientist protocol choices remain unresolved",
            "LAION DUA license and Sherlock risk disposition remain unresolved",
            "authenticated participant role split overlap repetition beta QC and mask metadata are absent",
            "role-filtered regular-only feature and scoring handoffs are absent",
            "pair IDs repetition disposition alpha policy and scientific settings are not locked",
            "critical model runtime and dependency bindings remain incomplete",
            "later empirical execution authorization is absent",
        ],
        "boundary": {
            "empirical_or_neural_value_accessed": False,
            "feature_or_scoring_handoff_opened": False,
            "ood_payload_accessed": False,
            "ep04_or_legacy_scratch_accessed": False,
            "model_or_checkpoint_read": False,
            "array_or_tensor_created": False,
            "device_created": False,
            "executor_created": False,
            "optimizer_or_inference_executed": False,
            "source_archive_reopened": False,
        },
        "protocol_effect": {
            "scientific_choice_adopted": False,
            "pair_id_format_adopted": False,
            "source_study_alpha_ported": False,
            "protocol_lock_emitted": False,
            "empirical_access_authorized": False,
            "empirical_readiness": False,
        },
        "next_action": "obtain scientist and steward dispositions plus authenticated role-filtered structural handoffs; only then build and independently validate a real lock and later execution authorization before enabling any empirical adapter entrypoint",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument(
        "--execution-context",
        required=True,
        choices=(ATTENDED_LOGIN_CONTEXT,),
    )
    parser.add_argument("--max-workers", required=True, type=int)
    args = parser.parse_args()
    execution = validate_attended_execution_context(
        args.execution_context, args.max_workers
    )
    print(
        json.dumps(
            {
                "event": "ep21_adapter_conformance_started",
                "attempt": args.attempt,
                "pid": os.getpid(),
                "execution_context": execution["kind"],
                "max_workers": execution["max_workers"],
                "started_at": utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    report = build_report(
        args.attempt, args.max_workers, args.execution_context
    )
    helper_raw, _ = read_stable_bytes(
        PUBLICATION_HELPER_PATH, label="publication helper before publish",
        maximum_bytes=MAX_CODE_BYTES, expected_sha256=PUBLICATION_HELPER_SHA256,
    )
    helper = load_verified_module(
        helper_raw,
        path=PUBLICATION_HELPER_PATH,
        module_name="ep21_atomic_publication_helper_attempt6_verified",
    )
    output_identity = helper.atomic_publish_json(OUTPUT_DIR, OUTPUT_NAME, report)
    print(
        json.dumps(
            {
                "event": "ep21_adapter_conformance_completed",
                "attempt": args.attempt,
                "pid": os.getpid(),
                "completed_at": report["completed_at"],
                "output": output_identity,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
