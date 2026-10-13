#!/usr/bin/env python3
"""Run outcome-blind, protocol-neutral EP21 mechanical conformance checks.

This runner reads only the pinned public source archive, episode contracts, and
the fail-closed source-qualification report.  It never reads empirical arrays,
stimulus payloads, neural values, derived outcomes, OOD material, or EP04 state.
It deliberately does not emit ``outputs/protocol_lock.yaml``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SOURCE_QUALIFICATION_TOOLS = (
    Path(__file__).resolve().parents[2] / "source_qualification" / "tools"
)
if str(SOURCE_QUALIFICATION_TOOLS) not in sys.path:
    sys.path.insert(0, str(SOURCE_QUALIFICATION_TOOLS))
import verified_source_archive as source_archive_verifier


SOURCE_MEMBERS = {
    "content_train": "NCC_content_loss/NCC_train.py",
    "content_models": "NCC_content_loss/models.py",
    "content_utils": "NCC_content_loss/utils.py",
    "brain_train": "NCC_brain_loss/ncc_train.py",
    "brain_params": "NCC_brain_loss/params/converter_params_VC.csv",
    "reconstruction": "reconstruction/recon_icnn_image_gd.py",
    "reconstruction_config": "reconstruction/config/recon_icnn_vgg19_relu7generator_gd_200iter_decoded_ImageNet.yaml",
}
PINNED_ARCHIVE_MD5 = "bb80dcc1a038737d6d8fc2c95b3e5586"
PINNED_ARCHIVE_SIZE_BYTES = 23_110_908
MAX_QUALIFICATION_REPORT_BYTES = 5 * 1024 * 1024
MAX_PROTOCOL_TABLE_BYTES = 2 * 1024 * 1024


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_stable_bytes(path: Path, *, label: str, maximum_bytes: int) -> tuple[bytes, dict[str, Any]]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open {label}: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise ValueError(f"{label} is not a single-link regular file")
        if initial.st_size <= 0 or initial.st_size > maximum_bytes:
            raise ValueError(f"{label} has an invalid size")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise ValueError(f"{label} exceeded its size bound")
        final = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(initial, field) != getattr(final, field) for field in fields):
            raise ValueError(f"{label} changed while reading")
        if byte_count != initial.st_size:
            raise ValueError(f"{label} byte count changed while reading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return payload, {
        "path": str(path),
        "size_bytes": byte_count,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_qualification_binding(
    qualification: Mapping[str, Any], expected_attempt: int
) -> None:
    if expected_attempt <= 0:
        raise ValueError("expected qualification attempt must be positive")
    if qualification.get("schema_version") != "ep21.source_qualification_report.v2":
        raise ValueError("qualification report schema is not the accepted v2 schema")
    if qualification.get("attempt") != expected_attempt:
        raise ValueError("qualification report attempt does not match the prepared binding")
    if (
        qualification.get("qualification_outcome")
        != "completed_with_empirical_and_scientific_decision_blockers"
    ):
        raise ValueError("qualification outcome does not retain required blockers")
    if qualification.get("empirical_or_ood_payload_accessed") is not False:
        raise ValueError("qualification report does not preserve the closed boundary")
    if qualification.get("protocol_lock_present") is not False:
        raise ValueError("qualification report unexpectedly claims a protocol lock")
    if qualification.get("laion_readiness") != "blocked":
        raise ValueError("qualification report does not retain blocked LAION readiness")
    if (
        qualification.get("external_readiness")
        != "blocked_unprovisioned_optional_branch"
    ):
        raise ValueError("qualification report does not retain blocked external readiness")
    expected_interfaces = {
        "laion_regular_feature_handoff": "absent",
        "laion_regular_scoring_handoff": "absent",
        "external_site_feature_handoff": "absent",
        "external_site_scoring_handoff": "absent",
    }
    if qualification.get("direct_interface_state") != expected_interfaces:
        raise ValueError("qualification report direct-interface blockers changed")
    discrepancies = qualification.get("paper_code_discrepancy_register", {})
    if set(discrepancies) != {
        "activation",
        "alexnet_backend_and_input_size",
        "content_loss_weighting",
        "reconstruction_layers",
    } or any(
        not isinstance(item, Mapping)
        or item.get("state") != "recorded_unresolved_for_pipeline_conformance"
        for item in discrepancies.values()
    ):
        raise ValueError("qualification report scientific discrepancies changed")
    gaps = qualification.get("source_pipeline_assets", {}).get(
        "unmanifested_or_unpinned_gaps", {}
    )
    expected_gap_states = {
        "alexnet_identification_checkpoint": "required_unmanifested_public_caffenet_not_substitutable",
        "main_bdpy_dependency": "declared_but_unpinned",
        "vgg19_identification_checkpoint_binding": "public_reconstruction_member_authenticated_evaluator_lab_copy_unbound",
    }
    if any(
        gaps.get(name, {}).get("state") != state
        for name, state in expected_gap_states.items()
    ):
        raise ValueError("qualification report source-asset blockers changed")
    if qualification.get("ood_disposition") != "not_evaluable_under_current_contract":
        raise ValueError("qualification report OOD disposition changed")
    runtime = qualification.get("runtime_and_reconstruction", {})
    if runtime.get("execution_environment_ready") is not False:
        raise ValueError("qualification report runtime blocker changed")
    model_audit = qualification.get("source_pipeline_assets", {}).get(
        "public_model_payload_audit", {}
    )
    if (
        model_audit.get("schema_version") != "ep21.public_model_payload_audit.v2"
        or model_audit.get("attempt") != expected_attempt
        or model_audit.get("empirical_or_decoded_payload_accessed") is not False
        or model_audit.get("ood_or_ep04_payload_accessed") is not False
        or model_audit.get("checkpoint_deserialization_performed") is not False
    ):
        raise ValueError("qualification report public-model boundary changed")


def validate_qualification_artifact_binding(
    qualification_path: Path,
    episode_dir: Path,
    identity: Mapping[str, Any],
    expected_attempt: int,
    expected_sha256: str,
) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected qualification SHA256 is not canonical")
    expected_path = (
        episode_dir
        / "outputs"
        / "source_qualification"
        / f"attempt-{expected_attempt:03d}"
        / "qualification_report.json"
    ).resolve(strict=True)
    observed_path = qualification_path.resolve(strict=True)
    if observed_path != expected_path:
        raise ValueError("qualification report path does not match the prepared attempt")
    if identity.get("sha256") != expected_sha256:
        raise ValueError("qualification report SHA256 does not match the prepared pin")
    return expected_path


def validate_pinned_archive_identity(
    qualification: Mapping[str, Any], observed_md5: str, observed_bytes: int
) -> None:
    identity = qualification.get("source_identity", {})
    if identity.get("state") != "verified":
        raise ValueError("source identity is not verified")
    if identity.get("archive_md5") != PINNED_ARCHIVE_MD5:
        raise ValueError("qualification archive MD5 does not match the EP21 pin")
    if identity.get("archive_size_bytes") != PINNED_ARCHIVE_SIZE_BYTES:
        raise ValueError("qualification archive size does not match the EP21 pin")
    if observed_md5 != PINNED_ARCHIVE_MD5 or observed_bytes != PINNED_ARCHIVE_SIZE_BYTES:
        raise ValueError("source archive identity mismatch")


def safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def read_source_members(archive: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    names = [name for name in archive.namelist() if safe_archive_name(name)]
    for key, suffix in SOURCE_MEMBERS.items():
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"expected one safe archive member ending {suffix!r}, got {matches}"
            )
        raw = archive.read(matches[0])
        members[key] = {
            "path": matches[0],
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8", errors="strict"),
        }
    return members


def argparse_default(source: str, option: str) -> Any:
    tree = ast.parse(source)
    matches: list[Any] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        options = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if option not in options:
            continue
        defaults = [kw.value for kw in node.keywords if kw.arg == "default"]
        if len(defaults) != 1:
            raise ValueError(f"{option} does not have exactly one default")
        matches.append(ast.literal_eval(defaults[0]))
    if len(matches) != 1:
        raise ValueError(f"expected one argparse declaration for {option}, got {len(matches)}")
    return matches[0]


def function_loads_name(source: str, function_name: str, name: str) -> bool:
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        raise ValueError(f"expected one function named {function_name}")
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
        for node in ast.walk(functions[0])
    )


def process_pool_calls(source: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name == "ProcessPoolExecutor":
            calls.append(
                {
                    "line": node.lineno,
                    "positional_argument_count": len(node.args),
                    "keyword_names": sorted(
                        kw.arg for kw in node.keywords if kw.arg is not None
                    ),
                }
            )
    return calls


def line_evidence(source: str, needle: str) -> dict[str, Any]:
    hits = [
        {"line": line_number, "text": line.strip()}
        for line_number, line in enumerate(source.splitlines(), start=1)
        if needle in line
    ]
    if len(hits) != 1:
        raise ValueError(f"expected one line containing {needle!r}, got {len(hits)}")
    return hits[0]


def protocol_count(protocol_text: str, key: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", re.MULTILINE)
    matches = [int(value) for value in pattern.findall(protocol_text)]
    if len(matches) != 1:
        raise ValueError(f"expected one protocol count for {key}, got {matches}")
    return matches[0]


def allocate_exact_row_shapes(
    row_count: int, layer_tail_shapes: Mapping[str, Sequence[int]]
) -> dict[str, list[int]]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    allocated: dict[str, list[int]] = {}
    for layer, tail in layer_tail_shapes.items():
        if any(int(size) <= 0 for size in tail):
            raise ValueError(f"non-positive feature dimension for {layer}")
        allocated[layer] = [row_count, *[int(size) for size in tail]]
    return allocated


def validate_content_rows(
    image_ids: Sequence[str],
    response_row_count: int,
    feature_row_counts: Mapping[str, int],
    allocation_hint: int,
    expected_rows: int,
) -> dict[str, Any]:
    actual_rows = len(image_ids)
    if actual_rows != expected_rows or len(set(image_ids)) != expected_rows:
        raise ValueError("content IDs do not have the exact required unique count")
    if response_row_count != actual_rows:
        raise ValueError("content response row count differs from image IDs")
    mismatches = {
        layer: count for layer, count in feature_row_counts.items() if count != actual_rows
    }
    if mismatches:
        raise ValueError(f"content feature row mismatch: {mismatches}")
    if allocation_hint != actual_rows:
        raise ValueError("content allocation hint differs from authenticated row count")
    return {
        "image_id_count": actual_rows,
        "response_row_count": response_row_count,
        "feature_row_counts": dict(feature_row_counts),
        "allocation_row_count": allocation_hint,
    }


def validate_tau_inputs(
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    tau_train_ids: Sequence[str],
    tau_test_ids: Sequence[str],
    expected_train: int,
    expected_test: int,
) -> dict[str, int]:
    train = list(tau_train_ids)
    test = list(tau_test_ids)
    if len(train) != expected_train or len(set(train)) != expected_train:
        raise ValueError("tau-train IDs do not have the exact required unique count")
    if len(test) != expected_test or len(set(test)) != expected_test:
        raise ValueError("tau-test IDs do not have the exact required unique count")
    if set(train) & set(test):
        raise ValueError("tau-train and tau-test overlap")
    if len(source_labels) != expected_train or set(source_labels) != set(train):
        raise ValueError("source brain-loss rows are not exactly tau train")
    if len(target_labels) != expected_train or set(target_labels) != set(train):
        raise ValueError("target brain-loss rows are not exactly tau train")
    if set(source_labels) & set(test) or set(target_labels) & set(test):
        raise ValueError("tau-test row leaked into brain-loss fitting")
    if set(source_labels) != set(target_labels):
        raise ValueError("source and target brain-loss label sets differ")
    sorted_train = sorted(train)
    sorted_source = sorted(source_labels)
    sorted_target = sorted(target_labels)
    if sorted_source != sorted_train or sorted_target != sorted_train:
        raise ValueError("independently sorted brain-loss labels do not align")
    return {
        "tau_train_count": len(train),
        "tau_test_count": len(test),
        "train_test_union_count": len(set(train) | set(test)),
        "source_fit_count": len(source_labels),
        "target_fit_count": len(target_labels),
        "train_test_overlap_count": 0,
        "source_target_input_order_equal": list(source_labels) == list(target_labels),
        "post_sort_pairwise_label_alignment": sorted_source == sorted_target,
        "post_sort_pairwise_label_count": len(sorted_source),
    }


def resolve_worker_count(
    requested_workers: int | None, environment: Mapping[str, str]
) -> dict[str, Any]:
    raw_allocation = environment.get("SLURM_CPUS_PER_TASK")
    allocation = int(raw_allocation) if raw_allocation else None
    if requested_workers is None:
        if allocation is None:
            raise ValueError("workers must be explicit outside a Slurm allocation")
        requested_workers = allocation
        source = "SLURM_CPUS_PER_TASK"
    else:
        source = "explicit_cli"
    if requested_workers <= 0:
        raise ValueError("workers must be positive")
    if allocation is not None and requested_workers > allocation:
        raise ValueError("requested workers exceed SLURM_CPUS_PER_TASK")
    return {
        "max_workers": requested_workers,
        "binding_source": source,
        "slurm_cpus_per_task": allocation,
    }


def expect_value_error(action: Any) -> bool:
    try:
        action()
    except ValueError:
        return True
    return False


def parse_brain_parameter_table(source: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(source)))
    required = {"Source", "Target", "Number of samples", "Alpha", "ROI", "Method"}
    if not rows or set(rows[0]) != required:
        raise ValueError("unexpected brain-loss parameter-table columns")
    pairs = [(row["Source"], row["Target"]) for row in rows]
    alphas = [int(row["Alpha"]) for row in rows]
    sample_counts = sorted({int(row["Number of samples"]) for row in rows})
    return {
        "row_count": len(rows),
        "unique_directed_pair_count": len(set(pairs)),
        "declared_sample_counts": sample_counts,
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "unique_alpha_values": sorted(set(alphas)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--expected-qualification-attempt", type=int, required=True)
    parser.add_argument("--expected-qualification-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()

    started_at = utc_now()
    process_id = os.getpid()
    print(
        json.dumps(
            {
                "event": "ep21_neutral_conformance_started",
                "pid": process_id,
                "started_at": started_at,
            }
        ),
        flush=True,
    )

    qualification_bytes, qualification_identity = read_stable_bytes(
        args.qualification_report,
        label="qualification report",
        maximum_bytes=MAX_QUALIFICATION_REPORT_BYTES,
    )
    qualification = json.loads(qualification_bytes.decode("utf-8", errors="strict"))
    validate_qualification_binding(qualification, args.expected_qualification_attempt)
    bound_qualification_path = validate_qualification_artifact_binding(
        args.qualification_report,
        args.episode_dir,
        qualification_identity,
        args.expected_qualification_attempt,
        args.expected_qualification_sha256,
    )

    with source_archive_verifier.verified_zip(
        args.source_archive, PINNED_ARCHIVE_MD5, PINNED_ARCHIVE_SIZE_BYTES
    ) as verified_source:
        observed_md5 = verified_source.md5
        observed_bytes = verified_source.size
        validate_pinned_archive_identity(qualification, observed_md5, observed_bytes)
        members = read_source_members(verified_source.archive)

    protocol_path = args.episode_dir / "PROTOCOL_TABLE.yaml"
    protocol_bytes, protocol_identity = read_stable_bytes(
        protocol_path,
        label="protocol table",
        maximum_bytes=MAX_PROTOCOL_TABLE_BYTES,
    )
    protocol_text = protocol_bytes.decode("utf-8", errors="strict")
    unique_count = protocol_count(protocol_text, "subject_unique_regular_per_participant")
    tau_train_count = protocol_count(protocol_text, "tau_train_regular_shared")
    tau_test_count = protocol_count(protocol_text, "tau_test_regular_shared")
    content = members["content_train"]["text"]
    models = members["content_models"]["text"]
    utilities = members["content_utils"]["text"]
    brain = members["brain_train"]["text"]
    reconstruction = members["reconstruction"]["text"]
    reconstruction_config = members["reconstruction_config"]["text"]

    batch_default = argparse_default(content, "--batchSize")
    number_size_default = argparse_default(content, "--number_size")
    n_cpu_default = argparse_default(content, "--n_cpu")
    iteration_default = argparse_default(content, "--n_iterations")
    learning_rate_default = argparse_default(content, "--lr")
    decay_default = argparse_default(content, "--decay_iteration")
    executor_calls = process_pool_calls(content)
    if len(executor_calls) != 1:
        raise ValueError("expected exactly one ProcessPoolExecutor call")

    static_checks = {
        "tagged_batch_default_is_6000": batch_default == 6000,
        "tagged_number_size_default_is_6000": number_size_default == 6000,
        "number_size_is_parser_only": content.count("number_size") == 1,
        "tagged_allocates_with_batch_size": content.count("Tensor(opt.batchSize") == 2,
        "tagged_n_cpu_default_is_16": n_cpu_default == 16,
        "tagged_executor_lacks_max_workers": (
            executor_calls[0]["positional_argument_count"] == 0
            and "max_workers" not in executor_calls[0]["keyword_names"]
        ),
        "tagged_n_cpu_is_parser_only": content.count("n_cpu") == 1,
        "brain_num_sample_unused_inside_fit_function": not function_loads_name(
            brain, "train_NCconverter", "num_sample"
        ),
        "brain_source_and_target_are_independently_sorted_by_labels": all(
            snippet in brain
            for snippet in [
                "x_index = np.argsort(x_labels.flatten())",
                "x_labels = x_labels[x_index]",
                "x = x[x_index, :]",
                "y_index = np.argsort(y_labels.flatten())",
                "y_labels = y_labels[y_index]",
                "y = y[y_index, :]",
            ]
        ),
        "tagged_optimizer_iterations_1024": iteration_default == 1024,
        "tagged_optimizer_learning_rate_5e_5": learning_rate_default == 0.00005,
        "tagged_optimizer_decay_starts_512": decay_default == 512,
        "tagged_optimizer_betas_point5_point999": "betas=(0.5, 0.999)" in content,
        "tagged_seed_is_42": "seed = 42" in content,
        "tagged_checkpoint_is_final_iteration_only": (
            "if iteration == opt.n_iterations - 1:" in content
        ),
        "tagged_model_has_two_leaky_relu_point2": models.count("nn.LeakyReLU(0.2)")
        == 2,
        "tagged_weighting_is_normalized_inverse_squared_norm": (
            "weights = 1. / (feat_norms ** 2)" in utilities
            and "weights /= weights.sum()" in utilities
        ),
        "tagged_reconstruction_uses_sgd": "optimizer=optim.SGD" in reconstruction,
        "tagged_reconstruction_iterations_200": "num_iteration: 200" in reconstruction_config,
        "tagged_reconstruction_learning_rate_endpoints": '"lr": (2., 1e-10)' in reconstruction,
        "tagged_reconstruction_momentum_endpoints": '"momentum": (0.9, 0.9)' in reconstruction,
        "tagged_reconstruction_decay_endpoints": '"decay": (0.01, 0.01)' in reconstruction,
        "tagged_reconstruction_snapshot_interval_10": "snapshot_interval=10" in reconstruction,
        "tagged_reconstruction_initialization_unseeded": (
            "initial_gen_feat = np.random.normal(0, 1, generator_cfg.latent_shape)"
            in reconstruction
            and "np.random.seed" not in reconstruction
        ),
    }
    if not all(static_checks.values()):
        failures = sorted(key for key, passed in static_checks.items() if not passed)
        raise ValueError(f"tagged source assertions failed: {failures}")

    content_ids = [f"subject-unique-{index:04d}" for index in range(unique_count)]
    content_guard_result = validate_content_rows(
        content_ids,
        unique_count,
        {"conv_fixture": unique_count, "fc_fixture": unique_count},
        unique_count,
        unique_count,
    )
    negative_content_tests = {
        "reject_tagged_6000_allocation_hint": expect_value_error(
            lambda: validate_content_rows(
                content_ids,
                unique_count,
                {"fc_fixture": unique_count},
                6000,
                unique_count,
            )
        ),
        "reject_4711_feature_rows": expect_value_error(
            lambda: validate_content_rows(
                content_ids,
                unique_count,
                {"fc_fixture": unique_count - 1},
                unique_count,
                unique_count,
            )
        ),
        "reject_4713_feature_rows": expect_value_error(
            lambda: validate_content_rows(
                content_ids,
                unique_count,
                {"fc_fixture": unique_count + 1},
                unique_count,
                unique_count,
            )
        ),
        "reject_duplicate_image_id": expect_value_error(
            lambda: validate_content_rows(
                [*content_ids[:-1], content_ids[0]],
                unique_count,
                {"fc_fixture": unique_count},
                unique_count,
                unique_count,
            )
        ),
    }
    if not all(negative_content_tests.values()):
        raise ValueError("one or more content-row guards did not fail closed")

    allocated_shapes = allocate_exact_row_shapes(
        unique_count,
        {"conv_fixture": (4, 4), "fc_fixture": (8,)},
    )
    if any(shape[0] != unique_count for shape in allocated_shapes.values()):
        raise ValueError("dynamic allocation fixture did not use the exact row count")

    tau_train = [f"regular-train-{index:04d}" for index in range(tau_train_count)]
    tau_test = [f"regular-test-{index:04d}" for index in range(tau_test_count)]
    role_guard_result = validate_tau_inputs(
        list(reversed(tau_train)),
        tau_train,
        tau_train,
        tau_test,
        tau_train_count,
        tau_test_count,
    )
    negative_role_tests = {
        "reject_train_test_overlap": expect_value_error(
            lambda: validate_tau_inputs(
                tau_train,
                tau_train,
                tau_train,
                [tau_train[0], *tau_test[1:]],
                tau_train_count,
                tau_test_count,
            )
        ),
        "reject_tau_test_fit_row": expect_value_error(
            lambda: validate_tau_inputs(
                [*tau_train[:-1], tau_test[0]],
                tau_train,
                tau_train,
                tau_test,
                tau_train_count,
                tau_test_count,
            )
        ),
        "reject_wrong_train_count": expect_value_error(
            lambda: validate_tau_inputs(
                tau_train[:-1],
                tau_train[:-1],
                tau_train[:-1],
                tau_test,
                tau_train_count,
                tau_test_count,
            )
        ),
    }
    if not all(negative_role_tests.values()):
        raise ValueError("one or more fail-closed role-guard tests did not reject")

    worker_binding = resolve_worker_count(args.workers, os.environ)
    oversubscription_rejected = expect_value_error(
        lambda: resolve_worker_count(3, {"SLURM_CPUS_PER_TASK": "2"})
    )
    if not oversubscription_rejected:
        raise ValueError("worker binding did not reject oversubscription")

    brain_table = parse_brain_parameter_table(members["brain_params"]["text"])
    if brain_table["row_count"] != 20 or brain_table["unique_directed_pair_count"] != 20:
        raise ValueError("unexpected source-study brain-loss parameter-table cardinality")

    execution_binding = {
        "analysis_id": "pipeline_conformance",
        "attempt": args.attempt,
        "stage": "protocol_neutral_synthetic_fixture",
        "pair_id": "synthetic-source_to_synthetic-target",
        "source_id": "synthetic-source",
        "target_id": "synthetic-target",
        "cpu_workers": worker_binding["max_workers"],
        "device": "cpu",
    }

    completed_at = utc_now()
    report = {
        "schema_version": "ep21.neutral_conformance_report.v1",
        "astra_analysis_id": "pipeline_conformance",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": completed_at,
        "process_id": process_id,
        "scope": "protocol_neutral_static_and_synthetic_mechanical_checks",
        "outcome": "passed_protocol_neutral_checks_with_protocol_and_access_blockers",
        "synthetic_only": True,
        "payload_accessed": False,
        "empirical_or_ood_payload_accessed": False,
        "qualification_report": {
            **qualification_identity,
            "path": str(bound_qualification_path),
            "schema_version": qualification["schema_version"],
            "attempt": qualification["attempt"],
            "qualification_outcome": qualification["qualification_outcome"],
            "closed_boundary_verified": True,
            "prepared_path_and_sha256_match": True,
        },
        "protocol_table": {
            **protocol_identity,
            "subject_unique_regular_per_participant": unique_count,
            "tau_train_regular_shared": tau_train_count,
            "tau_test_regular_shared": tau_test_count,
            "planning_counts_only": True,
        },
        "source_archive": {
            "path": str(args.source_archive),
            "md5": observed_md5,
            "size_bytes": observed_bytes,
            "identity_matches_qualification": True,
            "identity_matches_ep21_pin": True,
        },
        "source_members": {
            key: {"path": value["path"], "sha256": value["sha256"]}
            for key, value in members.items()
        },
        "planning_counts_are_unauthenticated_contract_fixtures": True,
        "mechanical_checks": {
            "tagged_source_static_assertions": static_checks,
            "dynamic_content_buffer_fixture": {
                "planned_row_count": unique_count,
                **content_guard_result,
                "allocated_shapes": allocated_shapes,
                "negative_tests": negative_content_tests,
                "optimizer_steps_executed": 0,
                "passed": True,
                "execution_requirement": "derive the actual authenticated row count at runtime and allocate every content-loss feature buffer with that exact count; never retain the tagged 6000 default implicitly",
            },
            "tau_role_guard_fixture": {
                **role_guard_result,
                "negative_tests": negative_role_tests,
                "passed": True,
                "execution_requirement": "route only authenticated tau-train rows to brain-loss fitting and enforce exact source/target label-set equality before the tagged fitter is called",
                "alignment_basis": "the tagged fitter independently sorts source and target rows by their unique image labels before normalization and fitting; the guard proves that both post-sort sequences equal the authenticated tau-train sequence",
                "repetition_scope": "synthetic fixture uses one row per unique image; authenticated repetition grouping and aggregation remain required before empirical execution",
            },
            "worker_binding_fixture": {
                **worker_binding,
                "binding": execution_binding,
                "executor_or_device_created": False,
                "oversubscription_rejected": oversubscription_rejected,
                "passed": True,
                "execution_requirement": "pass the resolved count as ProcessPoolExecutor(max_workers=...) because the tagged --n_cpu argument is unused",
            },
        },
        "tagged_source_hazards": {
            "content_row_allocation": {
                "state": "detected_guard_tested_patch_or_wrapper_required_before_empirical_run",
                "batch_default": batch_default,
                "number_size_default": number_size_default,
                "evidence": [
                    line_evidence(content, "Tensor(opt.batchSize, *shape[1:])"),
                    line_evidence(content, "Tensor(opt.batchSize, shape[1])"),
                    line_evidence(content, "copy_(torch.from_numpy(dnn_feature_batch))"),
                ],
            },
            "worker_binding": {
                "state": "detected_guard_tested_patch_required_before_empirical_run",
                "tagged_n_cpu_default": n_cpu_default,
                "executor_call": executor_calls[0],
            },
            "brain_loss_row_routing": {
                "state": "detected_external_prefit_guard_required_before_empirical_run",
                "num_sample_used_inside_train_function": False,
                "source_study_parameter_table": brain_table,
                "source_study_alpha_portability": "unresolved_scientific_choice_not_adopted",
                "evidence": [
                    line_evidence(brain, "num_sample,"),
                    line_evidence(brain, "x_index = np.argsort(x_labels.flatten())"),
                    line_evidence(brain, "y_index = np.argsort(y_labels.flatten())"),
                    line_evidence(brain, "model.fit(x_normalized, y_normalized, alpha=alpha, n_feat=0,"),
                ],
            },
        },
        "tagged_optimizer_facts_recorded_not_adopted": {
            "content_converter": {
                "iterations": iteration_default,
                "learning_rate": learning_rate_default,
                "adam_betas": [0.5, 0.999],
                "linear_decay_start_iteration": decay_default,
                "schedule_formula": "1 - max(0, iteration + offset - decay_start_iteration) / (n_iterations - decay_start_iteration)",
                "seed": 42,
                "step_order": [
                    "optimizer.zero_grad",
                    "loss.backward",
                    "optimizer.step",
                    "lr_scheduler.step",
                ],
                "checkpoint_rule": "final_iteration_only",
            },
            "reconstruction": {
                "optimizer": "SGD",
                "iterations": 200,
                "learning_rate_endpoints": [2.0, 1e-10],
                "momentum_endpoints": [0.9, 0.9],
                "decay_endpoints": [0.01, 0.01],
                "snapshot_interval": 10,
                "initialization": "unseeded_numpy_random_normal",
                "seed_and_restart_behavior": "unresolved",
            },
            "state": "outcome_blind_source_fact_pending_protocol_lock",
        },
        "scientific_decisions": {
            "activation": "unresolved_paper_relu_vs_tagged_leaky_relu_point2",
            "content_loss_weighting": "unresolved_paper_vs_tagged_normalized_per_block",
            "reconstruction_layers": "unresolved_paper_eight_vs_tagged_nineteen",
            "brain_loss_regularization": "unresolved_source_study_dyad_values_not_ported",
        },
        "protocol_lock_emitted": False,
        "empirical_readiness": "blocked",
        "blockers": [
            "scientist protocol choice remains unresolved",
            "LAION DUA and Sherlock risk classification remain unresolved",
            "role-filtered regular-only feature and scoring handoffs are absent",
            "authenticated participants roles tau split overlaps betas masks and array QC are absent",
            "AlexNet is unmanifested VGG binding is unproven and main bdpy is unpinned",
            "empirical neural execution is not authorized",
        ],
        "next_action": "after the scientist selects the paper- or tagged-code-faithful protocol and a steward provisions authenticated structural metadata, finalize the complete synthetic conformance suite and emit a fail-closed protocol lock; do not access empirical payloads before separate authorization",
    }
    atomic_json(args.output_dir / "neutral_conformance_report.json", report)
    print(
        json.dumps(
            {
                "event": "ep21_neutral_conformance_completed",
                "pid": process_id,
                "completed_at": completed_at,
                "output": str(args.output_dir / "neutral_conformance_report.json"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
