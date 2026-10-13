"""Annotation-only/source-schema preflight for EP12.

This module may read annotation values and Arrow schemas.  It never opens a
record batch from body statistics, graph weights, synapses, or transmitter
tables.  Connectivity outcome access remains disabled after preflight.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("source preflight needs the pinned pandas/pyarrow module environment") from exc

from .policy import digest_bytes, digest_object
from .synthetic import EPISODE_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


SOURCE_ROOT = Path(
    "/oak/stanford/groups/russpold/data/br_autoresearch_data/flyem_male_cns/"
    "gcs-male-cns-v1.0-flat-connectome"
)
ANNOTATION_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BODY_STATS_FILE = "body-stats-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def _schema_only(path: Path) -> dict[str, Any]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        return {
            "fields": [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in reader.schema
            ],
            "record_batches": int(reader.num_record_batches),
            "schema_sha256": digest_bytes(str(reader.schema).encode("utf-8")),
            "record_values_opened": False,
        }


def _type_hash(type_name: str) -> str:
    return hashlib.sha256(type_name.encode("utf-8")).hexdigest()


def _split_key(type_name: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}\0{type_name}".encode("utf-8")).hexdigest()


def _role_partition(
    annotations: pd.DataFrame, contract: Mapping[str, Any]
) -> dict[str, Any]:
    eligibility = contract["annotation_only_eligibility"]
    type_field = str(eligibility["provider_type_field"])
    side_field = str(eligibility["side_field"])
    status_field = str(eligibility["focal_status_field"])
    allowed_sides = [str(value) for value in eligibility["allowed_sides"]]
    allowed_status = [str(value) for value in eligibility["focal_status_allowed"]]
    minimum = int(eligibility["minimum_neurons_per_side"])
    if allowed_sides != ["L", "R"]:
        raise ValueError("this executor qualifies exactly the frozen bilateral L/R estimand")
    if eligibility["split_seed_derivation"] != "sha256_of_ep12_role_split_v1":
        raise ValueError("unsupported role split seed derivation")
    development_fraction = float(eligibility["development_fraction"])
    final_fraction = float(eligibility["final_fraction"])
    if not np.isclose(development_fraction + final_fraction, 1.0):
        raise ValueError("development and final role fractions must sum to one")
    namespace = "ep12-role-split-v1"
    type_name = annotations[type_field].astype("string").fillna("").str.strip()
    side = annotations[side_field].astype("string").fillna("")
    status = annotations[status_field].astype("string").fillna("")
    focal = type_name.ne("") & side.isin(allowed_sides) & status.isin(allowed_status)
    selected = pd.DataFrame(
        {"type": type_name[focal].astype(str), "side": side[focal].astype(str)}
    )
    counts = selected.groupby(["type", "side"]).size().unstack(fill_value=0)
    for required in ("L", "R"):
        if required not in counts:
            counts[required] = 0
    counts = counts[(counts["L"] >= minimum) & (counts["R"] >= minimum)].copy()
    counts["minimum_side_count"] = counts[["L", "R"]].min(axis=1)
    ordered = counts.sort_values(["minimum_side_count", "L", "R"], kind="mergesort")
    rank = pd.Series(np.arange(len(ordered)), index=ordered.index)
    ordered["minimum_side_count_decile"] = np.minimum(
        9, np.floor(rank * 10 / max(1, len(ordered))).astype(int)
    )
    blocks: list[tuple[int, list[str], float]] = []
    for decile, block in ordered.groupby("minimum_side_count_decile", sort=True):
        names = sorted(
            (str(name) for name in block.index),
            key=lambda name: _split_key(name, namespace),
        )
        exact_quota = len(names) * final_fraction
        blocks.append((int(decile), names, exact_quota))
    target_final = int(round(len(ordered) * final_fraction))
    quotas = {decile: int(np.floor(exact)) for decile, _, exact in blocks}
    remaining = target_final - sum(quotas.values())
    remainder_order = sorted(
        blocks,
        key=lambda item: (-(item[2] - np.floor(item[2])), item[0]),
    )
    for decile, _, _ in remainder_order[:remaining]:
        quotas[decile] += 1
    final_types: set[str] = set()
    for decile, names, _ in blocks:
        final_types.update(names[: quotas[decile]])
    rows: list[dict[str, Any]] = []
    for name, row in ordered.iterrows():
        role = "final" if str(name) in final_types else "development"
        rows.append(
            {
                "provider_type": str(name),
                "provider_type_sha256": _type_hash(str(name)),
                "left_neurons": int(row["L"]),
                "right_neurons": int(row["R"]),
                "minimum_side_count": int(row["minimum_side_count"]),
                "minimum_side_count_decile": int(row["minimum_side_count_decile"]),
                "role": role,
                "split_key_sha256": _split_key(str(name), namespace),
            }
        )
    development = sum(row["role"] == "development" for row in rows)
    final = sum(row["role"] == "final" for row in rows)
    return {
        "schema_version": "ep12.annotation_role_manifest.v1",
        "assignment_unit": "whole_provider_type",
        "eligibility": {
            "provider_type": f"nonempty {type_field}",
            "status": allowed_status,
            "side_field": side_field,
            "allowed_sides": allowed_sides,
            "minimum_neurons_per_side": minimum,
            "nonlateral_or_missing_side_rows": eligibility[
                "nonlateral_or_missing_side_rows"
            ],
        },
        "split": {
            "development_fraction_target": development_fraction,
            "final_fraction_target": final_fraction,
            "deterministic_key": f"sha256({namespace}\\0provider_type)",
            "balance": eligibility["split_balance_fields"],
        },
        "eligible_type_count": len(rows),
        "development_type_count": development,
        "final_type_count": final,
        "real_connectivity_used": False,
        "rows": rows,
        "rows_hash": digest_object(rows),
    }


def run_source_preflight(*, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    allowed_root = (EPISODE_ROOT / "outputs" / "prelaunch").resolve()
    if output_dir != allowed_root and allowed_root not in output_dir.parents:
        raise ValueError("source preflight outputs must stay under outputs/prelaunch")
    if SOURCE_ROOT.resolve() != SOURCE_ROOT:
        raise ValueError("source root must be canonical")
    source_dir = SOURCE_ROOT / "source"
    metadata_dir = SOURCE_ROOT / "source_metadata"
    contract_path = EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
    contract = load_yaml_subset(contract_path)
    if contract["real_connectivity_access_authorized"] not in (False, "development_only"):
        raise ValueError("source preflight permits at most development-only authorization")
    if contract.get("final_connectivity_access_authorized", False) is not False:
        raise ValueError("source preflight requires final connectivity to remain disabled")
    eligibility = contract["annotation_only_eligibility"]
    sha_map = json.loads((metadata_dir / "sha256_map.json").read_text(encoding="utf-8"))
    schemas = {
        filename: _schema_only(source_dir / filename)
        for filename in (ANNOTATION_FILE, BODY_STATS_FILE, WEIGHTS_FILE)
    }
    required_fields = {
        ANNOTATION_FILE: {"bodyId", "type", "somaSide", "status"},
        BODY_STATS_FILE: {
            "body",
            "pre",
            "post",
            "status_fine",
            "type",
            "downstream",
            "synweight",
            "rank",
        },
        WEIGHTS_FILE: {"body_pre", "body_post", "weight"},
    }
    schema_checks: list[dict[str, Any]] = []
    for filename, required in required_fields.items():
        observed = {field["name"] for field in schemas[filename]["fields"]}
        missing = sorted(required - observed)
        schema_checks.append(
            {
                "file": filename,
                "passed": not missing,
                "missing_fields": missing,
                "expected_sha256": sha_map[filename],
            }
        )

    annotation_path = source_dir / ANNOTATION_FILE
    annotation_table = feather.read_table(
        annotation_path,
        columns=[
            eligibility["provider_type_field"],
            eligibility["side_field"],
            eligibility["focal_status_field"],
        ],
        memory_map=True,
    )
    role_manifest = _role_partition(annotation_table.to_pandas(), contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_manifest = {
        "schema_version": "ep12.source_schema_manifest.v1",
        "provider_id": "male-cns:v1.0",
        "source_root": str(SOURCE_ROOT),
        "source_is_read_only": not bool(SOURCE_ROOT.stat().st_mode & 0o222),
        "files": {
            filename: {
                "path": str(source_dir / filename),
                "expected_sha256": sha_map[filename],
                **schemas[filename],
            }
            for filename in schemas
        },
        "checks": schema_checks,
        "all_checks_passed": all(check["passed"] for check in schema_checks),
    }
    exposure = {
        "schema_version": "ep12.prelaunch_exposure.v1",
        "annotation_schema_opened": True,
        "annotation_values_opened": True,
        "annotation_value_fields": ["type", "somaSide", "status"],
        "body_statistics_schema_opened": True,
        "body_statistics_values_opened": False,
        "connectome_weights_schema_opened": True,
        "connectome_weights_values_opened": False,
        "synapse_values_opened": False,
        "neurotransmitter_values_opened": False,
        "real_connectivity_outcome_values_opened": False,
        "scientific_contract_sha256": digest_bytes(contract_path.read_bytes()),
    }
    report = {
        "schema_version": "ep12.source_preflight_report.v1",
        "preflight_passed": schema_manifest["all_checks_passed"],
        "real_connectivity_outcome_values_opened": False,
        "source_schema_manifest": "source_schema_manifest.json",
        "role_manifest": "annotation_role_manifest.json",
        "exposure_record": "exposure_record.json",
        "eligible_type_count": role_manifest["eligible_type_count"],
        "development_type_count": role_manifest["development_type_count"],
        "final_type_count": role_manifest["final_type_count"],
        "role_rows_hash": role_manifest["rows_hash"],
    }
    atomic_json(output_dir / "source_schema_manifest.json", schema_manifest)
    atomic_json(output_dir / "annotation_role_manifest.json", role_manifest)
    atomic_json(output_dir / "exposure_record.json", exposure)
    atomic_json(output_dir / "preflight_report.json", report)
    manifest = {
        "preflight_report_sha256": digest_bytes((output_dir / "preflight_report.json").read_bytes()),
        "source_schema_manifest_sha256": digest_bytes(
            (output_dir / "source_schema_manifest.json").read_bytes()
        ),
        "annotation_role_manifest_sha256": digest_bytes(
            (output_dir / "annotation_role_manifest.json").read_bytes()
        ),
        "exposure_record_sha256": digest_bytes((output_dir / "exposure_record.json").read_bytes()),
        "preflight_passed": report["preflight_passed"],
        "real_connectivity_outcome_values_opened": False,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return report
