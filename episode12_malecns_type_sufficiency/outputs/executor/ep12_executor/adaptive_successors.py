"""Verify observed EP12 trials and freeze deterministic adaptive successors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .adaptive_search import validate_development_configuration
from .policy import EpisodePolicy, digest_bytes, digest_object
from .runtime import EPISODE_ROOT, atomic_json


GRAMMAR_KEYS = (
    "partner_vocabulary",
    "partner_minimum_development_type_count",
    "rare_partner_pooling_mass_fraction",
    "partner_hierarchy_depth",
    "representation",
    "rank",
    "composition_pseudocount",
    "nuisance_spline_df",
    "covariance",
    "covariance_rank",
    "component_count",
    "regularization",
)


def _inside_outputs(path: Path) -> Path:
    resolved = path.resolve()
    root = (EPISODE_ROOT / "outputs").resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path must stay under EP12 outputs: {resolved}")
    return resolved


def _read_configuration_payload(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(_inside_outputs(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        configurations = [dict(value) for value in payload]
        hashes = [digest_object(value) for value in configurations]
    elif isinstance(payload, dict) and isinstance(payload.get("configurations"), list):
        configurations = [dict(value) for value in payload["configurations"]]
        hashes = [str(value) for value in payload.get("configuration_sha256", [])]
        if len(hashes) != len(configurations):
            raise ValueError("configuration manifest hash list is incomplete")
    else:
        raise ValueError("configuration manifest has the wrong schema")
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    for configuration, expected_hash in zip(configurations, hashes):
        validate_development_configuration(policy, configuration)
        if digest_object(configuration) != expected_hash:
            raise ValueError(f"configuration hash mismatch: {configuration.get('trial_id')}")
    return configurations, hashes


def _verify_trial(
    *,
    result_root: Path,
    configuration: dict[str, Any],
    configuration_hash: str,
) -> dict[str, Any]:
    trial_id = str(configuration["trial_id"])
    trial_dir = _inside_outputs(result_root) / trial_id
    manifest_path = trial_dir / "manifest.json"
    report_path = trial_dir / "trial_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise ValueError(f"missing durable artifacts for {trial_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("final_connectivity_accessed") is not False:
        raise ValueError(f"final-connectivity boundary missing for {trial_id}")
    for name, expected in manifest.get("files", {}).items():
        artifact = trial_dir / str(name)
        if not artifact.is_file() or digest_bytes(artifact.read_bytes()) != str(expected):
            raise ValueError(f"artifact digest mismatch for {trial_id}/{name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("trial_id") != trial_id:
        raise ValueError(f"trial identity mismatch for {trial_id}")
    if report.get("final_connectivity_accessed") is not False:
        raise ValueError(f"final connectivity entered {trial_id}")
    if report.get("status") != "completed":
        raise ValueError(f"trial did not complete cleanly: {trial_id}")
    aggregate = report.get("aggregate", {})
    if int(aggregate.get("technical_failure_count", -1)) != 0:
        raise ValueError(f"trial has technical failures: {trial_id}")
    if int(aggregate.get("valid_type_count", 0)) <= 0:
        raise ValueError(f"trial has no valid comparisons: {trial_id}")
    if report.get("configuration", {}).get("configuration_sha256") != configuration_hash:
        raise ValueError(f"executed configuration identity mismatch: {trial_id}")
    directional = (
        float(aggregate["mean_left_to_right_delta"]),
        float(aggregate["mean_right_to_left_delta"]),
    )
    bilateral = float(aggregate["mean_bilateral_delta"])
    if not all(math.isfinite(value) for value in (*directional, bilateral)):
        raise ValueError(f"nonfinite objective for {trial_id}")
    return {
        "trial_id": trial_id,
        "configuration": configuration,
        "configuration_sha256": configuration_hash,
        "objective": {
            "left_to_right": directional[0],
            "right_to_left": directional[1],
            "bilateral": bilateral,
            "bilateral_floor": min(directional),
            "direction_disagreement": abs(directional[0] - directional[1]),
            "types_both_above_margin": int(
                aggregate["types_with_both_point_estimates_above_margin"]
            ),
        },
        "valid_type_count": int(aggregate["valid_type_count"]),
        "unscorable_type_count": int(aggregate["unscorable_type_count"]),
        "trial_report_sha256": digest_bytes(report_path.read_bytes()),
        "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
    }


def _ranking_key(record: dict[str, Any]) -> tuple[Any, ...]:
    objective = record["objective"]
    return (
        -float(objective["bilateral_floor"]),
        -float(objective["bilateral"]),
        float(objective["direction_disagreement"]),
        -int(objective["types_both_above_margin"]),
        str(record["configuration_sha256"]),
    )


def _scientific_signature(configuration: dict[str, Any]) -> str:
    return digest_object(
        {
            key: value
            for key, value in configuration.items()
            if key not in {"trial_id", "stage"}
        }
    )


def _successors(
    *,
    policy: EpisodePolicy,
    cycle: int,
    ranked_parents: list[dict[str, Any]],
    seen_configurations: list[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen = {_scientific_signature(value) for value in seen_configurations}
    candidates: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
    for parent_rank, record in enumerate(ranked_parents, start=1):
        parent = record["configuration"]
        for key_rank, key in enumerate(GRAMMAR_KEYS):
            for value_rank, value in enumerate(policy.grammar_choices[key]):
                if parent[key] == value:
                    continue
                candidate = dict(parent)
                candidate[key] = value
                candidate["stage"] = "adaptive"
                candidate["trial_id"] = "pending"
                signature = _scientific_signature(candidate)
                if signature in seen:
                    continue
                lineage = {
                    "parent_trial_id": record["trial_id"],
                    "parent_configuration_sha256": record["configuration_sha256"],
                    "parent_rank": parent_rank,
                    "changed_operator": key,
                    "prior_value": parent[key],
                    "successor_value": value,
                    "evidence": record["objective"],
                }
                sort_key = (parent_rank, key_rank, value_rank, signature)
                candidates.append((sort_key, candidate, lineage))
    candidates.sort(key=lambda value: value[0])
    selected_configurations: list[dict[str, Any]] = []
    selected_lineage: list[dict[str, Any]] = []
    for _, configuration, lineage in candidates:
        signature = _scientific_signature(configuration)
        if signature in seen:
            continue
        index = len(selected_configurations) + 1
        configuration["trial_id"] = f"adaptive_c{cycle}_{index:03d}"
        validate_development_configuration(policy, configuration)
        selected_configurations.append(configuration)
        selected_lineage.append(lineage)
        seen.add(signature)
        if len(selected_configurations) == count:
            break
    if len(selected_configurations) != count:
        raise ValueError(f"only {len(selected_configurations)} unseen successors were available")
    return selected_configurations, selected_lineage


def plan_successor_cycle(
    *,
    cycle: int,
    source_specs: list[tuple[Path, Path]],
    output_dir: Path,
    successor_count: int = 9,
) -> dict[str, Any]:
    if cycle not in {1, 2}:
        raise ValueError("the minimum search freezes exactly two successor cycles")
    output_dir = _inside_outputs(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"successor plan directory is not empty: {output_dir}")
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    all_records: list[dict[str, Any]] = []
    all_configurations: list[dict[str, Any]] = []
    records_by_source: list[list[dict[str, Any]]] = []
    for configuration_path, result_root in source_specs:
        configurations, hashes = _read_configuration_payload(configuration_path)
        records = [
            _verify_trial(
                result_root=result_root,
                configuration=configuration,
                configuration_hash=configuration_hash,
            )
            for configuration, configuration_hash in zip(configurations, hashes)
        ]
        records_by_source.append(records)
        all_records.extend(records)
        all_configurations.extend(configurations)
    expected_prior = 18 if cycle == 1 else 27
    if len(all_records) != expected_prior:
        raise ValueError(
            f"cycle {cycle} requires {expected_prior} prior valid trials, got {len(all_records)}"
        )
    parent_records = records_by_source[-1] if cycle == 2 else all_records
    ranked_parents = sorted(parent_records, key=_ranking_key)
    configurations, lineage = _successors(
        policy=policy,
        cycle=cycle,
        ranked_parents=ranked_parents,
        seen_configurations=all_configurations,
        count=successor_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cycle": cycle,
        "proposal_rule": (
            "rank parents lexicographically by bilateral directional floor, bilateral "
            "mean, direction agreement, count above the existing margin, then frozen "
            "hash; enumerate one-operator grammar neighbors in SEARCH_POLICY order"
        ),
        "configurations": configurations,
        "configuration_sha256": [digest_object(value) for value in configurations],
        "lineage": lineage,
        "final_connectivity_accessed": False,
    }
    configuration_output = output_dir / "successor_configurations.json"
    atomic_json(configuration_output, payload)
    report = {
        "cycle": cycle,
        "verified_prior_trial_count": len(all_records),
        "ranked_parent_trial_ids": [value["trial_id"] for value in ranked_parents],
        "successor_count": len(configurations),
        "successor_configuration_sha256": payload["configuration_sha256"],
        "final_connectivity_accessed": False,
    }
    atomic_json(output_dir / "successor_plan.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-plan-successors")
    parser.add_argument("--cycle", type=int, choices=(1, 2), required=True)
    parser.add_argument("--source", action="append", nargs=2, metavar=("CONFIG", "RESULTS"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = plan_successor_cycle(
        cycle=arguments.cycle,
        source_specs=[(Path(config), Path(results)) for config, results in arguments.source],
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
