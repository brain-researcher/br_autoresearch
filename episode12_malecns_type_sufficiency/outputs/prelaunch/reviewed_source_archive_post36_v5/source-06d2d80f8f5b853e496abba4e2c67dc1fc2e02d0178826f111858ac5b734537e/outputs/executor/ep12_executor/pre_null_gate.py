"""Evaluate EP12's frozen prediction-gain gate before a full-search null."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adaptive_successors import (
    _inside_outputs,
    _read_configuration_payload,
    _verify_trial,
)
from .policy import EpisodePolicy, digest_bytes
from .scientific_models import seed_from
from .scientific_qualification import paired_simultaneous_intervals
from .synthetic import EPISODE_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


def _trial_deltas(result_root: Path, trial_id: str) -> np.ndarray:
    path = _inside_outputs(result_root) / trial_id / "result_table.csv"
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                [float(row["left_to_right_delta"]), float(row["right_to_left_delta"])]
            )
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid directional result table for {trial_id}")
    return values


def evaluate_pre_null_gate(
    *,
    source_specs: list[tuple[Path, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = _inside_outputs(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"pre-null gate directory is not empty: {output_dir}")
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    contract = load_yaml_subset(
        EPISODE_ROOT / "outputs" / "executor" / "SCIENTIFIC_CONTRACT.yaml"
    )
    replicates = int(contract["execution_parameters"]["simultaneous_bootstrap_replicates"])
    records: list[dict[str, Any]] = []
    trial_results: list[dict[str, Any]] = []
    for configuration_path, result_root in source_specs:
        configurations, hashes = _read_configuration_payload(configuration_path)
        for configuration, configuration_hash in zip(configurations, hashes):
            verified = _verify_trial(
                result_root=result_root,
                configuration=configuration,
                configuration_hash=configuration_hash,
            )
            records.append(verified)
            deltas = _trial_deltas(result_root, str(configuration["trial_id"]))
            intervals = paired_simultaneous_intervals(
                deltas,
                margin=policy.meaningful_margin,
                replicates=replicates,
                seed=seed_from(
                    "ep12-pre-null-gate",
                    configuration_hash,
                    verified["trial_report_sha256"],
                ),
            )
            lower = np.asarray(intervals["lower"], dtype=float)
            passes = bool(np.all(lower > policy.meaningful_margin))
            trial_results.append(
                {
                    "trial_id": configuration["trial_id"],
                    "configuration_sha256": configuration_hash,
                    "valid_type_count": int(len(deltas)),
                    "intervals": intervals,
                    "prediction_gain_pre_null_gate_pass": passes,
                }
            )
    if len(records) != policy.minimum_valid_trials:
        raise ValueError(
            f"pre-null gate requires {policy.minimum_valid_trials} trials, got {len(records)}"
        )
    eligible = [
        value["trial_id"]
        for value in trial_results
        if value["prediction_gain_pre_null_gate_pass"]
    ]
    report = {
        "schema_version": "ep12.pre_null_prediction_gain_gate.v1",
        "verified_trial_count": len(records),
        "meaningful_margin": policy.meaningful_margin,
        "simultaneous_interval_method": contract["uncertainty"]["interval"],
        "bootstrap_replicates": replicates,
        "trial_results": trial_results,
        "prediction_gain_eligible_trial_ids": eligible,
        "prediction_gain_eligible_trial_count": len(eligible),
        "full_search_null_required": None if eligible else False,
        "next_gate": (
            "mode_support_and_falsifiers_before_full_search_null"
            if eligible
            else "full_search_null_skipped_no_prediction_gain_eligible_comparison"
        ),
        "procedure_locked": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "pre_null_gate.json", report)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": "ep12.pre_null_prediction_gain_gate_manifest.v1",
            "files": {
                "pre_null_gate.json": digest_bytes(
                    (output_dir / "pre_null_gate.json").read_bytes()
                )
            },
            "final_connectivity_accessed": False,
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-pre-null-gate")
    parser.add_argument("--source", action="append", nargs=2, metavar=("CONFIG", "RESULTS"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = evaluate_pre_null_gate(
        source_specs=[(Path(config), Path(results)) for config, results in arguments.source],
        output_dir=arguments.output_dir,
    )
    print(
        json.dumps(
            {
                "verified_trial_count": report["verified_trial_count"],
                "prediction_gain_eligible_trial_count": report[
                    "prediction_gain_eligible_trial_count"
                ],
                "full_search_null_required": report["full_search_null_required"],
                "next_gate": report["next_gate"],
                "final_connectivity_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
