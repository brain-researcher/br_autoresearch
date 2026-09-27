"""Finalize the 36-trial EP12 development search without opening final data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adaptive_successors import (
    _inside_outputs,
    _ranking_key,
    _read_configuration_payload,
    _scientific_signature,
    _verify_trial,
)
from .runtime import atomic_json


def finalize_observed_search(
    *,
    source_specs: list[tuple[Path, Path]],
    output_dir: Path,
) -> dict[str, object]:
    output_dir = _inside_outputs(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"observed finalization directory is not empty: {output_dir}")
    records: list[dict[str, object]] = []
    for configuration_path, result_root in source_specs:
        configurations, hashes = _read_configuration_payload(configuration_path)
        source_records = [
            _verify_trial(
                result_root=result_root,
                configuration=configuration,
                configuration_hash=configuration_hash,
            )
            for configuration, configuration_hash in zip(configurations, hashes)
        ]
        records.extend(source_records)
    if len(records) != 36:
        raise ValueError(f"observed minimum requires 36 valid trials, got {len(records)}")
    signatures = [_scientific_signature(record["configuration"]) for record in records]
    if len(set(signatures)) != len(signatures):
        raise ValueError("observed search contains duplicate scientific configurations")
    cycles = {
        str(record["configuration"]["stage"])
        if record["configuration"]["stage"] == "coverage"
        else str(record["trial_id"]).split("_")[1]
        for record in records
    }
    if cycles != {"coverage", "c1", "c2"}:
        raise ValueError(f"observed adaptive cycles are incomplete: {sorted(cycles)}")
    ranked = sorted(records, key=_ranking_key)
    archive_rows = [
        {
            "rank": index,
            "trial_id": record["trial_id"],
            "configuration_sha256": record["configuration_sha256"],
            "objective": record["objective"],
            "valid_type_count": record["valid_type_count"],
            "unscorable_type_count": record["unscorable_type_count"],
        }
        for index, record in enumerate(ranked, start=1)
    ]
    report: dict[str, object] = {
        "status": "observed_minimum_complete_pending_falsifiers_and_null",
        "verified_valid_trial_count": len(records),
        "coverage_trial_count": sum(
            record["configuration"]["stage"] == "coverage" for record in records
        ),
        "adaptive_cycle_1_trial_count": sum(
            str(record["trial_id"]).startswith("adaptive_c1_") for record in records
        ),
        "adaptive_cycle_2_trial_count": sum(
            str(record["trial_id"]).startswith("adaptive_c2_") for record in records
        ),
        "provisional_incumbent_trial_id": ranked[0]["trial_id"],
        "provisional_incumbent_configuration_sha256": ranked[0][
            "configuration_sha256"
        ],
        "provisional_only": True,
        "required_falsifiers_completed": False,
        "null_eligibility": "not_evaluated_pending_mode_support_and_falsifiers",
        "full_search_null_completed": False,
        "procedure_locked": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
        "archive": archive_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "observed_search_archive.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-finalize-observed")
    parser.add_argument("--source", action="append", nargs=2, metavar=("CONFIG", "RESULTS"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = finalize_observed_search(
        source_specs=[(Path(config), Path(results)) for config, results in arguments.source],
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
