#!/usr/bin/env python3
"""Validate and label EP10 v3 positive-only target-construction outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BY_GROUP = {
    "212074": 6,
    "220187": 35,
    "220188": 25,
    "220190": 16,
    "221091": 33,
    "221225": 39,
    "221418": 12,
    "221514": 5,
    "221539": 1,
    "221588": 7,
    "221751": 13,
    "233281": 2,
}
POLICY = "tolerate_both_mapping_classes"
ZERO_SEMANTICS = (
    "no mapped positive terminal observation recorded for this coordinate; "
    "not biological absence, verified non-detection, or a negative label"
)


class ValidationError(RuntimeError):
    pass


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"{path} is not a JSON object")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise ValidationError("PyYAML is required") from error
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValidationError(f"{path} is not a YAML mapping")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_contracts(
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    overlap: Mapping[str, Any],
) -> None:
    gate = contract.get("v3_structural_gate", {})
    endpoint = contract.get("positive_only_observation_endpoint", {})
    if (
        contract.get("episode_id") != "ep10"
        or contract.get("status")
        != "frozen_after_v2_outcome_blind_overlap_before_v3_target_construction"
        or endpoint.get("endpoint_name")
        != "mapped_positive_terminal_observation_set"
        or int(gate.get("expected_eligible_cells", -1)) != 194
        or int(gate.get("expected_excluded_cells", -1)) != 6
        or gate.get("expected_eligible_cells_by_group") != EXPECTED_BY_GROUP
        or int(gate.get("group_221588_authorized_support", -1)) != 7
        or gate.get("all_12_groups_must_match_expected_support") is not True
        or gate.get("no_further_relaxation_replacement_or_threshold_tuning")
        is not True
    ):
        raise ValidationError("v3 contract differs from the authorized frozen rule")

    scope = authorization.get("authorized_scope", {})
    boundaries = authorization.get("preserved_boundaries", {})
    if (
        authorization.get("status") != "explicit_user_authorization_received"
        or scope.get("analysis_label") != "exploratory"
        or scope.get("observation_policy") != "positive_only_unmapped_tolerant"
        or int(scope.get("expected_development_cells", -1)) != 194
        or int(scope.get("group_221588_support", -1)) != 7
        or boundaries.get("final_and_unassigned_remain_sealed_during_development")
        is not True
        or boundaries.get("do_not_relabel_v2_as_passed") is not True
    ):
        raise ValidationError("v3 user authorization record is incomplete or changed")

    policy_counts = overlap.get("eligible_cells_by_policy", {})
    value_order = overlap.get("group_policy_value_order", [])
    by_group = overlap.get("eligible_cells_by_policy_and_group", {})
    try:
        policy_index = value_order.index(POLICY)
    except (AttributeError, ValueError) as error:
        raise ValidationError("v2 overlap summary lacks the authorized policy") from error
    observed_by_group = {
        group: int(values[policy_index]) for group, values in by_group.items()
    }
    if (
        overlap.get("status") != "complete"
        or overlap.get("diagnostic_authorizes_target_construction") is not False
        or int(policy_counts.get(POLICY, -1)) != 194
        or observed_by_group != EXPECTED_BY_GROUP
        or overlap.get("target_vocabulary_loaded") is not False
        or overlap.get("projection_outcomes_computed_or_emitted") is not False
    ):
        raise ValidationError("target-blind v2 overlap evidence does not match v3")


def validate_rows(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    if len(rows) != 200:
        raise ValidationError(f"expected 200 eligibility rows, found {len(rows)}")
    identities = [row.get("provider_neuron_id", "") for row in rows]
    filenames = [row.get("file", "") for row in rows]
    if not all(identities) or len(set(identities)) != 200:
        raise ValidationError("eligibility rows do not have 200 unique neuron IDs")
    if not all(filenames) or len(set(filenames)) != 200:
        raise ValidationError("eligibility rows do not have 200 unique filenames")

    eligible_rows = [row for row in rows if row.get("global_eligible") == "yes"]
    invalid_labels = [
        row.get("global_eligible")
        for row in rows
        if row.get("global_eligible") not in {"yes", "no"}
    ]
    if invalid_labels:
        raise ValidationError("eligibility CSV contains labels other than yes/no")
    by_group = dict(
        sorted(Counter(row["fMOST_brain_id"] for row in eligible_rows).items())
    )
    failures = dict(
        sorted(
            Counter(
                row.get("global_eligibility_failure", "")
                for row in rows
                if row.get("global_eligible") == "no"
            ).items()
        )
    )
    if len(eligible_rows) != 194 or by_group != EXPECTED_BY_GROUP:
        raise ValidationError(
            f"v3 eligibility mismatch: eligible={len(eligible_rows)}, groups={by_group}"
        )
    if failures != {"soma_not_MOp": 6}:
        raise ValidationError(f"unexpected v3 exclusion counts: {failures}")
    return by_group, failures


def validate_primary_calls(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, int], dict[int, int]]:
    if len(rows) != 200:
        raise ValidationError(f"expected 200 primary-call rows, found {len(rows)}")
    identities = [row.get("provider_neuron_id", "") for row in rows]
    if not all(identities) or len(set(identities)) != 200:
        raise ValidationError("primary-call rows do not have 200 unique neuron IDs")
    if not rows:
        raise ValidationError("primary-call table is empty")

    detected_prefix = "detected__"
    zero_prefix = "binary_zero_semantics__"
    detected_columns = [
        column for column in rows[0] if column.startswith(detected_prefix)
    ]
    zero_columns = [column for column in rows[0] if column.startswith(zero_prefix)]
    detected_ids = [column.removeprefix(detected_prefix) for column in detected_columns]
    zero_ids = [column.removeprefix(zero_prefix) for column in zero_columns]
    if len(detected_ids) != 16 or detected_ids != zero_ids:
        raise ValidationError("primary-call table lacks the paired fixed 16 columns")

    eligible_by_id: dict[str, int] = {}
    k_distribution: Counter[int] = Counter()
    for row in rows:
        neuron_id = row["provider_neuron_id"]
        eligible = row.get("global_eligible")
        if eligible == "yes":
            values = [row[column] for column in detected_columns]
            if any(value not in {"0", "1"} for value in values):
                raise ValidationError(f"nonbinary v3 call for neuron {neuron_id}")
            observed_k = sum(value == "1" for value in values)
            try:
                reported_k = int(row.get("K", ""))
            except ValueError as error:
                raise ValidationError(f"invalid K for neuron {neuron_id}") from error
            if reported_k != observed_k:
                raise ValidationError(f"K disagrees with calls for neuron {neuron_id}")
            expected_set = [
                coordinate_id
                for coordinate_id, value in zip(detected_ids, values)
                if value == "1"
            ]
            reported_set = (
                row.get("S_coordinate_ids", "").split(";")
                if row.get("S_coordinate_ids", "")
                else []
            )
            if reported_set != expected_set:
                raise ValidationError(
                    f"S_coordinate_ids disagrees with calls for neuron {neuron_id}"
                )
            for coordinate_id, value in zip(detected_ids, values):
                zero_value = row[f"{zero_prefix}{coordinate_id}"]
                expected_zero = "" if value == "1" else ZERO_SEMANTICS
                if zero_value != expected_zero:
                    raise ValidationError(
                        f"v3 zero semantics mismatch for neuron {neuron_id}, "
                        f"coordinate {coordinate_id}"
                    )
            k_distribution[observed_k] += 1
            eligible_by_id[neuron_id] = 1
        elif eligible == "no":
            required_blank = [
                row.get("K", ""),
                row.get("S_coordinate_ids", ""),
                *(row[column] for column in detected_columns),
                *(row[column] for column in zero_columns),
            ]
            if any(value != "" for value in required_blank):
                raise ValidationError(
                    f"globally excluded neuron {neuron_id} has target calls"
                )
            eligible_by_id[neuron_id] = 0
        else:
            raise ValidationError(f"invalid eligibility label for neuron {neuron_id}")
    if sum(eligible_by_id.values()) != 194:
        raise ValidationError("primary-call table does not contain 194 eligible rows")
    return eligible_by_id, dict(sorted(k_distribution.items()))


def validate_and_relabel(
    output_root: Path,
    contract_path: Path,
    authorization_path: Path,
    overlap_path: Path,
) -> Mapping[str, Any]:
    contract = _load_yaml(contract_path)
    authorization = _load_yaml(authorization_path)
    overlap = _load_json(overlap_path)
    validate_contracts(contract, authorization, overlap)

    eligibility_path = output_root / "cell_eligibility_qc.csv"
    with eligibility_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_group, failures = validate_rows(rows)

    primary_path = output_root / "primary_16_target_calls.csv"
    with primary_path.open(newline="", encoding="utf-8") as handle:
        primary_rows = list(csv.DictReader(handle))
    primary_eligibility, observed_k_distribution = validate_primary_calls(
        primary_rows
    )
    eligibility_by_id = {
        row["provider_neuron_id"]: int(row["global_eligible"] == "yes")
        for row in rows
    }
    if primary_eligibility != eligibility_by_id:
        raise ValidationError("eligibility and primary-call tables disagree")

    summary_path = output_root / "aggregate_summary.json"
    summary = dict(_load_json(summary_path))
    k_distribution = summary.get("K_distribution", {})
    expected_k_distribution = {
        str(key): value for key, value in observed_k_distribution.items()
    }
    if (
        summary.get("status") != "complete"
        or int(summary.get("input_files", -1)) != 200
        or int(summary.get("input_groups", -1)) != 12
        or int(summary.get("globally_eligible_primary_model_cells", -1)) != 194
        or int(summary.get("globally_unknown_excluded_cells", -1)) != 6
        or summary.get("global_eligibility_failure_counts") != failures
        or int(summary.get("primary_coordinate_count", -1)) != 16
        or k_distribution != expected_k_distribution
        or summary.get("binary_zero_semantics") != ZERO_SEMANTICS
        or summary.get("final_or_unassigned_files_opened") is not False
        or summary.get("EP11_shared_units_opened") is not False
    ):
        raise ValidationError("constructor aggregate summary failed v3 postflight")

    summary.update(
        {
            "stage": "v3_positive_only_target_construction",
            "analysis_label": "exploratory",
            "observation_policy": "positive_only_unmapped_tolerant_v3",
            "eligible_cells_by_group": by_group,
            "binary_zero_semantics": ZERO_SEMANTICS,
            "complete_target_set_interpretation_allowed": False,
            "verified_non_detection_interpretation_allowed": False,
            "mapping_incompleteness_tolerated": True,
            "target_vocabulary_loaded": True,
            "projection_outcomes_computed": True,
            "projection_outcomes_emitted": True,
            "v3_contract": contract_path.name,
            "v3_user_authorization": authorization_path.name,
            "v2_structural_gate_relabelled_as_passed": False,
        }
    )
    _atomic_json(summary_path, summary)

    gate_summary = {
        "status": "pass",
        "stage": "v3_positive_only_postflight",
        "analysis_label": "exploratory",
        "input_files": 200,
        "input_groups": 12,
        "eligible_cells": 194,
        "excluded_cells": 6,
        "eligible_cells_by_group": by_group,
        "exclusion_counts": failures,
        "group_221588_support": by_group["221588"],
        "observation_policy": "positive_only_unmapped_tolerant_v3",
        "binary_zero_semantics": ZERO_SEMANTICS,
        "target_vocabulary_loaded": True,
        "projection_outcomes_emitted": True,
        "final_or_unassigned_files_opened": False,
        "v1_and_v2_relabelled_as_passed": False,
    }
    _atomic_json(output_root / "v3_gate_summary.json", gate_summary)
    return gate_summary


def run_self_test() -> Mapping[str, Any]:
    rows = []
    neuron_id = 1
    for group, count in EXPECTED_BY_GROUP.items():
        for index in range(count):
            rows.append(
                {
                    "provider_neuron_id": str(neuron_id),
                    "file": f"{group}_{index:03d}.swc",
                    "fMOST_brain_id": group,
                    "global_eligible": "yes",
                    "global_eligibility_failure": "",
                }
            )
            neuron_id += 1
    for index in range(6):
        rows.append(
            {
                "provider_neuron_id": str(neuron_id),
                "file": f"excluded_{index:03d}.swc",
                "fMOST_brain_id": "220187",
                "global_eligible": "no",
                "global_eligibility_failure": "soma_not_MOp",
            }
        )
        neuron_id += 1
    by_group, failures = validate_rows(rows)
    if by_group != EXPECTED_BY_GROUP or failures != {"soma_not_MOp": 6}:
        raise AssertionError("synthetic v3 validation did not reproduce the gate")

    coordinate_ids = [f"coordinate_{index:02d}" for index in range(16)]
    primary_rows = []
    for row in rows:
        eligible = row["global_eligible"] == "yes"
        primary_row = {
            "provider_neuron_id": row["provider_neuron_id"],
            "global_eligible": row["global_eligible"],
            "K": "1" if eligible else "",
            "S_coordinate_ids": coordinate_ids[0] if eligible else "",
        }
        for index, coordinate_id in enumerate(coordinate_ids):
            detected = eligible and index == 0
            primary_row[f"detected__{coordinate_id}"] = (
                str(int(detected)) if eligible else ""
            )
            primary_row[f"binary_zero_semantics__{coordinate_id}"] = (
                "" if detected or not eligible else ZERO_SEMANTICS
            )
        primary_rows.append(primary_row)
    primary_eligibility, k_distribution = validate_primary_calls(primary_rows)
    if sum(primary_eligibility.values()) != 194 or k_distribution != {1: 194}:
        raise AssertionError("synthetic v3 primary-call validation failed")
    return {
        "status": "pass",
        "checks": [
            "exact_194_cells",
            "exact_group_support",
            "six_soma_exclusions",
            "per_cell_positive_only_zero_semantics",
        ],
        "uses_real_data": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--overlap-summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0
    if args.preflight_only:
        missing = [
            name
            for name in ("contract", "authorization", "overlap_summary")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValidationError(f"missing preflight arguments: {missing}")
        validate_contracts(
            _load_yaml(args.contract),
            _load_yaml(args.authorization),
            _load_json(args.overlap_summary),
        )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "stage": "v3_positive_only_preflight",
                    "eligible_cells": 194,
                    "group_221588_support": 7,
                    "target_vocabulary_loaded": False,
                    "projection_outcomes_computed_or_emitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute:
        raise ValidationError("real postflight requires --execute")
    missing = [
        name
        for name in ("output_root", "contract", "authorization", "overlap_summary")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValidationError(f"missing required arguments: {missing}")
    result = validate_and_relabel(
        args.output_root, args.contract, args.authorization, args.overlap_summary
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, ValidationError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
