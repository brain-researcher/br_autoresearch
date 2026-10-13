#!/usr/bin/env python3
"""Materialize the sealed, target-blind EP10 v3 model interface.

This program deliberately has no target-vocabulary or target-call argument.
It opens exactly the frozen 200 development SWCs, evaluates target-blind
structural/mapping flags, and writes the exact 194 permitted interface rows.
Final and unassigned morphologies are never addressable by this interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import construct_mop_terminal_targets as core
import diagnose_mop_provider_native_v2_qc_overlap as diagnostic


OUTPUT_FIELDS = (
    "provider_neuron_id",
    "file",
    "fMOST_brain_id",
    "provider_sample_id",
    "source_acronym",
    "provider_hemisphere",
    "soma_x_ccfv3_um",
    "soma_y_ccfv3_um",
    "soma_z_ccfv3_um",
    "segment_unmapped",
    "terminal_unmapped_report_only",
    "mapping_overlap_class_report_only",
)
FORBIDDEN_OUTPUT_TOKENS = (
    "length",
    "terminal_count",
    "target",
    "coordinate_id",
    "primary_k",
    "projection",
)


def _load_contract(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as error:
        raise core.ConstructionError("interface execution requires PyYAML") from error
    with path.open("r", encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    if not isinstance(contract, dict):
        raise core.ConstructionError("v3 model-interface contract is not a mapping")
    if contract.get("status") != "frozen_after_independent_v3_policy_audit_before_real_model_fit":
        raise core.ConstructionError("v3 model-interface contract is not frozen")
    scope = contract.get("scope", {})
    if (
        scope.get("role") != "development_only"
        or int(scope.get("frozen_manifest_rows", -1)) != 200
        or int(scope.get("interface_rows", -1)) != 194
        or scope.get("final_or_unassigned_access") != "forbidden"
        or scope.get("target_vocabulary_or_calls_accepted_by_interface_builder") is not False
    ):
        raise core.ConstructionError("v3 model-interface scope changed")
    allowed = tuple(contract.get("interface", {}).get("allowed_columns", ()))
    if allowed != OUTPUT_FIELDS:
        raise core.ConstructionError("v3 model-interface columns/order changed")
    if any(token in field.lower() for field in OUTPUT_FIELDS for token in FORBIDDEN_OUTPUT_TOKENS):
        raise core.ConstructionError("forbidden outcome/count field reached interface")
    return contract


def _safe_row(result: diagnostic.DiagnosticResult) -> dict[str, object]:
    entry = result.entry
    segment_unmapped = bool(result.flags & diagnostic.SEGMENT_MAPPING_FLAGS)
    terminal_unmapped = bool(result.flags & diagnostic.TERMINAL_MAPPING_FLAGS)
    x, y, z = entry.canonical_soma_um
    row: dict[str, object] = {
        "provider_neuron_id": entry.provider_neuron_id,
        "file": entry.filename,
        "fMOST_brain_id": entry.brain_id,
        "provider_sample_id": entry.sample_id,
        "source_acronym": entry.source_acronym,
        "provider_hemisphere": entry.provider_hemisphere,
        "soma_x_ccfv3_um": format(x, ".12g"),
        "soma_y_ccfv3_um": format(y, ".12g"),
        "soma_z_ccfv3_um": format(z, ".12g"),
        "segment_unmapped": int(segment_unmapped),
        "terminal_unmapped_report_only": int(terminal_unmapped),
        "mapping_overlap_class_report_only": diagnostic._mapping_overlap_label(result),
    }
    if tuple(row) != OUTPUT_FIELDS:
        raise core.ConstructionError("safe interface row schema changed")
    return row


def _expected_maps(contract: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    gate = contract.get("exact_gate", {})
    groups = {str(key): int(value) for key, value in gate.get("eligible_cells_by_group", {}).items()}
    strata = {
        str(group): {"yes": int(counts["yes"]), "no": int(counts["no"])}
        for group, counts in gate.get("group_by_segment_unmapped", {}).items()
    }
    if len(groups) != 12 or len(strata) != 12 or sum(groups.values()) != 194:
        raise core.ConstructionError("interface contract exact group gate changed")
    return groups, strata


def _summarize(
    results: Sequence[diagnostic.DiagnosticResult], contract: Mapping[str, Any]
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    if len(results) != 200:
        raise core.ConstructionError("interface requires exactly 200 diagnostic rows")
    eligible = [result for result in results if not (result.flags & diagnostic.MANDATORY_FLAGS)]
    excluded = [result for result in results if result.flags & diagnostic.MANDATORY_FLAGS]
    if len(eligible) != 194 or len(excluded) != 6:
        raise core.ConstructionError(
            f"v3 interface gate changed: eligible={len(eligible)} excluded={len(excluded)}"
        )

    exclusion_counts: Counter[str] = Counter()
    for result in excluded:
        exclusion_counts.update(result.flags & diagnostic.MANDATORY_FLAGS)
    if dict(exclusion_counts) != {"soma_not_MOp": 6}:
        raise core.ConstructionError(
            f"v3 interface exclusion vector changed: {dict(exclusion_counts)}"
        )

    expected_groups, expected_strata = _expected_maps(contract)
    observed_groups = Counter(result.entry.brain_id for result in eligible)
    if dict(sorted(observed_groups.items())) != expected_groups:
        raise core.ConstructionError(
            f"v3 interface group vector changed: {dict(sorted(observed_groups.items()))}"
        )
    observed_strata: dict[str, dict[str, int]] = {}
    for group in sorted(expected_groups):
        group_results = [result for result in eligible if result.entry.brain_id == group]
        yes = sum(bool(result.flags & diagnostic.SEGMENT_MAPPING_FLAGS) for result in group_results)
        observed_strata[group] = {"yes": yes, "no": len(group_results) - yes}
    if observed_strata != expected_strata:
        raise core.ConstructionError(
            f"group by segment-unmapped support changed: {observed_strata}"
        )

    class_counts = Counter(diagnostic._mapping_overlap_label(result) for result in eligible)
    expected_classes = {
        str(key): int(value)
        for key, value in contract["exact_gate"]["mapping_overlap_class_counts"].items()
    }
    if dict(sorted(class_counts.items())) != dict(sorted(expected_classes.items())):
        raise core.ConstructionError(
            f"mapping-overlap class counts changed: {dict(sorted(class_counts.items()))}"
        )
    segment_yes = sum(values["yes"] for values in observed_strata.values())
    segment_no = sum(values["no"] for values in observed_strata.values())
    if (
        segment_yes != int(contract["exact_gate"]["segment_unmapped_cells"])
        or segment_no != int(contract["exact_gate"]["segment_mapped_cells"])
    ):
        raise core.ConstructionError("aggregate segment-unmapped support changed")

    rows = [_safe_row(result) for result in eligible]
    summary = {
        "status": "pass",
        "stage": "v3_target_blind_model_interface_preflight",
        "analysis_label": "exploratory",
        "input_development_rows": len(results),
        "interface_rows": len(rows),
        "excluded_rows": len(excluded),
        "exclusion_counts": dict(exclusion_counts),
        "eligible_cells_by_group": expected_groups,
        "group_by_segment_unmapped": observed_strata,
        "segment_unmapped_cells": segment_yes,
        "segment_mapped_cells": segment_no,
        "mapping_overlap_class_counts": dict(sorted(class_counts.items())),
        "interface_columns": list(OUTPUT_FIELDS),
        "model_mapping_quality_feature": "segment_unmapped",
        "terminal_unmapped_is_report_only": True,
        "raw_lengths_or_terminal_counts_emitted": False,
        "target_vocabulary_argument_supported": False,
        "target_vocabulary_or_calls_loaded": False,
        "projection_outcomes_opened_or_emitted": False,
        "final_or_unassigned_files_opened": False,
        "biological_absence_or_negative_label_implied": False,
    }
    return rows, summary


def validator_overlap_summary(overlap: Mapping[str, Any]) -> dict[str, Any]:
    """Add the frozen v3 validator's compact policy-by-group view.

    The overlap diagnostic's native summary is richer; the original v3
    validator consumes a compact rectangular projection.  This adapter is
    target-blind and changes no counts or eligibility decisions.
    """
    if (
        overlap.get("status") != "complete"
        or int(overlap.get("input_files", -1)) != 200
        or int(overlap.get("input_groups", -1)) != 12
        or overlap.get("target_vocabulary_loaded") is not False
        or overlap.get("projection_outcomes_computed_or_emitted") is not False
    ):
        raise core.ConstructionError("overlap summary is not eligible for v3 adapter")
    value_order = list(diagnostic.POLICY_DISQUALIFYING_FLAGS)
    by_group_detail = overlap.get("by_group", {})
    if not isinstance(by_group_detail, dict) or len(by_group_detail) != 12:
        raise core.ConstructionError("overlap summary lacks 12 group details")
    compact_by_group: dict[str, list[int]] = {}
    for group in sorted(by_group_detail):
        counts = by_group_detail[group].get("eligible_cells_by_policy", {})
        try:
            compact_by_group[group] = [int(counts[policy]) for policy in value_order]
        except (KeyError, TypeError, ValueError) as error:
            raise core.ConstructionError(
                f"overlap summary lacks policy counts for group {group}"
            ) from error
    adapted = dict(overlap)
    adapted.update(
        {
            "diagnostic_authorizes_target_construction": False,
            "group_policy_value_order": value_order,
            "eligible_cells_by_policy_and_group": compact_by_group,
        }
    )
    return adapted


def run_self_test() -> dict[str, Any]:
    diagnostic_summary = diagnostic.run_self_test()
    config, atlas, ontology, mapper = diagnostic._synthetic_assets()
    result = diagnostic._synthetic_result(
        """1 1 5 5 5 1 -1
2 10 15 5 5 1 1
3 2 25 5 5 1 2
""",
        atlas,
        config,
        ontology,
        mapper,
    )
    row = _safe_row(result)
    if tuple(row) != OUTPUT_FIELDS or row["segment_unmapped"] != 0:
        raise AssertionError("target-blind safe-row projection failed")
    if any(token in key.lower() for key in row for token in FORBIDDEN_OUTPUT_TOKENS):
        raise AssertionError("forbidden field reached target-blind interface")
    return {
        "status": "pass",
        "uses_real_atlas_swc_or_target_calls": False,
        "diagnostic_self_test": diagnostic_summary["status"],
        "checks": [
            "no_target_argument",
            "safe_boolean_mapping_projection",
            "no_raw_lengths_or_terminal_counts",
            "no_outcome_or_coordinate_fields",
        ],
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
    contract = _load_contract(args.interface_contract)
    revision = core._load_v2_revision(args.revision_contract)
    if revision["SWC_interpretation"]["mode"] != core.V2_INTERPRETATION:
        raise core.ConstructionError("revision contract is not provider-native v2")
    _, atlas, config, ontology, full_coordinates, coordinates = core._load_real_assets(
        args.contract,
        args.annotation,
        args.ontology,
        None,
        load_target_vocabulary=False,
    )
    if full_coordinates or coordinates:
        raise core.ConstructionError("target coordinates reached model-interface builder")
    mapper = core.StructureMapper(ontology, {}, config.grey_structure_id)

    entries = core.load_manifest(args.manifest)
    swc_root = core._validate_scratch_path(args.swc_root, "development SWC root")
    output_root = core._validate_scratch_path(args.output_root, "interface output root")
    if swc_root == output_root or swc_root in output_root.parents or output_root in swc_root.parents:
        raise core.ConstructionError("SWC input and interface output roots must be disjoint")
    swc_paths = core.validate_exact_swc_set(entries, swc_root)
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise core.ConstructionError("interface output root must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise core.ConstructionError("refusing symlink interface output root")

    results: list[diagnostic.DiagnosticResult] = []
    for entry in entries:
        try:
            nodes, _ = core.parse_swc(
                swc_paths[entry.provider_neuron_id],
                interpretation=core.V2_INTERPRETATION,
            )
        except (core.ConstructionError, OSError, UnicodeError) as error:
            result = diagnostic._parser_error_result(entry, error)
        else:
            result = diagnostic.diagnose_cell(entry, nodes, atlas, config, ontology, mapper)
        results.append(result)

    rows, summary = _summarize(results, contract)
    core._atomic_csv(output_root / "v3_model_interface.csv", OUTPUT_FIELDS, rows)
    core._atomic_json(output_root / "v3_model_interface_summary.json", summary)
    # Preserve a freshly reconstructed, target-blind aggregate in the schema
    # consumed by the v3 target validator.  This replaces the historical
    # sibling-sourced overlap artifact for all downstream replay work.
    overlap = diagnostic.build_summary(results)
    core._atomic_json(output_root / "aggregate_overlap_summary.json", overlap)
    core._atomic_json(
        output_root / "validator_overlap_summary.json",
        validator_overlap_summary(overlap),
    )
    print(json.dumps({**summary, "output_root": str(output_root)}, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--revision-contract", type=Path)
    parser.add_argument("--interface-contract", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--ontology", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--swc-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if args.execute:
            parser.error("--self-test and --execute are mutually exclusive")
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0
    if not args.execute:
        parser.error("real interface materialization requires --execute")
    required = (
        "contract",
        "revision_contract",
        "interface_contract",
        "annotation",
        "ontology",
        "manifest",
        "swc_root",
        "output_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    try:
        run_real(args)
    except core.ConstructionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
