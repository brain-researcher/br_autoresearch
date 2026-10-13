#!/usr/bin/env python3
"""Diagnose overlap among EP10 provider-native v2 structural QC flags.

This diagnostic is deliberately target-blind.  It validates and opens only the
same 200 frozen development SWCs, loads the Allen annotation and ontology with
an empty named-target map, and evaluates every applicable structural QC flag
for each successfully parsed cell.  It never accepts a target-vocabulary path
and never computes target calls, candidate coordinates, or K.

Real-data access is gated by ``--execute``.  ``--self-test`` is synthetic and
does not read a real SWC, annotation, manifest, or projection outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import construct_mop_terminal_targets as core


FLAG_ORDER = (
    "parser_error",
    "canonical_soma_mismatch",
    "provider_soma_hemisphere_mismatch",
    "soma_outside_volume",
    "soma_annotation_zero",
    "soma_not_MOp",
    "no_positive_provider_axon_edge",
    "provider_axon_positive_length_outside_volume",
    "provider_axon_positive_length_annotation_zero",
    "provider_terminal_outside_volume",
    "provider_terminal_annotation_zero",
    "provider_terminal_nongrey",
)

FLAG_DEFINITIONS = {
    "parser_error": "the provider-native v2 SWC parser rejected the tree",
    "canonical_soma_mismatch": "the SWC root coordinate differs from the frozen soma",
    "provider_soma_hemisphere_mismatch": (
        "the provider hemisphere differs from the side of the SWC root"
    ),
    "soma_outside_volume": "the SWC root coordinate is outside the annotation volume",
    "soma_annotation_zero": "the in-volume SWC root maps to annotation ID zero",
    "soma_not_MOp": "the nonzero SWC root annotation is not MOp or its descendant",
    "no_positive_provider_axon_edge": (
        "no positive-length edge has both endpoint raw types outside {3, 4}"
    ),
    "provider_axon_positive_length_outside_volume": (
        "some positive provider-axon centerline length lies outside the volume"
    ),
    "provider_axon_positive_length_annotation_zero": (
        "some positive provider-axon centerline length maps to annotation ID zero"
    ),
    "provider_terminal_outside_volume": (
        "at least one provider topology terminal lies outside the volume"
    ),
    "provider_terminal_annotation_zero": (
        "at least one in-volume provider topology terminal maps to annotation ID zero"
    ),
    "provider_terminal_nongrey": (
        "at least one nonzero provider topology terminal lies outside Allen grey"
    ),
}

PARSER_FLAGS = frozenset(("parser_error",))
SOMA_FLAGS = frozenset(
    (
        "canonical_soma_mismatch",
        "provider_soma_hemisphere_mismatch",
        "soma_outside_volume",
        "soma_annotation_zero",
        "soma_not_MOp",
    )
)
POSITIVE_AXON_FLAGS = frozenset(("no_positive_provider_axon_edge",))
SEGMENT_MAPPING_FLAGS = frozenset(
    (
        "provider_axon_positive_length_outside_volume",
        "provider_axon_positive_length_annotation_zero",
    )
)
TERMINAL_MAPPING_FLAGS = frozenset(
    (
        "provider_terminal_outside_volume",
        "provider_terminal_annotation_zero",
        "provider_terminal_nongrey",
    )
)
MANDATORY_FLAGS = PARSER_FLAGS | SOMA_FLAGS | POSITIVE_AXON_FLAGS
ALL_FLAGS = frozenset(FLAG_ORDER)

POLICY_DISQUALIFYING_FLAGS = {
    "strict_current": ALL_FLAGS,
    "tolerate_segment_outside_or_zero_only": MANDATORY_FLAGS
    | TERMINAL_MAPPING_FLAGS,
    "tolerate_terminal_nongrey_outside_or_zero_only": MANDATORY_FLAGS
    | SEGMENT_MAPPING_FLAGS,
    "tolerate_both_mapping_classes": MANDATORY_FLAGS,
}


@dataclass(frozen=True)
class DiagnosticResult:
    entry: core.ManifestEntry
    flags: frozenset[str]
    parser_error_detail: str = ""
    post_parse_flags_evaluated: bool = True
    provider_axon_edge_count: int = 0
    positive_provider_axon_edge_count: int = 0
    provider_terminal_count: int = 0
    provider_axon_outside_length_um: float = 0.0
    provider_axon_annotation_zero_length_um: float = 0.0
    provider_terminals_outside_count: int = 0
    provider_terminals_annotation_zero_count: int = 0
    provider_terminals_nongrey_count: int = 0


def _ordered_flags(flags: Iterable[str]) -> tuple[str, ...]:
    observed = frozenset(flags)
    unknown = observed - ALL_FLAGS
    if unknown:
        raise core.ConstructionError(f"unknown diagnostic flags: {sorted(unknown)}")
    return tuple(flag for flag in FLAG_ORDER if flag in observed)


def _combination_label(flags: Iterable[str]) -> str:
    ordered = _ordered_flags(flags)
    return "|".join(ordered) if ordered else "none"


def _eligible(result: DiagnosticResult, policy: str) -> bool:
    try:
        disqualifying = POLICY_DISQUALIFYING_FLAGS[policy]
    except KeyError as error:
        raise core.ConstructionError(f"unknown diagnostic policy {policy!r}") from error
    return not bool(result.flags & disqualifying)


def _mapping_overlap_label(result: DiagnosticResult) -> str:
    has_segment = bool(result.flags & SEGMENT_MAPPING_FLAGS)
    has_terminal = bool(result.flags & TERMINAL_MAPPING_FLAGS)
    if has_segment and has_terminal:
        return "both_segment_and_terminal_mapping_flags"
    if has_segment:
        return "segment_mapping_flags_only"
    if has_terminal:
        return "terminal_mapping_flags_only"
    return "neither_mapping_class"


def _parser_error_result(
    entry: core.ManifestEntry, error: BaseException
) -> DiagnosticResult:
    return DiagnosticResult(
        entry=entry,
        flags=frozenset(("parser_error",)),
        parser_error_detail=str(error),
        post_parse_flags_evaluated=False,
    )


def diagnose_cell(
    entry: core.ManifestEntry,
    nodes: Mapping[int, core.SwcNode],
    atlas: core.Atlas,
    config: core.GeometryConfig,
    ontology: core.Ontology,
    mapper: core.StructureMapper,
) -> DiagnosticResult:
    """Evaluate all applicable v2 structural flags without early exit."""

    if mapper.anchor_to_target:
        raise core.ConstructionError(
            "overlap diagnostic requires an empty named-target map"
        )
    flags: set[str] = set()
    roots = [node for node in nodes.values() if node.parent_id == -1]
    if len(roots) != 1 or roots[0].node_type != 1:
        raise core.ConstructionError("validated root/soma invariant was lost")
    root = roots[0]

    if root.point_um != entry.canonical_soma_um:
        flags.add("canonical_soma_mismatch")
    soma_hemisphere = core._soma_hemisphere(
        root.point_um[2], config.ml_midline_um
    )
    if core._normalize_hemisphere(entry.provider_hemisphere) != soma_hemisphere:
        flags.add("provider_soma_hemisphere_mismatch")

    soma_index = core._voxel_index(root.point_um, config)
    if soma_index is None:
        flags.add("soma_outside_volume")
    else:
        soma_structure = atlas.structure_id(soma_index)
        if soma_structure == 0:
            flags.add("soma_annotation_zero")
        else:
            mop_id = ontology.unique_id_for_acronym("MOp")
            if not ontology.is_descendant_or_self(soma_structure, mop_id):
                flags.add("soma_not_MOp")

    provider_axon_edge_count = 0
    positive_provider_axon_edge_count = 0
    allocation_qc = core.CellQC(
        provider_neuron_id=entry.provider_neuron_id,
        filename=entry.filename,
        brain_id=entry.brain_id,
        sample_id=entry.sample_id,
    )
    next_piece_id = 0
    total_provider_axon_length_um = 0.0
    for child in nodes.values():
        if child.parent_id == -1:
            continue
        parent = nodes[child.parent_id]
        if not core._is_axon_edge(parent, child, core.V2_INTERPRETATION):
            continue
        provider_axon_edge_count += 1
        edge_length = core.euclidean_length(parent.point_um, child.point_um)
        total_provider_axon_length_um += edge_length
        if edge_length > config.zero_tolerance_um:
            positive_provider_axon_edge_count += 1
        pieces = core.allocate_edge(
            child.node_id,
            parent.point_um,
            child.point_um,
            root.point_um[2],
            atlas,
            config,
            mapper,
            None,
            {},
            next_piece_id,
            allocation_qc,
        )
        next_piece_id += len(pieces)

    if positive_provider_axon_edge_count == 0:
        flags.add("no_positive_provider_axon_edge")
    allocated_length_um = sum(allocation_qc.category_lengths_um.values())
    allocation_tolerance = max(
        config.zero_tolerance_um, total_provider_axon_length_um * 1.0e-11
    )
    if not math.isclose(
        allocated_length_um,
        total_provider_axon_length_um,
        rel_tol=0.0,
        abs_tol=allocation_tolerance,
    ):
        raise core.ConstructionError(
            "provider-axon allocation failed conservation by "
            f"{allocated_length_um - total_provider_axon_length_um:.12g} um"
        )
    outside_length_um = allocation_qc.category_lengths_um["outside_volume"]
    annotation_zero_length_um = allocation_qc.category_lengths_um[
        "annotation_zero"
    ]
    if outside_length_um > config.zero_tolerance_um:
        flags.add("provider_axon_positive_length_outside_volume")
    if annotation_zero_length_um > config.zero_tolerance_um:
        flags.add("provider_axon_positive_length_annotation_zero")

    provider_terminal_ids = core._terminal_node_ids(
        nodes, core.V2_INTERPRETATION
    )
    terminal_outside_count = 0
    terminal_annotation_zero_count = 0
    terminal_nongrey_count = 0
    for terminal_id in provider_terminal_ids:
        terminal = nodes[terminal_id]
        terminal_index = core._voxel_index(terminal.point_um, config)
        if terminal_index is None:
            terminal_outside_count += 1
            continue
        structure_id = atlas.structure_id(terminal_index)
        if structure_id == 0:
            terminal_annotation_zero_count += 1
            continue
        category, target_index = mapper.classify(structure_id)
        if target_index is not None:
            raise core.ConstructionError(
                "named-target assignment reached target-blind overlap diagnostic"
            )
        if category != "unclaimed_grey":
            terminal_nongrey_count += 1

    if terminal_outside_count:
        flags.add("provider_terminal_outside_volume")
    if terminal_annotation_zero_count:
        flags.add("provider_terminal_annotation_zero")
    if terminal_nongrey_count:
        flags.add("provider_terminal_nongrey")

    return DiagnosticResult(
        entry=entry,
        flags=frozenset(flags),
        provider_axon_edge_count=provider_axon_edge_count,
        positive_provider_axon_edge_count=positive_provider_axon_edge_count,
        provider_terminal_count=len(provider_terminal_ids),
        provider_axon_outside_length_um=outside_length_um,
        provider_axon_annotation_zero_length_um=annotation_zero_length_um,
        provider_terminals_outside_count=terminal_outside_count,
        provider_terminals_annotation_zero_count=terminal_annotation_zero_count,
        provider_terminals_nongrey_count=terminal_nongrey_count,
    )


def _marginal_counts(results: Sequence[DiagnosticResult]) -> dict[str, int]:
    return {
        flag: sum(flag in result.flags for result in results)
        for flag in FLAG_ORDER
    }


def _combination_counts(results: Sequence[DiagnosticResult]) -> dict[str, int]:
    return dict(
        sorted(Counter(_combination_label(result.flags) for result in results).items())
    )


def _mapping_overlap_counts(results: Sequence[DiagnosticResult]) -> dict[str, int]:
    labels = (
        "neither_mapping_class",
        "segment_mapping_flags_only",
        "terminal_mapping_flags_only",
        "both_segment_and_terminal_mapping_flags",
    )
    observed = Counter(_mapping_overlap_label(result) for result in results)
    return {label: observed[label] for label in labels}


def _policy_counts(results: Sequence[DiagnosticResult]) -> dict[str, int]:
    return {
        policy: sum(_eligible(result, policy) for result in results)
        for policy in POLICY_DISQUALIFYING_FLAGS
    }


def build_summary(results: Sequence[DiagnosticResult]) -> dict[str, Any]:
    if len(results) != core.EXPECTED_FILES:
        raise core.ConstructionError(
            f"overlap summary requires {core.EXPECTED_FILES} development rows"
        )
    by_group: dict[str, Any] = {}
    for group in sorted({result.entry.brain_id for result in results}):
        group_results = [
            result for result in results if result.entry.brain_id == group
        ]
        eligible = _policy_counts(group_results)
        by_group[group] = {
            "input_cells": len(group_results),
            "marginal_flag_counts": _marginal_counts(group_results),
            "exact_flag_combination_counts": _combination_counts(group_results),
            "mapping_class_overlap_counts": _mapping_overlap_counts(group_results),
            "eligible_cells_by_policy": eligible,
            "ineligible_cells_by_policy": {
                policy: len(group_results) - count
                for policy, count in eligible.items()
            },
        }
    eligible = _policy_counts(results)
    policy_definitions = {}
    for policy, disqualifying in POLICY_DISQUALIFYING_FLAGS.items():
        policy_definitions[policy] = {
            "disqualifying_flags": [
                flag for flag in FLAG_ORDER if flag in disqualifying
            ],
            "tolerated_flags": [
                flag for flag in FLAG_ORDER if flag not in disqualifying
            ],
        }
    return {
        "status": "complete",
        "stage": "provider_native_v2_structural_QC_overlap_diagnostic",
        "role": "development",
        "input_files": len(results),
        "input_groups": len(by_group),
        "swc_interpretation": core.V2_INTERPRETATION,
        "provider_axon_edge_rule": (
            "both endpoint raw types are outside {3, 4}"
        ),
        "provider_terminal_rule": (
            "topology leaf whose raw type is outside {3, 4}"
        ),
        "flag_definitions": FLAG_DEFINITIONS,
        "policy_definitions": policy_definitions,
        "marginal_flag_counts": _marginal_counts(results),
        "exact_flag_combination_counts": _combination_counts(results),
        "mapping_class_overlap_counts": _mapping_overlap_counts(results),
        "eligible_cells_by_policy": eligible,
        "ineligible_cells_by_policy": {
            policy: len(results) - count for policy, count in eligible.items()
        },
        "by_group": by_group,
        "post_parse_flags_evaluated_together_without_early_exit": True,
        "parser_errors_prevent_downstream_flag_evaluation": True,
        "named_target_map_entries": 0,
        "target_vocabulary_argument_supported": False,
        "target_vocabulary_loaded": False,
        "candidate_target_ids_computed_or_emitted": False,
        "per_target_mapping_computed_or_emitted": False,
        "K_computed_or_emitted": False,
        "projection_outcomes_computed_or_emitted": False,
        "final_or_unassigned_files_opened": False,
        "EP11_shared_units_opened": False,
    }


def write_outputs(output_root: Path, results: Sequence[DiagnosticResult]) -> None:
    # The contract forbids pre-candidate per-cell raw lengths and terminal
    # counts.  Keep all per-cell measurements in memory and persist only the
    # requested aggregate overlap table.
    core._atomic_json(
        output_root / "aggregate_overlap_summary.json", build_summary(results)
    )


def _synthetic_assets() -> tuple[
    core.GeometryConfig, core.SyntheticAtlas, core.Ontology, core.StructureMapper
]:
    config = core.GeometryConfig((10.0, 10.0, 10.0), (5, 2, 3), 15.0)
    values = {
        (x, y, z): 985
        for x in range(config.shape[0])
        for y in range(config.shape[1])
        for z in range(config.shape[2])
    }
    atlas = core.SyntheticAtlas(config.shape, values)
    ontology = core._fixture_ontology()
    mapper = core.StructureMapper(ontology, {}, config.grey_structure_id)
    return config, atlas, ontology, mapper


def _synthetic_result(
    swc: str,
    atlas: core.Atlas,
    config: core.GeometryConfig,
    ontology: core.Ontology,
    mapper: core.StructureMapper,
    entry: core.ManifestEntry | None = None,
) -> DiagnosticResult:
    nodes, _ = core.parse_swc_lines(
        swc.splitlines(), interpretation=core.V2_INTERPRETATION
    )
    return diagnose_cell(
        entry or core._fixture_entry(), nodes, atlas, config, ontology, mapper
    )


def run_self_test() -> dict[str, Any]:
    config, atlas, ontology, mapper = _synthetic_assets()
    clean = _synthetic_result(
        """1 1 5 5 5 1 -1
2 10 15 5 5 1 1
3 2 25 5 5 1 2
""",
        atlas,
        config,
        ontology,
        mapper,
    )
    if clean.flags or not all(
        _eligible(clean, policy) for policy in POLICY_DISQUALIFYING_FLAGS
    ):
        raise AssertionError("clean provider-native cell failed a diagnostic policy")

    zero_values = {
        (x, y, z): 985
        for x in range(config.shape[0])
        for y in range(config.shape[1])
        for z in range(config.shape[2])
    }
    zero_values[(1, 0, 0)] = 0
    zero_atlas = core.SyntheticAtlas(config.shape, zero_values)
    zero_overlap = _synthetic_result(
        """1 1 5 5 5 1 -1
2 10 15 5 5 1 1
""",
        zero_atlas,
        config,
        ontology,
        mapper,
    )
    expected_zero = {
        "provider_axon_positive_length_annotation_zero",
        "provider_terminal_annotation_zero",
    }
    if zero_overlap.flags != expected_zero:
        raise AssertionError(
            f"segment/terminal zero overlap changed: {sorted(zero_overlap.flags)}"
        )
    if _eligible(zero_overlap, "strict_current"):
        raise AssertionError("strict policy tolerated segment/terminal zero")
    if _eligible(zero_overlap, "tolerate_segment_outside_or_zero_only"):
        raise AssertionError("segment-only tolerance also tolerated terminal zero")
    if _eligible(
        zero_overlap, "tolerate_terminal_nongrey_outside_or_zero_only"
    ):
        raise AssertionError("terminal-only tolerance also tolerated segment zero")
    if not _eligible(zero_overlap, "tolerate_both_mapping_classes"):
        raise AssertionError("combined mapping tolerance did not recover overlap cell")

    nongrey_values = {
        (x, y, z): 985
        for x in range(config.shape[0])
        for y in range(config.shape[1])
        for z in range(config.shape[2])
    }
    nongrey_values[(1, 0, 0)] = 1009
    nongrey_atlas = core.SyntheticAtlas(config.shape, nongrey_values)
    terminal_only = _synthetic_result(
        """1 1 5 5 5 1 -1
2 10 15 5 5 1 1
""",
        nongrey_atlas,
        config,
        ontology,
        mapper,
    )
    if terminal_only.flags != {"provider_terminal_nongrey"}:
        raise AssertionError(
            f"terminal-only mapping flag changed: {sorted(terminal_only.flags)}"
        )
    if not _eligible(
        terminal_only, "tolerate_terminal_nongrey_outside_or_zero_only"
    ):
        raise AssertionError("terminal tolerance failed to recover nongrey terminal")

    no_axon = _synthetic_result(
        """1 1 5 5 5 1 -1
2 3 15 5 5 1 1
""",
        atlas,
        config,
        ontology,
        mapper,
    )
    if "no_positive_provider_axon_edge" not in no_axon.flags:
        raise AssertionError("no-positive-provider-axon flag was not recorded")
    if any(_eligible(no_axon, policy) for policy in POLICY_DISQUALIFYING_FLAGS):
        raise AssertionError("a mapping tolerance relaxed the positive-axon check")

    try:
        core.parse_swc_lines(
            ["1 1 5 5 5 1 -1", "2 99 15 5 5 1 1"],
            interpretation=core.V2_INTERPRETATION,
        )
    except core.ConstructionError as error:
        parser_result = _parser_error_result(core._fixture_entry(), error)
    else:
        raise AssertionError("malformed provider-native tree passed parsing")
    if any(
        _eligible(parser_result, policy) for policy in POLICY_DISQUALIFYING_FLAGS
    ):
        raise AssertionError("a mapping tolerance relaxed parser integrity")

    return {
        "status": "pass",
        "uses_real_atlas_or_swc": False,
        "checks": [
            "empty_named_target_map",
            "provider_native_custom_type_traversal",
            "all_clean_policies",
            "segment_and_terminal_annotation_zero_overlap",
            "terminal_nongrey_only",
            "mapping_tolerances_retain_positive_axon_check",
            "mapping_tolerances_retain_parser_check",
        ],
    }


def run_real(args: argparse.Namespace) -> dict[str, Any]:
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
        raise core.ConstructionError("target coordinates reached overlap diagnostic")
    mapper = core.StructureMapper(ontology, {}, config.grey_structure_id)

    entries = core.load_manifest(args.manifest)
    swc_root = core._validate_scratch_path(args.swc_root, "development SWC root")
    output_root = core._validate_scratch_path(args.output_root, "output root")
    if (
        swc_root == output_root
        or swc_root in output_root.parents
        or output_root in swc_root.parents
    ):
        raise core.ConstructionError(
            "SWC input and diagnostic output roots must be disjoint"
        )
    swc_paths = core.validate_exact_swc_set(entries, swc_root)
    if output_root.exists() and not output_root.is_dir():
        raise core.ConstructionError(f"output root is not a directory: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise core.ConstructionError(
            "overlap output root must be absent or empty; refusing stale outputs"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink():
        raise core.ConstructionError(f"refusing symlink output root: {output_root}")

    results: list[DiagnosticResult] = []
    for entry in entries:
        try:
            nodes, _ = core.parse_swc(
                swc_paths[entry.provider_neuron_id],
                interpretation=core.V2_INTERPRETATION,
            )
        except (core.ConstructionError, OSError, UnicodeError) as error:
            result = _parser_error_result(entry, error)
        else:
            result = diagnose_cell(entry, nodes, atlas, config, ontology, mapper)
        results.append(result)
    if len(results) != core.EXPECTED_FILES:
        raise core.ConstructionError(
            "diagnostic did not emit one result per frozen manifest row"
        )
    write_outputs(output_root, results)
    summary = build_summary(results)
    print(
        json.dumps(
            {
                "status": "complete",
                "development_files": len(results),
                "development_groups": len(
                    {result.entry.brain_id for result in results}
                ),
                "eligible_cells_by_policy": summary["eligible_cells_by_policy"],
                "named_target_map_entries": 0,
                "target_vocabulary_loaded": False,
                "projection_outcomes_computed_or_emitted": False,
                "output_root": str(output_root),
                "final_or_unassigned_files_opened": False,
            },
            sort_keys=True,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run dependency-free synthetic tests without real-data access",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly permit opening the frozen development SWCs",
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--revision-contract", type=Path)
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
        parser.error("real structural-overlap diagnosis requires --execute")
    required = (
        "contract",
        "revision_contract",
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
