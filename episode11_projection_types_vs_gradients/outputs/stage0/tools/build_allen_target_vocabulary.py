#!/usr/bin/env python3
"""Build EP11's outcome-blind Allen CCFv3 target vocabulary.

This tool reads only the Allen structure graph.  It never reads an annotation
voxel or an SWC, so it cannot reveal a real projection outcome.  The named
targets are a fixed, pairwise-disjoint set of coarse grey-matter ancestors.
Nodes below ``grey`` not claimed by those anchors are counted as QC-only and
never become an outcome coordinate or enter its normalization denominator.
Laterality is expanded relative to each neuron's soma hemisphere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


PINNED_ONTOLOGY_SHA256 = (
    "b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386"
)
ROOT_ID = 997
GREY_ID = 8
LATERALITIES = ("ipsilateral", "contralateral")


@dataclass(frozen=True)
class Anchor:
    structure_id: int
    acronym: str
    name: str


# Isocortex is resolved into its 17 direct, layer-collapsed areas.  The rest of
# grey matter is kept at broad divisions chosen without looking at axons.
ANCHORS = (
    Anchor(184, "FRP", "Frontal pole, cerebral cortex"),
    Anchor(500, "MO", "Somatomotor areas"),
    Anchor(453, "SS", "Somatosensory areas"),
    Anchor(1057, "GU", "Gustatory areas"),
    Anchor(677, "VISC", "Visceral area"),
    Anchor(247, "AUD", "Auditory areas"),
    Anchor(669, "VIS", "Visual areas"),
    Anchor(31, "ACA", "Anterior cingulate area"),
    Anchor(972, "PL", "Prelimbic area"),
    Anchor(44, "ILA", "Infralimbic area"),
    Anchor(714, "ORB", "Orbital area"),
    Anchor(95, "AI", "Agranular insular area"),
    Anchor(254, "RSP", "Retrosplenial area"),
    Anchor(22, "PTLp", "Posterior parietal association areas"),
    Anchor(541, "TEa", "Temporal association areas"),
    Anchor(922, "PERI", "Perirhinal area"),
    Anchor(895, "ECT", "Ectorhinal area"),
    Anchor(698, "OLF", "Olfactory areas"),
    Anchor(1089, "HPF", "Hippocampal formation"),
    Anchor(703, "CTXsp", "Cortical subplate"),
    Anchor(477, "STR", "Striatum"),
    Anchor(803, "PAL", "Pallidum"),
    Anchor(549, "TH", "Thalamus"),
    Anchor(1097, "HY", "Hypothalamus"),
    Anchor(313, "MB", "Midbrain"),
    Anchor(771, "P", "Pons"),
    Anchor(354, "MY", "Medulla"),
    Anchor(512, "CB", "Cerebellum"),
)

CSV_FIELDS = (
    "coordinate_index",
    "target_index",
    "coordinate_id",
    "target_id",
    "allen_structure_id",
    "allen_acronym",
    "allen_name",
    "laterality",
    "anchor_graph_order",
    "mapped_ontology_node_count",
    "selection_rule",
    "ontology_sha256",
)


class VocabularyError(RuntimeError):
    """Raised when the pinned ontology or partition contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_tree(root: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    stack: list[tuple[dict[str, Any], int | None]] = [(root, None)]
    while stack:
        node, expected_parent = stack.pop()
        try:
            structure_id = int(node["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise VocabularyError("ontology node lacks an integer id") from error
        if structure_id in nodes:
            raise VocabularyError(f"duplicate ontology structure id {structure_id}")
        observed_parent = node.get("parent_structure_id")
        if observed_parent is not None:
            observed_parent = int(observed_parent)
        if expected_parent != observed_parent:
            raise VocabularyError(
                f"structure {structure_id} parent mismatch: "
                f"{observed_parent} != {expected_parent}"
            )
        nodes[structure_id] = node
        children = node.get("children", [])
        if not isinstance(children, list):
            raise VocabularyError(f"structure {structure_id} children is not a list")
        for child in reversed(children):
            stack.append((child, structure_id))
    return nodes


def ancestor_ids(
    structure_id: int, nodes: dict[int, dict[str, Any]]
) -> tuple[int, ...]:
    lineage: list[int] = []
    seen: set[int] = set()
    current: int | None = structure_id
    while current is not None:
        if current in seen:
            raise VocabularyError(f"cycle in ontology at structure {current}")
        seen.add(current)
        lineage.append(current)
        parent = nodes[current].get("parent_structure_id")
        current = None if parent is None else int(parent)
    return tuple(lineage)


def compile_vocabulary(
    nodes: dict[int, dict[str, Any]],
    anchors: Iterable[Anchor],
    ontology_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    anchor_list = tuple(anchors)
    anchor_ids = {anchor.structure_id for anchor in anchor_list}
    if len(anchor_ids) != len(anchor_list):
        raise VocabularyError("target anchors contain a duplicate structure id")
    if ROOT_ID not in nodes or GREY_ID not in nodes:
        raise VocabularyError("ontology lacks pinned root or grey structure")

    lineages = {
        structure_id: ancestor_ids(structure_id, nodes) for structure_id in nodes
    }
    for anchor in anchor_list:
        node = nodes.get(anchor.structure_id)
        if node is None:
            raise VocabularyError(f"missing target anchor {anchor.structure_id}")
        if node.get("acronym") != anchor.acronym or node.get("name") != anchor.name:
            raise VocabularyError(
                f"target anchor metadata changed for {anchor.structure_id}"
            )
        if GREY_ID not in lineages[anchor.structure_id]:
            raise VocabularyError(f"target anchor {anchor.acronym} is outside grey")

    for left in anchor_list:
        for right in anchor_list:
            if left == right:
                continue
            if left.structure_id in lineages[right.structure_id]:
                raise VocabularyError(
                    f"overlapping targets: {left.acronym} contains {right.acronym}"
                )

    mapped_counts = {anchor.structure_id: 0 for anchor in anchor_list}
    residual_count = 0
    grey_node_count = 0
    for structure_id, lineage in lineages.items():
        if GREY_ID not in lineage:
            continue
        grey_node_count += 1
        matches = anchor_ids.intersection(lineage)
        if len(matches) > 1:
            raise VocabularyError(
                f"structure {structure_id} maps to multiple target anchors"
            )
        if matches:
            mapped_counts[next(iter(matches))] += 1
        else:
            residual_count += 1

    targets: list[dict[str, object]] = []
    for target_index, anchor in enumerate(anchor_list):
        node = nodes[anchor.structure_id]
        targets.append(
            {
                "target_index": target_index,
                "target_id": f"allen_{anchor.structure_id}_{anchor.acronym}",
                "allen_structure_id": anchor.structure_id,
                "allen_acronym": anchor.acronym,
                "allen_name": anchor.name,
                "anchor_graph_order": int(node["graph_order"]),
                "mapped_ontology_node_count": mapped_counts[anchor.structure_id],
                "selection_rule": "anchor_descendant_or_self",
            }
        )
    rows: list[dict[str, object]] = []
    for target in targets:
        for laterality in LATERALITIES:
            rows.append(
                {
                    "coordinate_index": len(rows),
                    **target,
                    "coordinate_id": f"{target['target_id']}__{laterality}",
                    "laterality": laterality,
                    "ontology_sha256": ontology_sha256,
                }
            )

    summary = {
        "ontology_node_count": len(nodes),
        "grey_ontology_node_count": grey_node_count,
        "named_target_count": len(anchor_list),
        "residual_target_count": 0,
        "target_count": len(targets),
        "coordinate_count": len(rows),
        "named_mapped_grey_node_count": sum(mapped_counts.values()),
        "unclaimed_grey_qc_node_count": residual_count,
        "partitioned_grey_node_count": sum(mapped_counts.values()) + residual_count,
        "ontology_sha256": ontology_sha256,
    }
    if summary["partitioned_grey_node_count"] != grey_node_count:
        raise VocabularyError("grey-matter partition is not exhaustive")
    return rows, summary


def load_ontology(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        roots = payload["msg"]
    except (KeyError, TypeError) as error:
        raise VocabularyError("ontology JSON lacks msg roots") from error
    if not isinstance(roots, list) or len(roots) != 1:
        raise VocabularyError("ontology JSON must contain exactly one root")
    return flatten_tree(roots[0])


def write_csv(rows: list[dict[str, object]], handle: TextIO) -> None:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def self_test() -> None:
    fixture = {
        "id": ROOT_ID,
        "acronym": "root",
        "name": "root",
        "parent_structure_id": None,
        "graph_order": 0,
        "children": [
            {
                "id": GREY_ID,
                "acronym": "grey",
                "name": "grey",
                "parent_structure_id": ROOT_ID,
                "graph_order": 1,
                "children": [
                    {
                        "id": 10,
                        "acronym": "A",
                        "name": "Area A",
                        "parent_structure_id": GREY_ID,
                        "graph_order": 2,
                        "children": [
                            {
                                "id": 11,
                                "acronym": "A1",
                                "name": "Area A child",
                                "parent_structure_id": 10,
                                "graph_order": 3,
                                "children": [],
                            }
                        ],
                    },
                    {
                        "id": 12,
                        "acronym": "R",
                        "name": "Residual area",
                        "parent_structure_id": GREY_ID,
                        "graph_order": 4,
                        "children": [],
                    },
                ],
            },
            {
                "id": 1009,
                "acronym": "fiber tracts",
                "name": "fiber tracts",
                "parent_structure_id": ROOT_ID,
                "graph_order": 5,
                "children": [],
            },
        ],
    }
    nodes = flatten_tree(fixture)
    rows, summary = compile_vocabulary(
        nodes, (Anchor(10, "A", "Area A"),), "synthetic"
    )
    assert summary["target_count"] == 1
    assert summary["coordinate_count"] == 2
    assert summary["grey_ontology_node_count"] == 4
    assert summary["unclaimed_grey_qc_node_count"] == 2
    assert rows[0]["coordinate_id"] == "allen_10_A__ipsilateral"
    assert rows[1]["coordinate_id"] == "allen_10_A__contralateral"
    buffer = io.StringIO()
    write_csv(rows, buffer)
    assert buffer.getvalue().count("\n") == 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--allow-unpinned-ontology", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("synthetic vocabulary self-test: pass", file=sys.stderr)
        if args.ontology_json is None:
            return
    if args.ontology_json is None:
        raise SystemExit("--ontology-json is required unless only --self-test is used")

    observed_hash = sha256_file(args.ontology_json)
    if not args.allow_unpinned_ontology and observed_hash != PINNED_ONTOLOGY_SHA256:
        raise VocabularyError(
            f"ontology SHA-256 mismatch: {observed_hash} != {PINNED_ONTOLOGY_SHA256}"
        )
    rows, summary = compile_vocabulary(
        load_ontology(args.ontology_json), ANCHORS, observed_hash
    )

    if args.output_csv is None:
        write_csv(rows, sys.stdout)
    else:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            write_csv(rows, handle)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
