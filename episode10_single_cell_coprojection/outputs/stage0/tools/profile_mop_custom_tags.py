#!/usr/bin/env python3
"""Profile Gao SWC custom node tags without computing projection outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path


TAG_TYPES = frozenset((10, 11, 12))
AXON_TYPES = frozenset((0, 2))
DENDRITE_TYPES = frozenset((3,))


def compartment(node_type: int) -> str:
    if node_type in AXON_TYPES:
        return "axon"
    if node_type in DENDRITE_TYPES:
        return "dendrite"
    if node_type == 1:
        return "soma"
    return f"other_{node_type}"


def edge_length(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def load_tree(path: Path) -> dict[int, tuple[int, tuple[float, float, float], int]]:
    nodes: dict[int, tuple[int, tuple[float, float, float], int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 7:
                raise RuntimeError(f"{path}: line {line_number} has fewer than seven columns")
            node_id = int(fields[0])
            node_type = int(fields[1])
            point = (float(fields[2]), float(fields[3]), float(fields[4]))
            parent_id = int(fields[6])
            if node_id in nodes:
                raise RuntimeError(f"{path}: duplicate node {node_id}")
            nodes[node_id] = (node_type, point, parent_id)
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--swc-root", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 200:
        raise RuntimeError(f"expected 200 manifest rows, found {len(manifest)}")

    node_types: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    tag_types: Counter[str] = Counter()
    tag_roles: Counter[str] = Counter()
    tag_parent_types: Counter[str] = Counter()
    tag_child_types: Counter[str] = Counter()
    tag_boundary_signatures: Counter[str] = Counter()
    tag_component_resolution: Counter[str] = Counter()
    tag_edge_lengths: Counter[str] = Counter()
    cells: Counter[str] = Counter()
    per_brain = defaultdict(Counter)

    for row in manifest:
        brain = row["fMOST_brain_id"]
        path = args.swc_root / brain / row["file"]
        nodes = load_tree(path)
        per_brain[brain]["total"] += 1
        children: dict[int, list[int]] = defaultdict(list)
        for node_id, (node_type, point, parent_id) in nodes.items():
            node_types[str(node_type)] += 1
            if parent_id == -1:
                continue
            if parent_id not in nodes:
                raise RuntimeError(f"{path}: missing parent {parent_id}")
            children[parent_id].append(node_id)
            parent_type, parent_point, _ = nodes[parent_id]
            transitions[f"{parent_type}->{node_type}"] += 1
            if node_type in TAG_TYPES or parent_type in TAG_TYPES:
                key = "positive" if edge_length(parent_point, point) > 0.0 else "zero"
                tag_edge_lengths[key] += 1

        tags = {node_id for node_id, value in nodes.items() if value[0] in TAG_TYPES}
        if tags:
            cells["with_custom_tags"] += 1
            per_brain[brain]["with_custom_tags"] += 1
        else:
            cells["without_custom_tags"] += 1
            per_brain[brain]["without_custom_tags"] += 1
        all_components_resolvable = True
        for node_id in tags:
            node_type, _, parent_id = nodes[node_id]
            tag_types[str(node_type)] += 1
            outgoing = children.get(node_id, [])
            tag_roles["leaf" if not outgoing else "internal"] += 1
            tag_parent_types[str(nodes[parent_id][0]) if parent_id != -1 else "root"] += 1
            for child_id in outgoing:
                tag_child_types[str(nodes[child_id][0])] += 1

        unseen = set(tags)
        while unseen:
            seed = unseen.pop()
            component_nodes = {seed}
            queue = deque((seed,))
            boundary_types: list[int] = []
            while queue:
                current = queue.popleft()
                _, _, parent_id = nodes[current]
                neighbors = list(children.get(current, []))
                if parent_id != -1:
                    neighbors.append(parent_id)
                for neighbor in neighbors:
                    neighbor_type = nodes[neighbor][0]
                    if neighbor_type in TAG_TYPES:
                        if neighbor not in component_nodes:
                            component_nodes.add(neighbor)
                            unseen.discard(neighbor)
                            queue.append(neighbor)
                    else:
                        boundary_types.append(neighbor_type)
            boundary_classes = sorted({compartment(value) for value in boundary_types})
            signature = "+".join(boundary_classes) if boundary_classes else "none"
            tag_boundary_signatures[signature] += 1
            neurite = {value for value in boundary_classes if value in {"axon", "dendrite"}}
            unsupported = {
                value for value in boundary_classes if value not in {"axon", "dendrite", "soma"}
            }
            if len(neurite) == 1 and not unsupported:
                resolution = next(iter(neurite))
            elif len(neurite) > 1:
                resolution = "mixed_axon_dendrite"
                all_components_resolvable = False
            else:
                resolution = "ambiguous"
                all_components_resolvable = False
            tag_component_resolution[resolution] += 1
        if tags and all_components_resolvable:
            cells["all_tag_components_topologically_resolvable"] += 1
            per_brain[brain]["all_tag_components_topologically_resolvable"] += 1
        elif tags:
            cells["has_ambiguous_or_mixed_tag_component"] += 1
            per_brain[brain]["has_ambiguous_or_mixed_tag_component"] += 1

    print(
        json.dumps(
            {
                "status": "complete",
                "files": len(manifest),
                "groups": len({row["fMOST_brain_id"] for row in manifest}),
                "projection_outcomes_computed": False,
                "node_type_counts": dict(sorted(node_types.items(), key=lambda item: int(item[0]))),
                "edge_transition_counts": dict(sorted(transitions.items())),
                "custom_tag_type_counts": dict(sorted(tag_types.items())),
                "custom_tag_roles": dict(sorted(tag_roles.items())),
                "custom_tag_parent_type_counts": dict(sorted(tag_parent_types.items())),
                "custom_tag_child_type_counts": dict(sorted(tag_child_types.items())),
                "custom_tag_boundary_signatures": dict(sorted(tag_boundary_signatures.items())),
                "custom_tag_component_resolution": dict(sorted(tag_component_resolution.items())),
                "custom_tag_edge_length_counts": dict(sorted(tag_edge_lengths.items())),
                "cell_counts": dict(sorted(cells.items())),
                "per_brain_cell_counts": {
                    brain: dict(sorted(counts.items()))
                    for brain, counts in sorted(per_brain.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
