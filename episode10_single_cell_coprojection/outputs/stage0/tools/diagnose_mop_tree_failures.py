#!/usr/bin/env python3
"""Summarize parser failures among frozen EP10 development trees.

This diagnostic reads only rows already classified as
``tree_or_construction_failure``.  It emits parser categories and a few
representative messages; it does not compute or emit projection outcomes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_constructor(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ep10_constructor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load constructor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def category(message: str) -> str:
    transition = re.match(r"SWC transition (-?\d+)->(-?\d+) is not allowed", message)
    if transition:
        return f"disallowed_transition_{transition.group(1)}_to_{transition.group(2)}"
    if "must contain exactly one soma type 1 node" in message:
        return "noncanonical_soma_count_or_root"
    if "must have exactly one root" in message:
        return "noncanonical_root_count"
    if "unique SWC root must be soma type 1" in message:
        return "root_not_soma_type_1"
    if "unsupported node type" in message:
        match = re.search(r"unsupported node type (-?\d+)", message)
        return f"unsupported_node_type_{match.group(1) if match else 'unknown'}"
    if "marker node" in message and "must be a leaf" in message:
        return "marker_node_not_leaf"
    if "missing parent" in message or "refers to missing parent" in message:
        return "missing_parent"
    if "cycle" in message:
        return "cycle"
    if "fewer than seven columns" in message:
        return "too_few_columns"
    if "duplicate node ID" in message:
        return "duplicate_node_id"
    if "coordinates/radius" in message or "radius must" in message:
        return "invalid_geometry_field"
    return "other_parse_failure"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constructor", type=Path, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--swc-root", type=Path, required=True)
    args = parser.parse_args()

    constructor = load_constructor(args.constructor.resolve())
    with args.qc.open(newline="", encoding="utf-8") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row["global_eligibility_failure"] == "tree_or_construction_failure"
        ]

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    parse_successes: list[dict[str, str]] = []
    for row in selected:
        path = args.swc_root / row["fMOST_brain_id"] / row["file"]
        try:
            constructor.parse_swc(path)
        except (constructor.ConstructionError, OSError, UnicodeError) as error:
            reason = category(str(error))
            counts[reason] += 1
            if len(examples[reason]) < 3:
                examples[reason].append(
                    {
                        "file": row["file"],
                        "provider_neuron_id": row["provider_neuron_id"],
                        "message": str(error),
                    }
                )
        else:
            parse_successes.append(
                {
                    "file": row["file"],
                    "provider_neuron_id": row["provider_neuron_id"],
                }
            )

    if sum(counts.values()) + len(parse_successes) != len(selected):
        raise RuntimeError("diagnostic did not account for every selected row")
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_tree_or_construction_failures": len(selected),
                "parse_failure_counts": dict(sorted(counts.items())),
                "parse_failure_examples": dict(sorted(examples.items())),
                "parse_succeeded_but_construction_failed": parse_successes,
                "projection_outcomes_emitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
