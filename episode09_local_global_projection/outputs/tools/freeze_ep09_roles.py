#!/usr/bin/env python3
"""Freeze EP09 animal roles using metadata only."""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import socket
import stat
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

SEED = 20261017
AUDIT_N = 16
RULE_VERSION = "ep09_roles_v1_2026-10-17"

UNCLAIMED = "ep09_eligible_unclaimed"
ELIGIBLE = {
    UNCLAIMED,
    "ep09_eligible_shared_with_EP10_unassigned",
    "ep09_development_only_EP10_outcomes_opened",
}
TARGET_LAYERS = {
    "Cux2": {"L2/3", "L4"},
    "Wfs1": {"L2/3"},
}
LAYER_TOKENS = {
    "1": "L1",
    "2": "L2/3",
    "3": "L2/3",
    "2/3": "L2/3",
    "4": "L4",
    "5": "L5",
    "6a": "L6",
    "6b": "L6",
    "6": "L6",
}

LEDGER_FIELDS = (
    "fmost_id",
    "transgenic_line",
    "ep09_line_class",
    "ep09_largest_core_pool",
    "ep09_core_pool_ge5",
    "ep09_role_eligibility",
)
QC_FIELDS = ("cell", "brain", "line", "acronym", "passes")
ROLE_FIELDS = (
    "fmost_id",
    "line",
    "pool_class",
    "stratum",
    "ep09_largest_core_pool",
    "ep09_role_eligibility",
    "ep09_role",
    "audit_seed",
    "rule_version",
)
CELL_FIELDS = (
    "cell",
    "brain",
    "line",
    "area",
    "layer",
    "pool_key",
    "pool_size",
    "ep09_role",
)

ROLE_FILE = "ep09_role_manifest.csv"
CELL_FILE = "ep09_cell_manifest.csv"
SUMMARY_FILE = "ep09_role_summary.json"
OUTPUT_FILES = (ROLE_FILE, CELL_FILE, SUMMARY_FILE)


class FreezeError(ValueError):
    pass


def read_snapshot(path):
    """Hash exactly the bytes subsequently parsed."""
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def csv_rows(raw, path, required_fields):
    """Yield only the requested fields; preserve their string values."""
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames
        if header is None:
            raise FreezeError(f"{path}: missing CSV header")

        for field in required_fields:
            if header.count(field) != 1:
                raise FreezeError(
                    f"{path}: required column {field!r} must appear exactly once"
                )

        for row in reader:
            missing = [field for field in required_fields if row[field] is None]
            if missing:
                raise FreezeError(
                    f"{path}:{reader.line_num}: missing values for {missing}"
                )
            yield reader.line_num, {
                field: row[field] for field in required_fields
            }


def sorted_brains(brains):
    brains = list(brains)
    if all(brain.isdigit() for brain in brains):
        return sorted(brains, key=int)
    return sorted(brains)


def load_animals(path):
    raw, digest = read_snapshot(path)
    animals = {}
    seen = set()

    for row_number, row in csv_rows(raw, path, LEDGER_FIELDS):
        brain = row["fmost_id"]
        if not brain:
            raise FreezeError(f"{path}:{row_number}: empty fmost_id")
        if brain in seen:
            raise FreezeError(
                f"{path}:{row_number}: duplicate fmost_id {brain!r}"
            )
        seen.add(brain)

        if not (
            row["ep09_line_class"] == "core"
            and row["ep09_core_pool_ge5"] == "yes"
            and row["ep09_role_eligibility"] in ELIGIBLE
        ):
            continue

        transgenic_line = row["transgenic_line"].lower()
        if "cux2" in transgenic_line:
            line = "Cux2"
        elif "wfs1" in transgenic_line:
            line = "Wfs1"
        else:
            raise FreezeError(
                f"{path}:{row_number}: cannot derive a core line "
                f"for usable brain {brain!r}"
            )

        try:
            largest_pool = int(row["ep09_largest_core_pool"])
        except ValueError as exc:
            raise FreezeError(
                f"{path}:{row_number}: invalid ep09_largest_core_pool"
            ) from exc

        pool_class = "pool_10plus" if largest_pool >= 10 else "pool_5_9"
        animals[brain] = {
            "fmost_id": brain,
            "line": line,
            "pool_class": pool_class,
            "stratum": f"{line}|{pool_class}",
            "ep09_largest_core_pool": largest_pool,
            "ep09_role_eligibility": row["ep09_role_eligibility"],
            "ep09_role": "development",
            "audit_seed": SEED,
            "rule_version": RULE_VERSION,
        }

    return animals, digest


def draw_audit(animals):
    candidates = defaultdict(list)
    for brain, animal in animals.items():
        if animal["ep09_role_eligibility"] == UNCLAIMED:
            candidates[animal["stratum"]].append(brain)

    total_candidates = sum(len(brains) for brains in candidates.values())
    if total_candidates < AUDIT_N:
        raise FreezeError(
            f"Need {AUDIT_N} audit candidates; found {total_candidates}"
        )

    strata = sorted(candidates)
    allocation = {}
    remainder_groups = defaultdict(list)

    for stratum in strata:
        n = len(candidates[stratum])
        numerator = AUDIT_N * n
        base, remainder = divmod(numerator, total_candidates)
        allocation[stratum] = {
            "n": n,
            "quota": numerator / total_candidates,
            "base": base,
            "extra": 0,
            "drawn": 0,
        }
        # A common denominator makes integer remainders exact.
        remainder_groups[remainder].append(stratum)

    leftover = AUDIT_N - sum(item["base"] for item in allocation.values())
    if not 0 <= leftover <= len(strata):
        raise FreezeError("Invalid largest-remainder allocation")

    # This is the only Random instance in the script.
    rng = random.Random(SEED)

    # Rank every remainder group before any within-stratum selection.
    # Shuffle tied names in descending remainder order, including ties
    # whose strata ultimately receive no extra slot.
    remainder_order = []
    for remainder in sorted(remainder_groups, reverse=True):
        tied_names = sorted(remainder_groups[remainder])
        if len(tied_names) > 1:
            rng.shuffle(tied_names)
        remainder_order.extend(tied_names)

    for stratum in remainder_order[:leftover]:
        allocation[stratum]["extra"] = 1

    audit_brains = set()
    for stratum in strata:
        brains = sorted_brains(candidates[stratum])
        requested = (
            allocation[stratum]["base"] + allocation[stratum]["extra"]
        )
        if requested > len(brains):
            raise FreezeError(
                f"Stratum {stratum!r} needs {requested} brains "
                f"but contains {len(brains)}"
            )

        rng.shuffle(brains)
        selected = brains[:requested]
        allocation[stratum]["drawn"] = len(selected)
        audit_brains.update(selected)

    if (
        len(audit_brains) != AUDIT_N
        or sum(item["drawn"] for item in allocation.values()) != AUDIT_N
    ):
        raise FreezeError(f"Audit draw did not produce exactly {AUDIT_N} brains")

    for brain, animal in animals.items():
        animal["ep09_role"] = (
            "audit" if brain in audit_brains else "development"
        )

    return allocation


def load_structure_graph(path):
    raw, digest = read_snapshot(path)
    document = json.loads(raw.decode("utf-8-sig"))

    if not isinstance(document, dict):
        raise FreezeError(f"{path}: expected a JSON object")
    roots = document.get("msg")
    if not isinstance(roots, list) or not roots:
        raise FreezeError(f"{path}: 'msg' must be a nonempty list")

    required = {
        "id", "acronym", "name", "parent_structure_id", "children"
    }
    seen_ids = set()
    locations = {}

    # Entries carry the nested parent and whether any ancestor is 315.
    stack = [(root, None, None, False) for root in reversed(roots)]

    while stack:
        node, parent_id, parent_acronym, ancestor_isocortex = stack.pop()
        if not isinstance(node, dict) or not required.issubset(node):
            raise FreezeError(f"{path}: malformed structure node")

        node_id = node["id"]
        acronym = node["acronym"]
        name = node["name"]
        declared_parent = node["parent_structure_id"]
        children = node["children"]

        if type(node_id) is not int:
            raise FreezeError(f"{path}: structure IDs must be integers")
        if declared_parent is not None and type(declared_parent) is not int:
            raise FreezeError(f"{path}: invalid parent_structure_id")
        if not isinstance(acronym, str) or not acronym:
            raise FreezeError(f"{path}: invalid structure acronym")
        if not isinstance(name, str) or not isinstance(children, list):
            raise FreezeError(f"{path}: invalid name or children for {acronym!r}")
        if node_id in seen_ids or acronym in locations:
            raise FreezeError(
                f"{path}: duplicate structure ID or acronym at {acronym!r}"
            )
        if parent_id is not None and declared_parent != parent_id:
            raise FreezeError(
                f"{path}: parent_structure_id disagrees with nesting "
                f"for {acronym!r}"
            )

        seen_ids.add(node_id)
        in_isocortex = ancestor_isocortex or node_id == 315
        location = None

        if in_isocortex and "ayer" in name:
            token = name.rsplit("ayer", 1)[1].strip()
            if token not in LAYER_TOKENS:
                raise FreezeError(
                    f"{path}: unsupported isocortex layer token "
                    f"{token!r} for {acronym!r}"
                )
            if parent_acronym is None:
                raise FreezeError(
                    f"{path}: layer node {acronym!r} has no available parent"
                )
            location = (parent_acronym, LAYER_TOKENS[token])

        # None distinguishes a known non-layer node from an unknown acronym.
        locations[acronym] = location
        for child in reversed(children):
            stack.append((child, node_id, acronym, in_isocortex))

    return locations, digest


def load_cells(path, animals, locations):
    raw, digest = read_snapshot(path)
    cells = []
    seen = set()

    for row_number, row in csv_rows(raw, path, QC_FIELDS):
        if row["passes"] != "True":
            continue

        brain = row["brain"]
        animal = animals.get(brain)
        if animal is None:
            continue

        if row["line"] != animal["line"]:
            raise FreezeError(
                f"{path}:{row_number}: QC line {row['line']!r} disagrees "
                f"with ledger line {animal['line']!r} for brain {brain!r}"
            )

        cell = row["cell"]
        identity = (brain, cell)
        if not cell or identity in seen:
            raise FreezeError(
                f"{path}:{row_number}: empty or duplicate passing cell "
                f"{identity!r}"
            )
        seen.add(identity)

        acronym = row["acronym"]
        if acronym not in locations:
            raise FreezeError(
                f"{path}:{row_number}: unknown soma acronym {acronym!r}"
            )

        location = locations[acronym]
        if location is None:
            continue
        area, layer = location
        line = animal["line"]
        if layer not in TARGET_LAYERS[line]:
            continue

        cells.append({
            "cell": cell,
            "brain": brain,
            "line": line,
            "area": area,
            "layer": layer,
            "pool_key": f"{brain}|{line}|{area}|{layer}",
            "ep09_role": animal["ep09_role"],
        })

    pool_sizes = Counter(cell["pool_key"] for cell in cells)
    for cell in cells:
        cell["pool_size"] = pool_sizes[cell["pool_key"]]

    cells.sort(key=lambda cell: (cell["brain"], cell["cell"]))
    return cells, digest


def summarize_counts(animals, cells):
    animal_counts = Counter(
        (animal["ep09_role"], animal["line"], animal["pool_class"])
        for animal in animals.values()
    )
    cell_counts = Counter(
        (
            cell["ep09_role"],
            cell["line"],
            animals[cell["brain"]]["pool_class"],
        )
        for cell in cells
    )
    animals_per_role = Counter(
        animal["ep09_role"] for animal in animals.values()
    )
    cells_per_role = Counter(cell["ep09_role"] for cell in cells)

    roles = ("development", "audit")
    counts = []
    for role in roles:
        for line in sorted(TARGET_LAYERS):
            for pool_class in ("pool_5_9", "pool_10plus"):
                key = (role, line, pool_class)
                counts.append({
                    "ep09_role": role,
                    "line": line,
                    "pool_class": pool_class,
                    "animals": animal_counts[key],
                    "cells": cell_counts[key],
                })

    return {
        "counts_by_role_line_pool_class": counts,
        "animals_per_role": {
            role: animals_per_role[role] for role in roles
        },
        "cells_per_role": {
            role: cells_per_role[role] for role in roles
        },
    }


def refuse_existing_outputs(out_dir):
    existing = [
        str(out_dir / filename)
        for filename in OUTPUT_FILES
        if os.path.lexists(out_dir / filename)
    ]
    if existing:
        raise FreezeError(
            "Refusing to overwrite existing output paths: "
            + ", ".join(existing)
        )


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_outputs(out_dir, role_rows, cells, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    refuse_existing_outputs(out_dir)
    created = []

    try:
        with ExitStack() as stack:
            handles = {}

            # Reserve every output exclusively. Existing files, directories,
            # and symlinks cannot be overwritten, including during races.
            for filename in OUTPUT_FILES:
                path = out_dir / filename
                handle = path.open("x", encoding="utf-8", newline="")
                created.append(path)
                handles[filename] = stack.enter_context(handle)

            for filename, fields, rows in (
                (ROLE_FILE, ROLE_FIELDS, role_rows),
                (CELL_FILE, CELL_FIELDS, cells),
            ):
                handle = handles[filename]
                writer = csv.DictWriter(
                    handle, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

            # Both CSVs have been written and closed before hashing.
            summary["output_sha256"] = {
                ROLE_FILE: file_sha256(out_dir / ROLE_FILE),
                CELL_FILE: file_sha256(out_dir / CELL_FILE),
            }
            summary_text = json.dumps(
                summary, indent=2, sort_keys=True, allow_nan=False
            )
            handle = handles[SUMMARY_FILE]
            handle.write(summary_text + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        # All three files are complete and closed before permissions change.
        for path in created:
            permissions = stat.S_IMODE(path.stat().st_mode)
            os.chmod(path, permissions & ~0o222)

    except BaseException:
        # Roll back only files created by this invocation.
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(
                    f"Could not remove incomplete output {path}: {exc}",
                    file=sys.stderr,
                )
        raise

    return summary_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("LEDGER_CSV", type=Path)
    parser.add_argument("EP09_QC_CSV", type=Path)
    parser.add_argument("STRUCTURE_GRAPH_JSON", type=Path)
    parser.add_argument("OUT_DIR", type=Path)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or not job_id.strip():
        raise FreezeError("SLURM_JOB_ID must be set and nonempty")

    refuse_existing_outputs(args.OUT_DIR)

    # Animal roles depend exclusively on ledger metadata.
    animals, ledger_hash = load_animals(args.LEDGER_CSV)
    allocation = draw_audit(animals)

    locations, graph_hash = load_structure_graph(args.STRUCTURE_GRAPH_JSON)
    cells, qc_hash = load_cells(args.EP09_QC_CSV, animals, locations)

    role_rows = [
        animals[brain] for brain in sorted_brains(animals)
    ]
    summary = {
        "SLURM_JOB_ID": job_id,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "SEED": SEED,
        "AUDIT_N": AUDIT_N,
        "rule_version": RULE_VERSION,
        "input_paths": {
            "LEDGER_CSV": os.path.abspath(args.LEDGER_CSV),
            "EP09_QC_CSV": os.path.abspath(args.EP09_QC_CSV),
            "STRUCTURE_GRAPH_JSON": os.path.abspath(
                args.STRUCTURE_GRAPH_JSON
            ),
        },
        "input_sha256": {
            "LEDGER_CSV": ledger_hash,
            "EP09_QC_CSV": qc_hash,
            "STRUCTURE_GRAPH_JSON": graph_hash,
        },
        "allocation_per_stratum": allocation,
        "projection_outcomes_opened": False,
        **summarize_counts(animals, cells),
    }

    summary_text = write_outputs(
        args.OUT_DIR, role_rows, cells, summary
    )
    print(summary_text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("freeze_ep09_roles.py: interrupted", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"freeze_ep09_roles.py: error: {exc}", file=sys.stderr)
        sys.exit(1)
