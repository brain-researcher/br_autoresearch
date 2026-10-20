#!/usr/bin/env python3
"""Build a brain-level exposure ledger from metadata and role tables only."""

import argparse
import csv
import hashlib
import io
import json
import os
import socket
from collections import Counter
from pathlib import Path

CORE = {"Cux2": {"L2/3", "L4"}, "Wfs1": {"L2/3"}}
TARGETS = {**CORE, "Scnn1a": {"L4"}}
EXCLUDED = {"PlexinD1", "Tlx3", "Rorb"}
LINES = set(TARGETS) | EXCLUDED
LAYER_MAP = {
    "1": "L1", "2": "L2/3", "3": "L2/3", "2/3": "L2/3",
    "4": "L4", "5": "L5", "6": "L6", "6a": "L6", "6b": "L6",
}

# Fixed exposure facts supplied by the episode records.
EP10_OPENED = {
    "development": "yes", "final": "no", "unassigned": "no", "none": "no",
}
EP11_OPENED = "no"
EP09_OPENED = "no"

INPUTS = (
    "sample_json", "ep09_qc_csv", "structure_graph_json",
    "ep10_roles_csv", "ep10_summary_json", "ep11_inventory_csv",
)
FIELDS = (
    "fmost_id sample_id project_id transgenic_line ep09_line_class "
    "ep09_qc_pass_cells_in_target_layers ep09_largest_core_pool "
    "ep09_core_pool_ge5 ep10_role ep10_projection_outcomes_opened "
    "ep10_cross_episode_disposition ep11_inventory_member ep11_reserved "
    "ep11_projection_outcomes_opened ep09_projection_outcomes_opened "
    "ep09_role_eligibility"
).split()


def normalize_id(value):
    if value is None:
        raise ValueError("An identifier is null.")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text:
        raise ValueError("An identifier is empty.")
    return text


def soft_id(value):
    """Descriptive identifiers may be blank; they are not join keys."""
    text = "" if value is None else str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def metadata_line(value):
    text = str(value).casefold()
    matches = [line for line in LINES if line.casefold() in text]
    if len(matches) > 1:
        raise ValueError("A transgenic_line contains multiple canonical lines.")
    return matches[0] if matches else None


def build_layer_lookup(graph):
    lookup = {}

    def walk(node, ancestors, parent_acronym):
        path = ancestors + (normalize_id(node["id"]),)
        name = node["name"].casefold()
        if "315" in path and "ayer" in name:
            token = name.rsplit("ayer", 1)[1].strip()
            if token not in LAYER_MAP or parent_acronym is None:
                raise ValueError("An isocortex layer node cannot be resolved.")
            value = (parent_acronym, LAYER_MAP[token])
            acronym = node["acronym"]
            if acronym in lookup and lookup[acronym] != value:
                raise ValueError("Conflicting structure acronyms.")
            lookup[acronym] = value
        for child in node["children"]:
            walk(child, path, node["acronym"])

    for root in graph["msg"]:
        walk(root, (), None)
    return lookup


def eligibility(line_class, role, disposition, ep11_reserved):
    if line_class != "core":
        return "not_in_ep09_core"
    if role == "final":
        return "excluded_reserved_for_EP10_final"
    if ep11_reserved:
        return "excluded_reserved_for_EP11"
    if role == "development":
        return "ep09_development_only_EP10_outcomes_opened"
    if role == "unassigned" and disposition == "EP10_only":
        return "ep09_eligible_shared_with_EP10_unassigned"
    return "ep09_eligible_unclaimed"


def main():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.strip():
        raise SystemExit("Refusing to run: SLURM_JOB_ID must be set.")

    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument(name)
    parser.add_argument("out_dir")
    args = parser.parse_args()

    data, hashes = {}, {}
    for name in INPUTS:
        path = Path(getattr(args, name))
        raw = path.read_bytes()
        hashes[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        # Hash exactly the bytes being parsed.
        data[name] = (
            list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
            if name.endswith("_csv") else json.loads(raw)
        )

    samples, brain_lines = {}, {}
    for item in data["sample_json"]:
        brain = normalize_id(item["fMOST_id"])
        record = {
            "fmost_id": brain,
            "sample_id": soft_id(item.get("sample_id")),
            "project_id": soft_id(item.get("project_id")),
            "transgenic_line": item["transgenic_line"],
        }
        if brain in samples and samples[brain] != record:
            raise ValueError("Conflicting metadata for a normalized brain ID.")
        samples[brain] = record
        brain_lines[brain] = metadata_line(item["transgenic_line"])

    def known_brain(value):
        brain = normalize_id(value)
        if brain not in samples:
            raise ValueError("An input references a brain absent from SAMPLE_JSON.")
        return brain

    ep10 = {}
    for row in data["ep10_roles_csv"]:
        brain = known_brain(row["fMOST_brain_id"])
        role = row["episode10_role"]
        disposition = row["cross_episode_disposition"]
        if role not in {"development", "final", "unassigned"}:
            raise ValueError("Invalid EP10 role.")
        if disposition not in {"EP10_only", "reserved_unassigned_for_EP11"}:
            raise ValueError("Invalid EP10 cross-episode disposition.")
        value = (role, disposition)
        if brain in ep10 and ep10[brain] != value:
            raise ValueError("Conflicting EP10 assignments for a brain.")
        ep10[brain] = value

    inventory = {
        known_brain(row["fMOST_brain_id"])
        for row in data["ep11_inventory_csv"]
    }
    shared = {
        known_brain(value)
        for value in data["ep10_summary_json"]["shared_EP11_fMOST_ids"]
    }

    layers = build_layer_lookup(data["structure_graph_json"])
    target_counts, pool_counts, largest_core = Counter(), Counter(), Counter()
    seen_cells = set()

    for row in data["ep09_qc_csv"]:
        brain = known_brain(row["brain"])
        line = row["line"]
        if line not in LINES:
            raise ValueError("Invalid canonical EP09 QC line.")
        if brain_lines[brain] is None:
            brain_lines[brain] = line
        elif brain_lines[brain] != line:
            raise ValueError("Conflicting line assignments for a brain.")

        cell_key = (brain, normalize_id(row["cell"]))
        if cell_key in seen_cells:
            raise ValueError("Duplicate brain/cell entry in EP09 QC.")
        seen_cells.add(cell_key)

        if row["passes"] not in {"True", "False"}:
            raise ValueError("EP09 passes must be exactly True or False.")
        location = layers.get(row["acronym"])
        if row["passes"] != "True" or line not in TARGETS or location is None:
            continue
        area, layer = location
        if layer not in TARGETS[line]:
            continue

        target_counts[brain] += 1
        pool_key = (brain, line, area, layer)
        pool_counts[pool_key] += 1
        if line in CORE:
            largest_core[brain] = max(
                largest_core[brain], pool_counts[pool_key]
            )

    rows = []
    by_eligibility, ge5_by_eligibility = Counter(), Counter()
    for brain in sorted(samples):
        line = brain_lines[brain]
        line_class = (
            "core" if line in CORE else
            "sensitivity" if line == "Scnn1a" else
            "excluded_line" if line in EXCLUDED else
            "not_it_line"
        )
        role, disposition = ep10.get(brain, ("none", ""))
        reserved = (
            brain in inventory or brain in shared
            or disposition == "reserved_unassigned_for_EP11"
        )
        category = eligibility(line_class, role, disposition, reserved)
        ge5 = largest_core[brain] >= 5
        by_eligibility[category] += 1
        if line_class == "core" and ge5:
            ge5_by_eligibility[category] += 1

        rows.append({
            **samples[brain],
            "ep09_line_class": line_class,
            "ep09_qc_pass_cells_in_target_layers": target_counts[brain],
            "ep09_largest_core_pool": largest_core[brain],
            "ep09_core_pool_ge5": "yes" if ge5 else "no",
            "ep10_role": role,
            "ep10_projection_outcomes_opened": EP10_OPENED[role],
            "ep10_cross_episode_disposition": disposition,
            "ep11_inventory_member": "yes" if brain in inventory else "no",
            "ep11_reserved": "yes" if reserved else "no",
            "ep11_projection_outcomes_opened": EP11_OPENED,
            "ep09_projection_outcomes_opened": EP09_OPENED,
            "ep09_role_eligibility": category,
        })

    unclaimed = ge5_by_eligibility["ep09_eligible_unclaimed"]
    shared_unassigned = ge5_by_eligibility[
        "ep09_eligible_shared_with_EP10_unassigned"
    ]
    summary = {
        "SLURM_JOB_ID": job_id,
        "hostname": socket.gethostname(),
        "input_sha256": hashes,
        "total_brains": len(rows),
        "brains_by_ep09_role_eligibility": dict(sorted(by_eligibility.items())),
        "core_pool_ge5_brains_by_ep09_role_eligibility": dict(
            sorted(ge5_by_eligibility.items())
        ),
        "ep09_audit_available_pool_ge5": unclaimed + shared_unassigned,
        "ep09_audit_available_unclaimed_only_pool_ge5": unclaimed,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gao_exposure_ledger.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    (out_dir / "gao_exposure_ledger_summary.json").write_text(
        rendered + "\n", encoding="utf-8"
    )
    print(rendered)


if __name__ == "__main__":
    main()
