#!/usr/bin/env python3
"""Build EP03 task families, conditional lineage groups and folds from metadata.

Reads only three frozen metadata files and the selected methods specification.
It never imports an image reader, reads a path from a manifest, or opens a map.
This is preparation, not baseline access or a claim of authenticated independence.
Run the command and its tests in a coordinator-owned Slurm allocation.
"""

import argparse
import csv
import json
import os
import socket
from collections import Counter, defaultdict
from pathlib import Path


def taxonomy_lookup(spec, routes):
    assignments = {}
    for family, entry in spec["taxonomy"]["families"].items():
        for route in entry["routes"]:
            if route in assignments:
                raise ValueError(f"duplicate family assignment: {route}")
            assignments[route] = {"family": family, "rationale": entry["rationale"],
                                  "assumption": entry["assumption"]}
    for route, reason in spec["taxonomy"]["unresolved_routes"].items():
        if route in assignments:
            raise ValueError(f"both assigned and unresolved: {route}")
        assignments[route] = {"family": None, "rationale": reason, "assumption": None}
    actual = {r["route"] for r in routes}
    if len(actual) != len(routes):
        raise ValueError("duplicate route metadata")
    if actual != set(assignments):
        raise ValueError(f"taxonomy coverage mismatch: missing={sorted(actual-set(assignments))}; "
                         f"extra={sorted(set(assignments)-actual)}")
    return assignments


def lineage_groups(accessions, edges):
    """Merge only explicit overlap evidence, never equal local subject labels."""
    parent = {a: a for a in accessions}

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for edge in edges:
        a, b = edge["accessions"]
        if a not in parent or b not in parent or a == b:
            raise ValueError("overlap edge must name two different known accessions")
        if edge.get("relation") not in {"shared_participants", "shared_first_level_inputs",
                                         "shared_derivative_lineage"} or not edge.get("evidence"):
            raise ValueError("overlap requires a permitted relation and explicit source evidence")
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    members = defaultdict(list)
    for a in sorted(accessions):
        members[root(a)].append(a)
    groups = {a: "group:" + "+".join(members[root(a)]) for a in sorted(accessions)}
    return groups


def map_key(row):
    return "|".join((row["route"], row["session"], row["contrast"]))


def metadata_exclusions(row, family):
    reasons = []
    if family is None:
        reasons.append("family_unresolved")
    if row["s"] not in {"1", "-1"} or row["status"] in {"ineligible", "unresolved"}:
        reasons.append("adopted_orientation_ineligible")
    flags = row["flags"].split(";")
    if "duplicate_definition" in flags:
        reasons.append("duplicate_definition")
    if any(f.startswith("regressor_not_convolved:") for f in flags):
        reasons.append("never_convolved_regressor")
    if int(row["n_inputs"]) <= 2:
        reasons.append("fitted_n_at_most_2")
    if row["dataset_id"] == "ds003545":
        reasons.append("adopted_onset_timing_hold")
    return sorted(set(reasons))


def split_for_groups(rows, held_groups):
    """Return matched comparison support; unsupported maps remain explicit."""
    held = set(held_groups)
    eligible = [r for r in rows if not r["metadata_exclusions"]]
    train = [r for r in eligible if r["group"] not in held]
    test = [r for r in eligible if r["group"] in held]
    represented = {r["family"] for r in train}
    score = [r["map_id"] for r in test if r["family"] in represented]
    unsupported = [r["map_id"] for r in test if r["family"] not in represented]
    return {"train_map_ids": [r["map_id"] for r in train],
            "test_map_ids": [r["map_id"] for r in test],
            "matched_score_map_ids": score,
            "unrepresented_family_map_ids": unsupported}


def prepare(spec, route_rows, orientation_rows, roles):
    assignments = taxonomy_lookup(spec, route_rows)
    routes = {r["route"]: r for r in route_rows}
    accessions = {r["dataset_id"] for r in route_rows}
    groups = lineage_groups(accessions, spec["lineage"]["merge_edges"])
    seen, rows = set(), []
    for row in orientation_rows:
        route = row["route"]
        if route not in routes or row["dataset_id"] != routes[route]["dataset_id"]:
            raise ValueError("orientation/route accession mismatch")
        key = map_key(row)
        if key in seen:
            raise ValueError(f"duplicate map identifier: {key}")
        seen.add(key)
        assignment = assignments[route]
        exclusions = metadata_exclusions(row, assignment["family"])
        n = int(row["n_inputs"])
        # Completed, authorized lineage result; never rewrite the frozen census.
        if route == "ds003545/task-compL1":
            n = 20
        role_key = route if route in roles["routes"] else (
            "_mid_template" if row["task"].lower() in {"mid", "midmb4", "midmb8", "midsingle"}
            else None)
        role = roles["routes"].get(role_key, {})
        rows.append({"map_id": key, "route": route, "accession": row["dataset_id"],
                     "session": row["session"], "contrast": row["contrast"],
                     "family": assignment["family"], "group": groups[row["dataset_id"]],
                     "lineage_status": "cross_accession_nonoverlap_assumed",
                     "role": "historically_exposed_development_only",
                     "legacy_partition_status": row["partition_status"],
                     "polarity": int(row["s"]) if row["s"] in {"-1", "1"} else None,
                     "frozen_orientation_status": row["status"], "fitted_n": n,
                     "fitted_n_evidence": "experiment_log: 2026-10-17 jobs 48221634 and 48245679",
                     "condition_role_key": role_key,
                     "condition_factor_names": sorted(role.get("factors", {})),
                     "taxonomy_rationale": assignment["rationale"],
                     "taxonomy_assumption": assignment["assumption"],
                     "metadata_exclusions": exclusions,
                     "target_eligibility": "not_established_by_metadata_preparation"})
    counts = Counter(r["route"] for r in rows)
    for route, metadata in routes.items():
        if counts[route] != int(metadata["dataset_z_maps"]):
            raise ValueError(f"map count disagreement for {route}")
    rows.sort(key=lambda r: r["map_id"])
    eligible_groups = sorted({r["group"] for r in rows if not r["metadata_exclusions"]})
    folds = []
    for held in eligible_groups:
        split = split_for_groups(rows, [held])
        train_groups = [g for g in eligible_groups if g != held]
        # Inner folds are specified by identity, not by a random contrast split.
        # For each inner validation group call split_for_groups on OUTER TRAINING
        # rows only. Family representation is recomputed inside that inner fold.
        folds.append({"outer_validation_group": held, "outer_train_groups": train_groups,
                      "inner_validation_groups": train_groups, **split})
    support = defaultdict(set)
    for row in rows:
        if not row["metadata_exclusions"]:
            support[row["family"]].add(row["group"])
    return {
        "analysis_id": "grouped_evaluator_qualification", "evidence": "methods_metadata_only",
        "lineage": spec["lineage"],
        "source_routes": len(routes), "source_accessions": len(accessions), "source_maps": len(rows),
        "family_group_support": {f: sorted(gs) for f, gs in sorted(support.items())},
        "groups_with_any_outer_comparison": sum(bool(f["matched_score_map_ids"]) for f in folds),
        "unresolved_routes": spec["taxonomy"]["unresolved_routes"],
        "maps": rows, "outer_folds": folds,
        "normalization": spec["normalization"],
        "decisions_not_adopted": spec["decisions_not_adopted"],
        "baseline_target_access": False, "scientific_scoring": False,
        "remaining_dependencies": ["negative_control_and_inference_threshold_decision",
            "fixed_common_mask_and_space_binding", "remaining_source_eligibility",
            "methods_only_redacted_text_and_registered_model_specifications",
            "meaningful_evaluator_checks_on_compute"]}


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("outputs/config/methods_preparation.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run this metadata build in the coordinator's verified Slurm allocation.")
    episode = Path(__file__).resolve().parents[2]
    spec = json.loads(args.spec.read_text())
    fixed = episode / "outputs/runs/source_handoff_qualification/20261016-orientation-proposal"
    allowed = {"routes": fixed / "route_manifest.tsv", "orientation": fixed / "orientation_by_map.tsv",
               "roles": fixed / "condition_roles.json"}
    # Resolve exact local metadata paths; do not follow arbitrary embedded paths.
    for key, path in allowed.items():
        if (episode / spec["inputs"][key]).resolve() != path.resolve():
            raise ValueError(f"unexpected input path for {key}")
    result = prepare(spec, read_tsv(allowed["routes"]), read_tsv(allowed["orientation"]),
                     json.loads(allowed["roles"].read_text()))
    destination = args.output.resolve()
    if not destination.is_relative_to(episode / "outputs/runs/grouped_evaluator_qualification"):
        raise ValueError("output must be in the new grouped_evaluator_qualification run tree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("source_routes", "source_accessions", "source_maps",
        "groups_with_any_outer_comparison", "baseline_target_access", "scientific_scoring")}))


if __name__ == "__main__":
    main()
