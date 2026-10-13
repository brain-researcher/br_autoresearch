#!/usr/bin/env python3
"""Apply adopted EP16 geometry/numerical rules to permitted structural metadata.

Results remain conditional: no task-size adjudication, roles, neural data,
packet selection, support-driven threshold changes, or pair pruning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from describe_schedule_design import EPS, components, contrasts, geometry
from describe_structural_support import array_labels
from mat5_structural import field_vector


def geometry_with_spacing(points):
    result = geometry(points)
    if result is None:
        return None
    distances = pdist(result["points"])
    if not len(distances) or not np.isfinite(distances).all() or np.any(distances <= 0):
        return None
    result.update(minimum_spacing=float(distances.min()), median_spacing=float(np.median(distances)))
    result["near_duplicate_distinctness_unresolved"] = result["minimum_spacing"] <= result["tolerance"]
    return result


def match_geometry(left, right, config):
    if left is None or right is None:
        return {"state": "unresolved_degenerate_layout", "mapping": None}
    if left["near_duplicate_distinctness_unresolved"] or right["near_duplicate_distinctness_unresolved"]:
        return {"state": "unresolved_near_duplicate_distinctness", "mapping": None}
    if len(left["points"]) != len(right["points"]):
        return {"state": "different_observed_target_cardinality", "mapping": None}
    delta = min(config["median_spacing_fraction"] * min(left["median_spacing"], right["median_spacing"]),
                config["minimum_spacing_fraction"] * min(left["minimum_spacing"], right["minimum_spacing"]))
    distances, mapping = cKDTree(right["points"]).query(left["points"], k=1)
    error = float(distances.max())
    bijective = len(set(mapping.tolist())) == len(mapping)
    state = "compatible" if bijective and error < delta else (
        "unresolved_exact_tolerance_boundary" if bijective and error == delta else "outside_declared_geometry_rule")
    return {"state": state, "mapping": mapping.tolist() if state == "compatible" else None,
            "maximum_matched_error": error, "delta": delta, "bijective": bijective}


def compatibility_components(nodes, compatible_edges):
    edge_set = {tuple(sorted(edge)) for edge in compatible_edges}
    return [{"nodes": group, "complete_clique": all((a, b) in edge_set
             for position, a in enumerate(group) for b in group[position + 1:])}
            for group in components(nodes, sorted(edge_set))]


def contrast_information(weighted_nuisance, indicator, weights, edges):
    u0, singular0, _ = np.linalg.svd(weighted_nuisance, full_matrices=False)
    cutoff0 = max(weighted_nuisance.shape) * EPS * singular0[0]
    rank0 = int(np.sum(singular0 > cutoff0))
    weighted_indicator = np.sqrt(weights) * indicator
    residual = weighted_indicator - u0[:, :rank0] @ (u0[:, :rank0].T @ weighted_indicator)
    energy = float(residual @ residual)
    centered_energy = float(np.sum(weights * (indicator - np.sum(weights * indicator)) ** 2))
    node_energy = defaultdict(float)
    for squared, (left, right, _) in zip(residual * residual, edges):
        node_energy[left] += float(squared)
        node_energy[right] += float(squared)
    shares = {node: value / energy for node, value in node_energy.items()} if energy > 0 else {}
    return {"nuisance_rank": rank0, "residual_L2": float(np.sqrt(energy)),
            "retained_energy_fraction": energy / centered_energy if centered_energy > 0 else None,
            "node_shares": shares, "maximum_node_share": max(shares.values()) if shares else None,
            "node_share_sum": sum(shares.values()) if shares else None}


def support_gates(full_rank, condition, information, both_lags, config):
    ratio, maximum = information["retained_energy_fraction"], information["maximum_node_share"]
    return {"both_lag_classes": bool(both_lags), "full_constrained_rank": bool(full_rank),
        "condition_strictly_below_maximum": condition is not None and condition < config["condition_number_maximum"],
        "residual_L2_strictly_above_minimum": information["residual_L2"] > config["residual_L2_minimum"],
        "retained_energy_strictly_above_minimum": ratio is not None and ratio > config["retained_energy_minimum"],
        "node_share_strictly_below_maximum": maximum is not None and maximum < config["maximum_node_share"]}


def design_description(sessions, edges, config):
    sources = sorted({a for a, _, _ in edges})
    targets = sorted({b for _, b, _ in edges})
    source_index, target_index = {v: k for k, v in enumerate(sources)}, {v: k for k, v in enumerate(targets)}
    source_basis, target_basis = contrasts(len(sources)), contrasts(len(targets))
    target_counts = Counter(b for _, b, _ in edges)
    weights = np.array([1 / (len(targets) * target_counts[b]) for _, b, _ in edges])
    nuisance = np.column_stack((np.ones(len(edges)),
        source_basis[[source_index[a] for a, _, _ in edges]],
        target_basis[[target_index[b] for _, b, _ in edges]]))
    z = np.array([lag == "long" for _, _, lag in edges], dtype=float)
    weighted_nuisance = nuisance * np.sqrt(weights)[:, None]
    information = contrast_information(weighted_nuisance, z, weights, edges)
    design = np.column_stack((weighted_nuisance, np.sqrt(weights) * z))
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > max(design.shape) * EPS * singular[0]))
    full = rank == design.shape[1]
    column_norms = np.linalg.norm(design, axis=0)
    condition = None
    if full and np.all(column_norms > 0):
        normalized_singular = np.linalg.svd(design / column_norms, compute_uv=False)
        condition = float(normalized_singular[0] / normalized_singular[-1])
    hat = np.sum(u[:, :rank] ** 2, axis=1)
    hat_node = defaultdict(float)
    for value, (a, b, _) in zip(hat, edges):
        hat_node[a] += float(value)
        hat_node[b] += float(value)
    lag_counts = Counter(lag for _, _, lag in edges)
    gates = support_gates(full, condition, information, lag_counts["near"] and lag_counts["long"], config)
    result = {"session_days": [sessions[i]["day"] for i in sorted(hat_node)],
        "pairs": len(edges), "near_pairs": lag_counts["near"], "long_pairs": lag_counts["long"],
        "source_sessions": len(sources), "target_sessions": len(targets), "design_columns": design.shape[1],
        "design_rank": rank, "weighted_column_normalized_condition": condition,
        "residual_L2": information["residual_L2"], "retained_energy_fraction": information["retained_energy_fraction"],
        "maximum_contrast_node_share": information["maximum_node_share"], "node_share_sum": information["node_share_sum"],
        "contrast_node_shares_by_day": {str(sessions[i]["day"]): value for i, value in sorted(information["node_shares"].items())},
        "descriptive_maximum_full_hat_node_mass_over_rank": max(hat_node.values()) / rank,
        "gates": gates, "conditional_numerical_pass": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pair_pruning_performed": False, "eligibility_qualified": False,
        "interpretation": "conditional_on_pending_block_transition_size_and_exposure_qualification"}
    return result


def prepare(record):
    fields, summary = record["fields"], record["summary"]
    shape = geometry_with_spacing(fields["trial_targets"])
    blocks = field_vector(fields["trials"], "block_id").astype(int)
    starts = field_vector(fields["trials"], "trial_start_index").astype(int)
    if len(blocks) != len(starts) or len(blocks) != len(fields["trial_targets"]):
        raise ValueError("Structural trial vectors differ in length")
    target_sets, edge_support = [], set()
    if shape is not None:
        for block in sorted(set(blocks.tolist())):
            rows = np.flatnonzero(blocks == block)
            rows = rows[np.argsort(starts[rows], kind="stable")]
            sequence = shape["inverse"][rows].tolist()
            target_sets.append(tuple(sorted(set(sequence))))
            edge_support.update(zip(sequence[:-1], sequence[1:]))
    return {"participant": summary["participant"], "day": summary["post_implant_day"],
        "arrays": array_labels(fields["electrodes"]), "geometry": shape,
        "whole_block_capacity": summary["whole_blocks_can_supply_two_64_trial_sides"],
        "trials": len(blocks), "block_target_sets": target_sets, "transition_support": edge_support,
        "all_nonempty_blocks_cover_session_targets": shape is not None and bool(target_sets)
            and all(len(s) == len(shape["points"]) for s in target_sets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with args.structural_jsonl.open() as source:
        sessions = [prepare(json.loads(line)) for line in source if line.strip()]
    sessions.sort(key=lambda s: (s["participant"], s["day"]))
    if len({(s["participant"], s["day"]) for s in sessions}) != len(sessions):
        raise ValueError("Duplicate participant/day identity")
    strata = defaultdict(list)
    for index, session in enumerate(sessions):
        strata[(session["participant"], session["arrays"])].append(index)
    group_reports, total_designs, numerical_passes = [], 0, 0
    with (args.output_dir / "geometry_pair_diagnostics.jsonl").open("x") as diagnostics:
        for (participant, arrays), indices in sorted(strata.items()):
            compatible, mappings = [], {}
            for position, a in enumerate(indices):
                for b in indices[position + 1:]:
                    match = match_geometry(sessions[a]["geometry"], sessions[b]["geometry"], config["geometry"])
                    if match["state"] == "compatible":
                        compatible.append((a, b))
                        mappings[(a, b)] = match["mapping"]
                    diagnostics.write(json.dumps({"participant": participant, "source_day": sessions[a]["day"],
                        "target_day": sessions[b]["day"], **match}, allow_nan=False) + "\n")
            for description in compatibility_components(indices, compatible):
                group = description["nodes"]
                clique = description["complete_clique"]
                capable = [i for i in group if sessions[i]["whole_block_capacity"]]
                edges = []
                for position, a in enumerate(capable):
                    for b in capable[position + 1:]:
                        gap = sessions[b]["day"] - sessions[a]["day"]
                        if (a, b) in mappings and (1 <= gap <= 30 or gap >= 180):
                            edges.append((a, b, "near" if gap <= 30 else "long"))
                designs = []
                # No rank calculations on a greedily chosen subclique of an ambiguous group.
                if clique:
                    for connected in components(capable, [(a, b) for a, b, _ in edges]):
                        nodes = set(connected)
                        selected = [edge for edge in edges if edge[0] in nodes and edge[1] in nodes]
                        if selected:
                            designs.append(design_description(sessions, selected, config["component_support"]))
                transition_differences = []
                if clique and len(group) > 1:
                    reference = group[0]
                    for member in group[1:]:
                        inverse = {dst: src for src, dst in enumerate(mappings[(reference, member)])}
                        mapped = {(inverse[a], inverse[b]) for a, b in sessions[member]["transition_support"]}
                        if mapped != sessions[reference]["transition_support"]:
                            transition_differences.append(sessions[member]["day"])
                unresolved_geometry = any(sessions[i]["geometry"] is None or
                    sessions[i]["geometry"]["near_duplicate_distinctness_unresolved"] for i in group)
                group_reports.append({"group_id": f"geometry_{len(group_reports)+1:04d}", "participant": participant,
                    "physical_array_set": arrays, "session_days": [sessions[i]["day"] for i in group],
                    "complete_pairwise_clique": clique,
                    "geometry_status": "unresolved_nontransitive_component" if not clique else
                        "unresolved_target_distinctness_or_degenerate" if unresolved_geometry else "geometry_rule_pass",
                    "all_nonempty_blocks_cover_session_target_sets": all(sessions[i]["all_nonempty_blocks_cover_session_targets"] for i in group),
                    "observed_transition_support_status": "unresolved_nonclique" if not clique else
                        "observed_support_difference_sampling_vs_protocol_unresolved" if transition_differences else
                        "identical_observed_edge_support_not_verified_task_identity",
                    "transition_support_difference_days_relative_to_first": transition_differences,
                    "size_dependence": "unresolved_no_specific_size_disposition",
                    "conditional_designs": designs, "qualified_schedule_group": False,
                    "support_failure_removed_from_group": False})
                total_designs += len(designs)
                numerical_passes += sum(item["conditional_numerical_pass"] for item in designs)
            print(json.dumps({"participant": participant, "conditional_geometry_stage_complete": True}), flush=True)
    session_reports = []
    for session in sessions:
        shape = session["geometry"]
        session_reports.append({"participant": session["participant"], "day": session["day"], "trials": session["trials"],
            "exact_target_count": len(shape["points"]) if shape else 0,
            "minimum_normalized_spacing": shape["minimum_spacing"] if shape else None,
            "median_normalized_pairwise_spacing": shape["median_spacing"] if shape else None,
            "near_duplicate_distinctness_unresolved": shape["near_duplicate_distinctness_unresolved"] if shape else None,
            "whole_block_capacity": session["whole_block_capacity"],
            "all_nonempty_blocks_cover_session_targets": session["all_nonempty_blocks_cover_session_targets"]})
    report = {"analysis_id": "structural_support", "phase": "conditional_adopted_numerical_support",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "elapsed_seconds": time.monotonic()-started,
        "sessions": len(sessions), "retained_declared_trials": sum(s["trials"] for s in sessions),
        "groups": len(group_reports), "conditional_designs": total_designs,
        "conditional_numerical_passes": numerical_passes,
        "nontransitive_components_unresolved": sum(not g["complete_pairwise_clique"] for g in group_reports),
        "source_structural_view": str(args.structural_jsonl), "applied_config_path": str(args.config),
        "applied_config": config, "geometry_groups": group_reports, "session_descriptions": session_reports,
        "complete": True, "scientific_numerical_rules_applied_conditionally": True,
        "size_transition_eligibility_qualified": False, "roles_assigned": False, "trials_selected": False,
        "neural_cursor_impedance_values_decoded": False, "raw_source_files_opened": False,
        "pair_pruning_performed": False, "thresholds_changed_after_application": False}
    (args.output_dir / "conditional_support_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items()
        if key not in {"applied_config", "geometry_groups", "session_descriptions"}}), flush=True)


if __name__ == "__main__":
    main()
