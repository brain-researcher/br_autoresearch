#!/usr/bin/env python3
"""Describe outcome-blind EP16 geometry matches and conditional design rank.

Reads only the already-authorized structural JSONL. No source archives, cursor,
neural values, trial selection, participant roles, or eligibility decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial import cKDTree

from describe_structural_support import array_labels
from mat5_structural import field_vector


EPS = np.finfo(np.float64).eps


def geometry(points):
    """Exact raw deduplication; translation and positive isotropic scale only."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Invalid permitted target coordinates")
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    center = np.array([math.fsum(unique[:, axis].tolist()) / len(unique) for axis in range(2)])
    centered = unique - center
    radius = math.sqrt(math.fsum((centered * centered).reshape(-1).tolist()) / len(unique))
    if not np.isfinite(radius) or radius <= 0:
        return None
    # Units cancel. This is a roundoff-scale descriptor, not a formal error
    # bound, source quantization allowance or tolerance for target movement.
    tolerance = float(64 * EPS * max(1.0, float(np.max(np.abs(unique))) / radius))
    normalized = centered / radius
    return {"points": normalized, "inverse": inverse, "tolerance": tolerance,
            "center": center.tolist(), "rms_radius": radius}


def correspondence(left, right):
    if left is None or right is None or len(left["points"]) != len(right["points"]):
        return None
    tolerance = left["tolerance"] + right["tolerance"]
    distances, indices = cKDTree(right["points"]).query(left["points"], k=1)
    if np.max(distances) > tolerance or len(set(indices.tolist())) != len(indices):
        return None
    return indices


def components(nodes, edges):
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result = []
    while adjacency:
        pending, seen = [min(adjacency)], set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(adjacency[node] - seen)
        result.append(sorted(seen))
        for node in seen:
            del adjacency[node]
    return result


def contrasts(count):
    """Orthonormal Helmert columns, each exactly sum-zero algebraically."""
    result = np.zeros((count, max(0, count - 1)))
    for column in range(count - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        result[:column + 1, column] = 1 / scale
        result[column + 1, column] = -(column + 1) / scale
    return result


def design_description(sessions, edges):
    if not edges:
        return {"pairs": 0, "rank_evaluated": False}
    sources = sorted({left for left, _, _ in edges})
    targets = sorted({right for _, right, _ in edges})
    source_index = {value: index for index, value in enumerate(sources)}
    target_index = {value: index for index, value in enumerate(targets)}
    source_basis, target_basis = contrasts(len(sources)), contrasts(len(targets))
    target_counts = Counter(right for _, right, _ in edges)
    weights = np.array([1 / (len(targets) * target_counts[right]) for _, right, _ in edges])
    nuisance = np.concatenate((np.ones((len(edges), 1)),
        source_basis[[source_index[left] for left, _, _ in edges]],
        target_basis[[target_index[right] for _, right, _ in edges]]), axis=1)
    indicator = np.array([gap == "long" for _, _, gap in edges], dtype=float)
    root_weight = np.sqrt(weights)
    weighted_nuisance = nuisance * root_weight[:, None]
    u0, singular0, _ = np.linalg.svd(weighted_nuisance, full_matrices=False)
    cutoff0 = max(weighted_nuisance.shape) * EPS * singular0[0]
    rank0 = int(np.sum(singular0 > cutoff0))
    weighted_indicator = indicator * root_weight
    residual = weighted_indicator - u0[:, :rank0] @ (u0[:, :rank0].T @ weighted_indicator)
    design = np.column_stack((weighted_nuisance, weighted_indicator))
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    cutoff = max(design.shape) * EPS * singular[0]
    rank = int(np.sum(singular > cutoff))
    full = rank == design.shape[1]
    leverage = np.sum(u[:, :rank] ** 2, axis=1)
    node_leverage = defaultdict(float)
    for value, (left, right, _) in zip(leverage, edges):
        node_leverage[left] += float(value)
        node_leverage[right] += float(value)
    lag_counts = Counter(gap for _, _, gap in edges)
    return {"pairs": len(edges), "near_pairs": lag_counts["near"], "long_pairs": lag_counts["long"],
        "session_days": [sessions[index]["day"] for index in sorted(node_leverage)],
        "source_sessions": len(sources), "target_sessions": len(targets),
        "nuisance_columns": nuisance.shape[1], "nuisance_rank": rank0,
        "design_columns": design.shape[1], "design_rank": rank,
        "full_constrained_design_rank": full, "svd_rank_cutoff": float(cutoff),
        "weighted_design_condition_number": float(singular[0] / singular[-1]) if full else None,
        "residualized_long_indicator_weighted_l2": float(np.linalg.norm(residual)),
        "long_indicator_adds_rank": rank > rank0,
        "maximum_row_hat_leverage": float(leverage.max()),
        "maximum_session_incident_hat_leverage_sum": max(node_leverage.values()),
        "maximum_session_incident_hat_leverage_fraction_of_rank": max(node_leverage.values()) / rank,
        "rank_evaluated": True, "scientific_condition_information_leverage_thresholds_applied": False}


def prepare(record):
    fields, summary = record["fields"], record["summary"]
    shape = geometry(fields["trial_targets"])
    blocks = field_vector(fields["trials"], "block_id").astype(int)
    starts = field_vector(fields["trials"], "trial_start_index").astype(int)
    observations = []
    if shape is not None:
        for block in sorted(set(blocks.tolist())):
            positions = np.flatnonzero(blocks == block)
            positions = positions[np.argsort(starts[positions], kind="stable")]
            sequence = shape["inverse"][positions].tolist()
            observations.append({"block_id": block, "trials": len(sequence),
                "observed_targets": len(set(sequence)),
                "covers_session_coordinate_set": len(set(sequence)) == len(shape["points"]),
                "observed_directed_transition_types": len(set(zip(sequence[:-1], sequence[1:])))})
    return {"participant": summary["participant"], "day": summary["post_implant_day"],
        "arrays": array_labels(fields["electrodes"]), "geometry": shape,
        "whole_block_capacity": summary["whole_blocks_can_supply_two_64_trial_sides"],
        "blocks": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    sessions = []
    with args.structural_jsonl.open() as source:
        for line in source:
            sessions.append(prepare(json.loads(line)))
    sessions.sort(key=lambda row: (row["participant"], row["day"]))
    strata = defaultdict(list)
    for index, session in enumerate(sessions):
        strata[(session["participant"], session["arrays"])].append(index)
    groups, summaries = [], []
    for (participant, arrays), indices in sorted(strata.items()):
        compatible = set()
        for position, left in enumerate(indices):
            for right in indices[position + 1:]:
                if correspondence(sessions[left]["geometry"], sessions[right]["geometry"]) is not None:
                    compatible.add((left, right))
        for group in components(indices, compatible):
            complete = all((left, right) in compatible for position, left in enumerate(group)
                           for right in group[position + 1:])
            capable = [index for index in group if sessions[index]["whole_block_capacity"]]
            edges = []
            for position, left in enumerate(capable):
                for right in capable[position + 1:]:
                    if (left, right) not in compatible:
                        continue
                    gap = sessions[right]["day"] - sessions[left]["day"]
                    if 1 <= gap <= 30 or gap >= 180:
                        edges.append((left, right, "near" if gap <= 30 else "long"))
            designs = []
            for connected in components(capable, [(left, right) for left, right, _ in edges]):
                node_set = set(connected)
                selected = [edge for edge in edges if edge[0] in node_set and edge[1] in node_set]
                if selected:
                    designs.append(design_description(sessions, selected))
            entry = {"participant": participant, "physical_array_set": arrays,
                "group_id": f"geometry_{len(groups) + 1:04d}",
                "session_days": [sessions[index]["day"] for index in group],
                "coordinate_count": len(sessions[group[0]]["geometry"]["points"]) if sessions[group[0]]["geometry"] else 0,
                "pairwise_roundoff_compatibility_is_clique": complete,
                "no_transitive_chaining_eligibility_asserted": True,
                "whole_block_capacity_sessions": len(capable),
                "size_dependence": "unresolved_no_task_size_metadata",
                "transition_pattern_compatibility": "unresolved_observed_missing_edges_are_not_proven_conflicts",
                "geometry_only_conditional_designs": designs,
                "qualified_schedule_group": False}
            groups.append(entry)
        print(json.dumps({"participant": participant, "geometry_stage_completed": True}), flush=True)
    for session in sessions:
        shape = session.pop("geometry")
        session.update({"coordinate_count": len(shape["points"]) if shape else 0,
            "normalization_center": shape["center"] if shape else None,
            "normalization_rms_radius": shape["rms_radius"] if shape else None,
            "normalized_roundoff_tolerance": shape["tolerance"] if shape else None})
        summaries.append(session)
    report = {"analysis_id": "structural_support", "phase": "geometry_conditional_rank",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "elapsed_seconds": time.monotonic() - begin,
        "sessions": len(sessions), "groups": len(groups),
        "source_structural_view": str(args.structural_jsonl),
        "geometry_groups": groups, "session_structure": summaries,
        "roundoff_formula": "64*float64_epsilon*max(1,max(abs(raw_unique_xy))/RMS_radius); pair threshold sums the two tolerances",
        "roundoff_interpretation": "arithmetic_scale_descriptor_not_formal_error_bound_or_source_quantization_allowance",
        "centroid_rms_accumulation": "math.fsum",
        "transformations": ["translation", "positive_isotropic_scale"],
        "rotation_reflection_axiswise_scale_permitted": False,
        "weights": "each target total1/n_targets, shared equally over its chronological sources",
        "rank_tolerance": "max(n_rows,n_columns)*float64_epsilon*largest_singular_value",
        "structural_support_qualified": False, "roles_assigned": False,
        "trials_selected": False, "neural_cursor_impedance_values_decoded": False,
        "required_remaining": ["documented_size_dependence_rule", "stable_layout_and_transition_rule",
            "scientific_condition_information_leverage_limits", "participant_outcome_exposure_review"]}
    (args.output_dir / "schedule_design_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("phase", "sessions", "groups", "elapsed_seconds", "structural_support_qualified", "roles_assigned")}), flush=True)


if __name__ == "__main__":
    main()
