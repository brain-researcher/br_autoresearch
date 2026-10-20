#!/usr/bin/env python3
"""Apply the 2026-10-16 EP16 source-fact assumption rules to the permitted structural view.

Layout-only task-family classification (radial-8, radial-4, grid, Fitts/random,
unclassified), intended-target slots, placeholder go periods, transition and
coverage checks, then the unchanged adopted geometry and component-support
gates. Runs the primary rule set and each prespecified sensitivity variant from
config/source_fact_rules.json. Reads only the existing structural JSONL (one
target point per trial, block/trial indices, electrode identity); no neural,
cursor, success, yield, impedance or decoder values. Assigns no roles.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from apply_conditional_schedule_support import (compatibility_components, design_description,
                                                geometry_with_spacing, match_geometry)
from describe_schedule_design import components
from describe_structural_support import array_labels
from mat5_structural import field_vector


BASE = {"merge": 1e-9, "slot_tolerance": 0.01, "angle_tolerance": 1.0, "placeholder_factor": 1.5,
        "max_ratio": 1.25, "grid_sizes": (5, 6, 7, 8, 9, 10), "grid_min_coverage": 0.5,
        "max_nonpermitted_fraction": 0.05, "families": ("radial_8", "radial_4", "grid"),
        "canonical_center": "ring", "exclude_incomplete_blocks": False, "scale_locked": False,
        "placeholder_blocks_excluded": False, "split_mixed_sessions": False, "packet_coverage": True,
        "strict_group_rule": False, "finite_sampling_check": True, "finite_sampling_outward_trials": 16}
VARIANTS = {
    "P0": {},
    "S1_exclude_incomplete_blocks": {"exclude_incomplete_blocks": True},
    "S2_strict_slot_tolerance": {"slot_tolerance": 0.001, "canonical_center": "stored"},
    "S3_narrow_families": {"families": ("radial_8", "grid")},
    "S4_scale_locked_matching": {"scale_locked": True},
    "S5_placeholder_blocks_excluded": {"placeholder_blocks_excluded": True},
    "S6_strict_transitions": {"max_nonpermitted_fraction": 0.0},
    "S7_session_schedule_combinations": {"split_mixed_sessions": True},
    "S8_without_packet_coverage_condition": {"packet_coverage": False},
    "S9_strict_group_rule": {"strict_group_rule": True},
    "S10_finite_sampling_check_not_applied": {"finite_sampling_check": False},
}
DEVELOPMENT = ("T5", "T10", "T11")
NOT_HOLDABLE = ("T3",)
CANDIDATES = ("T2", "T6", "T7", "T8", "T9")
GAP_NEAR, GAP_LONG = (1, 30), 180


def distinct_targets(points, merge):
    """Merge stored coordinates closer than merge * RMS radius (precision rule P)."""
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    center = unique.mean(axis=0)
    rms = math.sqrt(float(np.mean(np.sum((unique - center) ** 2, axis=1)))) if len(unique) > 1 else 0.0
    parent = list(range(len(unique)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    gap_pairs = 0
    if rms > 0:
        tree = cKDTree(unique)
        for a, b in tree.query_pairs(merge * rms):
            parent[root(a)] = root(b)
        gap_pairs = len(tree.query_pairs(1e-4 * rms)) - len(tree.query_pairs(1e-12 * rms))
    roots = [root(i) for i in range(len(unique))]
    order = {r: k for k, r in enumerate(sorted(set(roots)))}
    labels = np.array([order[r] for r in roots], dtype=int)
    coords = np.array([unique[labels == k].mean(axis=0) for k in range(len(order))])
    return coords, labels[inverse.reshape(-1)], {"stored_unique": len(unique), "distinct": len(coords),
                                                  "merged_pairs": len(unique) - len(coords),
                                                  "pairs_between_1e-12_and_1e-4_rms": int(gap_pairs), "rms": rms}


def angular_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def fit_radial(coords, counts, cfg):
    """Fit center (with its variants) plus 45-degree direction slots; None if the layout is not radial."""
    best = None
    tol, factor = cfg["slot_tolerance"], cfg["placeholder_factor"]
    for ci in range(len(coords)):
        vectors = coords - coords[ci]
        distance = np.hypot(vectors[:, 0], vectors[:, 1])
        others = [i for i in range(len(coords)) if i != ci]
        if not others:
            continue
        r0 = float(np.median(distance[others]))
        if r0 <= 0:
            continue
        center = [ci] + [i for i in others if distance[i] <= tol * r0]
        rest = [i for i in others if i not in center]
        core = [i for i in rest if distance[i] <= factor * cfg["max_ratio"] * r0]
        placeholders = [i for i in rest if i not in core]
        if not core:
            continue
        extent = float(distance[core].max())
        if any(distance[i] <= factor * extent for i in placeholders):
            continue
        radius = float(np.median(distance[core]))
        if any(distance[i] > tol * radius for i in center):
            continue
        slots = defaultdict(list)
        valid = True
        for i in core:
            angle = math.degrees(math.atan2(vectors[i][1], vectors[i][0])) % 360.0
            direction = int(round(angle / 45.0)) % 8
            if angular_difference(angle, 45.0 * direction) > cfg["angle_tolerance"]:
                valid = False
                break
            slots[direction].append(i)
        if not valid:
            continue
        if any(len(idx) > 1 and pdist(coords[idx]).max() > tol * radius for idx in slots.values()):
            continue
        reach = [float(distance[idx].mean()) for idx in slots.values()]
        if max(reach) / min(reach) > cfg["max_ratio"]:
            continue
        key = (len(core) + len(center), int(sum(counts[i] for i in center)), -ci)
        if best is None or key > best[0]:
            best = (key, {"center": center, "slots": {j: list(idx) for j, idx in slots.items()},
                          "placeholders": placeholders, "radius": radius})
    return None if best is None else best[1]


def lattice_axis(values, tol_frac):
    values = np.sort(np.unique(values))
    span = values[-1] - values[0]
    if span <= 0:
        return None
    gaps = np.diff(values)
    spacing = float(gaps[gaps > 1e-6 * span].min())
    levels = [values[0]]
    for value in values[1:]:
        if value - levels[-1] > tol_frac * spacing:
            levels.append(value)
    index = (np.array(levels) - levels[0]) / spacing
    if np.any(np.abs(index - np.round(index)) > tol_frac):
        return None
    return {"origin": float(levels[0]), "spacing": spacing, "n": int(round(index[-1])) + 1}


def fit_grid(coords, cfg):
    """Square n x n lattice with per-axis spacing; far placeholders allowed outside the lattice."""
    if len(coords) < 4:
        return None
    center = np.median(coords, axis=0)
    distance = np.hypot(*(coords - center).T)
    core = distance <= cfg["placeholder_factor"] * np.quantile(distance, 0.9)
    tol = cfg["slot_tolerance"]
    axes = [lattice_axis(coords[core, k], tol) for k in range(2)]
    if None in axes or axes[0]["n"] != axes[1]["n"] or axes[0]["n"] not in cfg["grid_sizes"]:
        return None
    index = np.stack([(coords[:, k] - axes[k]["origin"]) / axes[k]["spacing"] for k in range(2)], axis=1)
    on_lattice = np.all(np.abs(index - np.round(index)) <= tol, axis=1) & np.all(
        (np.round(index) >= 0) & (np.round(index) < axes[0]["n"]), axis=1)
    far = np.any((index < -1.0) | (index > axes[0]["n"]), axis=1)
    if np.any(~on_lattice & ~far) or np.any(on_lattice & ~core):
        return None
    cells = {tuple(int(v) for v in np.round(index[i])) for i in np.flatnonzero(on_lattice)}
    return {"axes": axes, "n": axes[0]["n"], "cell_of": {int(i): tuple(int(v) for v in np.round(index[i]))
            for i in np.flatnonzero(on_lattice)}, "placeholders": [int(i) for i in np.flatnonzero(far)],
            "observed_cells": len(cells)}


def load_session(record, cfg):
    fields, summary = record["fields"], record["summary"]
    points = np.asarray(fields["trial_targets"], dtype=float)
    block_of = field_vector(fields["trials"], "block_id").astype(int)
    start = field_vector(fields["trials"], "trial_start_index").astype(int)
    end = field_vector(fields["trials"], "trial_end_index").astype(int)
    bounds = {int(k): (int(s), int(e)) for k, s, e in zip(field_vector(fields["blocks"], "block_id"),
              field_vector(fields["blocks"], "sample_start_index"), field_vector(fields["blocks"], "sample_end_index"))}
    whole = np.array([bounds[int(b)][0] <= s < e <= bounds[int(b)][1] for b, s, e in zip(block_of, start, end)])
    expected = {int(k): v for k, v in summary["whole_trial_counts_by_block"].items()}
    if {k: int(np.sum(whole & (block_of == k))) for k in bounds} != expected:
        raise ValueError("Whole-trial counts differ from the structural summary")
    coords, label, precision = distinct_targets(points, cfg["merge"])
    blocks = []
    for block in sorted(set(block_of.tolist())):
        rows = np.flatnonzero(block_of == block)
        rows = rows[np.argsort(start[rows], kind="stable")]
        blocks.append({"block_id": block, "rows": rows})
    return {"participant": summary["participant"], "day": int(summary["post_implant_day"]),
            "arrays": array_labels(fields["electrodes"]), "coords": coords, "label": label, "whole": whole,
            "blocks": blocks, "precision": precision, "trials": len(points)}


def radial_transitions(sequence):
    edges = list(zip(sequence[:-1], sequence[1:]))
    bad = sum(1 for a, b in edges if not ((a == "C") != (b == "C")))
    return len(edges), bad


def radial_mapping(fit, ids):
    mapping = {ids[i]: "C" for i in fit["center"]}
    for direction, idx in fit["slots"].items():
        for i in idx:
            mapping[ids[i]] = direction
    return mapping, {ids[i] for i in fit["placeholders"]}


def classify_session(session, cfg):
    """Return block labels, schedules (session-schedule combinations) and notes.

    A pooled radial fit over all session targets is tried first; mixed sessions fall back to
    block-level radial fits (>=3 distinct targets) and one pooled grid fit over the other blocks.
    """
    coords, label, tol = session["coords"], session["label"], cfg["slot_tolerance"]
    targets = sorted(set(label.tolist()))
    counts = Counter(label.tolist())
    pooled = fit_radial(coords[targets], np.array([counts[i] for i in targets]), cfg) if len(targets) <= 24 else None
    blocks = []
    for block in session["blocks"]:
        ids = label[block["rows"]].tolist()
        blocks.append({"block_id": block["block_id"], "rows": block["rows"], "trials": len(ids),
                       "distinct_targets": len(set(ids)), "label": "pending", "placeholder_ids": set()})
    if pooled is not None:
        fits = {b["block_id"]: radial_mapping(pooled, targets) for b in blocks}
        radius = {b["block_id"]: pooled["radius"] for b in blocks}
    else:
        fits, radius = {}, {}
        for b in blocks:
            present = sorted(set(label[b["rows"]].tolist()))
            if len(present) < 3 or len(present) > 24:
                continue
            fit = fit_radial(coords[present], np.array([counts[i] for i in present]), cfg)
            if fit is not None:
                fits[b["block_id"]] = radial_mapping(fit, present)
                radius[b["block_id"]] = fit["radius"]
    for b in blocks:
        if b["block_id"] not in fits:
            continue
        mapping, placeholder_ids = fits[b["block_id"]]
        ids = label[b["rows"]].tolist()
        sequence = [mapping[i] for i in ids if i not in placeholder_ids]
        edges, bad = radial_transitions(sequence)
        fraction = bad / edges if edges else 0.0
        b.update(fit="radial", mapping=mapping, placeholder_ids=placeholder_ids, radius=radius[b["block_id"]],
                 placeholder_trials=int(sum(i in placeholder_ids for i in ids)), edges=edges,
                 nonpermitted_edges=bad, nonpermitted_fraction=fraction,
                 outward_trials=int(sum(1 for x in sequence if x != "C")),
                 slots_observed=sorted({mapping[i] for i in ids if i not in placeholder_ids}, key=str))
        if not sequence:
            b["label"] = "empty_after_placeholders"
        else:
            b["label"] = "radial" if fraction <= cfg["max_nonpermitted_fraction"] else "radial_conflict"
    pending = [b for b in blocks if b["label"] == "pending"]
    grid = None
    if pending:
        pooled_ids = sorted({int(i) for b in pending for i in label[b["rows"]]})
        grid = fit_grid(coords[pooled_ids], cfg)
        if grid is not None:
            cell_of = {pooled_ids[i]: cell for i, cell in grid["cell_of"].items()}
            placeholder_ids = {pooled_ids[i] for i in grid["placeholders"]}
            for b in pending:
                ids = label[b["rows"]].tolist()
                kept = [i for i in ids if i not in placeholder_ids]
                b.update(fit="grid", label="grid" if kept else "empty_after_placeholders", grid_n=grid["n"],
                         mapping=cell_of, placeholder_ids=placeholder_ids,
                         placeholder_trials=len(ids) - len(kept), slots_observed=sorted({cell_of[i] for i in kept}))
    for b in blocks:
        if b["label"] == "pending":
            many = b["distinct_targets"] > 12 and b["distinct_targets"] > 0.5 * b["trials"]
            b["label"] = "fitts_or_random" if many else "unclassified"
    schedules, notes = [], []
    radial_blocks = [b for b in blocks if b["label"] == "radial"]
    if radial_blocks:
        ring = float(np.median([b["radius"] for b in radial_blocks]))
        centers, by_direction = set(), defaultdict(set)
        for b in radial_blocks:
            for target in set(label[b["rows"]].tolist()) - b["placeholder_ids"]:
                slot = b["mapping"][target]
                (centers.add(target) if slot == "C" else by_direction[slot].add(target))
        spread = lambda ids: pdist(coords[sorted(ids)]).max() if len(ids) > 1 else 0.0
        one_instance = spread(centers) <= tol * ring and all(spread(v) <= tol * ring for v in by_direction.values()) \
            and not (centers & {t for v in by_direction.values() for t in v})
        directions = set(by_direction)
        family = "radial_8" if directions == set(range(8)) else "radial_4" if directions == {0, 2, 4, 6} else None
        if not one_instance:
            notes.append("radial_multiple_instances")
        elif family is None:
            notes.append("radial_incomplete_or_irregular_direction_union")
        else:
            peripheral = {j: coords[sorted(v)].mean(axis=0) for j, v in by_direction.items()}
            ring_center = np.mean(list(peripheral.values()), axis=0)
            if cfg["canonical_center"] == "ring":
                center = ring_center
                if np.max(np.hypot(*(coords[sorted(centers)] - ring_center).T)) > tol * ring:
                    family = None
                    notes.append("stored_center_off_ring_center")
            else:
                center = coords[sorted(centers)].mean(axis=0)
            if family:
                schedules.append({"family": family, "blocks": [b["block_id"] for b in radial_blocks],
                                  "canonical": np.vstack([center] + [peripheral[j] for j in sorted(peripheral)]),
                                  "slots": ["C"] + sorted(peripheral),
                                  "block_slots": {b["block_id"]: b["slots_observed"] for b in radial_blocks}})
    grid_blocks = [b for b in blocks if b["label"] == "grid"]
    if grid_blocks:
        if grid["observed_cells"] >= cfg["grid_min_coverage"] * grid["n"] ** 2:
            ax, ay = grid["axes"]
            lattice = np.array([[ax["origin"] + i * ax["spacing"], ay["origin"] + j * ay["spacing"]]
                                for i in range(grid["n"]) for j in range(grid["n"])])
            schedules.append({"family": "grid", "grid_n": grid["n"], "blocks": [b["block_id"] for b in grid_blocks],
                              "canonical": lattice, "slots": [(i, j) for i in range(grid["n"]) for j in range(grid["n"])],
                              "block_slots": {b["block_id"]: b["slots_observed"] for b in grid_blocks}})
        else:
            notes.append("grid_coverage_below_minimum")
    return blocks, schedules, notes


def two_sides(counts, masks=None, full=None):
    """Whole-block split into two sides of >=64 trials (optionally each side covering every slot).

    Exact depth-first search over block assignments with count and coverage pruning.
    """
    total = sum(counts)
    if masks is None:
        possible = {0}
        for c in counts:
            possible |= {v + c for v in possible}
        return any(v >= 64 and total - v >= 64 for v in possible)
    index = {slot: k for k, slot in enumerate(sorted(full, key=str))}
    bits = [sum(1 << index[x] for x in m if x in index) for m in masks]
    target = (1 << len(index)) - 1
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    suffix_count, suffix_bits = [0] * (len(order) + 1), [0] * (len(order) + 1)
    for k in range(len(order) - 1, -1, -1):
        suffix_count[k] = suffix_count[k + 1] + counts[order[k]]
        suffix_bits[k] = suffix_bits[k + 1] | bits[order[k]]
    seen = set()

    def search(k, ca, ma, cb, mb):
        if ca >= 64 and cb >= 64 and ma == target and mb == target:
            return True
        if k == len(order):
            return False
        if min(ca, 64) + suffix_count[k] < 64 or min(cb, 64) + suffix_count[k] < 64:
            return False
        if (ma | suffix_bits[k]) != target or (mb | suffix_bits[k]) != target:
            return False
        key = (k, min(ca, 64), ma, min(cb, 64), mb)
        if key in seen:
            return False
        seen.add(key)
        i = order[k]
        return search(k + 1, ca + counts[i], ma | bits[i], cb, mb) or search(k + 1, ca, ma, cb + counts[i], mb | bits[i])
    return search(0, 0, 0, 0, 0)


def session_nodes(session, cfg):
    """Session-schedule combinations that qualify under cfg, with capacity."""
    blocks, schedules, notes = classify_session(session, cfg)
    by_id = {b["block_id"]: b for b in blocks}
    kept_rows = {b["block_id"]: [r for r in b["rows"] if int(session["label"][r]) not in b["placeholder_ids"]] for b in blocks}
    nonempty = [bid for bid, rows in kept_rows.items() if rows]
    assigned = {bid for x in schedules for bid in x["blocks"]}
    mixed = bool(set(nonempty) - assigned) or len(schedules) > 1
    status = "single_schedule" if schedules and not mixed else "mixed_layouts" if schedules else "no_qualifying_schedule"
    # Prespecified finite-sampling check: a radial block missing slots despite >=16 outward trials is
    # not explained by the documented without-replacement cycles, so it is a different layout.
    rejected = set()
    if cfg["finite_sampling_check"] and not cfg["exclude_incomplete_blocks"]:
        for x in schedules:
            if x["family"].startswith("radial"):
                rejected |= {bid for bid in x["blocks"] if set(x["block_slots"][bid]) != set(x["slots"])
                             and by_id[bid].get("outward_trials", 0) >= cfg["finite_sampling_outward_trials"]}
    if rejected:
        notes = notes + ["finite_sampling_rejected_blocks:" + ",".join(map(str, sorted(rejected)))]
        mixed = True
        status = "mixed_layouts"
    nodes = []
    if schedules and (not mixed or cfg["split_mixed_sessions"]):
        for schedule in schedules:
            if schedule["family"] not in cfg["families"]:
                continue
            full = set(schedule["slots"])
            counts, masks, kept_blocks = [], [], []
            for bid in schedule["blocks"]:
                b = by_id[bid]
                if bid in rejected:
                    continue
                if cfg["placeholder_blocks_excluded"] and b.get("placeholder_trials", 0):
                    continue
                if cfg["exclude_incomplete_blocks"] and set(schedule["block_slots"][bid]) != full:
                    continue
                counts.append(int(np.sum(session["whole"][kept_rows[bid]])))
                masks.append(set(schedule["block_slots"][bid]))
                kept_blocks.append(bid)
            capacity = two_sides(counts)
            coverage = two_sides(counts, masks, full) if capacity and cfg["packet_coverage"] else capacity
            nodes.append({"participant": session["participant"], "day": session["day"], "arrays": session["arrays"],
                          "family": schedule["family"], "grid_n": schedule.get("grid_n"),
                          "canonical": schedule["canonical"], "blocks": kept_blocks, "whole_trials": sum(counts),
                          "whole_block_capacity": bool(capacity), "packet_coverage_feasible": bool(coverage),
                          "capable": bool(capacity) and bool(coverage)})
    return {"blocks": blocks, "schedules": schedules, "notes": notes, "status": status, "nodes": nodes}


def match_raw(left, right, config):
    """Scale-locked variant: translation only, adopted delta formula in raw units."""
    a, b = left - left.mean(axis=0), right - right.mean(axis=0)
    if len(a) != len(b):
        return {"state": "different_observed_target_cardinality", "mapping": None}
    da, db = pdist(a), pdist(b)
    delta = min(config["median_spacing_fraction"] * min(np.median(da), np.median(db)),
                config["minimum_spacing_fraction"] * min(da.min(), db.min()))
    distances, mapping = cKDTree(b).query(a, k=1)
    bijective = len(set(mapping.tolist())) == len(mapping)
    ok = bijective and float(distances.max()) < delta
    return {"state": "compatible" if ok else "outside_declared_geometry_rule", "mapping": mapping.tolist() if ok else None}


def evaluate(sessions, cfg, support_config):
    classified = [session_nodes(s, cfg) for s in sessions]
    nodes = [node for c in classified for node in c["nodes"]]
    strata = defaultdict(list)
    for index, node in enumerate(nodes):
        strata[(node["participant"], node["arrays"])].append(index)
    groups = []
    for (participant, arrays), indices in sorted(strata.items()):
        shapes = {i: geometry_with_spacing(nodes[i]["canonical"]) for i in indices}
        compatible = []
        for a, b in combinations(indices, 2):
            if cfg["scale_locked"]:
                match = match_raw(nodes[a]["canonical"], nodes[b]["canonical"], support_config["geometry"])
            else:
                match = match_geometry(shapes[a], shapes[b], support_config["geometry"])
            if match["state"] == "compatible":
                compatible.append((a, b))
        compatible_set = set(compatible)
        for description in compatibility_components(indices, compatible):
            group = sorted(description["nodes"], key=lambda i: nodes[i]["day"])
            capable = [i for i in group if nodes[i]["capable"]]
            edges = []
            for a, b in combinations(capable, 2):
                first, second = (a, b) if nodes[a]["day"] < nodes[b]["day"] else (b, a)
                gap = nodes[second]["day"] - nodes[first]["day"]
                if tuple(sorted((a, b))) in compatible_set and (GAP_NEAR[0] <= gap <= GAP_NEAR[1] or gap >= GAP_LONG):
                    edges.append((first, second, "near" if gap <= GAP_NEAR[1] else "long"))
            designs = []
            if description["complete_clique"]:
                for connected in components(capable, [(a, b) for a, b, _ in edges]):
                    keep = set(connected)
                    selected = [e for e in edges if e[0] in keep and e[1] in keep]
                    if selected:
                        designs.append(design_description(nodes, selected, support_config["component_support"]))
            groups.append({"participant": participant, "physical_array_set": list(arrays),
                           "family": nodes[group[0]]["family"], "grid_n": nodes[group[0]]["grid_n"],
                           "session_days": [nodes[i]["day"] for i in group],
                           "capable_session_days": [nodes[i]["day"] for i in capable],
                           "complete_pairwise_clique": description["complete_clique"], "designs": designs})
    participants = {}
    for participant in sorted({s["participant"] for s in sessions}, key=lambda p: int(p[1:])):
        mine = [g for g in groups if g["participant"] == participant]
        best = {"near": 0, "long": 0}
        supported = False
        for g in mine:
            both = [d for d in g["designs"] if d["near_pairs"] and d["long_pairs"]]
            passing = [d for d in both if d["conditional_numerical_pass"]]
            if cfg["strict_group_rule"] and len(passing) != len(both):
                continue
            if passing:
                supported = True
                near, long = sum(d["near_pairs"] for d in passing), sum(d["long_pairs"] for d in passing)
                if min(near, long) > min(best["near"], best["long"]):
                    best = {"near": near, "long": long, "family": g["family"], "sessions": len(g["session_days"])}
        participants[participant] = {"supported": supported,
            "held_out_eligible": participant in CANDIDATES and best["near"] >= 6 and best["long"] >= 6,
            "best_group_passing_pairs": best,
            "both_lag_components": sum(1 for g in mine for d in g["designs"] if d["near_pairs"] and d["long_pairs"]),
            "passing_components": sum(1 for g in mine for d in g["designs"] if d["conditional_numerical_pass"])}
    eligible = [p for p in CANDIDATES if participants.get(p, {}).get("held_out_eligible")]
    options = []
    for triple in combinations(eligible, 3):
        development = list(DEVELOPMENT) + list(NOT_HOLDABLE) + [p for p in CANDIDATES if p not in triple]
        options.append({"held_out": list(triple), "development_supported":
                        sum(participants.get(p, {}).get("supported", False) for p in development)})
    design = {"held_out_eligible_candidates": eligible,
              "held_out_triples": options,
              "design_supported": any(o["development_supported"] >= 4 for o in options),
              "development_supported_if_all_candidates_held_out_except_two_best": max(
                  (o["development_supported"] for o in options), default=None)}
    return classified, groups, participants, design


def diagnostics(sessions, classified):
    out = {"precision_pairs_between_1e-12_and_1e-4_rms": sum(s["precision"]["pairs_between_1e-12_and_1e-4_rms"] for s in sessions),
           "sessions_with_merged_near_duplicates": [[s["participant"], s["day"], s["precision"]["merged_pairs"]]
                                                    for s in sessions if s["precision"]["merged_pairs"]]}
    families, statuses, block_labels = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    fractions, incomplete_long, placeholders = Counter(), [], Counter()
    for s, c in zip(sessions, classified):
        statuses[s["participant"]][c["status"]] += 1
        for schedule in c["schedules"]:
            families[s["participant"]][schedule["family"] + (f"_{schedule.get('grid_n')}" if schedule.get("grid_n") else "")] += 1
        for b in c["blocks"]:
            block_labels[s["participant"]][b["label"]] += 1
            placeholders[s["participant"]] += b.get("placeholder_trials", 0)
            if b.get("fit") == "radial":
                f = b["nonpermitted_fraction"]
                fractions["0" if f == 0 else "(0,0.01]" if f <= .01 else "(0.01,0.05]" if f <= .05 else "(0.05,0.2]" if f <= .2 else ">0.2"] += 1
        for schedule in c["schedules"]:
            if schedule["family"].startswith("radial"):
                full = set(schedule["slots"])
                for b in c["blocks"]:
                    if b["block_id"] in schedule["blocks"] and set(schedule["block_slots"][b["block_id"]]) != full \
                            and b.get("outward_trials", 0) >= 16:
                        incomplete_long.append([s["participant"], s["day"], b["block_id"], b["outward_trials"],
                                                sorted(map(str, full - set(schedule["block_slots"][b["block_id"]])))])
    out.update(session_status=statuses, schedule_families=families, block_labels=block_labels,
               radial_block_nonpermitted_fraction_bins=fractions,
               radial_blocks_missing_slots_with_at_least_16_outward_trials=incomplete_long,
               placeholder_trials=placeholders)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--support-config", type=Path, required=True)
    parser.add_argument("--rules-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classification-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Compute allocation required")
    support_config = json.loads(args.support_config.read_text())
    rules = json.loads(args.rules_config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = [json.loads(line) for line in args.structural_jsonl.open() if line.strip()]
    report = {"analysis_id": "structural_support", "phase": "source_fact_assumption_support",
              "observed_utc": None, "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "rules_config": str(args.rules_config), "rules_adopted_utc": rules["adopted_utc"],
              "support_config": str(args.support_config), "variants": {}}
    for name, override in VARIANTS.items():
        cfg = {**BASE, **override}
        sessions = sorted((load_session(r, cfg) for r in records), key=lambda s: (int(s["participant"][1:]), s["day"]))
        if name == "P0":
            classified = [session_nodes(s, cfg) for s in sessions]
            report["diagnostics_P0"] = diagnostics(sessions, classified)
            with (args.output_dir / "session_classification_P0.jsonl").open("x") as handle:
                for s, c in zip(sessions, classified):
                    handle.write(json.dumps({"participant": s["participant"], "day": s["day"],
                        "arrays": list(s["arrays"]), "trials": s["trials"], "precision": s["precision"],
                        "status": c["status"], "notes": c["notes"],
                        "schedules": [{"family": x["family"], "grid_n": x.get("grid_n"), "blocks": x["blocks"]} for x in c["schedules"]],
                        "nodes": [{k: v for k, v in n.items() if k not in {"canonical", "arrays"}} for n in c["nodes"]],
                        "blocks": [{k: v for k, v in b.items() if k not in {"mapping", "placeholder_ids", "rows"}}
                                   for b in c["blocks"]]}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            if args.classification_only:
                break
        _, groups, participants, design = evaluate(sessions, cfg, support_config)
        report["variants"][name] = {"override": override, "participants": participants, "design": design,
            "groups_with_designs": [g for g in groups if g["designs"]],
            "group_count": len(groups), "nontransitive_groups": sum(not g["complete_pairwise_clique"] for g in groups)}
        print(json.dumps({"variant": name, "eligible": design["held_out_eligible_candidates"],
                          "supported": [p for p, v in participants.items() if v["supported"]],
                          "design_supported": design["design_supported"]}), flush=True)
    report["observed_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = time.monotonic() - started
    report.update(roles_assigned=False, neural_cursor_impedance_values_decoded=False, raw_source_files_opened=False,
                  pair_pruning_performed=False, thresholds_changed_after_application=False)
    (args.output_dir / "source_fact_support_report.json").write_text(
        json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


if __name__ == "__main__":
    main()
