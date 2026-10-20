"""EP16 block-aware 128-trial packet builder (analysis_specification.json, packet).

Inputs are one structural JSONL record (P0 schedule, intended-target slots and placeholders
are recomputed with the adopted source-fact code) and, for eligibility, the cursor position at
each trial start. Neural values never enter packet construction. A session without a feasible
whole-block colouring is packet-unresolved; callers remove it with its pairs before scoring.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial.distance import pdist

from apply_source_fact_support import BASE, load_session, session_nodes
from mat5_structural import field_vector

P0 = dict(BASE)
PACKET, NESTED = 64, (16, 32)


def octants(labels):
    """Half-open counterclockwise octants from +x (analysis_definition.score)."""
    labels = np.asarray(labels, dtype=np.float64)
    angles = np.remainder(np.arctan2(labels[:, 1], labels[:, 0]), 2 * np.pi)
    return np.floor(angles / (np.pi / 4)).astype(np.int64) % 8


def session_schedule(record):
    """P0 schedule of a capable session: kept blocks, per-row slots, placeholders, layout spacing."""
    session = load_session(record, P0)
    result = session_nodes(session, P0)
    capable = [n for n in result["nodes"] if n["capable"]]
    if len(capable) != 1:
        raise ValueError(f"Session {session['participant']} day {session['day']} has {len(capable)} capable P0 nodes")
    node = capable[0]
    by_block = {b["block_id"]: b for b in result["blocks"]}
    n_rows = session["trials"]
    slot = np.array([None] * n_rows, dtype=object)
    placeholder = np.zeros(n_rows, dtype=bool)
    kept = np.zeros(n_rows, dtype=bool)
    signature = {}
    for block_id in node["blocks"]:
        block = by_block[block_id]
        centers = set()
        for row in block["rows"]:
            target = int(session["label"][row])
            kept[row] = True
            if target in block["placeholder_ids"]:
                placeholder[row] = True
                continue
            slot[row] = block["mapping"][target]
            if slot[row] == "C":
                centers.add(tuple(float(v) for v in session["coords"][target]))
        signature[block_id] = tuple(sorted(centers)) or ("no_center_trials",)
    canonical = np.asarray(node["canonical"], dtype=np.float64)
    distances = pdist(canonical)
    spacing = float(np.median(distances[distances > 0]))
    block_start = {int(b): int(s) for b, s in zip(field_vector(record["fields"]["blocks"], "block_id"),
                                                  field_vector(record["fields"]["blocks"], "sample_start_index"))}
    order = sorted(node["blocks"], key=lambda b: block_start[int(b)])
    slots = sorted({s for s in slot[kept & ~placeholder]}, key=str)
    return {"participant": session["participant"], "day": session["day"], "family": node["family"],
            "blocks": [int(b) for b in order], "slot": slot, "placeholder": placeholder, "kept": kept,
            "whole": session["whole"], "canonical": canonical, "median_spacing": spacing,
            "canonical_slots": ["C"] + [s for s in slots if s != "C"] if node["family"].startswith("radial") else slots,
            "block_signature": signature, "trial_block": field_vector(record["fields"]["trials"], "block_id").astype(int),
            "trial_start": field_vector(record["fields"]["trials"], "trial_start_index").astype(np.int64)}


def eligibility(schedule, cursor_start, target_start, window_inside_block, exclusion_fraction=0.05):
    """Ordered eligibility rules; returns the mask, labels and the count removed by each rule."""
    cursor_start = np.asarray(cursor_start, dtype=np.float64)
    target_start = np.asarray(target_start, dtype=np.float64)
    n = len(schedule["trial_start"])
    if cursor_start.shape != (n, 2) or target_start.shape != (n, 2) or len(window_inside_block) != n:
        raise ValueError("Cursor/target/window arrays must align with the structural trial order")
    steps = [("whole_trial", schedule["whole"]), ("kept_schedule_block", schedule["kept"]),
             ("not_placeholder", ~schedule["placeholder"]), ("window_inside_block", np.asarray(window_inside_block, bool)),
             ("finite_cursor_and_target", np.isfinite(cursor_start).all(axis=1) & np.isfinite(target_start).all(axis=1))]
    mask = np.ones(n, dtype=bool)
    removed = {}
    for name, condition in steps:
        removed[name] = int(np.sum(mask & ~condition))
        mask &= condition
    delta = target_start - cursor_start
    distance = np.hypot(delta[:, 0], delta[:, 1])
    far = np.zeros(n, dtype=bool)
    far[mask] = distance[mask] > exclusion_fraction * schedule["median_spacing"]
    removed["near_target_start"] = int(np.sum(mask & ~far))
    mask &= far
    labels = np.full((n, 2), np.nan)
    labels[mask] = delta[mask] / distance[mask, None]
    return mask, labels, removed


def _composition(blocks, signature):
    values, counts = np.unique([repr(signature[b]) for b in blocks], return_counts=True)
    return dict(zip(values.tolist(), (counts / counts.sum()).tolist()))


def _total_variation(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def build_packet(schedule, cursor_start, target_start, window_inside_block, exclusion_fraction=0.05):
    """Whole-block 2-colouring search with hard slot/octant constraints and fixed preferences."""
    mask, labels, removed = eligibility(schedule, cursor_start, target_start, window_inside_block, exclusion_fraction)
    octant = np.full(len(mask), -1)
    octant[mask] = octants(labels[mask])
    rows_by_block = {}
    for block in schedule["blocks"]:
        rows = np.flatnonzero(mask & (schedule["trial_block"] == block))
        rows_by_block[block] = rows[np.argsort(schedule["trial_start"][rows], kind="stable")]
    required_slots = set(schedule["canonical_slots"])
    blocks = schedule["blocks"]
    best, feasible, checked = None, 0, 0
    for colours in product((0, 1), repeat=len(blocks) - 1):
        colouring = (0,) + colours  # the earliest kept block is on side A
        checked += 1
        sides = []
        for side in (0, 1):
            members = [b for b, c in zip(blocks, colouring) if c == side]
            rows = np.concatenate([rows_by_block[b] for b in members]) if members else np.array([], int)
            rows = rows[np.argsort(schedule["trial_start"][rows], kind="stable")][:PACKET]
            sides.append((members, rows))
        if any(len(rows) < PACKET for _, rows in sides):
            continue
        if any(set(schedule["slot"][rows]) != required_slots or len(set(octant[rows])) != 8 for _, rows in sides):
            continue
        feasible += 1
        cover32 = all(len(set(octant[rows[:32]])) == 8 for _, rows in sides)
        cover16 = all(len(set(octant[rows[:16]])) == 8 for _, rows in sides)
        compositions = [_composition(schedule["trial_block"][rows], schedule["block_signature"]) for _, rows in sides]
        tv = round(_total_variation(*compositions), 12)
        key = (not cover32, tv, not cover16, tuple(sorted(sides[0][0])))
        if best is None or key < best[0]:
            best = (key, sides, compositions, cover16, cover32, tv)
    base = {"participant": schedule["participant"], "day": schedule["day"], "eligible_trials": int(mask.sum()),
            "eligible_rows": np.flatnonzero(mask),
            "removed_by_rule": removed, "colourings_checked": checked, "feasible_colourings": feasible,
            "median_spacing": schedule["median_spacing"], "exclusion_distance": exclusion_fraction * schedule["median_spacing"]}
    if best is None:
        return {**base, "status": "packet_unresolved", "reason": "no whole-block colouring satisfies the hard constraints"}
    _, sides, compositions, cover16, cover32, tv = best
    packet = {**base, "status": "resolved", "configuration_total_variation": tv,
              "first32_octants_complete": cover32, "first16_octants_complete": cover16, "sides": {}}
    for name, (members, rows), composition in zip("AB", sides, compositions):
        packet["sides"][name] = {"blocks": [int(b) for b in members], "rows": rows.astype(int),
                                 "labels": labels[rows], "octants": octant[rows],
                                 "position_block": schedule["trial_block"][rows].astype(int),
                                 "trial_start": schedule["trial_start"][rows], "slots": schedule["slot"][rows].tolist(),
                                 "configuration_composition": composition,
                                 "configuration_pure": len(composition) == 1}
    packet["configuration_pure_session"] = (packet["sides"]["A"]["configuration_composition"]
                                            == packet["sides"]["B"]["configuration_composition"]
                                            and packet["sides"]["A"]["configuration_pure"])
    return packet
