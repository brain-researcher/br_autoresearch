"""Assemble EP16 realized session nodes from structural records and role-checked feature arrays.

Feature arrays come from extract_development_features.py (development) or, for T9, only from
the sealed evaluation's in-memory read. Packets use structural fields and cursor-at-start only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apply_conditional_schedule_support import design_description
from ep16_packets import build_packet, session_schedule
from ep16_panel import Session, chronological_edges
from mat5_structural import field_vector

FEATURES = ("sbp", "tx_4_5")


def passing_component_days(support_report, participant):
    report = json.loads(Path(support_report).read_text())
    days = [d["session_days"] for g in report["variants"]["P0"]["groups_with_designs"] if g["participant"] == participant
            for d in g["designs"] if d["conditional_numerical_pass"]]
    if len(days) != 1:
        raise ValueError(f"Expected one passing P0 component for {participant}")
    return sorted(int(d) for d in days[0])


def structural_records(structural_jsonl, participant, days):
    wanted, out = set(days), {}
    with open(structural_jsonl) as handle:
        for line in handle:
            if f'"participant":"{participant}"' not in line and f'"participant": "{participant}"' not in line:
                continue
            record = json.loads(line)
            day = int(record["summary"]["post_implant_day"])
            if record["summary"]["participant"] == participant and day in wanted:
                out[day] = record
    if set(out) != wanted:
        raise ValueError(f"Structural records missing for days {sorted(wanted - set(out))}")
    return out


def load_feature_dir(feature_dir, participant, days):
    out = {}
    for day in days:
        with np.load(Path(feature_dir) / f"{participant}_day_{day}.npz", allow_pickle=False) as data:
            out[day] = {key: data[key] for key in data.files}
    return out


def physical_keys(record, f):
    """Physical electrode identity (array label, Utah-grid x, y) per neural column.

    electrode_id is a per-file column number: T11 day 1549 lists the medial array first (ids 0-95),
    the other 63 sessions the lateral array. Grid position within an array is the physical identity.
    """
    from extract_development_features import per_electrode_arrays
    el = record["fields"]["electrodes"]
    ids = field_vector(el, "electrode_id").astype(np.int64)
    arrays = per_electrode_arrays(el, len(ids))
    if not (np.array_equal(ids, np.asarray(f["electrode_id"])) and np.array_equal(arrays, np.asarray(f["electrode_array"]).astype(str))):
        raise ValueError("Feature columns and structural electrode records differ")
    keys = [(str(a), float(x), float(y)) for a, x, y in zip(arrays, field_vector(el, "x"), field_vector(el, "y"))]
    if len(set(keys)) != len(keys):
        raise ValueError("Physical electrode keys are not unique within a session")
    return keys


DEAD_RELATIVE_SD = 1e-6
ARTIFACT_POWER_RATIO = 10.0


def validity_mask(feature, X):
    """Return session validity, counts, and per-channel artifact thresholds.

    Dead channels fail the relative finite-value SD cutoff. For sbp, iterate
    the ARTIFACT_POWER_RATIO times valid-median fence until its mask stabilizes,
    raising ValueError if 50 passes do not suffice. Nonfinite/nonpositive
    medians and non-sbp use +inf. fence_iterations counts all passes.
    """
    flat = X.reshape(-1, X.shape[-1])
    finite = np.isfinite(flat)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(np.where(finite, flat, np.nan), axis=0)
    ref = np.nanmedian(sd)
    dead = ~(sd > DEAD_RELATIVE_SD * ref)

    eligible = finite & ~dead[None, :]
    valid = eligible.copy()
    artifact_mask = np.zeros_like(valid)
    threshold = np.full(flat.shape[1], np.inf)
    fence_iterations = 0

    if feature == "sbp":
        for fence_iterations in range(1, 51):
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(np.where(valid, flat, np.nan), axis=0)
            positive = np.isfinite(med) & (med > 0)
            threshold.fill(np.inf)
            with np.errstate(over="ignore"):
                threshold[positive] = ARTIFACT_POWER_RATIO * med[positive]

            # Reclassify all eligible windows so threshold reproduces the mask.
            next_artifacts = eligible & (flat > threshold[None, :])
            if np.array_equal(next_artifacts, artifact_mask):
                break
            artifact_mask = next_artifacts
            valid = eligible & ~artifact_mask
        else:
            raise ValueError(
                "sbp artifact fence did not stabilize within 50 iterations"
            )

    return (
        valid.reshape(X.shape),
        {
            "dead_channels": int((dead & finite.any(axis=0)).sum()),
            "artifact_windows": int(artifact_mask.sum()),
            "fence_iterations": fence_iterations,
        },
        threshold,
    )


def block_normalize(f, name, schedule_blocks, trial_block, eligible_rows, valid_full):
    """z-score every channel within each block with that block's eligible trials (label-free).

    Provider convention ('block-wise normalization before fitting decoding models'); proposed primary
    pending the scientist's decision because evaluation blocks use their own unlabelled statistics.
    """
    values = np.where(valid_full, f[name], np.nan)
    out = np.full_like(values, np.nan)
    for block in schedule_blocks:
        rows = eligible_rows[trial_block[eligible_rows] == block]
        if len(rows) < 2:
            continue
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(values[rows], axis=0)
            sd = np.nanstd(values[rows], axis=0)
        enough = np.isfinite(sd) & (np.isfinite(values[rows]).sum(axis=0) >= 2)
        spread = enough & (sd > 0)
        all_rows = np.flatnonzero(trial_block == block)
        z = (values[all_rows] - mu) / np.where(spread, sd, 1.0)
        # A channel constant within a block carries no within-block deviation: 0, not missing, so that
        # a later scaler never rests on a few surviving values. Invalid windows stay missing.
        z = np.where(spread, z, 0.0)
        out[all_rows] = np.where(enough & np.isfinite(values[all_rows]), z, np.nan)
    return out


def assemble(records, features, features_used=FEATURES, extra=("sbp_odd_even",), normalization="block"):
    """Build packets and Session nodes; returns (sessions by feature set, packet reports, axis)."""
    if normalization not in ("block", "none"):
        raise ValueError(f"Unknown normalization {normalization!r}")
    packets, schedules = {}, {}
    for day in sorted(records):
        f = features[day]
        schedule = session_schedule(records[day])
        if not np.array_equal(f["trial_start"], schedule["trial_start"]):
            raise ValueError(f"Day {day}: feature and structural trial order differ")
        if not np.allclose(f["target_start"], np.asarray(records[day]["fields"]["trial_targets"]), equal_nan=True):
            raise ValueError(f"Day {day}: target at start differs from the structural view")
        packets[day] = build_packet(schedule, f["cursor_start"], f["target_start"], f["window_inside_block"])
        schedules[day] = schedule
    resolved = [d for d in sorted(packets) if packets[d]["status"] == "resolved"]
    keys = {d: physical_keys(records[d], features[d]) for d in resolved}
    axis = sorted({key for d in resolved for key in keys[d]})
    position = {key: i for i, key in enumerate(axis)}
    sessions = []
    for day in resolved:
        f, packet = features[day], packets[day]
        columns = np.array([position[key] for key in keys[day]])
        available = np.zeros(len(axis), dtype=bool)
        available[columns] = True
        X, quality, valid_rows = {}, {}, {}
        schedule = schedules[day]
        eligible_rows = packet["eligible_rows"]  # every eligible trial of the kept blocks (block statistics)
        for name in tuple(features_used):
            full = np.full((2, 64, len(axis)), np.nan)
            for k, side in enumerate("AB"):
                full[k][:, columns] = f[name][packet["sides"][side]["rows"]]
            valid, quality[name], threshold = validity_mask(name, full)
            if normalization == "block":
                # Reuse the packet's pre-artifact threshold for every trial of the session, then
                # z-score each block with its eligible trials.
                session_valid = np.isfinite(f[name])
                dead = ~np.isfinite(np.where(valid, full, np.nan)).any(axis=(0, 1))
                session_valid[:, dead[columns]] = False
                if name == "sbp":
                    session_valid &= ~(f[name] > threshold[columns][None, :])
                valid_rows[name] = session_valid
                normalized = block_normalize(f, name, schedule["blocks"], schedule["trial_block"], eligible_rows, session_valid)
                full = np.full((2, 64, len(axis)), np.nan)
                for k, side in enumerate("AB"):
                    full[k][:, columns] = normalized[packet["sides"][side]["rows"]]
                X[name] = np.where(valid, full, np.nan)
            else:
                X[name] = np.where(valid, full, np.nan)
        for name in tuple(extra):  # odd/even halves of sbp share the full-window validity and normalization
            if name not in f:
                continue
            full = np.full((2, 2, 64, len(axis)), np.nan)
            for h in (0, 1):
                half = f[name][h]
                if normalization == "block":
                    ok = valid_rows["sbp"] & np.isfinite(half)
                    with np.errstate(invalid="ignore"):
                        med = np.nanmedian(np.where(ok[eligible_rows], half[eligible_rows], np.nan), axis=0)
                    ok &= ~(half > ARTIFACT_POWER_RATIO * med[None, :])  # the same +10 dB rule on each half
                    half = block_normalize({"half": half}, "half", schedule["blocks"], schedule["trial_block"], eligible_rows, ok)
                for k, side in enumerate("AB"):
                    full[h, k][:, columns] = half[packet["sides"][side]["rows"]]
            X[name] = np.where(np.isfinite(X["sbp"])[None], full, np.nan)
        packet["feature_validity"] = quality
        packet["normalization"] = normalization
        sides = [packet["sides"][s] for s in "AB"]
        sessions.append(Session(day, X, np.stack([s["labels"] for s in sides]), np.stack([s["octants"] for s in sides]),
                                np.stack([s["position_block"] for s in sides]), available,
                                packet["configuration_pure_session"], np.stack([s["trial_start"] for s in sides])))
    return sessions, packets, axis


def realized_design(sessions, support_config):
    """Re-gate the realized chronological design with the unchanged component limits."""
    days = [s.day for s in sessions]
    src, tgt, long = chronological_edges(days)
    edges = [(int(a), int(b), "long" if l else "near") for a, b, l in zip(src, tgt, long)]
    nodes = [{"day": d} for d in days]
    return design_description(nodes, edges, support_config["component_support"])


def packet_summary(packets):
    """Count-only description of packet construction (no neural values)."""
    resolved = [p for p in packets.values() if p["status"] == "resolved"]
    removed = {}
    for p in packets.values():
        for key, value in p["removed_by_rule"].items():
            removed[key] = removed.get(key, 0) + value
    return {"sessions": len(packets), "resolved": len(resolved),
            "unresolved_days": sorted(p["day"] for p in packets.values() if p["status"] != "resolved"),
            "removed_by_rule": removed,
            "first32_octants_complete": sum(p["first32_octants_complete"] for p in resolved),
            "first16_octants_complete": sum(p["first16_octants_complete"] for p in resolved),
            "configuration_pure_sessions": sum(p["configuration_pure_session"] for p in resolved),
            "configuration_total_variation": {"zero": sum(p["configuration_total_variation"] == 0 for p in resolved),
                                              "max": max((p["configuration_total_variation"] for p in resolved), default=None)},
            "feasible_colourings_median": float(np.median([p["feasible_colourings"] for p in resolved])) if resolved else None,
            "eligible_trials_min": min((p["eligible_trials"] for p in resolved), default=None),
            "feature_validity": {f: {"sessions_with_dead_channels": sum(p["feature_validity"][f]["dead_channels"] > 0 for p in resolved
                                                                       if "feature_validity" in p),
                                     "dead_channel_sessions_total": sum(p["feature_validity"][f]["dead_channels"] for p in resolved
                                                                        if "feature_validity" in p),
                                     "artifact_windows": sum(p["feature_validity"][f]["artifact_windows"] for p in resolved
                                                             if "feature_validity" in p)}
                                 for f in FEATURES}}
