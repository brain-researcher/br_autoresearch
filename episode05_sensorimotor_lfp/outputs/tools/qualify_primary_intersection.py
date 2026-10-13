#!/usr/bin/env python3
"""Freeze all original unit-guide/LMP intersections without neural value reads."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import resource
import signal
import socket
import time

import numpy as np

from primary_covariance_svd import numeric_trial_ids
from primary_reference import ANIMALS, RECIPE as SCALAR_RECIPE, SESSIONS
from run_primary_reference import trial_records
from source_adapter import EVENTS, M1_SESSIONS, M1Session, compute_only

RECIPES = ("lmp_HW_unit_intersection_v1", "lmp_HW_unit_intersection_random_support8_v1")
REFERENCE = "lmp_history4_repeatability_weight_ridge_v1"
REPLICATES = tuple(range(8))


def guide_intersection(lfp_guide, unit_guide, original_electrodes):
    lfp, units = np.asarray(lfp_guide), np.asarray(unit_guide)
    electrodes = np.asarray(original_electrodes)
    if (lfp.ndim != 2 or lfp.shape[1] != 3 or units.ndim != 2 or units.shape[1] != 2
            or not np.isfinite(lfp).all() or not np.isfinite(units).all()
            or len(set(map(tuple, lfp))) != len(lfp) or len(set(map(tuple, units))) != len(units)
            or electrodes.ndim != 1 or not len(electrodes)
            or not np.issubdtype(electrodes.dtype, np.integer)
            or not np.array_equal(electrodes, np.unique(electrodes)) or np.any(electrodes < 1)):
        raise ValueError("Finite unique guides and ordered original physical LMP identities required")
    for guide in (lfp, units):
        if np.any(guide[:, 0] < 1) or np.any(guide[:, 0] != np.floor(guide[:, 0])):
            raise ValueError("Physical guide electrode identities must be positive integers")
    observed = lfp[(lfp[:, 1] == 0) & (lfp[:, 2] == 0), 0]
    if len(observed) != len(set(observed)) or sorted(observed.tolist()) != electrodes.tolist():
        raise ValueError("The original stable LMP physical support changed")
    unit_electrodes = sorted({int(value) for value in units[:, 0]})
    intersection = sorted(set(electrodes.tolist()) & set(unit_electrodes))
    if not intersection:
        raise ValueError("Empty original LMP/unit-guide intersection; no cohort reduction is permitted")
    return {"original_electrode_ids": electrodes.tolist(), "unit_guide_pairs": units.tolist(),
        "unit_electrode_ids": unit_electrodes, "intersection_electrode_ids": intersection,
        "original_electrodes_without_listed_units": sorted(set(electrodes.tolist())-set(intersection)),
        "unit_electrodes_outside_original_LMP": sorted(set(unit_electrodes)-set(electrodes.tolist())),
        "original_electrode_count": len(electrodes), "intersection_electrode_count": len(intersection),
        "listed_unit_count": len(units), "unit_quality_filter_applied": False}


def support_sets(electrodes, intersection, session):
    electrodes, intersection = list(electrodes), list(intersection)
    if (electrodes != sorted(set(electrodes)) or intersection != sorted(set(intersection))
            or not intersection or not set(intersection) <= set(electrodes)):
        raise ValueError("Nonempty original-order physical intersection required")
    candidate = {"recipe": RECIPES[0], "replicate": None, "electrode_ids": intersection,
        "electrode_positions": [electrodes.index(e) for e in intersection],
        "full_support": intersection == electrodes}
    controls = []
    for replicate in REPLICATES:
        literal = f"ep05-HW-unit-intersection-random-support-v1|20261004|{session}|{replicate}"
        queue = electrodes.copy()
        generator = random.Random()
        generator.seed(literal, version=2)
        generator.shuffle(queue)
        selected = sorted(queue[:len(intersection)])
        controls.append({"recipe": RECIPES[1], "replicate": replicate, "seed_literal": literal,
            "electrode_queue": queue, "electrode_ids": selected,
            "electrode_positions": [electrodes.index(e) for e in selected],
            "full_support": selected == electrodes})
    return [candidate, *controls]


def original_metadata(scalar, history, session, rows):
    if (scalar["status"] != "complete_session_score_bank" or scalar["recipe"] != SCALAR_RECIPE
            or scalar["session_id"] != session or history["session_id"] != session
            or history["status"] != "complete_two_history_repeatability_weight_session_banks"
            or history["exact_original_LMP_baseline_target_trial_replay"] is not True
            or history["exact_original_current_weight_state_and_maps_replay"] is not True):
        raise ValueError("Complete original scalar and HW qualification records required")
    first, second = scalar["source"], history["source"]
    for source in (first, second):
        if source["session_id"] != session or source["complete_trials"] != len(rows):
            raise ValueError("Original session/trial source support changed")
    if first["common_LMP_electrodes"] != second["common_LMP_electrodes"]:
        raise ValueError("Original scalar and HW physical supports disagree")
    if (second["native_LMP_offsets"] != [-3, 14] or second["native_target_offsets"] != [0, 14]
            or second["native_bin_seconds"] != .03 or second["lag_offsets"] != [0, -1, -2, -3]):
        raise ValueError("Original native HW feature/target windows required")
    labels = [int(row["direction"]) for row in rows]
    counts = [labels.count(d) for d in range(8)]
    numeric_trial_ids(np.asarray([row["trial_id"] for row in rows]), session)
    if (set(labels) != set(range(8)) or min(counts) < 2 or scalar["direction_counts"] != counts
            or history["banks"][REFERENCE]["direction_counts"] != counts):
        raise ValueError("Original complete eight-direction support must agree across records")
    return {"session_id": session, "original_electrode_ids": first["common_LMP_electrodes"],
            "complete_trials": len(rows), "direction_counts": counts}


def qualify_session(session, rows, expected, check=lambda: None):
    if session.session_id != expected["session_id"] or len(rows) != expected["complete_trials"]:
        raise ValueError("Exact original source session required")
    first_lfp = first_units = None
    description, guide_bytes = None, 0
    for row in rows:
        check()
        trial = int(row["source_trial_index_zero_based"])
        meta = session.metadata(trial)
        if (meta["trial_id"] != row["trial_id"] or meta["direction"] != int(row["direction"])
                or meta["result"] != "R" or meta["direction_circular_error"] > .01
                or meta["native_bin_seconds"] != .03
                or any(meta[event] != int(row[event]) for event in EVENTS)):
            raise ValueError("Original trial/event metadata changed before guide qualification")
        # The qualified adapter converts only guide arrays to NumPy. Its
        # neural dataset handles are inspected for shape, never for values.
        lfp, units = session.guides(trial)
        current = guide_intersection(lfp, units, expected["original_electrode_ids"])
        lfp_node = session.field(trial, "M1_lfp")
        spike_node = session.field(trial, "M1_spikes")
        velocity_node = session.field(trial, "vel")
        if (len(lfp_node.shape) != 2 or len(spike_node.shape) != 2
                or velocity_node.shape != (2, lfp_node.shape[1])
                or spike_node.shape[1] != lfp_node.shape[1]
                or meta["idx_movement_on"] < 3 or meta["idx_movement_on"]+15 > lfp_node.shape[1]):
            raise ValueError("Original simultaneous native source dimensions/windows changed")
        if first_lfp is None:
            first_lfp, first_units = lfp.copy(), units.copy()
            description = current
        elif not np.array_equal(first_lfp, lfp) or not np.array_equal(first_units, units):
            raise ValueError("The publication-qualified within-session guide invariance changed")
        guide_bytes += int(lfp.nbytes+units.nbytes)
    if description is None:
        raise ValueError("No original retained trials")
    supports = support_sets(description["original_electrode_ids"], description["intersection_electrode_ids"],
                            expected["session_id"])
    unique_new = {tuple(row["electrode_ids"]) for row in supports if not row["full_support"]}
    p, n = 4*description["intersection_electrode_count"], len(rows)
    return {**expected, **description, "animal": ANIMALS[expected["session_id"]],
        "trial_ids": [row["trial_id"] for row in rows],
        "source_trial_indices_zero_based": [int(row["source_trial_index_zero_based"]) for row in rows],
        "trial_directions": [int(row["direction"]) for row in rows],
        "source_path": str(session.path), "source_trial_count": session.count,
        "lfp_guide": first_lfp.tolist(), "constant_guides_on_all_original_trials": True,
        "guide_logical_bytes": guide_bytes, "support_sets": supports,
        "unique_new_fit_supports": len(unique_new),
        "workload_proxy": len(unique_new)*(105*n*p*p+8*p*p*p),
        "spike_values_read": False, "LFP_XY_values_read": False, "scores_computed": False}


def choose_pilot(sessions):
    if len(sessions) != 16 or {row["session_id"] for row in sessions} != set(SESSIONS):
        raise ValueError("All original sessions must qualify before choosing a resource pilot")
    by_id = {row["session_id"]: row for row in sessions}
    ordered = [by_id[s] for s in SESSIONS]
    best = max(ordered, key=lambda row: row["workload_proxy"])
    return best["session_id"] if best["workload_proxy"] > 0 else None


def qualify_all(source, records, original, check=lambda: None, session_factory=None):
    if set(records) != set(SESSIONS) or set(original) != set(SESSIONS):
        raise ValueError("Every original session must reconcile before source guides are opened")
    factory = M1Session if session_factory is None else session_factory
    descriptions = []
    for session_id in SESSIONS:
        check()
        specification = next(s for s in M1_SESSIONS if f"{s[0]}_{s[2]}" == session_id)
        with factory(source, *specification) as session:
            descriptions.append(qualify_session(session, records[session_id], original[session_id], check))
    return {"sessions": descriptions, "pilot_session": choose_pilot(descriptions),
        "pilot_selection_rule": "maximum u*(105*N*(4*k)^2+8*(4*k)^3); original session-order ties",
        "global_full_HW_duplicate": all(row["unique_new_fit_supports"] == 0 for row in descriptions),
        "all_sixteen_qualified_before_values": True,
        "spike_values_read": False, "LFP_XY_values_read": False, "scores_computed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "trial-table", "scalar-bank-root", "hw-bank-root", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args()
    compute_only()
    if not 0 < args.max_seconds <= 60:
        raise ValueError("The adopted guide-only internal bound is at most60seconds")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Explicit one-thread settings required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "started", "scientific_trial_count_increment": 0,
        "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "source": str(args.source),
        "trial_table": str(args.trial_table), "scalar_bank_root": str(args.scalar_bank_root),
        "HW_bank_root": str(args.hw_bank_root), "spike_values_read": False,
        "LFP_XY_values_read": False, "scores_computed": False}
    def deadline(*_):
        raise TimeoutError("Selected guide-only qualification deadline reached")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        records = {s: trial_records(args.trial_table, s) for s in SESSIONS}
        original = {}
        for session in SESSIONS:
            check()
            scalar = json.loads((args.scalar_bank_root/session/"score_bank.json").read_text())
            history = json.loads((args.hw_bank_root/session/"score_banks.json").read_text())
            original[session] = original_metadata(scalar, history, session, records[session])
        report.update(qualify_all(args.source, records, original, check))
        check()
        report["status"] = "complete_primary_intersection_metadata"
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            process_CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.out/"primary_intersection_metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds", "pilot_session", "scientific_trial_count_increment")}), flush=True)


if __name__ == "__main__":
    main()
