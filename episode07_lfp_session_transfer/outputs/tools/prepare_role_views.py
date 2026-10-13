#!/usr/bin/env python3
"""EP07 trusted preparation: fixed sessions, eligibility, roles and views only.

No fitted representation, neural summary, prediction or score is produced.
Only the two required native windows are read; released values are not changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import math
import os
from pathlib import Path
import random
import re
import time

import h5py
import numpy as np

from source_support import (
    EP06_RESERVED_SESSIONS, ORIGINAL_SESSION_POSITIONS, PREPARATION_SESSIONS,
    compute_only, hdf_metadata_value, session_path, write_json,
)
from view_access import CandidateViews

BANDS = ((0.0, 0.0), (0.5, 4.0), (4.0, 8.0), (8.0, 12.0),
         (12.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 400.0))
SPLIT_SEED = "ep07-target-role-split-v1"
REGIONS = ("M1", "PMd")


class TrialUnavailable(Exception):
    pass


class SourceInsufficient(Exception):
    pass


def field(handle, trial, name):
    if name not in handle["trial_data"]:
        raise TrialUnavailable("required_field_absent")
    refs = handle["trial_data"][name]
    coordinate = np.unravel_index(trial, refs.shape, order="F")
    reference = refs[coordinate]
    if not reference:
        raise TrialUnavailable("missing_field_reference")
    return handle[reference]


def scalar(handle, trial, name):
    node = field(handle, trial, name)
    if node.size != 1:
        raise TrialUnavailable("non_scalar_event_metadata")
    value = float(np.asarray(node).ravel()[0])
    if not math.isfinite(value):
        raise TrialUnavailable("nonfinite_event_metadata")
    return value


def event_index(handle, trial, name):
    value = scalar(handle, trial, name)
    if value != math.floor(value) or value < 1:
        raise TrialUnavailable("invalid_event_index")
    return int(value) - 1


def identity(handle, trial, animal, day):
    if hdf_metadata_value(field(handle, trial, "monkey")) != animal:
        raise SourceInsufficient("source animal identity disagrees with fixed filename")
    date = hdf_metadata_value(field(handle, trial, "date_time"))
    match = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s", date)
    if not match or "".join(f"{int(part):02d}" for part in match.groups()) != day:
        raise SourceInsufficient("source date identity disagrees with fixed filename")
    if scalar(handle, trial, "bin_size") != 0.03:
        raise SourceInsufficient("source does not provide the contracted native 30-ms bins")
    if hdf_metadata_value(field(handle, trial, "result")) != "R":
        raise TrialUnavailable("unsuccessful_reach")
    angle = scalar(handle, trial, "tgtDir")
    direction = int(math.floor(angle / (math.pi / 4) + 0.5)) % 8
    error = abs(math.atan2(math.sin(angle - direction * math.pi / 4),
                           math.cos(angle - direction * math.pi / 4)))
    if error > 0.01:
        raise TrialUnavailable("direction_outside_fixed_tolerance")
    start = event_index(handle, trial, "idx_startTime")
    if start != 0:
        raise SourceInsufficient("provider trial-start origin is not MATLAB one-based index 1")
    movement = event_index(handle, trial, "idx_movement_on")
    go = event_index(handle, trial, "idx_goCueTime")
    windows = {"M1": (movement + 5, movement + 15), "PMd": (go - 15, go)}
    if any(left < 0 for left, right in windows.values()):
        raise TrialUnavailable("required_native_window_absent")
    return direction, windows, angle, error


def guides(handle, trial, region, feature_count, neuron_count):
    lfp = np.asarray(field(handle, trial, f"{region}_lfp_guide")).T
    units = np.asarray(field(handle, trial, f"{region}_unit_guide")).T
    if lfp.shape != (feature_count, 3) or units.shape != (neuron_count, 2):
        raise TrialUnavailable("guide_array_shape_mismatch")
    if not np.isfinite(lfp).all() or not np.isfinite(units).all():
        raise TrialUnavailable("nonfinite_identity_guide")
    if {tuple(row) for row in lfp[:, 1:]} != set(BANDS):
        raise SourceInsufficient("released band guide differs from the nine observed classes")
    identities = [tuple(row) for row in units.tolist()]
    if len(set(identities)) != len(identities) or len({tuple(row) for row in lfp}) != len(lfp):
        raise SourceInsufficient("duplicate neural channel/unit identity within one trial")
    return lfp, units, identities


def eligible_trial(handle, trial, animal, day):
    direction, windows, angle, direction_error = identity(handle, trial, animal, day)
    result = {"source_index": trial, "trial_id": f"trial_{trial + 1:06d}",
              "direction": direction, "direction_radians": angle,
              "direction_circular_error": direction_error, "windows": windows, "unit_ids": {}}
    lengths = []
    for region in REGIONS:
        xnode = field(handle, trial, f"{region}_lfp")
        ynode = field(handle, trial, f"{region}_spikes")
        if len(xnode.shape) != 2 or len(ynode.shape) != 2 or min(xnode.shape + ynode.shape) == 0:
            raise TrialUnavailable("missing_neural_array")
        if xnode.shape[1] != ynode.shape[1]:
            raise TrialUnavailable("lfp_spike_time_axis_mismatch")
        lengths.append(xnode.shape[1])
        left, right = windows[region]
        if right > xnode.shape[1]:
            raise TrialUnavailable("required_native_window_absent")
        _, _, unit_ids = guides(handle, trial, region, xnode.shape[0], ynode.shape[0])
        # These are validity predicates only. Never emit any value or magnitude
        # summary; eligibility is joint across both regions of the same trial.
        x, y = xnode[:, left:right], ynode[:, left:right]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise TrialUnavailable("nonfinite_required_window")
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise TrialUnavailable("not_native_nonnegative_integer_counts")
        result["unit_ids"][region] = unit_ids
    if lengths[0] != lengths[1]:
        raise TrialUnavailable("M1_PMd_trial_time_axis_mismatch")
    return result


def assign_roles(eligible, day_role, rng):
    counts, role_counts = {}, {}
    for direction in range(8):
        group = [row for row in eligible if row["direction"] == direction]
        counts[str(direction)] = len(group)
        if len(group) < 15:
            raise SourceInsufficient(f"direction {direction} has {len(group)} complete trials, needs 15")
        if day_role == "earlier":
            for rank, row in enumerate(group):
                row.update(role="training", within_direction_order=rank)
        else:
            rng.shuffle(group)
            n_cal = math.floor(0.2 * len(group) + 0.5)
            n_dev = (len(group) - n_cal) // 2
            n_hold = len(group) - n_cal - n_dev
            if n_cal < 3 or n_dev < 6 or n_hold < 6:
                raise SourceInsufficient("fixed split lacks required calibration/development/held-out counts")
            for rank, row in enumerate(group):
                role = "calibration" if rank < n_cal else "development" if rank < n_cal + n_dev else "held_out"
                row.update(role=role, within_direction_order=rank)
        role_counts[str(direction)] = dict(Counter(row["role"] for row in group))
    return counts, role_counts


def common_units(eligible):
    common = {}
    for region in REGIONS:
        retained = set(eligible[0]["unit_ids"][region])
        for row in eligible[1:]:
            retained.intersection_update(row["unit_ids"][region])
        common[region] = [unit for unit in eligible[0]["unit_ids"][region] if unit in retained]
        if not common[region]:
            raise SourceInsufficient(f"no consistently present {region} neuron identity")
    return common


def write_record(handle, source, animal, day, region, record, kept_units, content):
    index = record["source_index"]
    left, right = record["windows"][region]
    node = handle.create_group(f"{animal}/{day}/{region}/{record['trial_id']}")
    node.attrs["role"] = record["role"]
    node.attrs["direction"] = record["direction"]
    node.attrs["direction_radians"] = record["direction_radians"]
    node.attrs["direction_circular_error"] = record["direction_circular_error"]
    node.attrs["source_trial_index_zero_based"] = index
    node.attrs["native_bin_seconds"] = 0.03
    if content in ("features", "both"):
        node.create_dataset("lfp", data=field(source, index, f"{region}_lfp")[:, left:right].T)
        node.create_dataset("lfp_guide", data=np.asarray(field(source, index, f"{region}_lfp_guide")).T)
        node.create_dataset("bin_offsets", data=np.arange(5, 15) if region == "M1" else np.arange(-15, 0))
    if content in ("targets", "both"):
        positions = {unit: j for j, unit in enumerate(record["unit_ids"][region])}
        selected = [positions[unit] for unit in kept_units]
        # HDF5 fancy indexing requires sorted indices; take the native window
        # first, then restore the fixed common-unit order in memory.
        y = field(source, index, f"{region}_spikes")[:, left:right].T[:, selected]
        node.create_dataset("spikes", data=y)
        node.create_dataset("unit_guide", data=np.asarray(kept_units))


def separation_check(paths):
    counts = Counter()
    candidate_trials = set()
    for name in ("fit", "development_features"):
        with h5py.File(paths[name], "r") as handle:
            for animal in handle:
                for day in handle[animal]:
                    identities = {}
                    for region in REGIONS:
                        group = handle[f"{animal}/{day}/{region}"]
                        identities[region] = set(group.keys())
                        for trial in group:
                            item = group[trial]
                            role = item.attrs["role"]
                            if name == "fit" and role not in ("training", "calibration"):
                                raise ValueError("fitting file exposes a non-fitting role")
                            if name == "development_features" and (role != "development" or "spikes" in item):
                                raise ValueError("candidate development view exposes targets")
                            key = (animal, day, region, trial)
                            if key in candidate_trials:
                                raise ValueError("trial overlaps fitting and development roles")
                            candidate_trials.add(key)
                            counts[name] += 1
                    if identities["M1"] != identities["PMd"]:
                        raise ValueError("M1 and PMd behavioral trials have different roles")
    # Read only group names/role metadata in scorer and sealed files; never
    # fetch a development or held-out spike array for this access test.
    for name, expected in (("development_targets", "development"), ("heldout", "held_out")):
        with h5py.File(paths[name], "r") as handle:
            for animal in handle:
                for day in handle[animal]:
                    identities = {}
                    for region in REGIONS:
                        group = handle[f"{animal}/{day}/{region}"]
                        identities[region] = set(group.keys())
                        for trial in group:
                            item = group[trial]
                            if item.attrs["role"] != expected:
                                raise ValueError("closed file contains mismatched role")
                            key = (animal, day, region, trial)
                            if name == "heldout" and key in candidate_trials:
                                raise ValueError("held-out trial exposed to candidate")
                            counts[name] += 1
                    if identities["M1"] != identities["PMd"]:
                        raise ValueError("regions disagree on closed whole-trial roles")
    reader = CandidateViews(paths["fit"].parent)
    for forbidden in ("development_targets", "heldout", "scorer"):
        try:
            next(reader.iter_records(forbidden, animal="Mihili", target_day="20140304",
                                     region="M1", stream="history_assisted"))
        except PermissionError:
            pass
        else:
            raise ValueError("candidate reader accepted a closed role")
    return {"analysis_id": "data_separation", "passed": True,
            "proof_scope": "separate_files_and_runner_interface_not_OS_isolation",
            "closed_neural_values_read_by_access_check": False,
            "region_trial_records": dict(counts), "candidate_closed_role_requests_rejected": True}


def prepare(source_root, views, out):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "source_support_report.json"
    if report_path.exists():
        raise FileExistsError("preserve previous preparation report; use a new diagnosed attempt")
    views.mkdir(parents=True, exist_ok=False, mode=0o700)
    paths = {
        "fit": views / "candidate/fit.h5",
        "development_features": views / "candidate/development_features.h5",
        "development_targets": views / "scorer/development_targets.h5",
        "heldout": views / "sealed/heldout.h5",
    }
    partial = {key: path.with_suffix(".h5.partial") for key, path in paths.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    report = {"analysis_id": "source_support", "complete": False, "sessions": [],
              "scores_computed": False, "neural_magnitude_statistics_emitted": False,
              "opening_stage": "preparation_only_no_scientific_scores",
              "joint_excluded_session": {"animal": "Mihili", "day": "20140306",
                                         "regions": list(REGIONS),
                                         "original_session_position": 5,
                                         "reason": "failed_native_count_validity"},
              "ep06_exposure_note": "session roles only; no compatible shared trial split or independent confirmation implied",
              "source": str(source_root), "view_root": str(views),
              "boundary": "runner_enforced_procedural_same_account_raw_source_remains_readable",
              "implant_identity": "Chewie-L is the contract assignment for the six fixed 2016 files; no hemisphere field in source",
              "band_pairs_hz": [list(pair) for pair in BANDS],
              "band_0_0_interpretation": "LMP authenticated by published methods and author zero-frequency signal-construction code",
              "split": {"seed": SPLIT_SEED, "implementation": "Python_random_Random_string_seed_version2_shuffle",
                        "order": "contract_session_order_then_direction_0_to_7_then_native_trial_order",
                        "shared_across_M1_PMd": True},
              "bridge_identifiability": "not_tested_here_training_calibration_only_later_stage"}
    rng = random.Random(SPLIT_SEED)
    rows = []
    try:
        with ExitStack() as stack:
            outputs = {key: stack.enter_context(h5py.File(path, "w")) for key, path in partial.items()}
            for animal, day, day_role in PREPARATION_SESSIONS:
                session = {"animal": animal, "implant_group": "Chewie-L" if animal == "Chewie" else "Mihili",
                           "day": day, "day_role": day_role,
                           "original_session_position": ORIGINAL_SESSION_POSITIONS[(animal, day)],
                           "ep06_session_role": "reserved" if (animal, day) in EP06_RESERVED_SESSIONS else "development",
                           "eligibility_rejections": {}}
                report["sessions"].append(session)
                path = session_path(source_root, animal, day)
                with h5py.File(path, "r") as handle:
                    trial_count = handle["trial_data"]["monkey"].size
                    session["source_trials"] = int(trial_count)
                    eligible, rejected = [], Counter()
                    for trial in range(trial_count):
                        try:
                            eligible.append(eligible_trial(handle, trial, animal, day))
                        except TrialUnavailable as error:
                            rejected[str(error)] += 1
                    session["eligible_trials"] = len(eligible)
                    session["eligibility_rejections"] = dict(rejected)
                    session["complete_trials_by_direction"] = {
                        str(direction): sum(row["direction"] == direction for row in eligible)
                        for direction in range(8)}
                    counts, roles = assign_roles(eligible, day_role, rng)
                    units = common_units(eligible)
                    session["roles_by_direction"] = roles
                    session["consistent_neuron_identity_counts"] = {region: len(units[region]) for region in REGIONS}
                    group_animal = session["implant_group"]
                    for record in eligible:
                        rows.append({"animal_implant": group_animal, "day": day,
                                     "day_role": day_role, "trial_id": record["trial_id"],
                                     "original_session_position": session["original_session_position"],
                                     "ep06_session_role": session["ep06_session_role"],
                                     "regions": "M1;PMd",
                                     "source_index_zero_based": record["source_index"],
                                     "direction": record["direction"], "role": record["role"],
                                     "direction_radians": record["direction_radians"],
                                     "direction_circular_error": record["direction_circular_error"],
                                     "within_direction_order": record["within_direction_order"]})
                        for region in REGIONS:
                            role = record["role"]
                            if role in ("training", "calibration"):
                                write_record(outputs["fit"], handle, group_animal, day, region, record, units[region], "both")
                            elif role == "development":
                                write_record(outputs["development_features"], handle, group_animal, day, region, record, units[region], "features")
                                write_record(outputs["development_targets"], handle, group_animal, day, region, record, units[region], "targets")
                            else:
                                write_record(outputs["heldout"], handle, group_animal, day, region, record, units[region], "both")
                print(f"prepared {animal} {day}: eligibility and role metadata only", flush=True)
        separation = separation_check(partial)
        with (out / "trial_roles.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for key, path in paths.items():
            partial[key].replace(path)
            path.chmod(0o400)
        report["complete"] = True
        report["status"] = "roles_prepared_no_model_fit_or_score"
        report["files"] = {key: str(path) for key, path in paths.items()}
        report["bytes"] = {key: path.stat().st_size for key, path in paths.items()}
        report["seconds"] = time.monotonic() - started
        write_json(out / "data_separation_report.json", separation)
        write_json(report_path, report)
        return 0
    except SourceInsufficient as error:
        report["status"] = "source_insufficient_stop_before_scores"
        report["reason"] = str(error)
        report["partial_views_not_published"] = True
        report["seconds"] = time.monotonic() - started
        write_json(report_path, report)
        print(f"source support stopped: {error}", flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    raise SystemExit(prepare(args.source, args.views, args.out))


if __name__ == "__main__":
    main()
