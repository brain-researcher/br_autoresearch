#!/usr/bin/env python3
"""Describe EP16 permitted structural metadata without choosing participant roles.

Raw coordinate layouts and observed within-block transition graphs are exact
descriptions, not a scientific schedule-equivalence decision. Time-pair counts
ignore schedule restrictions and are explicitly upper bounds on eligible
support. A bound below the required six pairs proves insufficient support; a
bound above six does not establish eligibility or fixed-effect identifiability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

import numpy as np

from mat5_structural import field_vector


def array_labels(electrodes):
    records = electrodes if isinstance(electrodes, list) else [electrodes]
    labels = []
    for record in records:
        values = record["array"]
        if isinstance(values, str):
            labels.append(values)
        else:
            for value in values:
                if not isinstance(value, str):
                    raise ValueError("Unexpected array-label encoding in structural view")
                labels.append(value)
    return tuple(sorted(set(labels)))


def summarize(records):
    participants = defaultdict(list)
    layout_ids, layouts, session_layouts = {}, [], []
    for record in records:
        fields, summary = record["fields"], record["summary"]
        participant = summary["participant"]
        arrays = array_labels(fields["electrodes"])
        targets = np.asarray(fields["trial_targets"], dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("Invalid target points in structural view")
        trial_blocks = field_vector(fields["trials"], "block_id").astype(int)
        starts = field_vector(fields["trials"], "trial_start_index").astype(int)
        if len(targets) != len(trial_blocks):
            raise ValueError("Trial target and block counts differ")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Nonfinite structural target coordinates need explicit source diagnosis")
        coordinates = tuple(sorted(set(map(tuple, targets.tolist()))))
        if coordinates not in layout_ids:
            layout_ids[coordinates] = f"raw_layout_{len(layout_ids)+1:03d}"
            layouts.append({"id": layout_ids[coordinates], "coordinates": coordinates,
                            "exact_unique_coordinates": len(coordinates)})
        coordinate_ids = {point: index for index, point in enumerate(coordinates)}
        blocks = []
        for block in sorted(set(trial_blocks.tolist())):
            indices = np.flatnonzero(trial_blocks == block)
            indices = indices[np.argsort(starts[indices], kind="stable")]
            sequence = [coordinate_ids[tuple(targets[index])] for index in indices]
            transitions = Counter(zip(sequence[:-1], sequence[1:]))
            blocks.append({"block_id": block, "declared_trials": len(indices),
                           "target_ids_observed": sorted(set(sequence)),
                           "target_counts": {str(key): count for key, count in sorted(Counter(sequence).items())},
                           "within_block_transition_counts": [
                               {"source_target_id": pair[0], "destination_target_id": pair[1], "count": count}
                               for pair, count in sorted(transitions.items())]})
        entry = {"participant": participant, "day": summary["post_implant_day"],
                 "physical_array_set": arrays,
                 "whole_blocks_can_supply_two_64_trial_sides": summary["whole_blocks_can_supply_two_64_trial_sides"],
                 "raw_layout_id": layout_ids[coordinates], "blocks": blocks}
        participants[participant].append(entry)
        session_layouts.append(entry)
    bounds = []
    for participant, sessions in sorted(participants.items()):
        by_arrays = defaultdict(list)
        for session in sessions:
            if session["whole_blocks_can_supply_two_64_trial_sides"]:
                by_arrays[session["physical_array_set"]].append(session)
        group_bounds = []
        for arrays, eligible in sorted(by_arrays.items()):
            counts = Counter()
            days = sorted(session["day"] for session in eligible)
            if len(set(days)) != len(days):
                raise ValueError("Duplicate participant/array/day session identity")
            for index, source in enumerate(days):
                for target in days[index+1:]:
                    gap = target-source
                    counts["near" if 1 <= gap <= 30 else "long" if gap >= 180 else "mid"] += 1
            group_bounds.append({"physical_array_set": arrays, "structurally_capable_sessions": len(days),
                                 "near_pair_upper_bound": counts["near"], "long_pair_upper_bound": counts["long"],
                                 "mid_pairs_descriptive": counts["mid"],
                                 "fails_six_per_gap_even_before_schedule_restrictions":
                                     counts["near"] < 6 or counts["long"] < 6})
        bounds.append({"participant": participant, "sessions": len(sessions),
                       "days": [min(item["day"] for item in sessions), max(item["day"] for item in sessions)],
                       "raw_exact_layouts": len(set(item["raw_layout_id"] for item in sessions)),
                       "by_physical_array_set": group_bounds,
                       "all_array_sets_fail_pair_count_upper_bound":
                           not group_bounds or all(item["fails_six_per_gap_even_before_schedule_restrictions"] for item in group_bounds)})
    return {"participant_support_upper_bounds": bounds, "raw_exact_layouts": layouts,
            "session_layouts_and_transitions": session_layouts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    with args.structural_jsonl.open() as source:
        records = [json.loads(line) for line in source if line.strip()]
    result = summarize(records)
    result.update({"analysis_id": "structural_support", "phase": "metadata_support_description",
                   "source_structural_view": str(args.structural_jsonl),
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                   "observed_utc": datetime.now(timezone.utc).isoformat(),
                   "participants": len(result["participant_support_upper_bounds"]), "sessions": len(records),
                   "schedule_equivalence_adjudicated": False, "coordinate_tolerance_selected": False,
                   "size_dependent_schedule_eligibility_adjudicated": False,
                   "fixed_effect_identifiability_evaluated": False, "roles_assigned": False,
                   "neural_cursor_impedance_values_decoded": False})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"structural_support_description.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"raw_exact_layouts", "session_layouts_and_transitions"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
