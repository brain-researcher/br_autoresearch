#!/usr/bin/env python3
"""Finish fixed-source validity inventory after the unpublished EP07 attempt.

Only the seven sessions not reached by attempt001 are inspected. Outputs are
eligibility/identity predicates and support counts, never neural values,
moments, scores or role views. No rounding, rescaling or exclusions are made.
"""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from diagnose_count_representation import near_integer
from prepare_role_views import (
    REGIONS, SourceInsufficient, TrialUnavailable, common_units, eligible_trial,
    field, identity,
)
from source_support import SESSIONS, compute_only, session_path, write_json


def inspect_session(source, animal, day, day_role):
    report = {"animal": animal, "day": day, "day_role": day_role}
    rejected, identity_directions = Counter(), Counter()
    regions = {region: Counter() for region in REGIONS}
    shapes = {region: set() for region in REGIONS}
    eligible = []
    with h5py.File(session_path(source, animal, day), "r") as handle:
        count = int(handle["trial_data"]["monkey"].size)
        report["source_trials"] = count
        for trial in range(count):
            try:
                direction, windows, _, _ = identity(handle, trial, animal, day)
            except (TrialUnavailable, SourceInsufficient) as error:
                rejected[str(error)] += 1
                continue
            identity_directions[str(direction)] += 1
            for region, summary in regions.items():
                try:
                    node = field(handle, trial, f"{region}_spikes")
                    if len(node.shape) != 2 or min(node.shape) == 0:
                        raise TrialUnavailable("missing_neural_array")
                    shapes[region].add((str(node.dtype), int(node.shape[0])))
                    left, right = windows[region]
                    if right > node.shape[1]:
                        raise TrialUnavailable("required_window_absent")
                    values = node[:, left:right]
                except (TrialUnavailable, SourceInsufficient) as error:
                    summary[str(error)] += 1
                    continue
                summary["windows_examined"] += 1
                if not np.isfinite(values).all():
                    summary["nonfinite_windows"] += 1
                    continue
                summary["finite_windows"] += 1
                if np.all(values >= 0):
                    summary["nonnegative_windows"] += 1
                if np.all(values == np.floor(values)):
                    summary["exact_integer_windows"] += 1
                elif near_integer(values):
                    summary["noninteger_but_within_8_ulp_of_integer_windows"] += 1
                else:
                    summary["noninteger_beyond_8_ulp_windows"] += 1
                    if near_integer(values * 0.03):
                        summary["noninteger_becomes_near_integer_if_multiplied_by_bin_seconds"] += 1
            try:
                eligible.append(eligible_trial(handle, trial, animal, day))
            except (TrialUnavailable, SourceInsufficient) as error:
                rejected[str(error)] += 1
    directions = Counter(str(row["direction"]) for row in eligible)
    report.update(
        eligibility_rejections=dict(rejected),
        metadata_valid_direction_counts={str(d): identity_directions[str(d)] for d in range(8)},
        complete_trial_direction_counts={str(d): directions[str(d)] for d in range(8)},
        complete_trials=len(eligible),
        eight_direction_support_passes=all(directions[str(d)] >= 15 for d in range(8)),
        regions={region: {"validity_counts": dict(summary),
                          "spike_dtype_and_column_counts": [list(x) for x in sorted(shapes[region])]}
                 for region, summary in regions.items()},
    )
    if eligible:
        try:
            report["common_unit_counts"] = {r: len(u) for r, u in common_units(eligible).items()}
        except SourceInsufficient as error:
            report["common_unit_failure"] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    if args.out.exists():
        raise FileExistsError("preserve existing diagnosis; do not overwrite")
    report = {"scope": "seven_fixed_sessions_unreached_by_attempt001",
              "source_version": "dryad-xd2547dkt-v5-payload",
              "neural_values_or_moments_emitted": False,
              "rounding_rescaling_or_exclusion_authorized": False,
              "roles_assigned_or_published": False, "sessions": []}
    for animal, day, day_role in SESSIONS[5:]:
        report["sessions"].append(inspect_session(args.source, animal, day, day_role))
        print(f"Qualified validity metadata: {animal} {day}", flush=True)
    write_json(args.out, report)


if __name__ == "__main__":
    main()
