#!/usr/bin/env python3
"""EP08 outcome-blind session, event, guide and array-shape inspection.

Reads scalar trial metadata and electrode guides from the public source.
Never loads an LFP or spike dataset's elements. No candidate or scoring API.
"""
import argparse
import json
import os
from pathlib import Path
import socket

import h5py
import numpy as np

SESSIONS = {
    "Mihili": ["20140217", "20140218", "20140303", "20140304", "20140306", "20140307"],
    "Chewie": ["20160929", "20161005", "20161006", "20161007", "20161014", "20161021"],
}
REQUIRED_BANDS = [(0., 0.), (100., 200.), (200., 400.)]


def mclass(node):
    value = node.attrs.get("MATLAB_class", b"")
    return value.decode() if isinstance(value, bytes) else str(value)


def field(handle, group, name, trial):
    node = group[name]
    if h5py.check_dtype(ref=node.dtype) is None:
        raise ValueError(f"Expected trial references in {name}")
    index = np.unravel_index(trial, node.shape, order="F")
    return handle[node[index]]


def scalar(handle, group, name, trial):
    node = field(handle, group, name, trial)
    if node.size != 1:
        raise ValueError(f"Nonscalar {name}")
    return float(node[...].ravel()[0])


def string(handle, group, name, trial):
    node = field(handle, group, name, trial)
    if mclass(node) != "char":
        raise ValueError(f"Noncharacter {name}")
    return "".join(chr(int(v)) for v in node[...].ravel(order="F"))


def inspect_session(path, animal, day, role):
    with h5py.File(path, "r") as handle:
        names = [n for n in handle if not n.startswith("#")]
        structs = [handle[n] for n in names if isinstance(handle[n], h5py.Group)
                   and mclass(handle[n]) == "struct"]
        if len(structs) != 1:
            raise ValueError(f"Expected one trial struct in {path.name}")
        group = structs[0]
        fields = list(group)
        required = ["bin_size", "monkey", "tgtDir", "idx_movement_on", "idx_endTime",
                    "M1_lfp", "M1_spikes", "M1_lfp_guide", "M1_unit_guide"]
        missing = sorted(set(required) - set(fields))
        if missing:
            return {"filename": path.name, "missing_fields": missing, "status": "unsupported_schema"}
        ntrials = group["bin_size"].size
        if any(group[n].size != ntrials for n in required):
            raise ValueError(f"Mismatched trial field lengths in {path.name}")
        guide = field(handle, group, "M1_lfp_guide", 0)[...].T
        if guide.ndim != 2 or guide.shape[1] != 3:
            raise ValueError("Expected feature/channel/band guide")
        channels_by_band = {
            f"{lo:g}:{hi:g}": sorted(set(int(row[0]) for row in guide
                                         if tuple(row[1:]) == (lo, hi)))
            for lo, hi in REQUIRED_BANDS
        }
        common_channels = sorted(set.intersection(*(set(v) for v in channels_by_band.values())))
        records = []
        exclusions = {}
        for trial in range(ntrials):
            dt = scalar(handle, group, "bin_size", trial)
            movement = scalar(handle, group, "idx_movement_on", trial)
            direction = scalar(handle, group, "tgtDir", trial)
            result_name = "Result" if "Result" in group else "result"
            success = string(handle, group, result_name, trial) == "R"
            neural_shapes = {n: list(reversed(field(handle, group, n, trial).shape))
                             for n in ("M1_lfp", "M1_spikes")}
            reason = None
            if not success:
                reason = "unsuccessful"
            elif not np.isfinite(dt) or not np.isclose(dt, .03, rtol=0, atol=1e-10):
                reason = "non_30ms_bins"
            elif not np.isfinite(movement) or movement != int(movement):
                reason = "missing_or_noninteger_movement"
            elif not np.isfinite(direction):
                reason = "missing_direction"
            else:
                direction_index = int(np.round(direction / (np.pi / 4))) % 8
                discrepancy = np.angle(np.exp(1j * (direction - direction_index * np.pi / 4)))
                # Source scalar codes are not all exact multiples of pi/4:
                # observed eight-target values deviate by up to ~0.003 rad.
                # Decode only within 0.01 rad, far below half-target spacing.
                if abs(discrepancy) > .01:
                    reason = "non_eight_direction_code"
                start = int(movement) - 1 + 5
                end = start + 10
                if any(len(shape) != 2 or start < 0 or end > shape[0]
                       for shape in neural_shapes.values()):
                    reason = "incomplete_movement_window"
            if reason:
                exclusions[reason] = exclusions.get(reason, 0) + 1
            else:
                records.append({"trial_index_zero_based": trial, "direction_index": direction_index,
                                "source_direction_radians": direction,
                                "nominal_direction_deviation_radians": float(discrepancy),
                                "window_start_zero_based": start, "window_end_exclusive": end,
                                "array_shapes": neural_shapes})
        counts = [sum(r["direction_index"] == d for r in records) for d in range(8)]
        return {
            "filename": path.name, "animal": animal, "date": day, "session_role": role,
            "monkey_field": string(handle, group, "monkey", 0),
            "implant_label": "Chewie-L" if animal == "Chewie" else "Mihili",
            "implant_label_basis": "contract; hemisphere not independently encoded in inspected scalar metadata",
            "trial_count": ntrials, "fields": fields, "eligible_metadata_count_by_direction": counts,
            "exclusions": exclusions, "band_pairs": np.unique(guide[:, 1:], axis=0).tolist(),
            "required_band_channels": channels_by_band, "common_physical_channels": common_channels,
            "geometry_fields": [n for n in fields if any(k in n.lower() for k in ("geometry", "coord", "position"))],
            "metadata_eligible_trials": records,
            "status": "metadata_inspected_neural_finiteness_and_semantics_not_yet_qualified",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Source inspection requires an actual Slurm compute shell")
    if args.out.exists():
        raise FileExistsError(args.out)
    report = {
        "stage": "source_support", "neural_elements_read": False, "scores_computed": False,
        "split_rule": "first four chronological fixed sessions per animal development; last two held",
        "split_chosen_before_any_policy_score": True, "sessions": [],
        "scope": "metadata support only; neither reference nor new-branch scoring activated",
    }
    for animal, dates in SESSIONS.items():
        for index, day in enumerate(dates):
            path = args.source / f"{animal}_CO_{day}.mat"
            report["sessions"].append(inspect_session(path, animal, day, "development" if index < 4 else "held"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"sessions": len(report["sessions"]), "neural_elements_read": False,
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
