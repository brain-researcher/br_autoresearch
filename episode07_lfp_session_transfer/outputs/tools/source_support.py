#!/usr/bin/env python3
"""Bounded EP07 source steward. No fitting or scientific-score interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import numpy as np
import h5py
from scipy.io import loadmat, whosmat

# Historical twelve-session qualification roster. Keep its order and entries:
# qualify_remaining_sources.py intentionally addresses the historical suffix.
SESSIONS = (
    ("Mihili", "20140217", "earlier"), ("Mihili", "20140218", "earlier"),
    ("Mihili", "20140303", "earlier"), ("Mihili", "20140304", "new"),
    ("Mihili", "20140306", "new"), ("Mihili", "20140307", "new"),
    ("Chewie", "20160929", "earlier"), ("Chewie", "20161005", "earlier"),
    ("Chewie", "20161006", "earlier"), ("Chewie", "20161007", "new"),
    ("Chewie", "20161014", "new"), ("Chewie", "20161021", "new"),
)
# October 10 adopted joint EP06/EP07 amendment; no other automatic exclusion.
PREPARATION_SESSIONS = tuple(
    session for session in SESSIONS if session[:2] != ("Mihili", "20140306")
)
ORIGINAL_SESSION_POSITIONS = {
    (animal, day): position
    for animal in ("Mihili", "Chewie")
    for position, (_, day, _) in enumerate(
        (session for session in SESSIONS if session[0] == animal), start=1
    )
}
EP06_RESERVED_SESSIONS = frozenset({
    ("Mihili", "20140303"), ("Mihili", "20140307"),
    ("Chewie", "20161006"), ("Chewie", "20161021"),
})
IDENTITY_EVENT_FIELDS = (
    "bin_size", "data_time", "date_time", "date", "monkey", "Result", "result",
    "tgtDir", "idx_startTime", "idx_tgtOnTime", "idx_goCueTime",
    "idx_movement_on", "idx_endTime", "trial_id", "trialID", "task",
)


def compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("source steward requires an actual Slurm compute shell")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def simple_metadata(value):
    array = np.asarray(value)
    if array.dtype.kind in "US":
        return "".join(array.astype(str).ravel().tolist())
    if array.dtype.kind in "biuf" and array.size <= 6:
        values = [float(x) if np.isfinite(x) else None for x in array.ravel()]
        return values[0] if len(values) == 1 else values
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def session_path(source, animal, day):
    path = source / f"{animal}_CO_{day}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"contract session missing: {path.name}")
    return path


def load_trials(path):
    variables = whosmat(path)
    structs = [name for name, shape, kind in variables if kind == "struct"]
    if len(structs) != 1:
        raise ValueError(f"expected one trial struct in {path.name}; metadata={variables}")
    # MATLAB v5 stores fields together. The steward must deserialize that
    # struct, but only explicit metadata fields may reach the inspection report.
    data = loadmat(path, variable_names=structs, squeeze_me=True, struct_as_record=False)
    return np.atleast_1d(data[structs[0]]).ravel(), variables


def matlab_class(obj):
    value = obj.attrs.get("MATLAB_class", b"")
    return value.decode() if isinstance(value, bytes) else str(value)


def hdf_metadata_value(dataset):
    if not isinstance(dataset, h5py.Dataset):
        return {"kind": "group", "fields": list(dataset.keys())}
    if matlab_class(dataset) == "char":
        return "".join(chr(int(v)) for v in np.asarray(dataset).ravel(order="F"))
    if dataset.size <= 6:
        return simple_metadata(np.asarray(dataset).T)
    return {"shape": list(reversed(dataset.shape)), "dtype": str(dataset.dtype)}


def first_field(handle, group, name):
    node = group[name]
    if isinstance(node, h5py.Dataset) and h5py.check_dtype(ref=node.dtype) is not None:
        ref = node[tuple(0 for _ in node.shape)]
        if not ref:
            raise ValueError(f"empty reference for metadata field {name}")
        return handle[ref]
    return node


def inspect_hdf(path, include_trial_schema):
    with h5py.File(path, "r") as handle:
        names = [name for name in handle if not name.startswith("#")]
        variables = [{"name": name, "class": matlab_class(handle[name]),
                      "kind": "group" if isinstance(handle[name], h5py.Group) else "dataset",
                      "shape": list(handle[name].shape) if isinstance(handle[name], h5py.Dataset) else None}
                     for name in names]
        entry = {"format": "MATLAB_v7.3_HDF5", "variables": variables}
        if not include_trial_schema:
            return entry
        candidates = [name for name in names if isinstance(handle[name], h5py.Group)
                      and matlab_class(handle[name]) == "struct"]
        if len(candidates) != 1:
            raise ValueError(f"ambiguous MATLAB struct metadata in {path.name}: {variables}")
        group = handle[candidates[0]]
        fields = list(group.keys())
        reference_fields = [group[name] for name in fields if isinstance(group[name], h5py.Dataset)
                            and h5py.check_dtype(ref=group[name].dtype) is not None]
        lengths = {field.size for field in reference_fields}
        if len(lengths) != 1:
            raise ValueError(f"trial field reference counts disagree in {path.name}")
        entry["trial_count"] = int(next(iter(lengths)))
        entry["field_names"] = fields
        entry["first_trial_identity_and_event_metadata"] = {
            name: hdf_metadata_value(first_field(handle, group, name))
            for name in IDENTITY_EVENT_FIELDS if name in fields}
        entry["neural_array_metadata"] = {}
        entry["guides"] = {}
        for name in fields:
            if name.endswith(("_spikes", "_lfp")):
                node = first_field(handle, group, name)
                entry["neural_array_metadata"][name] = {
                    "shape": list(reversed(node.shape)), "HDF5_storage_shape": list(node.shape),
                    "dtype": str(node.dtype)}
            elif name.endswith(("_unit_guide", "_lfp_guide")):
                node = first_field(handle, group, name)
                guide = np.asarray(node).T
                details = {"shape": list(guide.shape), "dtype": str(guide.dtype)}
                if guide.ndim == 2 and guide.dtype.kind in "biuf":
                    details["first_three_rows"] = [[float(x) if np.isfinite(x) else None for x in row]
                                                    for row in guide[:3]]
                    if name.endswith("_lfp_guide") and guide.shape[1] == 3:
                        details["band_pairs"] = [[float(x) if np.isfinite(x) else None for x in row]
                                                 for row in np.unique(guide[:, 1:], axis=0)]
                entry["guides"][name] = details
        return entry


def inspect_source(source):
    report = {"stage": "source_support_structural_inspection", "sessions": [],
              "neural_values_or_statistics_emitted": False,
              "heldout_scores_computed": False}
    # Two fixed earlier days are enough to establish both animal field schemas.
    schema_days = {"20140217", "20160929"}
    for animal, day, role in SESSIONS:
        path = session_path(source, animal, day)
        entry = {"animal": animal, "day": day, "day_role": role,
                 "filename": path.name, "bytes": path.stat().st_size}
        if h5py.is_hdf5(path):
            entry.update(inspect_hdf(path, day in schema_days))
            report["sessions"].append(entry)
            continue
        entry["format"] = "MATLAB_pre_v7.3"
        entry["variables"] = whosmat(path)
        if day in schema_days:
            trials, _ = load_trials(path)
            first = trials[0]
            fields = list(first._fieldnames)
            entry["trial_count"] = len(trials)
            entry["field_names"] = fields
            entry["first_trial_identity_and_event_metadata"] = {
                name: simple_metadata(getattr(first, name)) for name in IDENTITY_EVENT_FIELDS if name in fields}
            entry["neural_array_metadata"] = {
                name: {"shape": list(np.asarray(getattr(first, name)).shape),
                       "dtype": str(np.asarray(getattr(first, name)).dtype)}
                for name in fields if name.endswith(("_spikes", "_lfp"))}
            entry["guides"] = {}
            for name in fields:
                if name.endswith(("_unit_guide", "_lfp_guide")):
                    guide = np.asarray(getattr(first, name))
                    details = {"shape": list(guide.shape), "dtype": str(guide.dtype)}
                    if guide.ndim == 2 and guide.dtype.kind in "biuf":
                        details["first_three_rows"] = [[float(x) if np.isfinite(x) else None for x in row]
                                                        for row in guide[:3]]
                        if name.endswith("_lfp_guide") and guide.shape[1] == 3:
                            details["band_pairs"] = [[float(x) if np.isfinite(x) else None for x in row]
                                                     for row in np.unique(guide[:, 1:], axis=0)]
                    entry["guides"][name] = details
            del trials, first
        report["sessions"].append(entry)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect",))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    result = inspect_source(args.source)
    write_json(args.out, result)
    print(json.dumps({"stage": args.stage, "complete": True, "sessions": len(result["sessions"])}))


if __name__ == "__main__":
    main()
