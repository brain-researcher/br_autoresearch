#!/usr/bin/env python3
"""EP06 fixed-session metadata inventory. Never reads neural/velocity values.

The HDF5 reference traversal and reversed MATLAB shapes follow the existing
EP07 source_support/prepare_role_views implementation. No sibling module,
outcome, role view, native-count check or pre-v7.3 deserializer is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time

EPISODE = Path(__file__).resolve().parents[2]
OUTPUT = EPISODE / "outputs/runs/source_session_support/metadata001"
SOURCE_RELATIVE = "data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source"
SESSIONS = (
    ("Mihili", "20140217"), ("Mihili", "20140218"),
    ("Mihili", "20140303"), ("Mihili", "20140304"),
    ("Mihili", "20140306"), ("Mihili", "20140307"),
    ("Chewie", "20160929"), ("Chewie", "20161005"),
    ("Chewie", "20161006"), ("Chewie", "20161007"),
    ("Chewie", "20161014"), ("Chewie", "20161021"),
)
REGIONS = ("M1", "PMd")
EVENTS = ("idx_startTime", "idx_tgtOnTime", "idx_goCueTime",
          "idx_movement_on", "idx_endTime")
META = ("monkey", "date_time", "data_time", "date", "bin_size", "result",
        "Result", "tgtDir", "trial_id", "trialID", "task") + EVENTS
GUIDES = tuple(f"{r}_{kind}_guide" for r in REGIONS for kind in ("lfp", "unit"))
ARRAYS = tuple(f"{r}_{kind}" for r in REGIONS for kind in ("lfp", "spikes"))
BANDS = ((0., 0.), (.5, 4.), (4., 8.), (8., 12.), (12., 25.),
         (25., 50.), (50., 100.), (100., 200.), (200., 400.))
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
MAX_TRIALS = 10000                  # engineering bound; never a selected subset
MAX_GUIDE_ELEMENTS = 100000
MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class MetadataError(ValueError):
    pass


class DeadlineReached(RuntimeError):
    pass


def compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("requires an actual Slurm compute shell")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("requires exactly one allocated CPU per task")
    for key in ("SLURM_NTASKS", "SLURM_JOB_NUM_NODES"):
        if os.environ.get(key, "1") != "1":
            raise RuntimeError("requires one task on one node")
    if any(os.environ.get(key) != "1" for key in THREAD_VARS):
        raise RuntimeError("set all six thread caps to one before Python starts")
    if Path.cwd().resolve() != EPISODE:
        raise RuntimeError("run from the EP06 episode directory")


def load_runtime():
    global np, h5py
    import numpy as np
    import h5py


def matlab_class(node):
    value = node.attrs.get("MATLAB_class", b"")
    return value.decode() if isinstance(value, bytes) else str(value)


def field(handle, group, trial, name):
    """Dereference one field; reading the reference is not reading its array."""
    if name not in group:
        raise MetadataError(f"missing_field:{name}")
    refs = group[name]
    if not isinstance(refs, h5py.Dataset) or h5py.check_dtype(ref=refs.dtype) is None:
        raise MetadataError(f"not_trial_reference_field:{name}")
    coordinate = np.unravel_index(trial, refs.shape, order="F")
    ref = refs[coordinate]
    if not ref:
        raise MetadataError(f"missing_field_reference:{name}")
    node = handle[ref]
    if not isinstance(node, h5py.Dataset):
        raise MetadataError(f"not_dataset:{name}")
    return node


def metadata_value(node, name):
    """The only value-read boundary. Neural and velocity field names are denied."""
    if name not in META and name not in GUIDES:
        raise PermissionError(f"value access forbidden for {name}")
    if name in GUIDES:
        columns = 3 if name.endswith("lfp_guide") else 2
        if node.ndim != 2 or node.shape[0] != columns or node.size > MAX_GUIDE_ELEMENTS:
            raise MetadataError(f"invalid_or_oversize_guide_shape:{name}")
        if node.dtype.kind not in "biuf":
            raise MetadataError(f"nonnumeric_guide:{name}")
        values = np.asarray(node).T
        if not np.isfinite(values).all():
            raise MetadataError(f"nonfinite_guide:{name}")
        return values.tolist()
    if matlab_class(node) == "char":
        if node.size > 256:
            raise MetadataError(f"oversize_text_metadata:{name}")
        return "".join(chr(int(v)) for v in np.asarray(node).ravel(order="F"))
    if node.size != 1 or node.dtype.kind not in "biuf":
        raise MetadataError(f"not_scalar_metadata:{name}")
    value = float(np.asarray(node).ravel()[0])
    if not math.isfinite(value):
        raise MetadataError(f"nonfinite_metadata:{name}")
    return value


def array_header(node):
    # Do not coerce, slice, iterate, sample or check finiteness of this object.
    return {"shape": list(reversed(node.shape)), "hdf5_shape": list(node.shape),
            "dtype": str(node.dtype)}


def acquisition_day(value):
    if not isinstance(value, str):
        raise MetadataError("acquisition_date_not_text")
    # The source reader documents YYYY/M/D followed by a time string.
    try:
        day = datetime.strptime(value.split()[0], "%Y/%m/%d")
    except (ValueError, IndexError) as error:
        raise MetadataError("unsupported_acquisition_date") from error
    return day.strftime("%Y%m%d")


def check_trial(row, animal, day):
    """Only structural/identity assertions; no trial is dropped on failure."""
    problems = row["problems"]
    metadata = row["metadata"]
    if metadata.get("monkey") != animal:
        problems.append("animal_identity_mismatch")
    dates = [metadata[key] for key in ("date_time", "data_time", "date") if key in metadata]
    if not dates:
        problems.append("acquisition_date_missing")
    else:
        try:
            parsed = [acquisition_day(value) for value in dates]
            row["acquisition_day"] = parsed[0]
            if set(parsed) != {day}:
                problems.append("acquisition_date_mismatch")
        except MetadataError as error:
            problems.append(str(error))
    if not isinstance(metadata.get("bin_size"), (int, float)) or metadata["bin_size"] <= 0:
        problems.append("invalid_bin_size")
    if "result" not in metadata and "Result" not in metadata:
        problems.append("trial_success_field_missing")
    if "tgtDir" not in metadata:
        problems.append("direction_missing")
    lengths = []
    for name in ARRAYS + ("vel",):
        header = row["array_headers"].get(name)
        if header is None:
            continue  # exact field error is already retained
        shape = header["shape"]
        if len(shape) != 2 or min(shape) <= 0:
            problems.append(f"invalid_array_shape:{name}")
        else:
            lengths.append(shape[0])
            if name == "vel" and shape[1] != 2:
                problems.append("velocity_not_two_coordinates")
    if len(lengths) != 5 or len(set(lengths)) != 1:
        problems.append("paired_time_axis_unestablished")
    else:
        row["shared_time_bins"] = lengths[0]
    for key in EVENTS:
        value = metadata.get(key)
        if not isinstance(value, (int, float)) or value < 1 or value != math.floor(value):
            problems.append(f"invalid_one_based_event:{key}")
        elif lengths and value > min(lengths):
            problems.append(f"event_outside_array:{key}")
    if metadata.get("idx_startTime") != 1:
        problems.append("trial_start_not_one")
    # Report guide support, without choosing electrodes or checking neural quality.
    for region in REGIONS:
        lfp = row["guides"].get(f"{region}_lfp_guide")
        units = row["guides"].get(f"{region}_unit_guide")
        for kind, guide in (("lfp", lfp), ("spikes", units)):
            header = row["array_headers"].get(f"{region}_{kind}")
            if guide is not None and header is not None and len(header["shape"]) == 2:
                if len(guide) != header["shape"][1]:
                    problems.append(f"guide_array_size_mismatch:{region}_{kind}")
                if len({tuple(item) for item in guide}) != len(guide):
                    problems.append(f"duplicate_guide_identity:{region}_{kind}")
        if lfp is not None:
            pairs = sorted({tuple(item[1:]) for item in lfp})
            row["guide_support"][region] = {
                "band_pairs": [list(item) for item in pairs],
                "physical_electrodes_per_band": {
                    f"{lo:g},{hi:g}": len({item[0] for item in lfp if item[1:] == [lo, hi]})
                    for lo, hi in pairs},
                "all_registered_band_labels_present": set(pairs) == set(BANDS),
                "usable_electrodes_not_assessed": True,
            }
    row["metadata_and_shape_checks_pass"] = not problems


def inspect_trial(handle, group, index, animal, day):
    row = {"source_trial_index_zero_based": index,
           "structural_trial_key": f"{animal}_CO_{day}:row_{index + 1:06d}",
           "identity_basis": "shared_trial_struct_position_not_invented_provider_id",
           "metadata": {}, "array_headers": {}, "guides": {},
           "guide_support": {}, "problems": []}
    for name in META:
        if name not in group:
            continue
        try:
            row["metadata"][name] = metadata_value(field(handle, group, index, name), name)
        except MetadataError as error:
            row["problems"].append(str(error))
    for name in ARRAYS + ("vel",):
        try:
            row["array_headers"][name] = array_header(field(handle, group, index, name))
        except MetadataError as error:
            row["problems"].append(str(error))
    for name in GUIDES:
        try:
            row["guides"][name] = metadata_value(field(handle, group, index, name), name)
        except MetadataError as error:
            row["problems"].append(str(error))
    check_trial(row, animal, day)
    return row


def inspect_session(path, animal, day):
    result = {"filename": path.name, "source_animal": animal,
              "contract_animal": "Chewie-L" if animal == "Chewie" else animal,
              "declared_day": day, "status": "pending", "trials": [], "problems": [],
              "guide_catalog": {name: [] for name in GUIDES}}
    guide_lookup = {name: {} for name in GUIDES}
    if not path.is_file():
        result.update(status="source_missing", problems=["fixed_source_file_missing"])
        return result
    if not h5py.is_hdf5(path):
        result.update(status="unsupported_format", problems=["requires_HDF5_no_full_MAT_deserialization"])
        return result
    with h5py.File(path, "r") as handle:
        if "trial_data" not in handle or not isinstance(handle["trial_data"], h5py.Group):
            raise MetadataError("missing_trial_data_struct")
        group = handle["trial_data"]
        result["field_names"] = sorted(group.keys())
        if "monkey" not in group:
            raise MetadataError("missing_monkey_reference_field")
        n = int(group["monkey"].size)
        if not 0 < n <= MAX_TRIALS:
            raise MetadataError("trial_count_outside_engineering_bound_no_subsampling")
        result["source_trial_count"] = n
        for name in META + GUIDES + ARRAYS + ("vel",):
            if name not in group:
                continue
            node = group[name]
            if (not isinstance(node, h5py.Dataset) or h5py.check_dtype(ref=node.dtype) is None
                    or node.size != n):
                raise MetadataError(f"trial_reference_count_or_type_mismatch:{name}")
        for index in range(n):
            row = inspect_trial(handle, group, index, animal, day)
            # Guides describe identities, not neural values. Store each exact
            # guide once per session, retaining a binding from every trial.
            row["guide_catalog_indices"] = {}
            for name, values in row.pop("guides").items():
                key = tuple(tuple(item) for item in values)
                if key not in guide_lookup[name]:
                    guide_lookup[name][key] = len(result["guide_catalog"][name])
                    result["guide_catalog"][name].append(values)
                row["guide_catalog_indices"][name] = guide_lookup[name][key]
            result["trials"].append(row)
    counts = Counter(problem for row in result["trials"] for problem in row["problems"])
    result["problem_counts"] = dict(sorted(counts.items()))
    result["direction_radians_counts"] = dict(sorted(Counter(
        str(row["metadata"]["tgtDir"]) for row in result["trials"]
        if "tgtDir" in row["metadata"]).items()))
    result["date_values"] = sorted({row["acquisition_day"] for row in result["trials"]
                                    if "acquisition_day" in row})
    result["all_source_trials_recorded"] = len(result["trials"]) == n
    result["status"] = "metadata_inventory_complete" if not counts else "metadata_inventory_with_problems"
    return result


def atomic_json(path, report):
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) > MAX_OUTPUT_BYTES:
        raise RuntimeError("metadata report exceeds 64 MiB cap; no truncated publication")
    fd, temp = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)  # only inside this invocation's exclusively claimed directory
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def expire(signum, frame):
    raise DeadlineReached("internal_metadata_deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=1680)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1680:
        parser.error("max-seconds must be in 1..1680")
    compute_only()
    source = Path(os.environ["OAK"]) / SOURCE_RELATIVE
    if source != Path("/oak/stanford/groups/russpold") / SOURCE_RELATIVE:
        raise RuntimeError("OAK does not match the already-bound shared source")
    load_runtime()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"analysis_id": "source_session_support", "stage": "metadata_only",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "source_root": str(source), "python": sys.version,
              "numpy": np.__version__, "h5py": h5py.__version__,
              "max_seconds": args.max_seconds, "status": "started", "sessions": [],
              "neural_or_velocity_values_read": False, "neural_quality_assessed": False,
              "scientific_eligibility_assessed": False, "session_roles_assigned": False,
              "trial_roles_assigned": False, "neural_fitting_or_scoring_authorized": False,
              "ep07_access_dependency_unchanged": True, "scientific_trials_consumed": 0,
              "expected_sessions": [f"{animal}_CO_{day}.mat" for animal, day in SESSIONS]}
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    code = 1
    try:
        for animal, day in SESSIONS:
            path = source / f"{animal}_CO_{day}.mat"
            try:
                session = inspect_session(path, animal, day)
            except MetadataError as error:
                session = {"filename": path.name, "status": "structural_error", "error": str(error)}
            report["sessions"].append(session)
            atomic_json(OUTPUT / "session_metadata.json", report)
        report["all_12_sessions_recorded"] = len(report["sessions"]) == 12
        good = all(row["status"] == "metadata_inventory_complete" for row in report["sessions"])
        report["status"] = "metadata_inventory_complete" if good else "metadata_inventory_with_problems"
        code = 0 if good else 2
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="technical_failure", error_type=type(error).__name__, error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report["elapsed_seconds"] = time.monotonic() - started
        seen = {row["filename"] for row in report["sessions"]}
        report["pending_sessions"] = [name for name in report["expected_sessions"] if name not in seen]
        atomic_json(OUTPUT / "session_metadata.json", report)
    print(json.dumps({key: report[key] for key in ("stage", "status", "elapsed_seconds", "pending_sessions")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
