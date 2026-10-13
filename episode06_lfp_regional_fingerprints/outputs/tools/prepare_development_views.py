"""EP06's fixed seven-session development export, not a model or score.

The coordinator supplies an episode-local metadata handoff recording EP07's
actual completed freeze/opening. This procedural same-account boundary is not
authentication or OS isolation. No sibling file is read. The historical EP06
metadata report fixes original trial positions/guides; current source metadata
must match before neural access. All native units and all whole trials remain.

Training alone determines finite nine-band electrodes and one fixed15 draw.
No calibration/evaluation value can select a replacement. Full native LFP is
retained for later registered controls; ordinary readers return only the fixed
15. Nonfinite *unselected* LFP is retained, not imputed. Every selected LFP
value and every native count in the used window must be valid. Failed exports
retain their partials and cannot be resumed or overwritten. No CLI or launch
authority is supplied; resource limits/deadline are owned by the coordinator.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
import socket


EPISODE = Path(__file__).resolve().parents[2]
METADATA_PATH = EPISODE / "outputs/runs/source_session_support/metadata001/session_metadata.json"
HANDOFF_PATH = EPISODE / "outputs/runs/source_session_support/ep07_primary_handoff.json"
KIND = "EP06_seven_session_development_views"
REGIONS = ("M1", "PMd")
ROLES = ("training", "calibration", "evaluation")
BANDS = ((0., 0.), (.5, 4.), (4., 8.), (8., 12.), (12., 25.),
         (25., 50.), (50., 100.), (100., 200.), (200., 400.))
OFFSETS = tuple(range(5, 15))
ROLE_SEED, ELECTRODE_SEED = 6006101, 6006102
ANGLE_TOLERANCE = .005
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


@dataclass(frozen=True)
class SessionSpec:
    animal: str
    animal_index: int
    source_animal: str
    day: str
    original_position: int
    trials: int

    @property
    def filename(self):
        return f"{self.source_animal}_CO_{self.day}.mat"

    @property
    def session_id(self):
        return self.filename[:-4]


SESSIONS = (
    SessionSpec("Mihili", 0, "Mihili", "20140217", 1, 208),
    SessionSpec("Mihili", 0, "Mihili", "20140218", 2, 225),
    SessionSpec("Mihili", 0, "Mihili", "20140304", 4, 203),
    SessionSpec("Chewie-L", 1, "Chewie", "20160929", 1, 208),
    SessionSpec("Chewie-L", 1, "Chewie", "20161005", 2, 202),
    SessionSpec("Chewie-L", 1, "Chewie", "20161007", 4, 168),
    SessionSpec("Chewie-L", 1, "Chewie", "20161014", 5, 740),
)


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource/cancellation check required")
    check()


def _compute_only():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("neural export requires an actual coordinated Slurm compute shell")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise RuntimeError("all six numerical thread limits must be one")


def _runtime():
    import numpy as np
    import h5py
    import inspect_session_metadata as source
    source.load_runtime()
    return np, h5py, source


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, maximum_bytes):
    with Path(path).open("rb") as handle:
        content = handle.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise ValueError("metadata file exceeds its explicit engineering bound")
    return json.loads(content)


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("explicit UTC opening/freeze timestamps required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError("opening/freeze timestamps must carry UTC offset zero")
    return result


def _handoff(path, reports):
    # It records observed events, not a permission token. It may not be created
    # by this exporter or inferred from a job being submitted/running.
    if Path(path) != HANDOFF_PATH:
        raise ValueError("the fixed episode-local EP07 metadata handoff is required")
    handoff = _read_json(path, 4 * 1024 * 1024)
    if (handoff.get("source_episode") != "episode07_lfp_session_transfer"
            or handoff.get("primary_opening_complete") is not True
            or handoff.get("ep07_views") != "source_support_attempt002"
            or handoff.get("excluded_session") != "Mihili_CO_20140306.mat"):
        raise PermissionError("EP07 full freeze and completed primary opening are not recorded")
    times = tuple(_timestamp(handoff.get(name)) for name in
                  ("analysis_frozen_utc", "primary_opening_started_utc", "primary_opening_ended_utc"))
    if not times[0] <= times[1] <= times[2] <= datetime.now(timezone.utc):
        raise PermissionError("EP07 freeze/opening event order is inconsistent")
    expected = {(spec.animal, spec.filename, row["source_trial_index_zero_based"])
                for spec, report in zip(SESSIONS, reports) for row in report["trials"]}
    records = handoff.get("records")
    if not isinstance(records, list):
        raise ValueError("per-trial EP07 role metadata required")
    by_key = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("one explicit EP07 role record per original trial required")
        index, role = record.get("source_trial_index_zero_based"), record.get("ep07_role")
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or not isinstance(role, str) or not role or len(role) > 256):
            raise ValueError("EP07 trial position/role is missing or invalid")
        key = record.get("animal"), record.get("session_name"), index
        if key in by_key:
            raise ValueError("duplicate EP07 trial-role record")
        by_key[key] = role
    if set(by_key) != expected:
        raise ValueError("EP07 role handoff must exactly cover the seven development sessions")
    return handoff, by_key


def _metadata():
    metadata = _read_json(METADATA_PATH, 64 * 1024 * 1024)
    if metadata.get("status") != "metadata_inventory_complete" or metadata.get("stage") != "metadata_only":
        raise ValueError("the completed historical metadata inventory is required")
    reports = []
    for spec in SESSIONS:
        matches = [row for row in metadata["sessions"] if row.get("filename") == spec.filename]
        if len(matches) != 1:
            raise ValueError("a fixed development session is absent or duplicated")
        report = matches[0]
        if (report.get("contract_animal") != spec.animal or report.get("declared_day") != spec.day
                or report.get("status") != "metadata_inventory_complete" or report.get("problem_counts")
                or report.get("source_trial_count") != spec.trials
                or len(report.get("trials", ())) != spec.trials):
            raise ValueError("fixed session metadata/counts differ; no trial/session may be dropped")
        reports.append(report)
    return metadata, tuple(reports)


def plan_session_roles(spec, report, *, check):
    """Deterministic metadata-only partition; no source arrays are accepted."""
    _check(check)
    np, _, _ = _runtime()
    if spec not in SESSIONS or report.get("filename") != spec.filename:
        raise ValueError("one of the seven fixed development sessions is required")
    trials = report["trials"]
    if len(trials) != spec.trials or tuple(row["source_trial_index_zero_based"] for row in trials) != tuple(range(spec.trials)):
        raise ValueError("complete original source-row order is required")
    planned = []
    for row in trials:
        _check(check)
        meta = row["metadata"]
        angle, movement, end = meta.get("tgtDir"), meta.get("idx_movement_on"), meta.get("idx_endTime")
        if (not row.get("metadata_and_shape_checks_pass") or meta.get("bin_size") != .03
                or meta.get("result", meta.get("Result")) != "R"
                or isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle)
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       or x < 1 or x != math.floor(x) for x in (movement, end))):
            raise ValueError("source trial metadata is outside the selected EP06 convention")
        nearest = math.floor(angle / (math.pi / 4) + .5)
        if abs(angle - nearest * (math.pi / 4)) > ANGLE_TOLERANCE:
            raise ValueError("reach angle is outside the fixed circular octant tolerance")
        start = int(movement) - 1 + OFFSETS[0]
        stop = int(movement) - 1 + OFFSETS[-1] + 1
        if start < 0 or stop > row["shared_time_bins"] or stop > int(end):
            raise ValueError("complete half-open150-450ms window is unavailable; no trial dropping")
        planned.append({"source_trial_index_zero_based": row["source_trial_index_zero_based"],
            "trial_id": row["structural_trial_key"], "direction_raw": angle,
            "direction": nearest % 8, "window_start": start, "window_stop": stop,
            "role": None, "calibration_half": -1})
    for direction in range(8):
        rows = [row["source_trial_index_zero_based"] for row in planned if row["direction"] == direction]
        training = len(rows) // 2
        calibration = (len(rows) - training) // 2
        evaluation = len(rows) - training - calibration
        if training < 7 or calibration < 4 or evaluation < 4:
            raise ValueError("a direction lacks7 training/4 calibration/4 evaluation trials")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [ROLE_SEED, spec.animal_index, spec.original_position, direction])))
        shuffled = rng.permutation(rows).tolist()
        for role, indices in zip(ROLES, (shuffled[:training], shuffled[training:training + calibration],
                                        shuffled[training + calibration:])):
            for ordinal, index in enumerate(sorted(indices)):
                planned[index]["role"] = role
                if role == "calibration":
                    planned[index]["calibration_half"] = ordinal % 2
    return tuple(planned)


def _atomic_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_metadata(handle, spec, report, source, check):
    group = handle.get("trial_data")
    if group is None or "monkey" not in group or group["monkey"].size != spec.trials:
        raise ValueError("source trial structure differs from the fixed metadata")
    for expected in report["trials"]:
        _check(check)
        actual = source.inspect_trial(handle, group, expected["source_trial_index_zero_based"],
                                      spec.source_animal, spec.day)
        for field in ("metadata", "array_headers", "shared_time_bins", "structural_trial_key"):
            if actual.get(field) != expected.get(field):
                raise ValueError("current source metadata differs from the fixed inventory")
        if actual["problems"]:
            raise ValueError("current source has structural problems")
        for name, guide in actual["guides"].items():
            if guide != report["guide_catalog"][name][expected["guide_catalog_indices"][name]]:
                raise ValueError("current source guide differs from the fixed inventory")
    return group


def _axes(report, region, np):
    names = (region + "_lfp_guide", region + "_unit_guide")
    if any(len(report["guide_catalog"][name]) != 1 for name in names):
        raise ValueError("within-session native guide changes require explicit review, not intersection")
    guide = np.asarray(report["guide_catalog"][names[0]][0], dtype=np.float64)
    units = np.asarray(report["guide_catalog"][names[1]][0], dtype=np.float64)
    if (guide.ndim != 2 or guide.shape[1] != 3 or units.ndim != 2 or units.shape[1] != 2
            or not len(units) or not np.isfinite(guide).all() or not np.isfinite(units).all()
            or len({tuple(row) for row in guide}) != len(guide)
            or len({tuple(row) for row in units}) != len(units)):
        raise ValueError("complete unique finite native guides required")
    electrodes = tuple(sorted(set(guide[:, 0].tolist())))
    lookup = {tuple(row): index for index, row in enumerate(guide)}
    if len(electrodes) < 15 or len(lookup) != len(electrodes) * len(BANDS):
        raise ValueError("complete nine-band physical-electrode grid required")
    try:
        columns = np.asarray([[lookup[(electrode, *band)] for band in BANDS]
                              for electrode in electrodes], dtype=np.intp)
    except KeyError as error:
        raise ValueError("a native electrode lacks a registered feature band") from error
    return electrodes, columns, units


def _window(handle, group, planned, name, width, source, np):
    node = source.field(handle, group, planned["source_trial_index_zero_based"], name)
    if node.ndim != 2 or node.shape[0] != width or node.dtype.kind not in "fiu":
        raise ValueError("source neural array axes/dtype differ from native guide")
    result = np.asarray(node[:, planned["window_start"]:planned["window_stop"]]).T
    if result.shape != (len(OFFSETS), width):
        raise ValueError("source array does not contain the complete native window")
    return result


def _export_session(handle, spec, report, plan, destination, source, np, h5py, check):
    group = _source_metadata(handle, spec, report, source, check)
    session = destination.create_group(spec.session_id)
    session.attrs.update(animal=spec.animal, source_filename=spec.filename,
                         original_position=spec.original_position, complete=False)
    session.create_dataset("bands", data=np.asarray(BANDS))
    session.create_dataset("bin_offsets", data=np.asarray(OFFSETS, dtype=np.int64))
    text_dtype = h5py.string_dtype("utf-8")
    summaries = []
    for region_index, region in enumerate(REGIONS):
        _check(check)
        electrodes, columns, units = _axes(report, region, np)
        training_rows = [row for row in plan if row["role"] == "training"]
        training = {}
        finite = np.ones(len(electrodes), dtype=bool)
        for row in training_rows:
            _check(check)
            values = _window(handle, group, row, region + "_lfp", columns.size, source, np)[:, columns]
            training[row["source_trial_index_zero_based"]] = values
            finite &= np.isfinite(values).all(axis=(0, 2))
        eligible = np.flatnonzero(finite)
        if len(eligible) < 15:
            raise ValueError("fewer than15 training-finite complete physical electrodes; no fallback")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [ELECTRODE_SEED, spec.animal_index, spec.original_position, region_index])))
        selected = np.sort(rng.choice(eligible, size=15, replace=False))
        region_group = session.create_group(region)
        region_group.create_dataset("electrode_ids", data=np.asarray(electrodes))
        region_group.create_dataset("training_finite_electrodes", data=finite)
        region_group.create_dataset("selected_electrode_positions", data=selected)
        region_group.create_dataset("unit_guide", data=units)
        for role in ROLES:
            rows = [row for row in plan if row["role"] == role]
            role_group = region_group.create_group(role)
            role_group.create_dataset("source_trial_indices", data=np.asarray(
                [row["source_trial_index_zero_based"] for row in rows], dtype=np.int64))
            role_group.create_dataset("trial_ids", data=[row["trial_id"] for row in rows], dtype=text_dtype)
            for key in ("direction", "direction_raw", "calibration_half", "window_start", "window_stop"):
                role_group.create_dataset(key, data=np.asarray([row[key] for row in rows],
                    dtype=np.float64 if key == "direction_raw" else np.int64))
            lfp = role_group.create_dataset("lfp", shape=(len(rows), len(OFFSETS), len(electrodes), 9),
                dtype=np.float64, chunks=(1, len(OFFSETS), len(electrodes), 9))
            spikes = role_group.create_dataset("spikes", shape=(len(rows), len(OFFSETS), len(units)),
                dtype=np.float64, chunks=(1, len(OFFSETS), len(units)))
            for ordinal, row in enumerate(rows):
                _check(check)
                index = row["source_trial_index_zero_based"]
                values = (training[index] if role == "training" else
                    _window(handle, group, row, region + "_lfp", columns.size, source, np)[:, columns])
                if not np.isfinite(values[:, selected, :]).all():
                    raise ValueError("fixed selected LFP contains nonfinite values; no trial drop or redraw")
                counts = _window(handle, group, row, region + "_spikes", len(units), source, np)
                if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
                    raise ValueError("invalid native counts in the used window; no count conversion or unit dropping")
                lfp[ordinal] = values
                spikes[ordinal] = counts
        summaries.append({"region": region, "native_units": len(units),
            "native_electrodes": len(electrodes), "training_finite_electrodes": int(finite.sum()),
            "selected_electrode_ids": [electrodes[index] for index in selected]})
        del training
    session.attrs["complete"] = True
    return summaries


def prepare_development_views(*, opening_handoff=HANDOFF_PATH, output_directory, check):
    """Export all seven sessions after the coordinator's observed EP07 handoff.

    The explicit output directory must be new and task-owned. A successful
    return is role preparation only; it is not a scientific fit or score.
    """
    _compute_only()
    _check(check)
    metadata, reports = _metadata()
    handoff, ep07_roles = _handoff(opening_handoff, reports)
    # Both event admission and exact closed-role metadata coverage precede any
    # scientific import or source-file opening in this empirical entrypoint.
    np, h5py, source = _runtime()
    source_root = Path(os.environ["OAK"]) / source.SOURCE_RELATIVE
    if str(source_root) != metadata.get("source_root"):
        raise ValueError("source root differs from the fixed shared read-only release")
    plans = tuple(plan_session_roles(spec, report, check=check) for spec, report in zip(SESSIONS, reports))
    directory = Path(output_directory)
    if not directory.is_absolute():
        raise ValueError("explicit absolute task-owned output directory required")
    directory.mkdir(parents=False, exist_ok=False)
    report_path = directory / "report.json"
    report = {"kind": KIND, "status": "preparing", "started_utc": _utc(),
        "stage": "preparation_only_no_scientific_scores", "job_id": os.environ["SLURM_JOB_ID"],
        "ep07_analysis_frozen_utc": handoff["analysis_frozen_utc"],
        "ep07_primary_opening_ended_utc": handoff["primary_opening_ended_utc"],
        "reserved_sessions_opened": False, "sessions": [], "neural_access_started_utc": None,
        "original_trials": sum(spec.trials for spec in SESSIONS), "role_seed": ROLE_SEED,
        "electrode_seed": ELECTRODE_SEED, "native_offsets": OFFSETS}
    _atomic_report(report_path, report)
    try:
        with (directory / "trial_roles.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("animal", "session_name", "original_session_position", "source_trial_index_zero_based",
                "trial_id", "direction_raw", "direction", "ep06_role", "calibration_half", "ep07_role"))
            for spec, plan in zip(SESSIONS, plans):
                for row in plan:
                    writer.writerow((spec.animal, spec.filename, spec.original_position,
                        row["source_trial_index_zero_based"], row["trial_id"], row["direction_raw"],
                        row["direction"], row["role"], row["calibration_half"],
                        ep07_roles[(spec.animal, spec.filename, row["source_trial_index_zero_based"]) ]))
            handle.flush()
            os.fsync(handle.fileno())
        partial, final = directory / "development_views.h5.partial", directory / "development_views.h5"
        with h5py.File(partial, "x") as destination:
            destination.attrs.update(kind=KIND, completed=False, expected_sessions=len(SESSIONS))
            sessions = destination.create_group("sessions")
            # Conservatively persist contact before the first source neural
            # read; interruption never erases that contact-start fact.
            report["neural_access_started_utc"] = _utc()
            _atomic_report(report_path, report)
            for spec, source_report, plan in zip(SESSIONS, reports, plans):
                _check(check)
                with h5py.File(source_root / spec.filename, "r") as handle:
                    support = _export_session(handle, spec, source_report, plan, sessions,
                                              source, np, h5py, check)
                destination.flush()
                report["sessions"].append({"filename": spec.filename, "status": "complete",
                    "trial_counts": {role: sum(row["role"] == role for row in plan) for role in ROLES},
                    "regions": support})
                _atomic_report(report_path, report)
            _check(check)
            destination.attrs["completed"] = True
            destination.flush()
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        _check(check)
        os.link(partial, final)
        partial.unlink()
        report.update(status="completed", ended_utc=_utc(), views_path=str(final))
        _atomic_report(report_path, report)
        return final
    except BaseException as error:
        report.update(status="incomplete", ended_utc=_utc(), error_type=type(error).__name__, error=str(error))
        _atomic_report(report_path, report)
        raise
