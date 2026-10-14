"""Prepare session-standardized EP23 A0 image patterns for one participant."""

import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
STRATA = ("laion", "things", "thingsplus")
META_COLUMNS = ("uniqueID", "Word", "WordNet ID", "Wordnet ID4")
ROI_NAMES = [
    "laion-ventral",
    "laion-lateral",
    "laion-dorsal",
    "retinotopic-EVC",
]
IMAGE_PATTERN = re.compile(r"^unique_THINGS\+?_(.+)_p\d+\.jpg$")
HANDOFF_RELATIVE_ROOT = Path(
    "outputs/runs/source_readiness/20261003-development_neural_full01"
)
VOXEL_CHUNK = 1024
INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass
class Session:
    """Hold a read-only beta mapping, statistics, and selected presentations."""

    name: str
    betas: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    beta_rows: np.ndarray
    image_rows: np.ndarray
    axis_indices: np.ndarray
    audit: dict


@dataclass
class GroupPlan:
    """Describe eligible images and alternating presentation assignments."""

    rows: np.ndarray
    n_valid: np.ndarray
    half_a_counts: np.ndarray
    half_b_counts: np.ndarray
    assignments: list


def require_fields(obj, fields, where):
    """Require an object and its named fields."""
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: expected a JSON object")
    missing = [field for field in fields if field not in obj]
    if missing:
        raise ValueError(f"{where}: missing fields {missing}")


def require_string(value, where):
    """Require a nonempty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: expected a nonempty string")
    return value


def require_integer(value, where, minimum=0):
    """Require an int64-compatible integer within the allowed range."""
    if type(value) is not int or not minimum <= value <= INT64_MAX:
        raise ValueError(
            f"{where}: expected an integer between {minimum} and {INT64_MAX}"
        )
    return value


def read_json(path):
    """Read JSON and attach the input path to parse errors."""
    with path.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def jsonl_rows(path):
    """Yield numbered JSON objects without silently skipping rows."""
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            where = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"{where}: blank JSONL line")
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{where}: expected a JSON object")
            yield line_number, row


def read_things(path):
    """Read the ordered THINGS identities with replacement decoding."""
    rows = []
    seen_uids = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = reader.fieldnames or []
        missing = sorted(set(META_COLUMNS) - set(headers))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate column names")
        for index, row in enumerate(reader):
            where = f"{path}: concept row {index}"
            if None in row or any(row[name] is None for name in META_COLUMNS):
                raise ValueError(f"{where}: malformed TSV row")
            record = {name: row[name].strip() for name in META_COLUMNS}
            uid = require_string(record["uniqueID"], f"{where}: uniqueID")
            if uid in seen_uids:
                raise ValueError(f"{where}: duplicate uniqueID {uid!r}")
            seen_uids.add(uid)
            rows.append(record)
    if len(rows) != 1854:
        raise ValueError(f"{path}: expected 1854 concepts, found {len(rows)}")
    return rows


def read_development(path, participant, uid_to_index):
    """Select mapped images in increasing development output_row order."""
    participant_rows = []
    seen_output_rows = set()
    seen_images = set()
    total_rows = 0

    for line_number, row in jsonl_rows(path):
        where = f"{path}:{line_number}"
        require_fields(
            row, ("participant", "image_id", "source_stratum", "output_row"),
            where,
        )
        row_participant = require_string(
            row["participant"], f"{where}: participant"
        )
        image_id = require_string(row["image_id"], f"{where}: image_id")
        stratum = require_string(row["source_stratum"], f"{where}: source_stratum")
        output_row = require_integer(row["output_row"], f"{where}: output_row")
        if row_participant not in PARTICIPANTS:
            raise ValueError(
                f"{where}: unexpected participant {row_participant!r}"
            )
        if stratum not in STRATA:
            raise ValueError(f"{where}: unexpected source_stratum {stratum!r}")
        output_key = (row_participant, output_row)
        image_key = (row_participant, image_id)
        if output_key in seen_output_rows:
            raise ValueError(f"{where}: duplicate participant/output_row {output_key}")
        if image_key in seen_images:
            raise ValueError(f"{where}: duplicate participant/image_id {image_key}")
        seen_output_rows.add(output_key)
        seen_images.add(image_key)
        total_rows += 1
        if row_participant == participant:
            participant_rows.append({
                "image_id": image_id,
                "source_stratum": stratum,
                "output_row": output_row,
            })

    if not participant_rows:
        raise ValueError(f"{path}: no development rows for {participant}")
    participant_rows.sort(key=lambda row: row["output_row"])
    selected = []
    unmapped = []
    for row in participant_rows:
        if row["source_stratum"] == "laion":
            continue
        match = IMAGE_PATTERN.match(row["image_id"])
        concept = uid_to_index.get(match.group(1)) if match else None
        if concept is None:
            unmapped.append({
                **row,
                "reason": (
                    "unknown_uniqueID" if match else "image_id_pattern_mismatch"
                ),
            })
        else:
            selected.append({**row, "concept_index": concept})

    report = {
        "rows_in_shared_file": total_rows,
        "participant_development_rows": len(participant_rows),
        "selected_images": len(selected),
        "selected_image_order": "increasing development output_row",
        "unmapped_rows": len(unmapped),
        "unmapped_rows_by_stratum": {
            stratum: sum(row["source_stratum"] == stratum for row in unmapped)
            for stratum in ("things", "thingsplus")
        },
        "unmapped_row_details": unmapped,
    }
    catalog = {row["image_id"]: row for row in participant_rows}
    return selected, catalog, report


def load_array(path, mmap_mode=None):
    """Load a single NPY array with pickle disabled."""
    result = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    if not isinstance(result, np.ndarray):
        if hasattr(result, "close"):
            result.close()
        raise ValueError(f"{path}: expected a single NPY array")
    return result


def read_trials(
    path, participant, session_name, n_trials,
    selected_by_image, development_catalog, trial_image_strata,
):
    """Validate trial identities and retain selected beta-row references."""
    beta_rows = []
    image_rows = []
    axis_indices = []
    seen_axes = set()
    source_counts = Counter({stratum: 0 for stratum in STRATA})
    n_rows = 0

    for beta_row, (line_number, row) in enumerate(jsonl_rows(path)):
        where = f"{path}:{line_number}"
        require_fields(
            row,
            (
                "image_id", "participant", "session",
                "session_beta_axis_index", "source_stratum", "fold",
            ),
            where,
        )
        if beta_row >= n_trials:
            raise ValueError(
                f"{where}: more trial rows than the {n_trials} beta rows"
            )
        if row["participant"] != participant:
            raise ValueError(
                f"{where}: participant mismatch: "
                f"{row['participant']!r} != {participant!r}"
            )
        if row["session"] != session_name:
            raise ValueError(
                f"{where}: session mismatch: "
                f"{row['session']!r} != {session_name!r}"
            )
        image_id = require_string(row["image_id"], f"{where}: image_id")
        stratum = require_string(row["source_stratum"], f"{where}: source_stratum")
        if stratum not in STRATA:
            raise ValueError(f"{where}: unexpected source_stratum {stratum!r}")
        axis_index = require_integer(
            row["session_beta_axis_index"],
            f"{where}: session_beta_axis_index",
        )
        if axis_index in seen_axes:
            raise ValueError(
                f"{where}: duplicate session_beta_axis_index {axis_index}"
            )
        seen_axes.add(axis_index)

        expected = development_catalog.get(image_id)
        if expected is not None and stratum != expected["source_stratum"]:
            raise ValueError(
                f"{where}: source_stratum mismatch for image {image_id!r}: "
                f"{stratum!r} != development value {expected['source_stratum']!r}"
            )
        previous_stratum = trial_image_strata.setdefault(image_id, stratum)
        if previous_stratum != stratum:
            raise ValueError(
                f"{where}: inconsistent source_stratum for image {image_id!r}"
            )
        source_counts[stratum] += 1

        # JSONL position addresses the development beta array. The supplied
        # axis index orders presentations and need not equal this position.
        selected_row = selected_by_image.get(image_id)
        if selected_row is not None:
            beta_rows.append(beta_row)
            image_rows.append(selected_row)
            axis_indices.append(axis_index)
        # No fold filtering: every row of the development file is used.
        n_rows = beta_row + 1

    if n_rows != n_trials:
        raise ValueError(
            f"{path}: {n_rows} trial rows do not match {n_trials} beta rows"
        )
    return (
        np.asarray(beta_rows, dtype=np.int64),
        np.asarray(image_rows, dtype=np.int64),
        np.asarray(axis_indices, dtype=np.int64),
        dict(source_counts),
    )


def session_statistics(betas, where):
    """Compute float64 population statistics in 1024-voxel chunks."""
    n_trials, n_voxels = betas.shape
    if n_trials < 1:
        raise ValueError(f"{where}: a session must contain at least one trial")
    mean = np.full(n_voxels, np.nan, dtype=np.float64)
    sd = np.full(n_voxels, np.nan, dtype=np.float64)
    finite_columns = np.zeros(n_voxels, dtype=bool)

    for start in range(0, n_voxels, VOXEL_CHUNK):
        stop = min(start + VOXEL_CHUNK, n_voxels)
        block = np.asarray(betas[:, start:stop], dtype=np.float64)
        finite = np.isfinite(block).all(axis=0)
        finite_columns[start:stop] = finite
        if not finite.any():
            continue
        positions = start + np.flatnonzero(finite)
        values = block[:, finite]
        block_mean = values.mean(axis=0, dtype=np.float64)
        values -= block_mean
        np.square(values, out=values)
        block_sd = np.sqrt(values.mean(axis=0, dtype=np.float64))
        if (
            not np.isfinite(block_mean).all()
            or not np.isfinite(block_sd).all()
        ):
            raise ValueError(
                f"{where}: nonfinite float64 statistics for finite input columns"
            )
        mean[positions] = block_mean
        sd[positions] = block_sd
    return mean, sd, finite_columns


def contained_report_path(root, value, where):
    """Resolve a relative report path without allowing root escape."""
    value = require_string(value, f"{where}: report")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{where}: report must be relative to ep04-root")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{where}: report escapes ep04-root: {value!r}")
    return resolved


def read_sessions(root, participant, selected, development_catalog):
    """Validate all 30 handoffs and establish common voxel support."""
    handoff_path = root / HANDOFF_RELATIVE_ROOT / participant / "participant_handoff.json"
    handoff = read_json(handoff_path)
    require_fields(
        handoff, ("participant", "complete_participant_handoff", "sessions"),
        str(handoff_path),
    )
    if handoff["participant"] != participant:
        raise ValueError(f"{handoff_path}: participant identity mismatch")
    if handoff["complete_participant_handoff"] is not True:
        raise ValueError(f"{handoff_path}: complete_participant_handoff must be true")
    records = handoff["sessions"]
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError(f"{handoff_path}: sessions must be a list of 30 entries")

    names = set()
    prepared = []
    report_paths = set()
    for index, record in enumerate(records):
        where = f"{handoff_path}: sessions[{index}]"
        require_fields(record, ("session", "complete", "report"), where)
        name = require_string(record["session"], f"{where}: session")
        if name in names:
            raise ValueError(f"{where}: duplicate session {name!r}")
        if record["complete"] is not True:
            raise ValueError(f"{where}: complete must be true")
        report_path = contained_report_path(root, record["report"], where)
        if report_path in report_paths:
            raise ValueError(f"{where}: duplicate session report {report_path}")
        names.add(name)
        report_paths.add(report_path)
        prepared.append((name, report_path))
    prepared.sort(key=lambda entry: entry[0])

    selected_by_image = {
        row["image_id"]: index for index, row in enumerate(selected)
    }
    sessions = []
    canonical_voxels = None
    canonical_membership = None
    finite_every_session = None
    positive_sd_every_session = None
    payload_paths = set()
    trial_image_strata = {}

    for name, report_path in prepared:
        session_started = time.perf_counter()
        report = read_json(report_path)
        require_fields(
            report,
            (
                "payload_dir", "complete_neural_session_handoff",
                "roi_names", "union_voxel_count",
            ),
            str(report_path),
        )
        if report["complete_neural_session_handoff"] is not True:
            raise ValueError(
                f"{report_path}: complete_neural_session_handoff must be true"
            )
        if report["roi_names"] != ROI_NAMES:
            raise ValueError(
                f"{report_path}: roi_names must equal {ROI_NAMES!r}"
            )
        for field, expected in (("participant", participant), ("session", name)):
            if field in report and report[field] != expected:
                raise ValueError(f"{report_path}: {field} identity mismatch")
        n_voxels = require_integer(
            report["union_voxel_count"],
            f"{report_path}: union_voxel_count",
            minimum=1,
        )
        payload = Path(
            require_string(report["payload_dir"], f"{report_path}: payload_dir")
        )
        if not payload.is_absolute():
            raise ValueError(f"{report_path}: payload_dir must be absolute")
        payload = payload.resolve()
        if payload in payload_paths:
            raise ValueError(f"{report_path}: reused payload directory {payload}")
        payload_paths.add(payload)

        beta_path = payload / "development_betas.npy"
        betas = load_array(beta_path, mmap_mode="r")
        if betas.dtype.kind != "f" or betas.dtype.itemsize != 4:
            raise ValueError(f"{beta_path}: expected float32, found {betas.dtype}")
        if betas.ndim != 2 or betas.shape[1] != n_voxels:
            raise ValueError(
                f"{beta_path}: expected shape (n_trials, {n_voxels}), "
                f"found {betas.shape}"
            )
        if betas.shape[0] < 1:
            raise ValueError(f"{beta_path}: no development trials")

        voxel_path = payload / "voxel_indices_c.npy"
        voxels = load_array(voxel_path)
        if voxels.dtype.kind != "i" or voxels.dtype.itemsize != 8:
            raise ValueError(f"{voxel_path}: expected int64, found {voxels.dtype}")
        if voxels.shape != (n_voxels,):
            raise ValueError(
                f"{voxel_path}: expected shape {(n_voxels,)}, found {voxels.shape}"
            )
        if np.any(voxels < 0) or np.any(voxels[1:] <= voxels[:-1]):
            raise ValueError(
                f"{voxel_path}: indices must be nonnegative and strictly increasing"
            )

        membership_path = payload / "roi_membership.npy"
        membership = load_array(membership_path)
        if membership.dtype != np.dtype(bool):
            raise ValueError(
                f"{membership_path}: expected bool, found {membership.dtype}"
            )
        if membership.shape != (n_voxels, 4):
            raise ValueError(
                f"{membership_path}: expected shape {(n_voxels, 4)}, "
                f"found {membership.shape}"
            )

        if canonical_voxels is None:
            canonical_voxels = np.array(voxels, dtype=np.int64, copy=True)
            canonical_membership = np.array(membership, dtype=bool, copy=True)
            finite_every_session = np.ones(n_voxels, dtype=bool)
            positive_sd_every_session = np.ones(n_voxels, dtype=bool)
        else:
            if not np.array_equal(voxels, canonical_voxels):
                raise ValueError(
                    f"{voxel_path}: voxel identities differ from the first session"
                )
            if not np.array_equal(membership, canonical_membership):
                raise ValueError(
                    f"{membership_path}: ROI membership differs from the first session"
                )

        beta_rows, image_rows, axis_indices, source_counts = read_trials(
            payload / "development_trials.jsonl",
            participant,
            name,
            betas.shape[0],
            selected_by_image,
            development_catalog,
            trial_image_strata,
        )
        stats_started = time.perf_counter()
        mean, sd, finite = session_statistics(betas, str(beta_path))
        stats_seconds = time.perf_counter() - stats_started
        positive_sd = np.isfinite(sd) & (sd > 0.0)
        finite_every_session &= finite
        positive_sd_every_session &= positive_sd

        audit = {
            "session": name,
            "report": str(report_path),
            "payload_dir": str(payload),
            "n_trials": int(betas.shape[0]),
            "trial_counts_by_source_stratum": source_counts,
            "selected_presentations": len(beta_rows),
            "finite_voxels_in_all_session_rows": int(np.count_nonzero(finite)),
            "finite_positive_sd_voxels": int(
                np.count_nonzero(finite & positive_sd)
            ),
            "standardization_seconds": stats_seconds,
            "read_and_standardize_seconds": time.perf_counter() - session_started,
        }
        sessions.append(Session(
            name=name,
            betas=betas,
            mean=mean,
            sd=sd,
            beta_rows=beta_rows,
            image_rows=image_rows,
            axis_indices=axis_indices,
            audit=audit,
        ))

    if canonical_voxels is None:
        raise ValueError(f"{handoff_path}: no usable sessions")
    support = finite_every_session & positive_sd_every_session
    return (
        sessions, canonical_voxels, canonical_membership,
        finite_every_session, support, handoff_path,
    )


def standardized_block(session, beta_rows, columns):
    """Read and standardize a presentation-by-voxel block in float64."""
    values = np.asarray(
        session.betas[np.ix_(beta_rows, columns)], dtype=np.float64
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        values -= session.mean[columns]
        values /= session.sd[columns]
    return values


def collect_valid_presentations(sessions, n_selected, group_columns):
    """Check finiteness on every supported column of each ROI group."""
    references = {
        group: [[] for _ in range(n_selected)]
        for group in group_columns
    }
    total_presentations = np.zeros(n_selected, dtype=np.int64)

    for session_index, session in enumerate(sessions):
        np.add.at(total_presentations, session.image_rows, 1)
        session.audit["selected_valid_presentations"] = {}
        session.audit["selected_invalid_presentations"] = {}
        for group, columns in group_columns.items():
            valid = np.ones(len(session.beta_rows), dtype=bool)
            if len(session.beta_rows):
                for start in range(0, len(columns), VOXEL_CHUNK):
                    chunk_columns = columns[start:start + VOXEL_CHUNK]
                    values = standardized_block(
                        session, session.beta_rows, chunk_columns
                    )
                    valid &= np.isfinite(values).all(axis=1)
            for position in np.flatnonzero(valid):
                selected_row = int(session.image_rows[position])
                references[group][selected_row].append((
                    session_index,
                    int(session.beta_rows[position]),
                    int(session.axis_indices[position]),
                ))
            session.audit["selected_valid_presentations"][group] = int(valid.sum())
            session.audit["selected_invalid_presentations"][group] = int(
                len(valid) - valid.sum()
            )
    return references, total_presentations


def make_group_plan(references, sessions):
    """Sort valid presentations and assign alternating A/B halves."""
    n_valid = np.asarray([len(items) for items in references], dtype=np.int64)
    rows = np.flatnonzero(n_valid >= 2).astype(np.int64, copy=False)
    assignments = [[] for _ in sessions]

    for output_row, selected_row in enumerate(rows):
        ordered = sorted(
            references[int(selected_row)],
            key=lambda ref: (sessions[ref[0]].name, ref[2]),
        )
        for position, (session_index, beta_row, axis_index) in enumerate(ordered):
            assignments[session_index].append(
                (beta_row, output_row, position % 2)
            )

    return GroupPlan(
        rows=rows,
        n_valid=n_valid,
        half_a_counts=(n_valid[rows] + 1) // 2,
        half_b_counts=n_valid[rows] // 2,
        assignments=[
            np.asarray(items, dtype=np.int64).reshape(-1, 3)
            for items in assignments
        ],
    )


def roi_patterns(roi_name, columns, plan, sessions):
    """Accumulate float64 means by voxel chunk and store float32 patterns."""
    shape = (len(plan.rows), len(columns))
    half_a = np.empty(shape, dtype=np.float32)
    half_b = np.empty(shape, dtype=np.float32)
    full = np.empty(shape, dtype=np.float32)
    if len(plan.rows) == 0:
        return half_a, half_b, full

    denominator_a = plan.half_a_counts.astype(np.float64)[:, None]
    denominator_b = plan.half_b_counts.astype(np.float64)[:, None]
    denominator_full = plan.n_valid[plan.rows].astype(np.float64)[:, None]
    if np.any(denominator_a < 1) or np.any(denominator_b < 1):
        raise ValueError(f"{roi_name}: an eligible image has an empty half")

    for start in range(0, len(columns), VOXEL_CHUNK):
        stop = min(start + VOXEL_CHUNK, len(columns))
        chunk_columns = columns[start:stop]
        sum_a = np.zeros((len(plan.rows), stop - start), dtype=np.float64)
        sum_b = np.zeros_like(sum_a)

        for session, entries in zip(sessions, plan.assignments):
            if len(entries) == 0:
                continue
            values = standardized_block(session, entries[:, 0], chunk_columns)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{roi_name}/{session.name}: a previously valid presentation "
                    "became nonfinite during pattern accumulation"
                )
            is_a = entries[:, 2] == 0
            # add.at correctly accumulates repeated images within a session.
            np.add.at(sum_a, entries[is_a, 1], values[is_a])
            np.add.at(sum_b, entries[~is_a, 1], values[~is_a])

        half_a[:, start:stop] = sum_a / denominator_a
        half_b[:, start:stop] = sum_b / denominator_b
        # Weight every presentation equally, including when the halves differ.
        full[:, start:stop] = (sum_a + sum_b) / denominator_full
    return half_a, half_b, full


def exclusion_records(selected, total_presentations, n_valid):
    """Explain every image excluded from a particular presentation group."""
    records = []
    for index in np.flatnonzero(n_valid < 2):
        index = int(index)
        total = int(total_presentations[index])
        valid = int(n_valid[index])
        if total == 0:
            reason = "no_presentations"
        elif valid == 0:
            reason = "no_valid_presentations"
        else:
            reason = "only_one_valid_presentation"
        records.append({
            "selected_row": index,
            "image_id": selected[index]["image_id"],
            "concept_index": selected[index]["concept_index"],
            "total_presentations": total,
            "valid_presentations": valid,
            "reason": reason,
        })
    return records


def main():
    """Validate the handoffs and write image-level ROI pattern artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True, choices=PARTICIPANTS)
    for name in ("ep04-root", "things-meta", "dev-rows", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in ("ep04_root", "things_meta", "dev_rows", "out_dir")
    }
    started = time.perf_counter()
    timings = {}

    phase_started = time.perf_counter()
    things = read_things(paths["things_meta"])
    uid_to_index = {row["uniqueID"]: index for index, row in enumerate(things)}
    selected, development_catalog, development_report = read_development(
        paths["dev_rows"], args.participant, uid_to_index
    )
    timings["metadata_seconds"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    (
        sessions, voxel_indices, membership,
        finite_every_session, support, handoff_path,
    ) = read_sessions(
        paths["ep04_root"], args.participant, selected, development_catalog
    )
    timings["session_read_and_standardization_seconds"] = (
        time.perf_counter() - phase_started
    )

    roi_columns = {}
    for roi_index, roi_name in enumerate(ROI_NAMES):
        columns = np.flatnonzero(
            support & membership[:, roi_index]
        ).astype(np.int64, copy=False)
        if len(columns) < 3:
            raise ValueError(
                f"{roi_name}: only {len(columns)} supported voxels; at least 3 required"
            )
        roi_columns[roi_name] = columns
    group_columns = {
        "primary": np.flatnonzero(
            support & membership[:, :3].any(axis=1)
        ).astype(np.int64, copy=False),
        "evc": roi_columns["retinotopic-EVC"],
    }

    phase_started = time.perf_counter()
    references, total_presentations = collect_valid_presentations(
        sessions, len(selected), group_columns
    )
    plans = {
        group: make_group_plan(group_references, sessions)
        for group, group_references in references.items()
    }
    timings["validity_and_split_seconds"] = time.perf_counter() - phase_started

    arrays = {
        "image_ids": np.asarray(
            [row["image_id"] for row in selected], dtype=np.str_
        ),
        "concept_indices": np.asarray(
            [row["concept_index"] for row in selected], dtype=np.int64
        ),
        "n_valid_primary": plans["primary"].n_valid,
        "n_valid_evc": plans["evc"].n_valid,
    }
    roi_counts = {}
    timings["patterns_seconds_by_roi"] = {}
    for roi_index, roi_name in enumerate(ROI_NAMES):
        phase_started = time.perf_counter()
        group = "primary" if roi_index < 3 else "evc"
        plan = plans[group]
        columns = roi_columns[roi_name]
        half_a, half_b, full = roi_patterns(
            roi_name, columns, plan, sessions
        )
        arrays[f"{roi_name}__rows"] = plan.rows
        arrays[f"{roi_name}__A"] = half_a
        arrays[f"{roi_name}__B"] = half_b
        arrays[f"{roi_name}__full"] = full
        arrays[f"{roi_name}__voxels"] = voxel_indices[columns].astype(
            np.int64, copy=False
        )
        roi_counts[roi_name] = {
            "presentation_group": group,
            "selected_images": len(selected),
            "eligible_images": len(plan.rows),
            "excluded_images": len(selected) - len(plan.rows),
            "supported_voxels": len(columns),
            "valid_presentations_for_eligible_images": int(
                plan.n_valid[plan.rows].sum()
            ),
            "half_A_presentations": int(plan.half_a_counts.sum()),
            "half_B_presentations": int(plan.half_b_counts.sum()),
        }
        timings["patterns_seconds_by_roi"][roi_name] = (
            time.perf_counter() - phase_started
        )

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    npz_path = paths["out_dir"] / f"brain_patterns_{args.participant}.npz"
    report_path = paths["out_dir"] / f"brain_patterns_{args.participant}.json"
    phase_started = time.perf_counter()
    np.savez(npz_path, **arrays)
    timings["npz_write_seconds"] = time.perf_counter() - phase_started

    primary_eligible = plans["primary"].n_valid >= 2
    evc_eligible = plans["evc"].n_valid >= 2
    report = {
        "participant": args.participant,
        "input_paths": {
            name: str(path) for name, path in paths.items() if name != "out_dir"
        },
        "participant_handoff": str(handoff_path),
        "output_npz": str(npz_path),
        "settings": {
            "roi_names": ROI_NAMES,
            "standardization_rows": "all rows of every development_betas.npy",
            "standardization_dtype": "float64",
            "mean_accumulation_dtype": "float64",
            "output_pattern_dtype": "float32",
            "standard_deviation_ddof": 0,
            "voxel_chunk_size": VOXEL_CHUNK,
            "support": (
                "finite in every row of every session and positive SD in every session"
            ),
            "validity_columns": {
                "primary": "supported union of ROIs 0, 1, and 2",
                "evc": "supported ROI 3",
            },
            "presentation_order": ["session", "session_beta_axis_index"],
            "half_A_positions": "0, 2, 4, ...",
            "half_B_positions": "1, 3, 5, ...",
            "minimum_valid_presentations": 2,
            "minimum_roi_voxels": 3,
        },
        "development": development_report,
        "selected_images": len(selected),
        "unmapped_rows": development_report["unmapped_rows"],
        "selected_presentations": int(total_presentations.sum()),
        "eligible_images": {
            "primary": int(primary_eligible.sum()),
            "evc": int(evc_eligible.sum()),
            "either_group": int(np.count_nonzero(primary_eligible | evc_eligible)),
            "both_groups": int(np.count_nonzero(primary_eligible & evc_eligible)),
        },
        "counts_per_roi": roi_counts,
        "support_sizes": {
            "union_voxels_before_support": len(voxel_indices),
            "finite_in_every_row_of_every_session": int(
                np.count_nonzero(finite_every_session)
            ),
            "supported_union_voxels": int(np.count_nonzero(support)),
            "removed_for_nonfinite_values": int(
                np.count_nonzero(~finite_every_session)
            ),
            "removed_for_nonpositive_sd_among_globally_finite_voxels": int(
                np.count_nonzero(finite_every_session & ~support)
            ),
            "primary_union_columns": len(group_columns["primary"]),
            "per_roi": {
                roi_name: len(columns)
                for roi_name, columns in roi_columns.items()
            },
        },
        "n_sessions_read": len(sessions),
        "sessions_read": [session.name for session in sessions],
        "session_details": [session.audit for session in sessions],
        "exclusions": {
            group: exclusion_records(
                selected, total_presentations, plan.n_valid
            )
            for group, plan in plans.items()
        },
        "timings_seconds": timings,
    }
    timings["total_seconds_before_report_write"] = time.perf_counter() - started
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")

    print(
        f"{args.participant}: {len(sessions)} sessions, "
        f"{len(selected)} selected images, "
        f"{development_report['unmapped_rows']} unmapped development rows"
    )
    for roi_name, counts in roi_counts.items():
        print(
            f"{roi_name}: {counts['eligible_images']} eligible images, "
            f"{counts['supported_voxels']} supported voxels"
        )
    print(f"Wrote {npz_path} and {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"a0_brain_patterns.py: {exc}") from exc
