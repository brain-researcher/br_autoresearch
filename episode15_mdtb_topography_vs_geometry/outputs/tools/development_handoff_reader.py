"""Map explicitly released EP15 development handoffs into the existing fit APIs.

This is an adapter, not access authority or an isolation mechanism. The caller
must first deploy the enforced reader boundary and release the twelve ordinary
development ZIPs. Trusted literal copying did not authorize numeric decoding.
No original-source path, audit container, fitted cache, output path, fitting,
score, or independence assertion is supplied by this module. A Python binding
or member allowlist does not establish operating-system isolation.

All metadata/ZIP directories and selected headers are checked before voxel
decoding. Only genuine Task-A noninstruction and eighteen B-only beta streams
are decoded, one image at a time, on the unchanged 208-location support.
"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
import io
import math
from pathlib import Path
import struct
from types import MappingProxyType
import zipfile

import numpy as np

from build_anatomical_support import (
    GRID_KEYS, MAX_MEMBER_BYTES, read_image, scale_values, selected_sform,
)
from calibration_falsifiers import (
    TASK_A_HALF_RUNS, TaskACalibrationPacket, TaskBObservation,
)
from development_lopo import DEVELOPMENT_IDS, DevelopmentParticipant
from inspect_registration_headers import nifti_header
from join_source_metadata import EXPECTED_BETAS, SESSIONS, beta_member, validate_session
from local_functional_source import TASK_A_CONDITIONS
from pilot_task_a_mapping import sample_beta, validate_beta_reference, validate_support
from prepare_role_handoffs import ANATOMY, directory, member_metadata
from registered_endpoint_metrics import CONDITIONS


POINTS = 208
PLAN_ROWS = len(DEVELOPMENT_IDS) * sum(EXPECTED_BETAS.values())
INTEGER_FIELDS = (
    "run", "reg_id", "instruction", "common", "half", "geometry_partition",
    "condition_map_included", "regressors_in_condition_run",
    "within_run_weight_numerator", "within_run_weight_denominator",
)


@dataclass(frozen=True)
class DevelopmentReleaseBinding:
    """Explicit paths/headers supplied by the deployed reader, never proof of it.

anatomy_headers and task_a_header are the existing participant header records.
archive_path identifies that participant's released development/<pid>.zip.
No default source location or release-authority flag exists.
"""
    participant_id: str
    archive_path: object
    anatomy_headers: object
    task_a_header: object


@dataclass(frozen=True)
class DevelopmentArrays:
    participants: tuple
    task_a_packets: tuple
    observations: tuple
    voxel_ids: tuple
    report: object
    # Populated only by the live released-handoff reader. Array-only callers
    # still provide their own independently qualified header to the panel.
    qualified_forward_header: object = None
    support_selection: object = None


def _integer(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("plan integers cannot be booleans")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value == str(int(value)):
        return int(value)
    raise ValueError("literal integer plan fields are required")


def _voxel_ids(values, points=POINTS):
    values = tuple(values)
    if (not 0 < points <= POINTS or len(values) != points or len(set(values)) != points
            or any(isinstance(value, (bool, np.bool_)) or
                   not isinstance(value, (int, np.integer)) for value in values)):
        raise ValueError("the complete ordered 208 integer voxel identities are required")
    return tuple(map(int, values))


def _readonly(values):
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite mapped data; no participant or support removal")
    result.flags.writeable = False
    return result


def _compile_plan(member_rows, summaries, check):
    """Validate the existing development-filtered plan before any sample call."""
    if set(summaries) != set(SESSIONS):
        raise ValueError("both original condition-multiplicity summaries are required")
    grouped = {(pid, session): [] for pid in DEVELOPMENT_IDS for session in SESSIONS}
    seen = set()
    for number, original in enumerate(member_rows):
        check()
        if number >= PLAN_ROWS:
            raise ValueError("development plan exceeds the fixed complete member count")
        row = dict(original)
        for field in INTEGER_FIELDS:
            row[field] = _integer(row[field])
        pid, session = row["participant_id"], row["session"]
        run, reg = row["run"], row["reg_id"]
        key = (pid, session, run, reg)
        if ((pid, session) not in grouped or key in seen or not 1 <= run <= 16 or reg < 0
                or row["participant_role"] != "development"
                or row["planned_handoff"] != "development"
                or row["task_set"] != SESSIONS[session]
                or row["instruction"] not in (0, 1) or row["common"] not in (0, 1)
                or bool(row["instruction"]) != (row["task_code"] == "instrct")
                or row["half"] != (1 if run <= 8 else 2)
                or row["geometry_partition"] != (run - 1) // 4 + 1
                or row["beta_member"] != beta_member(pid, session, run, reg)):
            raise ValueError("development member identity, role, partition or duplicate mismatch")
        seen.add(key)
        grouped[pid, session].append(row)
    if len(seen) != PLAN_ROWS:
        raise ValueError("all twelve development participants and original beta rows are required")
    selected, all_members = {}, {}
    for pid in DEVELOPMENT_IDS:
        all_members[pid] = set()
        for session in SESSIONS:
            rows = sorted(grouped[pid, session], key=lambda row: (row["run"], row["reg_id"]))
            if not rows:
                raise ValueError("all twelve complete development participant sessions are required")
            counts = Counter(row["beta_member"] for row in rows)
            summary = validate_session(rows, counts, EXPECTED_BETAS[session])
            if summary != summaries[session]:
                raise ValueError("condition identity or multiplicity differs from the existing source summary")
            roster = TASK_A_CONDITIONS if session == "ses-s1" else CONDITIONS
            if (session == "ses-s1" and tuple(sorted(summary)) != TASK_A_CONDITIONS
                    or session == "ses-s2" and
                    (len(summary) != 32 or set(summary) - set(TASK_A_CONDITIONS) != set(CONDITIONS)
                     or len(set(summary) & set(TASK_A_CONDITIONS)) != 14)):
                raise ValueError("the canonical 29 A, 18 B-only and 14 shared conditions are required")
            keep = []
            for row in rows:
                all_members[pid].add(row["beta_member"])
                instruction = bool(row["instruction"])
                primary = session == "ses-s2" and row["cond_name"] in CONDITIONS and not instruction
                use = ("instruction_excluded" if instruction else "Task_A_calibration" if session == "ses-s1"
                       else "Task_B_primary" if primary else "Task_B_shared_postdecision_diagnostic")
                count = 0 if instruction else summary[row["cond_name"]]["regressors_per_run"][row["run"] - 1]
                if (row["endpoint_use"] != use or row["condition_map_included"] != int(not instruction)
                        or row["regressors_in_condition_run"] != count
                        or row["within_run_weight_numerator"] != int(not instruction)
                        or row["within_run_weight_denominator"] != (count if count else 1)):
                    raise ValueError("the frozen endpoint labels and within-run arithmetic weights are required")
                if not instruction and row["cond_name"] in roster:
                    keep.append(MappingProxyType(row))
            selected[pid, session] = tuple(keep)
    return selected, all_members


def _assemble(plan, voxel_ids, sample, check):
    points = len(voxel_ids)
    participants, packets, observations = [], [], []
    completed = 0
    for pid in DEVELOPMENT_IDS:
        run_maps = {}
        for session, conditions in (("ses-s1", TASK_A_CONDITIONS), ("ses-s2", CONDITIONS)):
            lookup = {name: index for index, name in enumerate(conditions)}
            sums = np.zeros((16, len(conditions), points), dtype=np.float64)
            counts = np.zeros((16, len(conditions)), dtype=np.int64)
            for row in plan[pid, session]:
                check()
                raw = np.asarray(sample(pid, row))
                if raw.shape != (points,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
                    raise ValueError("mapped beta must retain every finite fixed-support location")
                slot = row["run"] - 1, lookup[row["cond_name"]]
                sums[slot] += np.asarray(raw, dtype=np.float64)
                counts[slot] += 1
                completed += 1
            if np.any(counts == 0):
                raise ValueError("incomplete run/condition support; no imputation")
            run_maps[session] = _readonly(sums / counts[..., None])
        a_halves = _readonly(np.stack((run_maps["ses-s1"][:8].mean(axis=0),
                                      run_maps["ses-s1"][8:].mean(axis=0))))
        a_mean = _readonly(a_halves.mean(axis=0))
        b_parts = _readonly(np.stack([run_maps["ses-s2"][start:start + 4].mean(axis=0)
                                     for start in (0, 4, 8, 12)]))
        participants.append(DevelopmentParticipant(pid, TASK_A_CONDITIONS, voxel_ids, a_mean, b_parts))
        packets.append(TaskACalibrationPacket(pid, TASK_A_CONDITIONS, voxel_ids,
                                             a_mean, a_halves, TASK_A_HALF_RUNS))
        observations.append(TaskBObservation(pid, CONDITIONS, voxel_ids, b_parts))
    check()
    report = MappingProxyType({
        "participants": len(participants), "support_locations": points,
        "Task_A_conditions": len(TASK_A_CONDITIONS), "Task_B_only_conditions": len(CONDITIONS),
        "mapped_beta_members": completed, "arithmetic": "equal_regressors_within_run_then_equal_runs",
        "Task_A_half_runs": TASK_A_HALF_RUNS, "Task_B_partition_runs": tuple(tuple(range(k, k + 4)) for k in (1, 5, 9, 13)),
        "Task_B_shared_or_instruction_values_read": False, "audit_values_read": False,
        "independent_run_errors_established": False, "reader_isolation_established_by_adapter": False,
        "fitting_or_scoring_performed": False,
    })
    return DevelopmentArrays(tuple(participants), tuple(packets), tuple(observations), voxel_ids, report)


def assemble_development_arrays(member_rows, *, condition_multiplicity_by_session,
                                voxel_ids, sample, check):
    """Assemble supplied mapped values; callback restrictions confer no authority.

The callback receives (participant_id, validated_row), only for A or B-primary
members. Every complete metadata check precedes its first invocation. Failure
returns no partial cohort and does not write any artifact.
"""
    if not callable(sample) or not callable(check):
        raise TypeError("explicit sample and budget callbacks are required")
    ids = _voxel_ids(voxel_ids)
    plan, _ = _compile_plan(member_rows, condition_multiplicity_by_session, check)
    return _assemble(plan, ids, sample, check)


class _SelectedArchive:
    """Exact stream restriction inside an externally isolated reader."""
    def __init__(self, archive, selected):
        self.archive, self.selected = archive, frozenset(selected)

    def getinfo(self, member):
        if member not in self.selected:
            raise ValueError("stream is outside selected development anatomy/A/B-primary members")
        return self.archive.getinfo(member)

    def open(self, member):
        name = member.filename if hasattr(member, "filename") else member
        self.getinfo(name)
        return self.archive.open(member)


def _header(archive, member):
    info = archive.getinfo(member)
    member_metadata(info)
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError("selected image exceeds the existing 32MiB image bound")
    with archive.open(info) as stream:
        raw = stream.read(348)
    header = nifti_header(io.BytesIO(raw))
    # These are exactly the existing read_image representation checks, moved
    # ahead of every participant's voxel decoding. The same live348 bytes
    # supply bitpix/scaling; no beta payload or second header read is needed.
    endian = "<" if struct.unpack_from("<i", raw)[0] == 348 else ">"
    itemsize = {2: 1, 4: 2, 16: 4}.get(header["datatype"])
    if itemsize is None:
        raise ValueError("unsupported selected-image datatype")
    if struct.unpack_from(endian + "h", raw, 72)[0] != itemsize * 8:
        raise ValueError("selected-image datatype and bitpix disagree")
    slope, intercept = struct.unpack_from(endian + "2f", raw, 112)
    if slope != 0 and not math.isnan(slope) and (not math.isfinite(slope) or not math.isfinite(intercept)):
        raise ValueError("undefined selected-image intensity scaling")
    offset = header["voxel_offset"]
    if not math.isfinite(offset) or offset != int(offset) or not 352 <= offset <= 1024 * 1024:
        raise ValueError("unsupported selected-image payload offset")
    if int(offset) + math.prod(header["shape"]) * itemsize > info.file_size:
        raise ValueError("truncated selected-image payload size")
    return header


def _native_coordinates(archive, forward_member, forward_header, indices):
    forward = read_image(archive, forward_member, forward_header)
    world = scale_values(forward, forward["values"][indices[:, 0], indices[:, 1], indices[:, 2], 0, :])
    if world.shape != (len(indices), 3) or not np.isfinite(world).all():
        raise ValueError("finite native-world pull coordinates are required on every fixed point")
    return world


def read_development_handoffs(*, release_bindings, member_rows,
                              condition_multiplicity_by_session, common_support, check,
                              derived_support=None):
    """Decode only after externally authorized release and enforced isolation.

Bindings must cover the twelve fixed development people in order. Header
records and common_support are the existing metadata/anatomy products supplied
by the caller, not locations to discover through mixed archives. This function
neither inspects unrelated directories nor launches a process or creates files.
The caller owns runtime/storage bounds through check(); no default budget,
release flag, CLI, or procedural substitute for isolation is provided.
"""
    if not callable(check):
        raise TypeError("an explicit budget callback is required")
    bindings = tuple(release_bindings)
    if (any(type(binding) is not DevelopmentReleaseBinding for binding in bindings)
            or tuple(binding.participant_id for binding in bindings) != DEVELOPMENT_IDS):
        raise ValueError("explicit release bindings for all twelve development participants are required")
    # Snapshot metadata before opening any container; caller changes cannot
    # switch the represented support/header convention midway through a read.
    bindings = tuple(DevelopmentReleaseBinding(value.participant_id, Path(value.archive_path),
                     deepcopy(value.anatomy_headers), deepcopy(value.task_a_header)) for value in bindings)
    if (len({value.archive_path.resolve() for value in bindings}) != len(DEVELOPMENT_IDS)
            or any(value.archive_path.name != value.participant_id + ".zip" for value in bindings)):
        raise ValueError("distinct explicit participant development ZIP paths are required")
    plan, all_members = _compile_plan(member_rows, condition_multiplicity_by_session, check)
    support = {name: np.array(value, copy=True) for name, value in common_support.items()}
    selection = (None if derived_support is None else
                 {name: np.array(value, copy=True) for name, value in derived_support.items()})
    reference = bindings[0].anatomy_headers["space-SUIT_xfm.nii"]
    with ExitStack() as stack:
        prepared = {}
        for binding in bindings:
            check()
            pid, headers = binding.participant_id, binding.anatomy_headers
            if any(headers["space-SUIT_xfm.nii"][key] != reference[key] for key in GRID_KEYS):
                raise ValueError("every forward header must preserve the same frozen support grid")
            validate_beta_reference(binding.task_a_header, headers["desc-cereb_mask.nii"])
            archive = stack.enter_context(zipfile.ZipFile(binding.archive_path))
            infos = directory(archive)
            anatomy = {f"{pid}/anat/{pid}_{suffix}" for suffix in ANATOMY}
            if set(infos) != all_members[pid] | anatomy:
                raise ValueError("container is not the exact complete development participant handoff")
            for info in infos.values():
                member_metadata(info)
            forward_member = f"{pid}/anat/{pid}_space-SUIT_xfm.nii"
            selected = {row["beta_member"] for session in SESSIONS for row in plan[pid, session]}
            checked = _SelectedArchive(archive, selected | {forward_member})
            forward_header = _header(checked, forward_member)
            for key in GRID_KEYS + ("datatype", "voxel_offset"):
                if forward_header[key] != headers["space-SUIT_xfm.nii"][key]:
                    raise ValueError("live forward header differs from existing anatomy metadata")
            selected_sform(forward_header)
            beta_headers = {}
            for session in SESSIONS:
                for row in plan[pid, session]:
                    check()
                    header = _header(checked, row["beta_member"])
                    validate_beta_reference(header, headers["desc-cereb_mask.nii"])
                    if session == "ses-s1" and any(header[key] != binding.task_a_header[key]
                            for key in GRID_KEYS + ("datatype", "voxel_offset")):
                        raise ValueError("live A header differs from the existing reference")
                    beta_headers[row["beta_member"]] = header
            prepared[pid] = checked, headers["desc-cereb_mask.nii"], forward_member, forward_header, beta_headers
        # Historical anatomy records omit units. Every live header above was
        # compared with all recorded grid fields and independently required to
        # use millimetres; only that qualified live reference can validate the
        # fixed support. Do not infer units or amend the historical records.
        live_reference = prepared[DEVELOPMENT_IDS[0]][3]
        from finite_development_support import select_support
        _, indices, ids = select_support(support, live_reference, selection)
        # Monotonic encoding of the existing lexicographic source-grid axis,
        # not a participant-native voxel index or a new support.
        _voxel_ids(ids, len(indices))
        # All twelve directories and selected headers are now known; voxel
        # decoding is sequential and returns no cohort if any person fails.
        current_pid, native_world = None, None

        def sample(pid, row):
            nonlocal current_pid, native_world
            archive, mask_header, forward_member, forward_header, beta_headers = prepared[pid]
            if current_pid != pid:
                check()
                native_world = _native_coordinates(archive, forward_member, forward_header, indices)
                current_pid = pid
            values, _ = sample_beta(archive, row["beta_member"], beta_headers[row["beta_member"]],
                                    mask_header, native_world)
            return values

        return replace(_assemble(plan, ids, sample, check),
                       qualified_forward_header=deepcopy(live_reference),
                       support_selection=selection)
