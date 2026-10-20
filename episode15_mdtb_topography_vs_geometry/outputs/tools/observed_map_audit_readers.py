"""Inactive, separately gated A-calibration and B-evaluation audit readers.

Source preparation does not adopt the proposed audit. Every reader requires
its recorded policy adoption before inspecting supplied audit metadata or
opening a container. B decoding additionally requires the complete frozen
prediction bundle. The boundary is procedural within the same account.
"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np

from build_anatomical_support import GRID_KEYS, selected_sform
from calibration_falsifiers import TASK_A_HALF_RUNS, TaskACalibrationPacket, TaskBObservation
from development_handoff_reader import (
    INTEGER_FIELDS, _SelectedArchive, _header, _integer, _native_coordinates, _readonly,
)
from finite_development_support import load_selection, select_support
from join_source_metadata import EXPECTED_BETAS, SESSIONS, beta_member, validate_session
from local_functional_source import TASK_A_CONDITIONS
from observed_map_audit_context import AUDIT_IDS, require_adoption, validate_prediction_bundle
from pilot_task_a_mapping import sample_beta, validate_beta_reference
from prepare_role_handoffs import ANATOMY, directory, member_metadata
from registered_endpoint_metrics import CONDITIONS


FROZEN_SUPPORT_PATH = (Path(__file__).resolve().parents[1]
    / "runs/observed_beta_development/finite_support001/selection.npz")


@dataclass(frozen=True)
class AuditCalibrationBinding:
    participant_id: str
    archive_path: object
    anatomy_headers: object
    task_a_header: object


@dataclass(frozen=True)
class AuditObservationBinding:
    participant_id: str
    archive_path: object


@dataclass(frozen=True)
class AuditCalibrationData:
    task_a_packets: tuple
    voxel_ids: tuple
    qualified_forward_header: object
    support_selection: object
    sampling_native_world: np.ndarray
    sampling_mask_headers: tuple


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bindings(values, cls):
    values = tuple(values)
    _require(all(type(value) is cls for value in values)
             and tuple(value.participant_id for value in values) == AUDIT_IDS,
             "all12_original_ordered_audit_bindings_required")
    paths = tuple(Path(value.archive_path) for value in values)
    _require(all(path.is_absolute() and path.name == person + ".zip" for path, person in zip(paths, AUDIT_IDS))
             and len({path.resolve() for path in paths}) == 12,
             "distinct_explicit_role_filtered_audit_archives_required")
    return values


def _compile_plan(member_rows, *, session, summary, check):
    """The existing run/regressor arithmetic, with one explicit audit role."""
    handoff = "audit_calibration" if session == "ses-s1" else "sealed_audit"
    grouped = {person: [] for person in AUDIT_IDS}
    seen = set()
    maximum = 12 * EXPECTED_BETAS[session]
    for number, original in enumerate(member_rows):
        check()
        _require(number < maximum, "audit_plan_exceeds_complete_session_count")
        row = dict(original)
        for field in INTEGER_FIELDS:
            row[field] = _integer(row[field])
        person, run, reg = row["participant_id"], row["run"], row["reg_id"]
        key = (person, run, reg)
        _require(person in grouped and row["session"] == session and key not in seen
                 and 1 <= run <= 16 and reg >= 0 and row["participant_role"] == "audit"
                 and row["planned_handoff"] == handoff and row["task_set"] == SESSIONS[session]
                 and row["instruction"] in (0, 1) and row["common"] in (0, 1)
                 and bool(row["instruction"]) == (row["task_code"] == "instrct")
                 and row["half"] == (1 if run <= 8 else 2)
                 and row["geometry_partition"] == (run - 1) // 4 + 1
                 and row["beta_member"] == beta_member(person, session, run, reg),
                 "audit_member_identity_role_or_partition_mismatch")
        seen.add(key)
        grouped[person].append(row)
    _require(len(seen) == maximum, "all12_complete_audit_sessions_required")
    selected, all_members = {}, {}
    conditions = TASK_A_CONDITIONS if session == "ses-s1" else CONDITIONS
    for person in AUDIT_IDS:
        rows = sorted(grouped[person], key=lambda row: (row["run"], row["reg_id"]))
        counts = Counter(row["beta_member"] for row in rows)
        actual = validate_session(rows, counts, EXPECTED_BETAS[session])
        _require(actual == summary, "audit_condition_multiplicity_differs_from_frozen_source")
        _require((session == "ses-s1" and tuple(sorted(actual)) == TASK_A_CONDITIONS)
                 or (session == "ses-s2" and len(actual) == 32
                     and set(actual) - set(TASK_A_CONDITIONS) == set(CONDITIONS)
                     and len(set(actual) & set(TASK_A_CONDITIONS)) == 14),
                 "unchanged29A_18Bonly_and14shared_conditions_required")
        keep = []
        for row in rows:
            instruction = bool(row["instruction"])
            primary = session == "ses-s2" and row["cond_name"] in CONDITIONS and not instruction
            use = ("instruction_excluded" if instruction else "Task_A_calibration" if session == "ses-s1"
                   else "Task_B_primary" if primary else "Task_B_shared_postdecision_diagnostic")
            count = 0 if instruction else actual[row["cond_name"]]["regressors_per_run"][row["run"] - 1]
            _require(row["endpoint_use"] == use and row["condition_map_included"] == int(not instruction)
                     and row["regressors_in_condition_run"] == count
                     and row["within_run_weight_numerator"] == int(not instruction)
                     and row["within_run_weight_denominator"] == (count if count else 1),
                     "unchanged_condition_roles_and_arithmetic_weights_required")
            if not instruction and row["cond_name"] in conditions:
                keep.append(row)
        selected[person] = tuple(keep)
        all_members[person] = set(counts)
    return selected, all_members


def _run_maps(rows, conditions, sample, check):
    lookup = {name: index for index, name in enumerate(conditions)}
    sums = np.zeros((16, len(conditions), 188), dtype=np.float64)
    counts = np.zeros((16, len(conditions)), dtype=np.int64)
    for row in rows:
        check()
        values = np.asarray(sample(row))
        _require(values.shape == (188,) and values.dtype.kind in "fiu" and np.isfinite(values).all(),
                 "every_fixed_audit_location_must_be_finite_no_support_reselection")
        slot = row["run"] - 1, lookup[row["cond_name"]]
        sums[slot] += values
        counts[slot] += 1
    _require(np.all(counts > 0), "complete_run_condition_support_required_no_imputation")
    maps = sums / counts[..., None]
    _require(np.isfinite(maps).all(), "finite_complete_audit_run_maps_required")
    return maps


def read_audit_calibration(*, policy, release_bindings, member_rows,
                           condition_multiplicity_by_session, common_support,
                           derived_support, check):
    require_adoption(policy)
    _require(callable(check), "explicit_resource_callback_required")
    bindings = _bindings(release_bindings, AuditCalibrationBinding)
    bindings = tuple(AuditCalibrationBinding(value.participant_id, Path(value.archive_path),
                     deepcopy(value.anatomy_headers), deepcopy(value.task_a_header)) for value in bindings)
    plan, all_members = _compile_plan(member_rows, session="ses-s1",
        summary=condition_multiplicity_by_session["ses-s1"], check=check)
    _require(derived_support is not None, "development_frozen188_selection_required")
    selection = {key: np.array(value, copy=True) for key, value in derived_support.items()}
    original_selection = load_selection(FROZEN_SUPPORT_PATH)
    _require(set(selection) == set(original_selection)
             and all(np.array_equal(selection[key], original_selection[key]) for key in selection),
             "selection_must_equal_the_existing_development_frozen_mask")
    reference = bindings[0].anatomy_headers["space-SUIT_xfm.nii"]
    prepared, packets, native_world, masks = {}, [], [], []
    with ExitStack() as stack:
        for binding in bindings:
            check()
            person, headers = binding.participant_id, binding.anatomy_headers
            _require(all(headers["space-SUIT_xfm.nii"][key] == reference[key] for key in GRID_KEYS),
                     "audit_forward_grid_must_match_fixed_common_support")
            validate_beta_reference(binding.task_a_header, headers["desc-cereb_mask.nii"])
            archive = stack.enter_context(zipfile.ZipFile(binding.archive_path))
            infos = directory(archive)
            anatomy = {f"{person}/anat/{person}_{suffix}" for suffix in ANATOMY}
            _require(set(infos) == all_members[person] | anatomy,
                     "exact_A_only_audit_calibration_container_required")
            for info in infos.values():
                member_metadata(info)
            forward_member = f"{person}/anat/{person}_space-SUIT_xfm.nii"
            allowed = {row["beta_member"] for row in plan[person]} | {forward_member}
            checked = _SelectedArchive(archive, allowed)
            forward = _header(checked, forward_member)
            _require(all(forward[key] == headers["space-SUIT_xfm.nii"][key]
                         for key in GRID_KEYS + ("datatype", "voxel_offset")),
                     "audit_forward_header_changed")
            selected_sform(forward)
            beta_headers = {}
            for row in plan[person]:
                check()
                header = _header(checked, row["beta_member"])
                validate_beta_reference(header, headers["desc-cereb_mask.nii"])
                _require(all(header[key] == binding.task_a_header[key]
                             for key in GRID_KEYS + ("datatype", "voxel_offset")),
                         "audit_A_header_changed")
                beta_headers[row["beta_member"]] = header
            prepared[person] = checked, forward_member, forward, beta_headers, headers["desc-cereb_mask.nii"]
        live_reference = prepared[AUDIT_IDS[0]][2]
        _, indices, ids = select_support(common_support, live_reference, selection)
        _require(len(ids) == 188, "original_frozen188_support_required")
        for person in AUDIT_IDS:
            checked, forward_member, forward, beta_headers, mask = prepared[person]
            world = _native_coordinates(checked, forward_member, forward, indices)
            def sample(row):
                return sample_beta(checked, row["beta_member"], beta_headers[row["beta_member"]], mask, world)[0]
            run_maps = _run_maps(plan[person], TASK_A_CONDITIONS, sample, check)
            halves = _readonly(np.stack((run_maps[:8].mean(axis=0), run_maps[8:].mean(axis=0))))
            packets.append(TaskACalibrationPacket(person, TASK_A_CONDITIONS, ids,
                           _readonly(halves.mean(axis=0)), halves, TASK_A_HALF_RUNS))
            native_world.append(np.array(world, copy=True))
            masks.append(deepcopy(mask))
    check()
    return AuditCalibrationData(tuple(packets), ids, deepcopy(live_reference), selection,
                                _readonly(np.stack(native_world)), tuple(masks))


def read_audit_observations(*, policy, frozen_predictions, release_bindings,
                            member_rows, condition_multiplicity_by_session, check):
    require_adoption(policy)
    frozen = validate_prediction_bundle(frozen_predictions, policy=policy, require_sampling_geometry=True)
    selection = load_selection(FROZEN_SUPPORT_PATH)
    _require(tuple(frozen["voxel_ids"]) == tuple(map(int, selection["selected_voxel_ids"])),
             "prediction_axis_must_equal_existing_development_frozen188_axis")
    _require(callable(check), "explicit_resource_callback_required")
    bindings = _bindings(release_bindings, AuditObservationBinding)
    plan, all_members = _compile_plan(member_rows, session="ses-s2",
        summary=condition_multiplicity_by_session["ses-s2"], check=check)
    observations = []
    with ExitStack() as stack:
        prepared = {}
        for index, binding in enumerate(bindings):
            check()
            archive = stack.enter_context(zipfile.ZipFile(binding.archive_path))
            infos = directory(archive)
            _require(set(infos) == all_members[binding.participant_id], "exact_sealed_B_only_container_required")
            for info in infos.values():
                member_metadata(info)
            checked = _SelectedArchive(archive, {row["beta_member"] for row in plan[binding.participant_id]})
            mask = frozen["sampling_mask_headers"][index]
            beta_headers = {}
            for row in plan[binding.participant_id]:
                check()
                header = _header(checked, row["beta_member"])
                validate_beta_reference(header, mask)
                beta_headers[row["beta_member"]] = header
            prepared[binding.participant_id] = checked, mask, beta_headers
        for index, person in enumerate(AUDIT_IDS):
            checked, mask, headers = prepared[person]
            world = frozen["sampling_native_world"][index]
            def sample(row):
                return sample_beta(checked, row["beta_member"], headers[row["beta_member"]], mask, world)[0]
            runs = _run_maps(plan[person], CONDITIONS, sample, check)
            partitions = _readonly(np.stack([runs[start:start + 4].mean(axis=0) for start in (0, 4, 8, 12)]))
            observations.append(TaskBObservation(person, CONDITIONS, tuple(frozen["voxel_ids"]), partitions))
    check()
    return tuple(observations)
