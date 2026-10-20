"""INACTIVE proposed audit followup: complete strict-sampler availability only.

The actual policy currently has no adoption for this changed spatial estimand.
This code does not add one. Every entry that can inspect audit inputs requires
that separate amendment; the older full188 audit adoption is insufficient.
"""
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np

from build_anatomical_support import read_image, strict_trilinear, world_to_voxel
from calibration_falsifiers import TaskBObservation
from development_handoff_reader import _SelectedArchive, _header
from finite_development_support import load_selection
from observed_map_audit_context import AUDIT_IDS, require_adoption, validate_prediction_bundle
from observed_map_audit_readers import (
    AuditObservationBinding, FROZEN_SUPPORT_PATH, _bindings, _compile_plan, _require,
)
from pilot_task_a_mapping import validate_beta_reference
from prepare_role_handoffs import directory, member_metadata
from registered_endpoint_metrics import CONDITIONS


ADOPTION_KEY = "scientist_amendment_20261020_observed_map_common_support"
SPECIFICATION = "outputs/paper_plan.md#proposed-daily-decision--october-19-2026"
ROUTE_ID = "observed_map_audit_common_support_exploratory"
FULL_SUPPORT = 188


def require_common_adoption(policy):
    """Reject before original prediction loading, audit metadata or any output."""
    require_adoption(policy)
    amendment = policy.get(ADOPTION_KEY) if isinstance(policy, Mapping) else None
    if (not isinstance(amendment, Mapping) or amendment.get("status") != "adopted"
            or amendment.get("specification") != SPECIFICATION
            or amendment.get("route_id") != ROUTE_ID
            or any(not isinstance(amendment.get(key), str) or not amendment[key].strip()
                   for key in ("authority", "adopted_at_utc"))):
        raise PermissionError("common-support exploratory followup is not explicitly adopted")
    return deepcopy(dict(amendment))


@dataclass(frozen=True)
class SupportScan:
    keep: np.ndarray
    observations: tuple
    selected_member_counts: tuple
    original_voxel_ids: tuple


def sample_availability(archive, member, header, mask_header, native_world):
    """Identical interpolation; expose invalidity instead of rejecting the map.

    No zero filling, corner repair, weight renormalization or alternative
    interpolation is used. Malformed images/headers still raise immediately.
    """
    validate_beta_reference(header, mask_header)
    image = read_image(archive, member, header)
    values, valid = strict_trilinear(image, world_to_voxel(image["affine"], native_world))
    return values, valid & np.isfinite(values)


def scan_compiled_plan(plan, *, original_voxel_ids, sample, policy, check):
    """Visit every selected member, including after the intersection is empty.

    `plan` must come from the existing full-session role/multiplicity validator.
    Only availability enters `keep`. Run/partition means are retained in memory
    for later scoring; no prediction, residual or predictive loss enters here.
    """
    require_common_adoption(policy)
    ids = tuple(original_voxel_ids)
    _require(len(ids) == FULL_SUPPORT and len(set(ids)) == FULL_SUPPORT
             and tuple(plan) == AUDIT_IDS and callable(sample) and callable(check),
             "all12_people_and_original188_axis_required")
    lookup = {name: index for index, name in enumerate(CONDITIONS)}
    keep = np.ones(FULL_SUPPORT, dtype=bool)
    observations, member_counts = [], []
    for person in AUDIT_IDS:
        rows = tuple(plan[person])
        _require(bool(rows), "every_participant_requires_selected_B_regressors")
        sums = np.zeros((16, 18, FULL_SUPPORT), dtype=np.float64)
        counts = np.zeros((16, 18), dtype=np.int64)
        seen = set()
        for row in rows:
            check()
            key = row["run"], row["reg_id"]
            _require(row["participant_id"] == person and key not in seen
                     and 1 <= row["run"] <= 16 and row["cond_name"] in lookup,
                     "complete_unique_selected_member_identity_required")
            seen.add(key)
            values, valid = sample(person, row)
            values, valid = np.asarray(values), np.asarray(valid)
            _require(values.shape == valid.shape == (FULL_SUPPORT,)
                     and values.dtype.kind in "fiu" and valid.dtype == np.dtype(bool),
                     "unchanged188_strict_sampler_output_required")
            available = valid & np.isfinite(values)
            keep &= available
            # NaN outside the prospective intersection remains missing, never
            # a measured zero. It cannot affect a later retained location.
            values = np.where(available, values, np.nan)
            slot = row["run"] - 1, lookup[row["cond_name"]]
            sums[slot] += values
            counts[slot] += 1
        _require(len(seen) == len(rows) and np.all(counts > 0),
                 "every_run_condition_and_selected_regressor_required")
        runs = sums / counts[..., None]
        partitions = np.stack([runs[start:start + 4].mean(axis=0) for start in (0, 4, 8, 12)])
        observations.append(TaskBObservation(person, CONDITIONS, ids, partitions))
        member_counts.append(len(seen))
    check()
    keep.setflags(write=False)
    return SupportScan(keep, tuple(observations), tuple(member_counts), ids)


def scan_audit_support(*, policy, frozen_predictions, release_bindings,
                       member_rows, condition_multiplicity_by_session, check):
    require_common_adoption(policy)
    frozen = validate_prediction_bundle(frozen_predictions, policy=policy, require_sampling_geometry=True)
    selection = load_selection(FROZEN_SUPPORT_PATH)
    _require(frozen["voxel_ids"] == tuple(map(int, selection["selected_voxel_ids"])),
             "prediction_axis_must_equal_existing_development_frozen188_axis")
    bindings = _bindings(release_bindings, AuditObservationBinding)
    plan, all_members = _compile_plan(member_rows, session="ses-s2",
        summary=condition_multiplicity_by_session["ses-s2"], check=check)
    with ExitStack() as stack:
        prepared = {}
        for index, binding in enumerate(bindings):
            check()
            archive = stack.enter_context(zipfile.ZipFile(Path(binding.archive_path)))
            infos = directory(archive)
            _require(set(infos) == all_members[binding.participant_id],
                     "exact_sealed_B_only_container_required")
            for info in infos.values():
                member_metadata(info)
            rows = plan[binding.participant_id]
            checked = _SelectedArchive(archive, {row["beta_member"] for row in rows})
            mask = frozen["sampling_mask_headers"][index]
            headers = {}
            for row in rows:
                check()
                header = _header(checked, row["beta_member"])
                validate_beta_reference(header, mask)
                headers[row["beta_member"]] = header
            prepared[binding.participant_id] = checked, mask, headers, frozen["sampling_native_world"][index]

        def sample(person, row):
            checked, mask, headers, world = prepared[person]
            return sample_availability(checked, row["beta_member"], headers[row["beta_member"]], mask, world)

        result = scan_compiled_plan(plan, original_voxel_ids=frozen["voxel_ids"],
            sample=sample, policy=policy, check=check)
    return result  # All audit archives are closed before the runner freezes support.


def validate_support_record(record, original_voxel_ids):
    """One complete frozen availability intersection, never a handpicked list."""
    ids = tuple(original_voxel_ids)
    _require(isinstance(record, Mapping) and record.get("route_id") == ROUTE_ID
             and record.get("status") == "complete_strict_availability_intersection"
             and tuple(record.get("participant_ids", ())) == AUDIT_IDS
             and tuple(record.get("condition_ids", ())) == CONDITIONS
             and tuple(record.get("original_voxel_ids", ())) == ids
             and len(ids) == FULL_SUPPORT and record.get("all_selected_members_scanned") is True
             and record.get("predictive_losses_computed") is False
             and isinstance(record.get("frozen_at_utc"), str) and record["frozen_at_utc"],
             "complete_common_support_freeze_required_before_scoring")
    counts = record.get("selected_member_counts", ())
    _require(len(counts) == 12 and all(type(value) is int and value >= 16 * 18 for value in counts),
             "complete_selected_member_counts_required")
    keep = np.asarray(record.get("keep"))
    _require(keep.shape == (FULL_SUPPORT,) and keep.dtype == np.dtype(bool)
             and tuple(record.get("retained_voxel_ids", ())) == tuple(np.asarray(ids)[keep])
             and record.get("retained_count") == int(keep.sum()),
             "frozen_intersection_and_retained_axis_must_agree")
    return keep
