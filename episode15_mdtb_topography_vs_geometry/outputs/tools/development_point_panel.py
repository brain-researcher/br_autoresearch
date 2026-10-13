"""Connect the existing EP15 development arrays and fixed point-only panel.

This library accepts caller-supplied arrays and anatomy metadata only. It does
not read paths, decode a handoff, deploy isolation, grant release, write results,
or select an empirical allocation. Real inputs still require the existing
enforced development reader and release; in-memory callbacks are not that
boundary. Audit participants and shared-B diagnostics are outside this API.

All twelve ordinary targets and all sixteen fixed variants are retained. The
existing condition/identity, random-basis and matched-warp controls are wired
without new draws or model definitions. There are 12 ordinary and 12 paired
source fits, and 48 complete calibration panels; the 132 participant-deletion
refits are not run. Native signatures, calibrated uncertainty and formal class
decisions remain incomplete. No ratio, threshold or winning class is inferred.

The caller supplies check() for allocation/deadline checks around finite
components. Existing bounded numerical solvers retain their own limits; check
does not interrupt an ongoing solver. Outer CPU/memory/time/storage admission
and the outcome-blind cost pilot remain the coordinator's responsibility.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from anatomical_group_baseline import AnatomicalGroupBaseline
from calibration_falsifiers import (
    TASK_A_HALF_RUNS, TaskACalibrationPacket, TaskBObservation,
    exploratory_falsifier_points, fit_calibration_falsifiers, validate_task_a_packets,
)
from development_handoff_reader import DevelopmentArrays
from development_lopo import (
    DEVELOPMENT_IDS, DevelopmentParticipant, _array,
    exploratory_points, fit_development_lopo,
)
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS
from local_isometry_variants import LocalIsometryVariants
from local_parcel_prediction import LocalParcelPrediction
from m2_alignment_signature import m2_alignment_points
from matched_random_basis import MatchedRandomLocalBasis
from matched_random_warp import MatchedRandomWarpPrediction
from paired_stretch_variants import PairedStretchVariants
from finite_development_support import select_support
from registered_endpoint_metrics import CONDITIONS
from smooth_relocation import SmoothRelocationPrediction


@dataclass(frozen=True)
class DevelopmentPointPanel:
    """Completed supplied-array fits and their existing signed point summaries.

    Frozen field bindings do not make the existing nested fit records deeply
    immutable. Retain these as one result; they grant no new data access.
    """
    ordinary_fit: object
    ordinary_points: object
    calibration_falsifier_fit: object
    calibration_falsifier_points: object
    m2_alignment_points: object


def _joined_arrays(development, common_support, forward_header, check):
    if type(development) is not DevelopmentArrays:
        raise TypeError("the existing complete DevelopmentArrays handoff is required")
    participants = tuple(development.participants)
    packets = tuple(development.task_a_packets)
    observations = tuple(development.observations)
    for records, cls in ((participants, DevelopmentParticipant), (packets, TaskACalibrationPacket),
                         (observations, TaskBObservation)):
        if (any(type(value) is not cls for value in records)
                or tuple(value.participant_id for value in records) != DEVELOPMENT_IDS):
            raise ValueError("all three views must retain the twelve ordered development participants")
    header = deepcopy(forward_header)
    support, indices, ids = select_support(common_support, header, development.support_selection)
    support_size = len(ids)
    if development.voxel_ids != ids:
        raise ValueError("handoff voxel IDs must match the complete frozen common-support axis")
    # Check all identities before any response conversion, then snapshot every
    # supplied view before fitting. No scoring array can drift independently
    # from the copy that supplied the participant's training/prediction record.
    for record, packet, observed in zip(participants, packets, observations):
        check()
        if (record.voxel_ids != ids or packet.voxel_ids != ids or observed.voxel_ids != ids
                or record.task_a_condition_ids != TASK_A_CONDITIONS
                or packet.task_a_condition_ids != TASK_A_CONDITIONS
                or packet.half_run_ids != TASK_A_HALF_RUNS
                or record.task_b_condition_ids != CONDITIONS
                or observed.task_b_condition_ids != CONDITIONS):
            raise ValueError("participant, genuine-A and observed-B views must preserve exact axes")
    packets = validate_task_a_packets(packets, DEVELOPMENT_IDS, ids, numerical_atol=1e-10,
                                     support_size=support_size)
    copied, joined_observations = [], []
    for record, packet, observed in zip(participants, packets, observations):
        check()
        task_a = _array(record.task_a_maps, (29, support_size), "participant Task A")
        task_b = _array(record.task_b_partitions, (4, 18, support_size), "participant Task B")
        observed_b = _array(observed.task_b_partitions, (4, 18, support_size), "observed Task B")
        if not np.array_equal(task_a, packet.task_a_maps):
            raise ValueError("participant and genuine-half packet must contain the same Task-A mean")
        if not np.array_equal(task_b, observed_b):
            raise ValueError("source records and scoring observations must contain the same Task-B partitions")
        copied.append(DevelopmentParticipant(record.participant_id, TASK_A_CONDITIONS, ids,
                                            packet.task_a_maps, task_b))
        joined_observations.append(TaskBObservation(record.participant_id, CONDITIONS, ids, task_b))
    coordinates = _array(support["world_xyz_mm"], (support_size, 3), "common-space coordinates")
    check()
    return tuple(copied), packets, tuple(joined_observations), coordinates, ids


def _fixed_callbacks(coordinates, voxel_ids, check):
    """Instantiate only the already selected source/model/control definitions."""
    check()
    tolerances = BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                                 rank_rtol=1e-8, numerical_atol=1e-10)
    anatomy = build_local_anatomical_basis(coordinates, np.asarray(voxel_ids),
        tolerances=tolerances, split_depth=3, modes_per_cell=2, support_size=len(voxel_ids))
    check()
    source = LocalFunctionalSource(anatomy, numerical_atol=1e-10, rank_rtol=1e-8)
    check()
    parcel = LocalParcelPrediction(source, anatomy)
    check()
    relocation = SmoothRelocationPrediction(source, anatomy)
    check()
    isometry = LocalIsometryVariants(source, anatomy, check=check)
    check()
    stretch = PairedStretchVariants(source, anatomy, check=check)
    check()
    random_bases = MatchedRandomLocalBasis(source, anatomy, check=check)
    check()
    matched_warps = MatchedRandomWarpPrediction(relocation, check=check)
    anatomical = AnatomicalGroupBaseline(voxel_ids, check=check, support_size=len(voxel_ids))
    check()
    calibration = dict(calibrate_m3=source.calibrate_m3, certify_m3=source.certify_m3,
        predict_m3_geometry=source.predict_m3_geometry, predict_m1=parcel,
        predict_m2=relocation, predict_m3_variants=isometry, predict_m4=stretch)
    controls = dict(fit_m0a=anatomical, predict_random_bases=random_bases,
                    predict_matched_warps=matched_warps)
    return source.fit_shared, calibration, controls


def fit_development_point_panel(development, *, common_support, forward_header, check):
    """Fit the complete fixed point panel, then score the same supplied cohort.

    No empirical access is conferred by providing arrays. common_support and
    forward_header are the original caller-supplied anatomy products. An
    adopted derived support enters only through development.support_selection,
    with exact original/subset identities checked. All identity/value joins precede the
    first fit. A failed component/deadline propagates without a partial result,
    retry, participant deletion, subset fallback, or early scientific score.
    """
    if not callable(check):
        raise TypeError("an explicit allocation/deadline callback is required")
    check()
    records, packets, observations, coordinates, voxel_ids = _joined_arrays(
        development, common_support, forward_header, check)
    fit_shared, calibration, controls = _fixed_callbacks(coordinates, voxel_ids, check)
    check()
    ordinary = fit_development_lopo(records, fit_shared=fit_shared, **calibration, **controls,
        check=check, include_participant_deletions=False, support_size=len(voxel_ids))
    check()
    references = tuple(fold.calibration_reference for fold in ordinary.full.folds)
    falsifiers = fit_calibration_falsifiers(packets, records, references, deleted_id=None,
        fit_shared=fit_shared, **calibration, check=check, support_size=len(voxel_ids))
    check()
    points = exploratory_points(ordinary.full)
    check()
    falsifier_points = exploratory_falsifier_points(falsifiers, observations, support_size=len(voxel_ids))
    check()
    alignment = m2_alignment_points(ordinary.full, support_size=len(voxel_ids))
    check()
    return DevelopmentPointPanel(ordinary, points, falsifiers, falsifier_points, alignment)
