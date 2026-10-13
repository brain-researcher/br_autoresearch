"""Prospective EP15 Task-A correspondence and identity controls.

Array-only fitting with explicit A-only packets, then a separate B scoring
interface. One fixed draw per control; no reader, empirical execution, formal
permutation inference or additional resource authority. Callback restrictions
are data-flow checks, not an operating-system access boundary.
"""
from dataclasses import dataclass

import numpy as np

from development_lopo import (
    DEVELOPMENT_IDS, M1_VARIANTS, M2_VARIANTS, M3_VARIANTS, M4_VARIANTS,
    CalibrationPanel, CalibrationReference, FoldContext, PairedIdentityContext,
    _array, _shared, _support_size, fit_calibration_panel, validate_source_context,
)
from local_functional_source import TASK_A_CONDITIONS
from registered_endpoint_metrics import (
    CONDITIONS, crossvalidated_geometries, geometry_covariance, map_covariance,
    project_gain_direction,
)


TASK_A_HALF_RUNS = (tuple(range(1, 9)), tuple(range(9, 17)))
VARIANT_FIELDS = (("m1_predictions", M1_VARIANTS), ("m2_predictions", M2_VARIANTS),
                  ("m3_variant_predictions", M3_VARIANTS), ("m4_predictions", M4_VARIANTS))
VARIANTS = M1_VARIANTS+M2_VARIANTS+M3_VARIANTS+M4_VARIANTS


@dataclass(frozen=True)
class TaskACalibrationPacket:
    participant_id: str
    task_a_condition_ids: tuple[str, ...]
    voxel_ids: tuple[str | int, ...]
    task_a_maps: np.ndarray
    task_a_halves: np.ndarray
    half_run_ids: tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class TaskBObservation:
    participant_id: str
    task_b_condition_ids: tuple[str, ...]
    voxel_ids: tuple[str | int, ...]
    task_b_partitions: np.ndarray


@dataclass(frozen=True)
class CalibrationControlFold:
    context: FoldContext | PairedIdentityContext
    source: object
    true_panel: CalibrationPanel
    control_panel: CalibrationPanel
    true_packet_id: str
    control_packet_id: str
    diagnostics: dict


@dataclass(frozen=True)
class CalibrationFalsifierRun:
    participant_ids: tuple[str, ...]
    deleted_id: str | None
    condition_folds: tuple[CalibrationControlFold, ...]
    identity_folds: tuple[CalibrationControlFold, ...]


def _retained_ids(deleted_id):
    if deleted_id not in (None, *DEVELOPMENT_IDS):
        raise ValueError("only one explicit development participant deletion is permitted")
    return tuple(item for item in DEVELOPMENT_IDS if item != deleted_id)


def _voxel_axis(voxel_ids, *, support_size=208):
    support_size = _support_size(support_size)
    strings = all(isinstance(item, str) and bool(item) for item in voxel_ids)
    integers = all(isinstance(item, (int, np.integer))
                   and not isinstance(item, (bool, np.bool_)) for item in voxel_ids)
    if (not isinstance(voxel_ids, tuple) or len(voxel_ids) != support_size
            or not (strings or integers) or len(set(voxel_ids)) != support_size):
        raise ValueError("the complete frozen selected location axis is required")


def _packet_metadata(packets, expected_ids, voxel_ids, *, support_size=208):
    """Complete identity/axis validation precedes ANY array conversion."""
    _voxel_axis(voxel_ids, support_size=support_size)
    packets = tuple(packets)
    if (any(type(packet) is not TaskACalibrationPacket for packet in packets)
            or tuple(packet.participant_id for packet in packets) != expected_ids):
        raise ValueError("explicit A-only packets must preserve exact ordered participant identities")
    for packet in packets:
        if (packet.task_a_condition_ids != TASK_A_CONDITIONS
                or packet.voxel_ids != voxel_ids or packet.half_run_ids != TASK_A_HALF_RUNS):
            raise ValueError("Task-A packets require canonical axes and genuine fixed run halves")
    return packets


def validate_task_a_packets(packets, expected_ids, voxel_ids, *, numerical_atol, support_size=208):
    """Validate supplied genuine halves; never synthesize them from a mean.

    A packet cannot authenticate its own upstream provenance. The eventual
    admitted reader must supply the actual disjoint-run maps and these labels.
    """
    if numerical_atol != 1e-10:
        raise ValueError("the existing fixed 1e-10 numerical tolerance is required")
    support_size = _support_size(support_size)
    packets = _packet_metadata(packets, expected_ids, voxel_ids, support_size=support_size)
    result = []
    for packet in packets:
        mean = _array(packet.task_a_maps, (29, support_size), "Task-A mean")
        halves = _array(packet.task_a_halves, (2, 29, support_size), "genuine Task-A halves")
        if np.max(np.abs(halves.mean(axis=0)-mean)) > numerical_atol:
            raise ValueError("Task-A halves must reproduce the frozen arithmetic mean")
        result.append(TaskACalibrationPacket(packet.participant_id, TASK_A_CONDITIONS,
                                             voxel_ids, mean, halves, TASK_A_HALF_RUNS))
    return tuple(result)


def condition_relabeling(training_packets, context, voxel_ids, *, numerical_atol, support_size=208):
    """Return canonical-destination -> moved-source row indices and diagnostics."""
    if type(context) is not FoldContext:
        raise ValueError("condition relabeling uses its ordinary LOPO source")
    validate_source_context(context)
    packets = validate_task_a_packets(training_packets, context.training_ids, voxel_ids,
                                     numerical_atol=numerical_atol, support_size=support_size)
    rest = TASK_A_CONDITIONS.index("rest")
    nonrest = tuple(i for i in range(29) if i != rest)
    correlations = []
    for packet in packets:
        halves = packet.task_a_halves[:, list(nonrest)]
        centered = halves-halves.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("undefined source Task-A half correlation; control unresolved")
        correlation = np.sum((centered[0]/norms[0, :, None])
                             *(centered[1]/norms[1, :, None]), axis=-1)
        if not np.isfinite(correlation).all():
            raise ValueError("undefined source Task-A half correlation; control unresolved")
        correlations.append(correlation)
    # The fixed rest singleton is never ranked or permuted. A constant rest
    # map needs no correlation statistic; record None, not an imputed number.
    reliability = dict(zip(nonrest, np.median(correlations, axis=0)))
    priority = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([20261007, 1501]))).permutation(nonrest)
    priority_rank = {int(item): index for index, item in enumerate(priority)}
    ranked = sorted(nonrest, key=lambda i: (reliability[i], i))
    mapping = list(range(29))
    strata = []
    ranges = []
    for start in range(0, 28, 7):
        members = ranked[start:start+7]
        cycle = sorted(members, key=priority_rank.__getitem__)
        for index, destination in enumerate(cycle):
            mapping[destination] = cycle[(index+1) % len(cycle)]
        strata.append(tuple(TASK_A_CONDITIONS[index] for index in members))
        ranges.append((float(reliability[members[0]]), float(reliability[members[-1]])))
    return tuple(mapping), {
        "destination_condition_ids": TASK_A_CONDITIONS,
        "source_condition_ids": tuple(TASK_A_CONDITIONS[index] for index in mapping),
        "source_training_ids": context.training_ids,
        "source_median_half_correlations": tuple(
            None if index == rest else float(reliability[index]) for index in range(29)),
        "strata": tuple(strata), "stratum_reliability_ranges": tuple(ranges),
        "fixed_condition": "rest", "rest_reliability": "not_used", "draws": 1,
        "seed_sequence": (20261007, 1501),
        "matches_exact_target_reliability": False, "permutation_inference": False,
    }


def relabel_task_a_packet(packet, mapping, *, support_size=208):
    """Move mean AND half rows; canonical destination labels stay fixed."""
    support_size = _support_size(support_size)
    _voxel_axis(packet.voxel_ids, support_size=support_size)
    if (len(mapping) != 29 or any(type(index) is not int for index in mapping)
            or set(mapping) != set(range(29)) or mapping[28] != 28
            or any(mapping[index] == index for index in range(28))):
        raise ValueError("the condition control requires a non-rest derangement and fixed rest")
    return TaskACalibrationPacket(
        packet.participant_id, packet.task_a_condition_ids, packet.voxel_ids,
        _array(packet.task_a_maps[list(mapping)], (29, support_size), "relabeled Task-A mean"),
        _array(packet.task_a_halves[:, list(mapping)], (2, 29, support_size), "relabeled Task-A halves"),
        packet.half_run_ids,
    )


def identity_pairings(deleted_id=None):
    """Remove a deletion from one fixed whole-cohort cycle; never redraw."""
    retained = _retained_ids(deleted_id)
    indices = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([20261007, 1502]))).permutation(len(DEVELOPMENT_IDS))
    cycle = tuple(DEVELOPMENT_IDS[int(index)] for index in indices
                  if DEVELOPMENT_IDS[int(index)] in retained)
    successors = {person: cycle[(index+1) % len(cycle)] for index, person in enumerate(cycle)}
    return tuple((person, successors[person]) for person in retained)


def _panel_predictions(panel):
    if not isinstance(panel, CalibrationPanel):
        raise ValueError("an explicit fitted calibration panel is required")
    values = []
    for field, names in VARIANT_FIELDS:
        predictions = getattr(panel, field)
        if tuple(value.name for value in predictions) != names:
            raise ValueError("all sixteen fixed variants must be present in canonical order")
        values.extend(predictions)
    return tuple(values)


def fit_calibration_falsifiers(packets, source_records, ordinary_references, *, deleted_id,
                               fit_shared, calibrate_m3, certify_m3, predict_m3_geometry,
                               predict_m1, predict_m2, predict_m3_variants, predict_m4, check,
                               support_size=208):
    """Fit one complete twelve-target run or one explicitly selected deletion.

    ordinary_references contains only A-only CalibrationReference objects,
    not FoldPrediction objects carrying recipient B. Source records may carry
    development B, but only the exact excluded training tuple reaches fitting.
    The condition control reuses each true source/panel. Identity fits one new
    paired source, then correct and donor A on that same source. All sixteen
    variants use the existing common gain refitted for their packet.

    No nested deletions run automatically. To prepare those later, provide
    their already refitted ordinary references and select deleted_id explicitly.
    """
    support_size = _support_size(support_size)
    retained = _retained_ids(deleted_id)
    callbacks = dict(calibrate_m3=calibrate_m3, certify_m3=certify_m3,
                     predict_m3_geometry=predict_m3_geometry, predict_m1=predict_m1,
                     predict_m2=predict_m2, predict_m3_variants=predict_m3_variants,
                     predict_m4=predict_m4, check=check)
    if not all(callable(callback) for callback in (fit_shared, *callbacks.values())):
        raise TypeError("the existing source and complete sixteen-variant callbacks are required")
    callbacks["support_size"] = support_size
    raw_packets, records, references = tuple(packets), tuple(source_records), tuple(ordinary_references)
    if (any(type(packet) is not TaskACalibrationPacket for packet in raw_packets)
            or tuple(packet.participant_id for packet in raw_packets) != DEVELOPMENT_IDS
            or tuple(record.participant_id for record in records) != DEVELOPMENT_IDS):
        raise ValueError("all twelve canonical development identities must be explicit")
    voxel_ids = raw_packets[0].voxel_ids
    _packet_metadata(raw_packets, DEVELOPMENT_IDS, voxel_ids, support_size=support_size)
    # Check all metadata before converting even one included A array. A deleted
    # person's arrays and every recipient/donor B remain unopened here.
    for record in records:
        if (record.task_a_condition_ids != TASK_A_CONDITIONS
                or record.voxel_ids != voxel_ids or record.task_b_condition_ids != CONDITIONS):
            raise ValueError("source records must preserve canonical condition and location axes")
    if (any(type(reference) is not CalibrationReference for reference in references)
            or tuple(reference.context.target_id for reference in references) != retained):
        raise ValueError("complete ordered ordinary A-only calibration references are required")
    for reference in references:
        context = reference.context
        if type(context) is not FoldContext or context.deleted_id != deleted_id:
            raise ValueError("ordinary references must match this exact deletion context")
        validate_source_context(context)
        if (reference.source.training_ids != context.training_ids
                or reference.source.task_a_condition_ids != TASK_A_CONDITIONS
                or reference.source.voxel_ids != voxel_ids or reference.source.numerical_atol != 1e-10):
            raise ValueError("ordinary source must retain the exact fixed context, axes and tolerance")
        _panel_predictions(reference.panel)
    validated = validate_task_a_packets(
        tuple(packet for packet in raw_packets if packet.participant_id != deleted_id),
        retained, voxel_ids, numerical_atol=1e-10, support_size=support_size,
    )
    by_id = {packet.participant_id: packet for packet in validated}
    record_by_id = {record.participant_id: record for record in records}
    for person in retained:
        source_a = _array(record_by_id[person].task_a_maps, (29, support_size), "source-record Task A")
        if np.max(np.abs(source_a-by_id[person].task_a_maps)) > 1e-10:
            raise ValueError("source records and calibration packets must share the frozen A mean")
    condition_folds, identity_folds = [], []
    donor_by_id = dict(identity_pairings(deleted_id))
    for reference in references:
        check()
        context, source, true_panel = reference.context, reference.source, reference.panel
        target = by_id[context.target_id]
        _shared(source, context, target)
        reference_a = _array(reference.task_a_maps, (29, support_size), "ordinary fitted Task A")
        if np.max(np.abs(reference_a-target.task_a_maps)) > 1e-10:
            raise ValueError("the reused true panel must have fitted this unchanged target A mean")
        mapping, diagnostics = condition_relabeling(
            tuple(by_id[person] for person in context.training_ids), context, voxel_ids,
            numerical_atol=source.numerical_atol, support_size=support_size,
        )
        relabeled = relabel_task_a_packet(target, mapping, support_size=support_size)
        condition_panel = fit_calibration_panel(source, relabeled.task_a_maps, context, **callbacks)
        _panel_predictions(condition_panel)
        condition_folds.append(CalibrationControlFold(
            context, source, true_panel, condition_panel,
            target.participant_id, target.participant_id, diagnostics,
        ))
        donor_id = donor_by_id[target.participant_id]
        training_ids = tuple(person for person in DEVELOPMENT_IDS
                             if person not in (target.participant_id, donor_id, deleted_id))
        paired = PairedIdentityContext(target.participant_id, training_ids, deleted_id, donor_id)
        validate_source_context(paired)
        check()
        paired_source = fit_shared(tuple(record_by_id[person] for person in training_ids), paired)
        _shared(paired_source, paired, target)
        if paired_source.numerical_atol != 1e-10:
            raise ValueError("paired source must retain the frozen numerical tolerance")
        paired_true = fit_calibration_panel(paired_source, target.task_a_maps, paired, **callbacks)
        paired_donor = fit_calibration_panel(paired_source, by_id[donor_id].task_a_maps, paired, **callbacks)
        _panel_predictions(paired_true)
        _panel_predictions(paired_donor)
        identity_folds.append(CalibrationControlFold(
            paired, paired_source, paired_true, paired_donor,
            target.participant_id, donor_id,
            {"source_training_ids": training_ids, "draws": 1,
             "seed_sequence": (20261007, 1502), "pairings": tuple(donor_by_id.items()),
             "estimand": "donor_minus_true_within_paired_reduced_source",
             "individual_anatomy_tested": False, "permutation_inference": False},
        ))
    check()
    return CalibrationFalsifierRun(retained, deleted_id, tuple(condition_folds), tuple(identity_folds))


def exploratory_falsifier_points(run, observations, *, support_size=208):
    """Signed R_control - R_true, separately for each source-matched design.

    Observed B enters only here, after all calibration fits. Conditions, models,
    donors and shared sources do not increase biological N. M3's gain-projected
    geometry contrast is algebraically zero; numerical noise is not evidence.
    """
    support_size = _support_size(support_size)
    retained = _retained_ids(run.deleted_id)
    if run.participant_ids != retained:
        raise ValueError("all retained canonical participants are required")
    observations = tuple(observations)
    if (any(type(value) is not TaskBObservation for value in observations)
            or tuple(value.participant_id for value in observations) != retained):
        raise ValueError("complete ordered observed-B packets are required only at scoring")
    for folds, paired in ((run.condition_folds, False), (run.identity_folds, True)):
        if len(folds) != len(retained) or tuple(fold.context.target_id for fold in folds) != retained:
            raise ValueError("all control targets must be retained in order")
        for fold, observation in zip(folds, observations):
            validate_source_context(fold.context)
            if (type(fold.context) is not (PairedIdentityContext if paired else FoldContext)
                    or fold.context.deleted_id != run.deleted_id
                    or fold.source.training_ids != fold.context.training_ids
                    or fold.true_packet_id != fold.context.target_id
                    or fold.control_packet_id != (fold.context.donor_id if paired else fold.context.target_id)
                    or observation.task_b_condition_ids != CONDITIONS
                    or observation.voxel_ids != fold.source.voxel_ids):
                raise ValueError("control/source/observation identities must preserve the paired design")
            if paired and fold.context.donor_id != dict(identity_pairings(run.deleted_id))[fold.context.target_id]:
                raise ValueError("the fixed recipient/donor cycle must be retained")
            _panel_predictions(fold.true_panel)
            _panel_predictions(fold.control_panel)
    observed = np.stack([_array(value.task_b_partitions, (4, 18, support_size), "observed Task B")
                         for value in observations])
    halves = (observed[:, (0, 2)]+observed[:, (1, 3)])/2
    geometries = crossvalidated_geometries(observed)
    result = {"participant_ids": retained, "deleted_id": run.deleted_id,
              "biological_n": len(retained), "draws_per_control": 1,
              "permutation_inference": False, "M3_projected_geometry_contrast_algebraically_zero": True}
    for label, folds in (("condition", run.condition_folds), ("identity", run.identity_folds)):
        G = np.stack([fold.source.G for fold in folds])
        K0 = np.stack([fold.source.K0 for fold in folds])
        observed_map = halves-G[:, None]
        observed_geometry = project_gain_direction(geometries-K0[:, None], K0[:, None])
        panels = {"true": [_panel_predictions(fold.true_panel) for fold in folds],
                  "control": [_panel_predictions(fold.control_panel) for fold in folds]}
        entries = {}
        for index, name in enumerate(VARIANTS):
            estimates = {}
            for role, values in panels.items():
                maps = np.stack([_array(value[index].maps, (18, support_size), "control maps") for value in values])
                matrices = np.stack([_array(value[index].geometry, (18, 18), "control geometry")
                                     for value in values])
                if not np.array_equal(matrices, matrices.swapaxes(-1, -2)):
                    raise ValueError("predicted geometries must retain signed symmetric entries")
                prediction = maps-G
                geometry_prediction = project_gain_direction(matrices-K0, K0)
                estimates[role] = {
                    "R_map": map_covariance(observed_map[:, 0]-prediction, observed_map[:, 1]-prediction),
                    "R_geometry": geometry_covariance(observed_geometry[:, 0]-geometry_prediction,
                                                       observed_geometry[:, 1]-geometry_prediction),
                }
            entries[name] = {**estimates, "control_minus_true": {
                key: estimates["control"][key]-estimates["true"][key] for key in ("R_map", "R_geometry")}}
        result[label] = {"source_training_counts": tuple(len(fold.context.training_ids) for fold in folds),
                         "variants": entries,
                         "estimand": ("donor_minus_true_within_paired_reduced_source" if label == "identity"
                                      else "relabeled_minus_true_within_ordinary_source")}
    return result
