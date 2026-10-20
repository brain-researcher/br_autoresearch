"""Inactive all-development-source audit prediction implementation.

No file reader, policy mutation, scoring, or audit-B argument exists here.
The explicit amendment must be adopted before supplied arrays are inspected.
Legacy development guards remain intact. Narrow audit adapters below retain
their numerical kernels, tolerances, fixed controls and genuine participant
seed identities while replacing only development-specific context admission.
"""
import numpy as np

from calibration_falsifiers import (
    TaskACalibrationPacket, VARIANTS, _panel_predictions, relabel_task_a_packet,
    validate_task_a_packets,
)
from development_lopo import (
    DEVELOPMENT_IDS, M1_VARIANTS, M2_VARIANTS, M3_VARIANTS, M4_VARIANTS,
    M1Prediction, M2Prediction, M3Prediction, M4Prediction, CalibrationPanel,
    MatchedWarpPrediction, RandomBasisPrediction, _array,
)
from development_point_panel import _joined_arrays
from local_anatomical_basis import BasisTolerances, build_local_anatomical_basis
from local_functional_source import LocalFunctionalSource, TASK_A_CONDITIONS, _frozen
from local_isometry_variants import LocalIsometryVariants
from local_parcel_prediction import LocalParcelPrediction
from matched_random_basis import MatchedRandomLocalBasis
from matched_random_warp import (
    DRAW_COUNT, MASTER_SEED, MAX_PROPOSALS_PER_DRAW, _geometry, _haar_rotation,
)
from observed_map_audit_context import (
    AUDIT_IDS, ROUTE_ID, AuditContext, PARTITION_RUN_IDS, audit_identity_pairings,
    require_adoption, validate_audit_context, validate_prediction_bundle,
)
from paired_stretch_variants import PairedStretchVariants
from proposed_development_point_panel import ProposedPairedStretchVariants
from registered_endpoint_metrics import CONDITIONS, crossvalidated_geometries
from smooth_relocation import SmoothRelocationPrediction, VARIANTS as WARP_VARIANTS
from spatial_deformation_operators import apply_m3


class _AuditFunctionalSource(LocalFunctionalSource):
    """Unchanged source estimator with a distinct, explicit audit-only context."""
    def __init__(self, anatomy, *, policy):
        require_adoption(policy)
        super().__init__(anatomy, numerical_atol=1e-10, rank_rtol=1e-8)

    def _context(self, context):
        validate_audit_context(context)


def _callbacks(coordinates, voxel_ids, *, policy, check):
    tolerances = BasisTolerances(eigen_tie_atol=1e-10, eigen_tie_rtol=1e-10,
                                 rank_rtol=1e-8, numerical_atol=1e-10)
    check()
    anatomy = build_local_anatomical_basis(coordinates, np.asarray(voxel_ids),
        tolerances=tolerances, split_depth=3, modes_per_cell=2, support_size=188)
    check()
    source = _AuditFunctionalSource(anatomy, policy=policy)
    parcel = LocalParcelPrediction(source, anatomy)
    relocation = SmoothRelocationPrediction(source, anatomy)
    isometry = LocalIsometryVariants(source, anatomy, check=check)
    stretch = ProposedPairedStretchVariants(PairedStretchVariants(source, anatomy, check=check))
    random_bases = MatchedRandomLocalBasis(source, anatomy, check=check)
    check()
    return source, (parcel, relocation, isometry, stretch), random_bases


def _calibration_panel(source_recipe, source, target_task_a, context, callbacks, check):
    """Audit counterpart of fit_calibration_panel; same kernels/certificates."""
    validate_audit_context(context)
    source_recipe._source(source, context)
    target_task_a = _array(target_task_a, (29, 188), "audit calibration Task A")
    check()
    numerical = dict(source_recipe.calibrate_m3(source, target_task_a, context))
    numerical["O"] = _array(numerical["O"], (16, 16), "calibrated audit O")
    gain = np.asarray(numerical["gain"])
    if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
        raise ValueError("one positive finite pre-operator common gain required")
    numerical["gain"] = float(gain)
    source_recipe.certify_m3(source, numerical, context)
    check()
    centered = source.G - source.G.mean(axis=-1, keepdims=True)
    prediction = _array(numerical["gain"] * apply_m3(centered, source.B, numerical["O"]),
                        (18, 188), "audit M3 prediction")
    geometry = _array(source_recipe.predict_m3_geometry(source, numerical, context),
                      (18, 18), "audit M3 geometry")
    if not np.array_equal(geometry, geometry.T):
        raise ValueError("M3 geometry must retain identical signed symmetric entries")
    groups = []
    for callback, names, prediction_type, label in zip(callbacks,
            (M1_VARIANTS, M2_VARIANTS, M3_VARIANTS, M4_VARIANTS),
            (M1Prediction, M2Prediction, M3Prediction, M4Prediction), ("M1", "M2", "M3", "M4")):
        check()
        calibration = numerical if label in ("M3", "M4") else numerical["gain"]
        supplied = tuple(callback(source, target_task_a, calibration, context))
        if (len(supplied) != 4 or tuple(value.name for value in supplied) != names
                or any(not isinstance(value, prediction_type) for value in supplied)):
            raise ValueError("all four fixed " + label + " variants required in canonical order")
        validated = []
        for value in supplied:
            maps = _array(value.maps, (18, 188), label + " audit prediction")
            matrix = _array(value.geometry, (18, 18), label + " audit geometry")
            if not np.array_equal(matrix, matrix.T):
                raise ValueError(label + " geometry must retain identical signed symmetric entries")
            validated.append(prediction_type(value.name, maps, matrix, value.diagnostics))
        groups.append(tuple(validated))
    if (np.max(np.abs(groups[2][0].maps - prediction)) > source.numerical_atol
            or np.max(np.abs(groups[2][0].geometry - geometry)) > source.numerical_atol):
        raise ValueError("M3 variant reference must reproduce the paired reference prediction")
    return CalibrationPanel(numerical, prediction, geometry, *groups)


def _condition_mapping(training_packets, voxel_ids):
    """Original reliability strata/seed using all twelve development A packets."""
    packets = validate_task_a_packets(training_packets, DEVELOPMENT_IDS, voxel_ids,
                                     numerical_atol=1e-10, support_size=188)
    rest = TASK_A_CONDITIONS.index("rest")
    nonrest = tuple(index for index in range(29) if index != rest)
    correlations = []
    for packet in packets:
        halves = packet.task_a_halves[:, list(nonrest)]
        centered = halves - halves.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(centered, axis=-1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("undefined development Task-A half correlation; control unresolved")
        correlation = np.sum((centered[0] / norms[0, :, None])
                             * (centered[1] / norms[1, :, None]), axis=-1)
        if not np.isfinite(correlation).all():
            raise ValueError("undefined development Task-A half correlation; control unresolved")
        correlations.append(correlation)
    reliability = dict(zip(nonrest, np.median(correlations, axis=0)))
    priority = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([20261007, 1501]))).permutation(nonrest)
    priority_rank = {int(item): index for index, item in enumerate(priority)}
    ranked = sorted(nonrest, key=lambda index: (reliability[index], index))
    mapping = list(range(29))
    for start in range(0, 28, 7):
        cycle = sorted(ranked[start:start+7], key=priority_rank.__getitem__)
        for index, destination in enumerate(cycle):
            mapping[destination] = cycle[(index + 1) % len(cycle)]
    return tuple(mapping)


def _matched_warps(predictor, source, predictions, common_gain, context, check):
    """Original 8 SO(3) draws, using each actual audit participant's seed ID."""
    validate_audit_context(context)
    predictor.source_recipe._source(source, context)
    if (tuple(value.name for value in predictions) != M2_VARIANTS
            or any(not isinstance(value, M2Prediction) for value in predictions)):
        raise ValueError("all four unchanged M2 predictions required")
    results = []
    for field, fitted in zip(predictor.fields, predictions):
        diagnostics = fitted.diagnostics
        if (diagnostics.get("training_ids") != DEVELOPMENT_IDS
                or diagnostics.get("variant") != field.variant.name
                or diagnostics.get("common_gain") != common_gain
                or not np.array_equal(predictor.representation.coordinates, field.coordinates)):
            raise ValueError("matched control requires its original source, field and common gain")
        T = _array(diagnostics["coefficients"], (8, 3), "Task-A-fitted field coefficients")
        check()
        _geometry(predictor.representation, field, T)
        parent_norm = float(np.linalg.norm(T))
        variant_index = WARP_VARIANTS.index(field.variant)
        for draw_index in range(DRAW_COUNT):
            seed = (MASTER_SEED, int(context.target_id.removeprefix("sub-")), variant_index, draw_index)
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            for proposal in range(1, MAX_PROPOSALS_PER_DRAW + 1):
                check()
                rotation = _haar_rotation(rng)
                rotated = T @ rotation
                norm_error = abs(float(np.linalg.norm(rotated)) - parent_norm)
                gram_error = float(np.max(np.abs(rotated @ rotated.T - T @ T.T)))
                tolerance = 1e-10 * max(1., parent_norm**2)
                if norm_error > tolerance or gram_error > tolerance:
                    raise ArithmeticError("random rotation broke exact field matching")
                try:
                    certificate = _geometry(predictor.representation, field, rotated)
                except (ValueError, ArithmeticError):
                    continue
                partitions = predictor.predict_partitions(field,
                    source.geometry_state.source_partitions, rotated, common_gain)
                results.append(MatchedWarpPrediction(field.variant.name, draw_index,
                    _frozen(partitions.mean(axis=0)),
                    _frozen(crossvalidated_geometries(partitions[None])[0].mean(axis=0)),
                    dict(certificate, seed_entropy=seed, accepted_proposal=proposal,
                         rejected_proposals=proposal-1, training_ids=DEVELOPMENT_IDS,
                         target_id=context.target_id, deleted_id=context.deleted_id,
                         common_gain=common_gain, coefficients=_frozen(rotated), rotation=_frozen(rotation),
                         coefficient_norm_matching_error_mm=norm_error,
                         coefficient_gram_matching_max_abs=gram_error,
                         degenerate_identity_control=not np.any(T),
                         proposal_law="Haar_SO3_conditioned_on_existing_geometric_certificates",
                         conditional_direction_falsifier=True, omnibus_random_field_null=False,
                         condition_geometry_preserved=False, exact_phi_jacobian_determinant_matched=False,
                         native_signature_certified=False, class_adequacy_claimed=False,
                         empirical_execution_authorized=False, draw_count=DRAW_COUNT,
                         aggregation="mean_of_per_draw_statistics")))
                break
            else:
                raise ArithmeticError(f"matched audit warp unresolved: {field.variant.name} draw {draw_index}; no shrink or rescue")
    expected = tuple((name, draw) for name in M2_VARIANTS for draw in range(8))
    if tuple((value.variant, value.draw_index) for value in results) != expected:
        raise ValueError("every fixed matched warp and draw required")
    return tuple(results)


def _certificate_summary(panel):
    """Keep actual scalar solver/certificate decisions, not parameter caches."""
    fields = ("numerical_certificate_method", "solution_within_frozen_gap",
              "unique_global_minimum_sufficient_profile", "full_domain_positive_curvature_asserted",
              "numerical_solver", "numerical_solver_converged", "local_optimizer_converged",
              "optimizer_iterations", "optimizer_ftol", "optimizer_max_iterations",
              "global_optimum_certified", "unique_solution_certified")
    result = []
    for value in _panel_predictions(panel):
        entry = {"variant": value.name}
        for key in fields:
            if key in value.diagnostics:
                supplied = value.diagnostics[key]
                if isinstance(supplied, np.generic):
                    supplied = supplied.item()
                if supplied is None or isinstance(supplied, (str, bool, int, float)):
                    entry[key] = supplied
        result.append(entry)
    return result


def fit_audit_predictions(development, audit_packets, *, common_support, forward_header, policy, check):
    """Produce the complete A-only panel; no empirical authority is inferred."""
    require_adoption(policy)
    if not callable(check):
        raise TypeError("an allocation/deadline callback is required")
    check()
    records, training_packets, _, coordinates, voxel_ids = _joined_arrays(
        development, common_support, forward_header, check)
    if len(voxel_ids) != 188 or development.support_selection is None:
        raise ValueError("the existing frozen 188-location development support is required")
    packets = validate_task_a_packets(tuple(audit_packets), AUDIT_IDS, voxel_ids,
                                     numerical_atol=1e-10, support_size=188)
    source_recipe, callbacks, random_bases = _callbacks(coordinates, voxel_ids, policy=policy, check=check)
    # Source fitting consumes only the original twelve development records.
    context = AuditContext(AUDIT_IDS[0])
    source = source_recipe.fit_shared(records, context)
    source_recipe._source(source, context)
    raw_partitions = np.stack([record.task_b_partitions for record in records])
    centered = raw_partitions - raw_partitions.mean(axis=-1, keepdims=True)
    m0a = _array(centered.mean(axis=0).mean(axis=0), (18, 188), "all-development anatomical M0a")
    mapping = _condition_mapping(training_packets, voxel_ids)
    donor_by_id, by_id = dict(audit_identity_pairings()), {packet.participant_id: packet for packet in packets}
    ordinary, condition, donor, bases, warps = [], [], [], [], []
    certificates = []
    for target in packets:
        check()
        context = AuditContext(target.participant_id)
        true = _calibration_panel(source_recipe, source, target.task_a_maps, context, callbacks, check)
        relabeled = relabel_task_a_packet(target, mapping, support_size=188)
        shuffled = _calibration_panel(source_recipe, source, relabeled.task_a_maps, context, callbacks, check)
        donor_id = donor_by_id[target.participant_id]
        paired = AuditContext(target.participant_id, donor_id=donor_id)
        transferred = _calibration_panel(source_recipe, source, by_id[donor_id].task_a_maps,
                                         paired, callbacks, check)
        ordinary.append(np.stack([value.maps for value in _panel_predictions(true)]))
        condition.append(np.stack([value.maps for value in _panel_predictions(shuffled)]))
        donor.append(np.stack([value.maps for value in _panel_predictions(transferred)]))
        supplied_bases = tuple(random_bases(source, target.task_a_maps, true.numerical_fit, context))
        if (tuple(value.draw_index for value in supplied_bases) != tuple(range(8))
                or any(not isinstance(value, RandomBasisPrediction) for value in supplied_bases)):
            raise ValueError("all eight unchanged random basis draws required")
        bases.append(np.stack([_array(value.maps, (18, 188), "random basis audit prediction")
                               for value in supplied_bases]))
        supplied_warps = _matched_warps(callbacks[1], source, true.m2_predictions,
                                        true.numerical_fit["gain"], context, check)
        warps.append(np.stack([_array(value.maps, (18, 188), "matched warp audit prediction")
                               for value in supplied_warps]).reshape(4, 8, 18, 188))
        certificates.append({"participant_id": target.participant_id,
            "ordinary": _certificate_summary(true), "condition": _certificate_summary(shuffled),
            "donor": _certificate_summary(transferred), "random_basis_draws_completed": 8,
            "matched_warp_seeds": [value.diagnostics["seed_entropy"] for value in supplied_warps],
            "matched_warp_accepted_proposals": [value.diagnostics["accepted_proposal"] for value in supplied_warps],
            "matched_warp_degenerate_identity_control": [bool(value.diagnostics["degenerate_identity_control"])
                                                         for value in supplied_warps],
            "matched_warp_control_interpretation": {
                "proposal_law": "Haar_SO3_conditioned_on_existing_geometric_certificates",
                "conditional_direction_falsifier": True, "omnibus_random_field_null": False,
                "condition_geometry_preserved": False, "exact_phi_jacobian_determinant_matched": False}})
    bundle = {"status": "complete_observed_map_audit_predictions", "route_id": ROUTE_ID,
        "development_ids": DEVELOPMENT_IDS, "audit_ids": AUDIT_IDS, "voxel_ids": voxel_ids,
        "condition_ids": CONDITIONS, "variant_ids": VARIANTS, "partition_run_ids": PARTITION_RUN_IDS,
        "source_training_ids": DEVELOPMENT_IDS, "M0a": m0a, "M0b": source.G,
        "predictions": np.stack(ordinary), "condition_control_predictions": np.stack(condition),
        "donor_control_predictions": np.stack(donor), "random_basis_predictions": np.stack(bases),
        "matched_warp_predictions": np.stack(warps),
        "donor_ids": tuple(donor_by_id[person] for person in AUDIT_IDS),
        "condition_mapping": mapping, "numerical_certificates_complete": True,
        "audit_Task_B_accessed": False, "numerical_certificates": certificates}
    check()
    return validate_prediction_bundle(bundle, policy=policy)
