"""EP15 development M0b/M3 and optional M1--M4 participant-deletion refits.

Array-only orchestration, not a reader or a selected empirical source recipe.
The supplied source fitter must implement the fixed development-only recipe;
the supplied certificate must enforce its fixed locality/capacity requirements.
No numerical defaults, source geometry, basis or certificate are invented here.
Calls have restricted arguments, but Python callbacks/closures are not an OS
access boundary. The deployed reader and published-parameter firewall remain
independent prerequisites. This module cannot release the opaque handoffs.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from registered_endpoint_metrics import (
    CONDITIONS, crossvalidated_geometries, geometry_covariance, geometry_inner,
    map_covariance, project_gain_direction,
)
from spatial_deformation_operators import apply_m3


DEVELOPMENT_IDS = (
    "sub-08", "sub-12", "sub-14", "sub-15", "sub-17", "sub-18",
    "sub-20", "sub-21", "sub-24", "sub-25", "sub-27", "sub-31",
)
M1_VARIANTS = ("M1_reference_probabilistic", "M1_hard_assignment",
               "M1_coarse_granularity", "M1_fine_granularity")
M2_VARIANTS = ("M2_reference_diffeomorphism", "M2_tight_regularity",
               "M2_loose_regularity", "M2_displacement_stress")
M3_VARIANTS = ("M3_reference_local_isometry", "M3_tight_locality",
               "M3_broad_locality", "M3_active_subspace_stress")
M4_VARIANTS = ("M4_anisotropic_stretch", "M4_localized_low_rank_stretch",
               "M4_localized_low_rank_stretch_and_shear", "M4_spectral_bound_stress")
MATCHED_WARP_DRAWS = 8
RANDOM_BASIS_DRAWS = 8


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real with shape {shape}")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite; no participant/condition/location removal")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DevelopmentParticipant:
    participant_id: str
    task_a_condition_ids: tuple[str, ...]
    voxel_ids: tuple[str | int, ...]
    task_a_maps: np.ndarray
    task_b_partitions: np.ndarray  # runs 1:4, 5:8, 9:12, 13:16
    task_b_condition_ids: tuple[str, ...] = CONDITIONS


@dataclass(frozen=True)
class FoldContext:
    target_id: str
    training_ids: tuple[str, ...]
    deleted_id: str | None


@dataclass(frozen=True)
class PairedIdentityContext:
    """A distinct paired-control source, excluding recipient AND donor."""
    target_id: str
    training_ids: tuple[str, ...]
    deleted_id: str | None
    donor_id: str


def validate_source_context(context):
    if type(context) not in (FoldContext, PairedIdentityContext):
        raise ValueError("an explicit ordinary or paired-identity source context is required")
    if (context.target_id not in DEVELOPMENT_IDS
            or context.deleted_id not in (None, *DEVELOPMENT_IDS)
            or context.target_id == context.deleted_id):
        raise ValueError("a valid development target/deletion context is required")
    excluded = (context.target_id, context.deleted_id)
    if type(context) is PairedIdentityContext:
        if context.donor_id not in DEVELOPMENT_IDS or context.donor_id in excluded:
            raise ValueError("the identity donor must be another retained development participant")
        excluded += (context.donor_id,)
    expected = tuple(item for item in DEVELOPMENT_IDS if item not in excluded)
    if context.training_ids != expected:
        raise ValueError("source identities must exactly retain all required participant exclusions")


@dataclass(frozen=True)
class SharedM3Source:
    """One actual source fit, with its own geometry and reference scale.

    K0 is supplied by the source estimator's frozen geometry recipe, never
    reconstructed from G here. In particular a cross-replicate K0 is not
    interchangeable with the Gram matrix of a mean source map.
    """
    training_ids: tuple[str, ...]
    task_a_condition_ids: tuple[str, ...]
    voxel_ids: tuple[str | int, ...]
    task_a_maps: np.ndarray
    G: np.ndarray
    K0: np.ndarray
    S_map: float
    B: np.ndarray
    task_a_weights: np.ndarray
    fit_global_gain: bool
    rank_rtol: float
    numerical_atol: float
    geometry_state: object  # actual source representation owned by its recipe


@dataclass(frozen=True)
class M0aPrediction:
    maps: np.ndarray
    geometry: np.ndarray
    training_ids: tuple[str, ...]


@dataclass(frozen=True)
class M1Prediction:
    name: str
    maps: np.ndarray
    geometry: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class M2Prediction:
    name: str
    maps: np.ndarray
    geometry: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class MatchedWarpPrediction:
    variant: str
    draw_index: int
    maps: np.ndarray
    geometry: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class RandomBasisPrediction:
    draw_index: int
    maps: np.ndarray
    geometry: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class M3Prediction:
    name: str
    maps: np.ndarray
    geometry: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class M4Prediction:
    name: str
    maps: np.ndarray
    geometry: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class CalibrationPanel:
    numerical_fit: dict
    m3_maps: np.ndarray
    m3_geometry: np.ndarray
    m1_predictions: tuple[M1Prediction, ...]
    m2_predictions: tuple[M2Prediction, ...]
    m3_variant_predictions: tuple[M3Prediction, ...]
    m4_predictions: tuple[M4Prediction, ...]


@dataclass(frozen=True)
class CalibrationReference:
    context: FoldContext
    source: SharedM3Source
    panel: CalibrationPanel
    task_a_maps: np.ndarray


@dataclass(frozen=True)
class FoldPrediction:
    context: FoldContext
    G: np.ndarray
    K0: np.ndarray
    S_map: float
    S_geometry: float
    m3_maps: np.ndarray
    m3_geometry: np.ndarray
    O: np.ndarray
    gain: float
    observed_partitions: np.ndarray
    m1_predictions: tuple[M1Prediction, ...] = ()
    m2_predictions: tuple[M2Prediction, ...] = ()
    matched_warp_predictions: tuple[MatchedWarpPrediction, ...] = ()
    m3_variant_predictions: tuple[M3Prediction, ...] = ()
    m4_predictions: tuple[M4Prediction, ...] = ()
    m0a_prediction: M0aPrediction | None = None
    random_basis_predictions: tuple[RandomBasisPrediction, ...] = ()
    calibration_reference: CalibrationReference | None = None


@dataclass(frozen=True)
class LOPORun:
    participant_ids: tuple[str, ...]
    deleted_id: str | None
    folds: tuple[FoldPrediction, ...]


@dataclass(frozen=True)
class DevelopmentLOPO:
    full: LOPORun
    deletions: tuple[LOPORun, ...]


def _support_size(value):
    """Explicit selected spatial size; the unchanged route remains 208."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 1 <= value <= 208):
        raise ValueError("support_size must be an explicit integer in [1, 208]")
    return int(value)


def _participants(records, *, support_size=208):
    support_size = _support_size(support_size)
    records = tuple(records)
    by_id = {record.participant_id: record for record in records}
    if len(records) != 12 or len(by_id) != 12 or set(by_id) != set(DEVELOPMENT_IDS):
        raise ValueError("exactly the twelve assigned development participants are required")
    ordered = tuple(by_id[participant] for participant in DEVELOPMENT_IDS)
    condition_ids, voxel_ids = ordered[0].task_a_condition_ids, ordered[0].voxel_ids
    if (not isinstance(condition_ids, tuple) or not condition_ids
            or any(not isinstance(item, str) or not item for item in condition_ids)
            or len(set(condition_ids)) != len(condition_ids)):
        raise ValueError("an explicit unique ordered Task-A condition axis is required")
    strings = all(isinstance(item, str) and bool(item) for item in voxel_ids)
    integers = all(isinstance(item, (int, np.integer))
                   and not isinstance(item, (bool, np.bool_)) for item in voxel_ids)
    if (not isinstance(voxel_ids, tuple) or len(voxel_ids) != support_size
            or not (strings or integers) or len(set(voxel_ids)) != support_size):
        raise ValueError("the complete selected voxel identities are required")
    validated = []
    for record in ordered:
        if (record.task_a_condition_ids != condition_ids or record.voxel_ids != voxel_ids
                or record.task_b_condition_ids != CONDITIONS):
            raise ValueError("all participants must preserve exact common condition/voxel identities")
        validated.append(DevelopmentParticipant(
            record.participant_id, condition_ids, voxel_ids,
            _array(record.task_a_maps, (len(condition_ids), support_size), "Task A"),
            _array(record.task_b_partitions, (4, 18, support_size), "Task B partitions"),
        ))
    return tuple(validated)


def _shared(source, context, target):
    support_size = _support_size(len(target.voxel_ids))
    if not isinstance(source, SharedM3Source):
        raise TypeError("source fitter must return its explicit SharedM3Source")
    if (source.training_ids != context.training_ids
            or source.task_a_condition_ids != target.task_a_condition_ids
            or source.voxel_ids != target.voxel_ids):
        raise ValueError("shared source must retain exact training/condition/voxel identities")
    source_a = _array(source.task_a_maps, target.task_a_maps.shape, "source Task A")
    G = _array(source.G, (18, support_size), "G")
    K0 = _array(source.K0, (18, 18), "K0")
    if not np.array_equal(K0, K0.T):
        raise ValueError("K0 must retain identical signed symmetric entries")
    s_geometry = float(geometry_inner(K0, K0))
    scale = np.asarray(source.S_map)
    if (scale.shape != () or scale.dtype.kind not in "fiu"
            or not np.isfinite(scale) or scale <= 0 or s_geometry <= 0):
        raise ValueError("each training-only source scale must be positive and finite")
    # The explicit calibrator/certificate own B, rank, weights, common gain
    # rule, locality and all recipe-specific numerical settings.
    return source_a, G, K0, float(scale), s_geometry


def fit_calibration_panel(source, target_task_a, context, *, calibrate_m3,
                          certify_m3, predict_m3_geometry, check,
                          predict_m1=None, predict_m2=None,
                          predict_m3_variants=None, predict_m4=None, support_size=208):
    """The existing Task-A-only fit, shared by LOPO and its paired falsifiers.

    Each packet gets one pre-operator gain from the existing calibrator. All
    selected variants receive that same gain and source. Source fitting and
    observed Task B are deliberately outside this interface.
    """
    support_size = _support_size(support_size)
    if len(source.voxel_ids) != support_size:
        raise ValueError("calibration must preserve the selected complete spatial axis")
    validate_source_context(context)
    if source.training_ids != context.training_ids:
        raise ValueError("the calibration source must match its exact exclusions")
    target_task_a = _array(target_task_a, np.asarray(source.task_a_maps).shape, "target Task A")
    check()
    numerical = dict(calibrate_m3(source, target_task_a, context))
    basis_shape = np.asarray(source.B).shape
    if len(basis_shape) != 2 or basis_shape[0] != support_size or not 0 < basis_shape[1] < support_size:
        raise ValueError("M3 basis must preserve the fixed support and nonempty active subspace")
    numerical["O"] = _array(numerical["O"], (basis_shape[1], basis_shape[1]), "calibrated O")
    gain = np.asarray(numerical["gain"])
    if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
        raise ValueError("calibration must return one positive finite global gain")
    numerical["gain"] = float(gain)
    certify_m3(source, numerical, context)
    check()
    centered = source.G-source.G.mean(axis=-1, keepdims=True)
    prediction = _array(numerical["gain"]*apply_m3(centered, source.B, numerical["O"]),
                        (18, support_size), "M3 prediction")
    geometry = _array(predict_m3_geometry(source, numerical, context), (18, 18), "M3 geometry")
    if not np.array_equal(geometry, geometry.T):
        raise ValueError("M3 geometry must retain identical signed symmetric entries")
    model_predictions = {"M1": (), "M2": (), "M3": (), "M4": ()}
    for callback, names, prediction_type, label in (
            (predict_m1, M1_VARIANTS, M1Prediction, "M1"),
            (predict_m2, M2_VARIANTS, M2Prediction, "M2"),
            (predict_m3_variants, M3_VARIANTS, M3Prediction, "M3"),
            (predict_m4, M4_VARIANTS, M4Prediction, "M4")):
        if callback is None:
            continue
        check()
        calibration = numerical if label in ("M3", "M4") else numerical["gain"]
        supplied = tuple(callback(source, target_task_a, calibration, context))
        if (len(supplied) != len(names) or not all(isinstance(value, prediction_type) for value in supplied)
                or tuple(value.name for value in supplied) != names):
            raise ValueError(f"all four fixed {label} variants are required in order")
        validated = []
        for value in supplied:
            maps = _array(value.maps, (18, support_size), f"{label} prediction")
            matrix = _array(value.geometry, (18, 18), f"{label} geometry")
            if not np.array_equal(matrix, matrix.T):
                raise ValueError(f"{label} geometry must retain identical signed symmetric entries")
            validated.append(prediction_type(value.name, maps, matrix, value.diagnostics))
        model_predictions[label] = tuple(validated)
    if model_predictions["M3"]:
        reference = model_predictions["M3"][0]
        if (np.max(np.abs(reference.maps-prediction)) > source.numerical_atol
                or np.max(np.abs(reference.geometry-geometry)) > source.numerical_atol):
            raise ValueError("M3 variant reference must reproduce the existing paired prediction")
    return CalibrationPanel(numerical, prediction, geometry,
                            model_predictions["M1"], model_predictions["M2"],
                            model_predictions["M3"], model_predictions["M4"])


def fit_development_lopo(records, *, fit_shared: Callable, calibrate_m3: Callable,
                         certify_m3: Callable, predict_m3_geometry: Callable, check: Callable,
                         include_participant_deletions: bool, predict_m1: Callable | None = None,
                         predict_m2: Callable | None = None, predict_matched_warps: Callable | None = None,
                         predict_m3_variants: Callable | None = None, predict_m4: Callable | None = None,
                         fit_m0a: Callable | None = None, predict_random_bases: Callable | None = None,
                         support_size=208):
    """Fit all 12 targets, optionally repeating LOPO for every participant deletion.

    fit_shared(training_records, context) receives eleven participants for the
    full run and ten for each nested deletion. Neither the target record nor
    deleted record is supplied. calibrate_m3(source, target_task_a, context)
    implements the fixed target rule and returns O/gain plus its diagnostics.
    It may enforce block locality or a separately fixed common gain; this
    orchestrator does not substitute unrestricted global Procrustes.
    certify_m3(source, numerical_fit, context)
    must raise if the fixed model certificates fail; successful return supplies
    no new scientific authority. The target fitter receives Task A alone.
    predict_m3_geometry(source, numerical_fit, context) implements the selected
    source geometry recipe using that same fitted operator. It must agree with
    the actual source maps/partitions used by the fixed prediction recipe;
    gain**2*K0 is not assumed for an otherwise unspecified K0 estimator.
    Optional predict_m1(source, target_task_a, common_gain, context) returns
    the four named M1 predictions using that same source and pre-operator gain.
    No target Task B enters this callback either. Native signatures and M1
    class certification are not supplied by these point predictions.
    Optional predict_m2 has the same restricted arguments and must return
    all four named M2 predictions. Its numerical certificates do not establish
    half-field stability, a matched-warp signature or a global fit optimum.
    Optional predict_matched_warps(source, m2_predictions, common_gain, context)
    consumes those existing fits and returns eight separate control draws per
    variant. It receives no target A or B; its conditioning is the M2 fit.
    Optional predict_m3_variants(source, target_task_a, reference_numerical_fit,
    context) returns the four named M3 variants without changing the shared
    source or common gain. Its reference must reproduce the existing M3 maps
    and actual four-partition geometry within the source's fixed tolerance.
    Optional predict_m4 uses those same arguments and returns four M4 variants
    all paired to the supplied reference numerical fit. It cannot substitute
    an alternative M3 operator, refit the common gain, or consume target B.
    Optional fit_m0a(training_records, context) returns the raw anatomical
    group baseline without target A, target B or functional alignment. It uses
    the same complete training exclusion as the shared functional source.
    Optional predict_random_bases(source, target_task_a, reference_numerical_fit,
    context) returns eight separate M3 controls on anatomy-seeded local bases,
    retaining the common gain and shared source. Target B is not supplied.

    The optional nested procedure performs 12*11 additional shared fits and
    target calibrations; it never substitutes deletion of already-fitted rows.
    check() is the caller's allocation/deadline check before each fit and at
    completion. Failures propagate: no skipped target or reduced-N result.
    No files, margins, confidence bounds, native signatures or class decision.
    """
    if type(include_participant_deletions) is not bool:
        raise ValueError("participant-deletion execution must be explicitly selected")
    if not all(callable(fn) for fn in (fit_shared, calibrate_m3, certify_m3, predict_m3_geometry, check)):
        raise TypeError("source fitting, calibration, geometry, certification and budget callbacks are required")
    if predict_m1 is not None and not callable(predict_m1):
        raise TypeError("the optional M1 predictor must be callable")
    if predict_m2 is not None and not callable(predict_m2):
        raise TypeError("the optional M2 predictor must be callable")
    if predict_m3_variants is not None and not callable(predict_m3_variants):
        raise TypeError("the optional M3 variant predictor must be callable")
    if predict_m4 is not None and not callable(predict_m4):
        raise TypeError("the optional M4 predictor must be callable")
    if fit_m0a is not None and not callable(fit_m0a):
        raise TypeError("the optional anatomical group fitter must be callable")
    if predict_random_bases is not None and not callable(predict_random_bases):
        raise TypeError("the optional random local-basis predictor must be callable")
    if predict_matched_warps is not None and (not callable(predict_matched_warps) or predict_m2 is None):
        raise TypeError("matched warps require a callable and the existing M2 predictor")
    support_size = _support_size(support_size)
    participants = _participants(records, support_size=support_size)

    def run(deleted_id):
        retained = tuple(record for record in participants if record.participant_id != deleted_id)
        folds = []
        for target in retained:
            check()
            training = tuple(record for record in retained if record.participant_id != target.participant_id)
            context = FoldContext(target.participant_id,
                                  tuple(record.participant_id for record in training), deleted_id)
            anatomical = None
            if fit_m0a is not None:
                anatomical = fit_m0a(training, context)
                if (not isinstance(anatomical, M0aPrediction)
                        or anatomical.training_ids != context.training_ids):
                    raise ValueError("M0a must preserve the exact current-fold training identities")
                anatomical_maps = _array(anatomical.maps, (18, support_size), "M0a maps")
                anatomical_geometry = _array(anatomical.geometry, (18, 18), "M0a geometry")
                if not np.array_equal(anatomical_geometry, anatomical_geometry.T):
                    raise ValueError("M0a geometry must retain identical signed symmetric entries")
                anatomical = M0aPrediction(anatomical_maps, anatomical_geometry, context.training_ids)
                check()
            source = fit_shared(training, context)
            _, G, K0, s_map, s_geometry = _shared(source, context, target)
            panel = fit_calibration_panel(
                source, target.task_a_maps, context,
                calibrate_m3=calibrate_m3, certify_m3=certify_m3,
                predict_m3_geometry=predict_m3_geometry, check=check,
                predict_m1=predict_m1, predict_m2=predict_m2,
                predict_m3_variants=predict_m3_variants, predict_m4=predict_m4,
                support_size=support_size,
            )
            numerical = panel.numerical_fit
            prediction, geometry = panel.m3_maps, panel.m3_geometry
            model_predictions = {"M1": panel.m1_predictions, "M2": panel.m2_predictions,
                                 "M3": panel.m3_variant_predictions, "M4": panel.m4_predictions}
            random_bases = ()
            if predict_random_bases is not None:
                check()
                supplied = tuple(predict_random_bases(source, target.task_a_maps, numerical, context))
                if (not all(isinstance(value, RandomBasisPrediction) for value in supplied)
                        or tuple(value.draw_index for value in supplied) != tuple(range(RANDOM_BASIS_DRAWS))
                        or any(type(value.draw_index) is not int for value in supplied)):
                    raise ValueError("all eight random local-basis draws are required in order")
                validated = []
                for value in supplied:
                    maps = _array(value.maps, (18, support_size), "random local-basis prediction")
                    matrix = _array(value.geometry, (18, 18), "random local-basis geometry")
                    if not np.array_equal(matrix, matrix.T):
                        raise ValueError("random local-basis geometry must retain identical signed symmetric entries")
                    validated.append(RandomBasisPrediction(value.draw_index, maps, matrix, value.diagnostics))
                random_bases = tuple(validated)
            controls = ()
            if predict_matched_warps is not None:
                check()
                supplied = tuple(predict_matched_warps(source, model_predictions["M2"], numerical["gain"], context))
                expected = tuple((variant, draw) for variant in M2_VARIANTS for draw in range(MATCHED_WARP_DRAWS))
                if (not all(isinstance(value, MatchedWarpPrediction) for value in supplied)
                        or tuple((value.variant, value.draw_index) for value in supplied) != expected
                        or any(type(value.draw_index) is not int for value in supplied)):
                    raise ValueError("all four variants and eight matched-warp draws are required in order")
                validated = []
                for value in supplied:
                    maps = _array(value.maps, (18, support_size), "matched-warp prediction")
                    matrix = _array(value.geometry, (18, 18), "matched-warp geometry")
                    if not np.array_equal(matrix, matrix.T):
                        raise ValueError("matched-warp geometry must retain identical signed symmetric entries")
                    validated.append(MatchedWarpPrediction(value.variant, value.draw_index, maps, matrix, value.diagnostics))
                controls = tuple(validated)
            folds.append(FoldPrediction(
                context, G, K0, s_map, s_geometry,
                _array(prediction, (18, support_size), "M3 prediction"),
                geometry,
                _array(numerical["O"], numerical["O"].shape, "O"),
                float(numerical["gain"]), target.task_b_partitions,
                model_predictions["M1"], model_predictions["M2"], controls, model_predictions["M3"],
                model_predictions["M4"],
                anatomical,
                random_bases,
                CalibrationReference(context, source, panel, target.task_a_maps),
            ))
        return LOPORun(tuple(record.participant_id for record in retained), deleted_id, tuple(folds))

    full = run(None)
    deletions = tuple(run(participant) for participant in DEVELOPMENT_IDS) if include_participant_deletions else ()
    check()
    return DevelopmentLOPO(full, deletions)


def exploratory_points(run):
    """M0b/M3 and optional four-variant M1--M4 point summaries.

    Each participant uses its own G, K0 and projection. Only positive scalar
    scales are averaged across targets. Signed signal/residual estimates are
    retained; there are no ratios, threshold decisions or calibrated bounds.
    These summaries do not complete the required M1--M4 panel or native signatures.
    """
    expected = tuple(participant for participant in DEVELOPMENT_IDS if participant != run.deleted_id)
    if (run.participant_ids != expected or len(run.folds) != len(expected)
            or tuple(fold.context.target_id for fold in run.folds) != expected
            or any(fold.context.deleted_id != run.deleted_id for fold in run.folds)):
        raise ValueError("LOPO folds must preserve the complete ordered participant/deletion identities")
    for fold in run.folds:
        if fold.context.training_ids != tuple(item for item in expected if item != fold.context.target_id):
            raise ValueError("fold training identities do not match the nested source exclusion")
    observed = np.stack([fold.observed_partitions for fold in run.folds])
    G = np.stack([fold.G for fold in run.folds])
    K0 = np.stack([fold.K0 for fold in run.folds])
    prediction = np.stack([fold.m3_maps for fold in run.folds])
    predicted_geometry = np.stack([fold.m3_geometry for fold in run.folds])
    halves = (observed[:, (0, 2)]+observed[:, (1, 3)])/2
    map_residual = halves-G[:, None]
    predicted_residual = prediction-G
    geometries = crossvalidated_geometries(observed)
    geometry_residual = project_gain_direction(geometries-K0[:, None], K0[:, None])
    geometry_prediction = project_gain_direction(predicted_geometry-K0, K0)
    vm = map_covariance(map_residual[:, 0], map_residual[:, 1])
    vg = geometry_covariance(geometry_residual[:, 0], geometry_residual[:, 1])
    result = {
        "participant_ids": run.participant_ids, "deleted_id": run.deleted_id,
        "V_map": vm, "V_geometry": vg,
        "R_map": {"M0b": vm, "M3": map_covariance(
            map_residual[:, 0]-predicted_residual, map_residual[:, 1]-predicted_residual)},
        "R_geometry": {"M0b": vg, "M3": geometry_covariance(
            geometry_residual[:, 0]-geometry_prediction, geometry_residual[:, 1]-geometry_prediction)},
        "S_map": float(np.mean([fold.S_map for fold in run.folds])),
        "S_geometry": float(np.mean([fold.S_geometry for fold in run.folds])),
    }
    anatomical_present = tuple(fold.m0a_prediction is not None for fold in run.folds)
    if any(anatomical_present):
        if not all(anatomical_present) or any(
                fold.m0a_prediction.training_ids != fold.context.training_ids for fold in run.folds):
            raise ValueError("M0a must preserve complete folds and training exclusions")
        anatomical_maps = np.stack([fold.m0a_prediction.maps for fold in run.folds])
        anatomical_geometry = np.stack([fold.m0a_prediction.geometry for fold in run.folds])
        difference = anatomical_maps-G
        geometry_difference = project_gain_direction(anatomical_geometry-K0, K0)
        result["R_map"]["M0a"] = map_covariance(
            map_residual[:, 0]-difference, map_residual[:, 1]-difference)
        result["R_geometry"]["M0a"] = geometry_covariance(
            geometry_residual[:, 0]-geometry_difference, geometry_residual[:, 1]-geometry_difference)
    for field, variants, label in (("m1_predictions", M1_VARIANTS, "M1"),
                                   ("m2_predictions", M2_VARIANTS, "M2"),
                                   ("m3_variant_predictions", M3_VARIANTS, "M3"),
                                   ("m4_predictions", M4_VARIANTS, "M4")):
        names = tuple(value.name for value in getattr(run.folds[0], field))
        if (names not in ((), variants)
                or any(tuple(value.name for value in getattr(fold, field)) != names for fold in run.folds)):
            raise ValueError(f"{label} variants must be complete and identical across all participant folds")
        for index, name in enumerate(names):
            maps = np.stack([getattr(fold, field)[index].maps for fold in run.folds])
            matrices = np.stack([getattr(fold, field)[index].geometry for fold in run.folds])
            difference = maps-G
            geometry_difference = project_gain_direction(matrices-K0, K0)
            result["R_map"][name] = map_covariance(map_residual[:, 0]-difference, map_residual[:, 1]-difference)
            result["R_geometry"][name] = geometry_covariance(
                geometry_residual[:, 0]-geometry_difference, geometry_residual[:, 1]-geometry_difference)
    basis_keys = tuple(value.draw_index for value in run.folds[0].random_basis_predictions)
    if (basis_keys not in ((), tuple(range(RANDOM_BASIS_DRAWS)))
            or any(tuple(value.draw_index for value in fold.random_basis_predictions) != basis_keys
                   for fold in run.folds)):
        raise ValueError("random local-basis draws must be complete and identical across participant folds")
    if basis_keys:
        map_values, geometry_values = [], []
        for draw in range(RANDOM_BASIS_DRAWS):
            maps = np.stack([fold.random_basis_predictions[draw].maps for fold in run.folds])
            matrices = np.stack([fold.random_basis_predictions[draw].geometry for fold in run.folds])
            difference = maps-G
            geometry_difference = project_gain_direction(matrices-K0, K0)
            map_values.append(map_covariance(map_residual[:, 0]-difference, map_residual[:, 1]-difference))
            geometry_values.append(geometry_covariance(
                geometry_residual[:, 0]-geometry_difference, geometry_residual[:, 1]-geometry_difference))
        result["random_basis_draw_residuals"] = {"R_map": tuple(map_values), "R_geometry": tuple(geometry_values)}
        result["R_map"]["M3_random_local_basis"] = float(np.mean(map_values))
        result["R_geometry"]["M3_random_local_basis"] = float(np.mean(geometry_values))
    control_keys = tuple((value.variant, value.draw_index) for value in run.folds[0].matched_warp_predictions)
    expected_controls = tuple((variant, draw) for variant in M2_VARIANTS for draw in range(MATCHED_WARP_DRAWS))
    if (control_keys not in ((), expected_controls)
            or any(tuple((value.variant, value.draw_index) for value in fold.matched_warp_predictions) != control_keys
                   for fold in run.folds)):
        raise ValueError("matched-warp variants/draws must be complete and identical across participant folds")
    if control_keys:
        result["matched_warp_draw_residuals"] = {}
        for variant_index, variant in enumerate(M2_VARIANTS):
            map_values, geometry_values = [], []
            for draw in range(MATCHED_WARP_DRAWS):
                index = variant_index*MATCHED_WARP_DRAWS+draw
                maps = np.stack([fold.matched_warp_predictions[index].maps for fold in run.folds])
                matrices = np.stack([fold.matched_warp_predictions[index].geometry for fold in run.folds])
                difference = maps-G
                geometry_difference = project_gain_direction(matrices-K0, K0)
                map_values.append(map_covariance(map_residual[:, 0]-difference, map_residual[:, 1]-difference))
                geometry_values.append(geometry_covariance(
                    geometry_residual[:, 0]-geometry_difference, geometry_residual[:, 1]-geometry_difference))
            result["matched_warp_draw_residuals"][variant] = {
                "R_map": tuple(map_values), "R_geometry": tuple(geometry_values)}
            # Squaring precedes draw averaging. Averaging predictions would
            # describe a different, uncertified ensemble operator.
            result["R_map"][variant+"_matched_random_warp"] = float(np.mean(map_values))
            result["R_geometry"][variant+"_matched_random_warp"] = float(np.mean(geometry_values))
    return result
