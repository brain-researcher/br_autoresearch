"""Prospective EP15 anchored functional source and block-local M3 recipe.

Array-only, float64, with no reader, empirical release, cache or file output.
This is a two-stage anatomical-anchor estimator, not iterative generalized
Procrustes or a global optimum claim. All source participants belong to the
current training fold. Target calibration sees Task A only. Reader isolation,
functional validity and the complete model/control panel remain separate.
"""
from dataclasses import dataclass

import numpy as np

from development_lopo import SharedM3Source, validate_source_context, _support_size
from registered_endpoint_metrics import (
    CONDITIONS, condition_weights, crossvalidated_geometries, geometry_inner,
)
from spatial_deformation_operators import apply_m3


# Exact noninstruction roster/order in the frozen condition_partition_plan.json.
TASK_A_CONDITIONS = (
    "CheckerBoard", "DigitJudgement", "FingerSeq", "FingerSimple", "Go",
    "HappyFaces", "IntervalTiming", "Math", "MotorImagery", "NoGo",
    "Object0Back", "Object2Back", "PleasantScenes", "SadFaces", "SpatialNavigation",
    "StroopCon", "StroopIncon", "ToM", "UnpleasantScenes", "VerbGen",
    "Verbal0Back", "Verbal2Back", "VideoAct", "VideoKnots", "VisualSearchLarge",
    "VisualSearchMed", "VisualSearchSmall", "WordRead", "rest",
)


def _tolerance(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isrealobj(value):
        raise ValueError(f"{name} must be an explicit finite scalar in (0, 1)")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be an explicit finite scalar in (0, 1)")
    return value


def _array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} requires a real array with shape {shape}")
    value = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite; no dropping or imputation")
    return value


def _frozen(value):
    value = np.array(value, dtype=np.float64, copy=True)
    value.flags.writeable = False
    return value


def _center(value):
    result = value-value.mean(axis=-1, keepdims=True)
    if not np.isfinite(result).all():
        raise ArithmeticError("nonfinite spatial centering")
    return result


def _full_rank(singular, dimension, rtol, label):
    singular = np.asarray(singular)
    if (singular.shape != (dimension,) or not np.isfinite(singular).all()
            or np.any(singular <= 0) or singular[-1] <= rtol*singular[0]):
        raise ValueError(f"{label} is not full rank at the fixed rank tolerance")


@dataclass(frozen=True)
class LocalSourceGeometry:
    source_partitions: np.ndarray
    training_anchor_task_a: np.ndarray
    training_rotations: tuple[np.ndarray, ...]
    training_diagnostics: tuple[dict, ...]
    training_ids: tuple[str, ...]


class LocalFunctionalSource:
    """Fixed reference recipe exposed as the four development_lopo callbacks.

    anatomy is a LocalAnatomicalBasis (8 cells, two modes each). Its fixed native
    voxel IDs/coordinates are reused without changing support or order. Supply
    numerical_atol=1e-10 and rank_rtol=1e-8 explicitly for the prospective
    reference; the basis must have been constructed with those same values.
    """

    def __init__(self, anatomy, *, numerical_atol, rank_rtol):
        self.atol = _tolerance(numerical_atol, "numerical_atol")
        self.rank_rtol = _tolerance(rank_rtol, "rank_rtol")
        if (anatomy.tolerances.numerical_atol != self.atol
                or anatomy.tolerances.rank_rtol != self.rank_rtol):
            raise ValueError("basis and fitting numerical settings must be identical")
        if (anatomy.split_depth != 3 or anatomy.modes_per_cell != 2
                or anatomy.blocks != tuple((2*i, 2*i+2) for i in range(8))):
            raise ValueError("the reference recipe requires eight ordered two-mode blocks")
        self.support_size = _support_size(len(anatomy.voxel_ids))
        self.B = _frozen(_array(anatomy.B, (self.support_size, 16), "anatomical B"))
        self.voxel_ids = tuple(anatomy.voxel_ids)
        if len(self.voxel_ids) != self.support_size or len(set(self.voxel_ids)) != self.support_size:
            raise ValueError("all selected unique native voxel identities must be retained")
        labels = np.asarray(anatomy.cell_labels)
        if labels.shape != (self.support_size,) or labels.dtype.kind not in "iu" or set(labels) != set(range(8)):
            raise ValueError("all eight cells must retain their fixed voxel labels")
        self.labels = labels.copy()
        self.blocks = anatomy.blocks
        coordinates = _array(anatomy.coordinates, (self.support_size, 3), "anatomical coordinates")
        diameter = float(anatomy.max_cell_diameter_mm)
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("an explicit positive anatomy-only cell diameter bound is required")
        self.max_cell_diameter_mm = diameter
        actual_diameter = 0.
        for cell, (start, stop) in enumerate(self.blocks):
            member = self.labels == cell
            if np.max(np.abs(self.B[~member, start:stop])) > self.atol:
                raise ValueError("basis modes must have zero support outside their anatomical cell")
            points = coordinates[member]
            distances = np.sqrt(np.sum((points[:, None]-points[None, :])**2, axis=-1))
            actual_diameter = max(actual_diameter, float(np.max(distances)))
        if actual_diameter > diameter+self.atol:
            raise ValueError("actual anatomical cell diameter exceeds its frozen bound")
        self.actual_max_cell_diameter_mm = actual_diameter
        self.weights = _frozen(np.full(len(TASK_A_CONDITIONS), 1/len(TASK_A_CONDITIONS)))
        # Includes full-space checks on the identity operator and fixed basis.
        self.operator_certificate(np.eye(16))

    def operator_certificate(self, O):
        """Inspect the actual full-support Q, including cell locality.

        The diameter bounds transport only because cross-cell mixing is zero.
        Positive global gain is kept outside Q and is not called an isometry.
        """
        O = _array(O, (16, 16), "O")
        allowed = np.zeros((16, 16), dtype=bool)
        for start, stop in self.blocks:
            allowed[start:stop, start:stop] = True
        identity = np.eye(self.support_size)
        Q = identity+self.B@(O-np.eye(16))@self.B.T
        complement = identity-self.B@self.B.T
        diagnostics = {
            "basis_orthonormality_max_abs": float(np.max(np.abs(self.B.T@self.B-np.eye(16)))),
            "basis_centering_max_abs": float(np.max(np.abs(self.B.sum(axis=0)))),
            "O_cross_block_max_abs": float(np.max(np.abs(O[~allowed]))),
            "O_orthogonality_max_abs": float(max(np.max(np.abs(O.T@O-np.eye(16))),
                                                  np.max(np.abs(O@O.T-np.eye(16))))),
            "Q_orthogonality_max_abs": float(max(np.max(np.abs(Q.T@Q-identity)),
                                                  np.max(np.abs(Q@Q.T-identity)))),
            "Q_constant_preservation_max_abs": float(np.max(np.abs(Q@np.ones(self.support_size)-1))),
            "Q_identity_complement_max_abs": float(np.max(np.abs((Q-identity)@complement))),
            "Q_cross_cell_max_abs": float(np.max(np.abs(Q[self.labels[:, None] != self.labels[None, :]]))),
        }
        if any(not np.isfinite(value) or value > self.atol for value in diagnostics.values()):
            raise ValueError("the actual M3 operator fails fixed algebra/locality certification")
        diagnostics["max_transport_distance_bound_mm"] = self.max_cell_diameter_mm
        return diagnostics

    def fit_block_operator(self, source_task_a, target_task_a, *, global_gain):
        """Unique weighted polar factor in each allowed block, no rank repair.

        Full global source-design rank is required. Global cross-product rank
        is diagnostic only: local cross-product rank establishes uniqueness in
        the selected block-diagonal family, including reflections. Each local
        minimum singular value must exceed rank_rtol times the global cross
        spectral norm, so a roundoff-only local block cannot pass relative to
        its own tiny norm. Global cross rank itself is not an admission gate.
        """
        source = _center(_array(source_task_a, (29, self.support_size), "source Task A"))
        target = _center(_array(target_task_a, (29, self.support_size), "target Task A"))
        scalar = np.asarray(global_gain)
        if scalar.shape != () or scalar.dtype.kind not in "fiu" or not np.isfinite(scalar) or scalar <= 0:
            raise ValueError("one fixed positive finite global gain is required")
        gain = float(scalar)
        active_source = np.sqrt(self.weights)[:, None]*(source@self.B)
        active_target = np.sqrt(self.weights)[:, None]*(target@self.B)
        source_singular = np.linalg.svd(active_source, compute_uv=False)
        _full_rank(source_singular, 16, self.rank_rtol, "global Task-A active source design")
        cross = active_source.T@active_target
        global_cross_singular = np.linalg.svd(cross, compute_uv=False)
        local_cutoff = self.rank_rtol*global_cross_singular[0]
        O = np.zeros((16, 16), dtype=np.float64)
        local_singular = []
        for start, stop in self.blocks:
            left, singular, right_t = np.linalg.svd(cross[start:stop, start:stop], full_matrices=False)
            if not np.isfinite(singular).all() or singular[-1] <= local_cutoff:
                raise ValueError("local Task-A cross-product is not full rank at global objective scale")
            O[start:stop, start:stop] = left@right_t
            local_singular.append(_frozen(singular))
        certificate = self.operator_certificate(O)
        residual = gain*apply_m3(source, self.B, O)-target
        loss = float(np.sum(self.weights[:, None]*residual**2))
        if not np.isfinite(loss):
            raise ArithmeticError("nonfinite Task-A objective")
        return {"O": _frozen(O), "gain": gain, "weighted_task_a_squared_error": loss,
                "source_design_singular_values": _frozen(source_singular),
                "global_cross_singular_values_diagnostic_only": _frozen(global_cross_singular),
                "local_cross_rank_cutoff": float(local_cutoff),
                "local_cross_rank_scale": "global_cross_product_spectral_norm",
                "local_cross_singular_values": tuple(local_singular),
                "operator_certificate": certificate,
                "unique_within_fixed_block_family": True}

    def _context(self, context):
        validate_source_context(context)

    def fit_shared(self, training, context):
        self._context(context)
        training = tuple(training)
        if tuple(record.participant_id for record in training) != context.training_ids:
            raise ValueError("training records must exactly match the current source fold")
        A, B = [], []
        for record in training:
            if (record.task_a_condition_ids != TASK_A_CONDITIONS or record.task_b_condition_ids != CONDITIONS
                    or record.voxel_ids != self.voxel_ids):
                raise ValueError("source records must preserve the complete fixed condition/support identities")
            A.append(_center(_array(record.task_a_maps, (29, self.support_size), "training Task A")))
            B.append(_center(_array(record.task_b_partitions, (4, 18, self.support_size), "training B partitions")))
        A, B = np.stack(A), np.stack(B)
        anchor = A.mean(axis=0)
        mapped_a, mapped_b, rotations, diagnostics = [], [], [], []
        for person in range(len(training)):
            fitted = self.fit_block_operator(anchor, A[person], global_gain=1.)
            inverse = fitted["O"].T
            mapped_a.append(apply_m3(A[person], self.B, inverse))
            mapped_b.append(apply_m3(B[person].reshape(72, self.support_size), self.B, inverse).reshape(4, 18, self.support_size))
            rotations.append(fitted["O"])
            diagnostics.append(fitted)
        source_a = np.mean(mapped_a, axis=0)
        partitions = np.mean(mapped_b, axis=0)
        G = partitions.mean(axis=0)
        K0 = crossvalidated_geometries(partitions[None])[0].mean(axis=0)
        halves = partitions.reshape(2, 2, 18, self.support_size).mean(axis=1)
        s_map = float(np.sum(condition_weights()*np.mean(halves[0]*halves[1], axis=-1)))
        if not np.isfinite(s_map) or s_map <= 0 or float(geometry_inner(K0, K0)) <= 0:
            raise ValueError("actual training-only source map/geometry scales must be positive")
        state = LocalSourceGeometry(_frozen(partitions), _frozen(anchor), tuple(rotations),
                                    tuple(diagnostics), context.training_ids)
        return SharedM3Source(context.training_ids, TASK_A_CONDITIONS, self.voxel_ids,
                              _frozen(source_a), _frozen(G), _frozen(K0), s_map,
                              self.B, self.weights, True, self.rank_rtol, self.atol, state)

    def _source(self, source, context):
        self._context(context)
        if (not isinstance(source.geometry_state, LocalSourceGeometry)
                or source.training_ids != context.training_ids
                or source.geometry_state.training_ids != context.training_ids
                or source.voxel_ids != self.voxel_ids or source.task_a_condition_ids != TASK_A_CONDITIONS
                or source.rank_rtol != self.rank_rtol or source.numerical_atol != self.atol
                or not np.array_equal(source.B, self.B) or not np.array_equal(source.task_a_weights, self.weights)):
            raise ValueError("source must be this fixed recipe's actual current-fold fit")

    def calibrate_m3(self, source, target_task_a, context):
        self._source(source, context)
        X = _center(_array(source.task_a_maps, (29, self.support_size), "source Task A"))
        Y = _center(_array(target_task_a, (29, self.support_size), "target Task A"))
        energy = float(np.sum(self.weights[:, None]*X**2))
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("the common-gain source energy must be positive")
        gain = float(np.sum(self.weights[:, None]*X*Y)/energy)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("the pre-operator common Task-A gain is nonpositive or nonfinite")
        # Fixed before fitting O; never recomputed after the model-specific fit.
        result = self.fit_block_operator(X, Y, global_gain=gain)
        result["gain_rule"] = "positive_scalar_from_untransformed_source_Task_A_before_model_fit"
        return result

    def certify_m3(self, source, numerical_fit, context):
        self._source(source, context)
        if (numerical_fit.get("unique_within_fixed_block_family") is not True
                or numerical_fit.get("gain_rule") != "positive_scalar_from_untransformed_source_Task_A_before_model_fit"):
            raise ValueError("the fixed block-family identification and common-gain fit are required")
        _full_rank(numerical_fit["source_design_singular_values"], 16, self.rank_rtol, "global source design")
        if len(numerical_fit["local_cross_singular_values"]) != 8:
            raise ValueError("all eight local identification diagnostics are required")
        cutoff = numerical_fit["local_cross_rank_cutoff"]
        if (not np.isfinite(cutoff) or cutoff < 0
                or cutoff != self.rank_rtol*numerical_fit["global_cross_singular_values_diagnostic_only"][0]):
            raise ValueError("the fixed global-objective local rank cutoff is required")
        for singular in numerical_fit["local_cross_singular_values"]:
            singular = np.asarray(singular)
            if singular.shape != (2,) or not np.isfinite(singular).all() or singular[-1] <= cutoff:
                raise ValueError("local cross-product is unidentified at the global objective scale")
        gain = np.asarray(numerical_fit["gain"])
        if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
            raise ValueError("common gain must remain positive and finite")
        return self.operator_certificate(numerical_fit["O"])

    def predict_m3_geometry(self, source, numerical_fit, context):
        self.certify_m3(source, numerical_fit, context)
        partitions = source.geometry_state.source_partitions
        predicted = numerical_fit["gain"]*apply_m3(
            partitions.reshape(72, self.support_size), self.B, numerical_fit["O"],
        ).reshape(4, 18, self.support_size)
        return crossvalidated_geometries(predicted[None])[0].mean(axis=0)
