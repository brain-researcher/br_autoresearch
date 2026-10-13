"""Prospective EP15 M2 direction falsifier, conditional on a Task-A-fit field.

Eight SO(3) rotations preserve the field's pointwise displacement magnitude,
displacement Gram kernel and derivative singular spectrum. They are not an
omnibus 24-dimensional null, condition-geometry preservation, or exact matching
of det(I+Du). The conditional rotation orbit has at most three degrees of
freedom; this is not an equal fitting-capacity test. No reader, target B,
fitting, p-value or native-signature decision.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from development_lopo import (
    DEVELOPMENT_IDS, M2Prediction, M2_VARIANTS, MatchedWarpPrediction,
)
from local_functional_source import _array, _frozen
from registered_endpoint_metrics import crossvalidated_geometries
from smooth_relocation import VARIANTS


MASTER_SEED = 20261007
DRAW_COUNT = 8
MAX_PROPOSALS_PER_DRAW = 32


@dataclass(frozen=True)
class MatchedRandomField:
    parent_variant: str
    draw_index: int
    coefficients: np.ndarray
    rotation: np.ndarray
    diagnostics: dict


def _haar_rotation(rng):
    # QR of independent standard normals, corrected to positive R diagonal,
    # gives Haar O(3); a final fixed-column sign correction gives Haar SO(3).
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    diagonal = np.diag(r)
    if not np.isfinite(diagonal).all() or np.any(diagonal == 0):
        raise ArithmeticError("degenerate random rotation construction")
    q = q*np.sign(diagonal)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    if (np.max(np.abs(q.T@q-np.eye(3))) > 1e-12
            or abs(np.linalg.det(q)-1) > 1e-12):
        raise ArithmeticError("random rotation failed proper-orthogonality check")
    return q


def _geometry(representation, field, coefficients):
    """No response maps enter proposal admission, including source maps."""
    certificate = field.certificate(coefficients)
    displacement, _ = field.displacement(field.coordinates, coefficients)
    queries = field.coordinates+displacement
    weights, _ = representation.weights_and_derivatives(queries)
    interpolation = representation.certificate(weights)
    inverse, inverse_certificate = field.inverse(queries, coefficients)
    inverse_error = float(np.linalg.norm(inverse-field.coordinates, axis=1).max())
    if inverse_error > 2e-10:
        raise ArithmeticError("random pull/inverse composition failed")
    return {"field_certificate": certificate,
            "interpolation_certificate": interpolation,
            "inverse_certificate": inverse_certificate,
            "inverse_composition_max_mm": inverse_error}


def matched_random_fields(representation, field, coefficients, *, participant_id,
                          check: Callable):
    """All eight admissible draws or an explicit failure; never shrink a field.

    Independent PCG64 streams use SeedSequence([20261007, participant integer,
    fixed zero-based variant index, zero-based draw index]). A nested deletion
    keeps the same proposals; its refitted field can change geometry admission.
    Proposals are Haar before conditioning on the existing interpolation and
    inverse certificates. Neither Task-A fit quality nor any B value is used.
    """
    if participant_id not in DEVELOPMENT_IDS:
        raise ValueError("only an assigned development participant is supported")
    if not callable(check):
        raise TypeError("an allocation/deadline callback is required")
    if field.variant not in VARIANTS:
        raise ValueError("one of the four fixed M2 variants is required")
    if not np.array_equal(representation.coordinates, field.coordinates):
        raise ValueError("field and cardinal representation must share ordered coordinates")
    T = _array(coefficients, (8, 3), "Task-A-fitted field coefficients")
    check()
    # An invalid parent cannot be rescued by finding an admissible null.
    _geometry(representation, field, T)
    participant_integer = int(participant_id.removeprefix("sub-"))
    variant_index = VARIANTS.index(field.variant)
    degenerate = not np.any(T)
    parent_norm = float(np.linalg.norm(T))
    draws = []
    for draw_index in range(DRAW_COUNT):
        seed = (MASTER_SEED, participant_integer, variant_index, draw_index)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        rejected = 0
        for proposal in range(1, MAX_PROPOSALS_PER_DRAW+1):
            check()
            rotation = _haar_rotation(rng)
            rotated = T@rotation
            # A global right rotation preserves the field at every spatial
            # query up to an orthogonal output rotation, not just 208 samples.
            norm_error = abs(float(np.linalg.norm(rotated))-parent_norm)
            gram_error = float(np.max(np.abs(rotated@rotated.T-T@T.T)))
            tolerance = 1e-10*max(1., parent_norm**2)
            if norm_error > tolerance or gram_error > tolerance:
                raise ArithmeticError("random rotation broke exact field matching")
            try:
                geometry = _geometry(representation, field, rotated)
            except (ValueError, ArithmeticError):
                rejected += 1
                continue
            diagnostics = dict(geometry, seed_entropy=seed,
                accepted_proposal=proposal, rejected_proposals=rejected,
                coefficient_norm_matching_error_mm=norm_error,
                coefficient_gram_matching_max_abs=gram_error,
                degenerate_identity_control=degenerate,
                proposal_law="Haar_SO3_conditioned_on_existing_geometric_certificates",
                conditional_direction_falsifier=True,
                omnibus_random_field_null=False,
                condition_geometry_preserved=False,
                exact_phi_jacobian_determinant_matched=False,
                native_signature_certified=False)
            draws.append(MatchedRandomField(field.variant.name, draw_index,
                _frozen(rotated), _frozen(rotation), diagnostics))
            break
        else:
            raise ArithmeticError(
                f"matched random warp unresolved: {field.variant.name} draw {draw_index} "
                f"has no admissible proposal among {MAX_PROPOSALS_PER_DRAW}; no shrink or rescue")
    check()
    return tuple(draws)


class MatchedRandomWarpPrediction:
    """Reuse the four existing M2 fits; no target data or new calibration call.

    This optional source-only component is not wired into LOPO by importing it.
    The caller must preserve each draw through scoring: average per-draw
    residual/alignment statistics, never score mean maps or mean operators.
    Eight draws are nuisance randomizations, not independent participants.
    """

    def __init__(self, relocation_predictor, *, check: Callable):
        if not callable(check):
            raise TypeError("an allocation/deadline callback is required")
        if tuple(field.variant.name for field in relocation_predictor.fields) != M2_VARIANTS:
            raise ValueError("the complete fixed four-variant relocation predictor is required")
        self.predictor, self.check = relocation_predictor, check

    def __call__(self, source, fitted_predictions, common_gain, context):
        self.predictor.source_recipe._source(source, context)
        predictions = tuple(fitted_predictions)
        if (len(predictions) != 4 or not all(isinstance(p, M2Prediction) for p in predictions)
                or tuple(p.name for p in predictions) != M2_VARIANTS):
            raise ValueError("all four existing M2 predictions are required in fixed order")
        gain = np.asarray(common_gain)
        if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
            raise ValueError("one positive finite common gain is required")
        results = []
        for field, fitted in zip(self.predictor.fields, predictions):
            diagnostics = fitted.diagnostics
            if (diagnostics.get("training_ids") != context.training_ids
                    or diagnostics.get("variant") != field.variant.name
                    or diagnostics.get("common_gain") != float(gain)):
                raise ValueError("matched control must reuse its exact fold, variant and common gain")
            draws = matched_random_fields(self.predictor.representation, field,
                diagnostics["coefficients"], participant_id=context.target_id, check=self.check)
            for draw in draws:
                self.check()
                partitions = self.predictor.predict_partitions(field,
                    source.geometry_state.source_partitions, draw.coefficients, float(gain))
                results.append(MatchedWarpPrediction(field.variant.name, draw.draw_index,
                    _frozen(partitions.mean(axis=0)),
                    _frozen(crossvalidated_geometries(partitions[None])[0].mean(axis=0)),
                    dict(draw.diagnostics, coefficients=draw.coefficients, rotation=draw.rotation,
                         training_ids=context.training_ids, target_id=context.target_id,
                         deleted_id=context.deleted_id, common_gain=float(gain),
                         draw_count=DRAW_COUNT, aggregation="mean_of_per_draw_statistics",
                         native_signature_certified=False, class_adequacy_claimed=False,
                         empirical_execution_authorized=False)))
        self.check()
        return tuple(results)
