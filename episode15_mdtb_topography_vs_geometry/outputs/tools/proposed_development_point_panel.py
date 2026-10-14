"""Adopted integration adapter for the EP15 M4 certificate (October 13, 2026).

The observed-map runner uses this qualified adapter under
scientist_amendment_20261013_m4_certificate. Numerical refusal still stops the
whole panel; resource admission and private publication remain unchanged.

Reuse the original constructor and every existing panel helper. The canonical
diagonal family can use a global profile/exclusion certificate when the
sufficient full-domain curvature bound refuses. Other families retain their
existing certificate. Actual operator checks still precede every prediction.
The copied fit_development_point_panel body is unchanged; its _fixed_callbacks
binding wraps only predict_m4 after all original callbacks are constructed.
Thus the same alternative certificate applies to all four M4 variants in
ordinary fitting and every existing calibration falsifier. Nothing can return
a reduced panel when a numerical check or resource guard refuses.

Activation is one explicit import replacement inside the existing
run_development_point_panel.run_released_development under the adopted amendment. The
existing reader, sole frozen support, execute(), publication mode and CLI
remain unchanged. This file performs no such replacement or monkeypatch.
"""
import numpy as np

import development_point_panel as original_panel
from development_point_panel import (
    DevelopmentPointPanel, _joined_arrays, exploratory_falsifier_points,
    exploratory_points, fit_calibration_falsifiers, fit_development_lopo,
    m2_alignment_points,
)
from paired_stretch_variants import (
    MAX_ITERATIONS, M4CertificateUnverified, M4Prediction, NUMERICAL_ATOL, OPTIMIZER_FTOL,
    PairedM4TaskAObjective, PairedStretchVariants, RANK_RTOL,
    _array, _center, _frozen, _solution_certificate,
    apply_m3, apply_m4, crossvalidated_geometries, fit_m4_task_a, stretch_specifications,
)
from diagonal_m4_global import fit_diagonal_global
from proposed_m4_certificate import proposed_curvature_certificate


def _fit_with_certificate(objective, specification, X, Y, B, O, settings, *, check):
    """Fit the same model and domain, retaining all numerical acceptance limits."""
    try:
        curvature = proposed_curvature_certificate(objective, specification, check=check)
    except M4CertificateUnverified:
        if specification.name != stretch_specifications()[0].name:
            raise
        # The global helper independently requires the exact canonical modes
        # and domain. Neither its local growth bound nor its exterior exclusion
        # is a full-domain Hessian bound for the legacy solution certificate.
        check()
        numerical, certificate = fit_diagonal_global(objective, specification, check=check)
        check()
        theta = _array(numerical["coefficients"], (specification.parameter_count,), "fitted coefficients")
        C = objective.matrix(theta)
        supplied = _array(numerical["C"], (16, 16), "fitted C")
        if (np.max(np.abs(C-supplied)) > NUMERICAL_ATOL
                or not np.array_equal(numerical["O"], objective.O)
                or numerical["gain"] != objective.gain):
            raise ValueError("the fitted C, reference O and common gain must remain paired")
        certificate = {**certificate,
            "weighted_task_a_squared_error": numerical["weighted_task_a_squared_error"],
            "fitted_tangent_singular_values": _frozen(objective.tangent_singular_values(theta)),
            "numerical_certificate_method": "global_profile_and_full_domain_exclusion"}
        return C, numerical, certificate
    check()
    numerical = fit_m4_task_a(X, Y, B, O, **settings,
        spectral_norm_bound=specification.spectral_norm_bound,
        optimizer_ftol=OPTIMIZER_FTOL, max_iterations=MAX_ITERATIONS)
    check()
    C, certificate = _solution_certificate(objective, specification, numerical, curvature)
    return C, numerical, {**certificate,
        "numerical_certificate_method": "sufficient_full_domain_curvature"}


class ProposedPairedStretchVariants:
    """Wrap one original, successfully constructed callback without refitting."""

    def __init__(self, original):
        if type(original) is not PairedStretchVariants:
            raise TypeError("the unchanged constructed M4 callback is required")
        self.reference, self.check = original.reference, original.check
        self.support_size, self.specifications = original.support_size, original.specifications
        self.operator_certificate = original.operator_certificate

    def __call__(self, source, targetA, reference_numerical_fit, context):
        self.check()
        self.reference._source(source, context)
        self.reference.certify_m3(source, reference_numerical_fit, context)
        X = _center(_array(source.task_a_maps, (29, self.support_size), "source Task A"))
        Y = _center(_array(targetA, (29, self.support_size), "target Task A"))
        gain = reference_numerical_fit["gain"]
        common_gain = float(np.sum(self.reference.weights[:, None]*X*Y)
                            /np.sum(self.reference.weights[:, None]*X**2))
        if not np.isfinite(common_gain) or abs(common_gain-gain) > NUMERICAL_ATOL*max(1., abs(gain)):
            raise ValueError("reference gain must be the same pre-operator target-A common gain")
        O = _array(reference_numerical_fit["O"], (16, 16), "paired reference O")
        partitions = _array(source.geometry_state.source_partitions, (4, 18, self.support_size), "four source partitions")
        G = _array(source.G, (18, self.support_size), "shared source G")
        if not np.allclose(G, partitions.mean(axis=0), rtol=0, atol=NUMERICAL_ATOL):
            raise ValueError("shared G must be the mean of its actual four source partitions")
        result = []
        for specification in self.specifications:
            self.check()
            settings = dict(global_gain=gain, condition_weights=self.reference.weights,
                            mode_matrices=specification.modes, rank_bound=16,
                            dof_bound=specification.parameter_count,
                            rank_rtol=RANK_RTOL, numerical_atol=NUMERICAL_ATOL)
            objective = PairedM4TaskAObjective(X, Y, self.reference.B, O, **settings)
            C, numerical, certificate = _fit_with_certificate(
                objective, specification, X, Y, self.reference.B, O, settings, check=self.check)
            operator = self.operator_certificate(O, C, specification)
            zero = np.zeros((16, 16))
            zero_error = float(np.max(np.abs(gain*apply_m4(X, self.reference.B, O, zero)
                                           -gain*apply_m3(X, self.reference.B, O))))
            if zero_error > NUMERICAL_ATOL:
                raise ValueError("zero stretch fails paired reference M3 replay")
            predicted_partitions = gain*apply_m4(partitions.reshape(72, self.support_size), self.reference.B, O, C)
            predicted_partitions = predicted_partitions.reshape(4, 18, self.support_size)
            maps = gain*apply_m4(G, self.reference.B, O, C)
            geometry = crossvalidated_geometries(predicted_partitions[None])[0].mean(axis=0)
            if not np.isfinite(maps).all() or not np.isfinite(geometry).all():
                raise ArithmeticError("nonfinite paired M4 predictions")
            result.append(M4Prediction(specification.name, _frozen(maps), _frozen(geometry), {
                **certificate, **operator,
                "C": _frozen(C), "coefficients": _frozen(numerical["coefficients"]),
                "O": _frozen(O), "gain": gain, "training_ids": context.training_ids,
                "paired_model": "M3_reference_local_isometry",
                "spectral_norm_bound": specification.spectral_norm_bound,
                "rank_bound": 16, "per_cell_rank_bound": 2,
                "parameter_count_bound": specification.parameter_count,
                "degrees_of_freedom_interpretation": "fixed_parameter_count_and_prediction_tangent_rank_not_statistical_edf",
                "source_partitions_used": 4, "zero_stretch_replay_max_abs": zero_error,
                "local_optimizer_converged": numerical["local_optimizer_converged"],
                "optimizer_iterations": numerical["iterations"],
                "optimizer_ftol": OPTIMIZER_FTOL, "optimizer_max_iterations": MAX_ITERATIONS,
                "optimizer_role": ("primal_candidate_only" if "numerical_solver" in numerical
                                   else "curvature_certified_candidate"),
                "numerical_solver": numerical.get("numerical_solver", "SLSQP"),
                "numerical_solver_converged": numerical.get("numerical_solver_converged",
                                                            numerical["local_optimizer_converged"]),
                "global_optimum_certified": False, "unique_solution_certified": False,
                "native_stability_and_ablation_qualified": False,
            }))
        self.check()
        return tuple(result)


def _fixed_callbacks(coordinates, voxel_ids, check):
    fit_shared, calibration, controls = original_panel._fixed_callbacks(coordinates, voxel_ids, check)
    replacement = ProposedPairedStretchVariants(calibration["predict_m4"])
    return fit_shared, {**calibration, "predict_m4": replacement}, controls


def fit_development_point_panel(development, *, common_support, forward_header, check):
    """Unchanged complete panel orchestration using the prospective callback."""
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
