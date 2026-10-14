#!/usr/bin/env python3
"""Inspect the first Task-A certificate refusal through the unchanged reader.

Run only in the coordinator's Slurm allocation. This entrypoint changes no
fit or acceptance criterion. The ordinary runner still stops at its original
refusal; this diagnostic publishes no Task-B loss, prediction or participant
identity. Reported eigenvalues are floating-point diagnostics, not certificates.
"""
import argparse
from pathlib import Path

import run_development_point_panel as runner


def diagonal_diagnostic(objective, specification):
    import numpy as np
    from paired_stretch_variants import BLOCKS, CURVATURE_MARGIN, stretch_specifications
    from proposed_m4_certificate import _diagonal_minima

    canonical = stretch_specifications()[0]
    if (specification.name != canonical.name
            or not np.array_equal(specification.modes, canonical.modes)
            or specification.spectral_norm_bound != canonical.spectral_norm_bound):
        return {"scope": "non_diagonal_variant_not_diagnosed"}
    root_weight = np.sqrt(objective.weights)[:, None]
    a = objective.gain * root_weight * objective.rotated_source
    y = root_weight * objective.active_target
    p = np.sum(a * a, axis=0)
    d = np.sum(a * y, axis=0)
    energies = np.sum(a * a + y * y, axis=0)
    if (p.shape != (16,) or np.any(p <= 0)
            or not np.all(np.isfinite((p, d, energies)))):
        raise ValueError("qualified finite positive diagonal source required")
    rho = specification.spectral_norm_bound
    low, high = np.exp(-rho), np.exp(rho)
    vertex = np.clip(d / (4 * p), low, high)
    # Reuse the adopted outward-expanded scalar calculation, duplicated on a
    # two-coordinate diagonal only to obtain each individual lower bound.
    minima = np.array([_diagonal_minima(np.eye(2)*pi, np.eye(2)*di, rho)
                       for pi, di in zip(p, d)])
    H = np.diagonal(specification.modes, axis1=1, axis2=2).T
    lower_eigenvalues = np.linalg.eigvalsh(H.T @ (minima[:, None] * H))
    total_energy = float(energies.sum())
    # Each point obeys the exact mathematical box and trace constraints:
    # one coordinate is at its scalar-curvature minimum; the other fifteen
    # compensate equally. Include zero. A negative eigenvalue at such a
    # feasible point is evidence against global positive curvature, while
    # failure to find one does not establish convexity.
    witness_minimum = float("inf")
    witness_trace_error = witness_box_excess = 0.0
    for coordinate in range(-1, 16):
        c = np.zeros(16)
        if coordinate >= 0:
            value = float(np.log(vertex[coordinate]))
            c.fill(-value / 15)
            c[coordinate] = value
        hessian_diagonal = 4*p*np.exp(2*c) - 2*d*np.exp(c)
        eigenvalue = float(np.linalg.eigvalsh(H.T @ (hessian_diagonal[:, None]*H))[0])
        witness_minimum = min(witness_minimum, eigenvalue)
        witness_trace_error = max(witness_trace_error, abs(float(c.sum())))
        witness_box_excess = max(witness_box_excess, float(np.max(np.abs(c))-rho), 0.)
    return {
        "scope": "first_refused_anisotropic_Task_A_objective_only",
        "variant": specification.name,
        "reported_values_are_numerical_diagnostics_not_acceptance": True,
        "coordinate_curvature_lower_bounds_per_total_energy": (minima/total_energy).tolist(),
        "nonpositive_coordinate_lower_bounds": int(np.count_nonzero(minima <= 0)),
        "restricted_Helmert_lower_eigenvalues_per_total_energy": (lower_eigenvalues/total_energy).tolist(),
        "required_block_margins_per_total_energy": [
            float(CURVATURE_MARGIN*energies[start:stop].sum()/total_energy)
            for start, stop in BLOCKS],
        "feasible_point_count": 17,
        "minimum_feasible_point_eigenvalue_per_total_energy": witness_minimum/total_energy,
        "feasible_point_max_trace_roundoff": witness_trace_error,
        "feasible_point_max_box_roundoff": witness_box_excess,
        "negative_curvature_witness_at_original_margin": witness_minimum < -CURVATURE_MARGIN*total_energy,
        "negative_bound_does_not_prove_nonuniqueness": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-json", required=True, type=Path)
    args, remaining = parser.parse_known_args(argv)
    runner._compute()
    diagnostic_path = runner._path(args.diagnostic_json)
    if diagnostic_path.exists() or not diagnostic_path.parent.is_dir():
        parser.error("diagnostic output must be fresh under an existing directory")
    if runner.OUTPUTS / "runs" not in diagnostic_path.resolve().parents:
        parser.error("diagnostic output must be under this episode's outputs/runs")
    import proposed_development_point_panel as adapter
    from paired_stretch_variants import M4CertificateUnverified

    original = adapter.proposed_curvature_certificate
    captured = {}

    def inspect(objective, specification, *, check):
        try:
            return original(objective, specification, check=check)
        except M4CertificateUnverified:
            if not captured:
                check()
                captured.update(diagonal_diagnostic(objective, specification))
            raise

    adapter.proposed_curvature_certificate = inspect
    try:
        panel_exit = runner.main(remaining)
    finally:
        adapter.proposed_curvature_certificate = original
    report = dict(status="diagnosed_original_refusal" if captured else "no_refusal_captured",
                  original_panel_exit=panel_exit, acceptance_changed=False,
                  Task_B_scores_or_predictions_published=False, audit_values_read=False,
                  diagnostic=captured)
    runner._write_new(diagnostic_path, runner._encoded(report, runner.REPORT_BYTES))
    return 0 if captured else 2


if __name__ == "__main__":
    raise SystemExit(main())
