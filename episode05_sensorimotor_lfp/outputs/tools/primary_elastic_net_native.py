"""Native single-sweep acceleration; original Python fitting/stopping semantics."""
import ctypes
from pathlib import Path
import random

import numpy as np

from primary_elastic_net import (
    ANCHOR, RECIPES, REPLICATES, MAX_SWEEPS, LAMBDAS, DirectionTime,
    numeric_trial_ids, penalty_scale, kkt_certificate, trial_map,
    prepare_split, score_bank, average_control_banks, summarize_banks,
)

BACKEND = "c99_single_sweep_v1_original_numpy_KKT"
_LIBRARY = None


def native_library():
    global _LIBRARY
    if _LIBRARY is None:
        path = Path(__file__).with_name("elastic_net_native_kernel.so")
        library = ctypes.CDLL(str(path))
        matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED"))
        writable = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                                         flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE"))
        library.ep05_elastic_net_sweep.argtypes = [ctypes.c_size_t, matrix, matrix,
                                                  ctypes.c_double, ctypes.c_double, writable]
        library.ep05_elastic_net_sweep.restype = ctypes.c_int
        _LIBRARY = library
    return _LIBRARY


def cyclic_sweep(gram, cross, penalty, scale, coefficients):
    """Mutate W by exactly one physical-order sweep; no stopping decision."""
    if (gram.ndim != 2 or gram.shape[0] < 1 or gram.shape != (len(gram), len(gram))
            or cross.shape != (len(gram), 2) or coefficients.shape != cross.shape):
        raise ValueError("Exact physical Gram and paired native XY matrices required")
    status = native_library().ep05_elastic_net_sweep(len(gram), gram, cross,
                                                   penalty, scale, coefficients)
    if status:
        raise ArithmeticError(f"Native single-sweep kernel rejected nonfinite/invalid arithmetic, status={status}")


def solve_path(gram, cross, scale, check=lambda: None, max_sweeps=MAX_SWEEPS):
    """Cyclic physical-coordinate native-XY solves; fixed descending warm starts."""
    g, c = np.ascontiguousarray(gram, dtype=np.float64), np.ascontiguousarray(cross, dtype=np.float64)
    if (g.ndim != 2 or g.shape[0] < 1 or g.shape[0] != g.shape[1]
            or c.shape != (len(g), 2) or not np.isfinite(g).all() or not np.isfinite(c).all()
            or not np.isfinite(scale) or scale < 0 or type(max_sweeps) is not int
            or not 1 <= max_sweeps <= MAX_SWEEPS or np.any(np.diag(g) < 0)
            or not np.allclose(g, g.T, rtol=1e-13, atol=1e-13)):
        raise ValueError("Finite symmetric physical Gram, native XY covariance and bounded sweep count required")
    coefficients = np.zeros_like(c)
    path = np.empty((len(LAMBDAS), *c.shape))
    diagnostics = [None]*len(LAMBDAS)
    for index in reversed(range(len(LAMBDAS))):
        penalty = LAMBDAS[index]
        threshold = penalty*scale
        for sweep in range(1, max_sweeps+1):
            check()
            cyclic_sweep(g, c, penalty, scale, coefficients)
            if not np.isfinite(coefficients).all():
                raise ArithmeticError("Nonfinite elastic-net coefficients")
            certificate = kkt_certificate(g, c, coefficients, penalty, scale)
            if certificate["converged"]:
                active = coefficients != 0
                diagnostics[index] = {"lambda": penalty, "completed_sweeps": sweep, "sweep_backend": BACKEND,
                    "maximum_sweeps": MAX_SWEEPS, "penalty_scale": float(scale),
                    "L1_penalty_coefficient": threshold, "L2_penalty_coefficient": penalty,
                    "zero_coefficients": int(np.count_nonzero(~active)),
                    "active_X": int(np.count_nonzero(active[:, 0])),
                    "active_Y": int(np.count_nonzero(active[:, 1])),
                    "active_channel_union": int(np.count_nonzero(np.any(active, axis=1))),
                    "active_channel_intersection": int(np.count_nonzero(np.all(active, axis=1))),
                    "coefficient_L1_norm": float(np.abs(coefficients).sum()),
                    "coefficient_Frobenius_norm": float(np.linalg.norm(coefficients)),
                    **certificate}
                path[index] = coefficients
                break
        else:
            raise RuntimeError(f"Elastic-net lambda={penalty} did not satisfy KKT certificate within {max_sweeps} sweeps")
    return path, diagnostics



def fit_banks(features, targets, directions, trial_ids, session, check=lambda: None):
    x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    labels, ids = np.asarray(directions), np.asarray(trial_ids)
    numeric_trial_ids(ids, session)
    if (x.ndim != 3 or x.shape[:2] != (len(y), 15) or x.shape[-1] < 1
            or y.shape != (len(y), 15, 2) or labels.shape != (len(y),) or ids.shape != (len(y),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(labels.tolist()) != set(range(8))
            or any(np.count_nonzero(labels == d) < 2 for d in range(8))):
        raise ValueError("Complete original current-bin LMP/XY/trial support required")
    baseline = np.full_like(y, np.nan)
    paired = np.full((len(LAMBDAS), *y.shape), np.nan)
    shuffled = np.full((len(REPLICATES), len(LAMBDAS), *y.shape), np.nan)
    metadata = []
    for held in range(8):
        check()
        train, test = labels != held, labels == held
        template = DirectionTime.fit(labels[train]*np.pi/4, y[train], 1, labels[train])
        residual = (y[train]-template.predict(labels[train]*np.pi/4)).reshape(-1, 2)
        intercept = residual.mean(axis=0)
        true_u = residual-intercept
        base = template.predict(labels[test]*np.pi/4)+intercept
        baseline[test] = base
        z, state = prepare_split(x, train)
        fit_z = z[train].reshape(-1, x.shape[-1])
        eval_z = z[test].reshape(-1, x.shape[-1])
        gram = fit_z.T@fit_z/len(fit_z)
        scale = penalty_scale(true_u)
        diagnostics, maps = [], []
        for arm in range(1+len(REPLICATES)):
            check()
            arm_z = fit_z
            if arm:
                mapping, details = trial_map(ids[train], labels[train], session, held, arm-1)
                arm_z = z[train][mapping].reshape(-1, x.shape[-1])
                maps.append({"replicate": arm-1, "directions": details})
            # Exact same physical Gram; only training correspondence changes C.
            cross = arm_z.T@true_u/len(true_u)
            coefficients, fit_info = solve_path(gram, cross, scale, check)
            diagnostics.append(fit_info)
            for index in range(len(LAMBDAS)):
                value = base+(eval_z@coefficients[index]).reshape(base.shape)
                if arm == 0:
                    paired[index, test] = value
                else:
                    shuffled[arm-1, index, test] = value
        metadata.append({"held_direction": held, "fitting_bin_rows": len(true_u),
            "physical_mean": state["physical_mean"].tolist(), "physical_sd": state["physical_sd"].tolist(),
            "physical_zero_SD_columns": np.flatnonzero(state["physical_sd"] == 0).tolist(),
            "common_velocity_penalty_scale": scale, "physical_Gram_shared_exactly": True,
            "paired_fit_diagnostics_by_lambda": diagnostics[0],
            "control_fit_diagnostics_by_replicate_lambda": diagnostics[1:],
            "training_trial_maps_by_replicate": maps})
    candidate = score_bank(y, labels, baseline, paired, RECIPES[0])
    candidate.pop("ridge_fits", None)
    candidate.update(solver_backend=BACKEND, physical_electrode_count=x.shape[-1], direction_fit_metadata=metadata,
        elastic_net_two_response_fits=8*len(LAMBDAS), common_velocity_penalty_scale=True,
        native_XY_scaled=False, coordinatewise_L1=True, maximum_sweeps=MAX_SWEEPS,
        all_folds_zero_residual_scale=all(row["common_velocity_penalty_scale"] == 0 for row in metadata))
    controls = [score_bank(y, labels, baseline, shuffled[r], RECIPES[1]) for r in REPLICATES]
    control = average_control_banks(controls)
    control.update(training_LMP_permuted_as_whole_trials=True, physical_normalization_preserved=True,
                   fitting_targets_unpermuted=True, physical_Gram_shared_exactly=True,
                   diagnostics_location="candidate direction_fit_metadata")
    return {RECIPES[0]: candidate, RECIPES[1]: control}, baseline, {RECIPES[0]: paired, RECIPES[1]: shuffled}
