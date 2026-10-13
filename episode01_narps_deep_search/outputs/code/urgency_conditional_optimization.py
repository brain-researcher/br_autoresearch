"""One unchanged conditional participant block; no population fit or I/O.

The arithmetic and optimizer options follow hierarchical_behavior's subject
loop. An explicit likelihood permits the separately qualified batched urgency
implementation without mutating the reference fitter or any module globals.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize


def number(value):
    """JSON-safe numeric value, accompanied by explicit finiteness elsewhere."""
    value = float(value)
    return value if math.isfinite(value) else None


def optimize_subject_block(data, names, spec, mean, sd, starts, *, likelihood,
                           attempts, check=lambda: None, checkpoint=lambda: None):
    """Run starts sequentially, preserving returned and interrupted attempts.

    ``attempts`` is the caller-owned operational record. RuntimeError (including
    the caller's deadline) propagates; only the two reference numerical-error
    classes turn an objective into infinity. No success-based start filtering
    or additional objective evaluations are introduced.
    """
    optimizer = spec["optimizer"]
    bounds = [spec["parameters"][name]["bounds"] for name in names]
    results = []
    for start_index, start in enumerate(starts):
        check()
        record = {"start_index": start_index, "start_parameters": np.asarray(start).tolist(),
            "optimizer_entered": False, "returned": False, "elapsed_seconds": 0., "objective_calls": 0,
            "likelihood_calls_attempted": 0, "likelihood_calls_returned": 0,
            "likelihood_seconds": 0., "numerical_error_calls": 0,
            "nfev": None, "njev": None, "nit": None, "status": None, "success": None,
            "message": None, "parameters": None, "parameters_finite": None,
            "objective": None, "objective_finite": None, "nonfinite_reason": None,
            "support_contacts": []}
        attempts.append(record)
        began = time.perf_counter()
        checkpoint()

        def objective(candidate):
            check()
            record["objective_calls"] += 1
            try:
                record["likelihood_calls_attempted"] += 1
                call_started = time.perf_counter()
                try:
                    values = likelihood(candidate, names, data, spec)
                    record["likelihood_calls_returned"] += 1
                    nll = -values.sum()
                finally:
                    record["likelihood_seconds"] += time.perf_counter() - call_started
            except (FloatingPointError, ValueError):
                record["numerical_error_calls"] += 1
                return np.inf
            return float(nll + .5*np.sum(((candidate-mean)/sd)**2))

        try:
            check()
            record["optimizer_entered"] = True
            result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": optimizer["subject_maximum_iterations"],
                         "ftol": optimizer["subject_ftol"], "gtol": optimizer["subject_gtol"]})
            results.append(result)
            finite_x = bool(np.isfinite(result.x).all())
            finite_fun = bool(np.isfinite(result.fun))
            record.update(returned=True, nfev=int(result.nfev),
                njev=int(result.njev) if getattr(result, "njev", None) is not None else None,
                nit=int(result.nit), status=int(result.status), success=bool(result.success),
                message=str(result.message), parameters=[number(value) for value in result.x],
                parameters_finite=finite_x, objective=number(result.fun), objective_finite=finite_fun,
                nonfinite_reason=None if finite_x and finite_fun else "nonfinite_optimizer_return")
            for name, value, (low, high) in zip(names, result.x, bounds):
                tolerance = 1e-6*(1+high-low)
                if np.isfinite(value) and (value-low <= tolerance or high-value <= tolerance):
                    record["support_contacts"].append({"parameter": name,
                        "value": float(value), "bounds": [low, high]})
        except Exception as error:
            record["interruption"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - began
            checkpoint()
        check()
    finite = [(index, result) for index, result in enumerate(results) if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("No finite conditional fit across the original starts")
    index, best = min(finite, key=lambda pair: pair[1].fun)
    return {"start_index": index, "parameters": [number(value) for value in best.x],
        "parameters_finite": bool(np.isfinite(best.x).all()), "objective": number(best.fun),
        "objective_finite": True, "success": bool(best.success), "message": str(best.message)}
