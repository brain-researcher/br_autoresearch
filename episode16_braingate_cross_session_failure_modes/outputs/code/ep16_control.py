"""EP16 known-positive control: long-gap-only injections of known size and their pass rules.

Extra loss: Gaussian noise added only to the predictions of the transported source mapping in long
pairs, equivalent to i.i.d. feature noise of variance 0.20 * J_t / ||beta_s,k||_F^2, so the expected
R increase is 0.20; one fixed noise realization per (pair, fold, evaluation position).
Local signal: a degraded copy of each long-pair target (noise on both sides, level found by bisection
on the observed data) used only in long pairs, so its long-pair mean fractional change falls by 0.30.
Both return a panel variant (alternative per-position loss tables), or an explicit unresolved variant.
Unresolved injections are failed controls and must not be evaluated as an uninjected panel.
Pass rules use the frozen primary-family bounds relative to the uninjected observed point.
"""
from __future__ import annotations

import numpy as np

EXTRA_SIZE, LOCAL_SIZE, MARGIN_EXTRA, MARGIN_LOCAL, TOLERANCE = 0.20, -0.30, 0.10, -0.15, 0.05


def _unresolved(reason, report=None):
    return {"status": "unresolved", "reason": reason}, {**(report or {}), "status": "unresolved", "reason": reason}


def extra_loss_injection(panel, size=EXTRA_SIZE, seed=2026):
    T = panel.tables
    fit = T["fit"]
    J = fit["J"]
    long = np.flatnonzero(T["long"])
    if not len(long):
        return _unresolved("no_long_pairs", {"size": size})
    zeta = np.random.default_rng(seed).standard_normal((len(T["src"]), 2, 64, 2))
    sigma2 = np.zeros((len(T["src"]), 2))
    for e in long:
        jt = J[T["tgt"][e]]
        if not np.isfinite(jt) or jt <= 0:
            return _unresolved("nonpositive_or_nonfinite_target_J", {"size": size, "edge": int(e)})
        for k in (0, 1):
            norm2 = float(np.sum(fit["beta"][T["src"][e], k] ** 2))
            if not np.isfinite(norm2) or norm2 <= 0:
                return _unresolved("zero_or_nonfinite_source_decoder_norm", {"size": size, "edge": int(e), "fold": k})
            sigma2[e, k] = size * jt / norm2
            if not np.isfinite(sigma2[e, k]) or sigma2[e, k] < 0:
                return _unresolved("invalid_noise_variance", {"size": size, "edge": int(e), "fold": k})

    def noise(rows, k, beta):
        gram = beta.T @ beta
        chol = np.linalg.cholesky(gram + (1e-12 * np.trace(gram) + 1e-300) * np.eye(2))
        return np.sqrt(sigma2[rows, k])[:, None, None] * (zeta[rows, k] @ chol.T)

    lossF = panel.transported_losses(T["src"], T["tgt"], fit, extra_noise=noise)
    return {"status": "resolved", "lossF": lossF}, {
        "status": "resolved", "size": size, "sigma2_median": float(np.median(sigma2[T["long"]]))}


def local_signal_injection(panel, size=LOCAL_SIZE, seed=1616, iterations=40):
    """Per long-pair target: noise level tau with J'(tau) = J_t + size * H_t, H_t the harmonic mean of
    its long sources' J. Noise and per-side channel SDs are fixed per session (common random numbers).
    Nonpositive/nonfinite source J or an unbracketed goal produces an explicitly unresolved variant.
    """
    T = panel.tables
    fit = T["fit"]
    J = fit["J"]
    X, Y, w = panel.stacked()
    Xd = X.copy()
    report = {"status": "resolved"}
    targets = np.unique(T["tgt"][T["long"]])
    if not len(targets):
        return _unresolved("no_long_pairs", report)
    for t in targets:
        day = int(panel.sessions[t].day)
        sources = T["src"][T["long"] & (T["tgt"] == t)]
        if not np.isfinite(J[sources]).all() or np.any(J[sources] <= 0):
            return _unresolved("nonpositive_or_nonfinite_source_J", {**report, "failed_target_day": day})
        goal = J[t] + size / np.mean(1.0 / J[sources])
        if not np.isfinite(J[t]) or not np.isfinite(goal):
            return _unresolved("nonfinite_target_J_or_goal", {**report, "failed_target_day": day})
        xi = np.random.default_rng([seed, int(t)]).standard_normal(X[t].shape)
        sdev = np.stack([np.nanstd(X[t, k], axis=0) for k in (0, 1)])
        sdev = np.where(np.isfinite(sdev), sdev, 0.0)

        def J_at(level):
            return float(panel.fit_sessions((X[t] + level * sdev[:, None, :] * xi)[None], Y[t][None], w[t][None])["J"][0])

        lo, hi = 0.0, 1.0
        j_lo, j_hi = float(J[t]), J_at(hi)
        while np.isfinite(j_hi) and j_hi > goal and hi < 2 ** 12:
            hi *= 2.0
            j_hi = J_at(hi)
        if not np.isfinite(j_hi) or not (j_hi <= goal <= j_lo):
            return _unresolved("goal_not_bracketed", {**report, "failed_target_day": day, "goal": float(goal),
                                                     "J_at_zero": j_lo, "J_at_upper": j_hi, "upper_tau": hi})
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            j_mid = J_at(mid)
            if not np.isfinite(j_mid):
                return _unresolved("nonfinite_J_during_bisection", {**report, "failed_target_day": day})
            lo, hi = (mid, hi) if j_mid > goal else (lo, mid)
        level = 0.5 * (lo + hi)
        achieved = J_at(level)
        if not np.isfinite(achieved):
            return _unresolved("nonfinite_achieved_J", {**report, "failed_target_day": day})
        Xd[t] = X[t] + level * sdev[:, None, :] * xi
        report[day] = {"J": float(J[t]), "goal": float(goal), "achieved": achieved, "tau": level}
    degraded = panel.fit_sessions(Xd, Y, w)
    lossF = T["lossF"].copy()
    long = np.flatnonzero(T["long"])
    lossF[long] = panel.transported_losses(T["src"][long], T["tgt"][long], fit, X=Xd)
    return {"status": "resolved", "lossF": lossF, "lossL_degraded": degraded["lossL"]}, report


def _finite(value):
    return value is not None and np.isfinite(value)


def extra_loss_pass(observed_point, injected_point, injected_lower, margin=MARGIN_EXTRA, size=EXTRA_SIZE, tolerance=TOLERANCE):
    result = {}
    for name in ("adjusted", "robust"):
        o, i, lb = observed_point[name], injected_point[name], injected_lower[name]
        resolves = _finite(o) and _finite(i) and _finite(lb) and lb - o > margin
        unbiased = _finite(o) and _finite(i) and abs(i - o - size) <= tolerance
        result[name] = {"observed": o, "injected": i, "injected_lower_bound": lb,
                        "lower_bound_minus_observed": lb - o if _finite(o) and _finite(lb) else None,
                        "recovered_size": i - o if _finite(o) and _finite(i) else None,
                        "resolves_margin": bool(resolves), "unbiased": bool(unbiased)}
    result["pass"] = all(result[n]["resolves_margin"] and result[n]["unbiased"] for n in ("adjusted", "robust"))
    return result


def local_signal_pass(observed_point, injected_point, injected_upper, raw_margin, margin=MARGIN_LOCAL, size=LOCAL_SIZE,
                      tolerance=TOLERANCE):
    o_f, i_f, ub_f = observed_point["fractional_local"], injected_point["fractional_local"], injected_upper["fractional_local"]
    o_r, i_r, ub_r = observed_point["raw_transported"], injected_point["raw_transported"], injected_upper["raw_transported"]
    f_ok = _finite(o_f) and _finite(i_f) and _finite(ub_f) and ub_f - o_f < margin
    r_ok = _finite(o_r) and _finite(i_r) and _finite(ub_r) and ub_r - o_r < raw_margin
    unbiased = _finite(o_f) and _finite(i_f) and abs(i_f - o_f - size) <= tolerance
    result = {"fractional_local": {"observed": o_f, "injected": i_f, "injected_upper_bound": ub_f,
                                   "upper_bound_minus_observed": ub_f - o_f if _finite(o_f) and _finite(ub_f) else None,
                                   "recovered_size": i_f - o_f if _finite(o_f) and _finite(i_f) else None,
                                   "resolves_margin": bool(f_ok), "unbiased": bool(unbiased)},
              "raw_transported": {"observed": o_r, "injected": i_r, "injected_upper_bound": ub_r,
                                  "upper_bound_minus_observed": ub_r - o_r if _finite(o_r) and _finite(ub_r) else None,
                                  "recovered_size": i_r - o_r if _finite(o_r) and _finite(i_r) else None,
                                  "resolves_margin": bool(r_ok), "raw_margin": raw_margin}}
    result["pass"] = bool(f_ok and r_ok and unbiased)
    return result
