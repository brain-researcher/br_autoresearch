"""EP16 conditional branches and controls: learning curves, 32-label recalibration, label-blind
recording emulator with matched-random control, unlabelled tier, shifted-label null and leakage checks.

Every method is fit once on the realized fitting trials (first n positions of the fitting side) and
stored as per-position evaluation losses (edges, 2 evaluation sides, 64); bootstrap realizations
resample evaluation positions only. Methods and limits follow analysis_specification.json.
"""
from __future__ import annotations

import numpy as np

from ep16_models import Unresolved, fit_scaler, ridge, score, standardize
from ep16_panel import PACKET, adjusted_contrast, lag_weighted_contrast, octant_from_labels, octant_weights_or_nan, target_weights

BOUND_LOW, BOUND_HIGH = 1.0 / 3.0, 3.0
SUPERVISED = ("output_gain_bias", "output_rotation", "output_similarity", "low_rank_2", "low_rank_4", "bounded_supervised")
UNLABELLED = ("past_only", "whole_session", "covariance", "subspace")


# ----------------------------------------------------------------------------- supervised pieces
def _weighted_center(P, w):
    mean = w @ P
    return P - mean, mean


def _rotation(H):
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, d]) @ U.T


def output_correction(kind, P_fit, Y_fit, w):
    """Fit a bounded 2D output correction; returns (function on evaluation predictions, bound active)."""
    Pc, pm = _weighted_center(P_fit, w)
    Yc, ym = _weighted_center(Y_fit, w)
    if kind == "output_gain_bias":
        var = (w[:, None] * Pc ** 2).sum(axis=0)
        cov = (w[:, None] * Pc * Yc).sum(axis=0)
        raw = np.divide(cov, var, out=np.ones(2), where=var > 0)
        gain = np.clip(raw, BOUND_LOW, BOUND_HIGH)
        return (lambda P: (P - pm) * gain + ym), int(np.any(gain != raw))
    H = (w[:, None] * Pc).T @ Yc
    R = _rotation(H)
    if kind == "output_rotation":
        return (lambda P: (P - pm) @ R.T + ym), 0
    if kind == "output_similarity":
        denom = float((w * (Pc ** 2).sum(axis=1)).sum())
        s = float(np.trace(R @ H)) / denom if denom > 0 else 1.0
        clipped = float(np.clip(s, BOUND_LOW, BOUND_HIGH))
        return (lambda P: clipped * (P - pm) @ R.T + ym), int(clipped != s)
    raise ValueError(kind)


def _df_lambda(Z, w, grid, limit):
    """Smallest grid value whose effective degrees of freedom do not exceed the limit."""
    A = (Z - w @ Z) * np.sqrt(w)[:, None]
    eig = np.clip(np.linalg.eigvalsh(A @ A.T), 0, None)
    for lam in sorted(grid):
        if (eig / (eig + lam)).sum() <= limit:
            return lam
    raise Unresolved("No ridge-grid value satisfies the effective-df limit")


def decoder_update(kind, Z_fit, Y_fit, w, beta, c, lam, grid):
    """Low-rank or bounded supervised update of the frozen source decoder, ||Delta||_F <= ||beta||_F."""
    resid = Y_fit - (Z_fit @ beta + c)
    n = len(Y_fit)
    if kind.startswith("low_rank"):
        r = int(kind.split("_")[-1])
        _, _, Vt = np.linalg.svd(Z_fit - Z_fit.mean(axis=0), full_matrices=False)
        U = Vt[:r].T
        G, g0 = ridge((Z_fit @ U)[None], resid[None], w[None], lam)
        delta, dc = U @ G[0], g0[0]
    else:
        lam_b = _df_lambda(Z_fit, w, grid, n / 2.0)
        D, d0 = ridge(Z_fit[None], resid[None], w[None], lam_b)
        delta, dc = D[0], d0[0]
    norm, limit = np.linalg.norm(delta), np.linalg.norm(beta)
    active = int(norm > limit)
    if active:
        projected = delta * (limit / norm)
        dc = dc + (w @ Z_fit) @ (delta - projected)
        delta = projected
    return beta + delta, c + dc, active


def _edge_rows(panel, rows):
    T = panel.tables
    return np.arange(len(T["src"])) if rows is None else np.asarray(rows)


def repair_table(panel, method, n=32, rows=None, shuffle_seed=None):
    """Per-position A_st(n) losses; NaN for missing fitting octants or an unresolved df constraint."""
    T = panel.tables
    fit = T["fit"]
    X, Y, _ = panel.stacked()
    grid = [panel.lam * 10 ** (k / 4) for k in range(25)]
    table = np.full((len(T["src"]), 2, PACKET), np.nan)
    active = 0
    for e in _edge_rows(panel, rows):
        s, t = T["src"][e], T["tgt"][e]
        for k in (0, 1):
            ev = 1 - k
            Xf, Yf = X[t, k, :n], Y[t, k, :n]
            if shuffle_seed is not None:
                Yf = Yf[np.random.default_rng([shuffle_seed, int(e), k]).permutation(n)]
            w = octant_weights_or_nan(octant_from_labels(Yf))
            if not np.isfinite(w).all():
                continue
            sc = (fit["center"][s, k][None], fit["scale"][s, k][None], fit["usable"][s, k][None])
            Zf = standardize(Xf[None], *sc)[0]
            Ze = standardize(X[t, ev][None], *sc)[0]
            beta, c = fit["beta"][s, k], fit["c"][s, k]
            try:
                if method.startswith("output"):
                    correct, flag = output_correction(method, Zf @ beta + c, Yf, w)
                    P = correct(Ze @ beta + c)
                else:
                    b2, c2, flag = decoder_update(method, Zf, Yf, w, beta, c, panel.lam, grid)
                    P = Ze @ b2 + c2
            except Unresolved:
                continue
            active += flag
            table[e, ev] = ((Y[t, ev] - P) ** 2).sum(axis=-1)
    return table, active


def repair_statistics(real, table, J_min, D_min, n=32):
    """Long-gap recovery fraction and relative gain as ratios of target-equal weighted sums.

    Population (pre-score label support): long pairs whose target has octant-complete first-n
    fitting subsets on both sides; other long pairs are counted, never scored.
    """
    Lt, Bt, Jt = real.edge_target_scores()
    covered = np.array([all(real.panel.sessions[int(real.occ[t])].cover[n]) for t in real.tgt])
    long = np.flatnonzero(real.long & covered)
    out = {"recovery_fraction": None, "relative_gain": None}
    excluded = int((real.long & ~covered).sum())
    if len(long) < 6:
        return out, {"reason": "fewer_than_6_long_pairs_with_resolved_fitting_subsets", "excluded_long_pairs": excluded}
    A = real.edge_scores(table, "target", long)
    F = real.F[long]
    G, D, J = A - F, Lt[long] - F, Jt[long]
    w = target_weights(real.tgt[long])
    if not np.isfinite(A).all() or not np.all(J > J_min):
        return out, {"reason": "unresolved_packet_or_readiness", "excluded_long_pairs": excluded}
    sumD = float(w @ D)
    if not sumD > D_min:
        return out, {"reason": "aggregate_gap_not_above_D_min", "sum_wD": sumD}
    out["recovery_fraction"] = float(w @ G) / sumD
    out["relative_gain"] = float(w @ G) / float(w @ J)
    return out, {"sum_wG": float(w @ G), "sum_wD": sumD, "sum_wJ": float(w @ J), "long_pairs": int(len(long)),
                 "excluded_long_pairs": excluded}


def learning_curve(panel, ns=(16, 32)):
    X, Y, w = panel.stacked()
    return {n: panel.fit_sessions(X, Y, w, n=n)["L"] for n in ns}


# ----------------------------------------------------------------------------- recording emulator
def _summaries(Xs, Xt, center, scale, usable, avail_s, avail_t):
    rate_s = np.isfinite(Xs).mean(axis=0) * avail_s
    rate_t = np.isfinite(Xt).mean(axis=0) * avail_t
    Zt = standardize(Xt[None], center[None], scale[None], usable[None])[0]
    finite = np.isfinite(Xt) & usable[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(finite, Zt, 0).sum(axis=0) / finite.sum(axis=0)
        var = (np.where(finite, Zt - mean, 0) ** 2).sum(axis=0) / finite.sum(axis=0)
    return rate_s, rate_t, var


def no_better(avail_s, avail_t, rate_s, rate_t):
    """Direct observable order: target available set within source and no shared rate higher."""
    if np.any(avail_t & ~avail_s):
        return False
    shared = avail_s & avail_t
    return bool(np.all(rate_t[shared] <= rate_s[shared]))


FAMILIES = ("E1_dropout", "E2_theta0", "E2_theta0_floor", "E2_theta0.5")


def emulator_parameters(family, avail_s, avail_t, usable, rate_s, rate_t, var, n_s=None, n_t=None):
    """Drop, invalid-window mask, attenuation and added-noise variance per channel (fitting sides only).

    E2_theta0_floor treats a variance ratio inside its 95% sampling band, |log v| <= 1.96 *
    sqrt(2/(n_s-1) + 2/(n_t-1)), as unchanged (a=1, no noise), so finite-sample variance noise is not
    emulated as degradation."""
    drop = (avail_s & ~avail_t) | ~usable
    with np.errstate(invalid="ignore", divide="ignore"):
        mask = np.where(rate_s > 0, np.clip((rate_s - rate_t) / rate_s, 0, 1), 0.0)
    a, sigma2 = np.ones_like(var), np.zeros_like(var)
    if family != "E1_dropout":
        theta = {"E2_theta0": 0.0, "E2_theta0_floor": 0.0, "E2_theta0.5": 0.5}[family]
        v = np.where(np.isfinite(var) & (var >= 0), var, 1.0)
        if family == "E2_theta0_floor":
            with np.errstate(invalid="ignore", divide="ignore"):
                band = 1.96 * np.sqrt(2.0 / np.maximum(n_s - 1, 1) + 2.0 / np.maximum(n_t - 1, 1))
                v = np.where(np.abs(np.log(v)) <= band, 1.0, v)
        a2 = (1 - theta) * np.minimum(v, 1.0)
        a, sigma2 = np.sqrt(a2), np.clip(v - a2, 0, None)
    return drop, mask, a, sigma2


def _apply(Z, params, rng, observed):
    drop, mask, a, sigma2 = params
    keep = observed & (rng.random(Z.shape) >= mask[None, :]) & ~drop[None, :]
    return np.where(keep, a[None, :] * Z + np.sqrt(sigma2)[None, :] * rng.standard_normal(Z.shape), 0.0)


def _strata(array_labels, rate_s):
    if np.ptp(rate_s) > 0:
        tier = np.digitize(rate_s, np.quantile(rate_s, [1 / 3, 2 / 3]), right=True)
    else:
        tier = np.zeros(len(rate_s), int)
    groups = [np.flatnonzero((array_labels == a) & (tier == q)) for a in np.unique(array_labels) for q in np.unique(tier)]
    return [g for g in groups if len(g)]


def emulator_tables(panel, family, array_labels, permutations=20, seed=16, rows=None):
    """Per-position losses on SOURCE evaluation trials: emulated-local and clean-source models on
    transformed source features, identity-aware and matched-random (losses averaged over permutations)."""
    T = panel.tables
    fit = T["fit"]
    X, Y, w = panel.stacked()
    avail = np.stack([s.available for s in panel.sessions])
    shape = (len(T["src"]), 2, PACKET)
    out = {key: np.full(shape, np.nan) for key in ("lossLe", "lossFe", "lossLr", "lossFr")}
    eligible = np.zeros(len(T["src"]), dtype=bool)
    for e in _edge_rows(panel, rows):
        s, t = T["src"][e], T["tgt"][e]
        order = []
        for k in (0, 1):
            ev = 1 - k
            rng = np.random.default_rng([seed, int(s), int(t), k])
            sc = (fit["center"][s, k], fit["scale"][s, k], fit["usable"][s, k])
            rate_s, rate_t, var = _summaries(X[s, k], X[t, k], *sc, avail[s], avail[t])
            order.append(no_better(avail[s], avail[t], rate_s, rate_t))
            Zs = standardize(X[s][None], sc[0][None, None], sc[1][None, None], sc[2][None, None])[0]
            observed = np.isfinite(X[s]) & sc[2][None, None, :] & avail[s][None, None, :]
            params = emulator_parameters(family, avail[s], avail[t], sc[2], rate_s, rate_t, var,
                                         np.isfinite(X[s, k]).sum(axis=0), np.isfinite(X[t, k]).sum(axis=0))
            variants = [params]
            strata = _strata(array_labels, rate_s)
            for _ in range(permutations):
                perm = np.arange(len(var))
                for members in strata:
                    perm[members] = rng.permutation(members)
                variants.append(tuple(p[perm] for p in params))
            fit_x = np.stack([_apply(Zs[k], v, rng, observed[k]) for v in variants])
            eval_x = np.stack([_apply(Zs[ev], v, rng, observed[ev]) for v in variants])
            cf, sf, uf = fit_scaler(fit_x)
            beta, c = ridge(standardize(fit_x, cf, sf, uf), np.broadcast_to(Y[s, k], fit_x.shape[:2] + (2,)),
                            np.broadcast_to(w[s, k], fit_x.shape[:2]), panel.lam)
            P_local = standardize(eval_x, cf, sf, uf) @ beta + c[:, None, :]
            P_clean = eval_x @ fit["beta"][s, k] + fit["c"][s, k]
            loss_local = ((Y[s, ev][None] - P_local) ** 2).sum(axis=-1)
            loss_clean = ((Y[s, ev][None] - P_clean) ** 2).sum(axis=-1)
            out["lossLe"][e, ev], out["lossFe"][e, ev] = loss_local[0], loss_clean[0]
            out["lossLr"][e, ev], out["lossFr"][e, ev] = loss_local[1:].mean(axis=0), loss_clean[1:].mean(axis=0)
        eligible[e] = all(order)
    out["eligible"] = eligible
    return out


def recording_statistics(real, em, J_min):
    """Maximum absolute reproduction error (identity-aware) and random-minus-identity separation."""
    Lt, Bt, Jt = real.edge_target_scores()
    src, tgt, long = real.src, real.tgt, real.long
    Js, Bs = real.local["J"][src], real.local["B"][src]
    eligible = em["eligible"][real.e]
    out = {"identity_max_reproduction_error": None, "random_minus_identity_max_error": None}
    info = {"eligible_near": int((eligible & ~long).sum()), "eligible_long": int((eligible & long).sum())}
    if info["eligible_near"] < 6 or info["eligible_long"] < 6:
        return out, {**info, "reason": "fewer_than_6_near_or_long_eligible_transitions"}
    e = np.flatnonzero(eligible)
    if not (np.all(Js[e] > J_min) and np.all(Jt[e] > J_min)):
        return out, {**info, "reason": "observed_readiness_failure"}
    obs_local, obs_loss = (Jt[e] - Js[e]) / Js[e], (Lt[e] - real.F[e]) / Jt[e]
    _, m_obs_local = lag_weighted_contrast(tgt[e], long[e], obs_local)
    _, m_obs_loss = lag_weighted_contrast(tgt[e], long[e], obs_loss)
    c_obs = adjusted_contrast(src[e], tgt[e], long[e], obs_loss)
    errors = {}
    for tag in ("e", "r"):
        Le = real.edge_scores(em[f"lossL{tag}"], "source", e)
        Fe = real.edge_scores(em[f"lossF{tag}"], "source", e)
        Je = Le - Bs[e]
        if not np.all(Je > J_min) or not np.isfinite(Fe).all():
            return out, {**info, "reason": "emulated_readiness_failure"}
        emu_local, emu_loss = (Je - Js[e]) / Js[e], (Le - Fe) / Je
        _, m_emu_local = lag_weighted_contrast(tgt[e], long[e], emu_local)
        _, m_emu_loss = lag_weighted_contrast(tgt[e], long[e], emu_loss)
        c_emu = adjusted_contrast(src[e], tgt[e], long[e], emu_loss)
        if c_obs is None or c_emu is None:
            return out, {**info, "reason": "eligible_subgraph_rank_deficient"}
        parts = {"near_local": m_emu_local[False] - m_obs_local[False], "long_local": m_emu_local[True] - m_obs_local[True],
                 "near_loss": m_emu_loss[False] - m_obs_loss[False], "long_loss": m_emu_loss[True] - m_obs_loss[True],
                 "extra": c_emu - c_obs}
        errors[tag] = max(abs(v) for v in parts.values())
        info[f"components_{tag}"] = parts
    out["identity_max_reproduction_error"] = errors["e"]
    out["random_minus_identity_max_error"] = errors["r"] - errors["e"]
    return out, info


# ----------------------------------------------------------------------------- unlabelled tier
def _restandardize(X, mean, sd, fallback, scale):
    ok = np.isfinite(mean) & np.isfinite(sd) & (sd > 1e-6 * scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (X - mean) / np.where(ok, sd, 1.0)
    Z = np.where(ok & np.isfinite(X), Z, fallback)
    return np.where(np.isfinite(Z), Z, 0.0)


def unlabelled_table(panel, method, n=32, history=32, alpha=0.5, k_sub=8, rows=None):
    """Per-position losses for the unlabelled tier on target evaluation trials (labels never fit).

    past_only: each evaluation trial re-standardized with the mean/SD of the `history` most recent
    packet trials (either side) preceding it in time, requiring max(2, history//2) finite values per
    channel, otherwise falling back to source scaling. With block-normalized inputs this is
    history-restricted restandardization of block-normalized features, not literal past-only access.
    whole_session: all 128 packet trials (descriptive upper bound); covariance: bounded CORAL from
    the first n target fitting trials; subspace: top-k principal subspace alignment from those trials.
    """
    T = panel.tables
    fit = T["fit"]
    X, Y, _ = panel.stacked()
    table = np.full((len(T["src"]), 2, PACKET), np.nan)
    for e in _edge_rows(panel, rows):
        s, t = T["src"][e], T["tgt"][e]
        session = panel.sessions[t]
        for k in (0, 1):
            ev = 1 - k
            center, scale, usable = fit["center"][s, k], fit["scale"][s, k], fit["usable"][s, k]
            beta, c = fit["beta"][s, k], fit["c"][s, k]
            Xe = X[t, ev]
            Zsrc = standardize(Xe[None], center[None], scale[None], usable[None])[0]
            if method in ("past_only", "whole_session"):
                times = session.time.reshape(-1)
                pool = X[t].reshape(2 * PACKET, -1)
                Z = np.empty_like(Zsrc)
                for p in range(PACKET):
                    if method == "whole_session":
                        hist = np.arange(2 * PACKET)
                    else:
                        before = np.flatnonzero(times < session.time[ev, p])
                        hist = before[np.argsort(times[before])][-history:]
                    H = pool[hist]
                    finite = np.isfinite(H)
                    cnt = finite.sum(axis=0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = np.where(finite, H, 0).sum(axis=0) / cnt
                        sd = np.sqrt((np.where(finite, H - mean, 0) ** 2).sum(axis=0) / cnt)
                    mean = np.where(cnt >= max(2, history // 2), mean, np.nan)  # at least half the history finite
                    Z[p] = _restandardize(Xe[p], mean, sd, Zsrc[p], scale) * usable
            else:
                Zs = standardize(X[s, k][None], center[None], scale[None], usable[None])[0]
                Zt = standardize(X[t, k, :n][None], center[None], scale[None], usable[None])[0]
                mt, ms = Zt.mean(axis=0), Zs.mean(axis=0)
                if method == "covariance":
                    def shrunk(Zm):
                        C = np.cov(Zm, rowvar=False, bias=True)
                        return (1 - alpha) * C + alpha * np.diag(np.diag(C))
                    es, vs = np.linalg.eigh(shrunk(Zs))
                    et, vt = np.linalg.eigh(shrunk(Zt))
                    root_s = vs @ np.diag(np.sqrt(np.clip(es, 0, None))) @ vs.T
                    inv_root_t = vt @ np.diag(1 / np.sqrt(np.clip(et, 1e-12, None))) @ vt.T
                    u, sv, vh = np.linalg.svd(root_s @ inv_root_t)
                    Z = (Zsrc - mt) @ (u @ np.diag(np.clip(sv, BOUND_LOW, BOUND_HIGH)) @ vh).T + ms
                elif method == "subspace":
                    Us = np.linalg.svd(Zs - ms, full_matrices=False)[2][:k_sub].T
                    Ut = np.linalg.svd(Zt - mt, full_matrices=False)[2][:k_sub].T
                    u, _, vh = np.linalg.svd(Ut.T @ Us)
                    Z = (Zsrc - mt) @ Ut @ (u @ vh) @ Us.T + ms
                else:
                    raise ValueError(method)
            P = Z @ beta + c
            table[e, ev] = ((Y[t, ev] - P) ** 2).sum(axis=-1)
    return table


# ----------------------------------------------------------------------------- controls
def shifted_label_null(panel, fractions=(0.25, 0.5, 0.75)):
    """J of the local model fit on block-preserving shifted labels and scored on true labels."""
    values = []
    for s in panel.sessions:
        X = s.X[panel.feature][None]
        for f in fractions:
            Y = s.Y.copy()
            for k in (0, 1):
                for block in np.unique(s.block[k]):
                    pos = np.flatnonzero(s.block[k] == block)
                    shift = int(round(f * len(pos)))
                    if len(pos) > 1 and shift % len(pos):
                        Y[k, pos] = s.Y[k, np.roll(pos, -shift)]
            if np.array_equal(Y, s.Y):
                continue
            values.append(shifted_fit(panel, X, Y[None], s))
    return np.array(values)


def permutation_null(panel, permutations=3, seed=1601):
    """J of the local model fit on labels permuted within each block of the fitting packet (labels and
    their octant weights travel together), scored on true evaluation labels. Readiness null: unlike
    circular shifts, it cannot reproduce the centre-out-back label sequence (a one-trial shift maps an
    outward label onto its exact opposite)."""
    values = []
    for i, s in enumerate(panel.sessions):
        X = s.X[panel.feature][None]
        for r in range(permutations):
            rng = np.random.default_rng([seed, i, r])
            Y = s.Y.copy()
            for k in (0, 1):
                for block in np.unique(s.block[k]):
                    pos = np.flatnonzero(s.block[k] == block)
                    Y[k, pos] = s.Y[k, rng.permutation(pos)]
            values.append(shifted_fit(panel, X, Y[None], s))
    return np.array(values)


def shifted_fit(panel, X, Y_shift, session):
    """Local model on shifted fitting labels; scores use the true evaluation labels and weights."""
    w_fit = np.stack([octant_weights_or_nan(octant_from_labels(Y_shift[0, k])) for k in (0, 1)])[None]
    center, scale, usable = fit_scaler(X)
    Z = standardize(X, center, scale, usable)
    J = 0.0
    for k in (0, 1):
        ev = 1 - k
        beta, c = ridge(Z[:, k], Y_shift[:, k], w_fit[:, k], panel.lam)
        P = standardize(X[:, ev], center[:, k], scale[:, k], usable[:, k]) @ beta + c[:, None, :]
        ybar = w_fit[0, k] @ Y_shift[0, k]
        L = score(session.Y[ev], P[0], session.w[ev])
        B = score(session.Y[ev], np.broadcast_to(ybar, session.Y[ev].shape), session.w[ev])
        J += 0.5 * (L - B)
    return float(J)


def half_window_leakage(panel):
    """Deliberately invalid split: fit on odd-bin halves and score even-bin halves of the same trials,
    versus the valid cross-side half-window model at the same feature quality. Returns (leak, valid)."""
    leak, valid = [], []
    for s in panel.sessions:
        H = s.X.get("sbp_odd_even")
        if H is None:
            continue
        L_leak = L_valid = 0.0
        for k in (0, 1):
            center, scale, usable = fit_scaler(H[0, k][None])
            beta, c = ridge(standardize(H[0, k][None], center, scale, usable), s.Y[k][None], s.w[k][None], panel.lam)
            same = standardize(H[1, k][None], center, scale, usable) @ beta + c[:, None, :]
            other = standardize(H[1, 1 - k][None], center, scale, usable) @ beta + c[:, None, :]
            L_leak += 0.5 * score(s.Y[k], same[0], s.w[k])
            L_valid += 0.5 * score(s.Y[1 - k], other[0], s.w[1 - k])
        leak.append(float(L_leak))
        valid.append(float(L_valid))
    return np.array(leak), np.array(valid)


def random_trial_split(panel, rng):
    """Invalid split ignoring blocks: the 128 packet trials are randomly reassigned to two sides."""
    out = []
    for s in panel.sessions:
        X = s.X[panel.feature].reshape(2 * PACKET, -1)
        Y = s.Y.reshape(2 * PACKET, 2)
        order = rng.permutation(2 * PACKET)
        Xs, Ys = X[order].reshape(2, PACKET, -1), Y[order].reshape(2, PACKET, 2)
        w = np.stack([octant_weights_or_nan(octant_from_labels(Ys[k])) for k in (0, 1)])
        if not np.isfinite(w).all():
            out.append(np.nan)
            continue
        out.append(float(panel.fit_sessions(Xs[None], Ys[None], w[None])["L"][0]))
    return np.array(out)
