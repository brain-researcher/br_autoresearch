"""EP16 panel arithmetic: realized nodes, fold-averaged scores, estimands, bootstrap and decisions.

Implements config/analysis_specification.json for one participant and one connected component.
Every score is fold-averaged before differences or ratios. Pairs are chronological 1-30 day
(near) or >=180 day (long) edges between session occurrences; occurrences of the same session
are never paired. Nonestimable statistics are -inf for lower and +inf for upper bounds.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np

from describe_schedule_design import EPS, contrasts
from ep16_models import Unresolved, fit_scaler, octant_weights, ridge, score, standardize

PACKET = 64
GAPS = {"near": (1, 30), "long_min": 180}
PRIMARY = ("adjusted", "robust", "fractional_local", "raw_transported")


# ----------------------------------------------------------------------------- sessions
class Session:
    """One realized session node: two 64-trial sides on the participant's physical axis."""

    def __init__(self, day, X, Y, octant, block, available, configuration_pure=False, time=None):
        self.day = int(day)
        self.time = None if time is None else np.asarray(time, dtype=np.int64)  # (2, 64) trial starts
        self.X = {f: np.asarray(v, dtype=np.float64) for f, v in X.items()}  # (2, 64, E) raw, NaN missing
        self.Y = np.asarray(Y, dtype=np.float64)                            # (2, 64, 2)
        self.octant = np.asarray(octant, dtype=np.int64)                    # (2, 64)
        self.block = np.asarray(block)                                      # (2, 64)
        self.available = np.asarray(available, dtype=bool)                  # (E,)
        self.configuration_pure = bool(configuration_pure)
        self.w = np.stack([octant_weights(self.octant[k]) for k in (0, 1)])
        self.cover = {n: [len(set(self.octant[k, :n])) == 8 for k in (0, 1)] for n in (16, 32, 64)}
        # Resampling strata: positions sharing (block, octant) within a side.
        self.strata = []
        for k in (0, 1):
            keys = [(self.block[k, p], self.octant[k, p]) for p in range(PACKET)]
            index = {key: i for i, key in enumerate(dict.fromkeys(keys))}
            sid = np.array([index[key] for key in keys])
            order = np.argsort(sid, kind="stable")
            sizes = np.bincount(sid)
            offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            self.strata.append((sid, order, sizes, offsets))

    def resample_positions(self, rng):
        """Evaluation positions drawn with replacement inside (block, octant) strata; octant per position kept."""
        out = np.empty((2, PACKET), dtype=np.int64)
        for k in (0, 1):
            sid, order, sizes, offsets = self.strata[k]
            pick = offsets[sid] + np.floor(rng.random(PACKET) * sizes[sid]).astype(np.int64)
            out[k] = order[pick]
        return out


IDENTITY = np.tile(np.arange(PACKET), (2, 1))  # evaluation positions of an unresampled side


def chronological_edges(days, near=GAPS["near"], long_min=GAPS["long_min"]):
    """All chronological near/long pairs (i -> j) between occurrences with distinct days."""
    days = np.asarray(days)
    gap = days[None, :] - days[:, None]
    is_near = (gap >= near[0]) & (gap <= near[1])
    is_long = gap >= long_min
    src, tgt = np.nonzero(is_near | is_long)
    return src, tgt, is_long[src, tgt]


# ----------------------------------------------------------------------------- realizations
def aggregate(loss, pos, w):
    """Fold-averaged score from per-position losses: loss/pos/w (M, 2 evaluation sides, 64)."""
    picked = np.take_along_axis(loss, pos, axis=2)
    return -0.5 * (w * picked).sum(axis=(1, 2))


class Realization:
    """Session occurrences with evaluation positions; every score is an index-and-sum over the panel's
    per-position losses, so models stay fit to each session's realized fitting trials."""

    def __init__(self, panel, occ, pos, variant=None, edges=None):
        T = panel.tables
        self.panel, self.occ, self.pos, self.variant = panel, np.asarray(occ), np.asarray(pos), variant or {}
        self.days = T["days"][self.occ]
        self.w_occ = T["w"][self.occ]
        self.local = {"L": aggregate(T["lossL"][self.occ], self.pos, self.w_occ),
                      "B": aggregate(T["lossB"][self.occ], self.pos, self.w_occ)}
        self.local["J"] = self.local["L"] - self.local["B"]
        if edges is None:
            src, tgt, long = chronological_edges(self.days, panel.near, panel.long_min)
            if panel.direction == "reverse":
                src, tgt = tgt, src
        else:
            src, tgt, long = edges
        self.src, self.tgt, self.long = np.asarray(src), np.asarray(tgt), np.asarray(long, dtype=bool)
        self.e = T["edge_index"][self.occ[self.src], self.occ[self.tgt]]
        if np.any(self.e < 0):
            raise ValueError("Occurrence pair without a registered session edge")
        self.F = self.edge_scores(self.variant.get("lossF", T["lossF"]), side="target")

    def edge_scores(self, table, side="target", rows=None):
        """Aggregate an (edges, 2, 64) per-position loss table at the target's or source's positions."""
        rows = np.arange(len(self.e)) if rows is None else np.asarray(rows)
        who = self.tgt[rows] if side == "target" else self.src[rows]
        return aggregate(table[self.e[rows]], self.pos[who], self.w_occ[who])

    def occurrence_scores(self, table, occurrences):
        occurrences = np.asarray(occurrences)
        return aggregate(table[self.occ[occurrences]], self.pos[occurrences], self.w_occ[occurrences])

    def edge_target_scores(self):
        """L_t, B_t, J_t per edge; long edges use the degraded target under the local-signal injection."""
        L, B = self.local["L"][self.tgt].copy(), self.local["B"][self.tgt]
        if "lossL_degraded" in self.variant:
            longs = np.flatnonzero(self.long)
            L[longs] = self.occurrence_scores(self.variant["lossL_degraded"], self.tgt[longs])
        return L, B, L - B


# ----------------------------------------------------------------------------- estimators
def target_weights(tgt):
    counts = Counter(tgt.tolist())
    n = len(counts)
    return np.array([1.0 / (n * counts[t]) for t in tgt.tolist()])


def adjusted_contrast(src, tgt, long, R):
    """WLS long-indicator coefficient with Helmert source/target effects; None if rank-deficient."""
    sources, targets = np.unique(src), np.unique(tgt)
    if not long.any() or long.all():
        return None
    si = np.searchsorted(sources, src)
    ti = np.searchsorted(targets, tgt)
    w = target_weights(tgt)
    nuisance = np.column_stack((np.ones(len(src)), contrasts(len(sources))[si], contrasts(len(targets))[ti]))
    d = np.sqrt(w)
    design = np.column_stack((nuisance * d[:, None], d * long))
    singular = np.linalg.svd(design, compute_uv=False)
    if np.sum(singular > max(design.shape) * EPS * singular[0]) != design.shape[1]:
        return None
    q_basis, _ = np.linalg.qr(nuisance * d[:, None])
    z = d * long
    q = z - q_basis @ (q_basis.T @ z)
    return float(q @ (d * R) / (q @ q))


def robust_contrast(tgt, long, R):
    """Median over sources within target and lag, median over targets, long minus near."""
    values = {}
    for lag in (False, True):
        rows = long == lag
        if not rows.any():
            return None
        per_target = [np.median(R[rows & (tgt == t)]) for t in np.unique(tgt[rows])]
        values[lag] = float(np.median(per_target))
    return values[True] - values[False]


def lag_weighted_contrast(tgt, long, value):
    """Target-equal weights within each lag class, split equally over sources; long minus near."""
    means = {}
    for lag in (False, True):
        rows = np.flatnonzero(long == lag)
        if not len(rows):
            return None, None
        w = target_weights(tgt[rows])
        means[lag] = float(w @ value[rows])
    return means[True] - means[False], means


def primary_statistics(draw, J_min):
    """The four primary-family statistics with their own nonestimability conditions."""
    Lt, Bt, Jt = draw.edge_target_scores()
    F, src, tgt, long = draw.F, draw.src, draw.tgt, draw.long
    Js = draw.local["J"][src]
    out = {name: None for name in PRIMARY}
    info = {"pairs": int(len(src)), "near_pairs": int((~long).sum()), "long_pairs": int(long.sum())}
    finite = np.isfinite(F).all() and np.isfinite(Lt).all() and np.isfinite(Bt).all()
    if not finite or not len(src):
        return out, {**info, "nonestimable": "nonfinite_scores_or_no_pairs"}
    target_ready = bool(np.all(Jt > J_min))
    if target_ready:
        R = (Lt - F) / Jt
        out["adjusted"] = adjusted_contrast(src, tgt, long, R)
        out["robust"] = robust_contrast(tgt, long, R)
        info["R_near_weighted_mean"] = lag_weighted_contrast(tgt, long, R)[1]
    else:
        info["target_readiness_failures"] = int(np.unique(tgt[Jt <= J_min]).size)
    if np.all(Js > J_min):
        out["fractional_local"], means = lag_weighted_contrast(tgt, long, (Jt - Js) / Js)
        info["fractional_means"] = means
    else:
        info["source_readiness_failures"] = int(np.unique(src[Js <= J_min]).size)
    out["raw_transported"], raw_means = lag_weighted_contrast(tgt, long, F - Bt)
    info["raw_means"] = raw_means
    return out, info


# ----------------------------------------------------------------------------- panel
class Panel:
    """Fits every session's models once on its realized fitting trials and stores per-position losses."""

    def __init__(self, sessions, feature, lam, J_min, near=GAPS["near"], long_min=GAPS["long_min"], direction="chronological"):
        self.sessions = list(sessions)
        self.feature, self.lam, self.J_min = feature, float(lam), float(J_min)
        self.near, self.long_min, self.direction = near, long_min, direction
        self._tables = None

    def stacked(self):
        X = np.stack([s.X[self.feature] for s in self.sessions])
        Y = np.stack([s.Y for s in self.sessions])
        w = np.stack([s.w for s in self.sessions])
        return X, Y, w

    def fit_sessions(self, X, Y, w, n=PACKET, lam=None):
        """Per session and side: own scaler and ridge on the first n positions; L, B, J fold-averaged,
        plus per-position local and null losses on the opposite (evaluation) side."""
        lam = self.lam if lam is None else lam
        O = X.shape[0]
        Xf = X[:, :, :n].reshape(O * 2, n, -1)
        Yf = Y[:, :, :n].reshape(O * 2, n, 2)
        wf = w.reshape(O * 2, n) if n == PACKET else np.stack([octant_weights_or_nan(octant_from_labels(Yf[m]))
                                                              for m in range(O * 2)])
        center, scale, usable = fit_scaler(Xf)
        valid = np.isfinite(wf).all(axis=1)
        wf_safe = np.where(valid[:, None], wf, 1.0 / n)
        beta, c = ridge(standardize(Xf, center, scale, usable), Yf, wf_safe, lam)
        ybar = np.einsum("mn,mnd->md", wf_safe, Yf)
        shape = lambda a: a.reshape((O, 2) + a.shape[1:])
        center, scale, usable, beta, c, ybar, valid = map(shape, (center, scale, usable, beta, c, ybar, valid))
        lossL = np.empty((O, 2, PACKET))
        lossB = np.empty((O, 2, PACKET))
        for k in (0, 1):
            ev = 1 - k
            P = standardize(X[:, ev], center[:, k], scale[:, k], usable[:, k]) @ beta[:, k] + c[:, k][:, None, :]
            lossL[:, ev] = ((Y[:, ev] - P) ** 2).sum(axis=-1)
            lossB[:, ev] = ((Y[:, ev] - ybar[:, k][:, None, :]) ** 2).sum(axis=-1)
        ident = np.broadcast_to(np.arange(PACKET), (O, 2, PACKET))
        L = np.where(valid.all(axis=1), aggregate(lossL, ident, w), np.nan)
        B = aggregate(lossB, ident, w)
        return {"center": center, "scale": scale, "usable": usable, "beta": beta, "c": c, "ybar": ybar,
                "valid": valid, "lossL": lossL, "lossB": lossB, "L": L, "B": B, "J": L - B}

    def transported_losses(self, src, tgt, fit, X=None, Y=None, extra_noise=None):
        """Per-position loss (edges, 2 evaluation sides, 64) of source models on target evaluation trials."""
        X0, Y0, _ = self.stacked()
        X = X0 if X is None else X
        Y = Y0 if Y is None else Y
        out = np.empty((len(src), 2, PACKET))
        for k in (0, 1):
            ev = 1 - k
            for i in np.unique(src):
                rows = np.flatnonzero(src == i)
                Z = standardize(X[tgt[rows], ev], fit["center"][i, k][None], fit["scale"][i, k][None], fit["usable"][i, k][None])
                P = Z @ fit["beta"][i, k] + fit["c"][i, k]
                if extra_noise is not None:
                    P = P + extra_noise(rows, k, fit["beta"][i, k])
                out[rows, ev] = ((Y[tgt[rows], ev] - P) ** 2).sum(axis=-1)
        return out

    @property
    def tables(self):
        if self._tables is None:
            X, Y, w = self.stacked()
            fit = self.fit_sessions(X, Y, w)
            days = np.array([s.day for s in self.sessions])
            src, tgt, long = chronological_edges(days, self.near, self.long_min)
            if self.direction == "reverse":
                src, tgt = tgt, src
            index = np.full((len(days), len(days)), -1)
            index[src, tgt] = np.arange(len(src))
            self._tables = {"days": days, "w": w, "fit": fit, "lossL": fit["lossL"], "lossB": fit["lossB"],
                            "src": src, "tgt": tgt, "long": long, "edge_index": index,
                            "lossF": self.transported_losses(src, tgt, fit)}
        return self._tables

    def observed(self, variant=None):
        occ = np.arange(len(self.sessions))
        return Realization(self, occ, np.broadcast_to(IDENTITY, (len(occ), 2, PACKET)), variant)

    def draw(self, rng, variant=None):
        occ = rng.integers(0, len(self.sessions), size=len(self.sessions))
        pos = np.stack([self.sessions[i].resample_positions(rng) for i in occ])
        return Realization(self, occ, pos, variant)


def octant_from_labels(Y):
    angles = np.remainder(np.arctan2(Y[:, 1], Y[:, 0]), 2 * np.pi)
    return np.floor(angles / (np.pi / 4)).astype(np.int64) % 8


def octant_weights_or_nan(octant):
    try:
        return octant_weights(octant)
    except Unresolved:
        return np.full(len(octant), np.nan)


# ----------------------------------------------------------------------------- bootstrap
_SHARED = {}


def _worker(seeds):
    panel, statistic, variant = _SHARED["panel"], _SHARED["statistic"], _SHARED["variant"]
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows.append(statistic(panel.draw(rng, variant)))
    return rows


def bootstrap(panel, statistic, draws, seed, workers=1, variant=None, chunks=None):
    """statistic(draw) -> dict name -> float or None (None = nonestimable). Returns name -> array.

    Draw b uses SeedSequence(seed).spawn(draws)[b], so results do not depend on the worker count.
    Workers are forked and inherit the panel; only seeds cross process boundaries.
    """
    import multiprocessing
    seeds = np.random.SeedSequence(seed).spawn(draws)
    chunks = chunks or max(1, min(draws, 8 * workers))
    groups = [list(range(i, draws, chunks)) for i in range(chunks)]
    panel.tables  # fit once before forking
    _SHARED.update(panel=panel, statistic=statistic, variant=variant)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_worker, [[seeds[i] for i in g] for g in groups]))
    else:
        parts = [_worker([seeds[i] for i in g]) for g in groups]
    order = [i for g in groups for i in g]
    rows = [None] * draws
    for i, row in zip(order, [r for part in parts for r in part]):
        rows[i] = row
    names = rows[0].keys()
    return {name: np.array([np.nan if r[name] is None else r[name] for r in rows], dtype=np.float64) for name in names}


def one_sided_bound(values, alpha, side):
    """Percentile order statistic k = alpha*(B+1); every nonfinite draw is nonestimable."""
    if side not in ("lower", "upper"):
        raise ValueError(f"Unknown bound side {side!r}")
    values = np.asarray(values, dtype=np.float64)
    filled = np.where(~np.isfinite(values), -np.inf if side == "lower" else np.inf, values)
    ordered = np.sort(filled)
    k = int(round(alpha * (len(values) + 1)))
    if k < 1:
        raise ValueError("Too few draws for the requested level")
    return float(ordered[k - 1] if side == "lower" else ordered[len(values) - k])


# ----------------------------------------------------------------------------- decisions
def decide_lower(bound, margin):
    if not np.isfinite(bound):
        return "unresolved"
    if bound == margin:
        return "unresolved"
    return "supported" if bound > margin else "not_established"


def decide_upper(bound, margin):
    if not np.isfinite(bound):
        return "unresolved"
    if bound == margin:
        return "unresolved"
    return "supported" if bound < margin else "not_established"


def observed_decision(point, bound, margin, side):
    """A bound cannot resolve a decision whose observed statistic is nonestimable."""
    if side not in ("lower", "upper"):
        raise ValueError(f"Unknown decision side {side!r}")
    if point is None or not np.isfinite(point):
        return "unresolved"
    return decide_lower(bound, margin) if side == "lower" else decide_upper(bound, margin)


def combine(states):
    """Conjunction of component decisions: any unresolved -> unresolved; all supported -> supported."""
    if "unresolved" in states:
        return "unresolved"
    return "supported" if all(s == "supported" for s in states) else "not_established"


SIGNATURES = ("local_signal", "recording", "repair")
STATES = ("supported", "not_established", "unresolved")


def conclusion(states, positive_control, validity_failure=False):
    """Map decisions to exactly one registered single-case conclusion (tested exhaustively).

    states: extra_loss, local_signal -> supported | not_established | unresolved;
            recording, repair -> one of STATES when extra_loss is supported, else "not_evaluated".
    positive_control: extra_loss/local_signal -> bool; a 'not_established' reading of an account
    whose known-positive check failed on this design becomes 'underidentified'.
    """
    readings = {}
    for account in ("extra_loss", "local_signal"):
        state = states[account]
        if state not in STATES:
            raise ValueError(f"Unknown state {state!r}")
        readings[account] = "underidentified" if state == "not_established" and not positive_control[account] else state
    for account in ("recording", "repair"):
        state = states[account]
        if states["extra_loss"] == "supported":
            if state not in STATES:
                raise ValueError("Mechanism branches need a decision when extra loss is supported")
        elif state != "not_evaluated":
            raise ValueError("Mechanism branches are evaluated only when extra loss is supported")
        readings[account] = state
    if validity_failure:
        return "data_or_analysis_validity_failure", readings
    signatures = [a for a in SIGNATURES if readings[a] == "supported"]
    if len(signatures) > 1:
        return "combination_of_operational_signatures", readings
    if signatures:
        return {"local_signal": "later_session_signal_reduction",
                "recording": "observable_recording_change_sufficiency",
                "repair": "label_32_remappable_mismatch"}[signatures[0]], readings
    if readings["extra_loss"] == "supported":
        return "extra_long_gap_loss_without_supported_signature", readings
    pair = {readings["extra_loss"], readings["local_signal"]}
    if pair == {"not_established"}:
        return "additional_long_gap_loss_not_established_without_equivalence_claim", readings
    if "unresolved" in pair:
        return "support_or_uncertainty_prevents_an_answer", readings
    return "underidentified_null", readings
