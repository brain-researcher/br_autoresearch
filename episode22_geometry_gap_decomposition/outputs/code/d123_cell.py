#!/usr/bin/env python3
"""EP22 D1–D3 for one participant × held-out fold (development responses only).

Inputs are EP04's frozen, closed development artifacts (read-only): the
development embedding handoff, the per-fold saved mean patterns and their image
validity tables, and EP04's observed native panel (for the R0 reproduction
check only). No OOD, regular-shared, raw beta or raw image value is read.

D1  eigenvalue × eigenvector factorial on evaluation images (+ image bootstrap)
D2  matched-flexibility ladder: R0 native, R1 spectral exponent, R2 per-axis
    non-negative weights, R3 ridge-encoding-predicted geometry
D3  ED-maximizing selection in each model's own space versus random subsets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from scipy.stats import rankdata

EP04 = Path("/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry")
EMBEDDINGS = EP04 / "outputs/runs/source_readiness/20261003-development_embeddings01"
OBSERVED = EP04 / "outputs/runs/observed_native_panel/20261005-observed01/observed_native_panel.json"
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
MODELS = ("DINOv2", "CLIP", "PEcore", "SigLIP2")
REF = "DINOv2"
VLMS = ("CLIP", "PEcore", "SigLIP2")
ROIS = ("laion-ventral", "laion-lateral", "laion-dorsal", "retinotopic-EVC")
PRIMARY = ROIS[:3]
ALPHAS = np.round(np.arange(0.0, 4.0001, 0.1), 2)
K_R1, K_R2 = 256, 64
ENCODING_LAMBDA = 1e-3          # EP04 identical-fold encoding setting
R2_PAIRS, R2_LAMBDAS = 500_000, (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BOOTSTRAPS = 100
D3_RANDOM, D3_SEED_SET, D3_PCA = 20, 10, 64
SEED = 20261012


def preprocessing_report(person: str) -> Path:
    if person == "sub-05":
        return EP04 / "outputs/runs/neural_preprocessing/20261003-sub05-folds01/preprocessing.json"
    return EP04 / "outputs/runs/neural_preprocessing/20261003-remaining-folds01" / person / "preprocessing.json"


def json_lines(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def l2(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1)
    if not np.all(np.isfinite(x)) or np.any(n == 0):
        raise ValueError("non-finite or zero-norm row")
    return x / n[:, None]


def corr_rdm(p):
    p = np.asarray(p, dtype=np.float64)
    p = l2(p - p.mean(axis=1, keepdims=True))
    d = 1.0 - p @ p.T
    np.fill_diagonal(d, 0.0)
    return d


def cos_rdm(x):
    x = l2(x)
    d = 1.0 - x @ x.T
    np.fill_diagonal(d, 0.0)
    return d


def sq_rdm(y):
    g = y @ y.T
    q = np.diag(g)
    d = q[:, None] + q[None, :] - 2.0 * g
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


_TRIU: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def upper(d):
    n = d.shape[0]
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n, 1)
    return d[_TRIU[n]]


def ranks(v):
    return rankdata(v, method="average")


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    if den == 0:
        return float("nan")
    return float(a @ b / den)


def fz(r):
    return math.atanh(r) if math.isfinite(r) and abs(r) < 1 else float("nan")


def rsa_z(model_ranks, neural_ranks):
    return fz(pearson(model_ranks, neural_ranks))


def centered_svd(x):
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    return u, s


def participation_ratio(x):
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    w = s ** 2
    return float(w.sum() ** 2 / np.sum(w ** 2))


# --------------------------------------------------------------------------

def d1_factorial(x_eval, neural_full, rng):
    """Native and swapped geometries; RSA per ROI; bootstrap over images."""
    svd = {m: centered_svd(x_eval[m]) for m in MODELS}
    geoms = {}
    for m in MODELS:
        u, s = svd[m]
        geoms[f"{m}|{m}"] = sq_rdm(u * s)
    for m in VLMS:
        (um, sm), (ur, sr) = svd[m], svd[REF]
        k = min(len(sm), len(sr))
        geoms[f"{m}|{REF}"] = sq_rdm(um[:, :k] * sr[:k])   # VLM eigenvectors, DINOv2 eigenvalues
        geoms[f"{REF}|{m}"] = sq_rdm(ur[:, :k] * sm[:k])   # DINOv2 eigenvectors, VLM eigenvalues
    neural_ranks = {roi: ranks(upper(d)) for roi, d in neural_full.items()}
    point = {g: {roi: rsa_z(ranks(upper(d)), nr) for roi, nr in neural_ranks.items()} for g, d in geoms.items()}
    n = next(iter(geoms.values())).shape[0]
    iu = np.triu_indices(n, 1)
    boot = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        a, b = idx[iu[0]], idx[iu[1]]
        keep = a != b
        a, b = a[keep], b[keep]
        nr = {roi: ranks(d[a, b]) for roi, d in neural_full.items()}
        boot.append({g: {roi: rsa_z(ranks(d[a, b]), r) for roi, r in nr.items()} for g, d in geoms.items()})
    spectra = {m: (svd[m][1][:64] ** 2 / np.sum(svd[m][1] ** 2)).tolist() for m in MODELS}
    return point, boot, spectra


def d2_ladder(x_train, x_eval, train_sets, neural_train_full, neural_eval_ranks, patterns, eval_idx, rng):
    """R0–R3 per model and ROI at matched flexibility; fitting on training images only."""
    out = {m: {} for m in MODELS}
    for m in MODELS:
        xt = x_train[m]
        mu = xt.mean(axis=0)
        xtc = xt - mu
        _, s, vt = np.linalg.svd(xtc, full_matrices=False)
        v = vt.T
        k1 = min(K_R1, len(s))
        t1 = xtc @ v[:, :k1]                                   # = U_k S_k
        xec = x_eval[m] - mu
        # R1: one exponent per model and fold, selected on the mean primary training RSA.
        objective = []
        for alpha in ALPHAS:
            phi = t1 * (s[:k1] ** (alpha - 1.0))
            model_ranks = ranks(upper(sq_rdm(phi)))
            objective.append(float(np.mean([rsa_z(model_ranks, neural_train_full[roi]["ranks"])
                                            for roi in PRIMARY])))
        best = int(np.nanargmax(objective))
        alpha_hat = float(ALPHAS[best])
        r1_eval = ranks(upper(sq_rdm(xec @ v[:, :k1] * (s[:k1] ** (alpha_hat - 1.0)))))
        r1_alpha1 = ranks(upper(sq_rdm(xec @ v[:, :k1])))
        out[m]["R1_alpha_hat"] = alpha_hat
        out[m]["R1_training_objective"] = objective
        r0 = ranks(upper(cos_rdm(x_eval[m])))
        # R3 design (EP04 ridge): normalized loss, training-mean intercept.
        enc = {}
        for roi in ROIS:
            rows = train_sets[roi]
            if rows is None or neural_eval_ranks.get(roi) is None:
                continue
            row_pos = np.searchsorted(train_sets["_primary_rows"], rows)
            x = xt[row_pos]
            xm = x.mean(axis=0)
            xc = x - xm
            gram = xc.T @ xc
            gram.flat[:: gram.shape[0] + 1] += len(x) * ENCODING_LAMBDA
            factor = cho_factor(gram, lower=True)
            y = patterns[roi][rows]
            ym = y.mean(axis=0)
            w = cho_solve(factor, xc.T @ (y - ym))
            pred = (x_eval[m] - xm) @ w + ym
            enc[roi] = ranks(upper(corr_rdm(pred)))
        # R2: non-negative per-axis weights on the leading 64 training axes, per ROI.
        k2 = min(K_R2, len(s))
        scores_train = xtc @ v[:, :k2]
        scores_eval = xec @ v[:, :k2]
        for roi in ROIS:
            nr_eval = neural_eval_ranks.get(roi)
            if nr_eval is None:
                continue
            entry = {"R0": rsa_z(r0, nr_eval), "R1": rsa_z(r1_eval, nr_eval),
                     "R1_alpha1_k256": rsa_z(r1_alpha1, nr_eval)}
            rows = train_sets[roi]
            row_pos = np.searchsorted(train_sets["_primary_rows"], rows)
            sc = scores_train[row_pos]
            dfull = neural_train_full[roi]["full"]
            nimg = len(rows)
            perm = rng.permutation(nimg)
            half_a, half_b = perm[: nimg // 2], perm[nimg // 2:]

            def pairs(pool, count):
                i = pool[rng.integers(0, len(pool), count)]
                j = pool[rng.integers(0, len(pool), count)]
                keep = i != j
                return i[keep], j[keep]

            def design(i, j):
                return (sc[i] - sc[j]) ** 2

            ia, ja = pairs(half_a, R2_PAIRS // 2)
            ib, jb = pairs(half_b, R2_PAIRS // 2)
            xa, xb = design(ia, ja), design(ib, jb)
            sd = xa.std(axis=0)
            sd[sd == 0] = 1.0
            ya, yb = dfull[ia, ja], dfull[ib, jb]

            def fit(xd, yd, lam):
                xs = xd / sd
                xs = xs - xs.mean(axis=0)
                yc = yd - yd.mean()
                a = xs.T @ xs + lam * len(yd) * np.eye(xs.shape[1])
                lchol = np.linalg.cholesky(a)
                rhs = np.linalg.solve(lchol, xs.T @ yc)
                wts, _ = nnls(lchol.T, rhs)
                return wts / sd

            validation = []
            for lam in R2_LAMBDAS:
                wts = fit(xa, ya, lam)
                validation.append(pearson(ranks(xb @ wts), ranks(yb)) if np.any(wts > 0) else float("nan"))
            lam_hat = float(R2_LAMBDAS[int(np.nanargmax(validation))]) if np.any(np.isfinite(validation)) else 0.0
            i_all, j_all = pairs(np.arange(nimg), R2_PAIRS)
            wts = fit(design(i_all, j_all), dfull[i_all, j_all], lam_hat)
            if np.any(wts > 0):
                entry["R2"] = rsa_z(ranks(upper(sq_rdm(scores_eval * np.sqrt(wts)))), nr_eval)
            else:
                entry["R2"] = float("nan")
            entry["R2_lambda_hat"] = lam_hat
            entry["R2_nonzero_weights"] = int(np.sum(wts > 0))
            entry["R2_validation_spearman"] = validation
            entry["R3"] = rsa_z(enc[roi], nr_eval) if roi in enc else float("nan")
            out[m][roi] = entry
    return out


def d3_selection(x_eval, neural_full, laion_pos, participant_i, fold, rng):
    """Greedy ED maximization in each selector's own 64-d PCA space versus random subsets."""
    n_pool = len(laion_pos)
    m_sub = n_pool // 2
    seed_set = rng.choice(n_pool, size=D3_SEED_SET, replace=False)
    subsets = {}
    for s in MODELS:
        z = l2(x_eval[s][laion_pos])
        zc = z - z.mean(axis=0)
        u, sv, _ = np.linalg.svd(zc, full_matrices=False)
        sc = u[:, :D3_PCA] * sv[:D3_PCA]
        chosen = list(seed_set)
        mask = np.zeros(n_pool, dtype=bool)
        mask[chosen] = True
        c = sc[chosen].T @ sc[chosen]
        tr = float(np.trace(c))
        norms = np.sum(sc * sc, axis=1)
        while len(chosen) < m_sub:
            q = np.einsum("ij,jk,ik->i", sc, c, sc)
            tr2 = float(np.sum(c * c))
            ed = (tr + norms) ** 2 / (tr2 + 2.0 * q + norms ** 2)
            ed[mask] = -np.inf
            pick = int(np.argmax(ed))
            chosen.append(pick)
            mask[pick] = True
            c += np.outer(sc[pick], sc[pick])
            tr += float(norms[pick])
        subsets[f"selected_by_{s}"] = np.sort(np.array(chosen))
    for r in range(D3_RANDOM):
        subsets[f"random_{r:02d}"] = np.sort(rng.choice(n_pool, size=m_sub, replace=False))
    model_full = {m: cos_rdm(x_eval[m]) for m in MODELS}
    result = {"n_pool": n_pool, "n_subset": m_sub, "subsets": {}}
    for name, sub in subsets.items():
        pos = laion_pos[sub]
        ix = np.ix_(pos, pos)
        nr = {roi: ranks(upper(d[ix])) for roi, d in neural_full.items()}
        entry = {"ED": {m: participation_ratio(x_eval[m][pos]) for m in MODELS}, "RSA": {}}
        for m in MODELS:
            mr = ranks(upper(model_full[m][ix]))
            entry["RSA"][m] = {roi: rsa_z(mr, r) for roi, r in nr.items()}
        result["subsets"][name] = entry
    return result


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", choices=PARTICIPANTS)
    ap.add_argument("--fold", type=int, choices=range(5))
    ap.add_argument("--task-index", type=int, help="0–24: participant-major, fold-minor")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.task_index is not None:
        a.participant, a.fold = PARTICIPANTS[a.task_index // 5], a.task_index % 5
    t0 = time.time()
    out_path = a.out_dir / f"{a.participant}-fold-{a.fold}.json"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    p_i = PARTICIPANTS.index(a.participant)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, p_i, a.fold]))
    report = {"analysis_ids": ["d1_eigen_factorial", "d2_flexibility_ladder", "d3_selection_intervention"],
              "participant": a.participant, "held_fold": a.fold, "hostname": socket.gethostname(),
              "job_id": os.environ.get("SLURM_JOB_ID"), "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "neural_inputs": "EP04 saved development mean patterns only", "ood_or_shared_values_read": False,
              "settings": {"alphas": ALPHAS.tolist(), "k_r1": K_R1, "k_r2": K_R2, "encoding_lambda": ENCODING_LAMBDA,
                           "r2_pairs": R2_PAIRS, "r2_lambdas": R2_LAMBDAS, "bootstraps": BOOTSTRAPS,
                           "d3_random": D3_RANDOM, "d3_seed_set": D3_SEED_SET, "d3_pca": D3_PCA, "seed": SEED}}

    prep = json.loads(preprocessing_report(a.participant).read_text())
    payload = Path(prep["payload_dir"])
    images = json_lines(payload / "image_rows.jsonl")
    fold_dir = payload / f"fold-{a.fold}"
    validity = json_lines(fold_dir / "image_validity.jsonl")
    if len(images) != 4712 or [r["image_id"] for r in images] != [r["image_id"] for r in validity]:
        raise SystemExit("saved pattern rows and validity differ")
    eval_idx = np.array([i for i, (im, v) in enumerate(zip(images, validity))
                         if im["fold"] == a.fold and v["primary_eligible"]])
    train_primary = np.array([i for i, (im, v) in enumerate(zip(images, validity))
                              if im["fold"] != a.fold and v["primary_eligible"]])
    train_evc = np.array([i for i, (im, v) in enumerate(zip(images, validity))
                          if im["fold"] != a.fold and v["primary_eligible"] and v["evc_eligible_on_primary_images"]])
    evc_ok = all(validity[i]["evc_eligible_on_primary_images"] for i in eval_idx)
    train_sets = {roi: train_primary for roi in PRIMARY}
    train_sets[ROIS[3]] = train_evc if evc_ok else None
    train_sets["_primary_rows"] = train_primary
    strata = np.array([images[i]["source_stratum"] for i in eval_idx])
    report["counts"] = {"evaluation": int(len(eval_idx)), "training_primary": int(len(train_primary)),
                        "training_evc": int(len(train_evc)), "evc_evaluable": bool(evc_ok),
                        "evaluation_by_stratum": {s: int(np.sum(strata == s)) for s in ("laion", "things", "thingsplus")}}

    # Embeddings (EP04 handoff), aligned by image ID.
    dev_rows = {r["image_id"]: r["output_row"] for r in json_lines(EMBEDDINGS / "development_rows.jsonl")}
    raw = {m: np.load(EMBEDDINGS / f"{m}.npy", mmap_mode="r") for m in MODELS}
    eval_rows = [dev_rows[images[i]["image_id"]] for i in eval_idx]
    train_rows = [dev_rows[images[i]["image_id"]] for i in train_primary]
    x_eval = {m: l2(raw[m][eval_rows]) for m in MODELS}
    x_train = {m: l2(raw[m][train_rows]) for m in MODELS}

    # Neural geometry from saved mean patterns (correlation distance).
    patterns, neural_eval_full, neural_eval_ranks, neural_train_full = {}, {}, {}, {}
    for roi in ROIS:
        pat = np.load(fold_dir / f"{roi}_patterns.npy", mmap_mode="r")
        if pat.shape[0] != 4712:
            raise SystemExit(f"{roi}: unexpected pattern rows")
        if roi == ROIS[3] and not evc_ok:
            continue
        patterns[roi] = np.asarray(pat, dtype=np.float64)
        neural_eval_full[roi] = corr_rdm(patterns[roi][eval_idx])
        neural_eval_ranks[roi] = ranks(upper(neural_eval_full[roi]))
        rows = train_sets[roi]
        dfull = corr_rdm(patterns[roi][rows])
        neural_train_full[roi] = {"full": dfull, "ranks": ranks(upper(dfull)) if roi in PRIMARY else None}
    report["timing_load_seconds"] = time.time() - t0

    # R0 reproduction against EP04's observed native panel.
    observed = json.loads(OBSERVED.read_text())
    expected = {(r["model_label"], r["roi"]): r["native_fisher_z"] for r in observed["rows"]
                if r["participant"] == a.participant and r["held_fold"] == a.fold and r["native_status"] == "evaluable"}
    r0 = {m: {roi: rsa_z(ranks(upper(cos_rdm(x_eval[m]))), nr) for roi, nr in neural_eval_ranks.items()} for m in MODELS}
    diffs = [abs(r0[m][roi] - expected[(m, roi)]) for m in MODELS for roi in neural_eval_ranks if (m, roi) in expected]
    report["R0_reproduction"] = {"compared_rows": len(diffs), "max_abs_fisher_z_difference": max(diffs) if diffs else None,
                                 "reproduced": bool(diffs) and max(diffs) < 1e-8}
    report["R0"] = r0

    t1 = time.time()
    point, boot, spectra = d1_factorial(x_eval, neural_eval_full, rng)
    report["D1"] = {"point": point, "bootstrap": boot, "eigenvalue_shares_top64": spectra,
                    "native_matches_R0_max_abs": max(abs(point[f"{m}|{m}"][roi] - r0[m][roi])
                                                     for m in MODELS for roi in neural_eval_ranks)}
    report["timing_d1_seconds"] = time.time() - t1

    t2 = time.time()
    report["D2"] = d2_ladder(x_train, x_eval, train_sets, neural_train_full, neural_eval_ranks, patterns, eval_idx, rng)
    report["timing_d2_seconds"] = time.time() - t2

    t3 = time.time()
    laion_pos = np.flatnonzero(strata == "laion")
    report["D3"] = d3_selection(x_eval, neural_eval_full, laion_pos, p_i, a.fold, rng)
    report["timing_d3_seconds"] = time.time() - t3

    report["elapsed_seconds"] = time.time() - t0
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = out_path.with_suffix(".inprogress")
    tmp.write_text(json.dumps(report))
    tmp.replace(out_path)
    print(json.dumps({"cell": out_path.name, "R0_reproduction": report["R0_reproduction"],
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}))
    return 0 if report["R0_reproduction"]["reproduced"] else 3


if __name__ == "__main__":
    sys.exit(main())
