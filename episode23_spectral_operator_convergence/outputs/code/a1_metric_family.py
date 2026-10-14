#!/usr/bin/env python3
"""EP23 A1: nested comparison of metric families on development responses.

Participants are the independent units. Undefined RSA and insufficient
numerical PCA rank abort a cell; incomplete collections are rejected.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata

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
ENCODING_LAMBDA = 1e-3
R2_PAIRS, R2_LAMBDAS = 500_000, (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BOOTSTRAPS = 100
D3_RANDOM, D3_SEED_SET, D3_PCA = 20, 10, 64
SEED = 20261012
LAMBDA0 = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
ANALYSIS = "ep23_a1_metric_family_v1"
CONTRASTS = ("full_minus_diagonal", "power_minus_native")
FAMILIES = ("power", "diagonal", "full")


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


def checked_z(model_ranks, neural_ranks):
    value = rsa_z(model_ranks, neural_ranks)
    if not math.isfinite(value):
        raise ValueError("undefined Fisher RSA: constant ranks, non-finite data, or |r| >= 1")
    return value


def write_json(path, payload):
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    try:
        tmp.write_text(encoded, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def project(xt, xe, rank):
    mu = xt.mean(axis=0, keepdims=True)
    xc = xt - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    tol = np.finfo(np.float64).eps * max(xc.shape) * s[0]
    if rank > len(s) or s[rank - 1] <= tol:
        raise ValueError(f"requested PCA rank {rank} exceeds numerical training rank")
    basis = vt[:rank].T
    return xc @ basis, (xe - mu) @ basis, s[:rank] ** 2


def brain_q(pattern_rows, z):
    """Compute Z.T @ G_tilde @ Z exactly through the factored target."""
    p = np.asarray(pattern_rows, dtype=np.float64)
    p = p - p.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.mean(p * p, axis=1))
    if not np.all(np.isfinite(sd)) or np.any(sd == 0):
        raise ValueError("non-finite or constant training pattern")
    p /= sd[:, None]                         # Population SD, separately per row.
    trace_g = float(np.einsum("ij,ij->", p, p) / p.shape[1])
    pz = p.T @ z
    # G = P P.T / v; G_tilde = G * n / trace(G), without materializing G.
    return (pz.T @ pz) * (len(p) / (p.shape[1] * trace_g))


def fit_metric(d2, q, n_train, lambda0, family):
    c = float(n_train / d2.sum())
    lam = float(lambda0 * np.mean(d2 ** 2))
    # Gradient / 2 = Z.T @ (Z @ W @ Z.T - G_tilde) @ Z
    #                + lam * (W - c * I) = 0, with Z.T @ Z = diag(d2).
    if family == "diagonal":
        return np.maximum((np.diag(q) + lam * c) / (d2 ** 2 + lam), 0.0)
    w = (q + lam * c * np.eye(len(d2))) / (np.outer(d2, d2) + lam)
    # The symmetric closed form precedes the prescribed Euclidean PSD projection.
    # If clipping is active, that projection need not solve the constrained loss.
    eigenvalues, vectors = np.linalg.eigh((w + w.T) / 2.0)
    w = (vectors * np.maximum(eigenvalues, 0.0)) @ vectors.T
    return (w + w.T) / 2.0


def metric_rdm(z, w):
    g = (z * w) @ z.T if w.ndim == 1 else (z @ w) @ z.T
    q = np.diag(g)
    d = q[:, None] + q[None, :] - 2.0 * g
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def numerical_rank(w):
    eigenvalues = w if w.ndim == 1 else np.linalg.eigvalsh(w)
    return int(np.count_nonzero(eigenvalues > 1e-10 * eigenvalues.max(initial=0.0)))


def load_data(person, held, report):
    prep = json.loads(preprocessing_report(person).read_text())
    payload = Path(prep["payload_dir"])
    images = json_lines(payload / "image_rows.jsonl")
    fold_dir = payload / f"fold-{held}"
    validity = json_lines(fold_dir / "image_validity.jsonl")
    if len(images) != 4712 or [r["image_id"] for r in images] != [r["image_id"] for r in validity]:
        raise ValueError("saved pattern rows and validity differ")
    eval_idx = np.array([i for i, (im, v) in enumerate(zip(images, validity))
                         if im["fold"] == held and v["primary_eligible"]], dtype=int)
    train_primary = np.array([i for i, (im, v) in enumerate(zip(images, validity))
                              if im["fold"] != held and v["primary_eligible"]], dtype=int)
    train_evc = np.array([i for i, (im, v) in enumerate(zip(images, validity))
                          if im["fold"] != held and v["primary_eligible"]
                          and v["evc_eligible_on_primary_images"]], dtype=int)
    if len(eval_idx) < 3:
        raise ValueError("fewer than three eligible outer evaluation images")
    evc_ok = all(validity[i]["evc_eligible_on_primary_images"] for i in eval_idx)
    train_sets = {roi: train_primary for roi in PRIMARY}
    train_sets[ROIS[3]] = train_evc if evc_ok else None
    train_sets["_primary_rows"] = train_primary
    strata = np.array([images[i]["source_stratum"] for i in eval_idx])
    report["counts"] = {"evaluation": int(len(eval_idx)), "training_primary": int(len(train_primary)),
                        "training_evc": int(len(train_evc)), "evc_evaluable": bool(evc_ok),
                        "evaluation_by_stratum": {s: int(np.sum(strata == s))
                                                 for s in ("laion", "things", "thingsplus")}}
    dev_records = json_lines(EMBEDDINGS / "development_rows.jsonl")
    dev_rows = {r["image_id"]: r["output_row"] for r in dev_records}
    if len(dev_rows) != len(dev_records):
        raise ValueError("duplicate image IDs in development embedding handoff")
    raw = {m: np.load(EMBEDDINGS / f"{m}.npy", mmap_mode="r", allow_pickle=False) for m in MODELS}
    eval_rows = [dev_rows[images[i]["image_id"]] for i in eval_idx]
    train_rows = [dev_rows[images[i]["image_id"]] for i in train_primary]
    x_eval = {m: l2(raw[m][eval_rows]) for m in MODELS}
    x_train = {m: l2(raw[m][train_rows]) for m in MODELS}
    patterns, neural_eval = {}, {}
    for roi in ROIS:
        pat = np.load(fold_dir / f"{roi}_patterns.npy", mmap_mode="r", allow_pickle=False)
        if pat.ndim != 2 or pat.shape[0] != 4712 or pat.shape[1] < 2:
            raise ValueError(f"{roi}: unexpected pattern shape")
        if roi == ROIS[3] and not evc_ok:
            continue
        patterns[roi] = np.asarray(pat, dtype=np.float64)
        neural_eval[roi] = ranks(upper(corr_rdm(patterns[roi][eval_idx])))
    positions = np.full(4712, -1, dtype=int)
    positions[train_primary] = np.arange(len(train_primary))
    scopes = {}
    for name, rois in (("primary", PRIMARY), ("evc", (ROIS[3],))):
        rows = train_sets[rois[0]]
        if rows is None:
            continue
        labels = np.array([images[i]["fold"] for i in rows], dtype=int)
        folds = sorted(set(labels.tolist()))
        if folds != sorted(set(range(5)) - {held}):
            raise ValueError(f"{name}: expected all four outer training folds")
        groups = [np.flatnonzero(np.isin(labels, fs)) for fs in (folds[::2], folds[1::2])]
        if min(map(len, groups)) < 3:
            raise ValueError(f"{name}: insufficient inner images")
        splits = [(groups[0], groups[1]), (groups[1], groups[0])]
        neural = [{r: ranks(upper(corr_rdm(patterns[r][rows[test]]))) for r in rois}
                  for _, test in splits]
        scopes[name] = {"rows": rows, "rois": rois, "splits": splits,
                        "neural": neural, "x_positions": positions[rows]}
        report.setdefault("inner_fold_groups", {})[name] = [folds[::2], folds[1::2]]
    report["input_paths"] = {"preprocessing": str(preprocessing_report(person)),
                             "fold_directory": str(fold_dir), "embeddings": str(EMBEDDINGS)}
    return {"x_train": x_train, "x_eval": x_eval, "patterns": patterns,
            "neural_eval": neural_eval, "scopes": scopes}


def select_inner(views, scope, patterns):
    grids = {"power": ALPHAS, "diagonal": LAMBDA0, "full": LAMBDA0}
    rois, rows = scope["rois"], scope["rows"]
    scores = {f: np.empty((2, len(rois), len(grid))) for f, grid in grids.items()}
    for direction, (fit, _) in enumerate(scope["splits"]):
        zt, zv, d2 = views[direction]
        neural = scope["neural"][direction]
        for j, alpha in enumerate(ALPHAS):
            mr = ranks(upper(sq_rdm(zv * d2 ** ((float(alpha) - 1.0) / 2.0))))
            scores["power"][direction, :, j] = [checked_z(mr, neural[r]) for r in rois]
        for ri, roi in enumerate(rois):
            q = brain_q(patterns[roi][rows[fit]], zt)
            for family in ("diagonal", "full"):
                for j, lambda0 in enumerate(LAMBDA0):
                    w = fit_metric(d2, q, len(zt), lambda0, family)
                    mr = ranks(upper(metric_rdm(zv, w)))
                    scores[family][direction, ri, j] = checked_z(mr, neural[roi])
    chosen = {}
    for family, grid in grids.items():
        values = [float(v) for v in grid]
        means = scores[family].mean(axis=0).mean(axis=0)  # Directions, then ROIs.
        def tie_key(j):
            v = values[j]
            proximity = round(abs(v - 1.0), 12) if family == "power" else v
            return -float(means[j]), proximity, v
        best = min(range(len(values)), key=tie_key)
        chosen[family] = {"value": values[best], "boundary": best in (0, len(values) - 1),
                          "inner_mean_z": float(means[best]), "grid": values,
                          "mean_scores": means.tolist(),
                          "scores_by_roi": {r: scores[family][:, ri, :].tolist()
                                            for ri, r in enumerate(rois)}}
    return chosen


def run_cell(task_index, out_dir, rank):
    started = time.time()
    person, held = PARTICIPANTS[task_index // 5], task_index % 5
    report = {"analysis_id": ANALYSIS, "participant": person, "held_fold": held,
              "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "neural_inputs": "EP04 saved development mean patterns only",
              "ood_or_shared_values_read": False, "independent_unit": "participant",
              "settings": {"rank": rank, "alphas": ALPHAS.tolist(), "lambda0_grid": list(LAMBDA0),
                           "models": list(MODELS), "rois": list(ROIS), "numerical_rank_rtol": 1e-10,
                           "selection": "mean Fisher z over directions, then primary ROIs; EVC separately",
                           "full_fit": "symmetric closed form followed by Euclidean PSD projection",
                           "target": "row population-z-scored correlation Gram, trace normalized to n"},
              "models": {}}
    data = load_data(person, held, report)
    report["timing_load_seconds"] = time.time() - started
    for model in MODELS:
        model_started = time.time()
        reference = ranks(upper(cos_rdm(data["x_eval"][model])))
        result = {"selection": {}, "rois": {ROIS[3]: {"status": "not_evaluable"}},
                  "timings_seconds": {}}
        cache = {}
        for name, scope in data["scopes"].items():
            t0 = time.time()
            key = tuple(scope["rows"].tolist())
            if key not in cache:
                xt = data["x_train"][model][scope["x_positions"]]
                cache[key] = [project(xt[fit], xt[test], rank) for fit, test in scope["splits"]]
                cache[key].append(project(xt, data["x_eval"][model], rank))
            views = cache[key]
            t1 = time.time()
            chosen = select_inner(views, scope, data["patterns"])
            result["selection"][name] = chosen
            t2 = time.time()
            zt, ze, d2 = views[2]
            c, scale = float(len(zt) / d2.sum()), float(np.mean(d2 ** 2))
            alpha = chosen["power"]["value"]
            base = {"native": np.full(rank, c), "power": d2 ** (alpha - 1.0)}
            shared = {"native": ranks(upper(metric_rdm(ze, base["native"]))),
                      "power": ranks(upper(sq_rdm(ze * d2 ** ((alpha - 1.0) / 2.0))))}
            for roi in scope["rois"]:
                q = brain_q(data["patterns"][roi][scope["rows"]], zt)
                weights = dict(base)
                weights.update({f: fit_metric(d2, q, len(zt), chosen[f]["value"], f)
                                for f in ("diagonal", "full")})
                metrics = {}
                for family, w in weights.items():
                    mr = shared[family] if family in shared else ranks(upper(metric_rdm(ze, w)))
                    metrics[family] = {"rsa_z": checked_z(mr, data["neural_eval"][roi]),
                                       "numerical_rank": numerical_rank(w)}
                result["rois"][roi] = {
                    "status": "evaluable", "selection_scope": name, "training_n": len(zt),
                    "native_c": c, "lambda_scale": scale, "metrics": metrics,
                    "reference_cosine_z": checked_z(reference, data["neural_eval"][roi]),
                    "selected": {"alpha": alpha, "lambda0_diagonal": chosen["diagonal"]["value"],
                                 "lambda0_full": chosen["full"]["value"]},
                    "boundary": {f: chosen[f]["boundary"] for f in FAMILIES},
                    "lambda_absolute": {f: chosen[f]["value"] * scale for f in ("diagonal", "full")},
                    "contrasts": {
                        "full_minus_diagonal": metrics["full"]["rsa_z"] - metrics["diagonal"]["rsa_z"],
                        "power_minus_native": metrics["power"]["rsa_z"] - metrics["native"]["rsa_z"]}}
            result["timings_seconds"][name] = {
                "pca": t1 - t0, "inner_selection": t2 - t1, "outer_evaluation": time.time() - t2}
        result["timings_seconds"]["total"] = time.time() - model_started
        report["models"][model] = result
    report["elapsed_seconds"] = time.time() - started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path = out_dir / f"{person}-fold-{held}.json"
    write_json(path, report)
    print(path)


def collect(directory, output):
    cells, settings = {}, None
    for person in PARTICIPANTS:
        for fold in range(5):
            path = directory / f"{person}-fold-{fold}.json"
            if path.resolve() == output.resolve():
                raise ValueError("collection output would overwrite an input cell")
            cell = json.loads(path.read_text())
            if (cell["analysis_id"], cell["participant"], cell["held_fold"]) != (ANALYSIS, person, fold):
                raise ValueError(f"incorrect cell identity: {path}")
            if settings is None:
                settings = cell["settings"]
            elif cell["settings"] != settings:
                raise ValueError(f"mixed settings, including possibly mixed PCA ranks: {path}")
            cells[person, fold] = cell
    participant = {}
    for person in PARTICIPANTS:
        participant[person] = {}
        for model in MODELS:
            participant[person][model] = {}
            for contrast in CONTRASTS:
                values = np.array([[cells[person, f]["models"][model]["rois"][r]["contrasts"][contrast]
                                    for f in range(5)] for r in PRIMARY], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"non-finite contrast: {person}, {model}, {contrast}")
                participant[person][model][contrast] = float(values.mean(axis=1).mean())
        participant[person]["mean_over_models"] = {
            c: float(np.mean([participant[person][m][c] for m in MODELS])) for c in CONTRASTS}
    group = {}
    for model in (*MODELS, "mean_over_models"):
        group[model] = {}
        for contrast in CONTRASTS:
            values = [participant[p][model][contrast] for p in PARTICIPANTS]
            k = sum(v > 0 for v in values)
            group[model][contrast] = {
                "mean_delta_z": float(np.mean(values)), "n_participants": len(values),
                "k_positive": k, "n_zero": sum(v == 0 for v in values),
                "sign_p_one_sided": float(binomtest(k, len(values), p=0.5, alternative="greater").pvalue)}
    distributions = {s: {m: {f: {"counts": {}, "boundary_count": 0} for f in FAMILIES}
                          for m in MODELS} for s in ("primary", "evc")}
    for cell in cells.values():
        for model in MODELS:
            for scope, selections in cell["models"][model]["selection"].items():
                for family, selection in selections.items():
                    bucket = distributions[scope][model][family]
                    value = format(selection["value"], ".12g")
                    bucket["counts"][value] = bucket["counts"].get(value, 0) + 1
                    bucket["boundary_count"] += int(selection["boundary"])
    summary = {"analysis_id": ANALYSIS, "settings": settings, "n_cells": len(cells),
               "source_directory": str(directory.resolve()), "independent_unit": "participant",
               "aggregation": "equal folds within sector, equal primary sectors, then equal models",
               "primary_contrast": CONTRASTS[0], "secondary_contrast": CONTRASTS[1],
               "sign_test": "one-sided exact Binomial(n, 0.5); zeros count as nonpositive",
               "selection_distribution_unit": "participant-fold-model; descriptive counts only",
               "participant_contrasts": participant, "group": group,
               "selected_distributions": distributions,
               "evc_evaluable_cells": sum(c["counts"]["evc_evaluable"] for c in cells.values())}
    write_json(output, summary)
    print(f"{'model':<19} {'full-diag':>11} {'k/n':>5} {'p':>8} {'power-native':>13} {'k/n':>5} {'p':>8}")
    for model, results in group.items():
        a, b = (results[c] for c in CONTRASTS)
        print(f"{model:<19} {a['mean_delta_z']:11.6f} {a['k_positive']:>2}/5 "
              f"{a['sign_p_one_sided']:8.5f} {b['mean_delta_z']:13.6f} "
              f"{b['k_positive']:>2}/5 {b['sign_p_one_sided']:8.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-index", type=int, choices=range(25),
                      help="0–24: participant-major, fold-minor")
    mode.add_argument("--collect", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--rank", type=int, help="training PCA rank; default 64")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.collect is not None:
        if args.out is None or args.out_dir is not None or args.rank is not None:
            ap.error("collect mode requires --out and excludes --out-dir and --rank")
        collect(args.collect, args.out)
    else:
        if args.out_dir is None or args.out is not None:
            ap.error("cell mode requires --out-dir and excludes --out")
        rank = 64 if args.rank is None else args.rank
        if rank < 1:
            ap.error("--rank must be positive")
        run_cell(args.task_index, args.out_dir, rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
