"""EP23 A3: fixed-spectrum synthetic pooling simulations.

Task indices are zero-based; collection requires every task and seed.
Undefined cross-repeat correlation pairs are omitted and counted.
Results demonstrate possibilities and failure modes, not biological mechanisms.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

# Keep numerical reductions reproducible and avoid array-job oversubscription.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata, spearmanr

N, SIDE, K, TOP = 300, 64, 64, 8
A = np.arange(1, K + 1, dtype=float) ** -0.5
ALPHAS = np.arange(41, dtype=float) / 10.0
TOPOGRAPHIES = ("dct_ordered", "dct_shuffled", "white_random", "smooth_random")
MEASUREMENTS = (("identity", 0), ("pool_0", 0), ("pool_1", 1),
                ("pool_2", 2), ("pool_4", 4))
PAIRS = np.triu_indices(N, 1)
METRICS = ("mixing_fraction", "attenuation_spearman", "alpha_star",
           "rsa_max", "rsa_alpha1", "subspace_overlap", "rsa_valid_stimuli")
SCHEMA = "ep23-a3-v1"
GRID = [
    {"id": cid, "topography": topo, "measurement": label, "sigma": sigma,
     "downsample": label != "identity", "noise": noise}
    for cid, (topo, label, sigma, noise) in enumerate(
        (t, m, s, noise) for t in TOPOGRAPHIES
        for m, s in MEASUREMENTS for noise in ("none", "moderate"))
]

def rng(master, index, *tags):
    return np.random.default_rng(np.random.SeedSequence([master, index, *tags]))

def orthogonalize(values):
    q, r = np.linalg.qr(values, mode="reduced")
    return q * np.where(np.diag(r) < 0.0, -1.0, 1.0)

def dct_basis():
    coordinates = np.arange(SIDE, dtype=float) + 0.5
    frequencies = np.arange(SIDE, dtype=float)
    basis = np.sqrt(2.0 / SIDE) * np.cos(
        np.pi * frequencies[:, None] * coordinates[None, :] / SIDE)
    basis[0] /= np.sqrt(2.0)
    modes = sorted(
        ((u, v) for u in range(SIDE) for v in range(SIDE) if u or v),
        key=lambda uv: (uv[0] ** 2 + uv[1] ** 2, uv[0], uv[1]))[:K]
    return np.column_stack([np.outer(basis[u], basis[v]).ravel() for u, v in modes])

def topography(name, master, index, base):
    if name == "dct_ordered":
        return base
    if name == "dct_shuffled":
        return base[:, rng(master, index, 1).permutation(K)]
    # Share the initial Gaussian field between white and smooth topographies.
    field = rng(master, index, 2).standard_normal((SIDE, SIDE, K))
    if name == "smooth_random":
        field = gaussian_filter(field, (3.0, 3.0, 0.0), mode="reflect", truncate=4.0)
    return orthogonalize(field.reshape(SIDE * SIDE, K))

def measure(v, label, sigma):
    if label == "identity":
        return v
    field = v.reshape(SIDE, SIDE, K)
    if sigma:
        field = gaussian_filter(
            field, (sigma, sigma, 0), mode="reflect", truncate=4.0)
    return field.reshape(16, 4, 16, 4, K).mean(axis=(1, 3)).reshape(256, K)

def rank_unit(values):
    ranks = rankdata(values, method="average", axis=-1)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=-1, keepdims=True)
    return np.divide(ranks, norms, out=np.full_like(ranks, np.nan), where=norms > 0)

def rsa(gram, model_distances, model_ranks):
    diagonal = np.diag(gram)
    valid = np.isfinite(diagonal) & (diagonal > 0.0)
    count = int(valid.sum())
    if count < 3:
        return None, None, None, count
    mask = valid[PAIRS[0]] & valid[PAIRS[1]]
    i, j = PAIRS[0][mask], PAIRS[1][mask]
    # Cross-repeat distances may lie outside [0, 2]; do not clip or repair them.
    distances = 1.0 - gram[i, j] / (np.sqrt(diagonal[i]) * np.sqrt(diagonal[j]))
    models = model_ranks if mask.all() else rank_unit(model_distances[:, mask])
    scores = np.clip(models @ rank_unit(distances), -1.0, 1.0)
    if not np.isfinite(scores).any():
        return None, None, None, count
    best = int(np.nanargmax(scores))  # First maximum selects the smallest alpha.
    at_one = float(scores[10]) if np.isfinite(scores[10]) else None
    return float(ALPHAS[best]), float(scores[best]), at_one, count

def simulate(args):
    chosen = [c for c in GRID if c["id"] % args.n_tasks == args.task_index]
    if not chosen:
        return []
    base, records = dct_basis(), []
    for index in range(args.seeds):
        u = orthogonalize(rng(args.seed, index, 0).standard_normal((N, K)))
        x = u * A
        differences = (u[PAIRS[0]] - u[PAIRS[1]]) ** 2
        distances = (A[None, :] ** (2.0 * ALPHAS[:, None])) @ differences.T
        model_ranks = rank_unit(distances)
        for name in TOPOGRAPHIES:
            relevant = [c for c in chosen if c["topography"] == name]
            if not relevant:
                continue
            v = topography(name, args.seed, index, base)
            for mi, (label, sigma) in enumerate(MEASUREMENTS):
                conditions = [c for c in relevant if c["measurement"] == label]
                if not conditions:
                    continue
                vm = measure(v, label, sigma)
                cmat = vm.T @ vm
                diagonal = np.diag(cmat)
                off = cmat - np.diag(diagonal)
                mixing = float(np.sum(off * off) / np.sum(cmat * cmat))
                flat = np.ptp(diagonal) <= 1e-12 * np.max(np.abs(diagonal))
                attenuation = None if flat else float(
                    spearmanr(diagonal, np.arange(1, K + 1)).statistic)
                signal = x @ vm.T
                noise_sd = float(np.sqrt(np.var(signal, axis=0, ddof=0).mean()))
                for condition in conditions:
                    if condition["noise"] == "none":
                        cross = x @ cmat @ x.T  # Exactly B B^T algebraically.
                    else:
                        # Common standardized noise across topographies.
                        random = rng(args.seed, index, 3, mi)
                        b1 = signal + noise_sd * random.standard_normal(signal.shape)
                        b2 = signal + noise_sd * random.standard_normal(signal.shape)
                        cross = b1 @ b2.T
                    gram = (cross + cross.T) * 0.5
                    alpha, best, at_one, valid = rsa(gram, distances, model_ranks)
                    _, q = eigh(gram, subset_by_index=(N - TOP, N - 1))
                    overlap = np.sum((q.T @ u[:, :TOP]) ** 2) / TOP
                    records.append({
                        "condition_id": condition["id"], "seed_index": index,
                        "mixing_fraction": mixing, "attenuation_spearman": attenuation,
                        "alpha_star": alpha, "rsa_max": best, "rsa_alpha1": at_one,
                        "subspace_overlap": float(np.clip(overlap, 0.0, 1.0)),
                        "rsa_valid_stimuli": valid})
    return records

def configuration(args):
    return {
        "seed": args.seed, "seeds": args.seeds, "n": N, "sheet": [SIDE, SIDE], "K": K,
        "spectrum": "a_k=k^(-0.5), k=1..64", "alphas": ALPHAS.tolist(),
        "dct_order": "(u*u+v*v, u, v); exclude DC",
        "smooth_random_sigma": 3.0, "block_size": 4,
        "gaussian_boundary": "reflect", "gaussian_truncate": 4.0,
        "noise_variance": "mean(var(B, axis=0, ddof=0))",
        "rng": "PCG64; SeedSequence([seed, seed_index, stream, optional_measurement])",
        "pairing": "Shared U, white/smooth Gaussian field, and standardized noise",
        "rdm": "Normalized raw Gram; no voxel centering or distance clipping",
        "invalid_rsa_policy": "Omit pairs touching nonpositive/nonfinite diagonals",
        "constant_attenuation_rtol": 1e-12, "sd_ddof": 1,
        "alpha_ties": "smallest alpha", "subspace_dimension": TOP,
        "purpose": "Possibilities and failure modes; no biological mechanism or empirical tuning",
        "verdict_scope": "Descriptive condition means, not a significance test",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__}}

def statistics(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not values.size:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "min": float(values.min()), "max": float(values.max())}

def summarize(records, seeds):
    summaries, verdicts = [], []
    for condition in GRID:
        rows = sorted(
            (r for r in records if r["condition_id"] == condition["id"]),
            key=lambda r: r["seed_index"])
        if not rows:
            continue
        summaries.append({
            **condition, "seed_indices": [r["seed_index"] for r in rows],
            "alpha_star_values": [r["alpha_star"] for r in rows],
            "metrics": {key: statistics([r[key] for r in rows]) for key in METRICS}})
    for label, sigma in MEASUREMENTS:
        for noise in ("none", "moderate"):
            group = [s for s in summaries
                     if s["measurement"] == label and s["noise"] == noise]
            means = {s["topography"]: s["metrics"]["alpha_star"]["mean"] for s in group}
            complete = len(group) == len(TOPOGRAPHIES) and all(
                s["metrics"]["alpha_star"]["n"] == seeds for s in group)
            spread = float(max(means.values()) - min(means.values())) if complete else None
            verdicts.append({
                "measurement": label, "sigma": sigma, "noise": noise,
                "complete": complete, "topography_means": means,
                "alpha_star_mean_range": spread,
                "observed_mean_difference": None if spread is None else spread > 1e-12})
    return summaries, verdicts

def collect(directory, output):
    chunks = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.resolve() == output.resolve() or path.suffix == ".tmp":
            continue
        try:
            chunk = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(chunk, dict) and chunk.get("schema") == SCHEMA and chunk.get("kind") == "task":
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No EP23 A3 task outputs found.")
    config = chunks[0]["config"]
    n_tasks = chunks[0]["task"]["n_tasks"]
    seen, records = set(), []
    for chunk in chunks:
        if chunk["config"] != config or chunk["grid"] != GRID:
            raise ValueError("Incompatible task configuration or condition grid.")
        task = chunk["task"]
        index = task["index"]
        if task["n_tasks"] != n_tasks or not 0 <= index < n_tasks or index in seen:
            raise ValueError("Inconsistent, invalid, or duplicate task index.")
        expected = {(c["id"], s) for c in GRID if c["id"] % n_tasks == index
                    for s in range(config["seeds"])}
        actual = [(r["condition_id"], r["seed_index"]) for r in chunk["records"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"Task {index} has missing, duplicate, or unexpected records.")
        seen.add(index)
        records.extend(chunk["records"])
    if seen != set(range(n_tasks)):
        raise ValueError(f"Missing task indices: {sorted(set(range(n_tasks)) - seen)}")
    return config, records

def print_table(summaries, verdicts):
    def fmt(value):
        return "NA" if value is None else f"{value:.3f}"
    print("topography       measure   noise       alpha*(sd)   RSAmax   RSA@1     mix     rho overlap")
    columns = ("rsa_max", "rsa_alpha1", "mixing_fraction",
               "attenuation_spearman", "subspace_overlap")
    for row in summaries:
        metrics, alpha = row["metrics"], row["metrics"]["alpha_star"]
        print(f"{row['topography']:<16} {row['measurement']:<9} {row['noise']:<8} "
              f"{fmt(alpha['mean'])}({fmt(alpha['sd'])}) "
              + " ".join(f"{fmt(metrics[key]['mean']):>7}" for key in columns))
    for verdict in verdicts:
        if verdict["complete"]:
            print(f"alpha* mean range {verdict['measurement']}/{verdict['noise']}: "
                  f"{verdict['alpha_star_mean_range']:.3f}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--n-tasks", type=int)
    parser.add_argument("--collect", type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or args.seed < 0:
        parser.error("--seeds must be positive and --seed nonnegative.")
    try:
        if args.collect is not None:
            if args.task_index is not None or args.n_tasks is not None:
                parser.error("--collect cannot be combined with task partition options.")
            config, records = collect(args.collect, args.out)
            task = None
        else:
            if (args.task_index is None) != (args.n_tasks is None):
                parser.error("--task-index and --n-tasks must be supplied together.")
            if args.task_index is None:
                args.task_index, args.n_tasks = 0, 1
            if args.n_tasks < 1 or not 0 <= args.task_index < args.n_tasks:
                parser.error("Require --n-tasks >= 1 and 0 <= --task-index < --n-tasks.")
            config, records = configuration(args), simulate(args)
            task = {"index": args.task_index, "n_tasks": args.n_tasks}
        records.sort(key=lambda r: (r["condition_id"], r["seed_index"]))
        summaries, verdicts = summarize(records, config["seeds"])
        document = {"schema": SCHEMA, "kind": "collection" if task is None else "task",
                    "config": config, "task": task, "grid": GRID, "records": records,
                    "conditions": summaries, "topography_dependence": verdicts}
        serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=args.out.parent,
                prefix=f".{args.out.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(serialized)
        os.replace(handle.name, args.out)
        print_table(summaries, verdicts)
    except (OSError, ValueError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()
