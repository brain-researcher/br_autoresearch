#!/usr/bin/env python3
"""EP23 A2: coarse/fine model geometry and reliability-relative brain alignment."""
import argparse
import csv
import itertools
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

N_ROWS, N_CONCEPTS = 23560, 1854
MODELS = {"CLIP": 1024, "PEcore": 1024, "SigLIP2": 1152, "DINOv2": 1024}
ROIS = ("laion-ventral", "laion-lateral", "laion-dorsal", "retinotopic-EVC")
SCALES = ("coarse", "fine")
COMMON_INPUTS = ("embeddings_dir", "dev_rows", "things_meta", "things_c27")
IMAGE_RE = re.compile(r"^unique_THINGS\+?_(.+)_p\d+\.jpg$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_safe(value):
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def read_design(args):
    with args.things_meta.open(encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require((reader.fieldnames or []).count("uniqueID") == 1, "Missing/duplicate uniqueID column")
        unique_ids = [row["uniqueID"] for row in reader]
    require(len(unique_ids) == N_CONCEPTS, "THINGS metadata must contain 1,854 concepts")
    require(all(unique_ids) and len(set(unique_ids)) == N_CONCEPTS, "Invalid/duplicate uniqueIDs")
    concept_lookup = {name: i for i, name in enumerate(unique_ids)}
    with args.things_c27.open(encoding="utf-8-sig", errors="replace") as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    require(len(table) == N_CONCEPTS + 1, "category27 row count mismatch")
    names = table[0]
    require(len(names) == 27 and len(set(names)) == 27 and all(names), "Invalid category header")
    require(all(len(row) == 27 and all(x in ("0", "1") for x in row) for row in table[1:]),
            "category27 must contain exactly 27 binary columns")
    membership = np.asarray(table[1:], dtype=np.int64)
    single_labels = membership.argmax(axis=1)
    single_labels[membership.sum(axis=1) != 1] = -1

    seen = np.zeros(N_ROWS, dtype=bool)
    concepts = np.full(N_ROWS, -1, dtype=np.int64)
    lookup = {}
    unmatched = 0
    with args.dev_rows.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            output_row = row["output_row"]
            require(type(output_row) is int and 0 <= output_row < N_ROWS,
                    f"Invalid output_row at development line {line_number}")
            require(not seen[output_row], f"Duplicate output_row: {output_row}")
            require(all(isinstance(row[k], str) and row[k]
                        for k in ("participant", "image_id", "source_stratum")), "Invalid dev identity")
            key = (row["participant"], row["image_id"])
            require(key not in lookup, f"Duplicate participant/image identity: {key}")
            if row["source_stratum"] in ("things", "thingsplus"):
                match = IMAGE_RE.match(row["image_id"])
                concepts[output_row] = concept_lookup.get(match.group(1), -1) if match else -1
                unmatched += int(concepts[output_row] < 0)
            lookup[key] = output_row
            seen[output_row] = True
    require(seen.all(), "Development rows must cover every output_row from 0 through 23559")
    participants = sorted({key[0] for key in lookup})
    require(len(participants) == 5, "Expected exactly five development participants")
    require(all(re.fullmatch(r"sub-\d{2}", p) for p in participants), "Invalid participant names")
    labels = np.where(concepts >= 0, single_labels[np.maximum(concepts, 0)], -1)
    return dict(names=names, concepts=concepts, labels=labels, lookup=lookup,
                participants=participants, unmatched_things_rows=unmatched)


def select_images(candidates, labels, minimum):
    candidates = np.asarray(candidates, dtype=np.int64)
    keep = candidates[labels[candidates] >= 0]
    categories, counts = np.unique(labels[keep], return_counts=True)
    categories = categories[counts >= minimum]
    keep = keep[np.isin(labels[keep], categories)]
    encoded = np.searchsorted(categories, labels[keep])
    rank = min(8, (len(categories) - 1) // 2)
    require(rank >= 2, f"Insufficient categories: C={len(categories)}, minimum images={minimum}")
    return keep, encoded, categories, rank


def describe_set(selection, output_rows, names):
    keep, labels, categories, rank = selection
    require(len(output_rows) == len(keep), "Selected embedding-row count mismatch")
    counts = np.bincount(labels, minlength=len(categories))
    return {
        "n_images": len(keep), "C": len(categories), "r": rank,
        "categories": [{"index": int(c), "name": names[c], "count": int(counts[i])}
                       for i, c in enumerate(categories)],
        "dev_output_rows": np.asarray(output_rows).tolist(),
    }


def centroids(matrix, labels):
    return np.vstack([matrix[labels == c].mean(axis=0) for c in range(int(labels.max()) + 1)])


def top_vectors(left, rank, right=None):
    """Largest algebraic eigenvectors of J K J; large kernels stay matrix-free."""
    require(left.ndim == 2 and left.shape[0] > rank, "Invalid kernel feature shape")
    require(right is None or right.shape == left.shape, "Cross-kernel feature shape mismatch")
    left = np.asarray(left, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    cross = right is not None
    right = left if right is None else np.asarray(right, dtype=np.float64)
    if cross:
        right = right - right.mean(axis=0, keepdims=True)
    n = left.shape[0]

    def multiply(vector):
        vector = vector - vector.mean(axis=0, keepdims=True)
        result = left @ (right.T @ vector)
        if cross:
            result = (result + right @ (left.T @ vector)) / 2
        return result - result.mean(axis=0, keepdims=True)

    if n <= 1024:
        kernel = left @ right.T
        kernel = (kernel + kernel.T) / 2
        kernel = kernel - kernel.mean(0, keepdims=True) - kernel.mean(1, keepdims=True) + kernel.mean()
        kernel = (kernel + kernel.T) / 2
        values, vectors = eigh(kernel, subset_by_index=(n - rank, n - 1), check_finite=False)
    else:
        operator = LinearOperator((n, n), matvec=multiply, matmat=multiply,
                                  rmatvec=multiply, dtype=np.float64)
        # This fixed solver start is independent of all permutation RNG streams.
        values, vectors = eigsh(operator, k=rank, which="LA", tol=1e-10, maxiter=10000,
                                ncv=min(n, max(32, 2 * rank + 1)),
                                v0=np.random.default_rng(0).normal(size=n))
    vectors = vectors[:, np.argsort(values)[::-1]]
    require(np.isfinite(vectors).all() and
            np.allclose(vectors.T @ vectors, np.eye(rank), atol=1e-7, rtol=0),
            "Eigensolver returned invalid eigenvectors")
    return vectors


def overlap(first, second):
    value = float(np.sum((first.T @ second) ** 2) / first.shape[1])
    return float(np.clip(value, 0, 1))  # Remove only floating-point boundary excursions.


def null_statistics(comparisons, labels, args, participant_index, scale_index, roi_index):
    shape = next(iter(comparisons.values()))[0].shape
    require(all(u.shape == v.shape == shape for u, v in comparisons.values()), "Basis shape mismatch")
    n = shape[0]
    require(scale_index == 0 or labels.shape == (n,), "Fine-null label shape mismatch")
    rng = np.random.default_rng(np.random.SeedSequence(
        [args.seed, participant_index, scale_index, roi_index]))
    blocks = [np.flatnonzero(labels == c) for c in np.unique(labels)] if scale_index == 1 else []
    observed = {key: overlap(u, v) for key, (u, v) in comparisons.items()}
    nulls = {key: np.empty(args.n_perm) for key in comparisons}
    for draw in range(args.n_perm):
        permutation = rng.permutation(n) if scale_index == 0 else np.arange(n)
        for block in blocks:
            permutation[block] = rng.permutation(block)
        # One draw is shared by all models and self-consistency in this call.
        for key, (u, v) in comparisons.items():
            nulls[key][draw] = overlap(u[permutation], v)
    results = {}
    for key, null in nulls.items():
        omega, mean = observed[key], float(null.mean())
        require(mean < 1, f"Undefined kappa: saturated permutation null for {key}")
        results[key] = {
            "omega": omega, "null_mean": mean,
            "null_p95": float(np.quantile(null, 0.95, method="linear")),
            "p": float((1 + np.count_nonzero(null >= omega)) / (1 + args.n_perm)),
            "kappa": (omega - mean) / (1 - mean),
        }
    return results


def distance_vectors(matrix, labels, metric):
    return {
        "coarse": pdist(centroids(matrix, labels), metric=metric),
        "fine": np.concatenate([pdist(matrix[labels == c], metric=metric)
                                for c in range(int(labels.max()) + 1)]),
    }


def rsa(first, second):
    require(first.ndim == second.ndim == 1 and first.shape == second.shape, "RSA pair mismatch")
    result = {"n_pairs": len(first), "spearman": None, "fisher_z": None}
    if len(first) < 2 or not (np.isfinite(first).all() and np.isfinite(second).all()):
        return result | {"status": "undefined_distances"}
    a, b = rankdata(first, method="average"), rankdata(second, method="average")
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return result | {"status": "constant_distances"}
    correlation = float(np.clip(a @ b / denominator, -1, 1))
    return result | {"spearman": correlation,
                     "fisher_z": float(np.arctanh(np.clip(correlation, -0.999999, 0.999999))),
                     "status": "ok"}


def model_geometry(directory, output_rows, labels, rank, with_rsa):
    result = {}
    for model, dimension in MODELS.items():
        raw = np.load(directory / f"{model}.npy", mmap_mode="r", allow_pickle=False)
        require(raw.shape == (N_ROWS, dimension) and raw.dtype == np.float16,
                f"{model}: embedding shape/dtype mismatch")
        matrix = np.asarray(raw[output_rows], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        require(np.isfinite(matrix).all() and np.all(norms > 0), f"{model}: invalid embedding rows")
        matrix /= norms
        means = centroids(matrix, labels)
        result[model] = {"coarse": top_vectors(means, rank),
                         "fine": top_vectors(matrix - means[labels], rank)}
        if with_rsa:
            result[model]["rdm"] = distance_vectors(matrix, labels, "cosine")
    return result


def read_patterns(archive, roi, rows, selected):
    positions = {int(row): i for i, row in enumerate(rows)}
    take = np.asarray([positions[int(row)] for row in selected], dtype=np.int64)
    matrices, expected_shape = [], None
    for suffix in ("A", "B", "full"):
        raw = archive[f"{roi}__{suffix}"]
        require(raw.ndim == 2 and raw.shape[0] == len(rows) and raw.shape[1] > 0,
                f"{roi}/{suffix}: pattern shape mismatch")
        require(raw.dtype == np.float32 and np.isfinite(raw).all(),
                f"{roi}/{suffix}: invalid pattern dtype/values")
        expected_shape = raw.shape if expected_shape is None else expected_shape
        require(raw.shape == expected_shape, f"{roi}: voxel count mismatch across A/B/full")
        matrices.append(np.asarray(raw[take], dtype=np.float64))
    return matrices


def analyze_roi(patterns, models, labels, rank, args, participant_index, roi_index):
    a, b, full = patterns
    ca, cb = centroids(a, labels), centroids(b, labels)
    result = {"self_consistency": {}, "models": {m: {} for m in MODELS}}
    for scale_index, scale in enumerate(SCALES):
        left, right = (ca, cb) if scale == "coarse" else (a - ca[labels], b - cb[labels])
        brain_basis = top_vectors(left, rank, right)
        comparisons = {"self": (top_vectors(left, rank), top_vectors(right, rank))}
        comparisons.update({m: (models[m][scale], brain_basis) for m in MODELS})
        stats = null_statistics(comparisons, labels, args, participant_index, scale_index, roi_index)
        self_stats = stats.pop("self")
        reliable = self_stats["p"] < 0.05
        require(not reliable or self_stats["kappa"] != 0, "Significant self-consistency has zero kappa")
        status = "reliable" if reliable else "unreliable"
        result["self_consistency"][scale] = self_stats | {"reliable": reliable, "status": status}
        for model in MODELS:
            relative = stats[model]["kappa"] / self_stats["kappa"] if reliable else None
            result["models"][model][scale] = stats[model] | {"rho": relative, "status": status}
    brain_distances = distance_vectors(full, labels, "correlation")
    for model, cell in result["models"].items():
        cell["rsa"] = {s: rsa(models[model]["rdm"][s], brain_distances[s]) for s in SCALES}
        coarse, fine = cell["coarse"], cell["fine"]
        cell["rho_coarse_minus_fine"] = (
            coarse["rho"] - fine["rho"] if coarse["rho"] is not None and fine["rho"] is not None else None)
        cell["kappa_coarse_minus_fine"] = coarse["kappa"] - fine["kappa"]
    return result


def primary_contrasts(rois):
    cells = [rois[roi]["models"][model] for roi in ROIS[:3] for model in MODELS]
    relative = [c["rho_coarse_minus_fine"] for c in cells if c["rho_coarse_minus_fine"] is not None]
    return {"rho_coarse_minus_fine": float(np.mean(relative)) if relative else None,
            "kappa_coarse_minus_fine": float(np.mean([c["kappa_coarse_minus_fine"] for c in cells])),
            "n_reliable_cells": len(relative), "n_cells": 12}


def participant_analysis(args, data):
    participant = args.participant
    require(participant in data["participants"], f"Unknown participant: {participant}")
    participant_index = data["participants"].index(participant)
    rois, timings, cache = {}, {}, {}
    with np.load(args.brain_npz, allow_pickle=False) as archive:
        ids, concepts = archive["image_ids"], archive["concept_indices"]
        require(ids.ndim == 1 and ids.dtype.kind in "US" and len(ids) > 0, "Invalid brain image_ids")
        ids = ids.astype(str)
        require(len(set(ids)) == len(ids), "Duplicate brain image_ids")
        require(concepts.shape == ids.shape and concepts.dtype == np.int64, "Invalid brain concept_indices")
        embedding_rows = np.asarray([data["lookup"][(participant, image_id)] for image_id in ids])
        require(np.array_equal(concepts, data["concepts"][embedding_rows]), "Brain/dev concept identity mismatch")
        rows = {roi: archive[f"{roi}__rows"] for roi in ROIS}
        for roi, indices in rows.items():
            require(indices.ndim == 1 and indices.dtype == np.int64 and
                    len(np.unique(indices)) == len(indices) and
                    np.all((indices >= 0) & (indices < len(ids))), f"{roi}: invalid/duplicate row indices")
        common = rows[ROIS[0]]
        for roi in ROIS[1:3]:
            common = np.intersect1d(common, rows[roi], assume_unique=True)
        labels = data["labels"][embedding_rows]
        common = common[np.argsort(embedding_rows[common])]
        primary = select_images(common, labels, 10)
        evc_rows = np.intersect1d(primary[0], rows[ROIS[3]], assume_unique=True)
        evc = select_images(evc_rows[np.argsort(embedding_rows[evc_rows])], labels, 10)
        for scope, selection, selected_rois in (("primary", primary, ROIS[:3]), ("evc", evc, ROIS[3:])):
            selected, encoded, _, rank = selection
            output_rows = embedding_rows[selected]
            started = time.perf_counter()
            key = tuple(output_rows)
            if key not in cache:
                cache[key] = model_geometry(args.embeddings_dir, output_rows, encoded, rank, True)
            timings[f"{scope}_models_seconds"] = time.perf_counter() - started
            info = describe_set(selection, output_rows, data["names"])
            for roi in selected_rois:
                started = time.perf_counter()
                patterns = read_patterns(archive, roi, rows[roi], selected)
                rois[roi] = info | analyze_roi(patterns, cache[key], encoded, rank, args,
                                              participant_index, ROIS.index(roi))
                rois[roi]["descriptive"] = roi == ROIS[3]
                timings[f"{roi}_seconds"] = time.perf_counter() - started
    return describe_set(primary, embedding_rows[primary[0]], data["names"]) | {
        "participant": participant, "rois": rois, "contrasts": primary_contrasts(rois), "timings": timings}


def pooled_analysis(args, data):
    selection = select_images(np.arange(N_ROWS), data["labels"], 20)
    output_rows, labels, _, rank = selection
    started = time.perf_counter()
    models = model_geometry(args.embeddings_dir, output_rows, labels, rank, False)
    timings = {"models_seconds": time.perf_counter() - started}
    pairs = list(itertools.combinations(MODELS, 2))
    results = {f"{a}__{b}": {"models": [a, b]} for a, b in pairs}
    for scale_index, scale in enumerate(SCALES):
        comparisons = {f"{a}__{b}": (models[a][scale], models[b][scale]) for a, b in pairs}
        stats = null_statistics(comparisons, labels, args, 5, scale_index, 0)
        for key in results:
            results[key][scale] = stats[key]
    contrasts = {}
    for key, cell in results.items():
        contrasts[key] = cell["coarse"]["kappa"] - cell["fine"]["kappa"]
        cell["kappa_coarse_minus_fine"] = contrasts[key]
    return describe_set(selection, output_rows, data["names"]) | {
        "participant": None, "model_pairs": results, "contrasts": contrasts, "timings": timings}


def sign_test(values):
    values = [v for v in values if v is not None]
    require(all(math.isfinite(v) for v in values), "Nonfinite collected contrast")
    n, k = len(values), sum(v > 0 for v in values)
    return {"n_nonnull": n, "k_positive": k,
            "p_one_sided": sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)}


def collect_analysis(args):
    documents, files = [], []
    for path in sorted(args.collect_dir.glob("*.json")):
        if path.resolve() == args.out.resolve():
            continue
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if isinstance(document, dict) and document.get("analysis") == "EP23_A2":
            if document.get("mode") in ("participant", "pooled"):
                require(document.get("schema_version") == 1, f"Unsupported schema: {path}")
                documents.append(document)
                files.append(str(path.resolve()))
    participants = [d for d in documents if d["mode"] == "participant"]
    pooled = [d for d in documents if d["mode"] == "pooled"]
    require(len(participants) == 5 and len(pooled) == 1, "Collect requires five participant files and one pooled file")
    settings = pooled[0]["settings"]
    by_participant = {d["participant"]: d for d in participants}
    require(len(by_participant) == 5 and set(by_participant) == set(settings["participant_indices"]),
            "Collected participant identities mismatch")
    identity_keys = COMMON_INPUTS
    for document in documents:
        require(document["settings"] == settings, "Collected settings mismatch")
        require(all(document["inputs"][k] == pooled[0]["inputs"][k] for k in identity_keys),
                "Collected input identities mismatch")
    for document in participants:
        require(set(document["rois"]) == set(ROIS), "Collected ROI identity mismatch")
        require(all(set(document["rois"][r]["models"]) == set(MODELS) for r in ROIS),
                "Collected model identity mismatch")
        require(document["contrasts"] == primary_contrasts(document["rois"]), "Collected contrast mismatch")
    contrasts = {p: by_participant[p]["contrasts"] for p in sorted(by_participant)}
    means = {}
    for roi in ROIS:
        means[roi] = {}
        for model in MODELS:
            means[roi][model] = {}
            for scale in SCALES:
                values = [d["rois"][roi]["models"][model][scale]["kappa"] for d in participants]
                require(all(v is not None and math.isfinite(v) for v in values), "Invalid collected kappa")
                means[roi][model][scale] = {"mean_kappa": float(np.mean(values)), "n_participants": 5}
    group = {"participant_contrasts": contrasts, "mean_kappa": means,
             "primary_sign_test": sign_test([c["rho_coarse_minus_fine"] for c in contrasts.values()]),
             "raw_kappa_sign_test": sign_test([c["kappa_coarse_minus_fine"] for c in contrasts.values()])}
    print("Participant  Reliable     Delta rho   Delta kappa")
    for participant, contrast in contrasts.items():
        relative = contrast["rho_coarse_minus_fine"]
        formatted = "null" if relative is None else f"{relative:+.6f}"
        print(f"{participant:11s} {contrast['n_reliable_cells']:2d}/12     {formatted:>10s}"
              f"   {contrast['kappa_coarse_minus_fine']:+.6f}")
    for key in ("primary_sign_test", "raw_kappa_sign_test"):
        test = group[key]
        print(f"{key}: k={test['k_positive']}/{test['n_nonnull']}, p={test['p_one_sided']:.6g}")
    return {"settings": settings, "inputs": {"collect_dir": str(args.collect_dir.resolve()), "files": files},
            "participants": by_participant, "pooled": pooled[0], "group": group}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("participant", "pooled", "collect"), required=True)
    parser.add_argument("--participant")
    parser.add_argument("--out", type=Path, required=True)
    for name in ("brain-npz", "embeddings-dir", "dev-rows", "things-meta", "things-c27", "collect-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    require(args.n_perm >= 1 and args.seed >= 0, "n-perm must be positive; seed must be nonnegative")
    started = time.perf_counter()
    if args.mode == "collect":
        if args.collect_dir is None:
            parser.error("collect mode requires --collect-dir")
        result = collect_analysis(args)
    else:
        if any(getattr(args, key) is None for key in COMMON_INPUTS):
            parser.error("participant/pooled modes require embeddings, development rows, and both THINGS files")
        if args.mode == "participant" and (args.participant is None or args.brain_npz is None):
            parser.error("participant mode requires --participant and --brain-npz")
        data = read_design(args)
        loaded = time.perf_counter()
        inputs = {key: str(getattr(args, key).resolve()) for key in COMMON_INPUTS}
        inputs["unmatched_things_rows"] = data["unmatched_things_rows"]
        if args.mode == "participant":
            inputs["brain_npz"] = str(args.brain_npz.resolve())
        result = participant_analysis(args, data) if args.mode == "participant" else pooled_analysis(args, data)
        result["inputs"] = inputs
        result["settings"] = {
            "seed": args.seed, "n_perm": args.n_perm, "models": MODELS,
            "participant_indices": {p: i for i, p in enumerate(data["participants"])},
            "pooled_participant_index": 5, "pooled_roi_index": 0,
            "scale_indices": {s: i for i, s in enumerate(SCALES)},
            "roi_indices": {r: i for i, r in enumerate(ROIS)}, "primary_rois": list(ROIS[:3]),
            "minimum_images": {"participant": 10, "pooled": 20}, "rank_rule": "min(8, (C-1)//2)",
            "reliability_rule": "self p < 0.05", "eigenvectors": "largest algebraic",
            "scope": "Exploratory exposed LAION-fMRI development; single-label category27 manual",
        }
        result["timings"]["design_loading_seconds"] = loaded - started
    result.update(analysis="EP23_A2", schema_version=1, mode=args.mode)
    result.setdefault("timings", {})["total_seconds"] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(result), handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
