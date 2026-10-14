"""EP23 A0 alignment analysis; execute only inside the orchestrator's allocation.

operators.npz supplies authoritative THINGS identities; things-meta is recorded
as an input path only.
Smaller EVC sets inherit shared random orders, restricted within their own bins
or signature groups, preserving both valid marginal nulls and shared draws.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as partial_eigh

PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
TASKS = PARTICIPANTS + ("pooled",)
SOURCES = ("vg", "captions")
PRIMARY_ROIS = ("laion-ventral", "laion-lateral", "laion-dorsal")
ROIS = PRIMARY_ROIS + ("retinotopic-EVC",)
MODELS = {"CLIP": 1024, "PEcore": 1024, "SigLIP2": 1152, "DINOv2": 1024}
STATS = ("primary", "taxonomy_residual", "frequency_preserving",
         "visual_residual", "taxonomy_visual_residual")
BANDS = {"band_1_8": (0, 8), "band_9_16": (8, 16), "band_17_32": (16, 32)}
PAIR_SLICES = {"pair_q8": (0, 8), **{f"pair_{k}": v for k, v in BANDS.items()}}
STAT_INDEX = {k: i for i, k in enumerate(
    STATS + ("q4", "q16", "full_mean", "vg_direct") + tuple(BANDS) + tuple(PAIR_SLICES))}
PATH_KEYS = ("operators", "brain_dir", "embeddings_dir", "lowlevel_dir",
             "things_meta", "things_c27", "dev_rows")
SCHEMA = "ep23_a0_alignment_v1"
N_IMAGES, N_CONCEPTS = 23560, 1854


def require(condition, message):
    """Raise an explicit validation error, including under python -O."""
    if not condition:
        raise ValueError(message)


def vector(value, kinds, name, length=None):
    """Validate a typed vector and optionally its length."""
    a = np.asarray(value)
    require(a.ndim == 1 and a.dtype.kind in kinds, f"{name}: invalid vector")
    require(length is None or len(a) == length, f"{name}: length mismatch")
    return a.astype(str) if kinds == "US" else a


def center(k):
    """Double-center a square kernel and symmetrize it."""
    k = np.asarray(k, dtype=np.float64)
    require(k.ndim == 2 and k.shape[0] == k.shape[1] and len(k) > 0,
            "Kernel must be a nonempty square matrix")
    require(np.isfinite(k).all(), "Nonfinite kernel")
    c = k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()
    return (c + c.T) * 0.5


def top(k, q, partial=False):
    """Return the leading q eigenvectors in descending algebraic order."""
    require(k.ndim == 2 and k.shape[0] == k.shape[1] and 0 < q <= len(k),
            f"Invalid eigenspace shape {k.shape}, q={q}")
    require(np.isfinite(k).all(), "Nonfinite eigendecomposition input")
    k = (k + k.T) * 0.5
    if partial:
        w, u = partial_eigh(k, subset_by_index=(len(k) - q, len(k) - 1),
                            driver="evr", check_finite=False)
    else:
        w, u = np.linalg.eigh(k)
    require(np.isfinite(w).all() and np.isfinite(u).all(), "Nonfinite eigensystem")
    return u[:, -q:][:, ::-1]


def overlap(u, v):
    """Compute normalized subspace overlap, clipping only roundoff."""
    require(u.shape == v.shape and u.shape[1] > 0, "Eigenspace shape mismatch")
    value = float(np.sum((u.T @ v) ** 2) / u.shape[1])
    require(np.isfinite(value) and -1e-9 <= value <= 1 + 1e-9, "Invalid overlap")
    return float(np.clip(value, 0, 1))


def residual(k, basis):
    """Compute (I-BB') K (I-BB') efficiently for orthonormal B."""
    bk = basis.T @ k
    left = basis @ bk
    result = k - left - left.T + basis @ (bk @ basis) @ basis.T
    return (result + result.T) * 0.5


def means(values, labels, concepts, normalize=False):
    """Average image rows by concept, optionally normalizing each image first."""
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 2 and len(x) == len(labels) and x.shape[1] > 0,
            "Concept aggregation shape mismatch")
    require(np.isfinite(x).all(), "Nonfinite feature or brain pattern")
    ix = np.searchsorted(concepts, labels)
    require(np.all(ix < len(concepts)), "Image concept outside requested set")
    require(np.array_equal(concepts[ix], labels), "Image concept identity mismatch")
    counts = np.bincount(ix, minlength=len(concepts))
    require(np.all(counts > 0), "Requested concept has no images")
    if normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        require(np.all(norms > 0) and np.isfinite(norms).all(), "Invalid embedding norm")
        x = x / norms
    result = np.zeros((len(concepts), x.shape[1]), dtype=np.float64)
    np.add.at(result, ix, x)
    return result / counts[:, None]


def read_inputs(args):
    """Validate identities, operators, metadata and feature-file alignment."""
    paths = {k: Path(getattr(args, k)).resolve() for k in PATH_KEYS}
    for key, path in paths.items():
        require(path.is_dir() if key.endswith("_dir") else path.is_file(),
                f"Missing or invalid input: {key}={path}")
        require(args.out != path and not (path.is_dir() and path in args.out.parents),
                f"Output would modify an input location: {path}")
    sources = {}
    with np.load(paths["operators"], allow_pickle=False) as z:
        uids = vector(z["things_uids"], "US", "things_uids", N_CONCEPTS)
        require(len(set(uids)) == N_CONCEPTS and np.all(uids != ""), "Invalid THINGS UIDs")
        for source in SOURCES:
            c = vector(z[f"{source}_concepts"], "iu", f"{source} concepts")
            require(len(c) >= 32 and np.all(np.diff(c.astype(np.int64)) > 0)
                    and np.all((c >= 0) & (c < N_CONCEPTS)), "Invalid source concepts")
            counts = vector(z[f"{source}_counts"], "iuf", f"{source} counts", len(c))
            require(np.isfinite(counts).all() and np.all(counts > 0)
                    and np.all(counts == np.floor(counts)), "Invalid operator counts")
            entry = {"concepts": c, "counts": counts.astype(np.float64)}
            for variant in ("kernel", "direct"):
                k = np.asarray(z[f"{source}_{variant}"], dtype=np.float64)
                require(k.shape == (len(c), len(c)) and np.isfinite(k).all(),
                        f"{source}/{variant}: kernel shape or finiteness mismatch")
                require(np.allclose(k, k.T, rtol=1e-8, atol=1e-10),
                        f"{source}/{variant}: asymmetric kernel")
                entry[variant] = (k + k.T) * 0.5
            direct = entry["direct"]
            require(np.all(direct >= 0) and np.all(np.diag(direct) == 0),
                    f"{source}: invalid direct counts")
            degrees = direct.sum(axis=1)  # Source-wide, before any restriction.
            inv = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees),
                            where=degrees > 0)
            entry["direct"] = direct * inv[:, None] * inv[None, :]
            sources[source] = entry
    with paths["things_c27"].open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    require(len(table) == N_CONCEPTS + 1 and all(len(r) == 27 for r in table),
            "category27_manual.tsv must have a 27-column header and 1854 rows")
    require(len(set(table[0])) == 27 and all(table[0]), "Invalid category names")
    require(all(v in ("0", "1") for row in table[1:] for v in row), "Invalid category bit")
    categories = np.asarray(table[1:], dtype=np.float64)
    with paths["dev_rows"].open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    require(len(records) == N_IMAGES and all(type(r["output_row"]) is int for r in records),
            "Invalid development-row count or output_row type")
    records.sort(key=lambda r: r["output_row"])
    require([r["output_row"] for r in records] == list(range(N_IMAGES)),
            "output_row must cover 0..23559 exactly once")
    uid_index, by_key = {uid: i for i, uid in enumerate(uids)}, {}
    concepts = np.full(N_IMAGES, -1, dtype=np.int64)
    for i, r in enumerate(records):
        require(r["participant"] in PARTICIPANTS and isinstance(r["image_id"], str)
                and r["image_id"] and isinstance(r["source_stratum"], str),
                f"Invalid development row {i}")
        key = (r["participant"], r["image_id"])
        require(key not in by_key, f"Duplicate participant/image identity: {key}")
        by_key[key] = i
        if r["source_stratum"] in ("things", "thingsplus"):
            match = re.match(r"^unique_THINGS\+?_(.+)_p\d+\.jpg$", r["image_id"])
            # Truncated THINGSplus names stay unmapped (-1), as in the brain patterns.
            if match is not None and match.group(1) in uid_index:
                concepts[i] = uid_index[match.group(1)]
    embeddings = {}
    for model, width in MODELS.items():
        x = np.load(paths["embeddings_dir"] / f"{model}.npy",
                    mmap_mode="r", allow_pickle=False)
        require(x.shape == (N_IMAGES, width) and x.dtype == np.float16,
                f"{model}: embedding shape or dtype mismatch")
        embeddings[model] = x
    visual = np.load(paths["lowlevel_dir"] / "low_level_features.npy",
                     mmap_mode="r", allow_pickle=False)
    require(visual.shape == (N_IMAGES, 3150) and visual.dtype == np.float64,
            "Low-level feature shape or dtype mismatch")
    with (paths["lowlevel_dir"] / "image_ids.json").open(encoding="utf-8") as handle:
        require(json.load(handle) == [r["image_id"] for r in records],
                "Low-level image identities do not match output_row order")
    return dict(paths=paths, sources=sources, categories=categories, by_key=by_key,
                concepts=concepts, embeddings=embeddings, visual=visual,
                category_names=table[0])


def read_brain(data, participant):
    """Validate participant image identities and ROI indexing metadata."""
    path = data["paths"]["brain_dir"] / f"brain_patterns_{participant}.npz"
    rois, counts = {}, {}
    with np.load(path, allow_pickle=False) as z:
        ids = vector(z["image_ids"], "US", "brain image_ids")
        require(len(ids) > 0 and len(set(ids)) == len(ids), "Duplicate or empty brain IDs")
        concepts = vector(z["concept_indices"], "iu", "brain concepts", len(ids))
        rows = np.array([data["by_key"][(participant, image)] for image in ids])
        require(np.array_equal(concepts, data["concepts"][rows]),
                "Brain and development-row concept identities differ")
        for key in ("n_valid_primary", "n_valid_evc"):
            value = np.asarray(z[key])
            require(value.dtype.kind in "iu" and value.shape in ((), (len(ids),))
                    and np.all(value >= 0), f"{key}: invalid count shape or value")
            counts[key] = value.tolist()
        for roi in ROIS:
            rr = vector(z[f"{roi}__rows"], "iu", f"{roi} rows")
            vox = np.asarray(z[f"{roi}__voxels"])
            require(len(rr) > 0 and len(np.unique(rr)) == len(rr)
                    and np.all((rr >= 0) & (rr < len(ids))), f"{roi}: invalid image rows")
            require(vox.dtype.kind in "iu" and vox.ndim in (0, 1)
                    and np.all(vox >= 0), f"{roi}: invalid voxel metadata")
            require(vox.ndim == 0 or len(np.unique(vox)) == len(vox),
                    f"{roi}: duplicate voxel identities")
            nvox = int(vox) if vox.ndim == 0 else len(vox)
            require(nvox > 0, f"{roi}: no voxels")
            rois[roi] = {"rows": rr, "n_voxels": nvox}
    return dict(path=path, concepts=concepts, global_rows=rows, rois=rois,
                counts={"selected_images": len(ids), **counts})


def brain_means(brain, roi, local_rows, labels, concepts):
    """Load one pattern matrix at a time and aggregate selected concept images."""
    result = []
    expected = (len(brain["rois"][roi]["rows"]), brain["rois"][roi]["n_voxels"])
    with np.load(brain["path"], allow_pickle=False) as z:
        for half in ("A", "B", "full"):
            x = z[f"{roi}__{half}"]
            require(x.shape == expected and x.dtype == np.float32,
                    f"{roi}/{half}: pattern shape or dtype mismatch")
            require(np.isfinite(x).all(), f"{roi}/{half}: nonfinite patterns")
            result.append(means(x[local_rows], labels, concepts))
            del x
    return result


def feature_kernels(data, rows, concepts):
    """Build model and unscaled visual kernels from exactly the supplied images."""
    labels, targets = data["concepts"][rows], {}
    for model, features in data["embeddings"].items():
        x = means(features[rows], labels, concepts, normalize=True)
        targets[model] = center(x @ x.T)
    visual = means(data["visual"][rows], labels, concepts)
    return targets, center(visual @ visual.T)


class Permutations:
    """Cache one seeded random-order bank per statistic and share its draws."""

    def __init__(self, seed, task, source, n_perm, concepts, categories, counts):
        self.seed, self.task, self.source, self.n_perm = seed, task, source, n_perm
        self.concepts, self.categories, self.counts = concepts, categories, counts
        self.ranks, self.cache = {}, {}

    def get(self, statistic, concepts):
        """Derive uniform group permutations, including for smaller EVC sets."""
        key = (statistic, tuple(concepts))
        if key in self.cache:
            return self.cache[key]
        if statistic not in self.ranks:
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.seed, self.task, self.source, STAT_INDEX[statistic]]))
            orders = np.stack([rng.permutation(len(self.concepts))
                               for _ in range(self.n_perm)])
            self.ranks[statistic] = np.argsort(orders, axis=1).astype(np.int32)
        loc = np.searchsorted(self.concepts, concepts)
        require(np.all(loc < len(self.concepts)), "Permutation universe mismatch")
        require(np.array_equal(self.concepts[loc], concepts), "Permutation identity mismatch")
        if statistic in ("taxonomy_residual", "taxonomy_visual_residual"):
            groups = {}
            for i, row in enumerate(self.categories[loc]):
                groups.setdefault(tuple(np.flatnonzero(row)), []).append(i)
            groups = [np.asarray(g) for g in groups.values()]
        elif statistic == "frequency_preserving":
            groups = np.array_split(
                np.argsort(np.log(self.counts[loc]), kind="stable"), 10)
        else:
            groups = [np.arange(len(concepts))]
        result = np.broadcast_to(np.arange(len(concepts), dtype=np.int32),
                                 (self.n_perm, len(concepts))).copy()
        for group in groups:
            if len(group):
                order = np.argsort(self.ranks[statistic][:, loc[group]], axis=1)
                result[:, group] = group[order]
        self.cache[key] = result
        return result


def report(observed, null):
    """Summarize a statistic and its own finite-sample permutation distribution."""
    null = np.asarray(null, dtype=np.float64)
    require(null.ndim == 1 and len(null) > 0 and np.isfinite(null).all()
            and np.isfinite(observed), "Invalid statistic or null")
    mean = float(null.mean())
    return dict(omega=float(observed), null_mean=mean,
                null_p95=float(np.quantile(null, 0.95)), delta=float(observed - mean),
                p=float((1 + np.count_nonzero(null >= observed)) / (1 + len(null))),
                n_perm=len(null))


def compare(u, v, permutations):
    """Compute overlap and a row-permutation null."""
    null = np.fromiter((overlap(u[p], v) for p in permutations),
                       dtype=np.float64, count=len(permutations))
    return report(overlap(u, v), null), null


def statistic_suite(operator, targets, visual, categories, concepts, bank):
    """Evaluate five statistics, sharing each recomputed operator across targets."""
    matrices = {"operator": operator, **targets}
    base = {name: top(k, 32) for name, k in matrices.items()}
    h = np.column_stack((np.ones(len(concepts)), categories))
    p = np.eye(len(concepts)) - h @ np.linalg.pinv(h)
    w = top(visual, 16)
    u, singular, _ = np.linalg.svd(np.column_stack((h, w)), full_matrices=False)
    combined = u[:, singular > 1e-10]
    results, draws = ({name: {} for name in targets} for _ in range(2))
    for statistic in STATS:
        basis = None
        if statistic in ("primary", "frequency_preserving"):
            eigenspaces = {name: value[:, :8] for name, value in base.items()}
        elif statistic == "taxonomy_residual":
            eigenspaces = {name: top(p @ k @ p, 8) for name, k in matrices.items()}
        else:
            basis = w if statistic == "visual_residual" else combined
            eigenspaces = {name: top(residual(k, basis), 8)
                           for name, k in matrices.items()}
        permutations = bank.get(statistic, concepts)
        if basis is None:
            for name in targets:
                results[name][statistic], draws[name][statistic] = compare(
                    eigenspaces["operator"], eigenspaces[name], permutations)
        else:
            nulls = {name: np.empty(len(permutations)) for name in targets}
            for draw, permutation in enumerate(permutations):
                permuted = operator[np.ix_(permutation, permutation)]
                permuted_u = top(residual(permuted, basis), 8, partial=True)
                for name in targets:
                    nulls[name][draw] = overlap(permuted_u, eigenspaces[name])
            for name in targets:
                draws[name][statistic] = nulls[name]
                results[name][statistic] = report(
                    overlap(eigenspaces["operator"], eigenspaces[name]), nulls[name])
    return results, draws, base


def sensitivities(operator_u, brain_u, full, direct, concepts, bank):
    """Evaluate separate q, full-mean, direct and band permutation nulls."""
    results = {}
    for statistic, (start, stop) in {"q4": (0, 4), "q16": (0, 16), **BANDS}.items():
        results[statistic] = compare(operator_u[:, start:stop], brain_u[:, start:stop],
                                     bank.get(statistic, concepts))[0]
    results["full_mean"] = compare(operator_u[:, :8], top(full, 8),
                                   bank.get("full_mean", concepts))[0]
    if direct is not None:
        results["vg_direct"] = compare(top(direct, 8), brain_u[:, :8],
                                       bank.get("vg_direct", concepts))[0]
    return results


def participant_task(data, args):
    """Analyze each ROI and form participant summaries from synchronized nulls."""
    brain, output = read_brain(data, args.task), {}
    for source_index, source in enumerate(SOURCES):
        started, src = time.perf_counter(), data["sources"][source]
        common = src["concepts"]
        for roi in PRIMARY_ROIS:
            common = np.intersect1d(common, brain["concepts"][brain["rois"][roi]["rows"]])
        require(len(common) >= 32, f"{source}: fewer than 32 common primary concepts")
        common_ix = np.searchsorted(src["concepts"], common)
        bank = Permutations(args.seed, TASKS.index(args.task), source_index, args.n_perm,
                            common, data["categories"][common], src["counts"][common_ix])
        rois, primary_nulls = {}, {}
        for roi in ROIS:
            rr = brain["rois"][roi]["rows"]
            concepts = common if roi in PRIMARY_ROIS else np.intersect1d(
                common, brain["concepts"][rr])
            require(len(concepts) >= 32, f"{source}/{roi}: fewer than 32 concepts")
            local = np.flatnonzero(np.isin(brain["concepts"][rr], concepts))
            selected = rr[local]
            labels, rows = brain["concepts"][selected], brain["global_rows"][selected]
            a, b, f = brain_means(brain, roi, local, labels, concepts)
            targets, visual = feature_kernels(data, rows, concepts)
            targets["brain"] = center((a @ b.T + b @ a.T) * 0.5)
            ix = np.searchsorted(src["concepts"], concepts)
            operator = center(src["kernel"][np.ix_(ix, ix)])
            results, nulls, eigenspaces = statistic_suite(
                operator, targets, visual, data["categories"][concepts], concepts, bank)
            direct = center(src["direct"][np.ix_(ix, ix)]) if source == "vg" else None
            brain_result = dict(
                statistics=results["brain"],
                sensitivities=sensitivities(eigenspaces["operator"], eigenspaces["brain"],
                                           center(f @ f.T), direct, concepts, bank),
                self_consistency=dict(q=8, omega=overlap(
                    top(center(a @ a.T), 8), top(center(b @ b.T), 8))))
            rois[roi] = dict(n_concepts=len(concepts), concept_indices=concepts,
                             n_images=len(rows), n_voxels=brain["rois"][roi]["n_voxels"],
                             image_counts=np.bincount(np.searchsorted(concepts, labels),
                                                      minlength=len(concepts)),
                             operator_counts=src["counts"][ix], brain=brain_result,
                             models={model: results[model] for model in MODELS})
            if roi in PRIMARY_ROIS:
                primary_nulls[roi] = nulls["brain"]
        summary = {}
        for statistic in STATS:
            observed = np.mean([rois[r]["brain"]["statistics"][statistic]["omega"]
                                for r in PRIMARY_ROIS])
            null = np.mean([primary_nulls[r][statistic] for r in PRIMARY_ROIS], axis=0)
            summary[statistic] = report(observed, null)
        output[source] = dict(n_concepts=len(common), concept_indices=common,
                              rois=rois, summary=summary,
                              seconds=time.perf_counter() - started)
    return output, brain["counts"]


def pooled_task(data, args):
    """Analyze pooled model kernels and all six descriptive model pairs."""
    output = {}
    for source_index, source in enumerate(SOURCES):
        started, src = time.perf_counter(), data["sources"][source]
        concepts = src["concepts"]
        rows = np.flatnonzero(np.isin(data["concepts"], concepts))
        targets, visual = feature_kernels(data, rows, concepts)
        bank = Permutations(args.seed, 5, source_index, args.n_perm, concepts,
                            data["categories"][concepts], src["counts"])
        results, _, eigenspaces = statistic_suite(
            center(src["kernel"]), targets, visual, data["categories"][concepts],
            concepts, bank)
        pairs = {}
        for left, right in itertools.combinations(MODELS, 2):
            pairs[f"{left}__{right}"] = {
                statistic.removeprefix("pair_"): compare(
                    eigenspaces[left][:, start:stop], eigenspaces[right][:, start:stop],
                    bank.get(statistic, concepts))[0]
                for statistic, (start, stop) in PAIR_SLICES.items()}
        output[source] = dict(
            n_concepts=len(concepts), concept_indices=concepts, n_images=len(rows),
            image_counts=np.bincount(np.searchsorted(concepts, data["concepts"][rows]),
                                     minlength=len(concepts)),
            operator_counts=src["counts"], models=results, model_pairs=pairs,
            seconds=time.perf_counter() - started)
    return output, {"development_images": N_IMAGES}


def json_safe(value):
    """Convert NumPy values and replace nonfinite numbers with JSON null."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, data):
    """Atomically write strict JSON to the designated output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(data), handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(directory, out):
    """Merge six compatible task files and compute participant sign tests."""
    started, documents = time.perf_counter(), {}
    require(directory.is_dir(), f"Missing collection directory: {directory}")
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        if not isinstance(item, dict) or item.get("task") not in TASKS:
            continue
        require(item.get("schema") == SCHEMA, f"Incompatible schema: {path}")
        require(path.resolve() != out, "Collection output cannot replace a task file")
        require(item["task"] not in documents, f"Duplicate task: {item['task']}")
        documents[item["task"]] = item
    require(set(documents) == set(TASKS), f"Missing tasks: {set(TASKS) - set(documents)}")
    reference = documents[PARTICIPANTS[0]]
    for item in documents.values():
        for field in ("settings", "input_paths"):
            require(item[field] == reference[field], f"Task mismatch in {field}")
        require(set(item["sources"]) == set(SOURCES), "Task source mismatch")
    groups, lines = {}, ["source    statistic                    k/5  sign_p   p<.05"]
    for source in SOURCES:
        groups[source] = {}
        for statistic in STATS:
            values = {p: documents[p]["sources"][source]["summary"][statistic]
                      for p in PARTICIPANTS}
            for value in values.values():
                require(np.isfinite([value["omega"], value["null_mean"],
                                     value["delta"], value["p"]]).all()
                        and 0 <= value["p"] <= 1
                        and math.isclose(value["delta"], value["omega"] - value["null_mean"],
                                         rel_tol=1e-10, abs_tol=1e-12),
                        "Invalid participant summary")
            k = sum(v["delta"] > 0 for v in values.values())
            sign_p = sum(math.comb(5, j) for j in range(k, 6)) / 32
            significant = sum(v["p"] < 0.05 for v in values.values())
            groups[source][statistic] = dict(
                participants={p: {"delta": v["delta"], "p": v["p"]}
                              for p, v in values.items()},
                k=k, sign_test_p=sign_p, n_p_lt_0_05=significant)
            lines.append(f"{source:<9} {statistic:<28} {k}/5  {sign_p:7.5f}  {significant}")
    write_json(out, dict(schema=SCHEMA, mode="collect", tasks=documents, group_results=groups,
                         settings=reference["settings"], input_paths=reference["input_paths"],
                         counts={"tasks": 6, "participants": 5},
                         timings={"seconds": time.perf_counter() - started}))
    print("\n".join(lines))


def main():
    """Parse task or collection mode and write one complete result."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", choices=TASKS)
    mode.add_argument("--collect", type=Path)
    for key in PATH_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.collect is not None:
        collect(args.collect.resolve(), args.out)
        return
    if any(getattr(args, key) is None for key in PATH_KEYS):
        parser.error("Task mode requires all seven input-path arguments")
    if args.n_perm < 1 or args.seed < 0:
        parser.error("--n-perm must be positive and --seed must be nonnegative")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    data = read_inputs(args)
    loaded = time.perf_counter()
    sources, counts = (pooled_task(data, args) if args.task == "pooled"
                       else participant_task(data, args))
    settings = dict(
        seed=args.seed, n_perm=args.n_perm, q=8, visual_q=16, bands=BANDS,
        combined_svd_absolute_tolerance=1e-10, statistic_indices=STAT_INDEX,
        participants=PARTICIPANTS, primary_rois=PRIMARY_ROIS,
        descriptive_roi=ROIS[-1], descriptive_models=MODELS,
        category_names=data["category_names"], frequency_bins=10,
        frequency_ties="stable concept order", direct_zero_degree_inverse=0,
        uid_authority="operators.npz/things_uids; things_meta is opaque provenance",
        permutation_coupling="shared random orders; EVC restricted within its own groups")
    write_json(args.out, dict(
        schema=SCHEMA, task=args.task, task_index=TASKS.index(args.task),
        settings=settings, input_paths=data["paths"],
        counts=counts, sources=sources,
        timings=dict(started_utc=started_utc, input_seconds=loaded - started,
                     analysis_seconds=time.perf_counter() - loaded,
                     total_seconds_before_write=time.perf_counter() - started),
        thread_environment={k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "SLURM_CPUS_PER_TASK")}))


if __name__ == "__main__":
    main()
