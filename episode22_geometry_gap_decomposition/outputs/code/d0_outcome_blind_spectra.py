#!/usr/bin/env python3
"""EP22 D0: outcome-blind embedding spectra (no neural data).

Reads only model embeddings and stimulus identity metadata:
  * EP04's frozen development embedding handoff (read-only reuse),
  * the provider stimulus metadata CSV (image names and roles),
  * the original (2026-05-12) and corrected (2026-09-28) provider embedding
    HDF5 files, to bind the transparency-corrected OOD rows.
No beta, ROI, reliability, RDM-to-brain or RSA value is opened.

Outputs (in --out-dir):
  ood_embedding_binding.json  row-level comparison of original vs corrected files
  ood_category_table.csv      provider OOD type and proposed subset per image
  d0_spectra_summary.json     per-stratum ED / decay / anisotropy / CKA / overlap
  d0_spectra_subsets.csv      per-subset values for plotting
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

MODELS = ("DINOv2", "CLIP", "PEcore", "SigLIP2")
REFERENCE = "DINOv2"
VLMS = tuple(m for m in MODELS if m != REFERENCE)
DEV_STRATA = ("laion", "things", "thingsplus")
OOD_TYPES = ("cropped", "gabor", "gaudy", "illusion-classic", "illusion-natural",
             "relations", "selfmade", "shape", "unusual")
PHOTOGRAPHIC = ("cropped", "illusion-natural", "relations", "selfmade", "unusual")
SYNTHETIC = ("gabor", "gaudy", "illusion-classic", "shape")
OOD_NAME = re.compile(r"^shared_4rep_OOD_(" + "|".join(re.escape(t) for t in OOD_TYPES) + r")_")
H5_NAME = "task-images_desc-{model}_embeddings.h5"
OVERLAP_K = (10, 32)
DECAY_RANKS = (2, 32)  # 1-indexed inclusive rank window for the power-law fit


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_h5(path: Path):
    with h5py.File(path, "r") as handle:
        embedding = handle["embedding"][...]
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["image_ids"][...]]
        valid = handle["valid"][...].astype(bool)
        attrs = {k: jsonable(v) for k, v in handle.attrs.items()}
    return embedding, ids, valid, attrs


def row_l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    if not np.all(np.isfinite(x)) or np.any(norms == 0):
        raise ValueError("non-finite or zero-norm embedding row")
    return x / norms[:, None]


# --------------------------------------------------------------------------
# Binding of the corrected OOD rows
# --------------------------------------------------------------------------

def bind_corrected(original_dir: Path, corrected_dir: Path, ood_names: dict[str, str],
                   dev_rows: list[dict], dev_arrays: dict[str, np.ndarray]):
    report = {"original_dir": str(original_dir), "corrected_dir": str(corrected_dir), "models": {}}
    download_record = corrected_dir / "download_record.json"
    report["download_record"] = json.loads(download_record.read_text()) if download_record.is_file() else None
    corrected_ood = {}
    changed_sets: dict[str, set[str]] = {}
    ok = True
    for model in MODELS:
        orig_e, orig_ids, orig_valid, orig_attrs = read_h5(original_dir / H5_NAME.format(model=model))
        corr_e, corr_ids, corr_valid, corr_attrs = read_h5(corrected_dir / H5_NAME.format(model=model))
        entry = {
            "shape_original": list(orig_e.shape), "shape_corrected": list(corr_e.shape),
            "dtype": str(corr_e.dtype),
            "attrs_original": orig_attrs, "attrs_corrected": corr_attrs,
            "image_ids_identical": orig_ids == corr_ids,
            "valid_identical": bool(np.array_equal(orig_valid, corr_valid)),
        }
        if orig_e.shape != corr_e.shape or orig_e.dtype != corr_e.dtype or not entry["image_ids_identical"]:
            entry["error"] = "layout differs between original and corrected files"
            report["models"][model] = entry
            ok = False
            continue
        bits = orig_e.view(np.uint16 if orig_e.dtype == np.float16 else np.uint8)
        changed = np.any(bits != corr_e.view(bits.dtype), axis=1)
        changed_ids = [corr_ids[i] for i in np.flatnonzero(changed)]
        index = {name: i for i, name in enumerate(corr_ids)}
        missing_ood = sorted(n for n in ood_names if n not in index)
        ood_rows = np.array([index[n] for n in ood_names if n in index], dtype=int)
        is_ood = np.zeros(len(corr_ids), dtype=bool)
        is_ood[ood_rows] = True
        entry.update({
            "rows_changed": int(changed.sum()),
            "changed_rows_all_ood": bool(np.all(is_ood[changed])),
            "ood_rows_changed": int(changed[is_ood].sum()),
            "ood_rows_unchanged": int((~changed[is_ood]).sum()),
            "non_ood_rows_identical": bool(not np.any(changed[~is_ood])),
            "changed_by_ood_type": dict(Counter(ood_names[n] for n in changed_ids if n in ood_names)),
            "missing_ood_ids": missing_ood,
            "ood_rows_valid": int(corr_valid[ood_rows].sum()),
        })
        # EP22's development rows come from EP04's handoff; check they match the current release.
        source_rows = np.array([r["source_rows"][model] for r in dev_rows], dtype=int)
        handoff = dev_arrays[model]
        entry["development_handoff_rows_identical_to_corrected_release"] = bool(
            np.array_equal(handoff.view(np.uint16), corr_e[source_rows].view(np.uint16)))
        ok = ok and entry["changed_rows_all_ood"] and entry["non_ood_rows_identical"] \
            and not missing_ood and entry["ood_rows_valid"] == len(ood_names)
        changed_sets[model] = set(changed_ids)
        corrected_ood[model] = {n: corr_e[index[n]] for n in ood_names if n in index}
        report["models"][model] = entry
    changed_union = set().union(*changed_sets.values()) if changed_sets else set()
    report["changed_ood_ids_union"] = sorted(changed_union)
    report["changed_ood_count_union"] = len(changed_union)
    report["changed_set_identical_across_models"] = (
        len(changed_sets) == len(MODELS) and all(s == changed_union for s in changed_sets.values()))
    report["provider_statement_116_ood_rows"] = len(changed_union) == 116
    report["binding_ok"] = bool(ok and len(changed_sets) == len(MODELS))
    return report, corrected_ood, changed_union


# --------------------------------------------------------------------------
# Spectral measures
# --------------------------------------------------------------------------

def subset_measures(z_by_model: dict[str, np.ndarray]):
    """Centered Gram eigensystems and per-model / per-pair measures for one subset."""
    n = next(iter(z_by_model.values())).shape[0]
    grams, vecs, model_values = {}, {}, {}
    lo, hi = DECAY_RANKS
    ranks = np.arange(lo, hi + 1)
    for model, z in z_by_model.items():
        zc = z - z.mean(axis=0, keepdims=True)
        gram = zc @ zc.T
        w, v = np.linalg.eigh(gram)
        order = np.argsort(w)[::-1]
        w = np.clip(w[order], 0.0, None)
        v = v[:, order]
        total = w.sum()
        slope = np.polyfit(np.log(ranks), np.log(w[ranks - 1]), 1)[0]
        cosd = 1.0 - z @ z.T
        np.fill_diagonal(cosd, np.inf)
        nn = cosd.min(axis=1)
        model_values[model] = {
            "ED": float(total ** 2 / np.sum(w ** 2)),
            "top1": float(w[0] / total),
            "decay": float(-slope),
            "nn_cv": float(nn.std() / nn.mean()),
        }
        grams[model] = gram
        vecs[model] = v
    pair_values = {}
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            ga, gb = grams[a], grams[b]
            values = {"CKA": float(np.sum(ga * gb) / np.sqrt(np.sum(ga * ga) * np.sum(gb * gb)))}
            for k in OVERLAP_K:
                m = vecs[a][:, :k].T @ vecs[b][:, :k]
                values[f"overlap_k{k}"] = float(np.sum(m * m) / k)
            pair_values[f"{a}|{b}"] = values
    return n, model_values, pair_values


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "q025": float(np.quantile(arr, 0.025)), "q975": float(np.quantile(arr, 0.975))}


def run_stratum(key: str, rows: np.ndarray, data: dict[str, np.ndarray], n_sub: int,
                repeats: int, seed_words: list[int], subset_rows_out: list):
    if rows.size < n_sub:
        return {"n_available": int(rows.size), "n_sub": int(n_sub), "skipped": "fewer rows than n_sub"}
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    per_model = defaultdict(lambda: defaultdict(list))
    per_pair = defaultdict(lambda: defaultdict(list))
    log_ratio = defaultdict(list)
    for rep in range(repeats):
        pick = rng.choice(rows, size=n_sub, replace=False)
        _, mv, pv = subset_measures({m: data[m][pick] for m in MODELS})
        for m, vals in mv.items():
            for name, val in vals.items():
                per_model[m][name].append(val)
        for p, vals in pv.items():
            for name, val in vals.items():
                per_pair[p][name].append(val)
        for m in VLMS:
            log_ratio[m].append(float(np.log(mv[m]["ED"] / mv[REFERENCE]["ED"])))
        for m in MODELS:
            subset_rows_out.append({"stratum": key, "subset": rep, "model": m, **mv[m],
                                    "log_ED_ratio_vs_DINOv2": (float(np.log(mv[m]["ED"] / mv[REFERENCE]["ED"]))
                                                               if m != REFERENCE else 0.0)})
    return {
        "n_available": int(rows.size), "n_sub": int(n_sub), "repeats": int(repeats),
        "models": {m: {name: summarize(v) for name, v in d.items()} for m, d in per_model.items()},
        "pairs": {p: {name: summarize(v) for name, v in d.items()} for p, d in per_pair.items()},
        "log_ED_ratio_vs_DINOv2": {m: summarize(v) for m, v in log_ratio.items()},
        "overlap_chance": {f"k{k}": k / (n_sub - 1) for k in OVERLAP_K},
    }


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev-embeddings", type=Path, required=True)
    ap.add_argument("--stimulus-metadata", type=Path, required=True)
    ap.add_argument("--original-embeddings", type=Path, required=True)
    ap.add_argument("--corrected-embeddings", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-sub", type=int, default=128)
    ap.add_argument("--n-sub-large", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    report = {
        "analysis_id": "d0_outcome_blind_spectra", "started_utc": utc_now(),
        "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__,
        "neural_values_read": False,
        "settings": {"n_sub": a.n_sub, "n_sub_large": a.n_sub_large, "repeats": a.repeats,
                     "seed": a.seed, "overlap_k": OVERLAP_K, "decay_ranks": DECAY_RANKS,
                     "row_normalization": "l2_float64", "centering": "column_mean_within_subset"},
        "errors": [],
    }

    # Development handoff (EP04, read-only).
    dev_rows = [json.loads(line) for line in (a.dev_embeddings / "development_rows.jsonl").open()]
    dev_raw = {m: np.load(a.dev_embeddings / f"{m}.npy") for m in MODELS}
    for m in MODELS:
        if dev_raw[m].shape[0] != len(dev_rows):
            raise SystemExit(f"{m}: handoff rows {dev_raw[m].shape[0]} != identity rows {len(dev_rows)}")
    dev = {m: row_l2(dev_raw[m]) for m in MODELS}
    participants = sorted({r["participant"] for r in dev_rows})
    cells = Counter((r["participant"], r["source_stratum"]) for r in dev_rows)
    report["development_counts"] = {f"{p}|{s}": cells[(p, s)] for p in participants for s in DEV_STRATA}

    # OOD identities and provider types (names only).
    meta = list(csv.DictReader(a.stimulus_metadata.open()))
    ood_names = {}
    for r in meta:
        if r["dataset"] == "OOD":
            match = OOD_NAME.match(r["image_name"])
            if not match:
                report["errors"].append(f"unparsed OOD name {r['image_name']}")
                continue
            ood_names[r["image_name"]] = match.group(1)

    # Binding of corrected OOD rows.
    corrected_ood, changed = None, set()
    if all((a.corrected_embeddings / H5_NAME.format(model=m)).is_file() for m in MODELS):
        binding, corrected_ood, changed = bind_corrected(
            a.original_embeddings, a.corrected_embeddings, ood_names, dev_rows, dev_raw)
    else:
        binding = {"binding_ok": False, "error": "corrected embedding files absent"}
    (a.out_dir / "ood_embedding_binding.json").write_text(json.dumps(binding, indent=1))
    report["binding_ok"] = bool(binding.get("binding_ok"))

    with (a.out_dir / "ood_category_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "ood_type", "proposed_subset", "file_extension", "transparency_corrected"])
        for name in sorted(ood_names):
            t = ood_names[name]
            writer.writerow([name, t, "photographic" if t in PHOTOGRAPHIC else "synthetic",
                             name.rsplit(".", 1)[-1].lower(), name in changed])
    report["ood_type_counts"] = dict(Counter(ood_names.values()))

    # Spectra.
    subset_rows: list[dict] = []
    strata = {}
    stratum_index = 0
    p_index = {p: i for i, p in enumerate(participants)}
    for s_i, s in enumerate(DEV_STRATA):
        for p in participants:
            rows = np.array([i for i, r in enumerate(dev_rows) if r["participant"] == p and r["source_stratum"] == s])
            key = f"dev|{p}|{s}|n{a.n_sub}"
            strata[key] = run_stratum(key, rows, dev, a.n_sub, a.repeats, [a.seed, 1, s_i, p_index[p], a.n_sub], subset_rows)
        rows = np.array([i for i, r in enumerate(dev_rows) if r["source_stratum"] == s])
        for n_sub in (a.n_sub, a.n_sub_large):
            key = f"dev|pooled|{s}|n{n_sub}"
            strata[key] = run_stratum(key, rows, dev, n_sub, a.repeats, [a.seed, 2, s_i, n_sub], subset_rows)
    if corrected_ood is not None and report["binding_ok"]:
        ood_order = sorted(ood_names)
        ood = {m: row_l2(np.stack([corrected_ood[m][n] for n in ood_order])) for m in MODELS}
        groups = {"photographic": PHOTOGRAPHIC, "synthetic": SYNTHETIC, "all": OOD_TYPES}
        for g_i, (g, types) in enumerate(groups.items()):
            rows = np.array([i for i, n in enumerate(ood_order) if ood_names[n] in types])
            key = f"ood|{g}|n{a.n_sub}"
            strata[key] = run_stratum(key, rows, ood, a.n_sub, a.repeats, [a.seed, 3, g_i, a.n_sub], subset_rows)
    else:
        report["errors"].append("OOD strata skipped: corrected binding not established")
    report["strata"] = strata

    # Selection signature: relative ED on LAION minus other strata (positive = flatter VLM spectrum on LAION).
    def mean_ratio(key, m):
        return strata[key]["log_ED_ratio_vs_DINOv2"][m]["mean"] if "skipped" not in strata[key] else float("nan")
    contrasts = {}
    for m in VLMS:
        c = {}
        for other in ("things", "thingsplus"):
            per_p = {p: mean_ratio(f"dev|{p}|laion|n{a.n_sub}", m) - mean_ratio(f"dev|{p}|{other}|n{a.n_sub}", m)
                     for p in participants}
            c[f"laion_minus_{other}_per_participant"] = per_p
            c[f"laion_minus_{other}_positive_participants"] = int(sum(v > 0 for v in per_p.values()))
            for n_sub in (a.n_sub, a.n_sub_large):
                c[f"laion_minus_{other}_pooled_n{n_sub}"] = (mean_ratio(f"dev|pooled|laion|n{n_sub}", m)
                                                             - mean_ratio(f"dev|pooled|{other}|n{n_sub}", m))
        for g in ("photographic", "synthetic"):
            key = f"ood|{g}|n{a.n_sub}"
            if key in strata:
                c[f"laion_minus_ood_{g}_pooled_n{a.n_sub}"] = (mean_ratio(f"dev|pooled|laion|n{a.n_sub}", m)
                                                               - mean_ratio(key, m))
        contrasts[m] = c
    report["selection_signature_log_ED_ratio_contrasts"] = contrasts

    with (a.out_dir / "d0_spectra_subsets.csv").open("w", newline="") as handle:
        fields = ["stratum", "subset", "model", "ED", "top1", "decay", "nn_cv", "log_ED_ratio_vs_DINOv2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subset_rows)
    report["elapsed_seconds"] = time.time() - t0
    report["finished_utc"] = utc_now()
    (a.out_dir / "d0_spectra_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"binding_ok": report["binding_ok"], "errors": report["errors"],
                      "strata": len(strata), "elapsed_seconds": round(report["elapsed_seconds"], 1)}))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
