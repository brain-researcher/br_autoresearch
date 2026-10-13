#!/usr/bin/env python3
"""EP22 locked OOD confirmation evaluator.

Modes
  dev-test  Full pipeline on pseudo-OOD development images; no OOD value is read.
            Verifies the trusted reader bit-for-bit against EP04's development
            handoff on two sessions, then runs fold-free standardization,
            patterns, RDMs, C1-C3 and the gate on pseudo categories sized like
            the OOD categories.
  confirm   The single procedurally sealed OOD opening. Requires the write-once
            lock whose code checksums and settings match this file, refuses to
            run if a confirmation report exists, keeps OOD values in memory
            only and releases nothing until the report is complete.

Neural reading, fold-free session standardization (EP04 `training_statistics`
with no held-out fold), presentation validity, repeat counts and mean patterns
reuse EP04's production functions read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EP22 = Path(__file__).resolve().parents[2]
EP04 = EP22.parent / "episode04_shared_scene_geometry"
EP04_CODE = EP04 / "outputs/code"
SOURCE_ROOT = Path("/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922")
METADATA = Path("outputs/runs/source_readiness/20261003-metadata_folds04")             # relative to EP04
ARRAYS = Path("outputs/runs/source_readiness/20261003-array_metadata02/array_metadata.json")
HANDOFF = Path("outputs/runs/source_readiness/20261003-development_neural_full01")
DEV_EMBEDDINGS = EP04 / "outputs/runs/source_readiness/20261003-development_embeddings01"
CORRECTED = EP22 / "inputs/laion_ood_embeddings_corrected"
CATEGORY_TABLE = EP22 / "outputs/runs/d0_outcome_blind_spectra/20261012-attempt01/results/ood_category_table.csv"
D2_CELLS = EP22 / "outputs/runs/d123_development/20261012-attempt02/cells"
EXPOSURE_REVIEW = EP22 / "outputs/runs/ood_exposure_review/20261012-review01/exposure_review.json"
LOCK = EP22 / "outputs/ood_confirmation_lock.json"
CODE_FILES = [Path(__file__).resolve(), EP04_CODE / "preprocess_development_neural.py",
              EP04_CODE / "prepare_development_neural_pilot.py", EP04_CODE / "prepare_development_neural_participant.py",
              EP04_CODE / "inspect_array_metadata.py"]

SETTINGS = {
    "participants": ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"],
    "primary_rois": ["laion-ventral", "laion-lateral", "laion-dorsal"],
    "descriptive_roi": "retinotopic-EVC",
    "models": ["DINOv2", "CLIP", "PEcore", "SigLIP2"],
    "reference": "DINOv2",
    "confirmatory_categories": ["cropped", "relations", "selfmade", "unusual"],
    "descriptive_categories": ["illusion-natural", "gabor", "gaudy", "illusion-classic", "shape"],
    "margin_fisher_z": 0.02,
    "participant_rule_min": 4,
    "permutations": 10000,
    "permutation_seed": 20261012,
    "null_quantile": 0.95,
    "null_quantile_method": "higher",
    "null_exceedance": "strictly_greater",
    "standardization": "per session: mean and population SD over all development trials of that session (EP04 training_statistics with no held-out fold)",
    "voxel_rule": "in fixed ROI mask, finite on every development presentation, positive SD in every session; >=3 voxels per ROI",
    "repeat_rule": "image eligible with >=2 valid primary presentations (finite on all retained primary-sector voxels); EVC also needs >=2 valid EVC presentations",
    "neural_distance": "correlation",
    "native_model_distance": "cosine on float64 L2-normalized rows",
    "spectral_geometry": "per participant x category on eligible images: SVD of column-centred L2 rows; squared Euclidean on U_a * s_b (rank order)",
    "c3_transform": "per participant: PCA of all 4,712 development embeddings (L2 rows, centred), leading 256 axes; Phi=(X-mu) V diag(s^(alpha-1)), alpha = median of the participant's five D2 R1 alpha-hat values per model",
    "rsa": "Spearman (average ranks) of upper triangles, Fisher z; equal weights over categories and primary sectors, then participants",
    "c1": "CLIP-DINOv2 native gap: spectral if mean <= -margin and >=4/5 participants <= -margin; selection_only if mean > -margin and >=4/5 participants > -margin; else indeterminate. PEcore classified the same way (secondary). Leave-one-category-out means reported.",
    "c2a": "Within-category: for CLIP and SigLIP2, E_val < 0 for the mean and >=4/5 participants (E_val = 0.5[(RSA(m|m)-RSA(m|ref)) + (RSA(ref|m)-RSA(ref|ref))], rows = eigenvectors|eigenvalues).",
    "c2b": "All pairs among each participant's eligible confirmatory images: for CLIP and SigLIP2, evaluable if the mean native all-pairs gap <= -margin; pass if |all-pairs gap with DINOv2 eigenvalues on VLM eigenvectors| <= 0.5 |native all-pairs gap| for the mean and for >=4/5 participants (each with a negative native gap).",
    "c2b_retention": "C2b is confirmatory only if the lock records that the pre-lock dev-test met the same C2b rule for both CLIP and SigLIP2; otherwise C2b is descriptive.",
    "c2_pecore_secondary": "Within-category PEcore: E_val < 0 and E_vec > 0 for the mean and >=4/5 participants.",
    "c3": "For each VLM: mean (C3 transform minus native) > 0 and >=4/5 participants > 0.",
    "gate": "DINOv2 native confirmatory aggregate > 95th percentile ('higher') of 10,000 within-category whole-image permutations of the model RDM, each permutation shared across participants and sectors; failure makes the opening uninformative.",
    "embeddings": "provider 2026-09-28 corrected HDF5 files in inputs/laion_ood_embeddings_corrected (ETags in download_record.json)",
}
PSEUDO = {  # dev-test only: pseudo category -> (source stratum, size), mirroring OOD category sizes
    "cropped": ("thingsplus", 60), "relations": ("laion", 72), "selfmade": ("things", 30), "unusual": ("laion", 64),
    "illusion-natural": ("things", 8), "gabor": ("things", 10), "gaudy": ("thingsplus", 24),
    "illusion-classic": ("thingsplus", 21), "shape": ("laion", 82),
}
OOD_NAME = re.compile(r"^shared_4rep_OOD_")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_checksums():
    return {str(p): sha256(p) for p in CODE_FILES}


def ep04_modules():
    os.chdir(EP04)                      # EP04 records use paths relative to its episode directory
    sys.path.insert(0, str(EP04_CODE))
    import prepare_development_neural_pilot as pilot  # noqa: E402
    import preprocess_development_neural as pdn  # noqa: E402
    return pilot, pdn


# ----------------------------------------------------------------------------- geometry

def l2(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1)
    if not np.all(np.isfinite(x)) or np.any(n == 0):
        raise ValueError("non-finite or zero-norm row")
    return x / n[:, None]


def corr_rdm(p):
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
    d = np.maximum(q[:, None] + q[None, :] - 2.0 * g, 0.0)
    np.fill_diagonal(d, 0.0)
    return d


def upper(d):
    return d[np.triu_indices(d.shape[0], 1)]


def ranks(v):
    from scipy.stats import rankdata
    return rankdata(v, method="average")


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def fz(r):
    return math.atanh(r) if math.isfinite(r) and abs(r) < 1 else float("nan")


def centred_svd(x):
    u, s, _ = np.linalg.svd(x - x.mean(axis=0, keepdims=True), full_matrices=False)
    return u, s


# ----------------------------------------------------------------------------- neural

def ood_trial_rows(participant, summary, ood_index):
    """Metadata only: OOD rows per session with their original TSV row position."""
    per_session = {}
    for rec in sorted((r for r in summary["trial_files"] if r["participant"] == participant), key=lambda r: r["session"]):
        with open(SOURCE_ROOT / "dataset" / rec["trial_file"]) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if len(rows) != rec["rows"]:
            raise ValueError(f"{participant} {rec['session']}: TSV rows differ from frozen accounting")
        per_session[rec["session"]] = [
            {"session": r["session"], "run": r["run"], "beta_index": int(r["beta_index"]), "label": r["label"],
             "session_beta_axis_index": pos, "target_index": ood_index[r["label"]]}
            for pos, r in enumerate(rows) if r["label"] in ood_index]
    return per_session


def stream_rows(pilot, arrays, participant, session, rows, union_indices):
    import gzip
    rec = [b for b in arrays["betas"] if (b["participant"], b["session"]) == (participant, session)]
    if len(rec) != 1:
        raise ValueError("expected one beta record")
    rec = rec[0]
    rows = pilot.validate_session_rows(rows, rec["header"]["shape"][3])
    out = np.empty((len(rows), len(union_indices)), dtype=np.float32)
    progress = {k: 0 for k in ("source_volumes_consumed", "uncompressed_beta_payload_bytes_consumed",
                               "selected_volumes_decoded", "selected_roi_nonfinite_values", "selected_roi_zero_values",
                               "excluded_volumes_discarded_as_opaque_bytes")}
    with gzip.open(SOURCE_ROOT / "dataset" / rec["path"], "rb") as stream:
        header, dtype = pilot.open_header(stream, rec["header"])
        pilot.stream_selected(stream, header["shape"], dtype, rows, union_indices, out, progress)
    return rows, out, progress


def participant_patterns(task):
    """Worker: fold-free statistics from development trials; target patterns per ROI."""
    participant, mode, target_ids = task["participant"], task["mode"], task["target_ids"]
    pilot, pdn = ep04_modules()
    images, sessions, union, membership = pdn.load_completed_inputs(
        participant, HANDOFF / participant / "participant_handoff.json", METADATA)
    means, scales, keep, finite_all, positive_all, counts = pdn.training_statistics(sessions, held_fold=-1)
    if np.any(counts < 2):
        raise ValueError("a session has fewer than two development trials")
    support = membership & keep[:, None]
    voxels = {roi: int(support[:, i].sum()) for i, roi in enumerate(pdn.ROI_NAMES)}
    if min(voxels.values()) < 3:
        raise ValueError("an ROI has fewer than three retained voxels")
    index = {name: i for i, name in enumerate(target_ids)}
    target_sessions, reader = [], {}
    if mode == "confirm":
        summary = json.loads((METADATA / "source_metadata_report.json").read_text())
        arrays = json.loads(ARRAYS.read_text())
        rows_by_session = ood_trial_rows(participant, summary, index)
        for s in sessions:
            rows = rows_by_session[s.name]
            if rows:
                rows, betas, progress = stream_rows(pilot, arrays, participant, s.name, rows, union)
                reader[s.name] = {k: progress[k] for k in ("selected_volumes_decoded", "excluded_volumes_discarded_as_opaque_bytes")}
            else:
                betas = np.empty((0, len(union)), dtype=np.float32)
            target_sessions.append(pdn.Session(s.name, betas, np.array([r["target_index"] for r in rows], dtype=np.int64),
                                               np.zeros(len(rows), dtype=np.int8), rows))
    else:  # dev-test: pseudo targets are development trials taken from the handoff
        for s in sessions:
            pick = [k for k, r in enumerate(s.rows) if r["image_id"] in index]
            target_sessions.append(pdn.Session(s.name, np.asarray(s.betas[pick]),
                                               np.array([index[s.rows[k]["image_id"]] for k in pick], dtype=np.int64),
                                               np.zeros(len(pick), dtype=np.int8), [s.rows[k] for k in pick]))
        if task.get("verify_reader"):
            arrays = json.loads(ARRAYS.read_text())
            checks = {}
            for s in sessions:
                if s.name in task["verify_reader"]:
                    rows, betas, progress = stream_rows(pilot, arrays, participant, s.name, list(s.rows), union)
                    order = [r["session_beta_axis_index"] for r in rows] == [r["session_beta_axis_index"] for r in s.rows]
                    checks[s.name] = {"rows": len(rows), "row_order_identical": order,
                                      "bitwise_identical_to_EP04_handoff": bool(order and np.array_equal(
                                          betas.view(np.uint32), np.asarray(s.betas).view(np.uint32)))}
            reader["dev_reader_verification"] = checks
    primary_valid, evc_valid = pdn.presentation_validity(target_sessions, means, scales, support)
    n = len(target_ids)
    primary_counts = pdn.repeat_counts(target_sessions, primary_valid, n)
    evc_counts = pdn.repeat_counts(target_sessions, evc_valid, n)
    eligible, complete4, evc_eligible = pdn.image_eligibility(primary_counts, evc_counts)
    patterns = {}
    for i, roi in enumerate(pdn.ROI_NAMES):
        cols = np.flatnonzero(support[:, i])
        if i < 3:
            patterns[roi] = pdn.mean_pattern_block(target_sessions, means, scales, cols, primary_valid, primary_counts, eligible)
        else:
            patterns[roi] = pdn.mean_pattern_block(target_sessions, means, scales, cols, evc_valid, evc_counts, evc_eligible)
    presentations = int(sum(len(t.rows) for t in target_sessions))
    return {"participant": participant, "patterns": patterns, "eligible": eligible, "evc_eligible": evc_eligible,
            "primary_counts": primary_counts, "presentations": presentations, "voxels": voxels, "reader": reader,
            "dev_image_ids": [im["image_id"] for im in images]}


# ----------------------------------------------------------------------------- models

def dev_embedding_rows():
    rows = [json.loads(line) for line in open(DEV_EMBEDDINGS / "development_rows.jsonl")]
    return {r["image_id"]: r["output_row"] for r in rows}


def model_features(ids, mode):
    import h5py
    out = {}
    if mode == "confirm":
        for m in SETTINGS["models"]:
            with h5py.File(CORRECTED / f"task-images_desc-{m}_embeddings.h5", "r") as f:
                names = [x.decode() if isinstance(x, bytes) else str(x) for x in f["image_ids"][...]]
                pos = {n: i for i, n in enumerate(names)}
                emb = f["embedding"][...]
            out[m] = l2(emb[[pos[i] for i in ids]])
    else:
        rows = dev_embedding_rows()
        for m in SETTINGS["models"]:
            arr = np.load(DEV_EMBEDDINGS / f"{m}.npy", mmap_mode="r")
            out[m] = l2(arr[[rows[i] for i in ids]])
    return out


def c3_bases(dev_ids_by_participant):
    """Participant PCA basis from all development embeddings and median D2 alpha per model."""
    rows = dev_embedding_rows()
    bases = {}
    for p, ids in dev_ids_by_participant.items():
        alphas = {m: [] for m in SETTINGS["models"]}
        for f in range(5):
            cell = json.loads((D2_CELLS / f"{p}-fold-{f}.json").read_text())
            for m in SETTINGS["models"]:
                alphas[m].append(cell["D2"][m]["R1_alpha_hat"])
        bases[p] = {}
        for m in SETTINGS["models"]:
            arr = np.load(DEV_EMBEDDINGS / f"{m}.npy", mmap_mode="r")
            x = l2(arr[[rows[i] for i in ids]])
            mu = x.mean(axis=0)
            _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
            bases[p][m] = {"mu": mu, "v": vt[:256].T, "s": s[:256], "alpha": float(np.median(alphas[m]))}
    return bases


# ----------------------------------------------------------------------------- analysis

def category_geometries(feats, idx, basis):
    """Rank vectors for native, spectral-swap and C3 geometries on images idx."""
    ref = SETTINGS["reference"]
    out = {f"native|{m}": ranks(upper(cos_rdm(feats[m][idx]))) for m in SETTINGS["models"]}
    svd = {m: centred_svd(feats[m][idx]) for m in SETTINGS["models"]}
    for m in SETTINGS["models"]:
        if m == ref:
            continue
        (um, sm), (ur, sr) = svd[m], svd[ref]
        k = min(len(sm), len(sr))
        out[f"swap|{m}|{ref}"] = ranks(upper(sq_rdm(um[:, :k] * sr[:k])))   # VLM eigenvectors, DINOv2 eigenvalues
        out[f"swap|{ref}|{m}"] = ranks(upper(sq_rdm(ur[:, :k] * sm[:k])))   # DINOv2 eigenvectors, VLM eigenvalues
    for m in SETTINGS["models"]:
        b = basis[m]
        phi = (feats[m][idx] - b["mu"]) @ b["v"] * (b["s"] ** (b["alpha"] - 1.0))
        out[f"c3|{m}"] = ranks(upper(sq_rdm(phi)))
    return out


def decomposition(a_p, c_p, d_p, b_p):
    """Native gap and its exact eigenvalue / eigenvector split from participant aggregates.

    a = VLM native, b = DINOv2 native, c = VLM eigenvectors with DINOv2 eigenvalues,
    d = DINOv2 eigenvectors with VLM eigenvalues.
    """
    parts = list(b_p)
    gap = {p: a_p[p] - b_p[p] for p in parts}
    swapped = {p: c_p[p] - b_p[p] for p in parts}
    e_val = {p: 0.5 * ((a_p[p] - c_p[p]) + (d_p[p] - b_p[p])) for p in parts}
    e_vec = {p: gap[p] - e_val[p] for p in parts}

    def mean(d):
        return float(np.mean(list(d.values())))
    return {"native_gap": gap, "native_gap_mean": mean(gap), "swapped_gap": swapped, "swapped_gap_mean": mean(swapped),
            "E_val": e_val, "E_vec": e_vec, "E_val_mean": mean(e_val), "E_vec_mean": mean(e_vec)}


def evaluate(results, feats, categories, bases, permutations, seed, c2b_confirmatory=None):
    parts = SETTINGS["participants"]
    rois = SETTINGS["primary_rois"] + [SETTINGS["descriptive_roi"]]
    conf, desc = SETTINGS["confirmatory_categories"], SETTINGS["descriptive_categories"]
    cells, counts, neural_ranks, eligible_pos = {}, {}, {}, {}
    for p in parts:
        r = results[p]
        for c in conf + desc:
            full = categories[p][c]
            elig = [i for i in full if r["eligible"][i]]
            eligible_pos[(p, c)] = [k for k, i in enumerate(full) if r["eligible"][i]]
            counts[f"{p}|{c}"] = {"images": len(full), "eligible": len(elig)}
            geoms = category_geometries(feats[p], np.array(elig), bases[p])
            for roi in rois:
                if roi == SETTINGS["descriptive_roi"]:
                    ev = [i for i in elig if r["evc_eligible"][i]]
                    if len(ev) != len(elig):
                        continue
                nr = ranks(upper(corr_rdm(r["patterns"][roi][np.array(elig)])))
                neural_ranks[(p, c, roi)] = nr
                cells[(p, c, roi)] = {g: fz(pearson(v, nr)) for g, v in geoms.items()}

    def agg(geom, cats, rois_used=SETTINGS["primary_rois"]):
        per_p = {p: float(np.mean([cells[(p, c, roi)][geom] for c in cats for roi in rois_used])) for p in parts}
        return per_p, float(np.mean(list(per_p.values())))

    ref, margin, rule = SETTINGS["reference"], SETTINGS["margin_fisher_z"], SETTINGS["participant_rule_min"]
    report = {"counts": counts}
    # Gate: shared within-category permutations of the DINOv2 model RDM.
    obs_p, obs = agg(f"native|{ref}", conf)
    rng = np.random.default_rng(seed)
    full_rdm = {}
    for c in conf:
        full_rdm[c] = {p: cos_rdm(feats[p][ref][np.array(categories[p][c])]) for p in parts}
    null = np.empty(permutations)
    for b in range(permutations):
        vals = {p: [] for p in parts}
        for c in conf:
            n_c = next(iter(full_rdm[c].values())).shape[0]
            perm = rng.permutation(n_c)
            for p in parts:
                e = np.array(eligible_pos[(p, c)])
                d = full_rdm[c][p][np.ix_(perm[e], perm[e])]
                mr = ranks(upper(d))
                for roi in SETTINGS["primary_rois"]:
                    vals[p].append(fz(pearson(mr, neural_ranks[(p, c, roi)])))
        null[b] = np.mean([np.mean(v) for v in vals.values()])
    threshold = float(np.quantile(null, SETTINGS["null_quantile"], method=SETTINGS["null_quantile_method"]))
    gate = bool(obs > threshold)
    report["gate"] = {"DINOv2_aggregate": obs, "participants": obs_p, "null_threshold": threshold,
                      "null_mean": float(null.mean()), "permutations": permutations, "passed": gate}

    def classify(m):
        gp = {p: agg(f"native|{m}", conf)[0][p] - obs_p[p] for p in parts}
        g = float(np.mean(list(gp.values())))
        spectral = g <= -margin and sum(v <= -margin for v in gp.values()) >= rule
        selection = g > -margin and sum(v > -margin for v in gp.values()) >= rule
        loco = {}
        for drop in conf:
            keep = [c for c in conf if c != drop]
            loco[drop] = agg(f"native|{m}", keep)[1] - agg(f"native|{ref}", keep)[1]
        return {"gap_mean": g, "gap_participants": gp, "leave_one_category_out_gap": loco,
                "classification": "spectral" if spectral else "selection_only" if selection else "indeterminate"}

    report["C1"] = {"CLIP": classify("CLIP"), "PEcore_secondary": classify("PEcore")}
    # C2a and the PEcore secondary use the within-category decomposition.
    within = {m: decomposition(agg(f"native|{m}", conf)[0], agg(f"swap|{m}|{ref}", conf)[0],
                               agg(f"swap|{ref}|{m}", conf)[0], obs_p) for m in ("CLIP", "SigLIP2", "PEcore")}
    # C2b uses all pairs among each participant's eligible confirmatory images.
    all_cells = {}
    for p in parts:
        r = results[p]
        union = np.array(sorted(i for c in conf for i in categories[p][c] if r["eligible"][i]))
        geoms = category_geometries(feats[p], union, bases[p])
        for roi in SETTINGS["primary_rois"]:
            nr = ranks(upper(corr_rdm(r["patterns"][roi][union])))
            all_cells[(p, roi)] = {g: fz(pearson(v, nr)) for g, v in geoms.items()}

    def agg_all(geom):
        return {p: float(np.mean([all_cells[(p, roi)][geom] for roi in SETTINGS["primary_rois"]])) for p in parts}

    ref_all = agg_all(f"native|{ref}")
    allpairs = {m: decomposition(agg_all(f"native|{m}"), agg_all(f"swap|{m}|{ref}"), agg_all(f"swap|{ref}|{m}"), ref_all)
                for m in ("CLIP", "SigLIP2", "PEcore")}
    c2a = {m: {"E_val_mean": within[m]["E_val_mean"], "E_val": within[m]["E_val"],
               "passed": bool(within[m]["E_val_mean"] < 0 and sum(v < 0 for v in within[m]["E_val"].values()) >= rule)}
           for m in ("CLIP", "SigLIP2")}
    c2b = {}
    for m in ("CLIP", "SigLIP2"):
        e = allpairs[m]
        g = e["native_gap_mean"]
        evaluable = bool(g <= -margin)
        ok_p = [e["native_gap"][p] < 0 and abs(e["swapped_gap"][p]) <= 0.5 * abs(e["native_gap"][p]) for p in parts]
        c2b[m] = {"native_gap_mean": g, "swapped_gap_mean": e["swapped_gap_mean"],
                  "closure_mean": (1.0 - abs(e["swapped_gap_mean"]) / abs(g)) if g else None,
                  "participants_meeting_rule": int(sum(ok_p)), "evaluable": evaluable,
                  "passed": bool(abs(e["swapped_gap_mean"]) <= 0.5 * abs(g) and sum(ok_p) >= rule) if evaluable else None}
    pe = within["PEcore"]
    ok_pe = [pe["E_val"][p] < 0 and pe["E_vec"][p] > 0 for p in parts]
    report["C2"] = {"C2a_within_category": c2a, "C2b_all_pairs": c2b, "C2b_confirmatory": c2b_confirmatory,
                    "PEcore_secondary": {"E_val_mean": pe["E_val_mean"], "E_vec_mean": pe["E_vec_mean"],
                                         "passed": bool(pe["E_val_mean"] < 0 and pe["E_vec_mean"] > 0 and sum(ok_pe) >= rule)},
                    "within_category_detail": within, "all_pairs_detail": allpairs,
                    "all_pairs_cells": {"|".join(k): v for k, v in all_cells.items()}}
    c3 = {}
    for m in SETTINGS["models"]:
        nat_p, nat = agg(f"native|{m}", conf)
        tr_p, tr = agg(f"c3|{m}", conf)
        delta = {p: tr_p[p] - nat_p[p] for p in parts}
        c3[m] = {"delta_mean": tr - nat, "delta_participants": delta, "alpha": {p: bases[p][m]["alpha"] for p in parts},
                 "passed": bool(tr - nat > 0 and sum(v > 0 for v in delta.values()) >= rule) if m != ref else None}
    report["C3"] = c3
    report["descriptive"] = {
        "native_by_category": {c: {m: agg(f"native|{m}", [c])[1] for m in SETTINGS["models"]} for c in conf + desc},
        "EVC_native_confirmatory": {m: agg(f"native|{m}", conf, [SETTINGS["descriptive_roi"]])[1]
                                    for m in SETTINGS["models"]}
        if all((p, c, SETTINGS["descriptive_roi"]) in cells for p in parts for c in conf) else None,
    }
    report["cells"] = {"|".join(k): v for k, v in cells.items()}
    if not gate:
        report["verdict_status"] = "uninformative_gate_failed"
    else:
        report["verdict_status"] = "scored"
    return report


# ----------------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev-test", "confirm"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=None, help="dev-test only")
    ap.add_argument("--approval-record", default=None, help="confirm only: experiment-log anchor of the scientist's approval")
    a = ap.parse_args()
    out_dir = a.out_dir.resolve()
    t0 = time.time()
    report = {"evaluator": "ood_confirmation", "mode": a.mode, "started_utc": utc(), "hostname": socket.gethostname(),
              "job_id": os.environ.get("SLURM_JOB_ID"), "code_sha256": code_checksums(), "settings": SETTINGS}
    if a.mode == "confirm":
        lock = json.loads(LOCK.read_text())
        if lock.get("code_sha256") != report["code_sha256"] or lock.get("settings") != SETTINGS:
            raise SystemExit("lock does not match this evaluator; refusing to open OOD responses")
        if json.loads(EXPOSURE_REVIEW.read_text()).get("conclusion") != "pass" or not a.approval_record:
            raise SystemExit("exposure review pass and a recorded scientist approval are required")
        if (out_dir / "confirmation_report.json").exists():
            raise SystemExit("confirmation report already exists; the opening is one-shot")
        report["lock_sha256"] = sha256(LOCK)
        report["approval_record"] = a.approval_record
        permutations = SETTINGS["permutations"]
        table = list(csv.DictReader(open(CATEGORY_TABLE)))
        target_ids = sorted(r["image_id"] for r in table)
        type_of = {r["image_id"]: r["ood_type"] for r in table}
        shared = {c: [i for i, n in enumerate(target_ids) if type_of[n] == c]
                  for c in SETTINGS["confirmatory_categories"] + SETTINGS["descriptive_categories"]}
        categories = {p: shared for p in SETTINGS["participants"]}
        tasks = [{"participant": p, "mode": "confirm", "target_ids": target_ids} for p in SETTINGS["participants"]]
    else:
        permutations = a.permutations or 1000
        rng = np.random.default_rng(SETTINGS["permutation_seed"])
        meta = [json.loads(line) for line in open(EP04 / METADATA / "development_images.jsonl")]
        tasks, categories = [], {}
        for p in SETTINGS["participants"]:
            pool = {s: sorted(r["image_id"] for r in meta if r["participant"] == p and r["source_stratum"] == s)
                    for s in ("laion", "things", "thingsplus")}
            chosen, cat_pos = [], {}
            for c in SETTINGS["confirmatory_categories"] + SETTINGS["descriptive_categories"]:
                stratum, size = PSEUDO[c]
                avail = [i for i in pool[stratum] if i not in set(chosen)]
                pick = list(rng.choice(avail, size=size, replace=False))
                cat_pos[c] = list(range(len(chosen), len(chosen) + size))
                chosen += pick
            categories[p] = cat_pos
            tasks.append({"participant": p, "mode": "dev-test", "target_ids": chosen,
                          "verify_reader": ["ses-05", "ses-17"] if p == "sub-03" else None})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results = {r["participant"]: r for r in pool.map(participant_patterns, tasks)}
    report["neural"] = {p: {"presentations": r["presentations"], "eligible_images": int(r["eligible"].sum()),
                            "images": len(r["eligible"]), "primary_count_histogram": np.bincount(r["primary_counts"]).tolist(),
                            "voxels": r["voxels"], "reader": r["reader"]} for p, r in results.items()}
    if a.mode == "confirm":
        feats_all = model_features(target_ids, "confirm")
        feats = {p: feats_all for p in SETTINGS["participants"]}
    else:
        feats = {t["participant"]: model_features(t["target_ids"], "dev-test") for t in tasks}
    bases = c3_bases({p: r["dev_image_ids"] for p, r in results.items()})
    c2b_conf = lock["c2b_confirmatory"] if a.mode == "confirm" else None
    report["evaluation"] = evaluate(results, feats, categories, bases, permutations, SETTINGS["permutation_seed"],
                                    c2b_confirmatory=c2b_conf)
    report["elapsed_seconds"] = time.time() - t0
    report["finished_utc"] = utc()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "confirmation_report.json" if a.mode == "confirm" else "dev_test_report.json"
    with open(out_dir / name, "x") as f:          # write-once
        json.dump(report, f, indent=1, default=float)
    print(json.dumps({"mode": a.mode, "written": name, "elapsed_seconds": round(report["elapsed_seconds"], 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
