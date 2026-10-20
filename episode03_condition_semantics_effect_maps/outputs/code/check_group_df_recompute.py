#!/usr/bin/env python3
"""Check fitted N and contributor lineage (drafted by Codex gpt-6-astra, reviewed 2026-10-17).
Image inputs are Dataset-level t/z/effect/variance images and the feeding
node's effect images only. Read census metadata; write JSON, never an image.
"""
import argparse
import gc
import itertools
import json
import math
import os
import socket
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import norm, t as t_dist

DF_Z_TOL = 1e-3
EFFECT_REL_TOL = 1e-4
T_CORR_MIN = 0.9999
T_MEDIAN_TOL = 1e-3


def finite_float(value):
    return float(value) if np.isfinite(value) else None


def sanitize(obj):
    """Replace non-finite floats with None so one stray NaN cannot lose the report."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def differences(reference, estimate):
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        maximum = np.max(np.abs(reference - estimate))
        scale = np.max(np.abs(reference))
        relative = finite_float(maximum / scale) if scale > 0 else None
    return finite_float(maximum), relative


def find_inputs(base, row):
    found = []
    session = row.get("session")

    def scan(directory, allow_sessions):
        with os.scandir(directory) as entries:
            for entry in entries:
                if allow_sessions and entry.name.startswith("ses-") and entry.is_dir():
                    scan(entry.path, False)
                elif entry.is_file() and entry.name.endswith(".nii.gz"):
                    parts = entry.name[:-7].split("_")
                    entities = dict(p.split("-", 1) for p in parts[:-1] if "-" in p)
                    if (parts[-1] == "statmap" and entities.get("stat") == "effect"
                            and entities.get("contrast") == row["contrast"]
                            and "sub" in entities
                            and (session is None or entities.get("ses") == session)):
                        found.append((entities["sub"], Path(entry.path)))

    with os.scandir(base) as entries:
        for entry in entries:
            if entry.name.startswith("sub-") and entry.is_dir():
                scan(entry.path, True)
    return sorted(found, key=lambda item: (item[0], str(item[1])))


def infer_df(t_values, z_values, n, args, result):
    absolute_t = np.abs(t_values)
    indices = np.flatnonzero(
        (absolute_t > 1) & (absolute_t < 7) & np.isfinite(z_values) & (z_values != 0))
    result["df_candidates"] = int(indices.size)
    if indices.size < 50:
        result["df_status"] = "insufficient_voxels"
        return
    if indices.size > args.max_voxels:
        indices = np.random.default_rng(args.seed).choice(
            indices, size=args.max_voxels, replace=False)
    result["df_sample_voxels"] = int(indices.size)
    tv, zv = t_values[indices], z_values[indices]

    def score(df):
        predicted = np.sign(tv) * norm.isf(t_dist.sf(np.abs(tv), df))
        error = finite_float(np.median(np.abs(predicted - zv)))
        if error is None:
            raise ArithmeticError("Non-finite t-to-z median error")
        return error

    ranked = sorted((score(df), df) for df in range(1, args.df_max + 1))
    result["best_df_error"], result["best_df"] = ranked[0]
    if len(ranked) > 1:
        result["second_df_error"], result["second_df"] = ranked[1]
    result["error_at_n_minus_1"] = score(n - 1)
    result["df_status"] = "ok"


def check_map(row, args):
    started = time.perf_counter()
    result = dict.fromkeys((
        "mask_voxels nonfinite_input_voxels zero_sd_voxels comparison_voxels "
        "effect_max_abs_diff effect_rel_max_diff variance_rel_max_diff "
        "t_corr t_max_abs_diff t_median_abs_diff df_candidates best_df "
        "best_df_error second_df second_df_error error_at_n_minus_1 loo error"
    ).split())
    result.update(
        route=row.get("route"), session=row.get("session"), contrast=row.get("contrast"),
        path=row.get("path"), n_census=row.get("inputs"), n_found=None,
        inputs_found=[], subjects_found=[], subjects_missing=[], subjects_extra=[],
        duplicate_subject_inputs={}, geometry_mismatches=[], df_status="not_checked",
        df_sample_voxels=0, status="error", failed=[])
    try:
        dataset_root = args.analysis_root / row["dataset_id"]
        base = dataset_root / f"task-{row['task']}" / row["source_node"]
        found = find_inputs(base, row)
        n, expected = len(found), row["subjects"]
        counts = Counter(subject for subject, _ in found)
        result.update(n_found=n, n_census=int(row["inputs"]))
        result["inputs_found"] = [{"subject": s, "path": str(p)} for s, p in found]
        result["subjects_found"] = sorted(counts)
        result["subjects_missing"] = sorted(set(expected) - set(counts))
        result["subjects_extra"] = sorted(set(counts) - set(expected))
        result["duplicate_subject_inputs"] = {s: c for s, c in counts.items() if c > 1}
        if result["duplicate_subject_inputs"]:
            result["failed"].append("duplicate_subject_inputs")
            raise ValueError("More than one qualifying input for a subject")

        z_path = dataset_root / row["path"]
        if z_path.name.count("_stat-z_statmap") != 1:
            raise ValueError("Dataset basename must contain exactly one _stat-z_statmap")
        stats = ("z", "t", "effect", "variance")
        paths = [z_path.with_name(z_path.name.replace(
            "_stat-z_statmap", f"_stat-{stat}_statmap")) for stat in stats]
        images = [(p, nib.load(str(p))) for p in paths]
        sources = [(p, nib.load(str(p))) for _, p in found]
        reference = images[0][1]
        for path, img in images + sources:
            shape_ok = img.shape == reference.shape
            affine_ok = bool(np.allclose(img.affine, reference.affine, atol=1e-5))
            if not shape_ok or not affine_ok:
                result["geometry_mismatches"].append({
                    "path": str(path), "shape": [int(v) for v in img.shape],
                    "reference_shape": [int(v) for v in reference.shape],
                    "affine_matches": affine_ok})
        if result["geometry_mismatches"]:
            result["failed"].append("geometry_mismatch")
            raise ValueError("Image geometry mismatch; numerical checks skipped")

        data = {stat: np.asanyarray(img.dataobj, dtype=np.float64)
                for stat, (_, img) in zip(stats, images)}
        mask = np.isfinite(data["t"]) & (data["t"] != 0)
        result["mask_voxels"] = int(np.count_nonzero(mask))
        if result["mask_voxels"] == 0:
            result.update(status="empty_mask", failed=["empty_mask"])
            return result
        if n < 2:
            raise ValueError("At least two qualifying inputs are required for ddof=1")
        data = {stat: values[mask] for stat, values in data.items()}
        x = np.empty((n, result["mask_voxels"]), dtype=np.float64)
        finite_inputs = np.ones(result["mask_voxels"], dtype=bool)
        for i, (_, img) in enumerate(sources):
            x[i] = np.asanyarray(img.dataobj, dtype=np.float64)[mask]
            finite_inputs &= np.isfinite(x[i])
        del mask
        result["nonfinite_input_voxels"] = int(np.count_nonzero(~finite_inputs))

        with np.errstate(invalid="ignore", over="ignore"):
            mean = x.mean(axis=0)
            sd = x.std(axis=0, ddof=1)
        result["zero_sd_voxels"] = int(np.count_nonzero(finite_inputs & (sd == 0)))
        usable = finite_inputs & (sd != 0)
        result["comparison_voxels"] = int(np.count_nonzero(usable))
        if not np.any(usable):
            raise ValueError("No voxels remain after non-finite-input and zero-SD exclusions")
        mean = mean[usable]
        data = {stat: values[usable] for stat, values in data.items()}
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            variance_rc = sd[usable] ** 2 / n
            t_rc = mean / np.sqrt(variance_rc)
        if not all(np.isfinite(v).all() for v in (mean, variance_rc, t_rc)):
            raise ArithmeticError("Non-finite recomputation on otherwise usable voxels")
        effect, variance, t_values, z_values = (
            data[stat] for stat in ("effect", "variance", "t", "z"))
        result["effect_max_abs_diff"], result["effect_rel_max_diff"] = differences(
            effect, mean)
        _, result["variance_rel_max_diff"] = differences(variance, variance_rc)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            delta = np.abs(t_values - t_rc)
            result["t_max_abs_diff"] = finite_float(np.max(delta))
            result["t_median_abs_diff"] = finite_float(np.median(delta))
            result["t_corr"] = (finite_float(np.corrcoef(t_values, t_rc)[0, 1])
                                if t_values.size > 1 else None)
        infer_df(t_values, z_values, n, args, result)

        def within(value, limit):
            return value is not None and value <= limit

        criteria = {
            "best_df_equals_n_minus_1": result["best_df"] == n - 1,
            "df_error_tolerance": within(result["error_at_n_minus_1"], DF_Z_TOL),
            "effect_relative_tolerance": within(result["effect_rel_max_diff"], EFFECT_REL_TOL),
            "t_correlation": result["t_corr"] is not None and result["t_corr"] >= T_CORR_MIN,
            "t_median_tolerance": within(result["t_median_abs_diff"], T_MEDIAN_TOL),
            "input_count_matches_census": n == result["n_census"],
            "subjects_match_census": sorted(counts.elements()) == sorted(expected),
        }
        result["failed"] = [name for name, passed in criteria.items() if not passed]
        if result["df_status"] != "ok":
            result["failed"].insert(0, "df_" + result["df_status"])
        result["status"] = "not_confirmed" if result["failed"] else "confirmed"
        # Leave-k-out when the fitted df implies k = n - 1 - best_df dropped inputs
        # (added 2026-10-17: single leave-one-out cannot find two dropped contributors).
        k = (n - 1 - result["best_df"]) if result["best_df"] is not None else 0
        if result["status"] == "not_confirmed" and 2 <= k <= 3 and math.comb(n, k) <= 20000:
            xu = x[:, usable]
            total = mean * n
            best = None
            for combo in itertools.combinations(range(n), k):
                mean_without = (total - xu[list(combo)].sum(axis=0)) / (n - k)
                _, value = differences(effect, mean_without)
                if value is not None and (best is None or value < best[0]):
                    best = (value, combo, mean_without)
            if best is not None:
                keep = [i for i in range(n) if i not in best[1]]
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    t_sub = best[2] / (xu[keep].std(axis=0, ddof=1) / np.sqrt(n - k))
                    ok = np.isfinite(t_sub)
                    t_sub_corr = finite_float(np.corrcoef(t_values[ok], t_sub[ok])[0, 1])
                    t_sub_median = finite_float(np.median(np.abs(t_values[ok] - t_sub[ok])))
                result["leave_k_out"] = {
                    "k": int(k), "combinations": int(math.comb(n, k)),
                    "dropped_subjects": [found[i][0] for i in best[1]],
                    "effect_rel_max_diff": best[0], "t_corr": t_sub_corr,
                    "t_median_abs_diff": t_sub_median}
                if (best[0] <= EFFECT_REL_TOL and t_sub_corr is not None
                        and t_sub_corr >= T_CORR_MIN and t_sub_median is not None
                        and t_sub_median <= T_MEDIAN_TOL):
                    result["status"] = "confirmed_after_correction"
                    result["n_fitted"] = int(n - k)
            del xu
        if result["status"] == "not_confirmed" and n >= 3:
            result["loo"] = {
                "subject": None, "input_path": None, "effect_rel_max_diff": None,
                "full_set_effect_rel_max_diff": result["effect_rel_max_diff"]}
            for i, (subject, path) in enumerate(found):
                # Algebraic leave-one-out mean, using the same voxel subset.
                with np.errstate(invalid="ignore", over="ignore"):
                    mean_without = mean + (mean - x[i, usable]) / (n - 1)
                _, value = differences(effect, mean_without)
                best = result["loo"]["effect_rel_max_diff"]
                if value is not None and (best is None or value < best):
                    result["loo"].update(
                        subject=subject, input_path=str(path), effect_rel_max_diff=value)
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        if not result["failed"]:
            result["failed"] = ["exception"]
    finally:
        result["seconds"] = float(time.perf_counter() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-maps", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--routes", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--df-max", type=int, default=400)
    parser.add_argument("--max-voxels", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.df_max < 1 or args.max_voxels < 1 or args.seed < 0:
        parser.error("--df-max and --max-voxels must be positive; --seed must be nonnegative")
    started = time.perf_counter()
    with args.census_maps.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        parser.error("--census-maps must contain a JSON list of objects")
    results = []
    for row in rows:
        if args.routes is not None and row.get("route") not in args.routes:
            continue
        result = check_map(row, args)
        results.append(result)
        print(f"{result['route']} {result['contrast']} session={result['session']} "
              f"{result['status']} {result['seconds']:.3f}s", flush=True)
        gc.collect()
    overall = Counter(result["status"] for result in results)
    per_route = {}
    for result in results:
        per_route.setdefault(result["route"], Counter())[result["status"]] += 1
    report = {
        "check": "fitted_n_lineage", "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "inputs": {key: str(value) for key, value in vars(args).items()},
        "thresholds": {"DF_Z_TOL": DF_Z_TOL, "EFFECT_REL_TOL": EFFECT_REL_TOL,
                       "T_CORR_MIN": T_CORR_MIN, "T_MEDIAN_TOL": T_MEDIAN_TOL},
        "maps": results,
        "summary": {"overall": dict(overall),
                    "per_route": {route: dict(counts) for route, counts in per_route.items()}},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(report), handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
