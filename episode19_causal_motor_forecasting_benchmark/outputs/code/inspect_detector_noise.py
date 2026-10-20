#!/usr/bin/env python3
"""Noise-only inspection before the EP19 detector rules are written (October 16, 2026).

Describes development pre-cue sensor noise, position quantization, object/force
rest references, TRIG tone bursts and accelerometer axis variance. It computes
no threshold crossing, onset, label or provider annotation. Data: WAY P1-P12
series 1-7; self-paced development participants only.
"""
import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector_readers as R  # noqa: E402
import detector_summaries as S  # noqa: E402

Q = (50, 90, 99, 99.9, 100)


def pct(x, q=Q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return None if not len(x) else [float(v) for v in np.percentile(x, q)]


def robust(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = float(np.median(x))
    return m, float(1.4826 * np.median(np.abs(x - m)))


def way_participant(p):
    windows, values = [], []
    quanta = []
    for s in R.WAY_SERIES:
        w = R.load_way_series(p, s)
        summ = S.way_summaries(w.emg, w.positions_cm, w.n)
        pos = w.positions_cm
        d_digit = np.minimum(np.linalg.norm(pos[:, 1] - pos[:, 0], axis=1),
                             np.linalg.norm(pos[:, 2] - pos[:, 0], axis=1))
        diffs = np.abs(np.diff(pos[:, 1:4], axis=0)).reshape(-1)
        nz = diffs[diffs > 0]
        quanta.append(float(np.min(nz)) if len(nz) else None)
        for k, r in enumerate(w.rises):
            r = int(r)
            if r - 500 < 300:
                continue
            seg = slice(r - 500, r)
            hand = pos[seg, 1:4]
            spread = np.max(np.linalg.norm(hand - np.median(hand, axis=0), axis=2), axis=0)
            row = {"series": s, "cue": k, "spread_cm": spread.tolist(),
                   "clearance_cm": float(np.min(d_digit[r - 250:r])),
                   "p1_median": np.median(pos[seg, 0], axis=0).tolist(),
                   "p1_spread": float(np.max(np.linalg.norm(pos[seg, 0] - np.median(pos[seg, 0], axis=0), axis=1))),
                   "force_median": np.median(w.forces_n[seg], axis=0).tolist(),
                   "force_sd": np.std(w.forces_n[seg], axis=0).tolist(),
                   "summary_finite": bool(np.all(np.isfinite(summ[seg])))}
            windows.append(row)
            values.append(summ[seg])
        del w, summ
    out = {"participant": p, "windows": len(windows), "min_nonzero_position_step_cm": quanta}
    spread = np.array([r["spread_cm"] for r in windows])
    clear = np.array([r["clearance_cm"] for r in windows])
    for lim in (0.25, 0.5, 1.0):
        out[f"still_{lim}cm_and_clear"] = int(np.sum(np.all(spread <= lim, axis=1) & (clear >= 15)))
    out["clear_only"] = int(np.sum(clear >= 15))
    keep = np.all(spread <= 0.5, axis=1) & (clear >= 15) & np.array([r["summary_finite"] for r in windows])
    v = np.concatenate([values[i] for i in np.flatnonzero(keep)]) if keep.any() else np.zeros((0, 8))
    paths = {}
    for j, name in enumerate(S.WAY_PATHS):
        m, sd = robust(v[:, j])
        paths[name] = {"median": m, "mad_sd": sd, "pct_50_90_99_999_100": pct(v[:, j]),
                       "frac_zero": float(np.mean(v[:, j] == 0)),
                       "q99_minus_median_over_2326": float((np.percentile(v[:, j], 99) - m) / 2.326),
                       "exceed_m3sd": float(np.mean(v[:, j] > m + 3 * sd)) if sd > 0 else None,
                       "exceed_m6sd": float(np.mean(v[:, j] > m + 6 * sd)) if sd > 0 else None}
    out["paths"] = paths
    p1 = np.array([r["p1_median"] for r in windows])[keep]
    out["p1_median_across_windows"] = np.median(p1, axis=0).tolist() if len(p1) else None
    out["p1_abs_dev_pct_50_95_100"] = [pct(np.abs(p1[:, a] - np.median(p1[:, a])), (50, 95, 100)) for a in range(3)] if len(p1) else None
    out["p1_within_window_spread_pct"] = pct([r["p1_spread"] for r, k in zip(windows, keep) if k], (50, 95, 100))
    f = np.array([r["force_median"] for r in windows])[keep]
    out["force_median_across_windows"] = np.median(f, axis=0).tolist() if len(f) else None
    out["force_abs_dev_pct_95_100"] = np.percentile(np.abs(f - np.median(f, axis=0)), [95, 100], axis=0).tolist() if len(f) else None
    out["force_within_sd_median"] = np.median(np.array([r["force_sd"] for r in windows])[keep], axis=0).tolist() if len(f) else None
    series_force = {}
    for s in R.WAY_SERIES:
        idx = [i for i, r in enumerate(windows) if r["series"] == s and keep[i]]
        if idx:
            series_force[s] = np.median(np.array([windows[i]["force_median"] for i in idx]), axis=0).round(3).tolist()
    out["force_median_by_series"] = series_force
    out["stillness_window_count_kept"] = int(keep.sum())
    return out


def trig_bursts(trig, fs):
    d = np.abs(np.diff(trig, prepend=trig[0]))
    e = S.trailing_mean(d, int(round(0.02 * fs)))
    b = int(fs)
    nb = len(d) // b
    med = np.median(d[:nb * b].reshape(nb, b), axis=1)
    ref = np.full(len(d), np.nan)
    for j in range(2, nb + 1):
        ref[j * b:(j + 1) * b] = np.median(med[max(0, j - 10):j])
    ratio = e / ref
    hot = ratio > 20
    edges = np.flatnonzero(np.diff(np.concatenate([[0], hot.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))
    merged = []
    for a, z in runs:
        if merged and a - merged[-1][1] < int(0.05 * fs):
            merged[-1][1] = z
        else:
            merged.append([a, z])
    bursts = [{"onset_s": a / fs, "duration_s": (z - a) / fs,
               "peak_ratio": float(np.nanmax(ratio[a:z]))} for a, z in merged]
    return bursts, float(np.nanmedian(ref))


def sp_participant(participant):
    out = {"participant": participant, "runs": []}
    pre_dev, pre_sd = [], []
    for stem in R.development_stems():
        if os.path.basename(stem).split("_")[0] != participant:
            continue
        run = R.load_self_paced_run(stem)
        bursts, ref_median = trig_bursts(run.trig_uv, run.fs)
        acc, _ = R.reduce_to_detector_rate(run.acc_g, run.fs)
        dev = S.self_paced_departure(acc)
        sd1 = R.trailing_sd(acc, R.SP_DETECTOR_HZ)
        long_b = [x for x in bursts if x["duration_s"] >= 0.2]
        kinds = ["start" if i % 2 == 0 else "end" for i in range(len(long_b))]
        for x, k in zip(long_b, kinds):
            if k == "start":
                o = int(x["onset_s"] * R.SP_DETECTOR_HZ)
                if o - 250 >= 130:
                    pre_dev.append(dev[o - 250:o])
                    pre_sd.append(np.std(acc[o - 250:o], axis=0))
        out["runs"].append({
            "stem": run.stem, "fs": run.fs, "trig_ref_median_uV": ref_median, "bursts": len(bursts),
            "burst_durations_s": [round(x["duration_s"], 3) for x in bursts],
            "burst_onsets_s": [round(x["onset_s"], 3) for x in bursts],
            "burst_peak_ratio": [round(x["peak_ratio"], 1) for x in bursts],
            "long_burst_kinds_by_alternation": kinds,
            "trailing_1s_axis_sd_g_min": np.nanmin(sd1, axis=0).tolist(),
            "trailing_1s_axis_sd_g_p1": np.nanpercentile(sd1, 1, axis=0).tolist(),
            "acc_norm_median_g": float(np.median(np.linalg.norm(acc, axis=1)))})
    v = np.concatenate(pre_dev) if pre_dev else np.zeros(0)
    m, sd = robust(v) if len(v) else (None, None)
    out["pre_start_departure_g"] = {"windows": len(pre_dev), "median": m, "mad_sd": sd,
                                    "pct_50_90_99_999_100": pct(v),
                                    "pct_by_window_max_50_90_100": pct([np.nanmax(x) for x in pre_dev], (50, 90, 100))}
    out["pre_start_axis_sd_g_pct_50_95"] = np.percentile(np.array(pre_sd), [50, 95], axis=0).tolist() if pre_sd else None
    out["_pre_values"] = v.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run inside a Slurm allocation on a compute node")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        way = list(pool.map(way_participant, R.WAY_PARTICIPANTS))
        sp = list(pool.map(sp_participant, R.SP_DEVELOPMENT))
    pooled = np.concatenate([np.asarray(x.pop("_pre_values")) for x in sp])
    m, sd = robust(pooled)
    report = {"analysis": "detector_noise_inspection", "job_id": os.environ["SLURM_JOB_ID"],
              "host": socket.gethostname(), "crossings_onsets_labels_or_provider_annotations_computed": False,
              "way": way, "self_paced": sp,
              "self_paced_pooled_pre_start_departure_g": {"median": m, "mad_sd": sd, "pct": pct(pooled),
                                                          "frac_zero": float(np.mean(pooled == 0))},
              "elapsed_s": time.time() - t0}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
