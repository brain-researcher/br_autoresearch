#!/usr/bin/env python3
"""WAY-EEG-GAL development source-fact checks (EP19, October 16, 2026).

Scope: participants P1-P12, series 1-7 only. HS files come from the
job-46425841 development view; WS windows for the same series are read from
the provider archives. Series 8-9, AllLifts and EEG models are not touched.
EEG is read only for window alignment and the magnet-transient clock check.

Pre-declared predictions and decision rules are in
outputs/source_timing_handoff.md (October 16 section).
"""
import argparse
import io
import json
import os
import socket
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io as sio

FS = 500
ARCHIVES = os.path.join(os.environ.get("OAK", ""), "users/zijiao/br_autoresearch_data/"
                        "steward_acquisition/ep19_public_sources/archives/way")
VIEW = os.path.join(os.environ.get("SCRATCH", ""), "br_autoresearch/"
                    "episode19_causal_motor_forecasting_benchmark/development_view_job_46425841/way")
SERIES = range(1, 8)


def mat_header(raw):
    return raw[:116].decode("latin-1").strip()


def name_list(values):
    return [str(v).strip() for v in np.atleast_1d(values)]


def col(sub, name):
    return np.asarray(sub.sig, dtype=float)[:, name_list(sub.names).index(name)]


def edges(x, threshold=0.5):
    high = x > threshold
    return np.flatnonzero(high[1:] & ~high[:-1]) + 1, np.flatnonzero(~high[1:] & high[:-1]) + 1


def ncc_offset(hs, win, row0, k, lo, hi):
    """HS index n0 of window row 0 maximizing mean normalized correlation of first differences."""
    lo = max(lo, 0)
    hi = min(hi, len(hs) - row0 - k - 1)
    total = np.zeros(hi - lo)
    for c in range(win.shape[1]):
        a = np.diff(win[row0:row0 + k + 1, c])
        if a.std() == 0:
            continue
        a = (a - a.mean()) / a.std()
        b = np.diff(hs[lo + row0:hi + row0 + k, c])
        num = np.correlate(b, a, mode="valid")[:hi - lo]
        cs = np.concatenate([[0.0], np.cumsum(b)])
        cs2 = np.concatenate([[0.0], np.cumsum(b * b)])
        mu = (cs[k:] - cs[:-k]) / k
        var = (cs2[k:] - cs2[:-k]) / k - mu ** 2
        total += num[:hi - lo] / (k * np.sqrt(np.maximum(var[:hi - lo], 1e-24)))
    best = int(np.argmax(total))
    return lo + best, float(total[best] / win.shape[1])


def full_match(hs, win, n0):
    seg = hs[n0:n0 + len(win)]
    if len(seg) != len(win):
        return None
    diffs = np.abs(win - seg)
    slopes, icpts = [], []
    for c in range(win.shape[1]):
        if np.std(seg[:, c]) > 0:
            s, i = np.polyfit(seg[:, c], win[:, c], 1)
            slopes.append(s)
            icpts.append(abs(i))
    return {"exact": bool(np.all(diffs == 0)), "max_abs_diff": float(np.nanmax(diffs)),
            "slope_min": float(min(slopes)), "slope_max": float(max(slopes)),
            "max_abs_intercept": float(max(icpts))}


def equal_runs(x):
    change = np.flatnonzero(np.diff(x) != 0)
    return np.diff(change)


def liftoff_lag(lf_net, weight, z_obs, g=9.81):
    """FASTRAK object-rise lag behind the rise predicted from load force (centimetre reading).

    From the first sample where net load force reaches the trial's own static-hold
    weight W, integrate a = g (LF - W) / W twice; compare the times at which the
    predicted and observed vertical displacements first reach 0.2 and 0.5 cm.
    """
    out = {"liftoff_lag_02_s": None, "liftoff_lag_05_s": None}
    if not np.isfinite(weight) or weight < 0.5:
        return out
    above = np.flatnonzero(lf_net >= weight)
    if not len(above):
        return out
    t_f = int(above[0])
    acc = g * (lf_net[t_f:] - weight) / weight
    z_pred = 100.0 * np.cumsum(np.cumsum(acc) / FS) / FS
    for thr, key in ((0.2, "liftoff_lag_02_s"), (0.5, "liftoff_lag_05_s")):
        pred = np.flatnonzero(z_pred >= thr)
        obs = np.flatnonzero(z_obs[t_f - 50 if t_f >= 50 else 0:] >= thr)
        if len(pred) and len(obs):
            obs_t = (t_f - 50 if t_f >= 50 else 0) + int(obs[0])
            out[key] = (obs_t - (t_f + int(pred[0]))) / FS
    return out


def analyze(task):
    p, s = task
    t0 = time.time()
    role = "fit_and_inner_selection" if s <= 6 else "aggregate_development_feedback"
    with open(os.path.join(VIEW, role, f"HS_P{p}_S{s}.mat"), "rb") as f:
        raw = f.read()
    hs_header = mat_header(raw)
    hs = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)["hs"]
    with zipfile.ZipFile(os.path.join(ARCHIVES, f"P{p}.zip")) as z:
        raw = z.read(f"WS_P{p}_S{s}.mat")
    ws_header = mat_header(raw)
    ws = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)["ws"]
    del raw

    kin = np.asarray(hs.kin.sig, dtype=float)
    kin_names = name_list(hs.kin.names)
    eeg = np.asarray(hs.eeg.sig, dtype=float)
    emg = np.asarray(hs.emg.sig, dtype=float)
    n = len(kin)
    led = col(hs.misc, "ParticipantLED")
    button = col(hs.misc, "Button")
    magnet = col(hs.misc, "Magnet")
    weight_env = col(hs.env, "Weight")
    surf_env = col(hs.env, "Surface")
    weight_code = np.round(weight_env / 165.0)
    surf_code = np.round(surf_env)
    rises, falls = edges(led)
    pos = {f"P{a}{i}": col(hs.kin, f"P{a}{i} - position {a} sensor {i}") for a in "xyz" for i in range(1, 5)}
    P = {i: np.stack([pos[f"Px{i}"], pos[f"Py{i}"], pos[f"Pz{i}"]], 1) for i in range(1, 5)}
    lf = col(hs.kin, "FX1 - force x plate 1") + col(hs.kin, "FX2 - force x plate 2")
    gf = -(col(hs.kin, "FZ1 - force z plate 1") + col(hs.kin, "FZ2 - force z plate 2")) / 2
    d_index = np.linalg.norm(P[2] - P[1], axis=1)
    d_thumb = np.linalg.norm(P[3] - P[1], axis=1)
    d_digit = np.minimum(d_index, d_thumb)
    aperture = np.linalg.norm(P[2] - P[3], axis=1)
    wrist_speed = np.full(n, np.nan)
    wrist_speed[20:] = np.linalg.norm(P[4][20:] - P[4][:-20], axis=1) / 0.04

    wins = np.atleast_1d(ws.win)
    if (name_list(ws.names.eeg) != name_list(hs.eeg.names)
            or name_list(ws.names.emg) != name_list(hs.emg.names)):
        raise ValueError(f"P{p} S{s}: WS and HS EEG/EMG channel orders differ")
    ws_kin_names = name_list(ws.names.kin)
    shared = [nm for nm in kin_names if nm in ws_kin_names]
    hs_idx = [kin_names.index(nm) for nm in shared]
    ws_idx = [ws_kin_names.index(nm) for nm in shared]
    probe = [shared.index(nm) for nm in ("FX1 - force x plate 1", "Pz1 - position z sensor 1",
                                         "Px4 - position x sensor 4", "FZ1 - force z plate 1")]
    kin_shared = kin[:, hs_idx]
    eeg_probe = list(range(0, 32, 4))
    trials, reach_runs, reach_kinks = [], [], []
    for L, w in enumerate(wins):
        tst = float(w.trial_start_time)
        led_on, led_off = float(w.LEDon), float(w.LEDoff)
        wk = np.asarray(w.kin, dtype=float)[:, ws_idx]
        we = np.asarray(w.eeg, dtype=float)
        wm = np.asarray(w.emg, dtype=float)
        c = int(round(FS * tst))
        n0k, rk = ncc_offset(kin_shared[:, probe], wk[:, probe], 1000, 500, c - 1500, c + 500)
        n0e, re = ncc_offset(eeg[:, eeg_probe], we[:, eeg_probe], 1000, 500, c - 1500, c + 500)
        n0m, rm = ncc_offset(emg, wm, 8000, 2000, 8 * c - 12000, 8 * c + 4000)
        eeg_t = np.atleast_1d(w.eeg_t).astype(float)
        rec = {"lift": L + 1, "trial_start_time": tst, "LEDon": led_on, "LEDoff": led_off,
               "weight": int(w.weight), "weight_id": str(w.weight_id), "surf": int(w.surf),
               "surf_id": str(w.surf_id), "weight_prev": int(w.weight_prev), "surf_prev": int(w.surf_prev),
               "win_rows": len(wk), "eeg_t0": float(eeg_t[0]), "eeg_t_step": float(np.median(np.diff(eeg_t))),
               "emg_t0": float(np.atleast_1d(w.emg_t)[0]),
               "n0_kin": n0k, "r_kin": rk, "n0_eeg": n0e, "r_eeg": re, "n0_emg": n0m, "r_emg": rm,
               "n0_kin_minus_500tst": n0k - c,
               "kin_match": full_match(kin_shared, wk, n0k), "eeg_match": full_match(eeg, we, n0e),
               "emg_match": full_match(emg, wm, n0m)}
        if len(rises):
            rise = int(rises[np.argmin(np.abs(rises - (n0k + 1000)))])
            later = falls[falls > rise]
            rec["hs_rise"] = rise
            rec["rise_minus_n0_kin"] = rise - n0k
            if len(later) and rise >= 600 and later[0] + 10 < n:
                fall = int(later[0])
                rest = slice(rise - 500, rise - 100)
                hold = slice(fall - 300, fall - 100)
                prev = falls[falls < rise]
                start = int(prev[-1]) if len(prev) else 0
                ch = np.flatnonzero(np.diff(weight_code[start:rise + 1]) != 0)
                sch = np.flatnonzero(np.diff(surf_code[start:rise + 1]) != 0)
                rec.update({
                    "hs_fall": fall, "hs_high_s": (fall - rise) / FS, "ws_led_s": led_off - led_on,
                    "fall_minus_expected": fall - (n0k + int(round(FS * led_off)) - 1),
                    "weight_env_at_rise": float(weight_env[rise]), "surf_env_at_rise": float(surf_env[rise]),
                    "weight_code_constant_on_interval": bool(np.ptp(weight_code[rise:fall]) == 0),
                    "weight_code_last_change_before_rise_s": None if not len(ch) else (rise - (start + int(ch[-1]) + 1)) / FS,
                    "surf_code_last_change_before_rise_s": None if not len(sch) else (rise - (start + int(sch[-1]) + 1)) / FS,
                    "button_press_since_prev_fall": bool(np.any(button[start:rise] > 0.5)),
                    "LF_rest": float(np.nanmean(lf[rest])), "LF_hold_raw": float(np.nanmean(lf[hold])),
                    "GF_rest": float(np.nanmean(gf[rest])), "GF_hold_raw": float(np.nanmean(gf[hold])),
                    "lift_height": float(np.nanmean(pos["Pz1"][hold]) - np.nanmean(pos["Pz1"][rest])),
                    "aperture_hold": float(np.nanmean(aperture[hold])),
                    "d_digit_min_last1s": float(np.nanmin(d_digit[rise - 500:rise])),
                    "wrist_speed_max_pre_rise": float(np.nanmax(wrist_speed[rise - 500:rise])),
                    "wrist_disp_hold_xyz": [float(v) for v in
                                            np.nanmean(P[4][hold], axis=0) - np.nanmean(P[4][rest], axis=0)]})
                rec["LF_hold_net"] = rec["LF_hold_raw"] - rec["LF_rest"]
                rec["GF_hold_net"] = rec["GF_hold_raw"] - rec["GF_rest"]
                for label, dist in (("min", d_digit), ("index", d_index), ("thumb", d_thumb)):
                    below = np.flatnonzero(dist[start:rise] < 15.0)
                    rec[f"delay_since_{label}_below15_s"] = (None if not len(below)
                                                             else (rise - start - int(below[-1]) - 1) / FS)
                sp = wrist_speed[rise:min(rise + 1500, n)]
                ok = np.convolve((sp > 10.0).astype(int), np.ones(10, int), "valid") == 10
                rec["wrist_motion_latency_s"] = float(np.argmax(ok)) / FS if ok.any() else None
                reach = pos["Px4"][rise:min(rise + 1500, n)]
                reach_runs.append(equal_runs(reach))
                kinks = np.flatnonzero(np.abs(np.diff(reach, 2)) > 1e-9)
                reach_kinks.append(np.diff(kinks))
                rec.update(liftoff_lag(lf[rise:fall] - rec["LF_rest"], rec["LF_hold_net"],
                                       pos["Pz1"][rise:fall] - np.nanmean(pos["Pz1"][rest])))
        trials.append(rec)

    mag = np.flatnonzero(np.abs(np.diff(magnet)) > 0.5) + 1
    mag = mag[(mag > 200) & (mag < n - 200)]
    magnet_check = {"edges": int(len(mag))}
    if len(mag) >= 3:
        de = np.abs(np.diff(eeg, axis=0)).sum(1)
        prof = np.stack([de[e - 100:e + 100] for e in mag])
        mean = prof.mean(0)
        base = mean[:50]
        magnet_check.update({
            "peak_lag_samples": int(np.argmax(mean[75:125])) - 25,
            "peak_over_baseline_sd": float((mean[75:125].max() - base.mean()) / (base.std() + 1e-12)),
            "per_edge_lag_samples": [int(np.argmax(r[75:125])) - 25 for r in prof],
            "edge_times_s": [float(e / FS) for e in mag]})
    runs = np.concatenate(reach_runs) if reach_runs else np.array([])
    kink_gaps = np.concatenate(reach_kinks) if reach_kinks else np.array([])
    return {
        "px4_kink_spacing_median": float(np.median(kink_gaps)) if len(kink_gaps) else None,
        "px4_kink_spacing_mode": int(np.bincount(kink_gaps.astype(int)).argmax()) if len(kink_gaps) else None,
        "weight_code_values": sorted(float(v) for v in np.unique(weight_code)),
        "surf_code_values": sorted(float(v) for v in np.unique(surf_code)),
        "participant": p, "series": s, "role": role, "hs_header": hs_header, "ws_header": ws_header,
        "n_samples": n, "duration_s": n / FS, "emg_samples": int(len(emg)),
        "rates": [float(hs.eeg.samplingrate), float(hs.kin.samplingrate), float(hs.emg.samplingrate)],
        "misc_names": name_list(hs.misc.names), "env_names": name_list(hs.env.names),
        "ws_kin_names": ws_kin_names, "shared_kin_channels": len(shared),
        "led_exact_0_1_fraction": float(np.mean((led == 0) | (led == 1))),
        "led_min": float(led.min()), "led_max": float(led.max()),
        "hs_rises": int(len(rises)), "hs_falls": int(len(falls)), "ws_windows": int(len(wins)),
        "weight_env_values": sorted(float(v) for v in np.unique(weight_env)),
        "surf_env_values": sorted(float(v) for v in np.unique(surf_env)),
        "button_presses": int(len(edges(button)[0])),
        "px4_equal_run_median": float(np.median(runs)) if len(runs) else None,
        "px4_equal_run_p90": float(np.percentile(runs, 90)) if len(runs) else None,
        "px4_zero_diff_fraction": float(np.mean(np.diff(pos["Px4"]) == 0)),
        "magnet_check": magnet_check, "trials": trials, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--only", default=None, help="P:S for a single-file smoke run, e.g. 1:1")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run inside a Slurm allocation on a compute node")
    tasks = [(p, s) for p in range(1, 13) for s in SERIES]
    if args.only:
        tasks = [tuple(int(v) for v in args.only.split(":"))]
    if any(s > 7 for _, s in tasks):
        raise SystemExit("Series 8-9 are held out")
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(analyze, tasks))
    with open(args.output, "w") as f:
        json.dump({"analysis": "way_source_fact_checks", "job_id": os.environ["SLURM_JOB_ID"],
                   "host": socket.gethostname(), "series": sorted({s for _, s in tasks}),
                   "held_out_series_touched": False, "alllifts_read": False,
                   "elapsed_s": time.time() - started, "files": files}, f, indent=1)


if __name__ == "__main__":
    main()
