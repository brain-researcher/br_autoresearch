#!/usr/bin/env python3
"""Self-paced reaching development source-fact checks (EP19, October 16, 2026).

Scope: the 15 development participants only (155 raw runs in the job-46425841
development view) plus the development derivative file for sub-03/ses-02.
No held-out participant file is opened. EEG is not analysed.

Pre-declared predictions and decision rules are in
outputs/source_timing_handoff.md (October 16 section).
"""
import argparse
import io
import json
import os
import re
import socket
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import scipy.io as sio

DEVELOPMENT = ("sub-02", "sub-03", "sub-04", "sub-06", "sub-08", "sub-09", "sub-11", "sub-12",
               "sub-14", "sub-15", "sub-17", "sub-18", "sub-20", "sub-21", "sub-22")
VIEW = os.path.join(os.environ.get("SCRATCH", ""), "br_autoresearch/episode19_causal_motor_forecasting_benchmark/"
                    "development_view_job_46425841/self_paced/development")
ARCHIVE = os.path.join(os.environ.get("OAK", ""), "users/zijiao/br_autoresearch_data/steward_acquisition/"
                       "ep19_public_sources/archives/freewill/Freewill_EEG_Reaching_Grasping.zip")
DERIVATIVE = "derivatives/matfiles/sub-03/ses-02/sub-03_ses-02_task-reachingandgrasping_eeg.mat"
G_VOLT = 1.45e6  # manufacturer 3D sensor sensitivity, microvolts per g (typical, +/-5%)


def read_vhdr(path):
    text = open(path, encoding="utf-8").read()
    section, info, chans = None, {}, []
    for line in text.splitlines():
        if line.startswith("["):
            section = line.strip()
            continue
        if "=" not in line or line.startswith(";"):
            continue
        key, value = line.split("=", 1)
        if section == "[Common Infos]" or section == "[Binary Infos]":
            info[key.strip()] = value.strip()
        elif section == "[Channel Infos]" and key.startswith("Ch"):
            parts = value.split(",")
            chans.append((parts[0], float(parts[2]) if len(parts) > 2 and parts[2] else 1.0))
    return info, chans


def read_vmrk(path):
    text = open(path, encoding="utf-8").read()
    datafile = re.search(r"^DataFile=(.*)$", text, re.M).group(1).strip()
    marks = re.findall(r"^Mk\d+=(.*)$", text, re.M)
    return datafile, [m.split(",") for m in marks]


def stamp(value):
    return datetime.strptime(value[:20], "%Y%m%d%H%M%S%f").timestamp()


def longest_constant(x):
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate([[-1], change, [len(x) - 1]])
    return int(np.max(np.diff(edges)))


def trig_cues(trig, fs):
    d = np.abs(np.diff(trig))
    noise = float(np.median(d))
    w = max(1, int(0.02 * fs))
    env = np.convolve(d, np.ones(w) / w, "same")
    hot = env > 20 * noise
    starts = np.flatnonzero(hot[1:] & ~hot[:-1]) + 1
    onsets = []
    for s0 in starts:
        if not onsets or s0 - onsets[-1] >= 3 * fs:
            onsets.append(int(s0))
    onsets = np.array(onsets)
    gaps = np.diff(onsets) / fs
    kinds = ["start_end" if 11.5 <= g <= 12.5 else "end_start" if 6.5 <= g <= 7.6 else "irregular"
             for g in gaps]
    return noise, onsets, gaps, kinds


def static_vectors(acc, fs):
    w = int(0.5 * fs)
    m = len(acc) // w
    blocks = acc[:m * w].reshape(m, w, 3)
    sd = blocks.std(1)
    keep = np.all(sd < 5000.0, axis=1)
    return blocks.mean(1)[keep], sd


def sphere_fit(v):
    a = np.column_stack([2 * v, np.ones(len(v))])
    b = np.sum(v * v, 1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(sol[3] + center @ center))
    resid = np.linalg.norm(v - center, axis=1) - radius
    u = (v - center) / np.linalg.norm(v - center, axis=1, keepdims=True)
    spread = float(np.degrees(np.arccos(np.clip(np.min(u @ u.mean(0) / np.linalg.norm(u.mean(0))), -1, 1))))
    return {"center_uV": [float(c) for c in center], "radius_uV": radius,
            "radius_over_1g_spec": radius / G_VOLT, "rms_resid_over_radius": float(np.sqrt(np.mean(resid ** 2)) / radius),
            "max_angle_from_mean_deg": spread, "vectors": int(len(v))}


def analyze_run(stem):
    info, chans = read_vhdr(stem + ".vhdr")
    names = [c[0] for c in chans]
    res = np.array([c[1] for c in chans])
    nch = int(info["NumberOfChannels"])
    fs = 1e6 / float(info["SamplingInterval"])
    raw = np.fromfile(stem + ".eeg", dtype="<f4")
    if raw.size % nch:
        return {"stem": os.path.basename(stem), "error": "size not a multiple of channel count"}
    data = raw.reshape(-1, nch)
    datafile, marks = read_vmrk(stem + ".vmrk")
    seg = [m for m in marks if m[0] == "New Segment"]
    trig = data[:, names.index("TRIG")] * res[names.index("TRIG")]
    acc = np.column_stack([data[:, names.index(a)] * res[names.index(a)] for a in "XYZ"]).astype(float)
    noise, onsets, gaps, kinds = trig_cues(trig.astype(float), fs)
    vec, sd = static_vectors(acc, fs)
    base = os.path.basename(stem)
    out = {"stem": base, "fs": fs, "channels": nch, "frames": int(len(data)), "duration_s": len(data) / fs,
           "vhdr_datafile_ok": info.get("DataFile") == base + ".eeg",
           "vmrk_datafile": datafile, "vmrk_datafile_ok": datafile == base + ".eeg",
           "markers": len(marks), "new_segments": len(seg),
           "new_segment_position": int(seg[0][2]) if seg else None,
           "start_stamp": stamp(seg[0][5]) if seg and len(seg[0]) > 5 else None,
           "trig_noise_uV": noise, "cue_onsets": int(len(onsets)),
           "cue_onset_s": [float(o / fs) for o in onsets], "cue_gaps_s": [float(g) for g in gaps],
           "cue_gap_kinds": kinds,
           "static_vectors_uV": vec.tolist(),
           "axis_std_uV": [float(v) for v in acc.std(0)],
           "axis_rest_std_uV": [float(v) for v in np.median(sd, 0)],
           "acc_abs_max_uV": float(np.abs(acc).max()), "acc_norm_median_uV": float(np.median(np.linalg.norm(acc, axis=1))),
           "longest_constant_samples": {"eeg_ch1": longest_constant(data[:, 0]), "TRIG": longest_constant(data[:, names.index("TRIG")]),
                                        "X": longest_constant(data[:, names.index("X")])}}
    return out


def derivative_binding():
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read(DERIVATIVE)
    m = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)["data"]
    del raw
    table = m.dataTable
    header = [str(h) for h in table[0]]
    rows = table[1:-1]
    colv = lambda name: [r[header.index(name)] for r in rows]
    runs_col = [int(v) for v in colv("Run")]
    discard = [int(v) for v in colv("TrialDiscardIndex")]
    raw_eeg = np.atleast_1d(m.rawEEG)
    cnames = [str(c) for c in np.atleast_1d(m.channelName)]
    candidates = {}
    for r in range(7, 14):
        stem = os.path.join(VIEW, "sub-03/ses-02/eeg", f"sub-03_ses-02_task-reachingandgrasping_run-{r:04d}_eeg")
        info, chans = read_vhdr(stem + ".vhdr")
        data = np.fromfile(stem + ".eeg", dtype="<f4").reshape(-1, int(info["NumberOfChannels"]))
        candidates[r] = (data, [c[0] for c in chans], np.array([c[1] for c in chans]))
    runs = []
    for k, block in enumerate(raw_eeg, start=1):
        block = np.asarray(block, dtype=float)
        if block.shape[0] != len(cnames):
            block = block.T
        best = []
        for r, (data, names, res) in candidates.items():
            entry = {"raw_run": r, "raw_frames": int(len(data)), "frames_equal": len(data) == block.shape[1]}
            if entry["frames_equal"]:
                for ch in ("TRIG", "X", "Fz"):
                    x = data[:, names.index(ch)].astype(float)
                    y = block[cnames.index(ch)]
                    entry[f"{ch}_max_abs_diff_scaled"] = float(np.max(np.abs(y - x * res[names.index(ch)])))
                    entry[f"{ch}_max_abs_diff_unscaled"] = float(np.max(np.abs(y - x)))
            best.append(entry)
        idx = [i for i, rr in enumerate(runs_col) if rr == k]
        runs.append({"derivative_run": k, "frames": int(block.shape[1]),
                     "trials": len(idx), "discarded_trials": int(sum(discard[i] for i in idx)),
                     "raw_candidates": best})
    return {"file": DERIVATIVE, "numberOfRuns": int(m.numberOfRuns), "fs": float(m.fs),
            "string_columns_note": "EEGFileNumber and Comments are MATLAB string objects (MCOS) "
                                   "that scipy cannot decode; runs are bound by unique frame counts and samples.",
            "header": header, "channel_count": len(cnames), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--only", default=None, help="session prefix for a smoke run, e.g. sub-03_ses-02")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run inside a Slurm allocation on a compute node")
    stems = []
    for sub in DEVELOPMENT:
        if args.only and not args.only.startswith(sub):
            continue
        for ses in sorted(os.listdir(os.path.join(VIEW, sub))):
            folder = os.path.join(VIEW, sub, ses, "eeg")
            for f in sorted(os.listdir(folder)):
                if f.endswith("_eeg.vhdr") and (not args.only or f.startswith(args.only)):
                    stems.append(os.path.join(folder, f[:-5]))
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(analyze_run, stems))
    sessions = {}
    for r in runs:
        key = r["stem"].split("_task")[0]
        sessions.setdefault(key, []).append(r)
    timeline, spheres = {}, {}
    for key, rs in sessions.items():
        rs.sort(key=lambda r: r["stem"])
        rows = [(r["stem"][-8:-4], r["start_stamp"], r["start_stamp"] + r["duration_s"] if r["start_stamp"] else None)
                for r in rs]
        timeline[key] = {"runs": [a for a, _, _ in rows],
                         "between_run_gaps_s": [None if (b1 is None or a2 is None) else a2 - b1
                                                for (_, _, b1), (_, a2, _) in zip(rows[:-1], rows[1:])]}
        vec = np.array([v for r in rs for v in r["static_vectors_uV"]])
        spheres[key] = sphere_fit(vec) if len(vec) >= 10 else {"vectors": int(len(vec))}
    spheres["all_development_sessions"] = sphere_fit(np.array([v for r in runs for v in r["static_vectors_uV"]]))
    for r in runs:
        r["static_vector_count"] = len(r.pop("static_vectors_uV"))
    report = {"analysis": "self_paced_source_fact_checks", "job_id": os.environ["SLURM_JOB_ID"],
              "host": socket.gethostname(), "participants": list(DEVELOPMENT), "held_out_opened": False,
              "runs": runs, "timeline": timeline, "accelerometer_sphere_fits": spheres,
              "sub03_ses02_derivative": derivative_binding(), "elapsed_s": time.time() - started}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
