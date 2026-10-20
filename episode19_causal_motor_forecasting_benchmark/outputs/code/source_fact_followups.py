#!/usr/bin/env python3
"""Post hoc follow-ups to the October 16 source-fact checks (development data only).

Selected after job 48183086 results: (1) WAY FASTRAK velocity spectrum during
reaches (effective update rate / interpolation); (2) WAY env condition-code
timing around the LED rise; (3) self-paced 60-Hz mains-phase continuity as an
unmarked-gap detector. Labelled post hoc in all records.
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
BASE = os.path.join(os.environ.get("SCRATCH", ""), "br_autoresearch/episode19_causal_motor_forecasting_benchmark/"
                    "development_view_job_46425841")
ARCHIVES = os.path.join(os.environ.get("OAK", ""), "users/zijiao/br_autoresearch_data/steward_acquisition/"
                        "ep19_public_sources/archives/way")


def names(v):
    return [str(x).strip() for x in np.atleast_1d(v)]


def col(sub, name):
    return np.asarray(sub.sig, float)[:, names(sub.names).index(name)]


def way_file(task):
    p, s = task
    role = "fit_and_inner_selection" if s <= 6 else "aggregate_development_feedback"
    hs = sio.loadmat(os.path.join(BASE, "way", role, f"HS_P{p}_S{s}.mat"), squeeze_me=True,
                     struct_as_record=False)["hs"]
    with zipfile.ZipFile(os.path.join(ARCHIVES, f"P{p}.zip")) as z:
        ws = sio.loadmat(io.BytesIO(z.read(f"WS_P{p}_S{s}.mat")), squeeze_me=True, struct_as_record=False)["ws"]
    led = col(hs.misc, "ParticipantLED") > 0.5
    rises = np.flatnonzero(led[1:] & ~led[:-1]) + 1
    falls = np.flatnonzero(~led[1:] & led[:-1]) + 1
    wcode = np.round(col(hs.env, "Weight") / 165.0)
    scode = np.round(col(hs.env, "Surface"))
    button = col(hs.misc, "Button")
    win = np.hanning(750)
    psd = {k: np.zeros(376) for k in ("Px4", "Pz1", "FX1")}
    used = 0
    timing = {"w_pre200": 0, "w_pre10": 0, "w_post10": 0, "w_post200": 0, "w_hold": 0,
              "s_pre200": 0, "s_pre10": 0, "s_post10": 0, "s_post200": 0, "s_hold": 0, "n": 0}
    for w in np.atleast_1d(ws.win):
        n0 = int(round(FS * float(w.trial_start_time))) - 1000
        rise = n0 + 1000
        later = falls[falls > rise]
        if rise < 200 or not len(later) or rise + 750 >= len(led):
            continue
        fall = int(later[0])
        hold = fall - 200
        timing["n"] += 1
        for tag, idx in (("pre200", rise - 100), ("pre10", rise - 5), ("post10", rise + 5), ("post200", rise + 100),
                         ("hold", hold)):
            timing[f"w_{tag}"] += int(wcode[idx] == int(w.weight))
            timing[f"s_{tag}"] += int(scode[idx] == int(w.surf))
        for k, name in (("Px4", "Px4 - position x sensor 4"), ("Pz1", "Pz1 - position z sensor 1"),
                        ("FX1", "FX1 - force x plate 1")):
            seg = col(hs.kin, name)[rise:rise + 751]
            v = np.diff(seg) * FS
            v = (v - v.mean()) * win
            psd[k] += np.abs(np.fft.rfft(v)) ** 2
        used += 1
    freqs = np.fft.rfftfreq(750, 1 / FS)
    spec = {}
    for k, pw in psd.items():
        c = np.cumsum(pw) / pw.sum()
        spec[k] = {"f95_hz": float(freqs[np.searchsorted(c, 0.95)]), "f99_hz": float(freqs[np.searchsorted(c, 0.99)]),
                   "f999_hz": float(freqs[np.searchsorted(c, 0.999)]),
                   "power_above_20hz": float(pw[freqs > 20].sum() / pw.sum()),
                   "power_above_60hz": float(pw[freqs > 60].sum() / pw.sum()),
                   "log10_psd_by_5hz": [float(np.log10(pw[(freqs >= a) & (freqs < a + 5)].mean() + 1e-30))
                                        for a in range(0, 250, 5)]}
    b = button
    return {"participant": p, "series": s, "trials_used": used, "spectra": spec, "env_code_timing": timing,
            "button_high_fraction": float(np.mean(b > 0.5)), "button_min": float(b.min()), "button_max": float(b.max()),
            "button_falling_edges": int(np.sum((b[1:] < 0.5) & (b[:-1] >= 0.5)))}


def read_vhdr(path):
    info, chans, section = {}, [], None
    for line in open(path, encoding="utf-8").read().splitlines():
        if line.startswith("["):
            section = line.strip()
        elif "=" in line and not line.startswith(";"):
            key, value = line.split("=", 1)
            if section in ("[Common Infos]", "[Binary Infos]"):
                info[key.strip()] = value.strip()
            elif section == "[Channel Infos]" and key.startswith("Ch"):
                chans.append(value.split(",")[0])
    return info, chans


def lockin(x, fs, f, block):
    """Block-mean complex demodulation of every column of x at frequency f."""
    n = np.arange(len(x))
    z = x * np.exp(-2j * np.pi * f * n / fs)[:, None]
    m = len(z) // block
    return z[:m * block].reshape(m, block, -1).mean(1)


def sp_run(stem):
    info, chans = read_vhdr(stem + ".vhdr")
    nch = int(info["NumberOfChannels"])
    fs = 1e6 / float(info["SamplingInterval"])
    data = np.fromfile(stem + ".eeg", dtype="<f4").reshape(-1, nch)
    eeg_idx = [i for i, c in enumerate(chans[:nch]) if c not in ("TRIG", "X", "Y", "Z") and not c.isdigit()]
    block = int(0.5 * fs)  # 0.5 s spans whole cycles of 56, 60 and 64 Hz, so DC offsets cancel
    x = data[:, eeg_idx].astype(float)
    a60 = lockin(x, fs, 60.0, block)
    noise = 0.5 * (np.abs(lockin(x, fs, 56.0, block)) + np.abs(lockin(x, fs, 64.0, block)))
    snr_all = np.median(np.abs(a60) / (noise + 1e-12), axis=0)
    j = int(np.argmax(snr_all))
    best, best_snr, A, N = eeg_idx[j], float(snr_all[j]), a60[:, j], noise[:, j]
    dphi = np.degrees(np.angle(A[1:] * np.conj(A[:-1])))
    k = 21
    pad = np.pad(dphi, k // 2, mode="edge")
    local = np.array([np.median(pad[j:j + k]) for j in range(len(dphi))])
    resid = (dphi - local + 180) % 360 - 180
    good = (np.abs(A[1:]) > 3 * N[1:]) & (np.abs(A[:-1]) > 3 * N[:-1])
    jumps = np.flatnonzero(good & (np.abs(resid) > 30))
    return {"stem": os.path.basename(stem), "fs": fs, "channel": chans[best], "snr60_median": best_snr,
            "blocks": int(len(A)), "blocks_with_snr": int(good.sum()),
            "resid_sd_deg": float(np.std(resid[good])) if good.any() else None,
            "jumps": [{"t_s": float((j + 1) * block / fs), "resid_deg": float(resid[j])} for j in jumps]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run inside a Slurm allocation on a compute node")
    t0 = time.time()
    way_tasks = [(p, s) for p in range(1, 13) for s in range(1, 8)]
    dev = os.path.join(BASE, "self_paced/development")
    stems = sorted(os.path.join(dev, sub, ses, "eeg", f[:-5]) for sub in os.listdir(dev)
                   for ses in os.listdir(os.path.join(dev, sub))
                   for f in os.listdir(os.path.join(dev, sub, ses, "eeg")) if f.endswith("_eeg.vhdr"))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        way = list(pool.map(way_file, way_tasks))
        sp = list(pool.map(sp_run, stems))
    with open(args.output, "w") as f:
        json.dump({"analysis": "source_fact_followups_post_hoc", "job_id": os.environ["SLURM_JOB_ID"],
                   "host": socket.gethostname(), "way": way, "self_paced": sp, "elapsed_s": time.time() - t0}, f, indent=1)


if __name__ == "__main__":
    main()
