#!/usr/bin/env python3
"""Cue-input inspection: spectral content of development TRIG tone bursts (no outcome data).

For every burst lasting >= 0.25 s, report duration, peak ratio and the dominant
frequency / band-power ratio over its first 250 ms, with a schedule-context
reference class (interval to the previous long burst ~12.0 s -> end, ~7.08 s
-> start). Also reports the sub-15/ses-01/run-0001 bursts near 221.76 s.
"""
import argparse
import json
import os
import socket
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector_readers as R  # noqa: E402
from inspect_detector_noise import trig_bursts  # noqa: E402


def features(trig, fs, onset_s, span=0.25):
    a = int(round(onset_s * fs))
    seg = trig[a:a + int(span * fs)]
    seg = seg - seg.mean()
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win)) ** 2
    freq = np.fft.rfftfreq(len(seg), 1 / fs)
    band = (freq >= 5)
    peak = float(freq[band][np.argmax(spec[band])])
    lo = spec[(freq >= 40) & (freq < 70)].sum()
    hi = spec[(freq >= 70) & (freq < 100)].sum()
    sign = np.signbit(seg)
    zcr = float(np.mean(sign[1:] != sign[:-1]) * fs / 2)
    return peak, float(np.log10((lo + 1e-12) / (hi + 1e-12))), zcr


def run(stem):
    r = R.load_self_paced_run(stem)
    bursts, ref = trig_bursts(r.trig_uv, r.fs)
    rows, prev = [], None
    for b in bursts:
        if b["duration_s"] < 0.25:
            if r.stem.startswith("sub-15_ses-01") and r.run == 1 and 215 < b["onset_s"] < 230:
                rows.append({"onset_s": b["onset_s"], "duration_s": b["duration_s"], "short": True,
                             "peak_ratio": b["peak_ratio"]})
            continue
        peak, lratio, zcr = features(r.trig_uv, r.fs, b["onset_s"])
        ref_class = None
        if prev is not None:
            gap = b["onset_s"] - prev
            ref_class = "end" if abs(gap - 12.0) <= 0.3 else "start" if abs(gap - 7.08) <= 0.3 else None
        rows.append({"onset_s": b["onset_s"], "duration_s": b["duration_s"], "peak_ratio": b["peak_ratio"],
                     "peak_hz": peak, "log_40_70_over_70_100": lratio, "zcr_hz": zcr, "ref_class": ref_class})
        prev = b["onset_s"]
    return {"stem": r.stem, "fs": r.fs, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run inside a Slurm allocation")
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(run, R.development_stems()))
    summary = {}
    for fs in (250.0, 1000.0):
        for cls in ("start", "end"):
            rows = [x for r in runs if r["fs"] == fs for x in r["rows"] if x.get("ref_class") == cls]
            if not rows:
                continue
            summary[f"{int(fs)}_{cls}"] = {
                k: np.percentile([x[k] for x in rows], [0, 1, 5, 50, 95, 99, 100]).round(3).tolist()
                for k in ("duration_s", "peak_ratio", "peak_hz", "log_40_70_over_70_100", "zcr_hz")}
            summary[f"{int(fs)}_{cls}"]["n"] = len(rows)
    with open(args.output, "w") as f:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "summary": summary, "runs": runs}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
