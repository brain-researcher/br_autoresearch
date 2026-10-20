#!/usr/bin/env python3
"""Post hoc: look for a sample-level discontinuity inside the 4.92-s start->end
interval of sub-15/ses-01/run-0001 (development; candidate ~7.08-s unmarked gap)."""
import json
import os
import sys

import numpy as np

stem = os.path.join(os.environ["SCRATCH"], "br_autoresearch/episode19_causal_motor_forecasting_benchmark/"
                    "development_view_job_46425841/self_paced/development/sub-15/ses-01/eeg/"
                    "sub-15_ses-01_task-reachingandgrasping_run-0001_eeg")
t_start, t_end = float(sys.argv[1]), float(sys.argv[2])
hdr = open(stem + ".vhdr", encoding="utf-8").read().splitlines()
nch = int([l for l in hdr if l.startswith("NumberOfChannels=")][0].split("=")[1])
fs = 1e6 / float([l for l in hdr if l.startswith("SamplingInterval=")][0].split("=")[1])
sec, names, res = None, [], []
for l in hdr:
    if l.startswith("["):
        sec = l.strip()
    elif sec == "[Channel Infos]" and l.startswith("Ch") and "=" in l:
        parts = l.split("=", 1)[1].split(",")
        names.append(parts[0])
        res.append(float(parts[2]))
data = np.fromfile(stem + ".eeg", dtype="<f4").reshape(-1, nch).astype(float) * np.array(res[:nch])
acc = data[:, [names.index(c) for c in "XYZ"]]
eeg = data[:, [i for i, c in enumerate(names[:nch]) if c in ("Fp1", "Fz", "C3", "C4", "Pz", "O1", "Oz", "O2")]]
a, b = int(t_start * fs), int(t_end * fs)
da = np.linalg.norm(np.diff(acc, axis=0), axis=1)
de = np.abs(np.diff(eeg, axis=0)).mean(1)
ref = slice(int(30 * fs), int(200 * fs))
win = slice(a, b)
k = a + int(np.argmax(da[win]))
ke = a + int(np.argmax(de[win]))
lo = lambda x, i: x[max(0, i - int(0.2 * fs)):i].mean(0)
hi = lambda x, i: x[i + 1:i + 1 + int(0.2 * fs)].mean(0)
out = {"fs": fs, "window_s": [t_start, t_end],
       "acc_step_max_uV_in_window": float(da[win].max()), "acc_step_at_s": k / fs,
       "acc_step_p99999_reference_uV": float(np.percentile(da[ref], 99.999)),
       "acc_step_max_reference_uV": float(da[ref].max()),
       "gravity_dir_change_deg_at_max": float(np.degrees(np.arccos(np.clip(
           np.dot(lo(acc, k), hi(acc, k)) / np.linalg.norm(lo(acc, k)) / np.linalg.norm(hi(acc, k)), -1, 1)))),
       "eeg_step_max_uV_in_window": float(de[win].max()), "eeg_step_at_s": ke / fs,
       "eeg_step_max_reference_uV": float(de[ref].max()),
       "eeg_step_p99999_reference_uV": float(np.percentile(de[ref], 99.999))}
print(json.dumps(out, indent=1))
