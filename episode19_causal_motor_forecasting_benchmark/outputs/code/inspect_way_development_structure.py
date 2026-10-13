#!/usr/bin/env python3
"""Deserialize only role-selected development HS files; emit structure, not values.

MATLAB nested metadata cannot be reached through whosmat alone. This reader
therefore acknowledges access to development signal arrays while inspecting
only their shapes, declared sample rates and named channels. No onset/forecast
or held-out signal is computed or opened.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import socket

import numpy as np
from scipy.io import loadmat

from inspect_development_headers import SCIENTIFIC_CHANNELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("source inspection requires coordinator-owned compute")
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    # One explicitly selected fitting file; expanding coverage is a later step.
    for person in (1,):
        for series in (1,):
            role = "fit_and_inner_selection" if series <= 6 else "aggregate_development_feedback"
            source = args.development_view / "way" / role / f"HS_P{person}_S{series}.mat"
            content = loadmat(source, variable_names=["hs"], simplify_cells=True)
            hs = content["hs"]
            if int(hs["participant"]) != person or int(hs["series"]) != series:
                raise ValueError("development filename and source identity disagree")
            modalities = {}
            for modality in ("eeg", "emg", "kin", "env", "misc"):
                group = hs[modality]
                names = [str(name).strip() for name in np.asarray(group["names"], dtype=object).reshape(-1)]
                rate = float(group["samplingrate"])
                shape = list(np.shape(group["sig"]))
                modalities[modality] = {"names": names, "declared_sampling_rate_hz": rate,
                                       "array_shape": shape,
                                       "nominal_duration_seconds": shape[0] / rate if rate > 0 else None,
                                       "named_channel_count_matches": len(shape) == 2 and shape[1] == len(names)}
            rows.append({"participant": person, "series": series, "role": role, "source": str(source),
                         "modalities": modalities,
                         "missing_scientific_eeg_channels": sorted(set(SCIENTIFIC_CHANNELS) - set(modalities["eeg"]["names"]))})
            del content, hs, group
            gc.collect()
        print(f"inspected structure of WAY HS_P{person}_S1.mat only", flush=True)
    report = {
        "analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "status": "development_way_structure_inspected", "series_files": len(rows),
        "development_signal_arrays_deserialized": True, "signal_values_reported": False,
        "held_out_signal_access": False, "onsets_computed": False, "forecast_scores_computed": False,
        "timing_bound_verified": False, "clock_alignment_verified": False,
        "rows": rows,
    }
    (args.output_root / "way_development_structure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
