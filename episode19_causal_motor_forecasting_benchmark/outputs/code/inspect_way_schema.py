#!/usr/bin/env python3
"""Describe only field names, types and shapes in one selected development HS file."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

from scipy.io import loadmat


def describe(value, depth=0):
    result = {"python_type": type(value).__name__}
    if isinstance(value, dict):
        result["field_names"] = sorted(value)
        if depth < 2:
            result["fields"] = {key: describe(item, depth + 1) for key, item in value.items()}
    elif hasattr(value, "shape"):
        result["shape"] = list(value.shape)
        result["dtype"] = str(value.dtype)
    elif isinstance(value, (tuple, list)):
        result["length"] = len(value)
    # Never emit scalar or array values, including identifying strings.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator compute is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.development_view / "way/fit_and_inner_selection/HS_P1_S1.mat"
    content = loadmat(source, variable_names=["hs"], simplify_cells=True)
    report = {"analysis_id": "source_views_and_timing", "source": str(source),
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "development_signal_arrays_deserialized": True,
              "signal_or_identity_values_reported": False,
              "held_out_signal_access": False, "forecast_scores_computed": False,
              "hs_schema": describe(content["hs"])}
    (args.output_dir / "way_hs_schema.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
