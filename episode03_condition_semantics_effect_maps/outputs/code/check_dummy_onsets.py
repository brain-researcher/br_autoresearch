#!/usr/bin/env python3
"""Do ds003545 event onsets count from the first raw volume, including dummies?

The producer trimmed `dummy_volumes` (4) from BOLD and confounds without
shifting onsets. Reads only OpenNeuro events TSVs and BOLD JSON sidecars (model
inputs) plus the producer's raw volume counts; opens no image or header.
If events end after (n_raw - dummies) * TR, onsets must count from the raw start,
so the fitted regressors lag the trimmed data by dummies * TR.
"""

import argparse
import csv
import json
import os
import socket
from pathlib import Path


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--basic-details", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    details = json.loads(args.basic_details.read_text())["Tasks"]
    report = {"check": "ds003545_dummy_onsets", "job_id": os.environ.get("SLURM_JOB_ID"),
              "hostname": socket.gethostname(), "images_or_headers_opened": False, "tasks": {}}
    for task in args.tasks:
        info = details[task]
        n_raw, dummies = info.get("bold_volumes"), int(info.get("dummy_volumes") or 0)
        top = load_json(args.raw_root / f"task-{task}_bold.json") or {}
        files = sorted(args.raw_root.glob(f"sub-*/ses-*/func/*_task-{task}_*events.tsv")) + \
            sorted(args.raw_root.glob(f"sub-*/func/*_task-{task}_*events.tsv"))
        rows = []
        for f in files:
            side = load_json(f.with_name(f.name.replace("_events.tsv", "_bold.json"))) or {}
            tr = side.get("RepetitionTime", top.get("RepetitionTime"))
            try:
                with open(f, newline="") as handle:
                    ev = list(csv.DictReader(handle, delimiter="\t"))
                onsets = [float(r["onset"]) for r in ev if r.get("onset") not in (None, "", "n/a")]
                ends = [float(r["onset"]) + float(r["duration"]) for r in ev
                        if r.get("onset") not in (None, "", "n/a") and r.get("duration") not in (None, "", "n/a")]
            except (OSError, ValueError, KeyError):
                rows.append({"file": f.name, "error": "unreadable"})
                continue
            if not ends or tr is None or n_raw is None:
                rows.append({"file": f.name, "error": "missing onsets, TR or volume count"})
                continue
            rows.append({"file": f.name, "tr": tr, "first_onset": min(onsets), "last_end": max(ends),
                         "raw_span": n_raw * tr, "trimmed_span": (n_raw - dummies) * tr,
                         "ends_after_trimmed_span": max(ends) > (n_raw - dummies) * tr + 1e-6})
        ok = [r for r in rows if "error" not in r]
        report["tasks"][task] = {
            "n_raw_volumes": n_raw, "dummy_volumes": dummies, "events_files": len(files),
            "readable": len(ok), "trs": sorted({r["tr"] for r in ok}),
            "runs_ending_after_trimmed_span": sum(r["ends_after_trimmed_span"] for r in ok),
            "max_last_end": max((r["last_end"] for r in ok), default=None),
            "min_first_onset": min((r["first_onset"] for r in ok), default=None),
            "trimmed_span": ok[0]["trimmed_span"] if ok else None, "raw_span": ok[0]["raw_span"] if ok else None,
            "errors": [r for r in rows if "error" in r][:5]}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["tasks"], indent=1))


if __name__ == "__main__":
    main()
