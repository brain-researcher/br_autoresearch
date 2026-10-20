#!/usr/bin/env python3
"""Case-series design feasibility: do T9 and T11 support two chronological partitions?

Rule fixed before running: take the passing P0 radial-8 group's capable sessions,
split them at the count median (early = first ceil(n/2) days), and apply the
unchanged adopted component gates to the chronological near/long pairs inside
each partition. Structural metadata only; no neural, cursor or success values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket

from apply_source_fact_support import BASE, evaluate, load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--support-config", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Compute allocation required")
    support_config = json.loads(args.support_config.read_text())
    report = json.loads(args.support_report.read_text())
    passing = {g["participant"]: g for g in report["variants"]["P0"]["groups_with_designs"]
               if any(d["conditional_numerical_pass"] for d in g["designs"])}
    records = [json.loads(line) for line in args.structural_jsonl.open() if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = {"rule": "count-median chronological split of the passing P0 group's capable sessions; unchanged gates",
           "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "participants": {}}
    for participant in ("T9", "T11"):
        days = sorted(passing[participant]["capable_session_days"])
        cut = math.ceil(len(days) / 2)
        result = {}
        for name, part in (("early", days[:cut]), ("late", days[cut:])):
            keep = set(part)
            subset = [load_session(r, BASE) for r in records
                      if r["summary"]["participant"] == participant and int(r["summary"]["post_implant_day"]) in keep]
            _, groups, _, _ = evaluate(subset, BASE, support_config)
            designs = [d for g in groups if g["family"] == passing[participant]["family"] for d in g["designs"]
                       if d["near_pairs"] and d["long_pairs"]]
            result[name] = {"days": [part[0], part[-1]], "capable_sessions": len(part),
                            "both_lag_components": [{k: d[k] for k in ("near_pairs", "long_pairs", "design_rank",
                                "design_columns", "weighted_column_normalized_condition", "residual_L2",
                                "retained_energy_fraction", "maximum_contrast_node_share",
                                "conditional_numerical_pass", "failed_gates")} for d in designs]}
        out["participants"][participant] = result
        print(json.dumps({participant: {k: [(c["near_pairs"], c["long_pairs"], c["conditional_numerical_pass"],
              c["failed_gates"]) for c in v["both_lag_components"]] for k, v in result.items()}}), flush=True)
    (args.output_dir / "time_partition_support.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
