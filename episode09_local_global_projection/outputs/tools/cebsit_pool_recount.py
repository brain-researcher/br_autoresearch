#!/usr/bin/env python3
"""attempt005: donor-pool recount on CEBSIT cells that pass dendrite QC.

Input: dendrite_qc_cells.csv (non-axon fields only). A pool is animal x line x
CCF area x layer; only the line's target layers count; L1 is excluded.
Usage: cebsit_pool_recount.py QC_CSV STRUCTURE_GRAPH_JSON OUT_JSON
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import socket
import sys

TARGETS = {"Cux2": {"L2/3", "L4"}, "PlexinD1": {"L2/3", "L5"}, "Tlx3": {"L5"}, "Rorb": {"L4", "L5"},
           "Wfs1": {"L2/3"}, "Scnn1a": {"L4"}}
SCENARIOS = {
    "approved_core_after_cre_rule": ["Cux2"],
    "plus_Wfs1_needs_approval": ["Cux2", "Wfs1"],
    "plus_Wfs1_Scnn1a_needs_approval": ["Cux2", "Wfs1", "Scnn1a"],
    "post_hoc_all_six_lines_descriptive_only": list(TARGETS),
}


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    qc_csv, graph_json, out = map(Path, sys.argv[1:4])
    graph = json.loads(graph_json.read_text())
    by_acr, by_id, stack = {}, {}, [(n, "/") for n in graph["msg"]]
    while stack:
        n, p = stack.pop()
        n["path"] = p + f"{n['id']}/"
        by_acr[n["acronym"]] = n
        by_id[n["id"]] = n
        stack.extend((c, n["path"]) for c in n.get("children", []))

    def layer_area(acr):
        node = by_acr.get(acr)
        if node is None or "/315/" not in node["path"] or "ayer" not in node["name"]:
            return None, None
        tok = node["name"].rsplit("ayer", 1)[1].strip()
        lay = {"1": "L1", "2": "L2/3", "3": "L2/3", "2/3": "L2/3", "4": "L4", "5": "L5"}.get(tok, "L6")
        return lay, by_id.get(node["parent_structure_id"], {}).get("acronym")

    rows = list(csv.DictReader(open(qc_csv)))
    pools = Counter()
    for r in rows:
        if r["passes"] != "True":
            continue
        lay, area = layer_area(r["acronym"])
        if lay in TARGETS.get(r["line"], set()):
            pools[(r["line"], r["brain"], area, lay)] += 1
    report = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "qc_pass_cells": sum(r["passes"] == "True" for r in rows), "qc_cells": len(rows), "scenarios": {}}
    for name, lines in SCENARIOS.items():
        best = Counter()
        for (ln, b, a, l), n in pools.items():
            if ln in lines:
                best[b] = max(best[b], n)
        report["scenarios"][name] = {
            "lines": lines, "animals_with_pool_ge5": sum(v >= 5 for v in best.values()),
            "animals_with_pool_ge10": sum(v >= 10 for v in best.values()),
            "cells_in_pools_ge5": sum(n for (ln, b, a, l), n in pools.items() if ln in lines and n >= 5),
            "meets_registered_20": sum(v >= 5 for v in best.values()) >= 20}
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
