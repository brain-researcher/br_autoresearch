#!/usr/bin/env python3
"""attempt004: donor-pool census for the CEBSIT whole-cortex release, catalog only.

Reads the portal's public sample and neuron catalogs. Keeps only
non-axon fields: soma acronym, hemisphere, reconstruction type, ids/names,
and sample metadata. `class` and `type_array` are derived from projections and
are dropped on load without being inspected. No SWC or projection file is read.
Usage: survey_cebsit_catalog.py SAMPLE_JSON NEURON_JSON STRUCTURE_GRAPH_JSON OUT_JSON
"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import socket
import sys

NEURON_KEEP = ("acronym", "file", "hemisphere", "id", "name", "reconstruction_type")
IT_LINES = {"Cux2": {"L2/3", "L4"}, "PlexinD1": {"L2/3", "L5"}, "Plxnd1": {"L2/3", "L5"},
            "Tlx3": {"L5"}, "Rorb": {"L4", "L5"}, "Wfs1": {"L2/3"}, "Scnn1a": {"L4"}}
SCOPE = {"L2/3", "L4", "L5"}


def load_graph(path):
    graph = json.loads(Path(path).read_text())
    nodes, stack = {}, [(n, "/") for n in graph["msg"]]
    while stack:
        n, p = stack.pop()
        n["path"] = p + f"{n['id']}/"
        nodes[n["acronym"]] = n
        stack.extend((c, n["path"]) for c in n.get("children", []))
    return nodes


def layer_of(node):
    if node is None or "/315/" not in node["path"] or "ayer" not in node["name"]:
        return None
    tok = node["name"].rsplit("ayer", 1)[1].strip()
    return {"1": "L1", "2": "L2/3", "3": "L2/3", "2/3": "L2/3", "4": "L4", "5": "L5",
            "6a": "L6", "6b": "L6", "6": "L6"}.get(tok, "L" + tok)


def line_of(genotype):
    g = (genotype or "").lower()
    for k in IT_LINES:
        if k.lower() in g:
            return "PlexinD1" if k == "Plxnd1" else k
    return None


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    sample_json, neuron_json, graph_json, out = map(Path, sys.argv[1:5])
    samples = json.loads(sample_json.read_text())
    raw = json.loads(neuron_json.read_text())["neuron_data"]
    neurons = [{k: v.get(k) for k in NEURON_KEEP} for v in raw.values()]  # class/type_array dropped
    del raw
    nodes = load_graph(graph_json)

    by_fmost = defaultdict(list)
    for s in samples:
        by_fmost[str(s.get("fMOST_id"))].append(s)
    sample_ids = Counter(str(s.get("sample_id")) for s in samples)
    shared_sample_ids = {k: v for k, v in sample_ids.items() if v > 1 and k not in ("无", "NA", "", "None")}
    lines_all = Counter(str(s.get("transgenic_line")) for s in samples)
    virus_by_line = defaultdict(Counter)
    region_by_line = defaultdict(Counter)
    for s in samples:
        ln = line_of(s.get("transgenic_line"))
        if ln:
            virus_by_line[ln][str(s.get("injection_virus"))] += 1
            region_by_line[ln][str(s.get("injection_region"))] += 1

    def brain_of(n):
        for key in ("name", "file"):
            m = re.match(r"(\d+)", str(n.get(key) or ""))
            if m:
                return m.group(1)
        return None

    rtypes = Counter(str(n.get("reconstruction_type")) for n in neurons)
    pools, eligible, unmatched = Counter(), Counter(), 0
    l1 = Counter()
    for n in neurons:
        b = brain_of(n)
        recs = by_fmost.get(b)
        if not recs:
            unmatched += 1
            continue
        ln = line_of(recs[0].get("transgenic_line"))
        if ln is None or str(n.get("reconstruction_type")) != "Axon_and_dendrite":
            continue
        node = nodes.get(str(n.get("acronym")))
        lay = layer_of(node)
        if lay == "L1":
            l1[ln] += 1
        if lay not in IT_LINES[ln]:
            continue
        area = nodes.get(str(node["parent_structure_id"])) if False else None
        parent = next((x for x in nodes.values() if x["id"] == node["parent_structure_id"]), None)
        area = parent["acronym"] if parent else n.get("acronym")
        eligible[(ln, b)] += 1
        pools[(ln, b, area, lay)] += 1

    per_line = {}
    for ln in sorted({k[0] for k in eligible}):
        animals = {b for (l, b) in eligible if l == ln}
        best = {b: max(v for k, v in pools.items() if k[0] == ln and k[1] == b) for b in animals}
        per_line[ln] = {"animals_with_eligible_cells": len(animals),
                        "eligible_cells": sum(v for (l, b), v in eligible.items() if l == ln),
                        "animals_with_pool_ge5": sum(v >= 5 for v in best.values()),
                        "animals_with_pool_ge10": sum(v >= 10 for v in best.values()),
                        "largest_pool_by_animal": dict(sorted(best.items(), key=lambda x: -x[1]))}
    report = {
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "fields_used": {"neuron": list(NEURON_KEEP), "dropped_unread": ["class", "type_array"]},
        "samples": len(samples), "unique_fMOST_ids": len(by_fmost),
        "fMOST_ids_with_multiple_sample_rows": sum(len(v) > 1 for v in by_fmost.values()),
        "sample_ids_shared_by_multiple_rows": len(shared_sample_ids),
        "transgenic_lines": dict(lines_all.most_common()),
        "injection_virus_by_IT_line": {k: dict(v.most_common(6)) for k, v in virus_by_line.items()},
        "injection_region_by_IT_line": {k: dict(v.most_common(8)) for k, v in region_by_line.items()},
        "neurons": len(neurons), "neurons_without_sample_row": unmatched,
        "reconstruction_types": dict(rtypes), "IT_line_axon_and_dendrite_cells_in_L1": dict(l1),
        "per_line": per_line,
        "animals_with_pool_ge5_any_IT_line": len({b for ln, v in per_line.items() for b, m in v["largest_pool_by_animal"].items() if m >= 5}),
        "animals_with_pool_ge5_excluding_Wfs1_Scnn1a": len({b for ln, v in per_line.items() if ln not in ("Wfs1", "Scnn1a")
                                                            for b, m in v["largest_pool_by_animal"].items() if m >= 5}),
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_line"}, indent=1, ensure_ascii=False))
    for ln, v in per_line.items():
        print(ln, {k: v[k] for k in ("animals_with_eligible_cells", "eligible_cells", "animals_with_pool_ge5", "animals_with_pool_ge10")})


if __name__ == "__main__":
    main()
