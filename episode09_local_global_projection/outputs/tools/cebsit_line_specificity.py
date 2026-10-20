#!/usr/bin/env python3
"""attempt005: empirical test that CEBSIT labelling is Cre-dependent and line-specific.

Catalog only: soma CCF acronym (layer) and sample line. The projection-derived
`class`/`type_array` fields are dropped on load. Rules: attempt005/README.md.
Usage: cebsit_line_specificity.py SAMPLE_JSON NEURON_JSON STRUCTURE_GRAPH_JSON OUT_JSON
"""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import socket
import sys

import numpy as np

TARGETS = {"Cux2": {"L2/3", "L4"}, "PlexinD1": {"L2/3", "L5"}, "Tlx3": {"L5"}, "Rorb": {"L4", "L5"},
           "Wfs1": {"L2/3"}, "Scnn1a": {"L4"}, "Ntsr1": {"L6"}}
SEED, NBOOT = 20261016, 2000


def load_graph(path):
    graph = json.loads(Path(path).read_text())
    by_acr, by_id, stack = {}, {}, [(n, "/") for n in graph["msg"]]
    while stack:
        n, p = stack.pop()
        n["path"] = p + f"{n['id']}/"
        by_acr[n["acronym"]] = n
        by_id[n["id"]] = n
        stack.extend((c, n["path"]) for c in n.get("children", []))
    return by_acr, by_id


def layer_of(node):
    if node is None or "/315/" not in node["path"] or "ayer" not in node["name"]:
        return None
    tok = node["name"].rsplit("ayer", 1)[1].strip()
    return {"1": "L1", "2": "L2/3", "3": "L2/3", "2/3": "L2/3", "4": "L4", "5": "L5",
            "6a": "L6", "6b": "L6", "6": "L6"}.get(tok, "L" + tok)


def line_of(g):
    g = (g or "").lower()
    if "c57bl" in g:
        return "WT"
    for k in TARGETS:
        if k.lower() in g:
            return k
    return None


def mh_or(table):
    """Mantel-Haenszel odds ratio over strata of [a, b, c, d] (line in/out, WT in/out)."""
    num = den = 0.0
    for a, b, c, d in table:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        num += a * d / n
        den += b * c / n
    return num / den if den > 0 else float("inf")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    sample_json, neuron_json, graph_json, out = map(Path, sys.argv[1:5])
    samples = {str(s["fMOST_id"]): s for s in json.loads(sample_json.read_text())}
    raw = json.loads(neuron_json.read_text())["neuron_data"]
    cells = [{"acronym": v.get("acronym"), "name": v.get("name"), "file": v.get("file")} for v in raw.values()]
    del raw  # class/type_array never inspected
    by_acr, by_id = load_graph(graph_json)
    rows = []
    for c in cells:
        m = re.match(r"(\d+)", str(c["file"] or "")) or re.match(r"(\d+)", str(c["name"] or ""))
        s = samples.get(m.group(1)) if m else None
        ln = line_of(s.get("transgenic_line")) if s else None
        node = by_acr.get(str(c["acronym"]))
        lay = layer_of(node)
        if ln is None or lay is None:
            continue
        area = by_id.get(node["parent_structure_id"], {}).get("acronym")
        rows.append((ln, m.group(1), area, lay))
    report = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "cells_with_isocortex_layer": len(rows), "lines": {}}
    wt = [r for r in rows if r[0] == "WT"]
    rng = np.random.default_rng(SEED)
    for ln, tgt in TARGETS.items():
        lr = [r for r in rows if r[0] == ln]
        if not lr:
            continue
        areas = sorted({r[2] for r in lr})

        aidx = {a: i for i, a in enumerate(areas)}

        def counts(rs):
            brains = sorted({r[1] for r in rs})
            bidx = {b: i for i, b in enumerate(brains)}
            arr = np.zeros((len(brains), len(areas), 2))
            for r in rs:
                if r[2] in aidx:
                    arr[bidx[r[1]], aidx[r[2]], 0 if r[3] in tgt else 1] += 1
            return arr

        L, W = counts(lr), counts(wt)

        def mh(lsum, wsum):
            return mh_or([(lsum[i, 0], lsum[i, 1], wsum[i, 0], wsum[i, 1]) for i in range(len(areas))])
        est = mh(L.sum(0), W.sum(0))
        boots = []
        for _ in range(NBOOT):
            ls = L[rng.integers(0, len(L), len(L))].sum(0)
            ws = W[rng.integers(0, len(W), len(W))].sum(0)
            boots.append(mh(ls, ws))
        boots = np.array([b for b in boots if np.isfinite(b)])
        lkeys = sorted({r[1] for r in lr})
        lay_counts = Counter(r[3] for r in lr)
        wt_same_area = [r for r in wt if r[2] in areas]
        report["lines"][ln] = {
            "brains": len(lkeys), "cells": len(lr), "target_layers": sorted(tgt),
            "target_fraction": sum(r[3] in tgt for r in lr) / len(lr),
            "wt_target_fraction_same_areas": (sum(r[3] in tgt for r in wt_same_area) / len(wt_same_area)) if wt_same_area else None,
            "L1_fraction": lay_counts["L1"] / len(lr),
            "layer_counts": dict(lay_counts),
            "mh_odds_ratio": est,
            "or_ci95_brain_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else None,
            "passes": bool(est >= 3 and len(boots) and np.percentile(boots, 2.5) > 1)}
    report["positive_control_ntsr1_passes"] = report["lines"].get("Ntsr1", {}).get("passes")
    out.write_text(json.dumps(report, indent=1) + "\n")
    for ln, v in report["lines"].items():
        print(f"{ln:9s} brains {v['brains']:3d} cells {v['cells']:5d} target {v['target_fraction']:.2f} wt {v['wt_target_fraction_same_areas']:.2f} "
              f"L1 {v['L1_fraction']:.2f} OR {v['mh_odds_ratio']:.1f} CI {v['or_ci95_brain_bootstrap']} pass {v['passes']}")
    print("Ntsr1 control passes:", report["positive_control_ntsr1_passes"])


if __name__ == "__main__":
    main()
