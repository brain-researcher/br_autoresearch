#!/usr/bin/env python3
"""Soma-only check of the CEBSIT SWC coordinate convention (no axon read).

For each cell in the EP09 cell manifest, reads only the soma (type 1) row of its
SWC, looks the coordinate up in the 25 um CCFv3 annotation under each axis
permutation and unit scale, and compares with the catalog soma acronym.
Usage: check_swc_soma_convention.py ZIP CELL_MANIFEST_CSV CCF_PAYLOAD_DIR OUT_JSON
"""
import csv
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_ccf_soma_layers import load_graph, read_nrrd  # noqa: E402


def soma_xyz(z, member):
    with z.open(member) as f:
        for raw in f:
            p = raw.split()
            if len(p) >= 7 and p[0][:1] != b"#" and int(float(p[1])) == 1:
                return float(p[2]), float(p[3]), float(p[4])
    return None


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Slurm compute required")
    zpath, manifest, ccf, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    cells = list(csv.DictReader(open(manifest, newline="")))
    _, vol = read_nrrd(ccf / "annotation_25_ccf_2017.nrrd")
    nodes, by_acr = load_graph(ccf / "structure_graph_1.json")
    shape = np.array(vol.shape)
    xyz, acr = [], []
    with zipfile.ZipFile(zpath) as z:
        for c in cells:
            brain, cell = c["cell"].split("_", 1)
            s = soma_xyz(z, f"data_v3/{brain}/{cell}.swc")
            if s is not None:
                xyz.append(s)
                acr.append(f"{c['area']}|{c['layer']}")
    xyz = np.array(xyz)
    trials = {}
    for perm in itertools.permutations(range(3)):
        for scale in (1.0, 10.0, 25.0):
            idx = np.floor(xyz[:, perm] * scale / 25.0).astype(int)  # scale = um per SWC unit
            inside = np.all((idx >= 0) & (idx < shape), axis=1)
            hits_area = hits_exact = 0
            for k in np.flatnonzero(inside):
                node = nodes.get(int(vol[idx[k, 0], idx[k, 1], idx[k, 2]]))
                if node is None:
                    continue
                area, layer = acr[k].split("|")
                parent = nodes.get(node["parent_structure_id"])
                if parent is not None and parent["acronym"] == area:
                    hits_area += 1
                    tok = node["name"].rsplit("ayer", 1)[-1].strip() if "ayer" in node["name"] else ""
                    lay = {"1": "L1", "2": "L2/3", "3": "L2/3", "2/3": "L2/3", "4": "L4", "5": "L5"}.get(tok, "L6")
                    hits_exact += int(lay == layer)
            trials[f"perm={perm},um_per_unit={scale}"] = {"inside": int(inside.sum()),
                                                          "area_match": hits_area, "area_and_layer_match": hits_exact}
    best = max(trials, key=lambda k: trials[k]["area_and_layer_match"])
    rep = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(), "cells": len(cells),
           "somata_read": int(len(xyz)), "coordinate_ranges": [[float(xyz[:, i].min()), float(xyz[:, i].max())] for i in range(3)],
           "best": best, "trials": trials}
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("cells", "somata_read", "coordinate_ranges", "best")} | {"best_result": trials[best]}, indent=1))


if __name__ == "__main__":
    main()
