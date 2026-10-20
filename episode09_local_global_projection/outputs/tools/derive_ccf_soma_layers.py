#!/usr/bin/env python3
"""attempt003 Stage A: derive soma layers from CCFv3 soma coordinates (rule D25).

Decodes only whitelisted Full_morphometry columns (never K, Projection class),
derives a coordinate-only layer for every cell, validates the axis mapping
against the recorded Soma region, and reports agreement with recorded layers.
No blank is filled here; Stage B applies the pre-registered fill rule.

Usage: derive_ccf_soma_layers.py MORPHOMETRY_XLSX CCF_PAYLOAD_DIR JOIN_JSON SCRATCH_OUT_JSON REPORT_JSON
"""
import gzip
import itertools
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlsx_whitelist import whitelisted_rows  # noqa: E402

COLUMNS = {"A": "cell", "B": "brain", "C": "raw_x", "D": "raw_y", "E": "raw_z",
           "F": "ccf_x", "G": "ccf_y", "H": "ccf_z", "I": "manually_checked",
           "J": "soma_region", "L": "layer_recorded"}  # K deliberately absent
SEED = 20261016
N_BOOT = 2000


def read_nrrd(path):
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").rstrip("\n")
            if line == "":
                break
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip().lstrip("=").strip()
        payload = f.read()
    sizes = [int(s) for s in header["sizes"].split()]
    dtype = {"unsigned int": "<u4", "uint32": "<u4", "uint": "<u4", "unsigned short": "<u2"}[header["type"]]
    if header.get("endian", "little") == "big":
        dtype = dtype.replace("<", ">")
    raw = gzip.decompress(payload) if header["encoding"] == "gzip" else payload
    # NRRD sizes list the fastest axis first: index = i0 + s0*(i1 + s1*i2)
    vol = np.frombuffer(raw, dtype=dtype).reshape(sizes[::-1]).transpose(2, 1, 0)
    return header, vol


def load_graph(path):
    graph = json.loads(Path(path).read_text())
    nodes, stack = {}, [(n, "/") for n in graph["msg"]]
    while stack:
        n, path = stack.pop()
        n["structure_id_path"] = path + f"{n['id']}/"  # computed from parent links
        nodes[n["id"]] = n
        stack.extend((c, n["structure_id_path"]) for c in n.get("children", []))
    by_acr = {n["acronym"]: n for n in nodes.values()}
    return nodes, by_acr


def layer_token(node):
    if node is None or "/315/" not in node["structure_id_path"]:
        return None
    name = node["name"]
    if "ayer" not in name:
        return None
    tok = name.rsplit("ayer", 1)[1].strip()
    return {"1": "L1", "2": "L2/3", "3": "L2/3", "2/3": "L2/3", "4": "L4", "5": "L5",
            "6a": "L6a", "6b": "L6b", "6": "L6"}.get(tok, "L" + tok)


def harmonize(layer):
    if layer is None:
        return None
    return "L6" if layer.startswith("L6") else layer


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    xlsx, ccf, join_json, scratch_out, report_out = map(Path, sys.argv[1:6])
    rows = whitelisted_rows(xlsx, "xl/worksheets/sheet1.xml", COLUMNS.keys(), header_row=1)
    cells = [{COLUMNS[k]: v for k, v in r.items() if k in COLUMNS} | {"_row": r["_row"]}
             for r in rows if r.get("A")]
    header, vol = read_nrrd(ccf / "annotation_25_ccf_2017.nrrd")
    nodes, by_acr = load_graph(ccf / "structure_graph_1.json")
    join = json.loads(join_json.read_text())
    animal_of = {x["image_id"]: x["animal_id"] for x in join["joined"]}
    animal_of["236174"] = "236174"  # authenticated alias, job 46842156 (row 81)
    genotype_of = {x["image_id"]: x["genotype"] for x in join["joined"]}
    genotype_of["236174"] = "Gnb4-IRES2-CreERT2;Ai139"

    xyz = np.array([[float(c["ccf_x"]), float(c["ccf_y"]), float(c["ccf_z"])] for c in cells])
    shape = np.array(vol.shape)

    def lookup(perm, coords, offset=(0, 0, 0)):
        idx = np.floor(coords[:, perm] / 25.0).astype(int) + np.array(offset)
        inside = np.all((idx >= 0) & (idx < shape), axis=1)
        out = np.zeros(len(coords), dtype=np.int64)
        out[inside] = vol[idx[inside, 0], idx[inside, 1], idx[inside, 2]]
        return out

    def region_agree(labels):
        hits, n = 0, 0
        for c, lab in zip(cells, labels):
            reg = by_acr.get(c.get("soma_region") or "")
            if reg is None:
                continue
            n += 1
            node = nodes.get(int(lab))
            if node is not None and f"/{reg['id']}/" in node["structure_id_path"]:
                hits += 1
        return hits / n if n else float("nan"), n

    labels = lookup([0, 1, 2], xyz)
    agree0, n_reg = region_agree(labels)
    axis_check = {"mapping": "x=AP,y=DV,z=ML (Liu 2024 Methods)", "region_agreement": agree0,
                  "cells_with_known_region": n_reg, "permutation_search": None}
    perm = [0, 1, 2]
    if not agree0 >= 0.5:
        trials = {str(p): region_agree(lookup(list(p), xyz))[0] for p in itertools.permutations(range(3))}
        perm = list(map(int, max(trials, key=trials.get).strip("()").split(",")))
        labels = lookup(perm, xyz)
        axis_check["permutation_search"] = trials
        axis_check["selected_permutation"] = perm
    # 3x3x3 unanimity of the layer token around the soma voxel
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    neigh = np.stack([lookup(perm, xyz, o) for o in offsets], axis=1)
    for c, lab, nb in zip(cells, labels, neigh):
        node = nodes.get(int(lab))
        c["d25_structure"] = node["acronym"] if node else None
        c["d25_layer"] = layer_token(node)
        parent = nodes.get(node["parent_structure_id"]) if (node and c["d25_layer"]) else None
        c["d25_area"] = parent["acronym"] if parent else (node["acronym"] if node else None)
        toks = {layer_token(nodes.get(int(v))) for v in nb}
        c["d25_interior"] = bool(c["d25_layer"]) and toks == {c["d25_layer"]}
        c["animal"] = animal_of.get(c["brain"])
        c["genotype"] = genotype_of.get(c["brain"])

    rec = [c for c in cells if c.get("layer_recorded")]
    a = [harmonize(c["layer_recorded"]) for c in rec]
    b = [harmonize(c["d25_layer"]) or "undetermined" for c in rec]
    agree = np.array([x == y for x, y in zip(a, b)], dtype=float)
    animals = sorted({c["animal"] for c in rec}, key=str)
    by_animal = {an: [i for i, c in enumerate(rec) if c["animal"] == an] for an in animals}
    rng = np.random.default_rng(SEED)
    boot_a, boot_k = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(len(animals), size=len(animals), replace=True)
        ii = [i for p in pick for i in by_animal[animals[p]]]
        boot_a.append(agree[ii].mean())
        boot_k.append(kappa([a[i] for i in ii], [b[i] for i in ii]))
    conf = {}
    for x, y in zip(a, b):
        conf.setdefault(x, {}).setdefault(y, 0)
        conf[x][y] += 1
    it_drivers = ("Cux2-CreERT2", "Plxnd1-CreER", "Tlx3-Cre_PL56")
    it_idx = [i for i, c in enumerate(rec) if (c["genotype"] or "").startswith(it_drivers)]
    k = kappa(a, b)
    report = {
        "status": "stage_A_calibration_only_no_blank_filled", "job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(), "decoded_columns": sorted(COLUMNS), "projection_class_decoded": False,
        "annotation": {"sizes": header["sizes"], "type": header["type"], "space": header.get("space"),
                       "space_directions": header.get("space directions")},
        "cells": len(cells), "cells_with_recorded_layer": len(rec),
        "cells_with_blank_layer": len(cells) - len(rec), "axis_check": axis_check,
        "layer_agreement": {
            "raw": float(agree.mean()),
            "raw_ci95_animal_bootstrap": [float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5))],
            "kappa": k,
            "kappa_ci95_animal_bootstrap": [float(np.nanpercentile(boot_k, 2.5)), float(np.nanpercentile(boot_k, 97.5))],
            "animals": len(animals), "bootstrap_replicates": N_BOOT, "seed": SEED,
            "undetermined_among_recorded": int(sum(x == "undetermined" for x in b)),
            "it_driver_animals": {"cells": len(it_idx), "raw": float(agree[it_idx].mean()) if it_idx else None,
                                  "kappa": kappa([a[i] for i in it_idx], [b[i] for i in it_idx]) if it_idx else None},
            "confusion_recorded_by_derived": conf,
            "fill_condition_raw_ge_0.80_and_kappa_ge_0.60": bool(agree.mean() >= 0.80 and k >= 0.60)},
    }
    scratch_out.write_text(json.dumps(cells) + "\n")
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
