#!/usr/bin/env python3
"""attempt005: dendrite-only completeness check for CEBSIT IT-line cells.

Streams SWC members from the verified archive. Only soma and dendrite node
lines are parsed; every other line (axon) is skipped after reading its type
code, and no axon coordinate, length, count or target is retained per cell.
An aggregate type-code histogram (all cells pooled) identifies the codes.
Usage: cebsit_dendrite_qc.py ZIP SAMPLE_JSON NEURON_JSON OUT_DIR [NPROC]
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import zipfile

LINES = ("Cux2", "PlexinD1", "Tlx3", "Rorb", "Wfs1", "Scnn1a")
SOMA, DENDRITE = {1}, {3, 4}
MIN_STEMS, MIN_CABLE_UM = 2, 500.0
CSV_FIELDS = (
    "cell",
    "brain",
    "line",
    "acronym",
    "catalog_flag",
    "soma_nodes",
    "dendrite_nodes",
    "dendrite_cable_um",
    "stems",
    "orphan_dendrite_nodes",
    "apical_present",
    "passes",
)


def line_of(g):
    g = (g or "").lower()
    return next((k for k in LINES if k.lower() in g), None)


def member_key(member):
    path = Path(member)
    return f"{path.parent.name}_{path.stem}"


_ZIP = None


def _init(zpath):
    global _ZIP
    _ZIP = zipfile.ZipFile(zpath)  # opened once per worker process


def qc_member(member):
    types, nodes = Counter(), {}
    with _ZIP.open(member) as f:
        for raw in f:
            s = raw.decode("ascii", "replace").strip()
            if not s or s[0] == "#":
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            try:
                t = int(float(parts[1]))
            except ValueError:
                continue
            types[t] += 1
            if t in SOMA or t in DENDRITE:  # all other node types are skipped unparsed
                nodes[parts[0]] = (t, float(parts[2]), float(parts[3]), float(parts[4]), parts[6])
    soma = [k for k, v in nodes.items() if v[0] in SOMA]
    dend = {k: v for k, v in nodes.items() if v[0] in DENDRITE}
    cable, stems, orphan = 0.0, 0, 0
    for k, (t, x, y, zc, p) in dend.items():
        par = nodes.get(p)
        if par is None:
            orphan += 1
            continue
        if par[0] in SOMA:
            stems += 1
        cable += math.dist((x, y, zc), (par[1], par[2], par[3]))
    return member, {"soma_nodes": len(soma), "dendrite_nodes": len(dend), "dendrite_cable_um": round(cable, 1),
                    "stems": stems, "orphan_dendrite_nodes": orphan,
                    "apical_present": any(v[0] == 4 for v in dend.values())}, dict(types)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    zpath, sample_json, neuron_json, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    samples = {str(s["fMOST_id"]): s for s in json.loads(sample_json.read_text())}
    raw = json.loads(neuron_json.read_text())["neuron_data"]
    cat = {}
    for v in raw.values():
        f = str(v.get("file") or "")
        m = re.match(r"(\d+)", f)
        s = samples.get(m.group(1)) if m else None
        ln = line_of(s.get("transgenic_line")) if s else None
        if ln:
            cat[Path(f).stem] = {"brain": m.group(1), "line": ln, "acronym": v.get("acronym"),
                                 "catalog_flag": v.get("reconstruction_type")}
    del raw  # class/type_array never inspected
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".swc")]
    by_key = {member_key(n): n for n in members}
    todo = [by_key[k] for k in cat if k in by_key]
    agg_types, rows = Counter(), []
    with ProcessPoolExecutor(nproc, initializer=_init, initargs=(zpath,)) as ex:
        for member, res, types in ex.map(qc_member, todo, chunksize=16):
            agg_types.update(types)
            key = member_key(member)
            c = cat[key]
            res["passes"] = bool(res["soma_nodes"] >= 1 and res["stems"] >= MIN_STEMS and res["dendrite_cable_um"] >= MIN_CABLE_UM)
            rows.append({"cell": key, **c, **res})
    summary = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
               "archive_swc_members": len(members), "catalog_it_line_cells": len(cat),
               "it_line_cells_found_in_archive": len(todo),
               "it_line_cells_not_found_in_archive": len(cat) - len(todo),
               "aggregate_type_code_histogram_all_checked_cells": dict(sorted(agg_types.items())),
               "rule": {"soma_nodes_min": 1, "stems_min": MIN_STEMS, "dendrite_cable_um_min": MIN_CABLE_UM},
               "by_line_and_flag": {}}
    tally = Counter((r["line"], r["catalog_flag"], r["passes"]) for r in rows)
    for (ln, flag, ok), n in sorted(tally.items(), key=str):
        summary["by_line_and_flag"].setdefault(ln, {}).setdefault(str(flag), {})["pass" if ok else "fail"] = n
    (out / "dendrite_qc_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(out / "dendrite_qc_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=1))
    if not todo:
        raise SystemExit(
            "ERROR: zero catalog IT-line cells matched archive SWC members "
            f"({len(cat)} catalog cells, {len(members)} SWC members). "
            f"Summary JSON and header-only CSV written to {out}."
        )


if __name__ == "__main__":
    main()
