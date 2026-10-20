#!/usr/bin/env python3
"""attempt003 schema step: Peng 2021 supplement headers and atlas format only.

Emits header rows (no data rows) and atlas format facts. Run inside Slurm.
Usage: inspect_attempt003_schemas.py PENG_SUPP_DIR CCF_PAYLOAD_DIR OUT_JSON
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlsx_whitelist import header_rows  # noqa: E402


def nrrd_header(path):
    lines = []
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", "replace").rstrip("\n")
            if line == "":
                break
            lines.append(line)
    return lines


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    supp, ccf, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    report = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "data_rows_decoded": False, "supplements": {}}
    for f in sorted(supp.glob("*.xlsx")):
        report["supplements"][f.name] = {
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "sheets": header_rows(f)}
    ann = ccf / "annotation_25_ccf_2017.nrrd"
    report["annotation_header"] = nrrd_header(ann)
    graph = json.loads((ccf / "structure_graph_1.json").read_text())
    nodes, stack = [], [(n, "/") for n in graph["msg"]]
    while stack:
        n, path = stack.pop()
        n["structure_id_path"] = path + f"{n['id']}/"
        nodes.append(n)
        stack.extend((c, n["structure_id_path"]) for c in n.get("children", []))
    iso = [n for n in nodes if "/315/" in n["structure_id_path"]]
    layer_names = sorted({n["name"].rsplit(" ", 1)[-1] for n in iso if "ayer" in n["name"]})
    report["structure_graph"] = {
        "node_count": len(nodes), "isocortex_node_count": len(iso),
        "isocortex_layer_tokens": layer_names,
        "example_layer_nodes": [(n["id"], n["acronym"], n["name"]) for n in iso if "ayer" in n["name"]][:12],
        "node_keys": sorted(k for k in nodes[0].keys() if k != "children")}
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
