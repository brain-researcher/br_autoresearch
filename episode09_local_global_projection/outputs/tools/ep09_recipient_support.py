#!/usr/bin/env python3
"""Stage-A recipient-support recount (somata and atlas only).

Applies the contract v1.2 rule that the cells whose catalog soma area disagrees
with the 25 um coordinate lookup take coordinate-derived pool labels, then
recounts every-animal coverage (V_n) over donor-supported cells only (exact pool
of >= 5 cells, i.e. >= 4 non-self donors, before calipers). Reuses the reviewed
functions of ep09_feasibility_precheck_v1_2.py; reads no dendrite or axon rows.

Usage: ep09_recipient_support.py ZIP CELL_MANIFEST_CSV CCF_PAYLOAD_DIR TOOLS_DIR OUT_DIR
"""
import csv
import json
import os
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MIN_POOL = 5  # >= 4 non-self donors, the policy's minimum allowed donor count


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refusing to run: SLURM_JOB_ID is not set.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    zip_path, manifest_path, ccf_dir, tools_dir, out_dir = map(Path, sys.argv[1:6])
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / "recipient_support.json", out_dir / "relabeled_cells_qc.csv"]
    for path in outputs:
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")

    import numpy as np
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(tools_dir.resolve()))
    import derive_ccf_soma_layers as loader
    import ep09_feasibility_precheck_v1_2 as P

    P.np, P.cKDTree = np, cKDTree

    cells, brains, animal_roles = P.read_manifest(manifest_path)
    _h, volume = loader.read_nrrd(str(ccf_dir / "annotation_25_ccf_2017.nrrd"))
    nodes, by_acr = loader.load_graph(str(ccf_dir / "structure_graph_1.json"))
    graph = P.AtlasGraph(nodes, by_acr)
    manifest_areas = [graph.by_acr[c["area"]] for c in cells]
    xyz, exact, sides, coord_areas, found, _members = P.read_somata(
        zip_path, cells, volume, graph
    )

    # Coordinate-derived area and layer for every soma (atlas lookup only).
    relabeled = []
    pool_keys = []
    for i, cell in enumerate(cells):
        voxel = tuple(int(v // P.VOXEL_UM) for v in exact[i])
        label = int(volume[voxel])
        coord_area = coord_areas[i]
        coord_layer = None
        if label and graph.is_layer(label):
            parent_acr = graph.nodes[graph.parent[label]]["acronym"]
            coord_layer = "L" + graph.nodes[label]["acronym"][len(parent_acr):]
        area, layer = cell["area"], cell["layer"]
        if coord_area != manifest_areas[i]:
            coord_acr = None if coord_area is None else graph.nodes[coord_area]["acronym"]
            relabeled.append({
                "cell": cell["cell"], "brain": cell["brain"], "ep09_role": cell["ep09_role"],
                "line": cell["line"], "catalog_area": area, "catalog_layer": layer,
                "coordinate_area": coord_acr, "coordinate_layer": coord_layer,
                "soma_voxel_label": label,
            })
            area = coord_acr if coord_acr is not None else area
            layer = coord_layer if coord_layer is not None else layer
        pool_keys.append(f"{cell['brain']}|{cell['line']}|{area}|{layer}")

    pool_size = Counter(pool_keys)
    supported = np.array([pool_size[k] >= MIN_POOL for k in pool_keys])
    frozen_supported = np.array([int(c["pool_size"]) >= MIN_POOL for c in cells])

    labels, frequencies = np.unique(volume, return_counts=True)
    targets, label_owner, _report = P.build_vocabulary(graph, labels, frequencies)
    family_by_node = P.make_family_map(targets, graph)
    packed, offsets = P.pack_target_voxels(volume, labels, frequencies, label_owner, len(targets))
    column = {b: j for j, b in enumerate(brains)}
    animal_idx = np.array([column[c["brain"]] for c in cells], dtype=np.intp)

    def coverage(mask):
        keep = np.flatnonzero(mask)
        counts = P.count_pairs(
            targets, graph, packed, offsets, xyz[keep], [exact[i] for i in keep],
            sides[keep], [manifest_areas[i] for i in keep],
            [coord_areas[i] for i in keep], animal_idx[keep], len(brains),
        )
        minimum = counts.min(axis=1)
        rows = []
        for n in range(3, 15):
            survives = minimum >= n
            gate = P.family_gate(survives, targets, family_by_node, graph)
            rows.append({
                "n": n,
                "sided_target_count": int(survives.sum()),
                "supported_divisions": {d["acronym"]: d["sided_target_count"] for d in gate["supported_divisions"]},
            })
        return rows

    per_animal = []
    for b in brains:
        sel = animal_idx == column[b]
        per_animal.append({
            "brain": b, "ep09_role": animal_roles[b], "eligible_cells": int(sel.sum()),
            "donor_supported_cells": int((sel & supported).sum()),
            "donor_supported_cells_frozen_labels": int((sel & frozen_supported).sum()),
        })
    min_supported = min(a["donor_supported_cells"] for a in per_animal)
    summary = {
        "provenance": {
            "SLURM_JOB_ID": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [str(zip_path), str(manifest_path), str(ccf_dir), str(tools_dir)],
        },
        "rule": "pool = brain|line|area|layer; contract v1.2 relabels catalog-vs-coordinate area disagreements to coordinate labels; donor-supported = pool >= 5 cells (>= 4 non-self donors), before calipers",
        "somata_read": found,
        "relabeled_cell_count": len(relabeled),
        "donor_supported_cells": {
            role: int(sum(supported[i] for i, c in enumerate(cells) if c["ep09_role"] == role))
            for role in ("development", "audit")
        },
        "minimum_donor_supported_cells_per_animal": min_supported,
        "allowed_n_if_recipients_are_donor_supported": [3, min_supported],
        "per_animal": per_animal,
        "V_n_all_eligible_cells": coverage(np.ones(len(cells), dtype=bool)),
        "V_n_donor_supported_cells": coverage(supported),
        "axon_or_dendrite_geometry_read": False,
    }
    with outputs[0].open("x") as f:
        json.dump(summary, f, indent=1)
    with outputs[1].open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(relabeled[0]) if relabeled else ["cell"])
        writer.writeheader()
        writer.writerows(relabeled)
    for path in outputs:
        os.chmod(path, 0o444)
    print(json.dumps({k: summary[k] for k in (
        "relabeled_cell_count", "donor_supported_cells",
        "minimum_donor_supported_cells_per_animal")}, indent=1))
    for a, b in zip(summary["V_n_all_eligible_cells"], summary["V_n_donor_supported_cells"]):
        print(f"n={a['n']}: all cells {a['sided_target_count']}, donor-supported {b['sided_target_count']}")


if __name__ == "__main__":
    main()
