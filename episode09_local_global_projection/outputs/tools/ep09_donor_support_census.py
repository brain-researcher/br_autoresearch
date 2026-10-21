#!/usr/bin/env python3
"""Outcome-blind census of the companion draft's donor-support rule (somata, atlas, Laplace depth).

Rule under review (companion draft D02): exact pool (animal|line|area|layer, with the
contract's 24-cell coordinate relabel), same soma hemisphere, |Laplace depth difference|
<= 0.10, soma distance <= width, where width = animal-balanced development 90th percentile
of the distance to the 4th-nearest eligible peer, rounded up to 25 um and capped at 500 um.
Support graph: iteratively drop cells with < 4 caliper donors, then require a fixed-point-
free perfect matching per pool (Hall check), repeating to stability.
Also counts, per V_6 target, animals with >= 4 donor-supported A_pre-positive recipients
(option B's floor; an outcome-blind upper bound on observable supported recipients).
Reads no dendrite or axon rows.

The historical default is a preliminary degree/Hall census. The optional
--qualify-feasible-edges mode additionally removes donor edges that participate
in no complete plan and repeats pruning, as required by the proposed companion
graph definition. Neither mode adopts that companion or authorizes outcome access.

Usage: ep09_donor_support_census.py ZIP CELL_MANIFEST_CSV CCF_PAYLOAD_DIR TOOLS_DIR
       RELABELED_CELLS_CSV SOMA_FRAMES_CSV OUT_DIR
"""
import argparse
import csv
import json
import math
import os
import socket
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEPTH_CALIPER = 0.10
MIN_DONORS = 4
WIDTH_QUANTILE = 0.90
WIDTH_CAP_UM = 500
N_COVERAGE = 6


def weighted_quantile(values_weights, q):
    total = sum(w for _, w in values_weights)
    cumulative = 0.0
    for value, weight in sorted(values_weights):
        cumulative += weight / total
        if cumulative >= q - 1e-12:
            return value
    return max(v for v, _ in values_weights)


def supported_vertices(edges, *, minimum_donors=MIN_DONORS, qualify_feasible_edges=False):
    """Per-pool decision, with the historical census preserved by default."""
    if qualify_feasible_edges:
        from ep09_donor_graph import qualify_support_graph

        return set(qualify_support_graph(edges, minimum_donors=minimum_donors).edges)

    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    alive = set(edges)
    while True:
        drop = [i for i in alive if len(edges[i] & alive) < minimum_donors]
        if drop:
            alive.difference_update(drop)
            continue
        if alive:
            order = sorted(alive)
            index = {cell: k for k, cell in enumerate(order)}
            rows, cols = [], []
            for i in order:
                for j in edges[i] & alive:
                    rows.append(index[i])
                    cols.append(index[j])
            matrix = csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(len(order), len(order))
            )
            if (maximum_bipartite_matching(matrix, perm_type="column") < 0).any():
                alive.clear()
        return alive


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Refusing to run: SLURM_JOB_ID is not set.")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zip", "manifest", "ccf_dir", "tools_dir", "relabel", "frames", "out_dir"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--qualify-feasible-edges", action="store_true")
    args = parser.parse_args()
    zip_path, manifest_path, ccf_dir, tools_dir, relabel_path, frames_path, out_dir = (
        args.zip, args.manifest, args.ccf_dir, args.tools_dir, args.relabel, args.frames, args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "donor_support_census_variants.json"
    if out_json.exists():
        raise SystemExit(f"Refusing to overwrite {out_json}")

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
    xyz, exact, sides, coord_areas, _found, _m = P.read_somata(zip_path, cells, volume, graph)

    relabel = {r["cell"]: r for r in csv.DictReader(relabel_path.open())}
    frames = {r["cell"]: r for r in csv.DictReader(frames_path.open())}
    pool_of, depth, hemi = [], [], []
    for c in cells:
        area, layer = c["area"], c["layer"]
        if c["cell"] in relabel:
            r = relabel[c["cell"]]
            area = r["coordinate_area"] or area
            layer = r["coordinate_layer"] or layer
        pool_of.append(f"{c['brain']}|{c['line']}|{area}|{layer}")
        f = frames[c["cell"]]
        depth.append(float(f["depth"]) if f["depth"] else math.nan)
        hemi.append(f["hemisphere"])
    depth = np.array(depth)

    pools = defaultdict(list)
    for i, key in enumerate(pool_of):
        pools[key].append(i)

    def peers(i, members):
        return [j for j in members if j != i and hemi[j] == hemi[i]
                and math.isfinite(depth[i]) and math.isfinite(depth[j])
                and abs(depth[i] - depth[j]) <= DEPTH_CALIPER]

    # Width from development cells with >= 4 hemisphere/depth-eligible peers.
    per_animal = defaultdict(list)
    for key, members in pools.items():
        for i in members:
            if cells[i]["ep09_role"] != "development":
                continue
            p = peers(i, members)
            if len(p) >= MIN_DONORS:
                d = sorted(float(np.linalg.norm(xyz[i] - xyz[j])) for j in p)
                per_animal[cells[i]["brain"]].append(d[MIN_DONORS - 1])
    vw = [(d, 1.0 / (len(per_animal) * len(v))) for v in per_animal.values() for d in v]
    q90 = weighted_quantile(vw, WIDTH_QUANTILE)
    width = min(WIDTH_CAP_UM, 25 * math.ceil(q90 / 25))

    labels, frequencies = np.unique(volume, return_counts=True)
    targets, label_owner, _r = P.build_vocabulary(graph, labels, frequencies)
    packed, offsets = P.pack_target_voxels(volume, labels, frequencies, label_owner, len(targets))
    column = {b: j for j, b in enumerate(brains)}
    animal_idx = np.array([column[c["brain"]] for c in cells], dtype=np.intp)
    counts_all = P.count_pairs(
        targets, graph, packed, offsets, xyz, exact, sides, manifest_areas, coord_areas, animal_idx, len(brains),
    )
    v6 = counts_all.min(axis=1) >= N_COVERAGE
    family = P.make_family_map(targets, graph)
    tele = {315, 698, 1089, 703, 477, 803}
    v6_tele = np.array([v6[r] and family[targets[r // 2]] in tele for r in range(len(v6))])
    dev_cols = [column[b] for b in brains if animal_roles[b] == "development"]
    aud_cols = [column[b] for b in brains if animal_roles[b] == "audit"]

    def support(same_hemi, depth_cal, dist_cal):
        result = np.zeros(len(cells), dtype=bool)
        for key, members in pools.items():
            def ok(i, j):
                if j == i:
                    return False
                if same_hemi and hemi[j] != hemi[i]:
                    return False
                if depth_cal is not None and not (abs(depth[i] - depth[j]) <= depth_cal):
                    return False
                if dist_cal is not None and np.linalg.norm(xyz[i] - xyz[j]) > dist_cal:
                    return False
                return True
            edges = {i: {j for j in members if ok(i, j)} for i in members}
            alive = supported_vertices(
                edges, minimum_donors=MIN_DONORS,
                qualify_feasible_edges=args.qualify_feasible_edges,
            )
            for i in alive:
                result[i] = True
        return result

    variants = {
        "V0_exact_pool_only": (False, None, None),
        "V1_same_hemisphere": (True, None, None),
        "V2_hemi_depth010": (True, DEPTH_CALIPER, None),
        "V3_hemi_dist_uncapped_q90": (True, None, 25 * math.ceil(q90 / 25)),
        "V4_hemi_depth010_dist_uncapped_q90": (True, DEPTH_CALIPER, 25 * math.ceil(q90 / 25)),
        "V5_draft_hemi_depth010_dist500": (True, DEPTH_CALIPER, width),
    }
    variant_reports = {}
    for name, (same_hemi, depth_cal, dist_cal) in variants.items():
        sup = support(same_hemi, depth_cal, dist_cal)
        keep = np.flatnonzero(sup)
        counts_sup = P.count_pairs(
            targets, graph, packed, offsets, xyz[keep], [exact[i] for i in keep], sides[keep],
            [manifest_areas[i] for i in keep], [coord_areas[i] for i in keep], animal_idx[keep], len(brains),
        )
        per_animal_sup = np.bincount(animal_idx[keep], minlength=len(brains))
        c_eligible = per_animal_sup >= MIN_DONORS
        cols = np.flatnonzero(c_eligible)
        floor_rows = (counts_sup[:, cols] >= MIN_DONORS).all(axis=1) if cols.size else np.zeros(len(v6), bool)
        variant_reports[name] = {
            "calipers": {"same_hemisphere": same_hemi, "depth": depth_cal, "distance_um": dist_cal},
            "supported_cells": {"development": int(sup[[i for i, c in enumerate(cells) if c["ep09_role"] == "development"]].sum()),
                                "audit": int(sup[[i for i, c in enumerate(cells) if c["ep09_role"] == "audit"]].sum())},
            "development_animals_with_ge4_supported": int(c_eligible[dev_cols].sum()),
            "audit_animals_with_ge4_supported": int(c_eligible[aud_cols].sum()),
            "animals_with_zero_supported": int((per_animal_sup == 0).sum()),
            "V6_telencephalic_targets_with_ge4_supported_A_pre_in_every_C_eligible_animal": int((v6_tele & floor_rows).sum()),
            "V6_telencephalic_targets": int(v6_tele.sum()),
            "per_animal_supported": {b: int(per_animal_sup[column[b]]) for b in brains},
        }
    supported = support(*variants["V5_draft_hemi_depth010_dist500"])
    counts_sup = P.count_pairs(
        targets, graph, packed, offsets, xyz[np.flatnonzero(supported)], [exact[i] for i in np.flatnonzero(supported)],
        sides[supported], [manifest_areas[i] for i in np.flatnonzero(supported)],
        [coord_areas[i] for i in np.flatnonzero(supported)], animal_idx[supported], len(brains),
    )
    floor_ok = counts_sup.min(axis=1) >= MIN_DONORS

    animals = []
    for b in brains:
        sel = animal_idx == column[b]
        animals.append({"brain": b, "ep09_role": animal_roles[b], "eligible": int(sel.sum()),
                        "pool_ge5": int(sum(1 for i in np.flatnonzero(sel) if len(pools[pool_of[i]]) >= 5)),
                        "caliper_supported": int((sel & supported).sum())})
    summary = {
        "provenance": {"SLURM_JOB_ID": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                       "completed_utc": datetime.now(timezone.utc).isoformat(),
                       "soma_frames": str(frames_path), "relabeled_cells": str(relabel_path)},
        "rule": {"depth_caliper": DEPTH_CALIPER, "min_donors": MIN_DONORS, "width_quantile": WIDTH_QUANTILE,
                 "width_cap_um": WIDTH_CAP_UM, "same_hemisphere": True},
        "support_graph_algorithm": (
            "degree_pruning_Hall_check_and_feasible_edge_fixed_point"
            if args.qualify_feasible_edges else "historical_degree_pruning_and_Hall_check_only"
        ),
        "width_unrounded_um": q90, "width_um": width,
        "supported_cells": {role: int(sum(supported[i] for i, c in enumerate(cells) if c["ep09_role"] == role))
                            for role in ("development", "audit")},
        "animals_with_zero_supported": [a["brain"] for a in animals if a["caliper_supported"] == 0],
        "min_supported_per_animal": min(a["caliper_supported"] for a in animals),
        "per_animal": animals,
        "V6_sided_targets_all_cells": int(v6.sum()),
        "V6_targets_meeting_floor_4_supported_A_pre_in_every_animal": int((v6 & floor_ok).sum()),
        "V6_telencephalic_targets": int(v6_tele.sum()),
        "V6_telencephalic_targets_meeting_floor": int((v6_tele & floor_ok).sum()),
        "variants": variant_reports,
        "axon_or_dendrite_geometry_read": False,
    }
    with out_json.open("x") as f:
        json.dump(summary, f, indent=1)
    os.chmod(out_json, 0o444)
    for name, v in variant_reports.items():
        print(name, v["supported_cells"], "dev>=4:", v["development_animals_with_ge4_supported"], "aud>=4:", v["audit_animals_with_ge4_supported"], "zero:", v["animals_with_zero_supported"], "tele floor:", v["V6_telencephalic_targets_with_ge4_supported_A_pre_in_every_C_eligible_animal"], "/", v["V6_telencephalic_targets"])


if __name__ == "__main__":
    main()
