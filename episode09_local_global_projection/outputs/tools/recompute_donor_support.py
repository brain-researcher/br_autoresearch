#!/usr/bin/env python3
"""attempt003 Stage B: lineage, QC-field meaning, D25 diagnostic and donor-support census.

Reads Stage A per-cell output (non-axon columns only), Peng 2021 Supplementary
Table 2 columns A-J (never K Subclass_or_type, L Notes or the projection
matrix) and Supplementary Table 1 columns B-I. Applies the rules fixed in
attempt003/ASSUMPTIONS.md. Run inside Slurm.

Usage: recompute_donor_support.py STAGEA_CELLS_JSON PENG_SUPP_DIR CCF_PAYLOAD_DIR OUT_DIR
"""
from collections import Counter, defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import re
import socket
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xlsx_whitelist import whitelisted_rows  # noqa: E402
from derive_ccf_soma_layers import harmonize, kappa, layer_token, load_graph, read_nrrd  # noqa: E402

T2_COLS = {"A": "cell_name", "B": "formatted_name", "C": "soma_x", "D": "soma_y", "E": "soma_z",
           "F": "registered_region", "G": "manual_region", "H": "cortical_layer",
           "I": "transgenic_line", "J": "brain_id"}  # K, L and the matrix are never decoded
T1_COLS = {"B": "dataset_name", "C": "animal_id", "D": "fmost_id", "E": "age", "F": "sex",
           "G": "tamoxifen", "H": "targeted_types", "I": "estimated_neurons"}  # A (contacts) not needed
IT_TARGETS = {"Cux2": {"L2/3", "L4"}, "Plxnd1": {"L2/3", "L5"}, "Tlx3": {"L5"}}
SCOPE = {"L2/3", "L4", "L5"}
MIN_POOL = 5  # 4 nonself donors (policy minimum_allowed_value) + recipient


def norm(name):
    s = str(name).strip()
    return re.sub(r"\.(swc|eswc|SWC|ESWC)$", "", s)


def norm_layer(v):
    if v is None or str(v).strip().lower() in ("", "none", "nan", "n/a", "na"):
        return None
    v = str(v).strip()
    return v if v.startswith("L") else "L" + v


def driver(genotype):
    g = genotype or ""
    for d, prefix in (("Cux2", "Cux2-CreERT2"), ("Plxnd1", "Plxnd1-CreER"), ("Tlx3", "Tlx3-Cre_PL56")):
        if g.startswith(prefix):
            return d
    return None


def census(cells, layer_key, isocortex, include_offtarget=False):
    pools, eligible = Counter(), Counter()
    excluded = Counter()
    for c in cells:
        d = driver(c["genotype"])
        if d is None:
            continue
        if not isocortex(c["soma_region"]):
            excluded["it_driver_noncortical"] += 1
            continue
        lay = harmonize(c.get(layer_key))
        allowed = SCOPE if include_offtarget else IT_TARGETS[d]
        if lay is None:
            excluded["it_driver_cortical_unknown_layer"] += 1
            continue
        if lay not in allowed:
            excluded[f"it_driver_cortical_{'out_of_scope' if lay not in SCOPE else 'offtarget'}_{lay}"] += 1
            continue
        eligible[c["animal"]] += 1
        pools[(c["animal"], c["soma_region"], lay, d)] += 1
    supported = {k: v for k, v in pools.items() if v >= MIN_POOL}
    per_animal = {}
    for an in sorted({c["animal"] for c in cells if driver(c["genotype"])}, key=str):
        mine = {k: v for k, v in pools.items() if k[0] == an}
        per_animal[an] = {
            "eligible_cells": eligible[an],
            "largest_pool": max(mine.values()) if mine else 0,
            "supported_pools": sorted(f"{k[1]}|{k[2]}:{v}" for k, v in supported.items() if k[0] == an)}
    return {"eligible_cells": sum(eligible.values()),
            "supported_animals": len({k[0] for k in supported}),
            "supported_pools": len(supported), "cells_in_supported_pools": sum(supported.values()),
            "excluded_counts": dict(excluded), "per_animal": per_animal}


def cells_animal(cells, brain):
    return next((c["animal"] for c in cells if c["brain"] == brain), None)


def affine_summary(cells):
    out = []
    for brain in sorted({c["brain"] for c in cells}, key=str):
        rows = [c for c in cells if c["brain"] == brain]
        if len(rows) < 8:
            continue
        X = np.array([[float(c["raw_x"]), float(c["raw_y"]), float(c["raw_z"]), 1.0] for c in rows])
        Y = np.array([[float(c["ccf_x"]), float(c["ccf_y"]), float(c["ccf_z"])] for c in rows])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        M = beta[:3].T  # rows: CCF axis, cols: raw axis
        unit = np.abs(M) / np.linalg.norm(M, axis=0, keepdims=True)  # scale-free direction of each raw axis
        out.append({"brain": brain, "n": len(rows),
                    "rms_um": float(np.sqrt((resid ** 2).sum(1).mean())),
                    "ccf_axis_of_raw_x_y_z": [["AP", "DV", "ML"][int(i)] for i in unit.argmax(0)],
                    "direction_purity_raw_x_y_z": [float(v) for v in unit.max(0)],
                    "det_sign": int(np.sign(np.linalg.det(M)))})
    return out


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute required")
    stagea, supp, ccf, out = map(Path, sys.argv[1:5])
    cells = json.loads(stagea.read_text())
    for c in cells:  # absent XML cells mean blank values
        for k in ("layer_recorded", "soma_region", "manually_checked"):
            c.setdefault(k, None)
    nodes, by_acr = load_graph(ccf / "structure_graph_1.json")

    def isocortex(acr):
        n = by_acr.get(acr or "")
        return bool(n) and "/315/" in n["structure_id_path"]

    t2 = [{T2_COLS[k]: v for k, v in r.items() if k in T2_COLS}
          for r in whitelisted_rows(supp / "41586_2021_3941_MOESM4_ESM.xlsx", "xl/worksheets/sheet1.xml", T2_COLS, header_row=2)]
    t2 = [r for r in t2 if r.get("cell_name")]
    t1 = [{T1_COLS[k]: v for k, v in r.items() if k in T1_COLS}
          for r in whitelisted_rows(supp / "41586_2021_3941_MOESM3_ESM.xlsx", "xl/worksheets/sheet1.xml", T1_COLS, header_row=1)]
    t1 = [r for r in t1 if r.get("fmost_id") or r.get("animal_id")]

    # ---- lineage
    by_cell = {norm(r["cell_name"]): r for r in t2}
    by_fmt = {norm(r["formatted_name"]): r for r in t2 if r.get("formatted_name")}
    for c in cells:
        n = norm(c["cell"])
        c["peng_by_cell_name"] = n in by_cell
        c["peng_by_formatted_name"] = n in by_fmt
        hit = by_cell.get(n) or by_fmt.get(n)
        c["peng_cortical_layer"] = norm_layer(hit["cortical_layer"]) if hit else None
        c["peng_registered_region"] = hit["registered_region"] if hit else None
        c["peng_manual_region"] = hit["manual_region"] if hit else None
        c["peng_xyz"] = [hit["soma_x"], hit["soma_y"], hit["soma_z"]] if hit else None
    lineage = {"peng_table2_rows": len(t2),
               "seu_cells": len(cells),
               "match_cell_name": sum(c["peng_by_cell_name"] for c in cells),
               "match_formatted_name": sum(c["peng_by_formatted_name"] for c in cells),
               "match_either": sum(c["peng_by_cell_name"] or c["peng_by_formatted_name"] for c in cells),
               "examples_seu_names": [c["cell"] for c in cells[:3]] + [c["cell"] for c in cells[-3:]],
               "examples_peng_cell_name": [r["cell_name"] for r in t2[:3]] + [r["cell_name"] for r in t2[-3:]],
               "examples_peng_formatted_name": [r["formatted_name"] for r in t2[:3]] + [r["formatted_name"] for r in t2[-3:]]}
    unmatched = [c for c in cells if not (c["peng_by_cell_name"] or c["peng_by_formatted_name"])]
    lineage["unmatched_cells"] = len(unmatched)
    lineage["unmatched_brains"] = dict(Counter(c["brain"] for c in unmatched))
    lineage["unmatched_brain_count"] = len(lineage["unmatched_brains"])
    seu_names = {norm(c["cell"]) for c in cells}
    lineage["peng_rows_not_in_seu"] = sum(1 for r in t2 if norm(r["cell_name"]) not in seu_names
                                          and norm(r.get("formatted_name") or "") not in seu_names)

    # secondary key: within-brain mutual nearest soma, <= 50 um
    pxyz = np.array([[float(r["soma_x"]), float(r["soma_y"]), float(r["soma_z"])] for r in t2])
    unit = "um" if pxyz[:, 0].max() > 600 else "25um_voxel"
    if unit == "25um_voxel":
        pxyz = pxyz * 25.0
    lineage["peng_soma_unit_inferred"] = unit
    lineage["peng_soma_ranges"] = [[float(pxyz[:, i].min()), float(pxyz[:, i].max())] for i in range(3)]
    sec = 0
    for brain in {c["brain"] for c in cells}:
        sc = [c for c in cells if c["brain"] == brain]
        pr = [i for i, r in enumerate(t2) if str(r["brain_id"]).split(".")[0] == brain]
        if not pr:
            continue
        A = np.array([[float(c["ccf_x"]), float(c["ccf_y"]), float(c["ccf_z"])] for c in sc])
        B = pxyz[pr]
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        ai, bi = D.argmin(1), D.argmin(0)
        for i, j in enumerate(ai):
            if bi[j] == i and D[i, j] <= 50:
                sec += 1
                sc[i]["peng_soma_match"] = True
    lineage["secondary_mutual_nn_le_50um"] = sec
    both = [c for c in cells if c["peng_by_cell_name"] or c["peng_by_formatted_name"]]
    dist = [float(np.linalg.norm(np.array([float(c["ccf_x"]), float(c["ccf_y"]), float(c["ccf_z"])])
                                 - np.array([float(v) for v in c["peng_xyz"]]) * (25.0 if unit != "um" else 1.0)))
            for c in both]
    lineage["name_matched_soma_shift_um_quantiles"] = [float(np.percentile(dist, q)) for q in (5, 25, 50, 75, 95)] if dist else None

    # ---- isManuallyChecked meaning
    lin = lambda c: "peng2021_lineage" if (c["peng_by_cell_name"] or c["peng_by_formatted_name"]) else "not_in_peng2021"
    qc = {"values": dict(Counter(str(c["manually_checked"]) for c in cells)),
          "crosstab_isManuallyChecked_by_lineage": {f"{k[0]}|{k[1]}": v for k, v in
              Counter((str(c["manually_checked"]), lin(c)) for c in cells).items()}}

    # ---- provenance comparisons (non-axon labels only)
    prov = {}
    pairs = [(harmonize(c["layer_recorded"]), harmonize(c["peng_cortical_layer"])) for c in both]
    prov["seu_layer_vs_peng_layer"] = {f"{a}|{b}": v for (a, b), v in Counter(pairs).items()}
    prov["seu_region_vs_peng_manual_region_equal"] = sum(c["soma_region"] == c["peng_manual_region"] for c in both)
    prov["peng_registered_region_values_sample"] = sorted({str(c["peng_registered_region"]) for c in both})[:40]
    prov["peng_cortical_layer_values"] = dict(Counter(str(r["cortical_layer"]) for r in t2))
    blank_with_peng = [c for c in cells if not c["layer_recorded"] and c["peng_cortical_layer"]]
    prov["seu_blank_but_peng_layer"] = len(blank_with_peng)

    # D25 on Peng's own coordinates versus Peng's registered region (implementation diagnostic)
    _, vol = read_nrrd(ccf / "annotation_25_ccf_2017.nrrd")
    idx = np.floor(pxyz / 25.0).astype(int)
    inside = np.all((idx >= 0) & (idx < np.array(vol.shape)), axis=1)
    lab = np.zeros(len(t2), dtype=np.int64)
    lab[inside] = vol[idx[inside, 0], idx[inside, 1], idx[inside, 2]]
    exact = anc = n = 0
    lay_rr = Counter()
    for r, l in zip(t2, lab):
        reg = by_acr.get(str(r["registered_region"]))
        if reg is None:
            continue
        n += 1
        node = nodes.get(int(l))
        if node is not None and node["acronym"] == reg["acronym"]:
            exact += 1
        if node is not None and f"/{reg['id']}/" in node["structure_id_path"]:
            anc += 1
        lay_rr[(str(harmonize(layer_token(reg))), str(harmonize(layer_token(node))))] += 1
    prov["d25_on_peng_xyz_vs_peng_registered_region"] = {
        "rows_with_known_registered_region": n, "exact_acronym": exact, "registered_is_ancestor_or_self": anc,
        "registered_layer_vs_d25_layer": {f"{a}|{b}": v for (a, b), v in lay_rr.items()}}
    # Peng registered layer versus Peng curated Cortical_layer
    rc = Counter()
    for r in t2:
        reg = by_acr.get(str(r["registered_region"]))
        if norm_layer(r["cortical_layer"]):
            rc[(str(harmonize(layer_token(reg))) if reg else "unparsed", str(harmonize(norm_layer(r["cortical_layer"]))))] += 1
    prov["peng_registered_layer_vs_peng_cortical_layer"] = {f"{a}|{b}": v for (a, b), v in rc.items()}

    # ---- post hoc diagnostics, formed after Stage A failed calibration; no rule changes
    ccf_max = np.array([13200.0, 8000.0, 11400.0])
    valid_p = np.all((pxyz >= 0) & (pxyz < ccf_max), axis=1)
    post = {"label": "post_hoc_after_stageA_failure", "peng_rows_inside_ccf_bounds": int(valid_p.sum())}
    a1, b1 = [], []
    for r, l, ok in zip(t2, lab, valid_p):
        cl = norm_layer(r["cortical_layer"])
        if cl and ok:
            a1.append(harmonize(cl))
            b1.append(harmonize(layer_token(nodes.get(int(l)))) or "undetermined")
    post["d25_from_peng_coords_vs_peng_cortical_layer"] = {
        "n": len(a1), "raw": float(np.mean([x == y for x, y in zip(a1, b1)])) if a1 else None,
        "kappa": kappa(a1, b1) if a1 else None,
        "confusion": {f"{x}|{y}": v for (x, y), v in Counter(zip(a1, b1)).items()}}

    def anc_or_self(acr_ref, acr_node):
        ref, node = by_acr.get(acr_ref or ""), by_acr.get(acr_node or "")
        return bool(ref and node and f"/{ref['id']}/" in node["structure_id_path"])

    matched = [c for c in cells if c["peng_by_cell_name"] or c["peng_by_formatted_name"]]
    post["d25_from_seu_coords_vs_peng_registered_region"] = {
        "n": sum(1 for c in matched if by_acr.get(str(c["peng_registered_region"]))),
        "ancestor_or_self": sum(anc_or_self(c["peng_registered_region"], c["d25_structure"]) for c in matched)}
    for flag in ("0", "1"):
        grp = [c for c in cells if str(c["manually_checked"]) == flag and by_acr.get(c["soma_region"] or "")]
        post[f"d25_from_seu_coords_vs_seu_region_isManuallyChecked_{flag}"] = {
            "n": len(grp), "ancestor_or_self": sum(anc_or_self(c["soma_region"], c["d25_structure"]) for c in grp)}
    direct, mirrored = [], []
    for c in matched:
        p = np.array([float(v) for v in c["peng_xyz"]])
        s = np.array([float(c["ccf_x"]), float(c["ccf_y"]), float(c["ccf_z"])])
        m = p.copy()
        m[2] = 11400.0 - m[2]
        direct.append(float(np.linalg.norm(s - p)))
        mirrored.append(float(np.linalg.norm(s - m)))
    best = np.minimum(direct, mirrored)
    post["seu_vs_peng_soma_shift_um"] = {
        "n": len(best), "mirror_closer_fraction": float(np.mean(np.array(mirrored) < np.array(direct))),
        "min_of_direct_or_ML_mirror_quantiles_5_25_50_75_95": [float(np.percentile(best, q)) for q in (5, 25, 50, 75, 95)]}
    post["checked_but_unmatched_names"] = [c["cell"] for c in cells if str(c["manually_checked"]) == "1"
                                           and not (c["peng_by_cell_name"] or c["peng_by_formatted_name"])]
    post["peng_rows_not_in_seu_names"] = [r["cell_name"] for r in t2 if norm(r["cell_name"]) not in seu_names
                                          and norm(r.get("formatted_name") or "") not in seu_names]
    new = [c for c in cells if str(c["manually_checked"]) == "0"]
    post["isManuallyChecked_0_brains"] = dict(Counter(c["brain"] for c in new))
    prov["post_hoc"] = post

    # ---- census layers (rules fixed in ASSUMPTIONS.md; D25 fill failed calibration)
    for c in cells:
        c["layer_S0"] = c["layer_recorded"]
        c["layer_P"] = c["layer_recorded"] or c["peng_cortical_layer"]
        c["layer_S1"] = c["d25_layer"]
        c["layer_S2"] = c["layer_P"] or (c["d25_layer"] if c["d25_interior"] else None)
    scenarios = {
        "S0_recorded_only": census(cells, "layer_S0", isocortex),
        "P_recorded_then_peng2021_no_D25_fill": census(cells, "layer_P", isocortex),
        "S1_D25_every_cell_unvalidated": census(cells, "layer_S1", isocortex),
        "S2_P_plus_interior_D25_fills_unvalidated": census(cells, "layer_S2", isocortex),
        "S_IT_P_with_offtarget_layers_in_scope": census(cells, "layer_P", isocortex, include_offtarget=True),
        "S_IT_on_S2": census(cells, "layer_S2", isocortex, include_offtarget=True),
    }
    blank_it = [c for c in cells if driver(c["genotype"]) and not c["layer_recorded"]]
    blank_summary = {"it_driver_blank_layer_cells": len(blank_it),
                     "by_brain": dict(Counter(c["brain"] for c in blank_it)),
                     "isocortex_region": sum(isocortex(c["soma_region"]) for c in blank_it),
                     "regions": dict(Counter(c["soma_region"] for c in blank_it)),
                     "lineage": dict(Counter(lin(c) for c in blank_it)),
                     "peng_layer_available": sum(bool(c["peng_cortical_layer"]) for c in blank_it)}

    recovery = {}
    for brain in sorted({c["brain"] for c in cells if driver(c["genotype"])}, key=str):
        bl = [c for c in cells if c["brain"] == brain and not c["layer_recorded"]]
        cort = [c for c in bl if isocortex(c["soma_region"])]
        recovery[brain] = {"animal": cells_animal(cells, brain), "blank": len(bl), "blank_isocortex": len(cort),
                           "recovered_from_peng2021": sum(bool(c["peng_cortical_layer"]) for c in cort),
                           "newly_annotated_isManuallyChecked_0": sum(str(c["manually_checked"]) == "0" for c in cort),
                           "unknown_under_P": sum(not c["layer_P"] for c in cort)}
    batch = {"peng_table1_rows": len(t1),
             "peng_table1_fields": list(T1_COLS.values()),
             "seu_brains_in_peng_table1": sorted({c["brain"] for c in cells} & {str(r["fmost_id"]).split(".")[0] for r in t1}, key=str)}

    report = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "decoded": {"Full_morphometry": "A-J,L (K never)", "Peng_T2": "A-J (K,L,matrix never)", "Peng_T1": "B-I"},
              "lineage": lineage, "isManuallyChecked": qc, "provenance": prov,
              "blank_layer_it_driver_cells": blank_summary, "blank_recovery_by_it_brain": recovery, "census": scenarios,
              "raw_to_ccf_affine": affine_summary(cells), "batch_fields": batch,
              "floors": {"registered_development_plus_audit": 20, "unadopted_LOAO_proposal": 9}}
    (out / "stageB_report.json").write_text(json.dumps(report, indent=2) + "\n")
    keep = ["cell", "brain", "animal", "genotype", "soma_region", "layer_recorded", "manually_checked",
            "peng_by_cell_name", "peng_by_formatted_name", "peng_cortical_layer", "peng_registered_region",
            "peng_manual_region", "d25_structure", "d25_layer", "d25_area", "d25_interior",
            "layer_P", "layer_S2"]
    with open(out / "cell_nonaxon_ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow(c)
    t1_out = [{k: r.get(k) for k in ("animal_id", "fmost_id", "age", "sex", "tamoxifen", "targeted_types")} for r in t1]
    (out / "peng2021_table1_batch_fields.json").write_text(json.dumps(t1_out, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("lineage", "isManuallyChecked", "blank_layer_it_driver_cells")}, indent=2))


if __name__ == "__main__":
    main()
