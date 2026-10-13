#!/usr/bin/env python3
"""Collect EP22 D1–D3 cells: equal fold → sector → participant aggregation, figure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
MODELS = ("DINOv2", "CLIP", "PEcore", "SigLIP2")
REF, VLMS = "DINOv2", ("CLIP", "PEcore", "SigLIP2")
PRIMARY = ("laion-ventral", "laion-lateral", "laion-dorsal")
EVC = "retinotopic-EVC"
RUNGS = ("R0", "R1", "R2", "R3")
COLORS = {"CLIP": "#2a78d6", "PEcore": "#eb6834", "SigLIP2": "#1baf7a", "DINOv2": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def nanmean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def interval(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if arr.size else [None, None]


def components(rsa, m, rois):
    """Native gap and its exact eigenvalue / eigenvector split, averaged over rois."""
    g, ev = [], []
    for roi in rois:
        a, b = rsa[f"{m}|{m}"][roi], rsa[f"{REF}|{REF}"][roi]
        c, d = rsa[f"{m}|{REF}"][roi], rsa[f"{REF}|{m}"][roi]
        g.append(a - b)
        ev.append(0.5 * ((a - c) + (d - b)))
    gap, e_val = nanmean(g), nanmean(ev)
    return {"gap": gap, "E_val": e_val, "E_vec": gap - e_val}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--figure-dir", type=Path, required=True)
    a = ap.parse_args()
    cells = {}
    for p in PARTICIPANTS:
        for f in range(5):
            path = a.cells / f"{p}-fold-{f}.json"
            cells[(p, f)] = json.loads(path.read_text()) if path.is_file() else None
    missing = [f"{p}-fold-{f}" for (p, f), c in cells.items() if c is None]
    summary = {"analysis_ids": ["d1_eigen_factorial", "d2_flexibility_ladder", "d3_selection_intervention"],
               "created_utc": datetime.now(timezone.utc).isoformat(), "cells_present": 25 - len(missing),
               "missing_cells": missing, "aggregation": "equal folds, then primary sectors, then participants",
               "evidence_status": "exploratory development evidence (EP04-exposed responses)"}
    done = {k: c for k, c in cells.items() if c is not None}
    summary["R0_reproduction"] = {
        "all_reproduced": all(c["R0_reproduction"]["reproduced"] for c in done.values()),
        "max_abs_fisher_z_difference": max(c["R0_reproduction"]["max_abs_fisher_z_difference"] for c in done.values()),
        "rows_compared": sum(c["R0_reproduction"]["compared_rows"] for c in done.values())}

    def per_participant(fn):
        out = {}
        for p in PARTICIPANTS:
            vals = [fn(done[(p, f)]) for f in range(5) if (p, f) in done]
            out[p] = vals
        return out

    # D1
    d1 = {}
    for m in VLMS:
        entry = {}
        for scope, rois in (("primary", PRIMARY), ("EVC", (EVC,))):
            pp = {}
            for p in PARTICIPANTS:
                fold_vals = [components(done[(p, f)]["D1"]["point"], m, rois) for f in range(5) if (p, f) in done]
                pp[p] = {k: nanmean([v[k] for v in fold_vals]) for k in ("gap", "E_val", "E_vec")}
            mean = {k: nanmean([pp[p][k] for p in PARTICIPANTS]) for k in ("gap", "E_val", "E_vec")}
            boot = {k: [] for k in ("gap", "E_val", "E_vec")}
            n_boot = min(len(done[k]["D1"]["bootstrap"]) for k in done)
            per_p_boot = {p: {k: [] for k in boot} for p in PARTICIPANTS}
            for b in range(n_boot):
                part = {k: [] for k in boot}
                for p in PARTICIPANTS:
                    fold_vals = [components(done[(p, f)]["D1"]["bootstrap"][b], m, rois) for f in range(5) if (p, f) in done]
                    for k in boot:
                        v = nanmean([fv[k] for fv in fold_vals])
                        part[k].append(v)
                        per_p_boot[p][k].append(v)
                for k in boot:
                    boot[k].append(nanmean(part[k]))
            entry[scope] = {"participants": pp, "mean": mean,
                            "eigenvalue_share_of_gap": (mean["E_val"] / mean["gap"]) if mean["gap"] else None,
                            "bootstrap_95": {k: interval(v) for k, v in boot.items()},
                            "participant_bootstrap_95": {p: {k: interval(v) for k, v in d.items()}
                                                         for p, d in per_p_boot.items()},
                            "participants_E_val_same_sign_as_gap": int(sum(
                                np.sign(pp[p]["E_val"]) == np.sign(pp[p]["gap"]) for p in PARTICIPANTS)),
                            "participants_E_val_larger_than_E_vec_in_gap_direction": int(sum(
                                (pp[p]["E_val"] * np.sign(pp[p]["gap"])) > (pp[p]["E_vec"] * np.sign(pp[p]["gap"]))
                                for p in PARTICIPANTS))}
        d1[m] = entry
    summary["D1"] = d1

    # D2
    d2 = {"rsa": {}, "gap_vs_DINOv2": {}, "closure": {}, "alpha_hat": {}, "R2": {}}
    for m in MODELS:
        rsa = {}
        for rung in RUNGS + ("R1_alpha1_k256",):
            pp = {p: nanmean([nanmean([done[(p, f)]["D2"][m][roi][rung] for roi in PRIMARY])
                              for f in range(5) if (p, f) in done]) for p in PARTICIPANTS}
            rsa[rung] = {"participants": pp, "mean": nanmean(list(pp.values()))}
        d2["rsa"][m] = rsa
        d2["alpha_hat"][m] = {p: [done[(p, f)]["D2"][m]["R1_alpha_hat"] for f in range(5) if (p, f) in done]
                              for p in PARTICIPANTS}
        d2["R2"][m] = {"lambda_hat": [done[k]["D2"][m][roi]["R2_lambda_hat"] for k in done for roi in PRIMARY],
                       "nonzero_weights_mean": nanmean([done[k]["D2"][m][roi]["R2_nonzero_weights"]
                                                        for k in done for roi in PRIMARY])}
    for m in VLMS:
        gaps, clos = {}, {}
        for rung in RUNGS:
            pp = {p: d2["rsa"][m][rung]["participants"][p] - d2["rsa"][REF][rung]["participants"][p] for p in PARTICIPANTS}
            gaps[rung] = {"participants": pp, "mean": nanmean(list(pp.values())),
                          "negative_participants": int(sum(v < 0 for v in pp.values()))}
        g0 = gaps["R0"]["mean"]
        for rung in RUNGS[1:]:
            clos[rung] = 1.0 - abs(gaps[rung]["mean"]) / abs(g0) if g0 else None
        d2["gap_vs_DINOv2"][m] = gaps
        d2["closure"][m] = clos
    summary["D2"] = d2

    # D3
    d3 = {"own_selection_effect": {}, "delta": {}, "ED_ratio": {}}
    for s in MODELS:
        o_pp, delta_pp, ed_pp = {}, {m: {} for m in MODELS}, {m: {} for m in MODELS}
        for p in PARTICIPANTS:
            o_f, d_f, ed_f = [], {m: [] for m in MODELS}, {m: [] for m in MODELS}
            for f in range(5):
                if (p, f) not in done:
                    continue
                sub = done[(p, f)]["D3"]["subsets"]
                rand = [v for k, v in sub.items() if k.startswith("random_")]
                sel = sub[f"selected_by_{s}"]
                delta = {}
                for m in MODELS:
                    sel_rsa = nanmean([sel["RSA"][m][roi] for roi in PRIMARY])
                    rnd_rsa = nanmean([nanmean([r["RSA"][m][roi] for roi in PRIMARY]) for r in rand])
                    delta[m] = sel_rsa - rnd_rsa
                    d_f[m].append(delta[m])
                    ed_f[m].append(sel["ED"][m] / nanmean([r["ED"][m] for r in rand]))
                o_f.append(delta[s] - nanmean([delta[m] for m in MODELS if m != s]))
            o_pp[p] = nanmean(o_f)
            for m in MODELS:
                delta_pp[m][p] = nanmean(d_f[m])
                ed_pp[m][p] = nanmean(ed_f[m])
        d3["own_selection_effect"][s] = {"participants": o_pp, "mean": nanmean(list(o_pp.values())),
                                         "negative_participants": int(sum(v < 0 for v in o_pp.values()))}
        d3["delta"][s] = {m: {"participants": delta_pp[m], "mean": nanmean(list(delta_pp[m].values()))} for m in MODELS}
        d3["ED_ratio"][s] = {m: nanmean(list(ed_pp[m].values())) for m in MODELS}
    summary["D3"] = d3

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "d123_summary.json").write_text(json.dumps(summary, indent=1))
    render(summary, a.figure_dir)
    print(json.dumps({"cells_present": summary["cells_present"], "R0": summary["R0_reproduction"]}))
    return 0 if not missing else 2


def render(summary, figure_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)

    def tidy(ax, title):
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, lw=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # A: D1 decomposition of the native gap (primary sectors).
    ax = axes[0]
    for i, m in enumerate(VLMS):
        e = summary["D1"][m]["primary"]
        for j, (comp, hatch) in enumerate((("E_val", None), ("E_vec", "////"))):
            x = i + (j - 0.5) * 0.34
            ax.bar(x, e["mean"][comp], width=0.3, color=COLORS[m] if hatch is None else "white",
                   edgecolor=COLORS[m], hatch=hatch, linewidth=1.5, zorder=2)
            ax.scatter([x] * 5, [e["participants"][p][comp] for p in PARTICIPANTS], s=14, color=INK,
                       alpha=0.6, zorder=3, edgecolors="white", linewidths=0.6)
        ax.scatter([i], [e["mean"]["gap"]], marker="D", s=46, color=INK, zorder=4, edgecolors="white", linewidths=1.2)
    ax.set_xticks(range(3), VLMS)
    ax.set_ylabel("Contribution to native RSA gap vs DINOv2 (Fisher z)")
    tidy(ax, "A · D1 eigenvalue / eigenvector split")
    ax.text(0.02, 0.98, "solid = eigenvalues (spectrum)\nhatched = eigenvectors\n◆ = native gap", transform=ax.transAxes,
            fontsize=7.5, color=INK2, va="top")

    # B: D2 gap at matched flexibility.
    ax = axes[1]
    for m in VLMS:
        g = summary["D2"]["gap_vs_DINOv2"][m]
        ys = [g[r]["mean"] for r in RUNGS]
        ax.plot(range(4), ys, color=COLORS[m], lw=2, marker="o", ms=6, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(3.08, ys[-1], m, color=INK, fontsize=8, va="center")
    ax.set_xticks(range(4), ["R0\nnative", "R1\nα exponent", "R2\nper-axis", "R3\nencoding"])
    ax.set_xlim(-0.25, 3.6)
    ax.set_ylabel("VLM − DINOv2 held-out RSA (Fisher z)")
    tidy(ax, "B · D2 gap at matched flexibility")

    # C: D3 own-selection effect.
    ax = axes[2]
    for i, s in enumerate(MODELS):
        e = summary["D3"]["own_selection_effect"][s]
        ax.bar(i, e["mean"], width=0.55, color=COLORS[s], zorder=2)
        ax.scatter([i] * 5, list(e["participants"].values()), s=14, color=INK, alpha=0.6, zorder=3,
                   edgecolors="white", linewidths=0.6)
    ax.set_xticks(range(4), MODELS)
    ax.set_ylabel("Own-selection effect O(s) (Fisher z)")
    tidy(ax, "C · D3 selection in own space")
    fig.suptitle("EP22 D1–D3 · exploratory development evidence (five participants, EP04-exposed responses)",
                 x=0.01, ha="left", color=INK, fontsize=11.5, fontweight="bold")
    fig.text(0.01, -0.04, "Dots: participants (mean over five folds and three high-level sectors). Bars and lines: "
             "participant-balanced means. Negative O(s): selecting images to maximize a model's own ED lowers "
             "that model's RSA more than the other models'.", ha="left", va="top", color=INK2, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(figure_dir / f"d123_results.{ext}", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
