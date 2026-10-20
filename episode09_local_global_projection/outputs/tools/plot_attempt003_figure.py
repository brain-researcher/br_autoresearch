#!/usr/bin/env python3
"""attempt003 figure: layer calibration and within-animal donor support.

Usage: plot_attempt003_figure.py STAGEA_REPORT STAGEB_REPORT OUT_PNG
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DRIVER = {"Cux2": "Cux2", "Plxnd1": "Plxnd1", "Tlx3": "Tlx3"}


def main():
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})

    # Panel A: recorded layer versus coordinate-derived (D25) layer
    la = a["layer_agreement"]
    rows = ["L2/3", "L4", "L5", "L6"]
    cols = ["L1", "L2/3", "L4", "L5", "L6", "undetermined"]
    conf = la["confusion_recorded_by_derived"]
    m = np.array([[conf.get(r, {}).get(c, 0) for c in cols] for r in rows], dtype=float)
    frac = m / m.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    ax1.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if m[i, j]:
                ax1.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=8.5,
                         color="white" if frac[i, j] > 0.45 else INK)
    ax1.set_xticks(range(len(cols)), ["L1", "L2/3", "L4", "L5", "L6", "none"])
    ax1.set_yticks(range(len(rows)), [f"{r} (n={int(m[i].sum())})" for i, r in enumerate(rows)])
    ax1.set_xlabel("Layer derived from CCFv3 soma coordinates (25 µm annotation;\nnone = soma voxel outside an isocortex layer)")
    ax1.set_ylabel("Recorded (curated) layer")
    ax1.tick_params(length=0)
    for s in ax1.spines.values():
        s.set_visible(False)
    lo, hi = la["raw_ci95_animal_bootstrap"]
    klo, khi = la["kappa_ci95_animal_bootstrap"]
    ax1.set_title(f"A  Coordinate layers miss the fill condition\n"
                  f"agreement {la['raw']:.2f} [{lo:.2f}–{hi:.2f}], κ {la['kappa']:.2f} [{klo:.2f}–{khi:.2f}]; "
                  f"needed ≥0.80 and κ ≥0.60", loc="left", fontsize=9.5, color=INK)

    # Panel B: largest eligible within-animal pool per IT-driver animal
    cen = b["census"]
    sc = [("S0_recorded_only", "Recorded layers only (Oct 6 bound)", dict(marker="o", mfc=SURFACE, mec=MUTED, mew=1.5, ms=7)),
          ("P_recorded_then_peng2021_no_D25_fill", "Primary: + Peng 2021 curated layers", dict(marker="o", mfc=BLUE, mec=SURFACE, mew=1.5, ms=8.5)),
          ("S2_P_plus_interior_D25_fills_unvalidated", "+ interior coordinate fills (unvalidated)", dict(marker="^", mfc=INK2, mec=SURFACE, mew=1.0, ms=7))]
    animals = sorted(cen["P_recorded_then_peng2021_no_D25_fill"]["per_animal"],
                     key=lambda k: (-cen["P_recorded_then_peng2021_no_D25_fill"]["per_animal"][k]["largest_pool"], k))
    geno = {}
    ledger = Path(sys.argv[2]).with_name("cell_nonaxon_ledger.csv")
    import csv
    with open(ledger) as f:
        for r in csv.DictReader(f):
            geno.setdefault(r["animal"], r["genotype"])
    labels = []
    for an in animals:
        g = geno.get(an, "")
        d = next((v for k, v in DRIVER.items() if g.startswith(k)), "?")
        labels.append(f"{an}  {d}")
    y = np.arange(len(animals))[::-1]
    ax2.axvline(5, color=INK2, lw=1.0, zorder=1)
    ax2.text(5.3, 6.0, "donor floor: 5 cells\n(4 nonself donors\n+ recipient)", fontsize=8, color=INK2, va="center")
    offs = [0.26, 0.0, -0.26]
    for (key, lab, style), off in zip(sc, offs):
        xs = [cen[key]["per_animal"][an]["largest_pool"] for an in animals]
        ax2.plot(xs, y + off, ls="none", label=f"{lab}: {cen[key]['supported_animals']} animals ≥5", zorder=3, **style)
    ax2.set_yticks(y, labels)
    ax2.set_xlim(-0.8, 20)
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_xlabel("Largest within-animal pool of eligible IT cells (source × layer × driver)")
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax2.set_title("B  At most 3 of 12 IT-driver animals can supply matched donors\n"
                  "design needs 20 animals (12 dev + 8 audit); unadopted route needs 9",
                  loc="left", fontsize=9.5, color=INK)
    fig.text(0.01, 0.005, "SEU-A1876 metadata only; no axonal or projection-class value read. EP09 metadata_feasibility attempt003, Slurm job 48155880.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
