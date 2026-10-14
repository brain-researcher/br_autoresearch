"""EP23 stage-2 figure, rendered from collected reports and A1 cells only.

A: A1 held-out RSA by metric family (rank 64, with the rank-32 sensitivity).
B: A2 coarse vs fine subspace overlap (model-brain, brain self-consistency, model-model).
C: A3 synthetic pooling simulation: preferred power alpha* by topography.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


P = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
MODELS = ["CLIP", "PEcore", "SigLIP2", "DINOv2"]
PRIMARY = ["laion-ventral", "laion-lateral", "laion-dorsal"]
FAMILIES = ["native", "power", "diagonal", "full"]
FAMILY_LABELS = ["native", "power\nreweighting", "diagonal\nmetric", "full\nmetric"]
SCALES = ["coarse", "fine"]
TOPOGRAPHIES = ["dct_ordered", "dct_shuffled", "smooth_random", "white_random"]
TOPOGRAPHY_LABELS = ["DCT, frequency-ordered", "DCT, shuffled", "smooth random", "white random"]
MEASUREMENTS = ["identity", "pool_0", "pool_1", "pool_2", "pool_4"]
MEASUREMENT_LABELS = ["none", "σ = 0", "σ = 1", "σ = 2", "σ = 4"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated)
MARKERS = ["o", "s", "^", "D"]
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def a1_levels(cells_dir):
    """Participant means over 5 folds x 4 models x 3 primary sectors."""
    values = {p: {f: [] for f in ("reference", *FAMILIES)} for p in P}
    for path in sorted(Path(cells_dir).glob("sub-*-fold-*.json")):
        cell = load(path)
        for model in MODELS:
            for roi in PRIMARY:
                row = cell["models"][model]["rois"][roi]
                values[cell["participant"]]["reference"].append(row["reference_cosine_z"])
                for family in FAMILIES:
                    values[cell["participant"]][family].append(row["metrics"][family]["rsa_z"])
    expected = 5 * len(MODELS) * len(PRIMARY)
    if any(len(v) != expected for d in values.values() for v in d.values()):
        raise ValueError(f"{cells_dir}: expected {expected} values per participant and family")
    return {p: {f: float(np.mean(v)) for f, v in d.items()} for p, d in values.items()}


def mean_alpha(report, model):
    counts = report["selected_distributions"]["primary"][model]["power"]["counts"]
    return sum(float(v) * n for v, n in counts.items()) / sum(counts.values())


def marks(ax, x, y, color, marker="o", size=6, hollow=False, **kwargs):
    ax.plot(x, y, linestyle="none", marker=marker, markersize=size,
            markerfacecolor="white" if hollow else color,
            markeredgecolor=color if hollow else "white",
            markeredgewidth=1.1 if hollow else 0.8, **kwargs)


def panel_a1(ax, levels, reports):
    x = np.arange(len(FAMILIES))
    offsets = (-0.1, 0.14)
    for p in P:
        y = [levels[0][p][f] for f in FAMILIES]
        ax.plot(x + offsets[0], y, color=MUTED, linewidth=0.8, zorder=1)
        marks(ax, x + offsets[0], y, MUTED, size=5, zorder=2)
        marks(ax, x + offsets[1], [levels[1][p][f] for f in FAMILIES], MUTED,
              size=5, hollow=True, zorder=2)
    for level, offset, hollow in zip(levels, offsets, (False, True)):
        means = [np.mean([level[p][f] for p in P]) for f in FAMILIES]
        marks(ax, x + offset, means, INK, size=8, hollow=hollow, zorder=3)
    reference = np.mean([levels[0][p]["reference"] for p in P])
    ax.axhline(reference, color=SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ax.text(-0.42, reference + 0.006, f"cosine RSA, all dimensions ({reference:.2f})",
            fontsize=7.2, color=SECONDARY, va="bottom")
    delta = [r["group"]["mean_over_models"]["full_minus_diagonal"] for r in reports]
    k_min = min(r["group"][m]["full_minus_diagonal"]["k_positive"]
                for r in reports for m in MODELS)
    ax.text(0.02, 0.98,
            f"full − diagonal: Δz = {delta[0]['mean_delta_z']:+.2f} (rank 64), "
            f"{delta[1]['mean_delta_z']:+.2f} (rank 32)\n"
            f"positive in {k_min}/5 participants for every model; "
            f"sign p = {delta[0]['sign_p_one_sided']:.3f}",
            transform=ax.transAxes, fontsize=7.6, va="top", color=INK)
    handles = [
        Line2D([], [], color=MUTED, linewidth=0.8, marker="o", markersize=5,
               markeredgecolor="white", label="participant, rank 64"),
        Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor="white",
               markeredgecolor=MUTED, label="participant, rank 32"),
        Line2D([], [], linestyle="none", marker="o", markersize=8, color=INK,
               markeredgecolor="white", label="mean, rank 64"),
        Line2D([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="white",
               markeredgecolor=INK, label="mean, rank 32"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=7.2)
    ax.set_title("A  A1 · metric families on held-out images", loc="left")
    ax.set_xticks(x, FAMILY_LABELS)
    ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("held-out RSA, Fisher z\n(mean of 4 models × 3 sectors × 5 folds)")


def panel_a2(ax, a2):
    parts, x = a2["participants"], np.arange(len(SCALES))
    for p in P:
        rois = parts[p]["rois"]
        brain = [np.mean([rois[r]["models"][m][s]["kappa"] for r in PRIMARY for m in MODELS])
                 for s in SCALES]
        self_k = [np.mean([rois[r]["self_consistency"][s]["kappa"] for r in PRIMARY])
                  for s in SCALES]
        ax.plot(x + 0.06, self_k, color=SECONDARY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
        marks(ax, x + 0.06, self_k, SECONDARY, marker="D", size=5.5, zorder=2)
        ax.plot(x - 0.06, brain, color=INK, linewidth=0.9, zorder=2)
        marks(ax, x - 0.06, brain, INK, size=6.5, zorder=3)
    pairs = list(a2["pooled"]["model_pairs"].values())
    for i, scale in enumerate(SCALES):
        kappa = [pair[scale]["kappa"] for pair in pairs]
        ax.plot([i + 0.26] * 2, [min(kappa), max(kappa)], color=INK, linewidth=0.9, zorder=2)
        marks(ax, i + 0.26, np.mean(kappa), INK, marker="s", size=7, hollow=True, zorder=3)
    raw, primary = a2["group"]["raw_kappa_sign_test"], a2["group"]["primary_sign_test"]
    ax.text(0.98, 0.78,
            f"model–brain κ, coarse > fine:\n{raw['k_positive']}/{raw['n_nonnull']} participants, "
            f"p = {raw['p_one_sided']:.3f}\n"
            f"reliability-relative (primary):\n{primary['k_positive']}/{primary['n_nonnull']}, "
            f"p = {primary['p_one_sided']:.2f}",
            transform=ax.transAxes, fontsize=7.6, va="top", ha="right", color=INK)
    handles = [
        Line2D([], [], color=INK, linewidth=0.9, marker="o", markersize=6.5,
               markeredgecolor="white", label="model–brain (participant)"),
        Line2D([], [], color=SECONDARY, linewidth=0.8, linestyle=(0, (4, 2)), marker="D",
               markersize=5.5, markeredgecolor="white",
               label="brain self-consistency (participant)"),
        Line2D([], [], color=INK, linewidth=0.9, marker="s", markersize=7,
               markerfacecolor="white", label="model–model, pooled (mean, range of 6 pairs)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.2)
    ax.set_title("B  A2 · coarse vs fine alignment", loc="left")
    ax.set_xticks(x, ["coarse\n(category means)", "fine\n(within category)"])
    ax.set_xlim(-0.35, 1.55)
    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines["left"].set_bounds(0, 1)
    ax.set_ylabel("chance-corrected subspace overlap κ")


def panel_a3(ax, a3, alpha_hat):
    rows = {(c["topography"], c["measurement"]): c["metrics"]["alpha_star"]
            for c in a3["conditions"] if c["noise"] == "none"}
    x = np.arange(len(MEASUREMENTS))
    low, high = min(alpha_hat.values()), max(alpha_hat.values())
    ax.axhspan(low, high, color=GRID, linewidth=0, zorder=0)
    ax.text(-0.38, (low + high) / 2, f"A1 selected α̂, real data\n(4 model means, {low:.1f}–{high:.1f})",
            fontsize=7.2, color=SECONDARY, va="center")
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(2, color=SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
    ends = []
    for i, topography in enumerate(TOPOGRAPHIES):
        mean = np.array([rows[topography, m]["mean"] for m in MEASUREMENTS])
        sd = np.array([rows[topography, m]["sd"] for m in MEASUREMENTS])
        xi = x + (i - 1.5) * 0.07
        ax.errorbar(xi, mean, yerr=sd, color=SERIES[i], linewidth=1.4, elinewidth=0.8,
                    capsize=0, marker=MARKERS[i], markersize=6, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3, label=TOPOGRAPHY_LABELS[i])
        ends.append((mean[-1], xi[-1], i))
    placed = None
    for y, xe, i in sorted(ends):  # Direct labels, spaced at least 0.08 apart.
        placed = y if placed is None else max(y, placed + 0.08)
        ax.annotate(TOPOGRAPHY_LABELS[i], xy=(xe + 0.04, y), xytext=(x[-1] + 0.35, placed),
                    fontsize=7.2, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=SERIES[i], linewidth=0.9,
                                    shrinkA=0, shrinkB=0))
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=SECONDARY, linewidth=0.9, linestyle=(0, (4, 3)), label="α = 2"),
                Line2D([], [], color=MUTED, linewidth=0.8, linestyle=":", label="α = 1, native geometry"),
                Patch(color=GRID, label="A1 selected α̂ (real data)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.2, ncol=2,
              columnspacing=1.0, handlelength=1.8)
    ax.set_title("C  A3 · pooling simulation (synthetic, no noise)", loc="left")
    ax.set_xticks(x, MEASUREMENT_LABELS)
    ax.set_xlim(-0.45, len(MEASUREMENTS) + 1.35)
    ax.set_ylim(0.4, 2.75)
    ax.set_xlabel("measurement: 4×4 block average after Gaussian blur σ (voxels)")
    ax.set_ylabel("preferred power α* (mean ± sd, 20 seeds)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-cells", type=Path, nargs=2, required=True, help="rank 64, rank 32")
    parser.add_argument("--a1-reports", type=Path, nargs=2, required=True, help="rank 64, rank 32")
    parser.add_argument("--a2-report", type=Path, required=True)
    parser.add_argument("--a3-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [load(path) for path in args.a1_reports]
    if [r["settings"]["rank"] for r in reports] != [64, 32]:
        raise ValueError("--a1-reports must be the rank-64 then the rank-32 report")
    levels = [a1_levels(path) for path in args.a1_cells]
    alpha_hat = {m: mean_alpha(reports[0], m) for m in MODELS}

    plt.rcParams.update({
        "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "xtick.color": INK, "ytick.color": INK, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1.2, 1, 1.45]})
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.24, top=0.84, wspace=0.3)
    panel_a1(axes[0], levels, reports)
    panel_a2(axes[1], load(args.a2_report))
    panel_a3(axes[2], load(args.a3_report), alpha_hat)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(length=3, width=0.6, labelsize=8)

    fig.suptitle("EP23 stage 2 · metric families, coarse vs fine alignment, and pooling",
                 fontsize=11, y=0.98)
    fig.text(0.06, 0.02,
             "A–B: exploratory, exposed LAION-fMRI development data; 5 participants are the "
             "independent units; one-sided exact sign tests. B: participant subspaces use r = 3–4 "
             "modes from 7–10 categories; model–model uses the pooled set (r = 8).\n"
             "C: synthetic, model-based simulation (no biological claim); moderate noise changes "
             "α* by ≤ 0.01. The band marks the per-model mean α̂ that A1 selected on real data.",
             ha="left", va="bottom", fontsize=8, color=SECONDARY)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.out_dir / f"ep23_stage2.{extension}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
