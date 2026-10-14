import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


P = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
STATS = [
    "primary", "frequency_preserving", "visual_residual",
    "taxonomy_residual", "taxonomy_visual_residual",
]
LABELS = [
    "primary", "frequency null", "minus low-level\nvisual",
    "minus taxonomy", "minus taxonomy\n+ visual",
]
ROIS = ["laion-ventral", "laion-lateral", "laion-dorsal", "retinotopic-EVC"]
BANDS = ["band_1_8", "band_9_16", "band_17_32"]
MODELS = ["CLIP", "PEcore", "SigLIP2", "DINOv2"]
BLUE, ORANGE, GRAY, INK, GRID = (
    "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#e4e3df"
)


def mean_field(rows, field):
    return float(np.mean([row[field] for row in rows]))


def dots(ax, x, y, color, hollow=False, marker="o", size=6.5, label=None):
    ax.plot(
        x, y, linestyle="none", marker=marker, markersize=size,
        color=color, markerfacecolor="none" if hollow else color,
        markeredgecolor=color if hollow else "white",
        markeredgewidth=0.9, label=label, zorder=3,
    )


def null_tick(ax, x, y, color=GRAY, label=None):
    ax.plot(
        x, y, linestyle="none", marker="_", markersize=11,
        markeredgewidth=1, color=color, label=label, zorder=2,
    )


def curve(ax, x, y, color, label, linestyle="-", marker="o"):
    ax.plot(
        x, y, color=color, label=label, linestyle=linestyle,
        linewidth=1.1, marker=marker, markersize=6.5,
        markeredgecolor="white", markeredgewidth=0.8, zorder=3,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)

    tasks = report["tasks"]
    pooled = tasks["pooled"]["sources"]
    brain = {
        p: {roi: tasks[p]["sources"]["vg"]["rois"][roi]["brain"]
            for roi in ROIS}
        for p in P
    }

    plt.rcParams.update({
        "font.size": 8.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(
        1, 4, figsize=(17, 4.4),
        gridspec_kw={"width_ratios": [1.45, 1, 1, 1.15]},
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.28,
                        top=0.78, wspace=0.38)

    # A: participant summaries, with equal participant weighting.
    ax = axes[0]
    jitter = np.linspace(-0.065, 0.065, len(P))
    for src, offset, color, label in [
        ("vg", -0.18, BLUE, "Visual Genome"),
        ("captions", 0.18, ORANGE, "captions"),
    ]:
        for i, stat in enumerate(STATS):
            rows = [tasks[p]["sources"][src]["summary"][stat] for p in P]
            x = i + offset
            dots(ax, x + jitter, [r["omega"] for r in rows], color,
                 label=label if i == 0 else None)
            null_tick(ax, x, mean_field(rows, "null_mean"), color=INK)
            dots(ax, x, mean_field(rows, "omega"), color,
                 hollow=True, size=8)
    dots(ax, [], [], INK, hollow=True, size=8, label="participant mean")
    null_tick(ax, [], [], color=INK, label="null mean")
    ax.set_title("A  Brain: operator vs.\nparticipant subspaces", loc="left")
    ax.set_xticks(range(5), LABELS, rotation=20, ha="right",
                  rotation_mode="anchor")
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylabel("subspace overlap ω (q = 8; mean of 3 sectors)")

    # B: participant means for each ROI.
    ax = axes[1]
    for i, roi in enumerate(ROIS):
        for stat, offset, hollow, label in [
            ("primary", -0.2, False, "primary"),
            ("taxonomy_residual", 0, True, "minus taxonomy"),
        ]:
            rows = [brain[p][roi]["statistics"][stat] for p in P]
            dots(ax, i + offset, mean_field(rows, "omega"), BLUE,
                 hollow=hollow, label=label if i == 0 else None)
            null_tick(ax, i + offset, mean_field(rows, "null_mean"))
        rows = [brain[p][roi]["self_consistency"] for p in P]
        dots(ax, i + 0.2, mean_field(rows, "omega"), GRAY, marker="D",
             label="split-half self-consistency" if i == 0 else None)
    ax.set_title("B  Brain by sector (vg)", loc="left")
    ax.set_xticks(range(4), ["ventral", "lateral", "dorsal", "EVC (descr.)"])
    ax.set_xlim(-0.55, 3.55)

    # C: vg operator; brain averages exclude the descriptive EVC ROI.
    ax = axes[2]
    x = np.arange(3)
    band_rows = [
        [brain[p][roi]["sensitivities"][band] for p in P for roi in ROIS[:3]]
        for band in BANDS
    ]
    curve(ax, x, [mean_field(r, "omega") for r in band_rows],
          BLUE, "operator–brain (vg)")
    curve(ax, x, [mean_field(r, "null_mean") for r in band_rows],
          GRAY, "operator–brain null", linestyle="--")
    for has_dino, label, linestyle, marker in [
        (False, "VLM–VLM", "-", "s"),
        (True, "VLM–DINOv2", ":", "^"),
    ]:
        pairs = [
            value for name, value in pooled["vg"]["model_pairs"].items()
            if ("DINOv2" in name.split("__")) == has_dino
        ]
        if len(pairs) != 3:
            raise ValueError(f"Expected three {label} pairs; found {len(pairs)}")
        values = [np.mean([pair[band]["omega"] for pair in pairs])
                  for band in BANDS]
        curve(ax, x, values, INK, label, linestyle, marker)
    ax.set_title("C  Coarse-to-fine bands", loc="left")
    ax.set_xticks(x, ["1–8", "9–16", "17–32"])
    ax.set_xlim(-0.2, 2.2)

    # D: pooled model summaries, coloured by operator source.
    ax = axes[3]
    for src, stat, offset, color, hollow, label in [
        ("vg", "primary", -0.27, BLUE, False, "vg primary"),
        ("vg", "taxonomy_residual", -0.09, BLUE, True, "vg minus taxonomy"),
        ("captions", "primary", 0.09, ORANGE, False, "captions primary"),
        ("captions", "taxonomy_residual", 0.27, ORANGE, True,
         "captions minus taxonomy"),
    ]:
        for i, model in enumerate(MODELS):
            row = pooled[src]["models"][model][stat]
            dots(ax, i + offset, row["omega"], color, hollow=hollow,
                 label=label if i == 0 else None)
            null_tick(ax, i + offset, row["null_mean"])
    ax.set_title("D  Models: operator vs.\nmodel subspaces (pooled)", loc="left")
    ax.set_xticks(range(4), MODELS)
    ax.set_xlim(-0.55, 3.55)

    for i, ax in enumerate(axes):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRAY)
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(length=3, width=0.6, labelsize=8)
        ax.margins(y=0.18)
        ax.set_ylim(bottom=0)
        if i:
            ax.set_ylabel("subspace overlap ω")
        ax.legend(
            loc="best", frameon=False, fontsize=7.2, labelcolor=INK,
            ncol=2 if i == 0 else 1, handlelength=1.8,
            handletextpad=0.5, columnspacing=0.8,
        )
    axes[2].set_yscale("linear")
    axes[2].set_ylim(0, 0.8)

    fig.suptitle(
        "EP23 A0 · measured co-occurrence operator vs. model and brain "
        "subspaces (exploratory; exposed LAION development data)",
        fontsize=11, y=0.98,
    )
    fig.text(
        0.055, 0.035,
        "Dots: participants (A) or participant means (B–C); nulls: "
        "concept-permutation means of each statistic's own null; "
        "models and model–model are descriptive.",
        ha="left", va="bottom", fontsize=8.5,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.out_dir / f"a0_operator_alignment.{extension}",
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
