"""Render EP22 confirmation from --report PATH into --out-dir PATH."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

COLORS = {
    "CLIP": "#2a78d6",
    "PEcore": "#eb6834",
    "SigLIP2": "#1baf7a",
    "DINOv2": "#52514e",
}
INK = "#0b0b0b"
GRID = "#e4e3df"
PARTICIPANTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
OFFSETS = np.linspace(-0.16, 0.16, len(PARTICIPANTS))


def participant_values(mapping):
    if set(mapping) != set(PARTICIPANTS):
        raise ValueError("Expected exactly the five EP22 participants.")
    values = np.array([mapping[p] for p in PARTICIPANTS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Participant values must be finite.")
    return values


def dots(ax, x, values, color):
    ax.scatter(
        x + OFFSETS, values, s=49, color=color,
        edgecolors="white", linewidths=0.65, zorder=4,
    )


def bar_panel(ax, models, data, value_key, mean_key, labels):
    for x, model in enumerate(models):
        values = participant_values(data[model][value_key])
        mean = float(data[model][mean_key])
        if not np.isfinite(mean):
            raise ValueError(f"Non-finite mean for {model}.")
        color = COLORS[model]
        ax.bar(
            x, mean, width=0.52,
            color=matplotlib.colors.to_rgba(color, 0.18),
            edgecolor=color, linewidth=0.7, zorder=2,
        )
        dots(ax, x, values, color)
    ax.set_xticks(range(len(models)), labels)
    ax.set_xlim(-0.5, len(models) - 0.5)


def native_gap(cells, participant, category, rois):
    sectors = [cells[f"{participant}|{category}|{roi}"] for roi in rois]
    return np.mean([
        float(cell["native|CLIP"]) - float(cell["native|DINOv2"])
        for cell in sectors
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    evaluation = report["evaluation"]
    settings = report["settings"]
    categories = list(settings["confirmatory_categories"])
    rois = list(settings["primary_rois"])
    margin = float(settings["margin_fisher_z"])

    if len(categories) != 4 or set(categories) != {
        "cropped", "relations", "selfmade", "unusual"
    }:
        raise ValueError("Expected the four locked confirmatory categories.")
    if len(rois) != 3 or set(rois) != {
        "laion-ventral", "laion-lateral", "laion-dorsal"
    }:
        raise ValueError("Expected the three primary sectors.")
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("The locked margin must be finite and positive.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.usetex": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "axes.grid": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    bar_panel(
        axes[0], ("CLIP", "SigLIP2", "PEcore"),
        evaluation["C2"]["within_category_detail"], "E_val", "E_val_mean",
        ("CLIP\nconfirmatory", "SigLIP2\nconfirmatory", "PEcore\nsecondary"),
    )
    bar_panel(
        axes[1], ("CLIP", "PEcore", "SigLIP2", "DINOv2"),
        evaluation["C3"], "delta_participants", "delta_mean",
        ("CLIP", "PEcore", "SigLIP2", "DINOv2\nnot a prediction"),
    )

    # Each column contains five sector-balanced participant estimates.
    gaps = np.array([
        [native_gap(evaluation["cells"], p, c, rois) for c in categories]
        for p in PARTICIPANTS
    ])
    if not np.isfinite(gaps).all():
        raise ValueError("Native gaps must be finite; missing data are not dropped.")
    # Equal category weighting within each participant gives the final group.
    gaps = np.column_stack((gaps, gaps.mean(axis=1)))
    axes[2].axhspan(
        -margin, margin, color=COLORS["DINOv2"],
        alpha=0.10, linewidth=0, zorder=0,
    )
    for x, values in enumerate(gaps.T):
        dots(axes[2], x, values, COLORS["CLIP"])
        axes[2].scatter(
            x + 0.29, values.mean(), marker="D", s=49, color=INK,
            edgecolors="white", linewidths=0.65, zorder=5,
        )
    axes[2].set_xticks(range(gaps.shape[1]), categories + ["all four"])
    axes[2].set_xlim(-0.5, gaps.shape[1] - 0.5)

    titles = (
        "A  C2a: spectral penalty\nwithin categories",
        "B  C3: development-fitted α\ntransfer to OOD",
        "C  C1: native CLIP − DINOv2\nby confirmatory category",
    )
    ylabels = (
        r"$E_{\mathrm{val}}$ (Fisher z)",
        "Corrected − native RSA (Fisher z)",
        "Native RSA gap (Fisher z)",
    )
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=12)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.axhline(0, color=COLORS["DINOv2"], linewidth=0.8, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=0.7, length=3, labelsize=9)
        ax.margins(y=0.12)

    fig.suptitle(
        "EP22 · sealed OOD confirmation (five participants, procedurally sealed)",
        fontsize=15, fontweight="bold", y=0.97,
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.78, wspace=0.34)
    fig.text(
        0.065, 0.055,
        "Dots are participants; bars and diamonds are participant-balanced means. "
        f"Panel C shading is ±{margin:.3f} Fisher z (locked margin).",
        fontsize=9, color=INK,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "ood_confirmation.png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out_dir / "ood_confirmation.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
