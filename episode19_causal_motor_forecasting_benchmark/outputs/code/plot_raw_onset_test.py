#!/usr/bin/env python3
"""Render one EP19 diagnostic PNG at 160 dpi, inside a Slurm allocation.

Usage:
  python outputs/code/plot_raw_onset_test.py --input-dir RUN_OUTPUT_DIRECTORY
Optional:
  --output /absolute/path/raw_onset_test.png
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import json
import socket
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


KS = (3, 5, 8)
COLORS = {3: "#2a78d6", 5: "#eb6834", 8: "#1baf7a"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
MARGIN = 288


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname().lower():
        raise RuntimeError("Rendering requires an actual Slurm compute-node shell")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_number(value):
    if value in ("", None):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require_slurm()

    folder = args.input_dir.resolve()
    summary = read_json(folder / "raw_onset_summary.json")
    post = read_json(folder / "post_lock_comparison.json")
    raw = read_csv(folder / "raw_onsets.csv")
    controls = read_csv(folder / "positive_control.csv")
    participants = summary["participants"]

    if len(raw) != 3 * summary["accepted_events"]:
        raise ValueError("Unexpected onset/threshold row count")
    if post["accepted_events"] != summary["accepted_events"]:
        raise ValueError("Provider comparison has a different event count")
    if post.get("decision_summary_written_before_provider_access") is not True:
        raise ValueError("Missing Stage 3-before-provider ordering record")

    output = args.output or folder / "raw_onset_test.png"
    if output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.7))
    for ax in axes.flat:
        style_axis(ax)

    # (a) ECDF bounds over all accepted events, retaining unresolved rows.
    ax = axes[0, 0]
    for k in KS:
        rows = [row for row in raw if int(row["k"]) == k]
        lower = np.sort(np.asarray([float(row["lag_lower_ms"]) for row in rows]))
        upper = np.sort(np.asarray([float(row["lag_upper_ms"]) for row in rows]))
        grid = np.unique(np.r_[0.0, MARGIN, lower, upper[np.isfinite(upper)]])
        grid = np.r_[grid, max(grid[-1] + 1, 350)]

        cdf_lower = np.searchsorted(upper, grid, side="right") / len(rows)
        cdf_upper = np.searchsorted(lower, grid, side="right") / len(rows)
        unresolved = sum(row["status"] != "determinate" for row in rows)
        ax.step(
            grid, cdf_lower, where="post", color=COLORS[k],
            label=f"k={k}; unresolved {unresolved}/{len(rows)}",
        )
        ax.step(
            grid, cdf_upper, where="post", color=COLORS[k],
            linestyle="--", linewidth=1, alpha=0.8,
        )
        ax.fill_between(
            grid, cdf_lower, cdf_upper, step="post", color=COLORS[k], alpha=0.10
        )

    ax.axvline(MARGIN, color=MUTED, linestyle=":", label="288 ms")
    ax.set(
        title="(a) Primary lag ECDF with censoring bounds",
        xlabel="L − R (ms)", ylabel="Fraction of all accepted events",
        ylim=(0, 1.02), xlim=(0, None),
    )
    ax.legend(fontsize=8, loc="lower right")

    # (b) Actual detectable truth, not target or physical onset-to-label lag.
    ax = axes[0, 1]
    all_values = [0.0, 650.0]
    for k in KS:
        trials = [
            row for row in controls if int(row["k"]) == k and row["kind"] == "injected"
        ]
        usable = [row for row in trials if row["control_status"] == "ok"]
        pairs = [
            (float(row["true_lag_ms"]), float(row["recovered_lag_ms"]))
            for row in usable
            if row["status"] == "determinate"
            and finite_number(row["recovered_lag_ms"]) is not None
        ]
        label = (
            f"k={k}; unresolved {len(usable) - len(pairs)}, "
            f"unusable {len(trials) - len(usable)}"
        )
        if pairs:
            array = np.asarray(pairs)
            all_values.extend(array.ravel().tolist())
            ax.scatter(
                array[:, 0], array[:, 1], color=COLORS[k], s=7, alpha=0.18,
                edgecolors="none", rasterized=True, label=label,
            )
        else:
            ax.scatter([], [], color=COLORS[k], s=12, label=label)

    bound = max(all_values) * 1.04
    ax.plot([0, bound], [0, bound], color=MUTED, linewidth=1, label="Identity")
    ax.axvline(MARGIN, color=MUTED, linestyle=":", linewidth=1)
    ax.axhline(MARGIN, color=MUTED, linestyle=":", linewidth=1)
    ax.set(
        title="(b) Positive controls: detectable-onset truth",
        xlabel="True detectable lag, L − T_det (ms)",
        ylabel="Recovered primary lag, L − R (ms)",
        xlim=(0, bound), ylim=(0, bound),
    )
    ax.legend(fontsize=7.5, loc="upper left")

    # (c) Per-person p95; bars are censoring bounds, not confidence intervals.
    ax = axes[1, 0]
    person_metrics = summary["decision"]["per_person"]
    finite_values = [MARGIN]
    for k in KS:
        for participant in participants:
            item = person_metrics[str(k)][participant]
            for field in ("p95_lower_ms", "p95_upper_ms", "p95_determinate_ms"):
                if item[field] is not None:
                    finite_values.append(float(item[field]))

    top = max(350.0, max(finite_values) * 1.15 + 20)
    offsets = {3: -0.20, 5: 0.0, 8: 0.20}
    for k in KS:
        for index, participant in enumerate(participants):
            item = person_metrics[str(k)][participant]
            x = index + offsets[k]
            lower = float(item["p95_lower_ms"])
            upper, dot = item["p95_upper_ms"], item["p95_determinate_ms"]
            if upper is not None:
                ax.vlines(x, lower, upper, color=COLORS[k], linewidth=1.6)
                ax.plot(
                    [x, x], [lower, upper], linestyle="none",
                    marker="_", color=COLORS[k], markersize=6,
                )
            else:
                ax.annotate(
                    "", xy=(x, top * 0.96), xytext=(x, lower),
                    arrowprops={"arrowstyle": "->", "color": COLORS[k], "lw": 1.1},
                )
                ax.plot(x, lower, marker="_", color=COLORS[k], markersize=6)
            if dot is not None:
                ax.plot(x, dot, "o", color=COLORS[k], markersize=4)

    ax.axhline(MARGIN, color=MUTED, linestyle=":")
    ax.set_xticks(np.arange(len(participants)))
    ax.set_xticklabels(participants, rotation=55, ha="right")
    ax.set(
        title="(c) Per-person primary p95 lag",
        xlabel="Development participant", ylabel="Lag (ms)", ylim=(0, top),
    )
    handles = [
        Line2D([], [], marker="o", color=COLORS[k], label=f"k={k}", markersize=4)
        for k in KS
    ]
    handles.extend([
        Line2D([], [], color=MUTED, linestyle=":", label="288 ms"),
        Line2D([], [], color=MUTED, linewidth=1, label="Bars: censoring bounds"),
    ])
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    # (d) Provider agreement after the decision lock.
    ax = axes[1, 1]
    values_by_k = {
        k: np.asarray(
            post["by_k"][str(k)]["R_minus_AccStartIndex"]["values_ms"], dtype=float
        )
        for k in KS
    }
    nonempty = [values for values in values_by_k.values() if len(values)]
    if nonempty:
        combined = np.concatenate(nonempty)
        low, high = float(combined.min()), float(combined.max())
        if low == high:
            low, high = low - 2, high + 2
        bins = np.linspace(low, high, 51)
        for k in KS:
            values = values_by_k[k]
            counts, _ = np.histogram(values, bins=bins)
            ax.stairs(
                counts, bins, color=COLORS[k], linewidth=1.5,
                label=f"k={k}; n={len(values)}",
            )
    else:
        ax.text(
            0.5, 0.5, "No matched determinate estimates",
            transform=ax.transAxes, ha="center", color=MUTED,
        )
        for k in KS:
            ax.plot([], [], color=COLORS[k], label=f"k={k}; n=0")
    ax.axvline(0, color=MUTED, linestyle=":", label="Zero difference")
    ax.set(
        title="(d) Post-lock provider agreement",
        xlabel="Primary R − AccStartIndex (ms)", ylabel="Event count",
    )
    ax.legend(fontsize=8)

    outcome = summary["decision"]["outcome"]
    pc = "PASS" if summary["positive_control_primary_pass"] else "FAIL"
    fig.suptitle(
        f"EP19 self-paced raw-onset diagnostic — {summary['scope']}\n"
        f"{outcome}; all-participant primary controls: {pc}",
        fontsize=13, color=TEXT, y=0.985,
    )
    fig.text(
        0.02, 0.015,
        "Post hoc revision after job 48227088; its three development participants remain included.\n"
        "ECDF shading and p95 bars bracket unresolved events; dots are determinate-only p95; "
        "arrows denote unbounded upper p95.\n"
        "Controls use 2.8-s backgrounds and shorter searches than the main maximum. "
        "No sampling confidence intervals or worst-case certification.",
        fontsize=8.1, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.94))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
