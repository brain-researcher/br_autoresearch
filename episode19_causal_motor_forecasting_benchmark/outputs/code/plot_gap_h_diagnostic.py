#!/usr/bin/env python3
"""One EP19 six-panel figure and three lines from closed gap/H summaries."""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import gap_h_diagnostic as G
from gap_h_diagnostic import D

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
MUTED = "#898781"
INK = "#202020"
FACE = "#fcfcfb"
CLASS_COLORS = {"A": BLUE, "B": ORANGE, "C": MUTED}
KP_COLORS = {"KPG": GREEN, "KPB": ORANGE, "KPA": BLUE}


def formatted(value, digits=3):
    if value is None:
        return "unidentified"
    value = float(value)
    if math.isnan(value):
        return "unidentified"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def pair(values, digits=1):
    return "[" + ", ".join(formatted(v, digits) for v in values) + "]"


def metric_text(metric, digits=3):
    return (
        f"{formatted(metric['estimate'], digits)} "
        f"(95% person CI {pair(metric['ci95'], digits)})")


def p95_text(metric):
    return (
        f"{pair(metric['bounds'])} ms; "
        f"95% person CI {pair(metric['ci95'])}")


def panel_a(ax, figure):
    people = figure["people"]
    chosen = set(figure["decision_people"])
    positions = np.arange(len(people))
    bottom = np.zeros(len(people))
    for cls in ("A", "B", "C"):
        values = np.asarray([
            figure["tail_people"][p]["fractions"][cls] for p in people])
        ax.bar(positions, values, bottom=bottom, color=CLASS_COLORS[cls],
               width=.8, label=cls)
        bottom += values
    labels = [
        f"{p[4:]}{'*' if p in chosen else ''}\n"
        f"n={figure['tail_people'][p]['n']}" for p in people]
    ax.set_xticks(positions, labels, fontsize=7)
    ax.set(
        title="(a) Supported-tail classes; * decision set",
        xlabel="Participant", ylabel="Within-person fraction",
        ylim=(0, 1.04))
    ax.legend(ncol=3, fontsize=8, loc="upper right")


def panel_b(ax, figure, decision):
    a = decision["component_a"]
    for key, label, color in (
        ("v5", "Fixed v5 lag", MUTED),
        ("lambda", "Diagnostic lambda* (80 ms)", BLUE),
    ):
        item = figure["survival"][key]
        x = np.asarray(item["x_ms"])
        lower = np.asarray(item["lower"])
        upper = np.asarray(item["upper"])
        ax.fill_between(x, lower, upper, step="post", color=color, alpha=.22)
        ax.step(x, lower, where="post", color=color, label=label)
        ax.step(x, upper, where="post", color=color, alpha=.5, linewidth=.7)
    ax.axvline(288, color=INK, linestyle=":", label="288 ms")
    ax.axhline(.05, color=INK, linestyle="--", linewidth=.8)
    largest = max(
        max(item["x_ms"]) for item in figure["survival"].values())
    ax.set_xscale("log")
    ticks = [t for t in (200, 288, 500, 1000, 2000, 4000) if t <= largest]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set(
        title="(b) All-event survival bounds; all 15 people",
        xlabel="Interval (ms; logarithmic axis)", ylabel="Fraction > interval",
        xlim=(200, largest), ylim=(-.005, None))
    ax.text(
        .98, .97,
        "v5 P95 " + p95_text(a["v5_P95_all"]) + "\n"
        "P95* " + p95_text(a["P95_star_all"]),
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox={"facecolor": FACE, "edgecolor": "none", "alpha": .9})
    ax.legend(fontsize=7, loc="center right")


def panel_c(ax, figure):
    low, high = figure["theta3_theta5_band"]
    ax.axhspan(low, high, color=MUTED, alpha=.25,
               label="Person theta3/theta5 range")
    names = ("Real tail", "KPG", "KPB")
    colors = (BLUE, GREEN, ORANGE)
    for group_index, (name, color) in enumerate(zip(names, colors)):
        values = figure["depth"][name]
        shown = 0
        for person_index, item in enumerate(values):
            if item["median"] is None:
                continue
            offset = (person_index - 7) * .024
            ax.plot(group_index + offset, item["median"], "o",
                    color=color, markersize=4, alpha=.85)
            shown += 1
        ax.text(
            group_index, -.1, f"{shown} people",
            transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7)
    ax.set_xticks(range(3), names)
    ax.set(
        title="(c) Gap depth: one median per person",
        ylabel="Median r / theta5 at non-qualifying starts")
    ax.text(
        .02, .98, "Splitting run; otherwise longest internal run.\n"
        "Rows with no internal run are omitted.",
        transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.legend(fontsize=7, loc="best")


def panel_d(ax, figure):
    people = figure["people"]
    offsets = {"KPG": -.23, "KPB": 0, "KPA": .23}
    known = figure["known"]["people"]
    for i, p in enumerate(people):
        for kind in G.KINDS:
            item = known[p][kind]
            value = item["class_fraction"]["A"]
            if value is None:
                continue
            x = i + offsets[kind]
            color = KP_COLORS[kind]
            ax.plot(
                x, value, "o", color=color, markersize=4.5,
                markerfacecolor=color if item["covered"] else FACE)
            if kind in ("KPB", "KPA"):
                split = item["split_fraction"]
                if split is not None:
                    ax.plot(
                        x, split, "s", color=color, markerfacecolor="none",
                        markersize=5.5)
    ax.axhline(.8, color=GREEN, linestyle=":")
    ax.axhline(.2, color=INK, linestyle=":")
    ax.set_xticks(range(len(people)), [p[4:] for p in people], fontsize=7)
    ax.set(
        title="(d) Known positives at the 288-ms margin",
        xlabel="Participant", ylabel="Usable-event fraction", ylim=(-.04, 1.04))
    handles = [
        Line2D([], [], color=KP_COLORS[k], marker="o", linestyle="none", label=k)
        for k in G.KINDS
    ]
    handles.extend([
        Line2D([], [], color=INK, marker="s", markerfacecolor="none",
               linestyle="none", label="Pre-priority split: specificity gate"),
        Line2D([], [], color=INK, marker="o", markerfacecolor=FACE,
               linestyle="none", label="Under 24 usable"),
    ])
    ax.legend(handles=handles, fontsize=6.7, loc="best")
    ax.text(.02, .98, "Circles: class A; squares: lambda* < 288",
            transform=ax.transAxes, va="top", fontsize=7)


def panel_e(ax, figure, decision):
    counts = figure["counts"]
    for i, row in enumerate(counts["people"]):
        if row["fraction"] is None:
            continue
        lo, hi = row["wilson95"]
        ax.vlines(i, lo, hi, color=ORANGE, linewidth=1.3)
        ax.plot(i, row["fraction"], "o", color=ORANGE, markersize=4.5)
    if counts["pooled_fraction"] is not None:
        ax.axhline(counts["pooled_fraction"], color=INK, linestyle="--")
    ax.set_xticks(
        range(len(figure["people"])),
        [p[4:] for p in figure["people"]], fontsize=7)
    item = decision["component_b"]["T2"]
    ax.set(
        title="(e) Within-tail B prevalence",
        xlabel="Participant",
        ylabel="B fraction; descriptive Wilson 95% count intervals",
        ylim=(-.02, 1.02))
    ax.text(
        .02, .98,
        f"T2 p={formatted(item['test']['p'])}; "
        f"power={formatted(item['power']['estimate'])}\n"
        f"{item['status']}; B={counts['B']}/{counts['N']}",
        transform=ax.transAxes, va="top", fontsize=8)


def panel_f(ax, figure, decision):
    selected = [
        row for row in figure["counts"]["people"] if row["T3_included"]]
    for row in selected:
        x, y = row["half_logits"]
        ax.plot(x, y, "o", color=ORANGE, markersize=5)
        ax.annotate(
            row["participant"][4:], (x, y),
            xytext=(4, 3), textcoords="offset points", fontsize=7)
    item = decision["component_b"]["T3"]
    ax.set(
        title="(f) Run-half alignment of B prevalence",
        xlabel="Half 1 empirical logit", ylabel="Half 2 empirical logit")
    ax.text(
        .02, .98,
        f"n={len(selected)} people; r={formatted(item['test']['r'])}\n"
        f"T3 p={formatted(item['test']['p'])}; "
        f"power={formatted(item['power']['estimate'])}\n"
        f"{item['status']}\n"
        "Null preserves half totals and person denominators.",
        transform=ax.transAxes, va="top", fontsize=7.5)


def summary_lines(decision):
    a = decision["component_a"]
    b = decision["component_b"]
    t2, t3 = b["T2"], b["T3"]
    return [
        f"Component (a): {a['status']}; all-15 diagnostic P95* "
        f"{p95_text(a['P95_star_all'])}; decision set "
        f"{len(a['decision_people'])}/15, P95* "
        f"{p95_text(a['P95_star_decision'])}; "
        f"f_A={metric_text(a['f']['A'])}.",
        f"Component (b): {b['status']}; T2 {t2['status']}, "
        f"p={formatted(t2['test']['p'])}, "
        f"power={formatted(t2['power']['estimate'])}; "
        f"T3 {t3['status']}, p={formatted(t3['test']['p'])}, "
        f"power={formatted(t3['power']['estimate'])}, "
        f"eligible people={b['counts']['T3_n_people']}; "
        "count-model results do not establish reach identity.",
        "Status: development calibration/diagnostic evidence; post hoc. "
        "v5 D2/D1 remain UNDERIDENTIFIED; nothing frozen; "
        "lambda* is diagnostic only. v6, gates and p95-versus-worst-case "
        "remain scientist decisions; no EEG scoring clearance.",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    D.require_slurm(1)
    root = args.run_dir.resolve()
    metadata, predecessors = G.verify_chain(root, "provider")
    if any(m["health_only"] for m in metadata.values()):
        raise RuntimeError("Health-mode plotting is forbidden")
    if (
        root.parent != G.VIEWS.resolve()
        or not root.name.startswith("gap_h_diagnostic_job_")
        or G.PROTECTED.intersection(root.parts)
    ):
        raise RuntimeError("Plot input must be a closed new full gap/H run")
    if metadata["decision"].get("provider_access_has_not_occurred") is not True:
        raise RuntimeError("Decision was not closed before provider access")
    if args.output_dir.is_symlink() or args.output_dir.resolve() != root / "figure":
        raise RuntimeError("Figure output must be this new run's figure directory")

    # Read summaries only. No raw-event CSV, cached trace or provider CSV access.
    decision = D.read_json(root / "decision/decision.json")
    figure = D.read_json(root / "decision/figure_data.json")
    D.read_json(root / "provider/summary.json")
    if (
        decision["v5_D2_outcome"] != "UNDERIDENTIFIED"
        or decision["v5_D1_outcome"] != "UNDERIDENTIFIED"
        or decision["detector_frozen"]
    ):
        raise RuntimeError("Diagnostic acceptance boundary differs")
    output = D.fresh(args.output_dir)
    D.text_file(output / "predeclaration.md", G.DECLARATION)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "figure.facecolor": FACE,
        "axes.facecolor": FACE, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "axes.titlesize": 10, "lines.linewidth": 1.4,
    })
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.18, linewidth=.6)
        ax.set_axisbelow(True)
    panel_a(axes[0, 0], figure)
    panel_b(axes[0, 1], figure, decision)
    panel_c(axes[0, 2], figure)
    panel_d(axes[1, 0], figure)
    panel_e(axes[1, 1], figure, decision)
    panel_f(axes[1, 2], figure, decision)
    fig.suptitle(
        "EP19 fixed-label gap/H diagnostic — "
        "development calibration/diagnostic evidence; post hoc",
        fontsize=13, y=.99)
    fig.text(
        .5, .957,
        f"{decision['component_a']['status']} / "
        f"{decision['component_b']['status']}; fixed D2 labels and v5 k=5",
        ha="center", fontsize=10)
    fig.text(
        .02, .015,
        "A is an operational deep-pause class; B is a closed H>=288-ms departure. "
        "Neither identifies reach onset.\n"
        "P95 shading brackets unresolved events; p95 confidence intervals resample people. "
        "Wilson bars are descriptive count intervals.\n"
        "T2/T3 use conditional event-allocation assumptions; run dependence is unmodelled. "
        "Both v5 candidates remain UNDERIDENTIFIED; nothing is frozen.",
        fontsize=8)
    fig.tight_layout(rect=(0, .095, 1, .94), h_pad=2.8, w_pad=2)
    image_path = output / "ep19_gap_h_diagnostic_20261017.png"
    with image_path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=300, facecolor=FACE)
    plt.close(fig)
    lines = summary_lines(decision)
    if len(lines) != 3 or any("\n" in line for line in lines):
        raise RuntimeError("Summary must contain exactly three lines")
    D.text_file(output / "summary.txt", "\n".join(lines) + "\n")
    D.close(
        output, "ghd_figure",
        new_script_sha256=G.code_hashes(),
        predecessors=predecessors,
        run_root=str(root), health_only=False,
        png_dpi=300, independent_human_unit="participant",
    )
    print("\n".join(lines), flush=True)
    print(image_path, flush=True)


if __name__ == "__main__":
    main()
