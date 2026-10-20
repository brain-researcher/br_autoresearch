#!/usr/bin/env python3
"""One six-panel EP19 figure and three summary lines from closed summaries."""
from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import premature_r_diagnostic as P
from premature_r_diagnostic import D

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED, INK, SECONDARY = "#898781", "#0b0b0b", "#52514e"
GRID, AXIS, FACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CURVES = (
    ("linear", BLUE, "linear"),
    ("quadratic", ORANGE, "quadratic"),
    ("dev_linear_q10", AQUA, "dev linear q10"),
    ("dev_linear_q50", YELLOW, "dev linear q50"),
)


def marker_handle(label, color, marker="o", hollow=False):
    return Line2D(
        [], [], color=color, marker=marker, markersize=5.5, linestyle="none",
        markerfacecolor=FACE if hollow else color, label=label)


def interval(ax, y, metric, color, marker="D"):
    estimate, ci = metric.get("estimate"), metric.get("ci95", [None, None])
    if P.finite(estimate) and all(P.finite(v) for v in ci):
        ax.hlines(y, ci[0], ci[1], color=color, linewidth=1.5)
        ax.plot(estimate, y, marker, color=color, markersize=6)
    else:
        ax.text(
            .02, y, "not finitely identified", transform=ax.get_yaxis_transform(),
            fontsize=7, color=SECONDARY, va="center")


def person_labels(result):
    chosen = set(result["decision_people"])
    return [
        p if p in chosen else f"{p} (excluded)"
        for p in D.PEOPLE
    ]


def prd_forest(ax, result, feature, title, xlabel):
    selected = set(result["decision_people"])
    all_metric = result["all_people_summaries"][feature]
    for index, p in enumerate(D.PEOPLE):
        item = all_metric["people"][p]
        if item["included"] and P.finite(item.get("delta")):
            inside = p in selected
            ax.plot(
                item["delta"], index, "o", markersize=5.5,
                color=BLUE if inside else MUTED,
                markerfacecolor=BLUE if inside else FACE)
        else:
            ax.text(
                .02, index, "censored / unavailable",
                transform=ax.get_yaxis_transform(), fontsize=7,
                color=SECONDARY, va="center")
    interval(ax, 16, result["decision_summaries"][feature], INK)
    interval(ax, 17, result["KP"]["KPB"][feature], ORANGE, "s")
    interval(ax, 18, result["KP"]["KPA"][feature], AQUA, "s")
    labels = person_labels(result) + [
        "", f"Decision median (n={len(selected)})", "KPB synthetic", "KPA synthetic"]
    ax.set_yticks(range(19), labels, fontsize=7.3)
    ax.set_ylim(18.8, -.8)
    ax.axvline(0, color=AXIS, linestyle=":", linewidth=1.5)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.legend(handles=[
        marker_handle("Decision people", BLUE),
        marker_handle("Excluded people", MUTED, hollow=True),
        marker_handle("Decision median", INK, "D"),
        marker_handle("KPB", ORANGE, "s"),
        marker_handle("KPA", AQUA, "s"),
    ], fontsize=6.7, ncol=2, loc="best")


def drift_panel(ax, result):
    selected = set(result["decision_people"])
    series = (
        ("drift_explained", BLUE, -.15, "Drift explained"),
        ("prior_activity", ORANGE, .15, "Prior activity"),
    )
    for feature, color, offset, label in series:
        metric = result["all_people_summaries"][feature]
        for index, p in enumerate(D.PEOPLE):
            item = metric["people"][p]
            if item["included"] and P.finite(item.get("delta")):
                ax.plot(
                    item["delta"], index + offset, "o", color=color,
                    markerfacecolor=color if p in selected else FACE,
                    markersize=5.5)
        interval(ax, 16 + offset, result["decision_summaries"][feature], color)
    ax.set_yticks(
        list(range(15)) + [16],
        person_labels(result) + ["Decision medians"], fontsize=7.3)
    ax.set_ylim(16.8, -.8)
    ax.axvline(0, color=AXIS, linestyle=":", linewidth=1.5)
    ax.set(
        title="(C) Drift explanation and prior activity",
        xlabel="Tail minus reference proportion; aggregate 95% person CI")
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.legend(handles=[
        marker_handle("Drift explained", BLUE),
        marker_handle("Prior activity", ORANGE),
        marker_handle("Excluded person", MUTED, hollow=True),
    ], fontsize=7, loc="best")


def control_person_panel(ax, result, *, sham=False):
    bounds = []
    old_sub02 = None
    for index, p in enumerate(D.PEOPLE):
        precision = result["people"][p]["precision"]["5"]
        if sham:
            item = precision["CP4"]
            old = item["v5_sham"]["estimate"]
            new = item["all_background_sham"]
        else:
            item = precision["families"]["linear"]
            old = item["v5_metrics"]["near"]["estimate"]
            new = item["metrics"]["near"]
        if P.finite(old):
            ax.plot(index - .12, old, "x", color=MUTED, markersize=6)
            bounds.append(float(old))
            if p == "sub-02":
                old_sub02 = float(old)
        if P.finite(new["estimate"]):
            low, high = new["ci95"]
            if P.finite(low) and P.finite(high):
                ax.vlines(index + .12, low, high, color=BLUE, linewidth=1.5)
                bounds.extend([float(low), float(high)])
            ax.plot(index + .12, new["estimate"], "o", color=BLUE, markersize=5.5)
    gate = .05 if sham else .90
    ax.axhline(gate, color=INK, linestyle="--", linewidth=1.5)
    ax.set_xticks(range(len(D.PEOPLE)), D.PEOPLE, rotation=55, ha="right", fontsize=7)
    ax.grid(axis="y", color=GRID, linewidth=.5)
    ax.legend(handles=[
        marker_handle("v5: 32 backgrounds", MUTED, "x"),
        marker_handle("All eligible backgrounds", BLUE),
        Line2D([], [], color=INK, linestyle="--", linewidth=1.5,
               label=f"Gate {gate:.2f}"),
    ], fontsize=7, loc="best")
    if sham:
        ymax = max(.12, max(bounds, default=.05) + .055)
        ax.set(
            title="(F) Sham alarms at k = 5",
            ylabel="Alarm fraction; 95% background CI", ylim=(-.005, ymax))
        if old_sub02 is not None:
            ax.annotate(
                "v5 failure retained", xy=(-.12, old_sub02),
                xytext=(1.1, min(ymax - .015, old_sub02 + .045)),
                color=SECONDARY, fontsize=7.5,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1.0})
    else:
        ymin = max(0.0, min(bounds, default=.85) - .06)
        ax.set(
            title="(D) Linear near sensitivity at k = 5",
            ylabel="Sensitivity; 95% background CI", ylim=(ymin, 1.025))


def margin_panel(ax, result):
    ends = []
    for family, color, label in CURVES:
        curve = result["population"][family]["curve"]
        xs, ys, lows, highs, counts = [], [], [], [], []
        for target in sorted(curve, key=int):
            item = curve[target]
            if (item["n_people"] and P.finite(item["estimate"])
                    and all(P.finite(v) for v in item["ci95"])):
                xs.append(int(target))
                ys.append(item["estimate"])
                lows.append(item["ci95"][0])
                highs.append(item["ci95"][1])
                counts.append(item["n_people"])
        if xs:
            count_label = str(counts[0]) if len(set(counts)) == 1 else f"{min(counts)}-{max(counts)}"
            ax.plot(
                xs, ys, marker="o", markersize=5, linewidth=1.5, color=color,
                label=f"{label} (n={count_label})")
            ax.fill_between(xs, lows, highs, color=color, alpha=.13, linewidth=0)
            ends.append((ys[-1], xs[-1], label))
    ax.axvline(288, color=INK, linestyle=":", linewidth=1.5)
    ax.axhline(.9, color=INK, linestyle="--", linewidth=1.5)
    ends.sort()
    if ends:
        positions = []
        for y, _, _ in ends:
            positions.append(max(y, positions[-1] + .065) if positions else y)
        excess = max(0, positions[-1] - 1.015)
        positions = [y - excess for y in positions]
        if positions[0] < .02:
            shift = .02 - positions[0]
            positions = [y + shift for y in positions]
        for (y, x, label), label_y in zip(ends, positions):
            ax.annotate(
                label, xy=(x, y), xytext=(323, label_y), fontsize=7.2,
                color=SECONDARY, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": .7})
    else:
        ax.text(.5, .5, "No covered curves", transform=ax.transAxes,
                ha="center", color=SECONDARY)
    ax.set(
        title="(E) Margin curves: equal-weight people",
        xlabel="Original true targets (ms); bands: 95% person CI",
        ylabel="P(recovered lag ≥ 288 ms)", xlim=(270, 350), ylim=(-.02, 1.06))
    ax.set_xticks([272, 288, 304, 320])
    ax.grid(color=GRID, linewidth=.5)
    if ends:
        ax.legend(fontsize=6.8, loc="lower right")


def formatted(value, digits=3):
    return f"{float(value):.{digits}f}" if P.finite(value) else "unidentified"


def metric_text(metric, digits=1):
    return (
        f"{formatted(metric['estimate'], digits)} "
        f"(95% CI {formatted(metric['ci95'][0], digits)} to "
        f"{formatted(metric['ci95'][1], digits)})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prd-dir", type=Path, required=True)
    parser.add_argument("--cpc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    D.require_slurm(1)
    prd, cpc = args.prd_dir.resolve(), args.cpc_dir.resolve()
    km = P.check_stage(prd / "known_positive", "prd_known_positive")
    pm = P.check_stage(prd / "features", "prd_features")
    sm = P.check_stage(cpc / "strata", "cpc_strata")
    cm = P.check_stage(cpc / "controls", "cpc_controls")
    provider = P.check_stage(prd / "provider", "prd_provider")
    if any(m["health_only"] for m in (km, pm, sm, cm, provider)):
        raise RuntimeError("Health-mode plotting is forbidden")
    feature_hash = D.digest(prd / "features/closed.json")
    if (
        pm["known_positive_sha256"] != D.digest(prd / "known_positive/closed.json")
        or sm["prd_features_sha256"] != feature_hash
        or cm["prd_features_sha256"] != feature_hash
        or cm["strata_sha256"] != D.digest(cpc / "strata/closed.json")
        or provider["prd_features_sha256"] != feature_hash
        or provider["cpc_controls_sha256"] != D.digest(cpc / "controls/closed.json")
        or Path(cm["run_root"]).resolve() != cpc
        or Path(pm["run_root"]).resolve() != prd
    ):
        raise RuntimeError("Plot predecessor closures differ")
    if args.output_dir.is_symlink() or args.output_dir.resolve() != cpc / "figure":
        raise RuntimeError("Figure output must be the new CPC run's figure directory")
    output = D.fresh(args.output_dir)
    prd_result = D.read_json(prd / "features/decision.json")
    cpc_result = D.read_json(cpc / "controls/decision.json")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": SECONDARY, "ytick.color": SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": .8,
        "axes.facecolor": FACE, "figure.facecolor": FACE,
        "legend.frameon": False, "legend.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 5.5,
        "axes.titlesize": 10,
    })
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), facecolor=FACE)
    axes[0, 0].axvspan(-50, 50, color=MUTED, alpha=.12, linewidth=0)
    prd_forest(
        axes[0, 0], prd_result, "H", "(A) Substantial-departure lead",
        "ΔH (ms); aggregate bars: 95% person CI")
    axes[0, 0].axvline(100, color=MUTED, linestyle="--", linewidth=1.5)
    prd_forest(
        axes[0, 1], prd_result, "CE", "(B) Early sensor-vector continuity",
        "ΔC_E; aggregate bars: 95% person CI")
    axes[0, 1].axvline(-.2, color=MUTED, linestyle="--", linewidth=1.5)
    drift_panel(axes[0, 2], prd_result)
    control_person_panel(axes[1, 0], cpc_result)
    margin_panel(axes[1, 1], cpc_result)
    control_person_panel(axes[1, 2], cpc_result, sham=True)

    cp1 = cpc_result["CP1_k5_near_counts"]
    counts = [
        cp1.get(name, 0)
        for name in ("BELOW_GATE", "CROSSES_GATE", "ABOVE_GATE", "UNCOVERED")
    ]
    title = (
        "EP19 detector v2 / raw-onset v5 follow-up diagnostics — "
        "development calibration/diagnostic evidence; post hoc")
    subtitle = (
        f"PRD {prd_result['outcome']} / {prd_result['status']}; "
        f"decision set {len(prd_result['decision_people'])}/15 people; "
        "CPC CP1 k=5 near below/across/above/uncovered = "
        + "/".join(str(n) for n in counts))
    fig.suptitle(title, fontsize=12, y=.995)
    fig.text(.5, .963, subtitle, ha="center", fontsize=9, color=SECONDARY)
    fig.text(
        .02, .012,
        "A–C show individual person contrasts and aggregate person uncertainty; "
        "D/F show within-person background precision.\n"
        "Sensor-frame continuity is not reach direction or reach identity. "
        "Sub-02's reported v5 sham=0.070 and two unanchorable backgrounds are retained.\n"
        "Both v5 candidates remain UNDERIDENTIFIED; no detector is frozen.",
        fontsize=8, color=SECONDARY)
    fig.tight_layout(rect=(0, .09, 1, .94), h_pad=2.0, w_pad=1.5)
    image_path = output / "v5_diagnostics.png"
    with image_path.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=300, facecolor=FACE)
    plt.close(fig)

    population = cpc_result["population"]
    linear = population["linear"]
    q10, q50 = population["dev_linear_q10"]["all_near"], population["dev_linear_q50"]["all_near"]
    cp2 = cpc_result["CP2_k5_gated_counts"]
    lines = [
        f"PRD: {prd_result['outcome']} "
        f"({prd_result['status'].lower().replace('_', ' ')}), "
        f"{len(prd_result['decision_people'])} of 15 people; "
        f"delta H = {metric_text(prd_result['decision_summaries']['H'], 1)} ms; "
        f"delta early cosine = {metric_text(prd_result['decision_summaries']['CE'], 3)}. "
        f"{prd_result['next_design_implication']}",
        "CPC: mean linear near sensitivity was "
        f"{formatted(linear['v5_near']['estimate'])} with v5's 32 backgrounds and "
        f"{formatted(linear['all_near']['estimate'])} with all eligible backgrounds; "
        f"development linear q10/q50 means were {formatted(q10['estimate'])}/"
        f"{formatted(q50['estimate'])} across {q10['n_people']}/{q50['n_people']} people. "
        f"CP1 k=5 near: {counts[0]} below, {counts[1]} across, {counts[2]} above "
        f"the gate and {counts[3]} uncovered; "
        f"{cp2.get('TIMING_ERROR_CONSISTENT', 0)} gated cells were consistent with timing error.",
        "Status/limits: development calibration/diagnostic evidence; post hoc. "
        "D2 and D1 remain v5 UNDERIDENTIFIED; no detector is frozen and sensor continuity "
        "does not establish reach identity. Sub-02's reported v5 sham=0.070 and two "
        "unanchorable backgrounds are retained.",
    ]
    D.text_file(output / "summary.txt", "\n".join(lines) + "\n")
    D.close(
        output, "v5_diagnostics_figure",
        new_script_sha256=P.code_hashes(),
        prd_features_sha256=feature_hash,
        cpc_controls_sha256=D.digest(cpc / "controls/closed.json"),
        png_dpi=300, independent_human_unit="participant")
    print("\n".join(lines), flush=True)
    print(image_path, flush=True)


if __name__ == "__main__":
    main()
