#!/usr/bin/env python3
"""Plot only released compact PLS basis/head diagnostics and preserved references."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket


SUMMARY_RELATIVE = "runs/adaptive_search/primary_history_pls_basis_summary_attempt001/compact_evidence_attempt001.json"
MAX_SUMMARY_BYTES = 512*1024
POLICIES = ("HW", "blend", "paired_PLS", "basis_only", "full_target_control")
LABELS = ("Weighted history (HW)", "Current/history blend (W/HW)", "Paired PLS (P)",
          "Shuffled basis + true head (B)", "Shuffled basis + shuffled head (F)")
CONTRASTS = ("paired_PLS_minus_basis_only", "basis_only_minus_full_target")
CONTRAST_LABELS = ("Paired − shuffled projection\nwith true heads (P − B)",
                   "True − shuffled head targets\nwith the same projections (B − F)")
ANIMALS = ("Mihili", "Chewie")
METRICS = (("primary_increment", "primary_by_animal", .01, "Primary: full velocity − baseline"),
           ("identity_increment", "identity_by_animal", .005, "Identity: correct − wrong trial"))
REPLAY_FLAGS = ("all_new_saved_scores_replayed", "all_original_anchor_scores_replayed",
                "original_selectors_independently_replayed", "all_stored_transform_heads_replayed")


def aggregate_rows(records, names, margins):
    """Copy exact released scalar values, with equal animal weights checked."""
    rows = []
    support = None
    for name in names:
        record = records[name]
        sessions = record["sessions"]
        ids = {row["session_id"] for row in sessions}
        if (len(sessions) != 16 or len(ids) != 16
                or sum(row["animal"] == "Mihili" for row in sessions) != 6
                or sum(row["animal"] == "Chewie" for row in sessions) != 10
                or (support is not None and ids != support)):
            raise ValueError("Complete common sixteen-session/two-animal panels required")
        support = ids
        for metric, animal_key, threshold, _ in METRICS:
            mean = float(record[metric])
            by_animal = record[animal_key]
            if set(by_animal) != set(ANIMALS):
                raise ValueError("Exactly two original animal summaries required")
            values = [float(by_animal[animal]) for animal in ANIMALS]
            if not all(math.isfinite(value) for value in (mean, *values)):
                raise ValueError("Nonfinite released aggregate")
            if not math.isclose(mean, sum(values)/2, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Released mean must retain equal animal weighting")
            for unit, value in (("animal_balanced_mean", mean), *zip(ANIMALS, values)):
                rows.append({"policy_or_contrast": name, "metric": metric, "unit": unit,
                    "value": value, "registered_margin": threshold if margins else None,
                    "evidence": "exposed_development_two_animals",
                    "source": "outputs/"+SUMMARY_RELATIVE})
    return rows


def released_aggregates(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SUMMARY_BYTES+1)
    if len(raw) > MAX_SUMMARY_BYTES:
        raise ValueError("Only the compact evidence artifact of at most 512 KiB is permitted")
    document = json.loads(raw)
    if (document.get("status") != "complete_history_pls_basis_panel"
            or document.get("recipe") != "lmp_history4_repeatability_weight_pls2_basis_labels8_true_head_v1"
            or document.get("promotion_authorized") is not False
            or document.get("new_selector_fitted") is not False
            or document.get("updates_primary_incumbent") is not False
            or document.get("candidate_trial_count_increment") != 0
            or document.get("literal_null_control_count_increment") != 0
            or document.get("candidate_and_basis_capacity_matched") is not False
            or document.get("basis_and_full_target_score_Grams_identical") is not True
            or any(document.get(key) is not True for key in REPLAY_FLAGS)):
        raise ValueError("Complete replayed, preserved-selector compact basis diagnostic required")
    records = {name: document["original_anchors"][name]
               for name in ("HW", "blend", "paired_PLS", "full_target_control")}
    records["basis_only"] = document["basis_only"]
    return (aggregate_rows(records, POLICIES, margins=True),
            aggregate_rows(document["reference_contrasts"], CONTRASTS, margins=False))


def axis_limits(rows, margins):
    values = [row["value"] for row in rows]+[0.]
    if margins:
        values.extend([.01, .005])
    extent = max(values)-min(values)
    padding = max(extent*.09, .00015)
    return min(values)-padding, max(values)+padding


def draw_panels(rows, names, labels, title, captions, margins, output_stem):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    lookup = {(row["policy_or_contrast"], row["metric"], row["unit"]): row["value"] for row in rows}
    colors = {"Mihili": "#2674A8", "Chewie": "#C56C23"}
    offsets = {"Mihili": .12, "Chewie": -.12}
    height = 7.5 if margins else 6.6
    fig, axes = plt.subplots(1, 2, figsize=(13.1, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=.285, right=.98, top=.76, bottom=.34, wspace=.15)
    fig.suptitle(title, x=.055, ha="left", y=.955, fontsize=16, fontweight="bold")
    fig.text(.055, .90, "16 sessions · two animals · held-out directions · exposed development evidence",
             ha="left", color="#555555", fontsize=10)
    lower, upper = axis_limits(rows, margins)
    for ax, (metric, _, threshold, metric_title) in zip(axes, METRICS):
        ax.set_title(metric_title if margins else metric_title.split(":")[0]+": conditional contrast",
                     loc="left", fontsize=11, pad=12)
        ax.axvline(0, color="#777777", linewidth=1., zorder=0)
        if margins:
            ax.axvline(threshold, color="#777777", linestyle="--", linewidth=1.2, zorder=0)
            ax.text(threshold-.00008, .99, f"margin {threshold:.3f}",
                    transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=9, color="#555555")
        for index, name in enumerate(names):
            y = len(names)-1-index
            for animal in ANIMALS:
                ax.scatter(lookup[name, metric, animal], y+offsets[animal], s=37,
                    color=colors[animal], edgecolors="white", linewidths=.5, zorder=3)
            ax.scatter(lookup[name, metric, "animal_balanced_mean"], y, marker="D", s=42,
                color="#252525", edgecolors="white", linewidths=.6, zorder=4)
        ax.set_xlim(lower, upper)
        ax.set_ylim(-.65, len(names)-.30)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yticks(range(len(names)), list(reversed(labels)))
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", alpha=.16, linewidth=.6)
        ax.set_xlabel(r"Increment in $R^2_{SSE}$" if margins else r"Difference in $R^2_{SSE}$ increments", labelpad=9)
    legend = [Line2D([], [], color="#252525", marker="D", linestyle="none", markersize=6,
                     label="Animal-balanced mean")]
    legend += [Line2D([], [], color=colors[animal], marker="o", linestyle="none", markersize=6,
                      label=animal) for animal in ANIMALS]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.045, .225), ncol=3,
               frameon=False, fontsize=10)
    for y, caption in zip((.195, .155, .115, .075, .035), captions):
        fig.text(.055, y, caption, fontsize=9, color="#444444")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        help="Must be outputs/figures/primary_history_pls_basis; writes two PNG/SVG/CSV figures")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render in the coordinator's verified Slurm compute allocation")
    outputs = Path(__file__).resolve().parents[1]
    expected = outputs/"figures/primary_history_pls_basis"
    output = args.output_dir.resolve() if args.output_dir else expected
    if output != expected:
        raise ValueError("Only the assigned basis-only diagnostic figure directory is permitted")
    absolute, contrasts = released_aggregates(outputs/SUMMARY_RELATIVE)
    if len(absolute) != 30 or len(contrasts) != 12:
        raise ValueError("Exactly thirty absolute and twelve conditional-contrast rows required")
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False})
    draw_panels(absolute, POLICIES, LABELS,
        "EP05 · PLS projection and head correspondence", (
            "P uses true targets for both projection and head. B reuses shuffled projections with true-target heads.",
            "F uses those same shuffled projections and shuffled-target heads. B and F average 8 losses, not predictions.",
            "P/B need not have equal effective capacity. B/F share each projection and P's frozen penalty; references select independently.",
            "Dots are two animal summaries, not confidence intervals; 8 shuffles are not biological replicates.",
            "B is an unpromotable explanatory diagnostic. Dashed margins remain the original practical candidate criteria."),
        True, output/"primary_history_pls_basis_comparison")
    draw_panels(contrasts, CONTRASTS, CONTRAST_LABELS,
        "EP05 · two conditional PLS comparisons", (
            "P − B asks whether paired projection construction helps when every head receives true fitting targets.",
            "B − F asks whether true head targets help on the identical stored shuffled projections and penalties.",
            "Positive values favor the first arm. These are separate conditional comparisons, not a causal decomposition.",
            "P/B can differ in effective capacity. The eight matched draws are averaged losses, not biological replicates.",
            "Two animals; no confidence intervals or population inference. Zero is the directional reference; no new margin is introduced."),
        False, output/"primary_history_pls_basis_conditional_contrasts")
    print(json.dumps({"output_dir": str(output), "policies": 5, "contrasts": 2, "animals": 2,
        "absolute_tidy_rows": len(absolute), "contrast_tidy_rows": len(contrasts),
        "formats": ["png", "svg", "csv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
