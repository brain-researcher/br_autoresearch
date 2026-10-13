#!/usr/bin/env python3
"""Plot the complete, already-computed EP12 outgoing descriptive tables.

Reads only the fixed description01 report and its two CSVs. No source matrix,
snapshot, incoming profile, final focal profile, model fit or inference is read
or computed. Matplotlib is imported only in the selected compute allocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import socket
import time


EPISODE = "episode12_malecns_type_sufficiency"
RUN = "outputs/runs/observed_outgoing_description/20261005-description01"
FIGURE_ROOT = "outputs/figures/observed_outgoing_description"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fraction(value: str) -> float:
    result = float(value)
    require(math.isfinite(result) and 0 <= result <= 1, "Undefined or invalid saved fraction/TV")
    return result


def load_tables(root: Path) -> tuple[dict, dict, dict]:
    """Retain all types and both sides; quoted provider names use proper CSV parsing."""
    directory = root / RUN
    report = json.loads((directory / "descriptive_report.json").read_text(encoding="utf-8"))
    require(report.get("complete") is True
            and report.get("scope") == "descriptive_already_open_development_outgoing_only_one_male"
            and report.get("types_reported") == report.get("types_with_two_positive_mass_sides") == 816
            and report.get("type_side_rows_reported") == report.get("type_sides_with_positive_mass_profiles") == 1632
            and report.get("focal_neurons") == 84735 and report.get("partner_bins") == 11771
            and report.get("zero_mass_neurons") == 126
            and all(report.get(key) is False for key in ("final_focal_profiles_opened",
                "incoming_profiles_constructed", "historical_search_or_generated_chain_resumed")),
            "The exact completed development-only description is required")
    require(report.get("neuron_weighting") == "equal_among_positive_mass_neurons_within_type_and_side"
            and report.get("centroid_rule") == "neuron_equal_arithmetic_mean_in_full_raw_partner_vocabulary"
            and report.get("within_side_metric") == "mean_TV_of_each_positive_mass_profile_to_its_own_side_centroid"
            and report.get("between_side_metric") == "TV_between_the_two_neuron_equal_side_centroids",
            "Saved descriptive definitions differ")
    sides = {}
    with (directory / "outgoing_type_side_descriptions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["provider_type"], row["side"])
            require(key not in sides and key[0] and key[1] in {"L", "R"}, "Duplicate or invalid type/side")
            require(row["status"] == "positive_mass_profiles_described", "An expected side is undefined")
            item = {field: int(row[field]) for field in (
                "neurons", "positive_mass_neurons", "zero_mass_neurons", "outgoing_count")}
            require(all(value >= 0 for value in item.values()) and item["positive_mass_neurons"] > 0
                    and item["neurons"] == item["positive_mass_neurons"] + item["zero_mass_neurons"],
                    "Invalid saved side accounting")
            item["tv"] = fraction(row["mean_tv_to_side_centroid"])
            item["unknown"] = fraction(row["mean_untyped_or_missing_fraction"])
            sides[key] = item
    bilateral = {}
    with (directory / "outgoing_bilateral_descriptions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            provider_type = row["provider_type"]
            require(provider_type not in bilateral and row["status"] == "both_sides_described",
                    "Duplicate or undefined bilateral row")
            for side in ("L", "R"):
                require((provider_type, side) in sides and all(
                    int(row[f"{side}_{field}"]) == sides[(provider_type, side)][field]
                    for field in ("neurons", "positive_mass_neurons", "zero_mass_neurons")),
                    "Side and bilateral support differ")
            bilateral[provider_type] = fraction(row["between_side_centroid_tv"])
    require(len(bilateral) == 816 and len(sides) == 1632
            and set(sides) == {(name, side) for name in bilateral for side in ("L", "R")},
            "Every development type and both sides must appear in the figure")
    require(sum(row["neurons"] for row in sides.values()) == report["focal_neurons"]
            and sum(row["zero_mass_neurons"] for row in sides.values()) == report["zero_mass_neurons"],
            "Saved total support differs from the report")
    return report, sides, bilateral


def render(root: Path, output: Path, check) -> None:
    report, sides, bilateral = load_tables(root)
    check()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52606b", "axes.labelcolor": "#24333e",
        "xtick.color": "#52606b", "ytick.color": "#52606b",
        "text.color": "#24333e", "svg.fonttype": "none",
    })
    names = sorted(bilateral)
    blue, orange = "#227a9d", "#d17636"
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="white")
    fig.subplots_adjust(left=.060, right=.980, top=.770, bottom=.275, wspace=.35)
    fig.text(.060, .955, "Observed outgoing profiles across 816 development types", fontsize=18, weight="bold")
    fig.text(.060, .900, "One male specimen | Complete descriptive tables | All 11,771 partner bins retained", fontsize=11)

    axis = axes[0]
    axis.plot([0, 1], [0, 1], color="#a8b2bb", linestyle="--", linewidth=1, zorder=1)
    axis.scatter([sides[(name, "L")]["tv"] for name in names],
                 [sides[(name, "R")]["tv"] for name in names],
                 s=14, alpha=.40, color=blue, edgecolors="none", zorder=2)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Left: mean TV to own side centroid",
             ylabel="Right: mean TV to own side centroid", title="A  Within-side dispersion")
    axis.text(.04, .96, "One point per annotated type", transform=axis.transAxes,
              va="top", fontsize=9, color="#52606b")

    axis = axes[1]
    axis.hist([bilateral[name] for name in names], bins=[index / 20 for index in range(21)],
              color=blue, edgecolor="white", linewidth=.6)
    axis.set(xlim=(0, 1), xlabel="TV between left and right centroids",
             ylabel="Number of development types", title="B  Between-side difference")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.text(.96, .96, "All 816 types", transform=axis.transAxes, ha="right", va="top",
              fontsize=9, color="#52606b")

    axis = axes[2]
    for side, color, label in (("L", blue, "Left"), ("R", orange, "Right")):
        axis.scatter([sides[(name, side)]["positive_mass_neurons"] for name in names],
                     [sides[(name, side)]["unknown"] for name in names], s=12,
                     alpha=.30, color=color, edgecolors="none", label=label)
    axis.set_xscale("log")
    axis.set(ylim=(0, 1), xlabel="Positive-mass neurons per type and side\n(log scale)",
             ylabel="Mean untyped / missing partner fraction", title="C  Support and unknown mass")
    axis.legend(frameon=False, loc="upper right", markerscale=1.8, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", color="#e3e8ec", linewidth=.6, zorder=0)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)

    positive = sum(row["positive_mass_neurons"] for row in sides.values())
    fig.text(.060, .160,
             f"{report['focal_neurons']:,} neurons: {positive:,} positive-mass profiles; "
             f"{report['zero_mass_neurons']:,} zero-mass neurons counted but excluded from composition.", fontsize=10)
    fig.text(.060, .108, "In-sample summaries: each neuron contributes to its own type-side centroid. "
             "Dispersion depends on support; unknown partner mass stays in every denominator.", fontsize=9)
    fig.text(.060, .063, "No population inference, subtype test, held-out prediction or bilateral reproducibility claim. "
             "The provider taxonomy is connectivity-informed.", fontsize=9, color="#52606b")
    check()
    png = output / "observed_outgoing_description.png"
    svg = output / "observed_outgoing_description.svg"
    try:
        fig.savefig(png.with_suffix(".png.pending"), format="png", dpi=220, facecolor="white")
        check()
        fig.savefig(svg.with_suffix(".svg.pending"), format="svg", facecolor="white")
        check()
    finally:
        plt.close(fig)
    caption = (
        "# Observed outgoing composition in one male fly\n\n"
        "All 816 already-open development types are shown, without selecting types by their outcomes. "
        "A: each point compares left and right mean total-variation (TV) distances from positive-mass "
        "neuron profiles to their own neuron-equal type-side centroid; the dashed line marks equality. "
        "B: the complete distribution of TV between those two centroids, in fixed 0.05-wide bins. "
        "C: all 1,632 type-side observations show positive-mass support and the neuron-equal fraction "
        "of outgoing composition assigned to untyped/status or missing-annotation partner bins.\n\n"
        f"The accounting covers {report['focal_neurons']:,} neurons: {positive:,} have positive outgoing "
        f"mass, while {report['zero_mass_neurons']:,} have undefined composition and are explicitly "
        "counted rather than imputed. Every type has positive-mass profiles on both sides. "
        "All 11,771 raw partner bins, including unknown mass, remain in the composition denominator.\n\n"
        "These are in-sample summaries within one specimen. Each neuron contributes to its own "
        "centroid, so dispersion depends on sample size. The connectivity-informed provider taxonomy "
        "and prior development exposure remain limitations. No tests, population intervals, inferred "
        "subtypes, model winners, bilateral reproducibility or circuit consequences are established. "
        "The original T/U/M study remains unresolved; incoming profiles and 204 final focal types "
        "remain closed.\n\n"
        "Sources: the complete `outgoing_type_side_descriptions.csv`, "
        "`outgoing_bilateral_descriptions.csv` and `descriptive_report.json` in "
        f"`{root / RUN}`. No source snapshot or new neural values were read.\n"
    )
    pending_caption = output / "figure_caption.md.pending"
    pending_caption.write_text(caption, encoding="utf-8")
    check()
    png.with_suffix(".png.pending").replace(png)
    svg.with_suffix(".svg.pending").replace(svg)
    pending_caption.replace(output / "figure_caption.md")
    check()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname()
            and os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and all(os.environ.get(name) == "1" for name in THREAD_ENV),
            "Use the selected compute-node Slurm step with one CPU and all six thread caps=1")
    require(1 <= args.max_seconds <= 120, "Select an explicit bound of 1..120 seconds")
    root = (Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch" / EPISODE).resolve()
    output = args.output_dir.resolve()
    require((root / FIGURE_ROOT).resolve() in output.parents and not output.exists(),
            "A fresh explicit output directory under EP12's descriptive figure root is required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def check() -> None:
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError("Selected figure-rendering deadline reached")

    def interrupted(*_) -> None:
        raise TimeoutError("Figure rendering interrupted or deadline reached; partial files preserved")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        render(root, output, check)
        check()
        print(f"EP12 complete-table figure saved to {output}; elapsed={time.monotonic()-started:.3f}s", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
