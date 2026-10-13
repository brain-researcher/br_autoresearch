#!/usr/bin/env python3
"""Plot completed EP14 diagnostics and replay-check saved baseline predictions.

Reads existing artifacts only; no fitting or input payload/holdout access.
Execute after the diagnostics report completes, in the owned Slurm session.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1]
DEFAULT_RUN = OUTPUTS / "runs/discarded_direction_diagnostics/attempt001"
ORIGINAL = OUTPUTS / "runs/fixed_development_comparison/attempt001/outer"
FIGURES = OUTPUTS / "figures"
ATLASES = ("desikan", "destrieux")
COLORS = {"desikan": "#245B83", "destrieux": "#B85A28"}


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def keyed_rows(record):
    indexed = {}
    for row in record["appearances"]:
        key = (row["id"], row["view"], row["role"])
        if key in indexed:
            raise ValueError("Duplicate cached appearance")
        indexed[key] = row
    return indexed


def replay_check(run):
    checks = []
    atol, rtol = 1e-7, 1e-8
    for atlas in ATLASES:
        for view in range(3):
            new_path = run / "outer" / f"{atlas}_view{view}.json"
            new_record = read_json(new_path)
            if (new_record["atlas"], new_record["view"]) != (atlas, view):
                raise ValueError(f"Unexpected new cache identity: {new_path}")
            new = keyed_rows(new_record)
            for old_arm, geometry in (("R0", "raw"), ("R3", "rescaled")):
                old_path = ORIGINAL / f"{atlas}_{old_arm}_view{view}.json"
                old_record = read_json(old_path)
                if (old_record["atlas"], old_record["arm"], old_record["view"]) != (atlas, old_arm, view):
                    raise ValueError(f"Unexpected original cache identity: {old_path}")
                old = keyed_rows(old_record)
                if set(new) != set(old):
                    raise ValueError(f"Unmatched cached row identities: {old_path}")
                keys = sorted(old)
                if any((old[key]["domain"], old[key]["age"]) != (new[key]["domain"], new[key]["age"]) for key in keys):
                    raise ValueError(f"Unmatched cached domain/age: {old_path}")
                item = {"atlas": atlas, "view": view, "original_arm": old_arm,
                        "original_alpha": old_record["alpha"], "rows": len(keys),
                        "rows_and_metadata_match": True,
                        "new_prediction": f"geometry_{geometry}_alpha1000"}
                if float(old_record["alpha"]) != 1000.0:
                    item["status"] = "skipped_original_selected_different_alpha"
                else:
                    expected = np.array([old[key]["prediction"] for key in keys])
                    actual = np.array([new[key]["predictions"][item["new_prediction"]] for key in keys])
                    item["max_absolute_prediction_difference_years"] = float(np.max(np.abs(actual - expected)))
                    item["status"] = "passed" if np.allclose(actual, expected, atol=atol, rtol=rtol) else "failed"
                checks.append(item)
    failed = [item for item in checks if item["status"] == "failed"]
    compared = [item for item in checks if item["status"] == "passed"]
    result = {"analysis_id": "discarded_direction_diagnostics_baseline_replay",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "failed" if failed else "passed",
              "absolute_tolerance_years": atol, "relative_tolerance": rtol,
              "checks": checks, "compared_cache_count": len(compared),
              "skipped_cache_count": sum(item["status"].startswith("skipped") for item in checks),
              "maximum_compared_prediction_difference_years": max((item["max_absolute_prediction_difference_years"] for item in compared), default=None),
              "model_fitting": False, "holdout_access": False}
    (run / "baseline_replay_check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if failed:
        raise ValueError("Baseline replay failed; inspect baseline_replay_check.json before plotting")
    return result


def panel(report, atlas, population, metric):
    population_panel = report["atlases"][atlas]["pooled"]["external"][population]
    key = "absolute_on_centered_eligible_domains" if population == "common_support" and metric == "absolute" else metric
    return population_panel[key]


def make_figure(report, run):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"height_ratios": [1.35, 1.0]})
    series = [("desikan", "all_rows"), ("destrieux", "all_rows"),
              ("desikan", "common_support"), ("destrieux", "common_support")]
    offsets = (-0.24, -0.08, 0.08, 0.24)
    missing = []
    for column, metric in enumerate(("absolute", "centered")):
        baseline = "age_median" if metric == "absolute" else "centered_zero"
        discarded = [("discarded_site", baseline), ("discarded_site", "discarded_random"),
                     ("discarded_site", "discarded_permuted")]
        geometry = [("geometry_retained_alpha1000", "geometry_raw_alpha1000"),
                    ("geometry_rescaled_alpha1000", "geometry_retained_alpha1000")]
        labels = [("Site scores − training median" if metric == "absolute" else "Site scores − centered zero"),
                  "Site scores − random scores", "Site scores − permuted-site scores"]
        for row_number, (pairs, names) in enumerate(((discarded, labels), (geometry, ["Retained − raw", "Rescaled retained − retained"]))):
            ax = axes[row_number, column]
            for series_number, (atlas, population) in enumerate(series):
                scores = panel(report, atlas, population, metric)
                for index, (other, reference) in enumerate(pairs):
                    contrast = scores["contrasts"].get(f"{other}_minus_{reference}")
                    if contrast is None:
                        missing.append(f"{atlas} {population} {metric}: insufficient support")
                        continue
                    value = contrast["delta_mae_years"]
                    y = len(pairs) - 1 - index + offsets[series_number]
                    interval = contrast.get("descriptive_percentiles_2_5_97_5")
                    # Percentile intervals need not contain the point estimate;
                    # draw their actual endpoints rather than truncating errors.
                    if interval is not None:
                        ax.hlines(y, interval[0], interval[1], color=COLORS[atlas], linewidth=1.4, alpha=0.9)
                        ax.vlines(interval, y - 0.035, y + 0.035, color=COLORS[atlas], linewidth=1.1)
                    else:
                        missing.append(f"{atlas} {population} {metric}: interval unavailable")
                    common = population == "common_support"
                    ax.plot(value, y, marker="D" if common else "o", markersize=5.5,
                            markerfacecolor="white" if common else COLORS[atlas],
                            markeredgecolor=COLORS[atlas], markeredgewidth=1.4, linestyle="none", zorder=4)
            ax.axvline(0, color="#72787E", linestyle=(0, (3, 3)), linewidth=0.9, zorder=0)
            ax.set_yticks(np.arange(len(pairs)), list(reversed(names)))
            ax.tick_params(axis="y", length=0, pad=8)
            ax.set_ylim(-0.5, len(pairs) - 0.5)
            ax.grid(axis="x", color="#DDE2E6", linewidth=0.7, alpha=0.65)
            ax.set_axisbelow(True)
            ax.set_xlabel("Δ domain-macro MAE (years)")
            label = "Absolute age" if metric == "absolute" else "Within-domain centered age"
            title = f"{'AB'[column]}. {label}: discarded-score utility" if row_number == 0 else f"{'CD'[column]}. {label}: geometry at α = 1000"
            ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
    handles = []
    for atlas, population in series:
        common = population == "common_support"
        domains = len(panel(report, atlas, population, "centered")["included_domains"])
        population_label = "matched common support" if common else "all rows"
        handles.append(Line2D([], [], marker="D" if common else "o", linestyle="none",
                              markerfacecolor="white" if common else COLORS[atlas], markeredgecolor=COLORS[atlas],
                              markeredgewidth=1.4, markersize=6,
                              label=f"{atlas.capitalize()} · {population_label} ({domains} domains)"))
    fig.suptitle("What age-prediction utility lies in site-predictive directions?", fontsize=15, fontweight="bold", y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=9.5)
    footer = ("External development predictions only. Negative contrasts favor the first named model. Bars: descriptive 95% paired hierarchical bootstrap intervals.\n"
              "Filled circles: all rows. Hollow diamonds: identical eligible common-support domains for absolute and centered scoring.\n"
              "Centering uses evaluation ages; it is not a deployable correction. Atlases/views are not independent replications. No confirmatory or causal claim.")
    if missing:
        footer += "\nSome points or intervals are unavailable because of support; see the report for every domain."
    fig.text(0.025, 0.015, footer, fontsize=8.4, va="bottom", color="#40484F", linespacing=1.45)
    fig.subplots_adjust(left=0.23, right=0.985, top=0.84, bottom=0.16, hspace=0.48, wspace=0.95)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = FIGURES / f"discarded_direction_diagnostics.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Execute only in the coordinator's actual Slurm compute shell")
    run = args.run.resolve()
    if not run.is_relative_to(OUTPUTS / "runs/discarded_direction_diagnostics"):
        raise ValueError("Read only this episode's declared diagnostic run artifacts")
    report = read_json(run / "diagnostics_report.json")
    if report["analysis_id"] != "discarded_direction_diagnostics" or report["holdout_access"]:
        raise ValueError("Unexpected diagnostic evidence role")
    checked = replay_check(run)
    paths = make_figure(report, run)
    print(json.dumps({"complete": True, "figures": paths,
                      "replay_check": str(run / "baseline_replay_check.json"),
                      "maximum_prediction_difference_years": checked["maximum_compared_prediction_difference_years"],
                      "compared_cache_count": checked["compared_cache_count"],
                      "skipped_cache_count": checked["skipped_cache_count"]}), flush=True)


if __name__ == "__main__":
    main()
