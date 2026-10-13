#!/usr/bin/env python3
"""Plot the completed W panel and fixed-prediction contrasts; no source reads/fits."""
import csv
import json
from pathlib import Path
from statistics import mean

from primary_reference import ANIMALS, SESSIONS
from source_adapter import compute_only


ANIMAL_ORDER = ("Mihili", "Chewie")
METRICS = ("delta_R2", "identity_increment")
CANDIDATE = "lmp_repeatability_weight_ridge_v1"
CONTROL = "lmp_repeatability_weight_labels8_v1"
COLORS = {"Mihili": "#276f94", "Chewie": "#bb6632", "Overall": "#242424"}


def validate_rows(rows):
    import math

    if len(rows) != 16 or {r["session_id"] for r in rows} != set(SESSIONS):
        raise ValueError("Every original session must occur exactly once")
    if any(r["animal"] != ANIMALS[r["session_id"]]
           or not all(math.isfinite(r[k]) for k in METRICS) for r in rows):
        raise ValueError("Finite scores and original animal identities required")


def group_means(rows, metric):
    values = {animal: mean(r[metric] for r in rows if r["animal"] == animal)
              for animal in ANIMAL_ORDER}
    values["Overall"] = mean(values.values())
    return values


def description_rows(rows, condition):
    """Export session contrasts, equal-animal means and all fixed omissions."""
    result, summaries = [], {}
    for metric in METRICS:
        for row in rows:
            result.append({"condition": condition, "metric": metric, "level": "session",
                           "animal": row["animal"], "session": row["session_id"],
                           "excluded_session": "", "value": row[metric],
                           "omission_min": "", "omission_max": ""})
        centers = group_means(rows, metric)
        omitted = {row["session_id"]: group_means(
            [r for r in rows if r["session_id"] != row["session_id"]], metric) for row in rows}
        summaries[metric] = {}
        for animal, center in centers.items():
            lo = min(value[animal] for value in omitted.values())
            hi = max(value[animal] for value in omitted.values())
            summaries[metric][animal] = (center, lo, hi)
            result.append({"condition": condition, "metric": metric, "level": "mean",
                           "animal": animal, "session": "all", "excluded_session": "",
                           "value": center, "omission_min": lo, "omission_max": hi})
            for excluded, values in omitted.items():
                result.append({"condition": condition, "metric": metric,
                               "level": "fixed_prediction_omission", "animal": animal,
                               "session": "retained", "excluded_session": excluded,
                               "value": values[animal], "omission_min": "", "omission_max": ""})
    return result, summaries


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_repeatability_weight_summary_attempt001/primary_repeatability_weight.json"
    report = json.loads(source.read_text())
    if report.get("status") != "complete_repeatability_weight_control_panel":
        raise ValueError("Only the completed sixteen-session W panel may be plotted")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 2 or set(panels) != {CANDIDATE, CONTROL}:
        raise ValueError("Both completed W recipes required")
    for panel in panels.values():
        validate_rows(panel["sessions"])
    candidate = panels[CANDIDATE]
    groups = (report["anchor_contrasts"]["independently_selected"], report["weight_label_control"])
    names = ("W minus full-channel LMP", "W minus weight-label control (mean of 8)")
    for group in groups:
        validate_rows(group["sessions"])
    primary, identity = (group_means(candidate["sessions"], k)["Overall"] for k in METRICS)
    if abs(primary-candidate["primary_increment"]) > 1e-12 or abs(identity-candidate["identity_increment"]) > 1e-12:
        raise ValueError("Panel and recomputed equal-animal means differ")
    if primary >= .01 or identity >= .005:
        raise ValueError("This closed-result figure requires both original practical gates to fail")

    stem = root / "figures/primary_repeatability_weight_attempt001"
    if any(stem.with_suffix("."+ext).exists() for ext in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing result figures")
    ordered_ids = [s for animal in ANIMAL_ORDER for s in SESSIONS if ANIMALS[s] == animal]
    labels = [s.replace("Mihili_", "Mi ").replace("Chewie_", "Ch ") for s in ordered_ids]
    summary_y = {"Mihili": 17.5, "Chewie": 18.8, "Overall": 20.1}
    rows, summaries = [], []
    for group, name in zip(groups, names):
        exported, summary = description_rows(group["sessions"], name)
        rows.extend(exported)
        summaries.append(summary)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 12), sharey=True, sharex="row")
    for column, (group, name, summary) in enumerate(zip(groups, names, summaries)):
        by_id = {row["session_id"]: row for row in group["sessions"]}
        for row_index, metric in enumerate(METRICS):
            ax = axes[row_index, column]
            for position, session in enumerate(ordered_ids):
                value = by_id[session][metric]*1000
                ax.scatter(value, position, s=27, color=COLORS[ANIMALS[session]], zorder=3)
            for animal, position in summary_y.items():
                center, lo, hi = summary[metric][animal]
                ax.plot([lo*1000, hi*1000], [position, position],
                        color=COLORS[animal], lw=2.5, solid_capstyle="round", zorder=3)
                ax.scatter(center*1000, position, marker="D", s=44,
                           color=COLORS[animal], edgecolor="white", linewidth=.6, zorder=4)
            ax.axvline(0, color=".45", lw=.8)
            ax.axhline(5.5, color=".87", lw=.7)
            ax.axhline(16.3, color=".75", lw=.8)
            ax.grid(axis="x", color=".92", lw=.6)
            ax.set_axisbelow(True)
            ax.set_ylim(21.1, -.8)
            ax.set_yticks(list(range(16))+list(summary_y.values()),
                          labels+["Mihili mean", "Chewie mean", "Equal-animal mean"])
            ax.tick_params(axis="y", length=0, labelsize=8.8)
            ax.tick_params(axis="x", labelsize=9)
            ax.margins(x=.10)
            metric_label = "Primary increment" if metric == "delta_R2" else "Identity increment"
            ax.set_xlabel(f"Difference in {metric_label.lower()} (×10⁻³ R²)")
            ax.set_title(f"{'ABCD'[row_index*2+column]}  {metric_label}: {name}", loc="left", fontsize=10.5, pad=10)

    fig.suptitle("Repeatability weighting: full-cohort contrasts", x=.14, ha="left", y=.986, fontsize=17)
    fig.text(.14, .951,
             f"Both practical gates FAIL: primary {primary:.6f} < 0.010; identity {identity:.6f} < 0.005",
             fontsize=12, color="#9d2727", weight="bold")
    fig.legend(handles=[Line2D([], [], color=COLORS[a], marker="o", lw=0, label=a)
                        for a in ANIMAL_ORDER], loc="lower left", bbox_to_anchor=(.135, .083),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.14, .043,
             "Dots: all 16 sessions. Diamonds: means. Bars: ranges after each fixed-prediction session omission; not confidence intervals.\n"
             "Full-LMP anchor uses its own outer-selected λ; all 8 weight-label controls use W-selected λ and averaged SSE.\n"
             "Exposed development data · 2 animals · equal session weights within each animal · no promotion or new-animal test.",
             fontsize=9, color=".3")
    fig.tight_layout(rect=(.02, .125, .995, .925), h_pad=2.3, w_pad=2.0)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("."+extension), dpi=180)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(rows),
                      "source_summary": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
