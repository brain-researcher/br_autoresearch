#!/usr/bin/env python3
"""Plot complete residual/direct/control evidence on full unbroken signed scales."""
import csv
import json
from pathlib import Path

from plot_repeatability_weight import ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows
from primary_full_target_sensitivity import ANCHOR, RECIPES
from source_adapter import compute_only


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root/"runs/adaptive_search/primary_full_target_sensitivity_summary_attempt001/primary_full_target_sensitivity.json"
    report = json.loads(source.read_text())
    if (report.get("status") != "complete_full_target_sensitivity_control_panel"
            or report["sensitivity_only"] is not True or report["promotion_authorized"] is not False
            or report["reference_anchor"]["refitted"] is not False):
        raise ValueError("Complete nonpromotable sensitivity and preserved residual anchor required")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    anchor = report["reference_anchor"]["panel"]
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES) or anchor["recipe"] != ANCHOR:
        raise ValueError("Original residual anchor and both sensitivity recipes required")
    selected = (anchor, panels[RECIPES[0]], panels[RECIPES[1]])
    names = ("R: residual LMP", "F: direct full XY", "C: training-label control")
    labels = ("R\nResidual LMP", "F\nDirect full XY", "C\nMean of 8 target-label fits")
    groups, summaries, exported = [], [], []
    for panel, name in zip(selected, names):
        validate_rows(panel["sessions"])
        for metric, key in zip(METRICS, ("primary_increment", "identity_increment")):
            if abs(group_means(panel["sessions"], metric)["Overall"]-panel[key]) > 1e-12:
                raise ValueError("Reported and recomputed equal-animal means differ")
        rows, summary = description_rows(panel["sessions"], name)
        exported.extend({**row, "recipe": panel["recipe"], "source_summary": str(source)} for row in rows)
        groups.append(panel["sessions"])
        summaries.append(summary)
    stem = root/"figures/primary_full_target_sensitivity_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing figures")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    offsets = {"Mihili": -.17, "Chewie": .17, "Overall": 0.}
    for ax, metric, margin, title in zip(axes, METRICS, (.01, .005),
            ("A  Signed primary gain over direction/time baseline", "B  Correct versus wrong held-out trial pairing")):
        for position, (rows, summary) in enumerate(zip(groups, summaries)):
            for animal in ANIMAL_ORDER:
                values = [row for row in rows if row["animal"] == animal]
                for index, row in enumerate(values):
                    spread = .12*(index/(len(values)-1)-.5)
                    ax.scatter(position+offsets[animal]+spread, row[metric], color=COLORS[animal],
                               s=27, alpha=.48, linewidth=.4, edgecolor="white", zorder=3)
            for animal in (*ANIMAL_ORDER, "Overall"):
                center, low, high = summary[metric][animal]
                location = position+offsets[animal]
                ax.plot([location, location], [low, high], color=COLORS[animal], lw=2, zorder=4)
                for endpoint in (low, high):
                    ax.plot([location-.025, location+.025], [endpoint, endpoint], color=COLORS[animal], lw=1.1, zorder=4)
                ax.scatter(location, center, marker="s" if animal == "Overall" else "D",
                           color=COLORS[animal], s=49, edgecolor="white", linewidth=.6, zorder=5)
        ax.axhline(0, color=".45", lw=.8)
        ax.axhline(margin, color="#9d2727", ls="--", lw=1)
        ax.text(.02, margin, f" Equal-animal margin {margin:g}", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", color="#9d2727", fontsize=9)
        ax.set(title=title, ylabel="Increment in joint native-XY R²", xticks=range(3),
               xticklabels=labels, xlim=(-.48, 2.48))
        ax.title.set_fontsize(10.4)
        ax.grid(axis="y", color=".93", lw=.6)
        ax.set_axisbelow(True)
        # Keep all signed session scores, omission endpoints, zero and margin
        # in the full autoscaled linear axis; no truncation, inset or break.
        ax.margins(y=.12)
    direct = summaries[1]
    primary, identity = direct["delta_R2"]["Overall"][0], direct["identity_increment"]["Overall"][0]
    primary_status = "PASS" if primary >= .01 else "FAIL"
    identity_status = "PASS" if identity >= .005 else "FAIL"
    fig.suptitle("Direct full-target prediction loses to the movement baseline", x=.075, ha="left", y=.98, fontsize=16)
    fig.text(.075, .928,
             f"F: primary {primary:.5f} — {primary_status} at 0.010; identity {identity:.5f} — {identity_status} at 0.005. Sensitivity only.",
             fontsize=11.5, weight="bold", color="#9d2727")
    fig.legend(handles=[Line2D([], [], color=COLORS[a], marker="D", lw=0, label=a+" mean") for a in ANIMAL_ORDER]
               +[Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.067, .105), ncol=3, frameon=False, fontsize=10)
    fig.text(.075, .033,
             "Small dots: all 16 sessions per recipe. Bars: fixed-prediction session-omission ranges; not confidence intervals.\n"
             "R uses its preserved outer-selected λ; F and all 8 controls share F-selected λ. Complete signed scales retain the primary losses.\n"
             "Exposed development data · 2 animals · native unscaled XY · no primary promotion or future-session confirmation.",
             fontsize=9, color=".3")
    fig.tight_layout(rect=(.025, .19, .995, .89), w_pad=2.4)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("."+extension), dpi=180)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
        "source_data": str(stem.with_suffix(".csv")), "rows": len(exported), "source_summary": str(source),
        "preserved_anchor_summary": report["reference_anchor"]["path"]}, allow_nan=False))


if __name__ == "__main__":
    main()
