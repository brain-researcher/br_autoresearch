#!/usr/bin/env python3
"""Plot complete reduced-versus-full HW contrasts on identical retained targets."""
import csv
import json
from pathlib import Path
from statistics import mean

from plot_all_band import draw_groups
from plot_repeatability_weight import (
    ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows,
)
from primary_matched_budget import RECIPES, VIEWS
from source_adapter import compute_only


REFERENCE_NAMES = ("independently_selected", "diagnostic_lambda_matched")
LABELS = ("C: matched electrodes\n86 electrodes\nOriginal trial counts",
          "T: matched trials\n12 trials / direction\nOriginal electrode counts",
          "CT: both budgets\n86 electrodes\n12 trials / direction")


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_matched_budget_summary_attempt001/primary_matched_budget.json"
    report = json.loads(source.read_text())
    if (report.get("status") != "complete_matched_budget_sensitivity_panel"
            or report.get("sensitivity_only") is not True
            or report.get("promotion_authorized") is not False
            or report.get("incumbent_reset") is not False
            or report.get("reference_refitted") is not False
            or report["recipe_order"] != list(RECIPES)
            or set(report["reference_contrasts"]) != set(RECIPES)):
        raise ValueError("Complete original C/T/CT sensitivity and preserved full-HW references required")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 3 or set(panels) != set(RECIPES):
        raise ValueError("All three completed resource diagnostics required")
    own_primary = []
    for recipe in RECIPES:
        panel = panels[recipe]
        validate_rows(panel["sessions"])
        for metric, key in zip(METRICS, ("primary_increment", "identity_increment")):
            if abs(group_means(panel["sessions"], metric)["Overall"] - panel[key]) > 1e-12:
                raise ValueError("Own-baseline diagnostic means must replay from all sessions")
        own_primary.append(panel["primary_increment"])
    if any(value >= .01 for value in own_primary):
        raise ValueError("This complete-result framing requires all three original primary margins to fail")

    groups, exported = {}, []
    for reference in REFERENCE_NAMES:
        groups[reference] = []
        for short, recipe in zip(("C", "T", "CT"), RECIPES):
            references = report["reference_contrasts"][recipe]
            if set(references) != set(REFERENCE_NAMES):
                raise ValueError("Both original and diagnostic-matched full-HW selections required")
            panel = references[reference]
            if panel["primary_formula"] != "(SSE_full_HW-SSE_reduced)/D_retained":
                raise ValueError("Direct common-support prediction contrast required")
            rows = panel["sessions"]
            validate_rows(rows)
            for row in rows:
                views = row["view_contrasts"]
                if [view["view"] for view in views] != list(VIEWS):
                    raise ValueError("Every session contrast requires all eight ordered support views")
                for metric in METRICS:
                    if abs(mean(view[metric] for view in views) - row[metric]) > 1e-12:
                        raise ValueError("Session contrast must average already normalized view differences")
            for metric, key in zip(METRICS, ("primary_increment", "identity_increment")):
                if abs(group_means(rows, metric)["Overall"] - panel[key]) > 1e-12:
                    raise ValueError("Direct contrast means must replay from all sessions")
            description, summary = description_rows(rows, short + " minus full HW: " + reference)
            if any(summary[metric][animal][2] >= 0
                   for metric in METRICS for animal in ANIMAL_ORDER):
                raise ValueError("Headline requires both-animal losses through every fixed session omission")
            groups[reference].append((rows, summary, description))
            exported.extend({**row, "recipe": recipe, "reference_selection": reference,
                             "source_summary": str(source)} for row in description)

    stem = root / "figures/primary_matched_budget_attempt001"
    if any(stem.with_suffix("." + extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.2), sharey="col")
    for row_index, reference in enumerate(REFERENCE_NAMES):
        reference_title = ("Full HW at its original selected λ" if row_index == 0
                           else "Full HW at the reduced condition's selected λ")
        for column, metric in enumerate(METRICS):
            ax = axes[row_index, column]
            draw_groups(ax, groups[reference], metric, LABELS)
            measure = "Prediction gain" if column == 0 else "Identity increment"
            ax.set_title(f"{'ABCD'[2*row_index+column]}  {measure}: {reference_title}",
                         loc="left", fontsize=10.4, pad=10)
            ax.set_ylabel("Reduced − full HW (×10⁻³ R²)")
            ax.text(.98, .97, "Positive favors reduced HW", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=".35")

    fig.suptitle("Reduced budgets lower prediction relative to full HW in both animals",
                 x=.075, ha="left", y=.982, fontsize=16.5)
    fig.text(.075, .946,
             "Own-baseline gains persist, but miss 0.010: "
             + "; ".join(f"{name} {value:.6f}" for name, value in zip(("C", "T", "CT"), own_primary)) + ".",
             fontsize=11.5, weight="bold", color="#9d2727")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0,
                               label=animal + " mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.069, .120), ncol=3, frameon=False, fontsize=10)
    fig.text(.075, .022,
             "Small dots: all 16 sessions. Bars: fixed-prediction session-omission ranges; not confidence intervals. Shared column scales retain every signed value.\n"
             "Prediction gain = (SSEfull − SSEreduced) / Dretained; identity contrasts use the same retained trial pairs. These are full-prediction comparisons.\n"
             "Each session averages 8 normalized subset-view contrasts before animal aggregation; overlapping views are not independent observations.\n"
             "Full HW keeps its original fitting data and saved predictions. Reduced models refit their behavioral baseline and neural state on their retained support.\n"
             "C reuses full HW in the two 86-electrode sessions. C selects λ=1; T/CT select λ=10. Equal counts or λ do not imply equal anatomical coverage or effective df.\n"
             "Exposed development data · 2 animals · nonpromotable resource sensitivity · original primary failure remains.",
             fontsize=8.8, color=".3")
    fig.tight_layout(rect=(.025, .205, .995, .916), h_pad=2.6, w_pad=2.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(exported),
                      "source_summary": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
