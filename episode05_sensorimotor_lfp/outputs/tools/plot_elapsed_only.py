#!/usr/bin/env python3
"""Plot complete elapsed-only sensitivity, preserving the behavioral decomposition."""
import csv
import json
from pathlib import Path

from extract_primary_elapsed_only import CONTRAST_NAMES, PANEL_NAMES
from plot_all_band import draw_groups
from plot_repeatability_weight import (
    ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows,
)
from primary_elapsed_only import RECIPE
from source_adapter import compute_only


def describe(panel, name, source):
    rows = panel["sessions"]
    validate_rows(rows)
    for metric, key in zip(METRICS, ("primary_increment", "identity_increment")):
        if abs(group_means(rows, metric)["Overall"] - panel[key]) > 1e-12:
            raise ValueError("Complete-session and reported equal-animal means must agree")
    exported, summary = description_rows(rows, name)
    return rows, summary, [{**row, "source_summary": str(source)} for row in exported]


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_elapsed_only_summary_attempt001/primary_elapsed_only.json"
    report = json.loads(source.read_text())
    if (report.get("status") != "complete_elapsed_only_HW_panel" or report["recipe"] != RECIPE
            or report["sensitivity_only"] is not True or report["promotion_authorized"] is not False
            or report["updates_primary_incumbent"] is not False or report["original_HW_refitted"] is not False
            or report["all_original_HW_neural_state_replay"] is not True
            or report["all_original_HW_saved_arrays_and_scoring_replay"] is not True
            or report["original_HW_anchor"]["refitted"] is not False
            or set(report["full_model_anchor_contrasts"]) != set(CONTRAST_NAMES)):
        raise ValueError("Complete nonpromotable sensitivity and original-HW replays required")
    panels = {name: describe(report[name], name, source) for name in PANEL_NAMES}
    contrasts = {name: describe(report["full_model_anchor_contrasts"][name], name, source)
                 for name in CONTRAST_NAMES}
    original_name = "original_HW_versus_original_B_replayed"
    elapsed_name = "ET_HW_versus_original_B_descriptive"
    by_session = {name: {row["session_id"]: row for row in group[0]} for name, group in panels.items()}
    for session in by_session[original_name]:
        hw_b = by_session[original_name][session]["delta_R2"]
        b_t = by_session["original_B_versus_elapsed_baseline"][session]["delta_R2"]
        hw_t = by_session["original_HW_versus_elapsed_baseline"][session]["delta_R2"]
        et_b = by_session[elapsed_name][session]["delta_R2"]
        et_t = by_session["elapsed_sensitivity"][session]["delta_R2"]
        if abs(hw_b + b_t - hw_t) > 1e-12 or abs(et_b + b_t - et_t) > 1e-12:
            raise ValueError("Original and elapsed-only common-denominator baseline decompositions must agree")
    hw_b = panels[original_name][1]["delta_R2"]["Overall"][0]
    b_t = panels["original_B_versus_elapsed_baseline"][1]["delta_R2"]["Overall"][0]
    hw_t = panels["original_HW_versus_elapsed_baseline"][1]["delta_R2"]["Overall"][0]
    et_b = panels[elapsed_name][1]["delta_R2"]["Overall"][0]
    et_t = panels["elapsed_sensitivity"][1]["delta_R2"]["Overall"][0]
    if hw_b >= .01 or any(row["delta_R2"] >= 0 for row in panels[elapsed_name][0]):
        raise ValueError("Headline requires the original primary failure and all sixteen ET-HW losses to B")
    stem = root / "figures/primary_elapsed_only_attempt001"
    if any(stem.with_suffix("." + extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.8))
    model_groups = [panels[original_name], panels[elapsed_name]]
    contrast_groups = [contrasts[name] for name in CONTRAST_NAMES]
    model_labels = ("Original HW\nDirection/time residual fit", "ET-HW\nElapsed-only residual fit")
    contrast_labels = ("ET-HW − HW\nIndependently selected penalties",
                       "ET-HW − HW\nBoth at ET-HW-selected λ")
    titles = (("A  Full prediction relative to original direction/time B",
               "B  Correct versus wrong within-direction trial pairing"),
              ("C  Direct full-prediction contrast: ET-HW minus HW",
               "D  Difference in trial-pairing increment: ET-HW minus HW"))
    for column, metric in enumerate(METRICS):
        for row_index, (groups, labels) in enumerate(((model_groups, model_labels),
                                                     (contrast_groups, contrast_labels))):
            ax = axes[row_index, column]
            draw_groups(ax, groups, metric, labels)
            # Shared glyph helper scales by 1000 internally; display original
            # R² units so the decomposition and all four axes use explicit units.
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value/1000:g}"))
            ax.set_title(titles[row_index][column], loc="left", fontsize=10.3, pad=10)
            ax.set_ylabel("Increment in joint native-XY R²" if row_index == 0
                          else "Difference in increment (R²)")
            if row_index == 0:
                margin = .01 if column == 0 else .005
                ax.axhline(margin*1000, color="#9d2727", ls="--", lw=1)
                ax.text(.98, margin*1000, f" Original HW margin {margin:g}",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=8.6, color="#9d2727")
            else:
                ax.text(.98, .97, "Positive favors ET-HW", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9, color=".35")

    fig.suptitle("Elapsed-only fitting loses accuracy despite a larger pairing increment",
                 x=.075, ha="left", y=.982, fontsize=16.5)
    fig.text(.075, .942,
             f"HW − T = (B − T) + (HW − B) = {b_t:.6f} + {hw_b:.6f} = {hw_t:.6f} R²",
             fontsize=12, weight="bold", color="#263e57")
    fig.text(.075, .911,
             f"The large HW-over-T gain includes the behavioral baseline. ET-HW gains {et_t:.6f} over T, yet loses {abs(et_b):.6f} to B (all 16 sessions).",
             fontsize=10.8, color="#9d2727")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0,
                               label=animal + " mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.069, .113), ncol=3, frameon=False, fontsize=10)
    fig.text(.075, .025,
             "Small dots: all 16 sessions. Bars: fixed-prediction session-omission ranges; not confidence intervals. Each panel has its own full signed scale.\n"
             "B = original direction/time template + common residual intercept; T = equal-direction elapsed-only template + the same type of intercept.\n"
             "ET-HW refits Y−T and adds T back; the HW feature design and weights stay fixed. Original HW predictions are preserved, with no refit.\n"
             "Direct contrasts use full-prediction SSE and the same original-session denominator. Original HW selects λ=1; ET-HW selects λ=0.01 in every session.\n"
             "The matched-λ comparison preserves the ridge operator, while the supplied baseline and fitting response differ. Stronger pairing does not rescue accuracy.\n"
             "Exposed development data · 2 animals · ET-HW is nonpromotable · original HW primary margin still fails.",
             fontsize=8.8, color=".3")
    fig.tight_layout(rect=(.025, .195, .995, .876), h_pad=2.6, w_pad=2.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    exported = [row for group in (*panels.values(), *contrasts.values()) for row in group[2]]
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(exported),
                      "source_summary": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
