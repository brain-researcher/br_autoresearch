#!/usr/bin/env python3
"""Plot only the released COMPLETE HW capacity-surrogate compact evidence."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

from plot_all_band import draw_groups
from plot_repeatability_weight import (
    ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows,
)
from primary_capacity_matched_surrogate import RECIPE, REPLICATES
from source_adapter import compute_only


SPECS = (("delta_R2", "primary_increment", "primary_by_animal"),
         ("identity_increment", "identity_increment", "identity_by_animal"))


def describe(rows, published, condition, source, derivation="Published session effects"):
    validate_rows(rows)
    exported, summary = description_rows(rows, condition)
    for metric, key, animal_key in SPECS:
        centers = group_means(rows, metric)
        if abs(centers["Overall"]-published[key]) > 1e-12:
            raise ValueError("Published equal-animal and complete-session means must agree")
        for animal in ANIMAL_ORDER:
            expected = published["within_animal_omission_ranges"][animal][key]
            center, low, high = summary[metric][animal]
            if (abs(center-published[animal_key][animal]) > 1e-12
                    or abs(low-expected["minimum"]) > 1e-12
                    or abs(high-expected["maximum"]) > 1e-12):
                raise ValueError("Published animal means and fixed-omission ranges must agree")
    return rows, summary, [{**row, "replicate": "", "derivation": derivation,
                            "source_evidence": str(source)} for row in exported]


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_capacity_matched_surrogate_summary_attempt001/compact_evidence.json"
    report = json.loads(source.read_text())
    account, replay, capacity = (report[key] for key in ("accounting", "exact_replay", "capacity"))
    if (report["status"] != "complete_capacity_matched_surrogate_panel" or report["recipe"] != RECIPE
            or any(account[key] != 1 for key in ("scientific_trial_count_increment",
                "literal_null_control_count_increment", "alternative_explanation_count_increment"))
            or any(account[key] is not False for key in ("structural_original_HW_duplicate",
                "updates_primary_incumbent", "promotion_authorized", "new_promotion_gate"))
            or replay["original_HW_refitted"] is not False
            or any(replay[key] is not True for key in ("all_original_HW_neural_state_replay",
                "all_original_HW_saved_arrays_and_scoring_replay", "all_original_HW_weighted_covariance_replay"))
            or capacity["direction_folds"] != 128 or capacity["surrogate_direction_folds"] != 1024
            or capacity["shared_Gram_eigendecompositions"] != 128
            or capacity["surrogate_specific_Gram_eigendecompositions"] != 0):
        raise ValueError("Complete original-support capacity control and exact original replays required")

    null = report["mean_surrogate"]
    contrast = report["HW_minus_mean_surrogate"]
    validate_rows(null["sessions"])
    validate_rows(contrast["sessions"])
    by_id = {row["session_id"]: row for row in contrast["sessions"]}
    # The compact extractor keeps original HW aggregates, mean-null sessions,
    # and direct HW-minus-null sessions. Recover HW only by this exact linear
    # common-denominator identity, then check all published HW aggregates.
    hw_rows = []
    for row in null["sessions"]:
        difference = by_id[row["session_id"]]
        if row["lambda"] != 1.0 or difference["lambda"] != 1.0:
            raise ValueError("All surrogates must retain original HW-selected lambda=1")
        hw_rows.append({"session_id": row["session_id"], "animal": row["animal"],
                        **{metric: row[metric]+difference[metric] for metric in METRICS}})
    groups = [describe(hw_rows, report["original_HW"], "Original HW", source,
                       "Mean-surrogate session effect plus published direct HW-minus-mean effect"),
              describe(null["sessions"], null, "Mean of eight training surrogates", source)]
    direct = describe(contrast["sessions"], contrast, "HW minus mean training surrogate", source)
    if (report["original_HW"]["primary_increment"] >= .01
            or any(row[metric] <= 0 for row in direct[0] for metric in METRICS)
            or any(direct[1][metric][animal][1] <= 0 for metric in METRICS for animal in ANIMAL_ORDER)):
        raise ValueError("Framing requires original primary failure and positive complete paired effects")

    seeds = report["HW_minus_each_seed"]
    if [seed["replicate"] for seed in seeds] != list(REPLICATES):
        raise ValueError("Every one of the eight fixed ordinary draws is required")
    draw_values = {}
    for metric, key, animal_key in SPECS:
        draw_values[metric] = {animal: [seed[animal_key][animal] for seed in seeds]
                               for animal in ANIMAL_ORDER}
        draw_values[metric]["Overall"] = [seed[key] for seed in seeds]
        for animal, values in draw_values[metric].items():
            if (not all(math.isfinite(value) and value > 0 for value in values)
                    or abs(mean(values)-direct[1][metric][animal][0]) > 1e-12):
                raise ValueError("Fixed-draw means must reproduce the direct mean-surrogate contrast")
        for seed in seeds:
            if abs(mean(seed[animal_key].values())-seed[key]) > 1e-12:
                raise ValueError("Draw summaries must preserve equal-animal weighting")
            for animal, count in (("Mihili", 6), ("Chewie", 10)):
                sign_key = "primary_positive_session_counts" if metric == "delta_R2" else "identity_positive_session_counts"
                if (seed[sign_key][animal] != count
                        or seed["within_animal_omission_ranges"][animal][key]["minimum"] <= 0):
                    raise ValueError("Every draw must preserve all session and animal-omission signs")

    stem = root / "figures/primary_capacity_matched_surrogate_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")
    exported = [row for group in (*groups, direct) for row in group[2]]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 11.4))
    offsets = {"Mihili": -.18, "Chewie": .18, "Overall": 0.}
    for column, (metric, key, _) in enumerate(SPECS):
        ax = axes[0, column]
        draw_groups(ax, groups, metric, ("HW\nOriginal trial alignment", "Training surrogate\nMean SSE across 8 draws"))
        margin = .01 if column == 0 else .005
        ax.axhline(margin*1000, color="#9d2727", ls="--", lw=1)
        ax.text(.98, margin*1000, f" Original HW margin {margin:g}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.8, color="#9d2727")
        ax.set_title(("A  Prediction gain over direction/time baseline",
                      "B  Correct versus wrong within-direction trial pairing")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Increment in joint native-XY R² (×10⁻³)")
        ax = axes[1, column]
        draw_groups(ax, [direct], metric, ["HW − mean surrogate\n16 paired session effects"])
        for animal in (*ANIMAL_ORDER, "Overall"):
            values = draw_values[metric][animal]
            position = 1+offsets[animal]
            ax.plot([position, position], [min(values)*1000, max(values)*1000],
                    color=COLORS[animal], lw=1.4, ls=":", zorder=2)
            for replicate, value in enumerate(values):
                ax.scatter(position+.11*(replicate/7-.5), value*1000, marker="o", s=34,
                           facecolors="none", edgecolors=COLORS[animal], linewidth=.9, zorder=4)
                exported.append({"condition": "HW minus individual training surrogate", "metric": metric,
                    "level": "fixed_draw_mean", "animal": animal, "session": "all", "excluded_session": "",
                    "value": value, "omission_min": "", "omission_max": "", "replicate": replicate,
                    "derivation": "Published mean for one fixed ordinary draw", "source_evidence": str(source)})
        ax.set(xticks=(0, 1), xticklabels=("HW − mean surrogate\n16 paired session effects",
                                         "HW − each surrogate\n8 fixed draw means"), xlim=(-.48, 1.48))
        ax.set_title(("C  Direct prediction advantage: HW minus surrogate",
                      "D  Trial-pairing advantage: HW minus surrogate")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Difference in increment (×10⁻³ R²)")
        ax.text(.98, .97, "Positive favors HW", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=".35")
        ax.margins(y=.12)

    lo, hi = capacity["shared_selected_effective_df_range"]
    fig.suptitle("HW retains an advantage over capacity-matched training surrogates",
                 x=.075, ha="left", y=.982, fontsize=16)
    fig.text(.075, .944,
             f"All 16 paired effects are positive; HW primary gain {report['original_HW']['primary_increment']:.6f} still falls below 0.010.",
             fontsize=11.4, weight="bold", color="#9d2727")
    fig.text(.075, .916,
             f"Exact shared Gram and ridge df in every fold · selected λ=1 · df {lo:.3f}–{hi:.3f} across 128 direction folds",
             fontsize=10.5, color="#263e57")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0,
                               label=animal+" mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")
                  , Line2D([], [], color=".3", marker="o", markerfacecolor="none", lw=0,
                           label="Individual draw mean")],
               loc="lower left", bbox_to_anchor=(.069, .123), ncol=4, frameon=False, fontsize=9.5)
    fig.text(.075, .025,
             "Filled small dots: all 16 sessions. Solid bars: fixed-prediction session-omission ranges. Open dots/dotted spans: all 8 draw means and their range.\n"
             "These descriptive ranges are not confidence intervals or p-values; the 8 draws are finite controls, not independent biological replicates.\n"
             "Whole training-trial feature histories are permuted within direction. Time, direction structure, preprocessing, B, targets and selected λ remain fixed.\n"
             "Original HW is never refit. Evaluation features stay paired with their original trials; identity scores separately wrong-pair held-out trajectories.\n"
             "Surrogate correct/wrong-pair SSEs are averaged, not predictions. Direct contrasts share the original native-XY denominator. Every panel retains all signed values.\n"
             "This is a conditional capacity control, not a signal-free or raw-phase null. Exposed development data · 2 animals · original primary failure remains.",
             fontsize=8.7, color=".3")
    fig.tight_layout(rect=(.025, .205, .995, .887), h_pad=2.5, w_pad=2.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("."+extension), dpi=180)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(exported),
                      "source_evidence": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
