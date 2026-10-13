#!/usr/bin/env python3
"""Plot the released COMPLETE H2W compact evidence; no source or partial runs."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

from plot_all_band import draw_groups
from plot_capacity_matched_surrogate import SPECS, describe
from plot_repeatability_weight import ANIMAL_ORDER, COLORS, METRICS, validate_rows
from primary_history2_repeatability_weight import RECIPES, REPLICATES
from source_adapter import compute_only


def checked_group(panel, condition, source):
    """Check complete session effects against the published group summaries."""
    validate_rows(panel["sessions"])
    if panel["lambda_counts"] != {"1.0": 16}:
        raise ValueError("This released comparison requires lambda=1 for every session")
    for row in panel["sessions"]:
        if row["lambda"] != 1.0:
            raise ValueError("Complete original-support lambda selections required")
    for _, key, animal_key in SPECS:
        if not all(math.isfinite(value) for value in (panel[key], *panel[animal_key].values())):
            raise ValueError("Finite published animal and equal-animal effects required")
    return describe(panel["sessions"], panel, condition, source)


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_history2_repeatability_summary_attempt001/compact_evidence.json"
    report = json.loads(source.read_text())
    account, replay, capacity = (report[key] for key in ("accounting", "exact_replay", "capacity"))
    if (report["status"] != "complete_history2_repeatability_panel" or report["recipes"] != list(RECIPES)
            or account["recipe_order"] != list(RECIPES)
            or account["scientific_trial_count_increment"] != 2
            or account["scientific_trial_count_by_recipe"] != dict.fromkeys(RECIPES, 1)
            or account["structural_duplicate_of"] != {recipe: [] for recipe in RECIPES}
            or any(account[key] != 1 for key in ("literal_null_control_count_increment", "alternative_explanation_count_increment"))
            or any(account[key] is not False for key in ("all_reference_margin_screens_pass", "original_HW_refitted",
                "original_W_refitted", "new_promotion_gate", "promotion_authorized"))
            or any(account[key] is not True for key in ("blend_reference_available", "eligible_for_comparable_primary_incumbent"))
            or any(replay[key] is not True for key in ("all_original_replays", "all_new_saved_scores_replayed",
                "original_selectors_independently_replayed"))
            or capacity["direction_folds"] != 128 or capacity["surrogate_direction_folds"] != 1024
            or capacity["shared_Gram_eigendecompositions"] != 128
            or capacity["surrogate_specific_Gram_eigendecompositions"] != 0):
        raise ValueError("Complete replayed H2W candidate/control and preserved original references required")

    candidate, null, direct = (report[key] for key in
                              ("candidate", "mean_surrogate", "candidate_minus_mean_surrogate"))
    groups = [checked_group(candidate, "H2W: current and one preceding bin", source),
              checked_group(null, "Mean of eight H2W training surrogates", source)]
    direct_group = checked_group(direct, "H2W minus mean training surrogate", source)
    candidate_by_id = {row["session_id"]: row for row in candidate["sessions"]}
    null_by_id = {row["session_id"]: row for row in null["sessions"]}
    for row in direct["sessions"]:
        session = row["session_id"]
        for metric in METRICS:
            if abs(candidate_by_id[session][metric]-null_by_id[session][metric]-row[metric]) > 1e-12:
                raise ValueError("Direct and separate candidate/surrogate effects must share the same denominator")
    if (candidate["primary_increment"] >= .01 or candidate["identity_increment"] >= .005
            or any(account["reference_margin_screens"][key] is not False for key in
                   ("primary_at_least_0p01", "identity_at_least_0p005"))
            or any(row[metric] <= 0 for row in direct["sessions"] for metric in METRICS)):
        raise ValueError("Framing requires failed practical margins and all-positive direct surrogate effects")

    references = report["reference_contrasts"]
    if set(references) != {"W_independently_selected", "W_H2W_lambda_matched", "HW_independently_selected",
                           "HW_H2W_lambda_matched", "blend_originally_selected"}:
        raise ValueError("Complete original W, HW and saved blend comparisons required")
    anchor_groups = []
    for name in ("W", "HW"):
        independent = references[name+"_independently_selected"]
        matched = references[name+"_H2W_lambda_matched"]
        group = checked_group(independent, "H2W minus "+name+" (original and H2W-matched lambda coincide)", source)
        checked_group(matched, "H2W minus "+name+" at H2W lambda", source)
        matched_by_id = {row["session_id"]: row for row in matched["sessions"]}
        restored_rows = []
        for row in independent["sessions"]:
            session = row["session_id"]
            for metric in METRICS:
                if row[metric] != matched_by_id[session][metric]:
                    raise ValueError("Collapse original/matched-lambda panels only when all session effects coincide exactly")
            restored_rows.append({"session_id": session, "animal": row["animal"],
                                  **{metric: candidate_by_id[session][metric]-row[metric] for metric in METRICS}})
        # Check the common-denominator identities against all preserved original
        # W/HW published animal means and omission ranges, without refitting.
        describe(restored_rows, report["original_components"][name], name+" recovered from direct contrast", source)
        anchor_groups.append(group)
    blend = references["blend_originally_selected"]
    blend_group = checked_group(blend, "H2W minus originally selected W/HW blend", source)
    anchor_groups.append(blend_group)
    if (any(blend_group[1]["delta_R2"][animal][2] >= 0 for animal in ANIMAL_ORDER)
            or blend["identity_increment"] <= 0
            or not (blend_group[1]["identity_increment"]["Mihili"][1] < 0
                    < blend_group[1]["identity_increment"]["Mihili"][2])):
        raise ValueError("Framing requires primary loss in both animals/all omissions and fragile Mihili identity gain")

    seeds = report["candidate_minus_each_surrogate"]
    if [seed["replicate"] for seed in seeds] != list(REPLICATES):
        raise ValueError("All eight fixed ordinary training-surrogate draws required")
    draw_values = {}
    for metric, key, animal_key in SPECS:
        draw_values[metric] = {animal: [seed[animal_key][animal] for seed in seeds] for animal in ANIMAL_ORDER}
        draw_values[metric]["Overall"] = [seed[key] for seed in seeds]
        for animal, values in draw_values[metric].items():
            if (not all(math.isfinite(value) and value > 0 for value in values)
                    or abs(mean(values)-direct_group[1][metric][animal][0]) > 1e-12):
                raise ValueError("Finite positive draw means must reproduce the mean-surrogate contrast")
        for seed in seeds:
            if abs(mean(seed[animal_key].values())-seed[key]) > 1e-12:
                raise ValueError("Every fixed draw must preserve equal-animal weighting")
            sign_key = "primary_positive_session_counts" if metric == "delta_R2" else "identity_positive_session_counts"
            for animal, count in (("Mihili", 6), ("Chewie", 10)):
                if (seed[sign_key][animal] != count
                        or seed["within_animal_omission_ranges"][animal][key]["minimum"] <= 0):
                    raise ValueError("Every fixed draw must preserve all session and animal-omission signs")

    stem = root / "figures/primary_history2_repeatability_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")
    exported = [row for group in (*groups, direct_group, *anchor_groups) for row in group[2]]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(14.6, 15.3), sharey="row")
    offsets = {"Mihili": -.18, "Chewie": .18, "Overall": 0.}
    for column, (metric, key, _) in enumerate(SPECS):
        ax = axes[0, column]
        draw_groups(ax, groups, metric, ("H2W\nCurrent + preceding bin", "Training surrogate\nMean SSE across 8 draws"))
        margin = .01 if column == 0 else .005
        ax.axhline(margin*1000, color="#9d2727", ls="--", lw=1)
        ax.text(.98, margin*1000, f" Practical margin {margin:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.8, color="#9d2727")
        ax.set_title(("A  Prediction gain over direction/time baseline",
                      "B  Correct versus wrong within-direction trial pairing")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Increment in joint native-XY R² (×10⁻³)")

        ax = axes[1, column]
        draw_groups(ax, [direct_group], metric, ["H2W − mean surrogate\n16 paired session effects"])
        for animal in (*ANIMAL_ORDER, "Overall"):
            values = draw_values[metric][animal]
            position = 1+offsets[animal]
            ax.plot([position, position], [min(values)*1000, max(values)*1000],
                    color=COLORS[animal], lw=1.4, ls=":", zorder=2)
            for replicate, value in zip(REPLICATES, values):
                ax.scatter(position+.11*(replicate/7-.5), value*1000, marker="o", s=34,
                           facecolors="none", edgecolors=COLORS[animal], linewidth=.9, zorder=4)
                exported.append({"condition": "H2W minus individual training surrogate", "metric": metric,
                    "level": "fixed_draw_mean", "animal": animal, "session": "all", "excluded_session": "",
                    "value": value, "omission_min": "", "omission_max": "", "replicate": replicate,
                    "derivation": "Published mean for one fixed ordinary draw", "source_evidence": str(source)})
        ax.set(xticks=(0, 1), xticklabels=("H2W − mean surrogate\n16 paired session effects",
                                         "H2W − each surrogate\n8 fixed draw means"), xlim=(-.48, 1.48))
        ax.set_title(("C  Prediction advantage over training surrogates",
                      "D  Trial-pairing advantage over training surrogates")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Difference in increment (×10⁻³ R²)")
        ax.margins(y=.12)

        ax = axes[2, column]
        draw_groups(ax, anchor_groups, metric,
                    ("H2W − W\nCurrent only", "H2W − HW\nFour lag blocks", "H2W − blend\nSaved W/HW mixture"))
        ax.set_title(("E  Direct prediction gain over saved references",
                      "F  Direct trial-pairing gain over saved references")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Difference in increment (×10⁻³ R²)")
        for row_index in (1, 2):
            axes[row_index, column].text(.98, .97, "Positive favors H2W", transform=axes[row_index, column].transAxes,
                                        ha="right", va="top", fontsize=9, color=".35")
        # Same full signed linear scale for the two metrics within each row;
        # effects versus original references use their own explicitly marked row.
        axes[0, column].tick_params(labelleft=True)
        axes[1, column].tick_params(labelleft=True)
        axes[2, column].tick_params(labelleft=True)

    lo, hi = capacity["shared_selected_effective_df_range"]
    fig.suptitle("Two-lag history beats training surrogates but trails the blend in prediction",
                 x=.075, ha="left", y=.982, fontsize=15.5)
    fig.text(.075, .954,
             f"H2W primary gain {candidate['primary_increment']:.6f} < 0.010; identity gain {candidate['identity_increment']:.6f} < 0.005.",
             fontsize=11.4, weight="bold", color="#9d2727")
    fig.text(.075, .932,
             f"Own training controls share the exact Gram and ridge df ({lo:.3f}–{hi:.3f}, 128 direction folds); all selected λ=1.",
             fontsize=10.3, color="#263e57")
    fig.text(.075, .187,
             f"H2W − blend: primary {blend['primary_increment']:+.6f} (both animals and every within-animal omission negative); "
             f"identity {blend['identity_increment']:+.6f}.", fontsize=10.1, weight="bold", color="#263e57")
    mi_low, mi_high = (blend_group[1]["identity_increment"]["Mihili"][i] for i in (1, 2))
    fig.text(.075, .168,
             f"The small identity gain is fragile in Mihili: mean {blend['identity_by_animal']['Mihili']:+.7f}, "
             f"fixed-omission range [{mi_low:+.7f}, {mi_high:+.7f}].", fontsize=9.8, color=".3")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0,
                               label=animal+" mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean"),
                  Line2D([], [], color=".3", marker="o", markerfacecolor="none", lw=0, label="Individual draw mean")],
               loc="lower left", bbox_to_anchor=(.069, .137), ncol=4, frameon=False, fontsize=9.5)
    fig.text(.075, .024,
             "Filled small dots: all 16 sessions. Solid bars: fixed-prediction session-omission ranges. Open dots/dotted spans: 8 draw means and their range.\n"
             "These ranges are not confidence intervals or p-values. Eight finite controls do not add biological replicates; there are two animals (6 and 10 sessions).\n"
             "Whole training-trial feature histories are permuted within direction; direction/time structure and correctly paired evaluation trials are retained.\n"
             "Surrogate correct/wrong-pair SSEs are averaged, not predictions. Own true/null designs share exact capacity; this is not a signal-free or raw-phase null.\n"
             "H2W uses lag 0 and −1 with the unchanged current-only channel prior. W, four-lag HW and the saved blend remain original, with no reference refit.\n"
             "Original- and H2W-selected λ coincide for W/HW, so those contrasts overlap exactly. Equal λ or normalized trace does not match capacity across families.\n"
             "All direct contrasts use complete original support and the same native-XY denominator. Row scales differ; both metrics share each row's full signed scale.\n"
             "Exposed development data · descriptive model comparisons, not causal lag effects · both practical margins still fail; the saved blend remains the incumbent.",
             fontsize=8.6, color=".3", linespacing=1.45)
    fig.tight_layout(rect=(.025, .222, .995, .909), h_pad=2.7, w_pad=2.5)
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
