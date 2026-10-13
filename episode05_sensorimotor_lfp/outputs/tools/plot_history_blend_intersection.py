#!/usr/bin/env python3
"""Outcome-blind plotting specification; read only the COMPLETE I/R-blend compact."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

from extract_primary_history_blend_intersection import COMPLETE_REPLAY_FLAGS, REFERENCE_NAMES
from plot_all_band import draw_groups
from plot_capacity_matched_surrogate import SPECS, describe
from plot_repeatability_weight import ANIMAL_ORDER, COLORS, METRICS, validate_rows
from primary_history_blend import ALPHAS
from primary_history_blend_intersection import RECIPES, REPLICATES
from primary_reference import LAMBDAS, SESSIONS
from run_primary_history_blend_intersection import PANEL_STATUS
from source_adapter import compute_only


CONTRAST_KEYS = ("I_minus_R", "I_minus_full_blend", "R_minus_full_blend", "I_minus_HW_blend_lambda_matched",
                 "I_minus_original_I_HW", "R_minus_HW_blend_lambda_matched", "R_minus_original_R_HW")
LABELS = ("I − mean R", "I − full\nblend", "R − full\nblend", "I − own HW\nNew λ",
          "I − own HW\nOriginal λ", "R − own HW\nNew λ", "R − own HW\nOriginal λ")
CONDITIONS = ("I blend minus mean R blend", "I blend minus original full blend", "R blend minus original full blend",
              "I blend minus own HW at new lambda", "I blend minus originally selected I HW",
              "R blend minus own HW at new lambda", "R blend minus originally selected R HW")


def exported_rows(group, selected, *, replicate="", references=None):
    result = []
    for row in group[2]:
        choice = selected.get(row["session"], {})
        reference = (references or {}).get(row["session"], {})
        result.append({**row, "replicate": replicate, "selected_alpha": choice.get("alpha", ""),
                       "selected_lambda": choice.get("lambda", ""), "reference_lambda": reference.get("reference_lambda", "")})
    return result


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_history_blend_intersection_summary_attempt001/compact_evidence.json"
    report = json.loads(source.read_text())
    account = report["accounting"]
    if (report["status"] != PANEL_STATUS or report["recipes"] != list(RECIPES)
            or account["recipe_order"] != list(RECIPES) or account["sensitivity_only"] is not True
            or any(account[key] is not False for key in ("incumbent_reset", "eligible_for_comparable_primary_incumbent",
                "original_HW_refitted", "full_W_refitted", "promotion_authorized", "provider_source_values_read"))
            or account["literal_null_control_count_increment"] != 0
            or any(report["exact_replay"][key] is not True for key in COMPLETE_REPLAY_FLAGS)
            or set(report["selected_prediction_paths"]) != set(SESSIONS)):
        raise ValueError("Only complete replayed nonpromotable I/R-blend evidence is permitted")
    charges, duplicates = account["scientific_trial_count_by_recipe"], account["structural_duplicate_of"]
    if (set(charges) != set(RECIPES) or set(duplicates) != set(RECIPES)
            or any(type(charges[recipe]) is not int or charges[recipe] != int(not duplicates[recipe]) for recipe in RECIPES)
            or any(account[key] != sum(charges.values()) for key in ("scientific_trial_count_increment",
                "stress_diagnostic_count_increment", "alternative_explanation_count_increment"))):
        raise ValueError("Independent exact duplicate and support-stress accounting required")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES):
        raise ValueError("Both complete original-session I/R panels required")
    intersection, random = (panels[recipe] for recipe in RECIPES)
    own_groups = [describe(intersection["sessions"], intersection, "I blend: own baseline gain", source),
                  describe(random["sessions"], random, "R blend: mean own baseline gain", source)]
    selected = {row["session_id"]: row for row in intersection["sessions"]}
    random_by_id = {row["session_id"]: row for row in random["sessions"]}
    for session, row in selected.items():
        if (row["lambda"] not in LAMBDAS or row["alpha"] not in ALPHAS
                or any(random_by_id[session][key] != row[key] for key in ("lambda", "alpha"))):
            raise ValueError("All eight R views must use the I-selected original-grid lambda/alpha")
    contrasts = {"I_minus_R": report["intersection_minus_random_support"], **report["reference_contrasts"]}
    if set(report["reference_contrasts"]) != set(REFERENCE_NAMES) or set(contrasts) != set(CONTRAST_KEYS):
        raise ValueError("Full-blend, new-lambda HW and originally selected HW comparisons must remain distinct")
    groups = [describe(contrasts[key]["sessions"], contrasts[key], label, source)
              for key, label in zip(CONTRAST_KEYS, CONDITIONS)]
    for row in contrasts["I_minus_R"]["sessions"]:
        session = row["session_id"]
        for metric in METRICS:
            if abs(selected[session][metric]-random_by_id[session][metric]-row[metric]) > 1e-12:
                raise ValueError("I-minus-R effects must share the original native-XY denominator")
    for key in CONTRAST_KEYS[1:]:
        for row in contrasts[key]["sessions"]:
            if row["lambda"] != selected[row["session_id"]]["lambda"] or row["alpha"] != selected[row["session_id"]]["alpha"]:
                raise ValueError("Direct contrast mixture parameters changed")
            if "lambda_matched" in key and row["reference_lambda"] != row["lambda"]:
                raise ValueError("New-lambda own-HW reference must use I's selected penalty")

    seeds = report["intersection_minus_each_random_support"]
    if [seed["replicate"] for seed in seeds] != list(REPLICATES):
        raise ValueError("Every one of the eight fixed random-support contrasts required")
    seed_groups = [describe(seed["sessions"], seed, "I blend minus individual R blend", source) for seed in seeds]
    seed_maps = [{row["session_id"]: row for row in group[0]} for group in seed_groups]
    for row in groups[0][0]:
        for metric in METRICS:
            if abs(mean(seed[row["session_id"]][metric] for seed in seed_maps)-row[metric]) > 1e-12:
                raise ValueError("All eight paired draw effects must reproduce the mean-R effect")
    draw_values = {metric: {animal: [group[1][metric][animal][0] for group in seed_groups]
                           for animal in (*ANIMAL_ORDER, "Overall")} for metric in METRICS}
    exceptions = {metric: {animal: {
        "nonpositive_draw_means": [r for r, group in zip(REPLICATES, seed_groups) if group[1][metric][animal][0] <= 0],
        "draws_with_nonpositive_omissions": [r for r, group in zip(REPLICATES, seed_groups) if group[1][metric][animal][1] <= 0],
    } for animal in ANIMAL_ORDER} for metric in METRICS}
    diagnostics = report["selected_support_diagnostics"]
    if len(diagnostics) != 16 or {row["session_id"] for row in diagnostics} != set(SESSIONS):
        raise ValueError("Complete physical-support and component-diagnostic records required")
    counts = []
    for row in diagnostics:
        choice = selected[row["session_id"]]
        views = [row["intersection"], *row["random_supports"]]
        count = row["intersection"]["electrode_count"]
        if (len(views) != 9 or [view["replicate"] for view in views[1:]] != list(REPLICATES)
                or any(view["electrode_count"] != count or view["lambda"] != choice["lambda"]
                       or view["alpha"] != choice["alpha"] for view in views)
                or any(not math.isfinite(view["error_cross_product_over_D"]) for view in views)):
            raise ValueError("All nine count-matched support mixtures and descriptive component errors required")
        counts.append(count)

    stem = root / "figures/primary_history_blend_intersection_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")
    exported = [row for group in own_groups for row in exported_rows(group, selected)]
    for key, group in zip(CONTRAST_KEYS, groups):
        exported.extend(exported_rows(group, selected, references={row["session_id"]: row for row in contrasts[key]["sessions"]}))
    for replicate, group in zip(REPLICATES, seed_groups):
        exported.extend(exported_rows(group, selected, replicate=replicate))
    for row in intersection["sessions"]:
        for key in ("alpha", "lambda"):
            exported.append({"condition": "Shared I/R selected mixture parameter", "metric": key, "level": "selected_parameter",
                "animal": row["animal"], "session": row["session_id"], "excluded_session": "", "value": row[key],
                "omission_min": "", "omission_max": "", "replicate": "", "derivation": "Published original-grid I selector",
                "source_evidence": str(source), "selected_alpha": row["alpha"], "selected_lambda": row["lambda"], "reference_lambda": ""})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(17.8, 14.7))
    axes[0, 1].sharey(axes[0, 0])
    for column, metric in enumerate(METRICS):
        ax = axes[0, column]
        draw_groups(ax, groups, metric, LABELS)
        ax.set_title(("A  Direct primary prediction differences", "B  Direct trial-pairing differences")[column],
                     loc="left", fontsize=11, pad=10)
        ax.set_ylabel("Difference in native-XY increment (×10⁻³ R²)")
        ax.text(.98, .97, "Positive favors the I/R blend named first", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=".35")
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(labelleft=True)
        for boundary in (.5, 2.5, 4.5):
            ax.axvline(boundary, color=".93", lw=.7, zorder=0)

    ax = axes[1, 0]
    ordered = [row for animal in ANIMAL_ORDER for row in intersection["sessions"] if row["animal"] == animal]
    for index, row in enumerate(ordered):
        ax.scatter(row["alpha"], index, s=36, color=COLORS[row["animal"]], zorder=3)
    ax.axhline(5.5, color=".8", lw=.8)
    ax.set(yticks=range(16), yticklabels=[row["session_id"].replace("Mihili_", "Mi ").replace("Chewie_", "Ch ")
                                       + f"  λ={row['lambda']:g}" for row in ordered],
           xticks=ALPHAS, xlim=(-.055, 1.055), ylim=(15.7, -.7),
           xlabel="Selected HW weight α; W weight = 1 − α (shared by all eight R draws)")
    ax.tick_params(axis="y", labelsize=8.6)
    ax.grid(axis="x", color=".92", lw=.6)
    ax.set_title("C  Original-grid mixture selected using the other 15 sessions", loc="left", fontsize=11, pad=10)

    ax = axes[1, 1]
    offsets = {"Mihili": -.20, "Chewie": .20, "Overall": 0.}
    for position, metric in enumerate(METRICS):
        for animal in (*ANIMAL_ORDER, "Overall"):
            values = draw_values[metric][animal]
            center = groups[0][1][metric][animal][0]
            if abs(mean(values)-center) > 1e-12:
                raise ValueError("Draw means and aggregate paired effects must agree")
            location = position+offsets[animal]
            ax.plot([location, location], [min(values)*1000, max(values)*1000], color=COLORS[animal], ls=":", lw=1.5, zorder=2)
            for replicate, value in zip(REPLICATES, values):
                ax.scatter(location+.13*(replicate/7-.5), value*1000, s=36, facecolors="none", edgecolors=COLORS[animal], linewidth=1, zorder=4)
            ax.scatter(location, center*1000, marker="s" if animal == "Overall" else "D", s=44,
                       color=COLORS[animal], edgecolor="white", linewidth=.6, zorder=5)
    ax.axhline(0, color=".45", lw=.8)
    ax.grid(axis="y", color=".93", lw=.6)
    ax.set(xticks=(0, 1), xticklabels=("Primary prediction", "Trial identity"), xlim=(-.5, 1.5),
           ylabel="I blend − individual R blend (×10⁻³ R²)")
    ax.set_title("D  Every one of the 8 ordinary random-support draw means", loc="left", fontsize=11, pad=10)
    ax.margins(y=.18)
    ax.text(.98, .97, "Open circles: individual draws; all signs retained", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=".35")

    fig.suptitle("Guide-support robustness of the actual constant W/HW blend", x=.075, ha="left", y=.983, fontsize=17)
    fig.text(.075, .952,
             f"Own-baseline gains — I: primary {intersection['primary_increment']:+.6f}, identity {intersection['identity_increment']:+.6f}; "
             f"R: primary {random['primary_increment']:+.6f}, identity {random['identity_increment']:+.6f}.", fontsize=11.2, color="#263e57")
    fig.text(.075, .927,
             f"{min(counts)}–{max(counts)} electrodes/session; original reference margins 0.010 primary / 0.005 identity. "
             "Changed-support sensitivities cannot reset the original-support incumbent.", fontsize=10.4, color=".3")
    def ids(values):
        return ",".join(map(str, values)) if values else "none"
    for line, animal in enumerate(ANIMAL_ORDER):
        primary, identity = (exceptions[metric][animal] for metric in METRICS)
        fig.text(.075, .224-.019*line,
                 f"{animal} draw exceptions (IDs 0–7): nonpositive means, primary [{ids(primary['nonpositive_draw_means'])}], "
                 f"identity [{ids(identity['nonpositive_draw_means'])}]; nonpositive omission in primary [{ids(primary['draws_with_nonpositive_omissions'])}], "
                 f"identity [{ids(identity['draws_with_nonpositive_omissions'])}].", fontsize=8.9, color=".3")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0, label=animal+" mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean"),
                  Line2D([], [], color=".3", marker="o", markerfacecolor="none", lw=0, label="Individual draw mean (D)")],
               loc="lower left", bbox_to_anchor=(.069, .168), ncol=4, frameon=False, fontsize=9.5)
    fig.text(.075, .026,
             "A/B small dots: all 16 signed session effects; solid bars: fixed-prediction session-omission ranges. Both panels share one complete linear scale.\n"
             "D uses its own full signed scale: open circles/dotted spans show all 8 draw means/their range; filled markers show the mean-R contrast.\n"
             "These ranges are not confidence intervals or p-values. Overlapping support views are not biological replicates: 2 animals, with 6 and 10 sessions.\n"
             "I uses recorded guide membership; R uses the exact eight ordinary subsets of equal size. True trial pairing remains intact; these are not pairing-null models.\n"
             "Full blend and prior I/R-HW keep their original selections. New-λ own-HW comparisons use I-blend's selected penalty; both versions remain separately displayed.\n"
             "Only reduced current-W states are newly fit; saved HW predictions stay fixed. Mixtures retain error cross-terms, and R averages SSEs rather than predictions.\n"
             "The compact source retains full-native-error cross-products and separate W/HW trace/df: neither is an isolated neural mechanism. Equal counts or λ do not match capacity.\n"
             "Guide membership does not identify spike contamination. Exposed development data; no promotion. CSV includes every draw's sessions/omissions and selected α/λ values.",
             fontsize=8.6, color=".3", linespacing=1.45)
    fig.tight_layout(rect=(.025, .273, .995, .904), h_pad=3, w_pad=3)
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
                      "draw_exceptions": exceptions, "source_evidence": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
