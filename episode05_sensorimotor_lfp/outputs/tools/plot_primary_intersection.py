#!/usr/bin/env python3
"""Plot only the COMPLETE guide-intersection support sensitivity evidence."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

from plot_all_band import draw_groups
from plot_repeatability_weight import (
    ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows,
)
from primary_intersection import RECIPES, REPLICATES
from source_adapter import compute_only


SPECS = (("delta_R2", "primary_increment", "primary_by_animal"),
         ("identity_increment", "identity_increment", "identity_by_animal"))


def describe(rows, condition, source, replicate="", published=None):
    validate_rows(rows)
    exported, summary = description_rows(rows, condition)
    if published is not None:
        if published["lambda_counts"] != {"1.0": 16} or any(row["lambda"] != 1.0 for row in rows):
            raise ValueError("Complete intersection-selected lambda=1 panel required")
        for metric, key, animal_key in SPECS:
            centers = group_means(rows, metric)
            expected = published["fixed_prediction_omissions"][metric]
            for animal in (*ANIMAL_ORDER, "Overall"):
                published_center = published[key] if animal == "Overall" else published[animal_key][animal]
                interval = expected["overall_range"] if animal == "Overall" else expected["animal_ranges"][animal]
                if (not all(math.isfinite(value) for value in (published_center, *interval))
                        or abs(centers[animal]-published_center) > 1e-12
                        or any(abs(value-original) > 1e-12 for value, original in zip(summary[metric][animal][1:], interval))):
                    raise ValueError("Published equal-animal means and fixed-omission ranges must replay")
    return rows, summary, [{**row, "replicate": replicate, "source_evidence": str(source)} for row in exported]


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_intersection_sensitivity_summary_attempt001/compact_evidence.json"
    report = json.loads(source.read_text())
    if (report["status"] != "complete_intersection_sensitivity_panel" or report["recipe_order"] != list(RECIPES)
            or report["sensitivity_only"] is not True
            or any(report[key] is not False for key in ("promotion_authorized", "incumbent_reset", "spike_contamination_identified"))
            or any(report[key] != 2 for key in ("scientific_trial_count_increment", "stress_diagnostic_count_increment",
                                               "alternative_explanation_count_increment"))
            or report["literal_null_control_count_increment"] != 0
            or report["scientific_trial_count_by_recipe"] != dict.fromkeys(RECIPES, 1)
            or report["structural_duplicate_of"] != dict.fromkeys(RECIPES)):
        raise ValueError("Complete nonpromotable guide-support sensitivity with unchanged null accounting required")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES):
        raise ValueError("Both complete I/R panels required")
    intersection, random = (panels[recipe] for recipe in RECIPES)
    own_groups = [describe(intersection["sessions"], "I: guide intersection", source, published=intersection),
                  describe(random["sessions"], "R: mean of eight count-matched random supports", source, published=random)]
    direct = report["intersection_minus_random_support"]
    direct_group = describe(direct["sessions"], "I minus mean random support", source, published=direct)
    refs = report["reference_contrasts"]
    if set(refs) != {"independently_selected", "intersection_lambda_matched"}:
        raise ValueError("Original and intersection-lambda-matched full-HW references required")
    full = refs["independently_selected"]
    full_group = describe(full["sessions"], "I minus full HW (original and I-matched lambda coincide)", source, published=full)
    describe(refs["intersection_lambda_matched"]["sessions"], "I minus full HW at I lambda", source,
             published=refs["intersection_lambda_matched"])
    if full["sessions"] != refs["intersection_lambda_matched"]["sessions"]:
        raise ValueError("Collapse full-HW lambda references only when all session effects are identical")

    candidate_by_id = {row["session_id"]: row for row in intersection["sessions"]}
    random_by_id = {row["session_id"]: row for row in random["sessions"]}
    for row in direct["sessions"]:
        for metric in METRICS:
            session = row["session_id"]
            if abs(candidate_by_id[session][metric]-random_by_id[session][metric]-row[metric]) > 1e-12:
                raise ValueError("I/R differences must use the same original targets and denominator")
    if (any(panel["primary_increment"] >= .01 or panel["identity_increment"] >= .005 for panel in panels.values())
            or any(direct_group[1][metric][animal][1] <= 0 for metric in METRICS for animal in ANIMAL_ORDER)
            or any(full_group[1][metric][animal][0] >= 0 for metric in METRICS for animal in ANIMAL_ORDER)
            or not (full_group[1]["delta_R2"]["Mihili"][1] < 0 < full_group[1]["delta_R2"]["Mihili"][2])):
        raise ValueError("Framing requires failed margins, positive mean I/R omissions, full-HW losses and the Mihili omission exception")

    seeds = report["random_support_panels"]
    if [seed["replicate"] for seed in seeds] != list(REPLICATES):
        raise ValueError("Every one of the eight ordinary fixed support draws required")
    seed_groups, seed_maps = [], []
    for seed in seeds:
        describe(seed["sessions"], "Random support draw", source, replicate=seed["replicate"], published=seed)
        rows = [{"session_id": row["session_id"], "animal": row["animal"],
                 **{metric: candidate_by_id[row["session_id"]][metric]-row[metric] for metric in METRICS}}
                for row in seed["sessions"]]
        group = describe(rows, "I minus individual random support", source, replicate=seed["replicate"])
        for metric, key, animal_key in SPECS:
            for animal in ANIMAL_ORDER:
                if abs(group[1][metric][animal][0]-(intersection[animal_key][animal]-seed[animal_key][animal])) > 1e-12:
                    raise ValueError("Individual draw contrasts must preserve animal-balanced paired differences")
        seed_groups.append(group)
        seed_maps.append({row["session_id"]: row for row in seed["sessions"]})
    for session, row in random_by_id.items():
        for metric in METRICS:
            if abs(mean(seed[session][metric] for seed in seed_maps)-row[metric]) > 1e-12:
                raise ValueError("R must average all eight score/SSE views, not predictions")

    support = report["selected_support"]
    if len(support) != 16 or {row["session_id"] for row in support} != set(candidate_by_id):
        raise ValueError("Complete original-cohort support metadata required")
    counts = []
    for row in support:
        count = row["electrode_count"]
        views = row["random_supports"]
        if (count != len(set(row["intersection_electrodes"])) or row["lambda"] != 1.0
                or [view["replicate"] for view in views] != list(REPLICATES)
                or any(len(set(view["electrode_ids"])) != count or view["full_support"] is not False for view in views)
                or any(not set(row["intersection_electrodes"]).issubset(view["electrode_queue"])
                       or count >= len(view["electrode_queue"]) for view in views)
                or row["finite_support_ranges_are_confidence_intervals"] is not False):
            raise ValueError("Strict guide intersection and eight equally sized ordinary support subsets required")
        counts.append(count)

    exceptions = {metric: {animal: {
        "nonpositive_draw_means": [r for r, group in zip(REPLICATES, seed_groups) if group[1][metric][animal][0] <= 0],
        "draws_with_nonpositive_omissions": [r for r, group in zip(REPLICATES, seed_groups) if group[1][metric][animal][1] <= 0],
    } for animal in ANIMAL_ORDER} for metric in METRICS}
    if exceptions["delta_R2"]["Mihili"]["nonpositive_draw_means"] != [4]:
        raise ValueError("This released figure must retain the negative Mihili draw-4 primary contrast")

    stem = root / "figures/primary_intersection_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")
    exported = [row for group in (*own_groups, full_group, direct_group, *seed_groups) for row in group[2]]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 12.6), sharey="row")
    offsets = {"Mihili": -.20, "Chewie": .20, "Overall": 0.}
    for column, (metric, _, _) in enumerate(SPECS):
        ax = axes[0, column]
        draw_groups(ax, own_groups, metric, ("I: unit-guide intersection", "R: count-matched random subsets\nMean SSE across 8 draws"))
        margin = .01 if column == 0 else .005
        ax.axhline(margin*1000, color="#9d2727", ls="--", lw=1)
        ax.text(.98, margin*1000, f" Practical margin {margin:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8.8, color="#9d2727")
        ax.set_title(("A  Prediction gain over direction/time baseline",
                      "B  Correct versus wrong within-direction trial pairing")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Increment in joint native-XY R² (×10⁻³)")

        ax = axes[1, column]
        draw_groups(ax, [full_group, direct_group], metric,
                    ("I − full HW\nPreserved original reference", "I − mean R\n16 paired session effects"))
        for animal in (*ANIMAL_ORDER, "Overall"):
            values = [group[1][metric][animal][0] for group in seed_groups]
            position = 2+offsets[animal]
            ax.plot([position, position], [min(values)*1000, max(values)*1000],
                    color=COLORS[animal], lw=1.4, ls=":", zorder=2)
            for replicate, value in zip(REPLICATES, values):
                location = position+.11*(replicate/7-.5)
                ax.scatter(location, value*1000, marker="o", s=34, facecolors="none",
                           edgecolors=COLORS[animal], linewidth=.9, zorder=4)
                if metric == "delta_R2" and animal == "Mihili" and replicate == 4:
                    ax.annotate("draw 4", (location, value*1000), xytext=(-4, -17), textcoords="offset points",
                                ha="right", color=COLORS[animal], fontsize=8.2,
                                arrowprops={"arrowstyle": "-", "color": COLORS[animal], "lw": .6})
        ax.set(xticks=(0, 1, 2), xticklabels=("I − full HW\nPreserved original reference", "I − mean R\n16 paired session effects",
                                            "I − each random subset\n8 fixed draw means"), xlim=(-.48, 2.48))
        ax.set_title(("C  Direct prediction comparisons", "D  Direct trial-pairing comparisons")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Difference in increment (×10⁻³ R²)")
        ax.text(.98, .97, "Positive favors intersection", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=".35")
        ax.margins(y=.14)
        for row_index in range(2):
            axes[row_index, column].tick_params(labelleft=True)

    fig.suptitle("Guide-intersection HW beats mean random subsets, but trails full HW",
                 x=.075, ha="left", y=.982, fontsize=15.6)
    fig.text(.075, .949,
             f"Intersection primary gain {intersection['primary_increment']:.6f} < 0.010; identity gain {intersection['identity_increment']:.6f} < 0.005.",
             fontsize=11.4, weight="bold", color="#9d2727")
    fig.text(.075, .923,
             f"Metadata-defined support: {min(counts)}–{max(counts)} electrodes/session; eight ordinary subsets of the same size; all selected λ=1.",
             fontsize=10.4, color="#263e57")
    failing = exceptions["delta_R2"]["Mihili"]["draws_with_nonpositive_omissions"]
    fig.text(.075, .226,
             "Mean I − R stays positive in both animals after every omission; individual random subsets have exceptions.",
             fontsize=10.2, weight="bold", color="#263e57")
    fig.text(.075, .204,
             "Mihili primary: draw 4 favors the random subset; nonpositive omission results occur for draws "
             + ", ".join(str(r) for r in failing) + ". Draw IDs are 0–7.", fontsize=9.5, color=".3")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0,
                               label=animal+" mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean"),
                  Line2D([], [], color=".3", marker="o", markerfacecolor="none", lw=0, label="Individual draw mean")],
               loc="lower left", bbox_to_anchor=(.069, .164), ncol=4, frameon=False, fontsize=9.5)
    fig.text(.075, .025,
             "Small filled dots: all 16 sessions. Solid bars: fixed-prediction session-omission ranges. Open dots/dotted spans: 8 draw means and their range.\n"
             "Ranges are descriptive, not confidence intervals or p-values. The overlapping subsets are not independent biological replicates (2 animals: 6 and 10 sessions).\n"
             "Guide membership does not identify spike contamination; no spike values were used. This is a nonpromotable support sensitivity, not a pairing null.\n"
             "I/R preserve true neural–behavioral pairing, original trials, native XY targets and baseline. Each reduced support refits its own HW neural state.\n"
             "Matching electrode counts or λ does not match anatomical coverage, fitted covariance or effective df. R averages correct/wrong-pair SSEs, not predictions.\n"
             "Full HW predictions remain unchanged. Original- and I-selected λ coincide, so both full-HW contrasts overlap exactly; every contrast shares original D.\n"
             "I − full HW is negative in both animal means; Mihili's primary sign flips when 2014-03-07 is omitted. All signed session and draw values remain visible.\n"
             "Exposed development data · row scales differ, both metrics share each row's full linear scale · CSV includes all individual-draw sessions and omissions.",
             fontsize=8.5, color=".3", linespacing=1.45)
    fig.tight_layout(rect=(.025, .266, .995, .897), h_pad=2.8, w_pad=2.5)
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
                      "random_draw_exceptions": exceptions, "source_evidence": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
