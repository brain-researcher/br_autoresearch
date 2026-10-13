#!/usr/bin/env python3
"""Plot the complete saved W/HW blend and its original-support contrasts."""
import csv
import json
from pathlib import Path

from plot_all_band import draw_groups
from plot_repeatability_weight import ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows
from primary_history_blend import RECIPE
from primary_reference import SESSIONS
from source_adapter import compute_only


SPECS = (("delta_R2", "primary_increment", "primary_by_animal"),
         ("identity_increment", "identity_increment", "identity_by_animal"))


def describe(panel, compact, condition, source):
    rows = panel["sessions"]
    validate_rows(rows)
    exported, summary = description_rows(rows, condition)
    for metric, key, animal_key in SPECS:
        means = group_means(rows, metric)
        if any(abs(means["Overall"]-packet[key]) > 1e-12 for packet in (panel, compact)):
            raise ValueError("Complete session rows must reproduce both published equal-animal means")
        omissions = compact["fixed_prediction_omissions"][metric]
        for animal in (*ANIMAL_ORDER, "Overall"):
            center, low, high = summary[metric][animal]
            limits = omissions["overall_range"] if animal == "Overall" else omissions["animal_ranges"][animal]
            if abs(low-limits[0]) > 1e-12 or abs(high-limits[1]) > 1e-12:
                raise ValueError("Published and reconstructed fixed-prediction omissions differ")
            if animal != "Overall" and any(abs(center-packet[animal_key][animal]) > 1e-12
                                           for packet in (panel, compact)):
                raise ValueError("Published and reconstructed animal means differ")
    return rows, summary, [{**row, "source_summary": str(source),
                            "selection": "lambda=1 in every outer fold; blend alpha=0.5"} for row in exported]


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    directory = root / "runs/adaptive_search/primary_history_blend_summary_attempt001"
    source = directory / "primary_history_blend.json"
    report = json.loads(source.read_text())
    compact = json.loads((directory/"compact_evidence.json").read_text())
    for packet in (report, compact):
        if (packet["status"] != "complete_history_blend_panel" or packet["recipe"] != RECIPE
                or packet["scientific_trial_count_increment"] != 1 or packet["structural_duplicate_of"]
                or packet["literal_null_control_count_increment"] != 0
                or packet["alternative_explanation_count_increment"] != 0
                or packet["neural_refits"] != 0 or packet["provider_source_values_read"] is not False
                or packet["promotion_authorized"] is not False
                or any(packet[key] is not True for key in ("original_selectors_independently_replayed",
                    "all_selected_prediction_scores_replayed", "all_original_component_arrays_support_replay",
                    "all_original_component_scores_replay"))):
            raise ValueError("Only the completed distinct original-support saved-prediction blend may be plotted")
    candidate = report["blend"]
    panels = report["original_components"]
    if set(panels) != {"W", "HW"} or set(report["selected_prediction_paths"]) != set(SESSIONS):
        raise ValueError("Both preserved components and every selected session required")
    for panel in (candidate, *panels.values()):
        validate_rows(panel["sessions"])
        for row in panel["sessions"]:
            if (row["lambda"] != 1. or len(row["outer_training_sessions"]) != 15
                    or set(row["outer_training_sessions"]) != set(SESSIONS)-{row["session_id"]}):
                raise ValueError("Each original-support choice must exclude its evaluated outer session")
    if (any(row["alpha"] != .5 or row["literal_endpoint_reused"] is not None
            or any(row["exactly_matches_original_selected_components"].values()) for row in candidate["sessions"])
            or compact["alpha_counts"] != {"0.5": 16}
            or compact["joint_lambda_alpha_counts"] != {"lambda=1.0,alpha=0.5": 16}
            or candidate["primary_increment"] >= .01 or candidate["identity_increment"] >= .005):
        raise ValueError("Released framing requires the distinct half blend and both failed practical margins")

    own = [describe(panels["W"], compact["original_components"]["W"], "W: current only", source),
           describe(candidate, compact["candidate"], "Blend: half W plus half HW", source),
           describe(panels["HW"], compact["original_components"]["HW"], "HW: four-bin history", source)]
    candidate_by_id = {row["session_id"]: row for row in candidate["sessions"]}
    direct = []
    for name in ("W", "HW"):
        key = name+"_independently_selected"
        contrast = report["reference_contrasts"][key]
        matched = report["reference_contrasts"][name+"_blend_lambda_matched"]
        validate_rows(contrast["sessions"])
        validate_rows(matched["sessions"])
        originals = {row["session_id"]: row for row in panels[name]["sessions"]}
        matched_by_id = {row["session_id"]: row for row in matched["sessions"]}
        for row in contrast["sessions"]:
            session = row["session_id"]
            if row["lambda"] != 1. or row["reference_lambda"] != 1.:
                raise ValueError("Original and blend-matched penalties coincide only at the released selections")
            for metric in METRICS:
                expected = candidate_by_id[session][metric]-originals[session][metric]
                if abs(row[metric]-expected) > 1e-12 or row[metric] != matched_by_id[session][metric]:
                    raise ValueError("Direct effects must retain paired session arithmetic and identical matched-lambda contrast")
        direct.append(describe(contrast, compact["reference_contrasts"][key], "Blend minus "+name, source))
    hw_contrast = report["reference_contrasts"]["HW_independently_selected"]
    hw_omissions = compact["reference_contrasts"]["HW_independently_selected"]["fixed_prediction_omissions"]
    if (hw_contrast["primary_positive_session_counts"]["Chewie"] != 3
            or hw_omissions["delta_R2"]["animal_positive_sign_changes"]["Chewie"] != ["Chewie_20150630"]
            or not all(direct[1][1]["identity_increment"][animal][2] < 0 for animal in ANIMAL_ORDER)
            or not all(own[1][1][metric][animal][1] > 0 for metric in METRICS for animal in ANIMAL_ORDER)):
        raise ValueError("Keep own positive animal/omission scores separate from the fragile HW advantage")

    stem = root / "figures/primary_history_blend_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing completed-result figures")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.6), sharey="row")
    for column, (metric, _, _) in enumerate(SPECS):
        ax = axes[0, column]
        draw_groups(ax, own, metric, ("W\nCurrent LMP", "Blend\n½ W + ½ HW", "HW\nFour-bin LMP"))
        margin = .01 if column == 0 else .005
        ax.axhline(1000*margin, color="#9d2727", ls="--", lw=1)
        ax.text(.98, margin*1000, f" Required margin {margin:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color="#9d2727", fontsize=9)
        ax.set_title(("A  Prediction gain over direction/time baseline", "B  Correct versus wrong trial pairing")[column],
                     loc="left", fontsize=11, pad=10)
        ax.set_ylabel("Increment in joint native-XY R² (×10⁻³)")
        ax = axes[1, column]
        draw_groups(ax, direct, metric, ("Blend − W", "Blend − HW"))
        ax.set_title(("C  Direct primary prediction differences", "D  Direct trial-pairing differences")[column],
                     loc="left", fontsize=11, pad=10)
        ax.set_ylabel("Difference in increment (×10⁻³ R²)")
        ax.text(.98, .96, "Positive favors blend", transform=ax.transAxes,
                ha="right", va="top", color=".35", fontsize=9)
    bound = max(abs(row[metric])*1000 for group in direct for row in group[0] for metric in METRICS)
    bound = max(bound, *(abs(value)*1000 for group in direct for metric in METRICS
                         for triple in group[1][metric].values() for value in triple))
    axes[1, 0].set_ylim(-1.15*bound, 1.15*bound)
    fig.suptitle("Current/history blend: higher primary gain, lower identity gain than HW",
                 x=.07, y=.981, ha="left", fontsize=16)
    fig.text(.07, .944,
             f"Blend primary {candidate['primary_increment']:.6f} < 0.010; identity {candidate['identity_increment']:.6f} < 0.005. Both practical margins fail.",
             fontsize=11.5, weight="bold", color="#9d2727")
    fig.text(.07, .916,
             "Other 15 sessions select λ=1 and α=0.5 in every outer fold; W/HW anchors also select λ=1. No neural refits.",
             fontsize=10.5, color="#263e57")
    fig.legend(handles=[Line2D([], [], color=COLORS[a], marker="D", lw=0, label=a+" mean")
                        for a in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.064, .145), ncol=3, frameon=False, fontsize=10)
    fig.text(.07, .119,
             "Blend’s own animal/omission gains stay positive. Against HW, only 3/10 Chewie sessions improve; omitting June 30 makes its mean −0.00000305.",
             fontsize=9.8, color="#6b3622")
    fig.text(.07, .026,
             "Small dots: all 16 sessions. Diamonds/squares: animal/equal-animal means. Bars: fixed-prediction session-omission ranges, not confidence intervals.\n"
             "Each row shares a vertical scale; direct contrasts use a common symmetric scale and retain every signed effect. Identity falls versus HW in both animals.\n"
             "W uses current LMP; HW uses four LMP bins with the same current-bin repeatability prior. The blend intentionally averages their predicted trajectories.\n"
             "Scores use identical native XY targets, B and denominators. Independent and blend-λ-matched anchor contrasts coincide here. Omission scores do not refit.\n"
             "Exposed development data · 2 animals · source/target support unchanged · improved search incumbent does not imply eligibility or promotion.",
             fontsize=8.7, color=".3")
    fig.tight_layout(rect=(.025, .211, .995, .887), h_pad=2.5, w_pad=2.4)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("."+extension), dpi=180)
    plt.close(fig)
    exported = [row for group in (*own, *direct) for row in group[2]]
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(exported),
                      "source_summary": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
