#!/usr/bin/env python3
"""Plot only the released COMPLETE saved-prediction time-schedule evidence."""
import csv
import json
from collections import Counter
from pathlib import Path

from plot_all_band import draw_groups
from plot_capacity_matched_surrogate import describe
from plot_repeatability_weight import ANIMAL_ORDER, COLORS, METRICS, validate_rows
from primary_timevarying_history_blend import RECIPE, alpha_schedule
from run_primary_timevarying_history_blend import PANEL_STATUS, REPLAY_FLAGS
from source_adapter import compute_only


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_timevarying_history_blend_summary_attempt001/compact_evidence.json"
    report = json.loads(source.read_text())
    account, replay = report["accounting"], report["exact_replay"]
    if (report["status"] != PANEL_STATUS or report["recipe"] != RECIPE
            or account["scientific_trial_count_increment"] != 1 or account["structural_duplicate_of"] != []
            or account["eligible_for_comparable_primary_incumbent"] is not True
            or any(account[key] != 0 for key in ("literal_null_control_count_increment", "alternative_explanation_count_increment",
                "reversal_scientific_trial_count_increment", "neural_refits"))
            or any(account[key] is not False for key in ("provider_source_values_read", "promotion_authorized",
                "all_reference_margin_screens_pass"))
            or account["reversal_descriptive_only"] is not True
            or any(replay[key] is not True for key in (*REPLAY_FLAGS, "all_selected_prediction_scores_replayed",
                "all_selected_reversal_scores_replayed", "all_exact_duplicate_flags_replayed"))):
        raise ValueError("Complete original-support replayed schedule and uncharged descriptive reversal required")
    candidate, constant, reversed_panel = report["candidate"], report["original_anchors"]["blend"], report["reversed_slope"]
    constant_contrast = report["reference_contrasts"]["blend_independently_selected"]
    reverse_contrast = report["selected_minus_reversed"]
    candidate_group = describe(candidate["sessions"], candidate, "Selected time schedule", source)
    constant_group = describe(constant["sessions"], constant, "Preserved constant blend", source)
    direct_groups = [describe(constant_contrast["sessions"], constant_contrast, "Selected minus constant blend", source),
                     describe(reverse_contrast["sessions"], reverse_contrast, "Selected minus reversed slope at same lambda", source)]
    describe(reversed_panel["sessions"], reversed_panel, "Fixed reversed schedule", source)
    matched = report["reference_contrasts"]["half_blend_schedule_lambda_matched"]
    describe(matched["sessions"], matched, "Selected minus half-blend at selected lambda", source)
    if (constant_contrast["sessions"] != matched["sessions"]
            or any(panel["lambda_counts"] != {"1.0": 16} for panel in
                   (candidate, constant, reversed_panel, constant_contrast, reverse_contrast, matched))
            or candidate["primary_increment"] >= .01 or candidate["identity_increment"] >= .005
            or candidate["primary_increment"] >= constant["primary_increment"]):
        raise ValueError("Framing requires coinciding original/half-blend references, lambda=1 and failure to improve primary")

    selected = report["selected_sessions"]
    validate_rows(selected)
    counts = Counter(str(row["slope"]) for row in selected)
    if (dict(counts) != {"0.0": 9, "0.25": 7} or report["slope_counts"] != dict(counts)
            or report["slope_sign_counts"] != {"zero": 9, "positive": 7}):
        raise ValueError("All nine unchanged and seven increasing-weight selections must remain visible")
    selected_by_id = {row["session_id"]: row for row in selected}
    candidate_by_id = {row["session_id"]: row for row in candidate["sessions"]}
    constant_by_id = {row["session_id"]: row for row in constant["sessions"]}
    reverse_by_id = {row["session_id"]: row for row in reversed_panel["sessions"]}
    direct_maps = [{row["session_id"]: row for row in group[0]} for group in direct_groups]
    for row in selected:
        session, slope = row["session_id"], row["slope"]
        if (row["lambda"] != 1.0 or row["alpha_by_bin"] != alpha_schedule(slope).tolist()
                or row["exact_prediction_equals_originals"]["blend"] != (slope == 0)):
            raise ValueError("Literal saved schedules and zero-slope original-blend identity required")
        for metric in METRICS:
            if (row[metric] != candidate_by_id[session][metric]
                    or abs(row[metric]-constant_by_id[session][metric]-direct_maps[0][session][metric]) > 1e-12
                    or abs(row[metric]-reverse_by_id[session][metric]-direct_maps[1][session][metric]) > 1e-12):
                raise ValueError("All direct contrasts must replay on the same original denominator")
            if slope == 0 and any(mapping[session][metric] != 0 for mapping in direct_maps):
                raise ValueError("All nine zero-slope sessions must retain exact zero contrasts")
        # Wrong-minus-correct SSE cancels the quadratic prediction term. With
        # symmetric schedules, this identity contrast is algebraically doubled.
        if abs(direct_maps[1][session]["identity_increment"]-2*direct_maps[0][session]["identity_increment"]) > 1e-12:
            raise ValueError("Identity reversal must preserve its affine prediction identity")
    active = {animal: [row for row in selected if row["animal"] == animal and row["slope"] != 0]
              for animal in ANIMAL_ORDER}
    if (len(active["Mihili"]) != 4 or len(active["Chewie"]) != 3
            or any(mapping[row["session_id"]]["delta_R2"] >= 0 for row in active["Chewie"] for mapping in direct_maps)
            or constant_contrast["within_animal_omission_ranges"]["Chewie"]["identity_increment"]["nonpositive_omissions"] != ["Chewie_20150629"]
            or reverse_contrast["within_animal_omission_ranges"]["Chewie"]["identity_increment"]["nonpositive_omissions"] != ["Chewie_20150629"]):
        raise ValueError("Framing requires all three active Chewie primary losses and the identity omission exception")

    stem = root / "figures/primary_timevarying_history_blend_attempt001"
    if any(stem.with_suffix("."+extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing complete-result figures")
    exported = [{**row, "target_bin": "", "selected_slope": "", "selected_lambda": ""}
                for group in (candidate_group, constant_group, *direct_groups) for row in group[2]]
    for row in selected:
        for target_bin, alpha in enumerate(row["alpha_by_bin"]):
            exported.append({"condition": "Selected HW mixing coefficient", "metric": "alpha", "level": "selected_schedule_bin",
                "animal": row["animal"], "session": row["session_id"], "excluded_session": "", "value": alpha,
                "omission_min": "", "omission_max": "", "replicate": "", "derivation": "Published selected coefficient",
                "source_evidence": str(source), "target_bin": target_bin, "selected_slope": row["slope"], "selected_lambda": row["lambda"]})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 12.9))
    ax = axes[0, 0]
    for slope, label, color, style in ((0., "s=0: 9 sessions", ".35", "-"),
                                     (.25, "s=+0.25: 7 sessions", "#765a93", "-"),
                                     (-.25, "Reversal of active schedule", ".65", "--")):
        ax.plot(range(15), alpha_schedule(slope), color=color, ls=style, lw=2,
                marker="o" if slope >= 0 else None, ms=3, label=label)
    ax.set(xlabel="Original target bin (0–14; 30 ms/bin)", ylabel="HW coefficient α; W coefficient = 1 − α",
           xlim=(-.2, 14.2), ylim=(0, 1), xticks=(0, 7, 14), yticks=(0, .25, .5, .75, 1))
    ax.set_title("A  Selected coefficients and fixed reversal", loc="left", fontsize=10.5, pad=10)
    ax.grid(color=".93", lw=.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(.03, .08, "Mean coefficient 0.5\nNo neural refit; no per-bin outcome selection", transform=ax.transAxes,
            fontsize=9.5, color=".35")

    ax = axes[0, 1]
    ordered = [row for animal in ANIMAL_ORDER for row in selected if row["animal"] == animal]
    for index, row in enumerate(ordered):
        ax.scatter(row["slope"], index, s=35, color=COLORS[row["animal"]],
                   marker="s" if row["slope"] == 0 else "o", zorder=3)
    ax.axhline(5.5, color=".8", lw=.8)
    ax.set(yticks=range(16), yticklabels=[row["session_id"].replace("Mihili_", "Mi ").replace("Chewie_", "Ch ") for row in ordered],
           xticks=(0, .25), xticklabels=("0: unchanged blend", "+0.25: increasing HW weight"), xlim=(-.045, .295), ylim=(15.7, -.7))
    ax.tick_params(axis="y", labelsize=8.3)
    ax.tick_params(axis="x", labelsize=8.6)
    ax.grid(axis="x", color=".92", lw=.6)
    ax.set_title("B  Other-15-session selections: 4 Mihili + 3 Chewie active", loc="left", fontsize=10.3, pad=10)

    axes[1, 1].sharey(axes[1, 0])
    for column, metric in enumerate(METRICS):
        ax = axes[1, column]
        draw_groups(ax, direct_groups, metric, ("Selected − constant blend\nOriginal and matched-λ references coincide",
                                               "Selected − reversed slope\nSame λ; reversal never retuned"))
        ax.set_title(("C  Direct primary prediction differences", "D  Direct trial-pairing differences")[column],
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Difference in native-XY increment (×10⁻³ R²)")
        ax.text(.98, .97, "Positive favors selected schedule", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=".35")
        ax.text(.03, .04, "All 16 sessions shown, including 9 exact zeros", transform=ax.transAxes, fontsize=8.8, color=".35")
        ax.tick_params(labelleft=True)

    fig.suptitle("A time-varying mixture does not improve the constant blend's primary score",
                 x=.075, ha="left", y=.982, fontsize=15.1)
    fig.text(.075, .949,
             f"Primary {candidate['primary_increment']:.6f} < 0.010; identity {candidate['identity_increment']:.6f} < 0.005. "
             f"Primary change versus blend {constant_contrast['primary_increment']:+.7f}.",
             fontsize=11.1, weight="bold", color="#9d2727")
    fig.text(.075, .922, "All λ=1 · nine folds reuse the constant half-blend exactly · seven choose HW weight 0.25 → 0.75",
             fontsize=10.3, color="#263e57")
    fig.text(.075, .205, "Every active Chewie session loses primary accuracy against both constant and reversed schedules.",
             fontsize=10.2, weight="bold", color="#263e57")
    fig.text(.075, .184,
             f"Chewie identity change versus constant is only {constant_contrast['identity_by_animal']['Chewie']:+.7f}; "
             "omitting 2015-06-29 makes it negative.", fontsize=9.8, color=".3")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0, label=animal+" mean") for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.069, .144), ncol=3, frameon=False, fontsize=9.5)
    fig.text(.075, .025,
             "C/D small dots: all 16 signed session effects. Solid bars: fixed-prediction session-omission ranges; not confidence intervals or p-values.\n"
             "Both effect panels share one full signed linear scale. Two animals (6 and 10 sessions), equally weighted after within-animal averaging.\n"
             "The reversal is an unretuned descriptive contrast, not an independent control or extra trial. At zero slope it is literally the same prediction.\n"
             "Identity is affine in prediction: selected-minus-reversed is exactly twice selected-minus-half-blend here, not independent corroboration.\n"
             "All original W/HW predictions, trials, native XY baseline and denominator remain fixed. α was selected using only the other fifteen sessions.\n"
             "A fixed mean mixing coefficient does not fix mean prediction contribution or capacity. These schedules do not identify physiological timing or causal lag effects.\n"
             "Exposed development data · both practical margins still fail · no incumbent reset · CSV preserves all selected coefficients, session effects and omissions.",
             fontsize=8.5, color=".3", linespacing=1.45)
    fig.tight_layout(rect=(.025, .251, .995, .895), h_pad=3, w_pad=3)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("."+extension), dpi=180)
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(exported), "source_evidence": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
