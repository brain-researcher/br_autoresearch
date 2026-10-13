#!/usr/bin/env python3
"""Plot the complete all-band panel and five contrasts; no source reads or fits."""
import csv
import json
from pathlib import Path

from plot_repeatability_weight import (
    ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows,
)
from primary_all_band import RECIPES
from source_adapter import compute_only


CONTRAST_KEYS = (
    "joint_independently_selected", "joint_all_band_lambda_matched",
    "lmp_independently_selected", "lmp_all_band_lambda_matched",
)


def panel_description(panel, name, source):
    """Recompute displayed means and omission ranges from complete session rows."""
    validate_rows(panel["sessions"])
    for metric, key in zip(METRICS, ("primary_increment", "identity_increment")):
        if abs(group_means(panel["sessions"], metric)["Overall"] - panel[key]) > 1e-12:
            raise ValueError("Reported and recomputed equal-animal means differ")
    rows, summary = description_rows(panel["sessions"], name)
    exported = [{**row, "source_summary": str(source),
                 "recipe_or_contrast": panel.get("recipe", panel.get("contrast"))}
                for row in rows]
    return panel["sessions"], summary, exported


def draw_groups(ax, groups, metric, labels):
    offsets = {"Mihili": -.18, "Chewie": .18, "Overall": 0.}
    for position, (rows, summary, _) in enumerate(groups):
        for animal in ANIMAL_ORDER:
            selected = [row for row in rows if row["animal"] == animal]
            for index, row in enumerate(selected):
                spread = .12 * (index / (len(selected) - 1) - .5)
                ax.scatter(position + offsets[animal] + spread, row[metric] * 1000,
                           s=27, color=COLORS[animal], alpha=.52, linewidth=.4,
                           edgecolor="white", zorder=3)
        for animal in (*ANIMAL_ORDER, "Overall"):
            center, low, high = summary[metric][animal]
            location = position + offsets[animal]
            ax.plot([location, location], [low * 1000, high * 1000],
                    color=COLORS[animal], lw=2, zorder=4)
            for endpoint in (low, high):
                ax.plot([location - .025, location + .025], [endpoint * 1000] * 2,
                        color=COLORS[animal], lw=1.2, zorder=4)
            ax.scatter(location, center * 1000, marker="s" if animal == "Overall" else "D",
                       s=48, color=COLORS[animal], edgecolor="white", linewidth=.6, zorder=5)
    ax.axhline(0, color=".45", lw=.8)
    ax.set(xticks=range(len(groups)), xticklabels=labels, xlim=(-.48, len(groups) - .52))
    ax.grid(axis="y", color=".93", lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    # Autoscaling retains every signed session score, omission endpoint and zero.
    # Practical margins, when added by the caller, also enter the full linear axis.
    ax.margins(y=.12)


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    source = root / "runs/adaptive_search/primary_all_band_summary_attempt001/primary_all_band.json"
    report = json.loads(source.read_text())
    if (report.get("status") != "complete_all_band_control_panel"
            or report.get("promotion_authorized") is not False
            or set(report["anchor_contrasts"]) != set(CONTRAST_KEYS)
            or set(report["reference_anchors"]) != {"joint", "lmp"}
            or any(anchor["refitted"] is not False for anchor in report["reference_anchors"].values())):
        raise ValueError("Complete sixteen-session panel with both preserved anchor families required")
    panels = {panel["recipe"]: panel for panel in report["panels"]}
    if len(report["panels"]) != 2 or set(panels) != set(RECIPES):
        raise ValueError("Both complete all-band recipes required")
    expected_screens = {*CONTRAST_KEYS, "added_training_alignment"}
    if (set(report["complementarity_screens"]) != expected_screens
            or any(value is not False for value in report["complementarity_screens"].values())):
        raise ValueError("This completed-result framing requires all five complementarity screens to fail")

    candidate, control = (panels[recipe] for recipe in RECIPES)
    own_contrast = report["training_added_label_control"]
    a_by_id = {row["session_id"]: row for row in candidate["sessions"]}
    c_by_id = {row["session_id"]: row for row in control["sessions"]}
    validate_rows(own_contrast["sessions"])
    for row in own_contrast["sessions"]:
        for metric in METRICS:
            expected = a_by_id[row["session_id"]][metric] - c_by_id[row["session_id"]][metric]
            if abs(row[metric] - expected) > 1e-12:
                raise ValueError("Own-control contrasts must equal paired session score differences")
    absolute = [panel_description(candidate, "A: all-band LMP and power", source),
                panel_description(control, "C: mean of eight added-block training-label controls", source)]
    contrast_specs = [(own_contrast, "A minus C: added-block training-label control")]
    contrast_specs.extend((report["anchor_contrasts"][key], name) for key, name in zip(CONTRAST_KEYS, (
        "A minus independently selected J", "A minus J at A-selected lambda",
        "A minus independently selected LMP", "A minus LMP at A-selected lambda",
    )))
    contrasts = [panel_description(panel, name, source) for panel, name in contrast_specs]
    primary, identity = (absolute[0][1][metric]["Overall"][0] for metric in METRICS)
    if primary >= .01 or identity >= .005:
        raise ValueError("This completed-result framing requires both original practical gates to fail")
    stem = root / "figures/primary_all_band_attempt001"
    if any(stem.with_suffix("." + extension).exists() for extension in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing completed-result figures")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.4), gridspec_kw={"height_ratios": (1., 1.15)})
    contrast_labels = ("A − C\nAdded-label control", "A − J\nOwn-selected λ",
                       "A − J\nA-selected λ", "A − LMP\nOwn-selected λ",
                       "A − LMP\nA-selected λ")
    for column, (metric, margin, title) in enumerate(zip(METRICS, (.01, .005), (
            "Primary: improvement over direction/time baseline",
            "Identity: correct versus wrong held-out trial pairing"))):
        ax = axes[0, column]
        draw_groups(ax, absolute, metric, ("A\nAll-band LMP and power", "C\nMean of 8 added-label controls"))
        ax.axhline(margin * 1000, color="#9d2727", ls="--", lw=1)
        ax.text(.98, margin * 1000, f" Equal-animal practical margin {margin:g}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color="#9d2727")
        ax.set_title(f"{'AB'[column]}  {title}", loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Increment in joint native-XY R² (×10⁻³)")
        ax = axes[1, column]
        draw_groups(ax, contrasts, metric, contrast_labels)
        ax.set_title(f"{'CD'[column]}  {'Primary' if column == 0 else 'Identity'} contrasts: all-band minus comparator",
                     loc="left", fontsize=10.5, pad=10)
        ax.set_ylabel("Difference in increment (×10⁻³ R²)")

    fig.suptitle("All-band prediction misses practical margins and complementarity checks",
                 x=.065, ha="left", y=.981, fontsize=17)
    fig.text(.065, .945,
             f"A: primary {primary:.6f} < 0.010; identity {identity:.6f} < 0.005. Both practical gates and all 5 complementarity screens FAIL.",
             fontsize=11.5, weight="bold", color="#9d2727")
    fig.legend(handles=[Line2D([], [], color=COLORS[animal], marker="D", lw=0, label=animal + " mean")
                        for animal in ANIMAL_ORDER]
               + [Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.060, .097), ncol=3, frameon=False, fontsize=10)
    fig.text(.065, .026,
             "Small dots: all 16 sessions per group. Bars: fixed-prediction session-omission ranges; not confidence intervals. All axes retain signed values.\n"
             "J = LMP + 100–400 Hz; A adds all six remaining stored power bands. C permutes only the added training block within direction; evaluation stays intact.\n"
             "A selects λ = 10 in all 16 sessions; C uses the same λ. Own-selected anchors keep their original λ; matched anchors use A-selected λ.\n"
             "A/C share feature width, trace and λ, but their covariance spectra and effective degrees of freedom differ. Exposed development data · 2 animals · no promotion.",
             fontsize=9, color=".3")
    fig.tight_layout(rect=(.025, .17, .995, .915), h_pad=2.8, w_pad=2.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    exported = [row for group in (*absolute, *contrasts) for row in group[2]]
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps({"figure": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")),
                      "source_data": str(stem.with_suffix(".csv")), "rows": len(exported),
                      "source_summary": str(source)}, allow_nan=False))


if __name__ == "__main__":
    main()
