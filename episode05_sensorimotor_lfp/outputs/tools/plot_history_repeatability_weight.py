#!/usr/bin/env python3
"""Render the complete H, W and HW summaries without source access or new fits."""
import csv
import json
from pathlib import Path

from plot_repeatability_weight import ANIMAL_ORDER, COLORS, METRICS, description_rows, group_means, validate_rows
from source_adapter import compute_only


def main():
    compute_only()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parents[1]
    specifications = (
        ("H", "History4", "primary_history_control_summary_attempt001/primary_history_control.json",
         "complete_two_history_control_comparisons", "lmp_history4_ridge_v1"),
        ("W", "Current + repeatability", "primary_repeatability_weight_summary_attempt001/primary_repeatability_weight.json",
         "complete_repeatability_weight_control_panel", "lmp_repeatability_weight_ridge_v1"),
        ("HW", "History4 + repeatability", "primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json",
         "complete_history_repeatability_weight_control_panel", "lmp_history4_repeatability_weight_ridge_v1"),
    )
    groups, summaries, exported, sources = [], [], [], []
    for short, name, relative, status, recipe in specifications:
        source = root/"runs/adaptive_search"/relative
        report = json.loads(source.read_text())
        candidates = [p for p in report["panels"] if p["recipe"] == recipe]
        if report.get("status") != status or len(candidates) != 1:
            raise ValueError("Each source must be its complete preserved sixteen-session panel")
        candidate = candidates[0]
        validate_rows(candidate["sessions"])
        for metric, summary_key in zip(METRICS, ("primary_increment", "identity_increment")):
            if abs(group_means(candidate["sessions"], metric)["Overall"]-candidate[summary_key]) > 1e-12:
                raise ValueError("Reported and recomputed equal-animal means differ")
        rows, summary = description_rows(candidate["sessions"], short+": "+name)
        exported.extend({**row, "source_summary": str(source), "recipe": recipe} for row in rows)
        summaries.append(summary)
        groups.append(candidate["sessions"])
        sources.append(str(source))
    stem = root/"figures/primary_history_repeatability_weight_attempt002"
    if any(stem.with_suffix("."+ext).exists() for ext in ("png", "pdf", "csv")):
        raise FileExistsError("Preserve existing completed-result figures")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6))
    offsets = {"Mihili": -.17, "Chewie": .17, "Overall": 0.}
    labels = [short+"\n"+name.replace(" + ", " +\n") for short, name, *_ in specifications]
    for ax, metric, margin, title in zip(axes, METRICS, (.01, .005),
            ("A  Primary: improvement over direction/time baseline", "B  Identity: correct versus wrong trial pairing")):
        for position, (rows, summary) in enumerate(zip(groups, summaries)):
            for animal in ANIMAL_ORDER:
                selected = [r for r in rows if r["animal"] == animal]
                for index, row in enumerate(selected):
                    spread = .12*(index/(len(selected)-1)-.5)
                    ax.scatter(position+offsets[animal]+spread, row[metric], s=27,
                               color=COLORS[animal], alpha=.48, linewidth=.4,
                               edgecolor="white", zorder=3)
            for animal in (*ANIMAL_ORDER, "Overall"):
                center, low, high = summary[metric][animal]
                location = position+offsets[animal]
                ax.plot([location, location], [low, high], color=COLORS[animal], lw=2, zorder=4)
                for endpoint in (low, high):
                    ax.plot([location-.024, location+.024], [endpoint, endpoint],
                            color=COLORS[animal], lw=1.2, zorder=4)
                ax.scatter(location, center, marker="s" if animal == "Overall" else "D",
                           s=49, color=COLORS[animal], edgecolor="white", linewidth=.6, zorder=5)
        ax.axhline(0, color=".55", lw=.8)
        ax.axhline(margin, color="#9d2727", ls="--", lw=1)
        ax.text(.98, margin, f" Equal-animal practical margin {margin:g}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color="#9d2727")
        ax.set(title=title, ylabel="Increment in joint native-XY R²", xticks=range(3),
               xticklabels=labels, xlim=(-.48, 2.48))
        ax.title.set_fontsize(10.4)
        ax.grid(axis="y", color=".93", lw=.6)
        ax.set_axisbelow(True)
        ax.margins(y=.10)

    hw = summaries[2]
    primary, identity = hw["delta_R2"]["Overall"][0], hw["identity_increment"]["Overall"][0]
    primary_status, identity_status = ("PASS" if primary >= .01 else "FAIL"), ("PASS" if identity >= .005 else "FAIL")
    fig.suptitle("Combining fixed history and repeatability weighting", x=.075, ha="left", y=.98, fontsize=17)
    fig.text(.075, .928,
             f"HW: primary {primary:.6f} — {primary_status} at 0.010; identity {identity:.6f} — {identity_status} at 0.005",
             fontsize=12, weight="bold", color="#9d2727")
    fig.legend(handles=[Line2D([], [], color=COLORS[a], marker="D", lw=0, label=a+" mean") for a in ANIMAL_ORDER]
               +[Line2D([], [], color=COLORS["Overall"], marker="s", lw=0, label="Equal-animal mean")],
               loc="lower left", bbox_to_anchor=(.067, .107), ncol=3, frameon=False, fontsize=10)
    fig.text(.075, .037,
             "Small dots: all 16 sessions for every recipe. Bars: ranges after fixed-prediction session omissions; not confidence intervals.\n"
             "Practical thresholds apply to equal-animal means; animal-specific gates require positivity. Each recipe uses its own outer-selected λ.\n"
             "Exposed development data · 2 animals · no promotion or future-session confirmation.", fontsize=9, color=".3")
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
        "source_data": str(stem.with_suffix(".csv")), "rows": len(exported), "source_summaries": sources}, allow_nan=False))


if __name__ == "__main__":
    main()
