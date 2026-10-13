#!/usr/bin/env python3
"""Present the frozen observed descriptions; never read event or fit payloads."""
import argparse
import json
import math
import os
from pathlib import Path
import socket

EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs/exploratory_observed_behavior"
CATEGORIES = ("strong_accept", "weak_accept", "weak_reject", "strong_reject", "no_response")
LABELS = ("Strong accept", "Weak accept", "Weak reject", "Strong reject", "Miss")
COLORS = ("#2563eb", "#93c5fd", "#fdba74", "#c2410c", "#64748b")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptive-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render in an actual Slurm compute shell")
    source = args.descriptive_report.resolve()
    if RUNS not in source.parents or source.name != "observed_behavior_descriptive.json":
        raise ValueError("Use the frozen observed descriptive report in its declared run tree")
    outputs = [args.output_dir.resolve() / f"observed_behavior.{suffix}" for suffix in ("png", "svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve existing observed-behavior figures")
    report = json.loads(source.read_text())
    if (report.get("analysis_id") != "exploratory_observed_behavior"
            or report.get("phase") != "observed_behavior_describe"
            or report.get("status") != "descriptions_completed"
            or report.get("source_validated") is not True
            or report.get("participants_count") != 108
            or report.get("version_counts") != {"EI": 54, "ER": 54}
            or report.get("source_trials") != 27648):
        raise ValueError("Require the completed 108-person observed description")
    descriptions = report["descriptions"]
    if descriptions.get("rt_quantile_method") != "linear":
        raise ValueError("The saved empirical quantile convention changed")
    panels = descriptions["by_version"]
    for version in ("EI", "ER"):
        row = panels[version]
        counts = row["category_counts"]
        if (row["trials"] != 13824 or set(counts) != set(CATEGORIES)
                or any(type(value) is not int or value < 0 for value in counts.values())
                or sum(counts.values()) != row["trials"]):
            raise ValueError("Incomplete version-specific response counts")
        for choice, categories in (("accept", CATEGORIES[:2]), ("reject", CATEGORIES[2:4])):
            rt = row["responded_rt_by_choice"][choice]
            n = sum(counts[key] for key in categories)
            quantiles = rt["seconds_q10_q50_q90"]
            if rt["responded_trials"] != n:
                raise ValueError("RT response count disagrees with the saved categories")
            if n == 0:
                if quantiles is not None:
                    raise ValueError("An empty response stratum cannot have RT quantiles")
            elif (not isinstance(quantiles, list) or len(quantiles) != 3
                  or any(not math.isfinite(q) or not 0 < q <= 4 for q in quantiles)
                  or quantiles != sorted(quantiles)):
                raise ValueError("Invalid saved responded RT quantiles")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, (responses, rt_axis) = plt.subplots(1, 2, figsize=(11.8, 5.8),
                                                gridspec_kw={"width_ratios": [1, 1.35]})
        fig.subplots_adjust(left=.075, right=.97, top=.73, bottom=.28, wspace=.42)
        bottom = [0., 0.]
        for category, label, color in zip(CATEGORIES, LABELS, COLORS):
            heights = [panels[v]["category_counts"][category]/panels[v]["trials"] for v in ("EI", "ER")]
            responses.bar([0, 1], heights, bottom=bottom, width=.58, label=label,
                          color=color, edgecolor="white", linewidth=.6)
            bottom = [a+b for a, b in zip(bottom, heights)]
        responses.set_xticks([0, 1], ["EI\n54 participants", "ER\n54 participants"])
        responses.set_ylim(0, 1)
        responses.yaxis.set_major_formatter(PercentFormatter(1))
        responses.set_ylabel("Proportion of all trials, including misses")
        responses.set_title("Recorded response categories", loc="left", fontsize=12)
        responses.legend(frameon=False, fontsize=9, ncol=3, loc="upper left",
                         bbox_to_anchor=(-.02, -.25), columnspacing=1.1, handlelength=1.2)

        rt_labels = []
        for y, (version, choice) in enumerate((("EI", "accept"), ("EI", "reject"),
                                                ("ER", "accept"), ("ER", "reject"))):
            rt = panels[version]["responded_rt_by_choice"][choice]
            rt_labels.append(f"{version} · {choice}")
            color = "#2563eb" if choice == "accept" else "#c2410c"
            if rt["seconds_q10_q50_q90"] is None:
                rt_axis.text(.05, y, "No responses", va="center", color="#64748b")
            else:
                q10, q50, q90 = rt["seconds_q10_q50_q90"]
                rt_axis.errorbar(q50, y, xerr=[[q50-q10], [q90-q50]], fmt="o",
                                 color=color, capsize=5, markersize=6, linewidth=1.8)
        rt_axis.set_yticks(range(4), rt_labels)
        rt_axis.set_ylim(3.6, -.6)
        rt_axis.set_xlim(0, 4.1)
        rt_axis.set_xlabel("Responded RT (seconds)")
        rt_axis.set_title("Empirical RT quantiles", loc="left", fontsize=12)
        rt_axis.axvline(4, color="#cbd5e1", linestyle=":", linewidth=1)
        fig.suptitle("Observed behavior · exploratory", y=.96, fontsize=17, fontweight="bold")
        fig.text(.5, .88, "108 participants · 54 EI / 54 ER · 27,648 trials", ha="center", fontsize=12)
        fig.text(.5, .82, "Descriptive evidence only; no psychological mechanism claim", ha="center", color="#475569")
        fig.text(.51, .16, "RT points: medians. Whiskers: 10th–90th percentiles.", fontsize=10)
        fig.text(.51, .11, "Empirical quantiles, not confidence intervals; misses have no RT.", fontsize=9, color="#475569")
        fig.text(.075, .035, "EI and ER are task versions. Strong/weak labels are recorded categories, not independent confidence.",
                 fontsize=9, color="#475569")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
