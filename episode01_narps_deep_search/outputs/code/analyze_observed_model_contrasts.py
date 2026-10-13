#!/usr/bin/env python3
"""Post-hoc paired comparisons of saved observed-data predictions, without refits.

The full 15-pair menu is fixed before this analysis. All six existing models,
both original score directions, and all 108 participants remain represented.
Participant SEs and leave-one-person-out mean ranges are descriptive only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics

EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs/exploratory_observed_model_contrasts"
FIGURES = EPISODE / "outputs/figures"
SOURCE = EPISODE / "outputs/runs/exploratory_observed_behavior/20261005-closeout-46651952/observed_behavior_summary.json"
MODELS = ("M0", "MV", "MZ", "MC", "MVZ", "MVC")
PAIRS = (
    ("MV", "M0"), ("MZ", "M0"), ("MC", "M0"), ("MVZ", "M0"), ("MVC", "M0"),
    ("MV", "MZ"), ("MV", "MC"), ("MVZ", "MV"), ("MVC", "MV"),
    ("MC", "MZ"), ("MVZ", "MZ"), ("MVC", "MZ"), ("MVZ", "MC"),
    ("MVC", "MC"), ("MVC", "MVZ"),
)
ADDITIONS = ("MVZ-MV", "MVC-MV", "MVZ-MZ", "MVC-MC")
DENOMINATOR = "256 trials per participant after summing both primary directions"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def moments(values):
    return {
        "participants": len(values),
        "mean_nats_per_trial": statistics.fmean(values),
        "participant_SE": statistics.stdev(values) / math.sqrt(len(values)),
        "participant_median": statistics.median(values),
    }


def contrast_moments(values):
    total = math.fsum(values)
    omitted = [(total - value) / (len(values) - 1) for value in values]
    return {
        **moments(values),
        "positive_participants": sum(value > 0 for value in values),
        "zero_participants": sum(value == 0 for value in values),
        "negative_participants": sum(value < 0 for value in values),
        "positive_fraction": sum(value > 0 for value in values) / len(values),
        "leave_one_participant_out_mean_range": [min(omitted), max(omitted)],
    }


def analyze(source):
    if (source.get("analysis_id") != "exploratory_observed_behavior"
            or source.get("denominator") != DENOMINATOR
            or set(source.get("models", {})) != set(MODELS)):
        raise ValueError("Require the unchanged six-model observed-behavior summary")
    participants, versions, model_values = None, None, {}
    result = {
        "analysis_id": "exploratory_observed_model_contrasts",
        "source": str(SOURCE),
        "source_created_utc": source.get("created_utc"),
        "evidence_status": "Post-hoc exploratory analysis after observed outcomes were exposed",
        "score": "Saved plug-in joint choice/RT/deadline log density, nat/trial; larger is better",
        "denominator": DENOMINATOR,
        "participant_weighting": "equal",
        "uncertainty": "Participant SE is descriptive, not a cluster interval or superiority gate",
        "leave_one_participant_out": "Omit each saved paired score once; no model refit or new held-out prediction",
        "task_versions": "EI/ER are between-participant task versions, not independent replications or causal contrasts",
        "interpretation": "Predictive differences do not identify psychological mechanisms",
        "all_pairs_reported": True,
        "fresh_fit_calls": 0,
        "new_trial_likelihood_calls": 0,
        "logistic_benchmark": source.get("logistic_benchmark"),
        "models": {},
        "contrasts": {},
    }
    for model in MODELS:
        record = source["models"][model]
        if record.get("available") is not True:
            raise ValueError("Every model must retain both completed directions")
        directions = record.get("directions", [])
        if len(directions) != 2 or any(d.get("complete") is not True for d in directions):
            raise ValueError("Every model must retain both completed directions")
        scores = record["participant_scores"]
        people = sorted(scores)
        groups = [scores[p]["version"] for p in people]
        if len(people) != 108 or groups.count("EI") != 54 or groups.count("ER") != 54:
            raise ValueError("Require all 108 participants, 54 EI and 54 ER")
        if participants is None:
            participants, versions = people, groups
        elif people != participants or groups != versions:
            raise ValueError("Participant/version identities differ across models")
        values = [float(scores[p]["nats_per_trial"]) for p in people]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Every participant score must be finite")
        model_values[model] = values
        result["models"][model] = {
            "summary": moments(values),
            "by_version": {v: moments([x for x, group in zip(values, versions) if group == v])
                           for v in ("EI", "ER")},
            "directions": directions,
            "participant_scores": scores,
        }
    for positive, reference in PAIRS:
        values = [a - b for a, b in zip(model_values[positive], model_values[reference])]
        result["contrasts"][positive + "-" + reference] = {
            "positive_model": positive,
            "reference_model": reference,
            "summary": contrast_moments(values),
            "by_version": {v: contrast_moments([x for x, group in zip(values, versions) if group == v])
                           for v in ("EI", "ER")},
            "participant_differences": {p: {"version": v, "nats_per_trial": x}
                                        for p, v, x in zip(participants, versions, values)},
        }
    return result


def markdown(report):
    lines = [
        "# Observed behavior: saved-score comparisons",
        "", report["evidence_status"] + ". No new fits or trial likelihoods.",
        "All scores retain 108 participants and both original directions; larger is better.",
        "", "| Model | Mean joint score (nat/trial) | Participant SE | EI mean | ER mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["models"].items():
        s, v = row["summary"], row["by_version"]
        lines.append(f"| {name} | {s['mean_nats_per_trial']:.6f} | {s['participant_SE']:.6f} | "
                     f"{v['EI']['mean_nats_per_trial']:.6f} | {v['ER']['mean_nats_per_trial']:.6f} |")
    lines += ["", "All 15 paired contrasts are reported; no pair is selected by its outcome.", "",
              "| Contrast | Mean gain | Paired SE | Median | Positive / 108 | EI mean | ER mean | Omit-one mean range |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, row in report["contrasts"].items():
        s, v = row["summary"], row["by_version"]
        low, high = s["leave_one_participant_out_mean_range"]
        lines.append(f"| {name} | {s['mean_nats_per_trial']:.6f} | {s['participant_SE']:.6f} | "
                     f"{s['participant_median']:.6f} | {s['positive_participants']} | "
                     f"{v['EI']['mean_nats_per_trial']:.6f} | {v['ER']['mean_nats_per_trial']:.6f} | "
                     f"[{low:.6f}, {high:.6f}] |")
    lines += ["", "Gain, SE, median and omit-one ranges are in nat/trial.",
              "Participant SEs are descriptive; no inferential or superiority threshold is applied.",
              "Omitting a participant changes only the summary, not the already-fitted models.",
              "EI/ER are task versions, not independent replications or causal contrasts.",
              "Predictive differences do not identify psychological mechanisms; logistic and other benchmarks remain absent.",
              "", f"Source: `{report['source']}`.", ""]
    return "\n".join(lines)


def plot(report, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
        fig.subplots_adjust(left=.09, right=.98, top=.75, bottom=.29, wspace=.38)
        panels = [("Total gain over symmetric reference", [m + "-M0" for m in MODELS[1:]]),
                  ("Additional gain within nested models", list(ADDITIONS))]
        for axis, (title, names) in zip(axes, panels):
            for index, name in enumerate(names):
                row = report["contrasts"][name]
                for group, offset, color, marker in ((None, -.18, "#1f2937", "o"),
                                                       ("EI", 0, "#2563eb", "s"),
                                                       ("ER", .18, "#c2410c", "^")):
                    summary = row["summary"] if group is None else row["by_version"][group]
                    axis.errorbar(summary["mean_nats_per_trial"], index + offset,
                                  xerr=summary["participant_SE"], fmt=marker, color=color,
                                  markersize=5, capsize=2, label=(group or "All") if index == 0 else None)
            axis.axvline(0, color="#94a3b8", linewidth=1)
            axis.set_yticks(range(len(names)), [name.replace("-", " − ") for name in names])
            axis.set_ylim(len(names) - .5, -.5)
            axis.set_title(title, loc="left", fontsize=12)
            axis.set_xlabel("Paired mean joint log-score gain (nat/trial)")
            axis.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -.14))
        fig.suptitle("EP01 · What do the existing models add?", fontsize=17, fontweight="bold", y=.96)
        fig.text(.5, .89, "108 participants · both original run-half directions · saved predictions only", ha="center")
        fig.text(.5, .83, "Post-hoc exploratory comparison; no psychological mechanism conclusion", ha="center", color="#475569")
        fig.text(.09, .14, "Points: equal-participant means. Whiskers: ±1 participant SE, descriptive only. Panels use different x scales.", fontsize=10)
        fig.text(.09, .095, "M0: symmetric reference. V: valuation asymmetry parameter. Z: starting-bias parameter. C: drift-criterion parameter.", fontsize=9)
        fig.text(.09, .047, "All 15 pairwise comparisons are retained in the report. EI/ER are task versions, not replications. No fits or resampling added.", fontsize=9, color="#475569")
        for suffix in ("png", "svg"):
            fig.savefig(prefix.with_suffix("." + suffix), dpi=180, facecolor="white")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Saved-score analysis requires an actual Slurm compute shell")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < 1 or any(os.environ.get(n) != "1" for n in THREAD_ENV):
        raise RuntimeError("Allocate at least one CPU and cap numerical threads at one")
    output, figure = args.output_dir.resolve(), args.figure_prefix.resolve()
    if RUNS not in output.parents or output.exists():
        raise FileExistsError("Use a fresh saved-score analysis directory")
    if figure.parent != FIGURES or figure.suffix or any(figure.with_suffix("." + s).exists() for s in ("png", "svg")):
        raise FileExistsError("Use a fresh figure prefix directly in outputs/figures")
    report = analyze(json.loads(SOURCE.read_text()))
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  hostname=socket.gethostname(), job_id=os.environ["SLURM_JOB_ID"])
    output.mkdir(parents=True)
    (output / "observed_model_contrasts.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table = markdown(report)
    (output / "observed_model_contrasts.md").write_text(table)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot(report, figure)
    print(table)
    print(json.dumps({"report": str(output / "observed_model_contrasts.json"), "figure_prefix": str(figure)}))


if __name__ == "__main__":
    main()
