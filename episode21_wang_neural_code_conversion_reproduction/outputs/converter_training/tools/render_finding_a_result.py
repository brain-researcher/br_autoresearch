"""Render the completed amended Finding A report, without reopening payloads.

Run on compute only, after the existing finalizer succeeds. This separate
reporting step reads two JSON files; it never imports evaluation code, reads
HDF/model/neural/truth files, recomputes an endpoint or draws new resamples.
Undefined estimates and unavailable intervals remain visibly unavailable.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket


EPISODE = Path(__file__).resolve().parents[3]
RUN = EPISODE / "outputs/runs/finding_a_feature_decoding/amended_full001"
FIGURES = EPISODE / "outputs/figures/finding_a_amended_full001"
POLICY = "exact_constant_brain_comparator_all20_fresh_fits"
PEOPLE = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
LAYERS = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2",
          "conv3_3", "conv3_4", "conv4_1", "conv4_2", "conv4_3", "conv4_4",
          "conv5_1", "conv5_2", "conv5_3", "conv5_4", "fc6", "fc7", "fc8"]
METRICS = ("pattern", "profile")
CONTRAST = "content_minus_brain"
METHODS = (("within_source", "Within source", "#666666"),
           ("content_loss", "Content loss", "#0072B2"),
           ("brain_loss", "Amended brain loss", "#D55E00"))


def read_json(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Report exceeds bounded read: {path.name}")
    return json.loads(raw)


def load_completed_report():
    # Operational completion is checked before scientific results are read.
    phase = read_json(RUN / "finalize/report.json", 1024 * 1024)
    scientific = RUN / "finding_a_collection.finding_a_report.json"
    if not (phase.get("phase") == "finalize"
            and phase.get("status") == "complete_finalize"
            and phase.get("artifacts_published") is True
            and phase.get("model_policy") == POLICY
            and phase.get("decoded_slots") == 855
            and phase.get("unavailable_slots") == 0
            and str(scientific) in (phase.get("published_paths") or [])):
        raise ValueError("Successful complete amended finalizer required before result access")
    report = read_json(scientific, 64 * 1024 * 1024)
    if not (report.get("status") == "complete_requested_record"
            and report.get("finding") == "A" and report.get("model_policy") == POLICY
            and report["primary"]["point"]["participants"] == PEOPLE
            and report["primary"]["point"]["layers"] == LAYERS
            and len(report["primary"]["point"]["image_ids"]) == 224
            and report["comparability"]["status"] == "not_evaluable"
            and report["comparability"]["margin"] is None):
        raise ValueError("Completed fixed Finding A report required")
    return report, phase, scientific


def number(value):
    return "not evaluable" if value is None else f"{value:+.3f}"


def interval_text(interval):
    if not interval["available"]:
        return "unavailable"
    return f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]"


def summary_lines(report):
    primary = report["primary"]
    values = [primary["point"][metric]["group_means"][CONTRAST] for metric in METRICS]
    image = [interval_text(primary[metric]["group"][CONTRAST]) for metric in METRICS]
    dyadic = [interval_text(report["secondary_dyadic"][metric]["group_intervals"][CONTRAST])
              for metric in METRICS]
    return [
        f"Content minus amended brain loss: pattern Δr={number(values[0])}; "
        f"profile Δr={number(values[1])}, averaging 19 layers and five source participants.",
        f"95% image intervals (pattern/profile): {image[0]} / {image[1]}; "
        f"secondary dyadic intervals: {dyadic[0]} / {dyadic[1]}.",
        "Adapted held-out evaluation of 224 images in five fixed people; "
        "no population or comparability claim. Same-account separation and earlier tau-test processing are disclosed.",
    ]


def render(report, phase, scientific):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def array(value, shape):
        result = np.asarray(value, dtype=float)
        if result.shape != shape or np.isinf(result).any():
            raise ValueError("Unexpected result dimensions or infinite value")
        return result

    def bounds(value, shape):
        lo, hi = array(value["lower"], shape), array(value["upper"], shape)
        valid = np.asarray(value["available"], dtype=bool)
        draws = np.asarray(value["valid_draws"])
        if valid.shape != shape or draws.shape != shape:
            raise ValueError("Interval dimensions do not match the estimate")
        if np.any(valid & (~np.isfinite(lo) | ~np.isfinite(hi) | (lo > hi) | (draws != 1000))):
            raise ValueError("An available interval lacks complete finite bounds")
        return lo, hi, valid

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(18, 11),
                             gridspec_kw={"width_ratios": [1.1, 1.65, 1]})
    matrices = [array(report["primary"]["point"][m]["dyad_means"][CONTRAST], (5, 5))
                for m in METRICS]
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    limit = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 0.001)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ededed")

    for row, metric in enumerate(METRICS):
        point = report["primary"]["point"][metric]
        uncertainty = report["primary"][metric]
        ax = axes[row, 0]
        source = array(point["source_means"][CONTRAST], (5,))
        lo, hi, valid = bounds(uncertainty["source"][CONTRAST], (5,))
        for i, value in enumerate(source):
            if math.isfinite(value):
                ax.plot(value, i, "o", color="#0072B2")
                if valid[i]:
                    ax.hlines(i, lo[i], hi[i], color="#0072B2", linewidth=1.8)
                else:
                    ax.annotate("CI unavailable", (value, i), xytext=(5, 7),
                                textcoords="offset points", fontsize=8)
            else:
                ax.text(0, i, "not evaluable", color="#555555", fontsize=8, va="center")
        group = point["group_means"][CONTRAST]
        secondary = report["secondary_dyadic"][metric]
        for y, estimate, interval, color, marker in (
                (6, group, uncertainty["group"][CONTRAST], "#222222", "D"),
                (7, secondary["group_means"][CONTRAST], secondary["group_intervals"][CONTRAST], "#8E5AA6", "s")):
            lower, upper, available = bounds(interval, ())
            if estimate is not None:
                ax.plot(estimate, y, marker, color=color)
                if bool(available):
                    ax.hlines(y, float(lower), float(upper), color=color, linewidth=2)
                else:
                    ax.annotate("CI unavailable", (estimate, y), xytext=(5, 7),
                                textcoords="offset points", fontsize=8)
            else:
                ax.text(0, y, "not evaluable", fontsize=8, va="center")
        ax.axvline(0, color="#aaaaaa", linestyle="--", linewidth=1)
        ax.set_yticks([0, 1, 2, 3, 4, 6, 7], PEOPLE + ["Five-source mean", "Secondary dyadic"])
        ax.set_ylim(7.8, -0.8)
        ax.set_xlabel("Content − amended brain loss (Δ correlation)")
        ax.set_title(f"{metric.capitalize()} · source-level effects", loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=.15)

        ax = axes[row, 1]
        for method, label, color in METHODS:
            estimates = array(point["layer_group_means"][method], (19,))
            lower, upper, available = bounds(uncertainty["layer_group"][method], (19,))
            ax.plot(range(19), estimates, ".-", color=color, label=label, linewidth=1.3)
            ax.fill_between(range(19), lower, upper, where=available,
                            color=color, alpha=.13, linewidth=0)
            missing_interval = np.isfinite(estimates) & ~available
            ax.scatter(np.flatnonzero(missing_interval), estimates[missing_interval],
                       marker="x", color=color, s=25)
            for i in np.flatnonzero(~np.isfinite(estimates)):
                ax.text(i, .02, "NA", rotation=90, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform(), fontsize=7, color=color)
        ax.set_xticks(range(19), LAYERS, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(f"{metric.capitalize()} correlation")
        ax.set_title("All 19 layers · equal source weights", loc="left", fontweight="bold")
        ax.axhline(0, color="#aaaaaa", linewidth=.7)
        ax.grid(axis="y", alpha=.15)
        if row == 0:
            ax.legend(frameon=False, fontsize=8, ncol=1)

        ax = axes[row, 2]
        matrix = matrices[row]
        if np.isfinite(np.diag(matrix)).any():
            raise ValueError("Self-pair diagonal must remain inapplicable")
        heatmap = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit)
        for source_index in range(5):
            for target_index in range(5):
                if source_index == target_index:
                    continue
                value = matrix[source_index, target_index]
                ax.text(target_index, source_index, f"{value:+.2f}" if np.isfinite(value) else "NA",
                        ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(value) and abs(value) > .65 * limit else "#222222")
        ax.set_xticks(range(5), PEOPLE, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(5), PEOPLE, fontsize=8)
        ax.set_xlabel("Target participant")
        ax.set_ylabel("Source participant")
        ax.set_title("All 20 directed contrasts", loc="left", fontweight="bold")
        fig.colorbar(heatmap, ax=ax, fraction=.05, pad=.04, label="Δ correlation")

    fig.suptitle("EP21 · Held-out VGG19 feature prediction\n"
                 "224 images · 5 fixed participants · 20 directed pairs · 19 layers", fontsize=17, y=.98)
    short_caption = (
        "Points average all 19 layers; each source weights its four outgoing targets equally. The primary mean gives all five sources equal weight.\n"
        "Bars/bands: 95% pointwise intervals from 1,000 synchronized image draws, conditional on these people and fitted objects; purple: secondary dyadic resampling.\n"
        "No population inference or comparability margin. Adapted constant-coordinate brain comparator; same-account procedural separation; prior tau-test processing disclosed.\n"
        "Blank diagonal: inapplicable self-pairs. NA: unavailable estimate; ×: unavailable layer interval. The 20 dyads are dependent repeated-target observations."
    )
    fig.text(.06, .035, short_caption, ha="left", va="bottom", fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=.105, right=.965, bottom=.22, top=.88, wspace=.43, hspace=.63)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "finding_a_result"
    for extension in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = summary_lines(report)
    (FIGURES / "summary.txt").write_text("\n".join(lines) + "\n")
    caption = short_caption + (
        "\n\nPattern correlation is across feature units per image; profile correlation is across images per feature unit. "
        "Contrasts use the already qualified paired-support result tables, not subtraction of unmatched absolute means. "
        "All methods and layers are displayed without outcome selection. Both image and dyadic intervals use the existing "
        "1,000-draw, 2.5/97.5 percentile rules; secondary dyadic resampling cannot replace source-level inference. "
        "No simultaneous interval, hypothesis-test, equivalence or non-inferiority conclusion is made. "
        "The canceled original readout previously processed tau-test data; no final-score exposure was reported. "
        "This adapted evaluation is not a fresh blind confirmation, and regular-image evidence remains correlated with EP04. "
        "No B/C, OOD, external-site or reconstruction result is implied."
        f"\n\nScientific report: {scientific}\nFinalizer job: {phase['allocation_identity']['job_id']}"
        f"\nReport disposition: {report['quantitative_finding']}"
        "\nThis renderer reads completed JSON reports only and performs no new statistical analysis.\n"
    )
    (FIGURES / "caption.txt").write_text(caption)
    print(stem.with_suffix(".png"))
    print("\n".join(lines))


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Scientific rendering requires an actual compute-node Slurm shell")
    report, phase, scientific = load_completed_report()
    render(report, phase, scientific)


if __name__ == "__main__":
    main()
