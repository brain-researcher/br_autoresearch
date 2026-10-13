#!/usr/bin/env python3
"""Plot the completed EP08 mapping comparison; no source arrays or new fits."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAPPING = "mapping_design_value"
COMPARATORS = ("balanced", "mean_value")
ANIMALS = ("Chewie", "Mihili")
ORDERS = ("original", "0", "1", "2")
CHECKPOINTS = (32, 48, 64, 80, 96)
ELECTRODES = (8, 16, 32, 96)
COLORS = {"balanced": "#286d91", "mean_value": "#b96836"}
LABELS = {"balanced": "Mapping − balance", "mean_value": "Mapping − mean-value"}


def contrast(panel, comparator):
    return float(panel["mean_delta_Q_common_from_"+comparator])


def session_profile(session, comparator):
    cells = session["panels"][MAPPING]["cells"]
    key = "delta_Q_common_from_"+comparator
    return np.array([np.mean([cell[key] for cell in cells if cell["training_trials"] == n])
                     for n in CHECKPOINTS], dtype=float)


def validate_report(report):
    if (report["status"] != "completed_mapping_design_variance_comparison"
            or report["held_session_accessed"] is not False
            or tuple(report["orders"]) != ORDERS):
        raise ValueError("Complete development-only four-order mapping comparison required")
    sessions = sorted(report["sessions"], key=lambda row: (row["animal"], row["filename"]))
    if (len(sessions) != 8 or len({s["filename"] for s in sessions}) != 8
            or any(sum(s["animal"] == animal for s in sessions) != 4 for animal in ANIMALS)):
        raise ValueError("All eight development sessions and both four-session animals required")
    expected = {(n, e) for n in CHECKPOINTS for e in ELECTRODES}
    for session in sessions:
        if tuple(o["order"] for o in session["orders"]) != ORDERS:
            raise ValueError("Every session must retain all four fixed orders")
        for panels in [session["panels"], *(o["panels"] for o in session["orders"])]:
            for policy in (MAPPING, *COMPARATORS):
                cells = panels[policy]["cells"]
                if len(cells) != 20 or {(c["training_trials"], c["electrodes"]) for c in cells} != expected:
                    raise ValueError("Every policy must retain all twenty resource cells")
            mapping = {(c["training_trials"], c["electrodes"]): c for c in panels[MAPPING]["cells"]}
            for comparator in COMPARATORS:
                key = "delta_Q_common_from_"+comparator
                for cell in panels[comparator]["cells"]:
                    candidate = mapping[cell["training_trials"], cell["electrodes"]]
                    delta = candidate["Q_common"]-cell["Q_common"]
                    if not np.isfinite(delta) or abs(delta-candidate[key]) > 1e-12:
                        raise ArithmeticError("Stored paired contrast differs from its matched Q values")
                values = [cell[key] for cell in mapping.values()]
                if abs(float(np.mean(values))-contrast(panels[MAPPING], comparator)) > 1e-12:
                    raise ArithmeticError("Stored session/order summary differs from complete cell mean")
    if tuple(o["order"] for o in report["by_order"]) != ORDERS:
        raise ValueError("Complete recorded order aggregation required")
    overall = next(panel for panel in report["panels"] if panel["policy"] == MAPPING)
    for comparator in COMPARATORS:
        expected_mean = np.mean([contrast(s["panels"][MAPPING], comparator) for s in sessions])
        if abs(expected_mean-contrast(overall, comparator)) > 1e-12:
            raise ArithmeticError("Equal-session/equal-animal summary replay differs")
    return sessions, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Render inside the coordinator's verified compute allocation")
    outputs = {kind: args.output_stem.with_suffix("."+kind) for kind in ("png", "pdf", "csv", "json")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Preserve prior figure and provenance artifacts")
    raw = args.report.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    del raw
    sessions, overall = validate_report(report)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), gridspec_kw={"height_ratios": [1.08, 1]})
    rows = []
    session_ax, order_ax = axes[0]
    for index, session in enumerate(sessions):
        values = [contrast(session["panels"][MAPPING], comparator) for comparator in COMPARATORS]
        session_ax.plot(values, [index, index], color=".76", linewidth=1, zorder=1)
        for comparator, value in zip(COMPARATORS, values):
            session_ax.scatter(value, index, color=COLORS[comparator], s=34, zorder=2)
            rows.append({"panel": "session", "animal": session["animal"], "session": session["filename"],
                         "order": "all_four", "training_trials": "all_five", "comparator": comparator,
                         "delta_Q_common": value})
    session_ax.set_yticks(range(8), [s["filename"].replace("_CO_", " ").replace(".mat", "") for s in sessions])
    session_ax.invert_yaxis()
    session_ax.axhline(3.5, color=".86", linewidth=.8)
    session_ax.set_title("A  Every session · equal mean over 20 cells and four orders", loc="left", fontsize=11)
    for index, order in enumerate(report["by_order"]):
        panel = next(p for p in order["panels"] if p["policy"] == MAPPING)
        values = [contrast(panel, comparator) for comparator in COMPARATORS]
        order_ax.plot(values, [index, index], color=".76", linewidth=1, zorder=1)
        for comparator, value in zip(COMPARATORS, values):
            order_ax.scatter(value, index, color=COLORS[comparator], s=45, zorder=2)
            rows.append({"panel": "order", "animal": "both_equal", "session": "all_eight",
                         "order": order["order"], "training_trials": "all_five", "comparator": comparator,
                         "delta_Q_common": value})
    order_ax.set_yticks(range(4), ("Original", "Seed 0", "Seed 1", "Seed 2"))
    order_ax.invert_yaxis()
    order_ax.set_title("B  Every order · equal mean over cells, sessions and animals", loc="left", fontsize=11)
    for ax in axes[0]:
        ax.axvline(0, color=".4", linewidth=.8)
        ax.set_xlabel("Paired ΔQ_common")
        ax.grid(axis="x", alpha=.15)
    profiles = []
    for ax, animal, letter in zip(axes[1], ANIMALS, ("C", "D")):
        people = [s for s in sessions if s["animal"] == animal]
        for comparator in COMPARATORS:
            values = np.array([session_profile(s, comparator) for s in people])
            profiles.extend(values.ravel().tolist())
            for session, trace in zip(people, values):
                ax.plot(CHECKPOINTS, trace, color=COLORS[comparator], linewidth=.8, alpha=.22)
                for n, value in zip(CHECKPOINTS, trace):
                    rows.append({"panel": "session_checkpoint", "animal": animal, "session": session["filename"],
                                 "order": "all_four", "training_trials": n, "comparator": comparator,
                                 "delta_Q_common": float(value)})
            means = values.mean(axis=0)
            ax.plot(CHECKPOINTS, means, color=COLORS[comparator], marker="o", linewidth=2,
                    label=LABELS[comparator])
            for n, value in zip(CHECKPOINTS, means):
                rows.append({"panel": "animal_checkpoint", "animal": animal, "session": "all_four",
                             "order": "all_four", "training_trials": n, "comparator": comparator,
                             "delta_Q_common": float(value)})
        ax.axhline(0, color=".4", linewidth=.8)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlabel("Acquired fitting trials")
        ax.set_ylabel("Paired ΔQ_common")
        ax.set_title(f"{letter}  {animal} · four-session mean and individual session traces", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=.15)
    low, high = min(profiles+[0]), max(profiles+[0])
    margin = max(.0005, .07*(high-low))
    for ax in axes[1]:
        ax.set_ylim(low-margin, high+margin)
    handles, names = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(.18, .938), frameon=False, ncol=2)
    fig.suptitle("EP08 · Acquired-feature mapping allocation falls behind both comparators",
                 x=.18, y=.984, ha="left", fontsize=16)
    text = " · ".join(f"Mapping − {label}: {contrast(overall, comparator):+.6f} Q"
                      for comparator, label in (("balanced", "balance"), ("mean_value", "mean-value")))
    fig.text(.18, .947, text+" · development only", fontsize=10)
    fig.text(.18, .083, "All 20 cells retained: 8/16/32/96 electrodes × 32/48/64/80/96 trials. Fixed scarce-electrode repetitions are averaged first.", fontsize=9)
    fig.text(.18, .058, "Animal profiles average electrodes and all four orders. Initial32/final96 fits are shared; their contrasts remain zero. No confidence intervals.", fontsize=9)
    fig.text(.18, .033, "Common initializer normalization and equal acquired observations at each E,n. This tests one heuristic; four held sessions remain closed.", fontsize=9)
    fig.subplots_adjust(left=.18, right=.985, bottom=.16, top=.875, hspace=.48, wspace=.31)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for kind in ("png", "pdf"):
        fig.savefig(outputs[kind], dpi=180, facecolor="white")
    plt.close(fig)
    with outputs["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"source_report": str(args.report.resolve()), "source_sha256": digest,
                  "source_status": report["status"], "source_observed_utc": report["observed_utc"],
                  "plot_script": str(Path(__file__).resolve()), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "hostname": socket.gethostname(), "new_fits": False, "new_source_reads": False,
                  "held_sessions_accessed": False, "new_valid_evaluations": 0,
                  "plotted_rows": len(rows), "outputs": {key: str(path) for key, path in outputs.items()},
                  "weighting": "scarce subsets first, equal electrodes/checkpoints/orders/sessions within animal, equal animals",
                  "session_and_order_summary_replay_tolerance": 1e-12,
                  "source_endpoint_Q_replay_max_absolute_difference": report["endpoint_replay_max_absolute_difference"]}
    outputs["json"].write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
