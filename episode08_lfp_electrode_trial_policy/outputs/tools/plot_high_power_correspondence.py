#!/usr/bin/env python3
"""Plot three spectral pipelines from the completed embedded high-power report."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket

OUTPUTS = Path(__file__).resolve().parents[1]
POLICY = "initial_marginal_high_power_only_v1"
CONTROL = "initial_marginal_high_power_target_labels5_v1"
SPECTRA = (
    ("initial_marginal", "initial_marginal_target_labels5_v1", "Three bands", 3),
    ("initial_marginal_lmp_only_v1", "initial_marginal_lmp_only_target_labels5_v1", "LMP only", 1),
    (POLICY, CONTROL, "High power only", 2),
)
ANIMALS = ("Mihili", "Chewie")
COLORS = {"Mihili": "#2679A5", "Chewie": "#CA6B2C"}
CELLS = {(n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}
EXPECTED_SESSIONS = {
    "Mihili_CO_20140217.mat", "Mihili_CO_20140218.mat",
    "Mihili_CO_20140303.mat", "Mihili_CO_20140304.mat",
    "Chewie_CO_20160929.mat", "Chewie_CO_20161005.mat",
    "Chewie_CO_20161006.mat", "Chewie_CO_20161007.mat",
}


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Finite figure values required")
    return value


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("Nonempty fixed session support required")
    return sum(map(finite, values))/len(values)


def equal(actual, expected, message):
    if not math.isclose(finite(actual), finite(expected), rel_tol=0, abs_tol=1e-12):
        raise ArithmeticError(message)


def aggregate_values(panels, policy, key):
    matches = [panel for panel in panels if panel["policy"] == policy]
    if len(matches) != 1 or len(matches[0]["animals"]) != 2:
        raise ValueError("Exactly one complete two-animal policy panel required")
    panel = matches[0]
    values = {animal["animal"]: finite(animal[key]) for animal in panel["animals"]}
    if set(values) != set(ANIMALS):
        raise ValueError("Original two-animal support required")
    equal(panel[key], mean(values.values()), "Equal-animal aggregate does not replay")
    return values, finite(panel[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
        default=OUTPUTS/"runs/development_search/high_power_correspondence_attempt001/high_power_correspondence.json")
    parser.add_argument("--out-prefix", type=Path,
        default=OUTPUTS/"figures/high_power_correspondence_attempt001")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute required for outcome reads and rendering")
    paths = {suffix: args.out_prefix.with_suffix("."+suffix) for suffix in ("png", "pdf", "csv")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Use a fresh figure prefix and preserve existing artifacts")
    report = json.loads(args.report.read_text())
    if (report["status"] != "completed_two_high_power_correspondence_comparisons"
            or report["policy"] != POLICY or report["control"] != CONTROL
            or report["MC_seeds"] != list(range(5)) or report["held_session_accessed"]
            or not report["scientific_scores_released"] or report["valid_evaluations_charged"] != 2
            or report["input_coordinates_per_electrode"] != 2
            or report["native_power_guide_pairs"] != [[100., 200.], [200., 400.]]):
        raise ValueError("Complete adopted two-band comparison required")
    sessions = sorted(report["sessions"], key=lambda session: (session["animal"], session["filename"]))
    omissions = report["fixed_prediction_session_omission"]
    if (len(sessions) != 8 or {s["filename"] for s in sessions} != EXPECTED_SESSIONS
            or any(sum(s["animal"] == animal for s in sessions) != 4 for animal in ANIMALS)
            or len(omissions) != 8 or {o["excluded_session"] for o in omissions} != EXPECTED_SESSIONS):
        raise ValueError("All eight original sessions and fixed-prediction omissions required")

    source = str(args.report.resolve())
    rows, values = [], {}
    def keep(kind, policy, control, label, width, metric, animal, value, session=""):
        rows.append({"row_type": kind, "spectrum": label, "policy": policy, "own_control": control,
            "coordinates_per_electrode": width, "metric": metric, "animal": animal,
            "session_or_omission": session, "value": finite(value), "source_report": source,
            "interpretation": "exposed development; two animals; cross-spectrum information and capacity differ; no confidence interval"})

    for policy, control, label, width in SPECTRA:
        contrast = "mean_delta_Q_from_"+control
        for session in sessions:
            for recipe in (policy, control):
                panel = session["panels"][recipe]
                cells = panel["cells"]
                if len(cells) != 20 or {(c["training_trials"], c["electrodes"]) for c in cells} != CELLS:
                    raise ValueError("Every spectrum retains all twenty equal resource cells")
                for cell in cells:
                    electrode_trials = 32*96+(cell["training_trials"]-32)*cell["electrodes"]
                    if (cell["LFP_electrode_trials"] != electrode_trials
                            or cell["LFP_numeric_elements"] != electrode_trials*10*width
                            or cell["target_numeric_elements"] != cell["training_trials"]*session["target_unit_count"]*10):
                        raise ValueError("Actual spectral bandwidth or target exposure differs from the recipe")
                equal(panel["mean_Q"], mean(c["Q"] for c in cells), "Twenty-cell Q mean differs")
            candidate = session["panels"][policy]
            equal(candidate[contrast], candidate["mean_Q"]-session["panels"][control]["mean_Q"],
                  "Own-control contrast does not replay candidate minus control")
        for metric, key in (("candidate_Q", "mean_Q"), ("own_control_delta_Q", contrast)):
            animal_means, overall = aggregate_values(report["panels"], policy, key)
            for animal in ANIMALS:
                selected = [session for session in sessions if session["animal"] == animal]
                session_values = [(s["filename"], finite(s["panels"][policy][key])) for s in selected]
                equal(animal_means[animal], mean(value for _, value in session_values),
                      "Animal mean does not replay its four sessions")
                omission_values = []
                for omission in omissions:
                    omitted = omission["excluded_session"]
                    reduced, _ = aggregate_values(omission["panels"], policy, key)
                    equal(reduced[animal], mean(s["panels"][policy][key] for s in selected if s["filename"] != omitted),
                          "Omission does not replay fixed session predictions")
                    omission_values.append((omitted, reduced[animal]))
                    keep("fixed_prediction_omission", policy, control, label, width, metric,
                         animal, reduced[animal], omitted)
                values[animal, policy, metric] = {"mean": animal_means[animal],
                    "sessions": session_values, "omissions": omission_values}
                for filename, value in session_values:
                    keep("session", policy, control, label, width, metric, animal, value, filename)
                keep("animal_mean", policy, control, label, width, metric, animal, animal_means[animal])
            keep("equal_animal_mean", policy, control, label, width, metric, "equal animal mean", overall)
            for omission in omissions:
                _, reduced_mean = aggregate_values(omission["panels"], policy, key)
                keep("fixed_prediction_omission", policy, control, label, width, metric,
                     "equal animal mean", reduced_mean, omission["excluded_session"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex="col")
    fig.subplots_adjust(left=.16, right=.965, top=.80, bottom=.255, hspace=.48, wspace=.40)
    fig.suptitle("Spectral inputs: prediction and useful initializer correspondence", fontsize=15, y=.98)
    fig.text(.5, .935, "8 exposed development sessions · 2 animals · 20 equal-weight resource cells · 4 held sessions closed",
             ha="center", fontsize=10, color="#444444")
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#777777", alpha=.65, markersize=4, label="Individual session"),
        Line2D([], [], marker="D", linestyle="none", color="#444444", markersize=6, label="Animal mean (4 sessions)"),
        Line2D([], [], color="#777777", linewidth=2, marker="|", markersize=7,
                   label="Range after one session omission; no refit, not a CI")],
        loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=9)
    limits = {metric: [0.] for metric in ("candidate_Q", "own_control_delta_Q")}
    for row_index, animal in enumerate(ANIMALS):
        color = COLORS[animal]
        for column, metric in enumerate(("candidate_Q", "own_control_delta_Q")):
            ax = axes[row_index, column]
            for index, (policy, _, label, width) in enumerate(SPECTRA):
                item = values[animal, policy, metric]
                session_values = [value for _, value in item["sessions"]]
                omission_values = [value for _, value in item["omissions"]]
                low, high = min(omission_values), max(omission_values)
                for offset, value in zip((-.23, -.18, -.13, -.08), session_values):
                    ax.scatter(value, index+offset, s=23, color=color, alpha=.6, edgecolors="none", zorder=3)
                ax.scatter(item["mean"], index+.035, s=51, marker="D", color=color, zorder=4)
                ax.annotate(f"{item['mean']:+.5f}" if column else f"{item['mean']:.4f}",
                    (item["mean"], index+.035), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=color)
                ax.plot([low, high], [index+.25]*2, color=color, linewidth=2, alpha=.8,
                        marker="|", markersize=7, markeredgewidth=1.2)
                limits[metric].extend((*session_values, *omission_values, item["mean"]))
            title = "Candidate prediction Q" if column == 0 else "Candidate − its own target-label control"
            ax.set_title(animal+" · "+title, loc="left", color=color, fontweight="bold", pad=9)
            ax.set_yticks(range(3), [label+f"\n{width}E input coordinates" for _, _, label, width in SPECTRA])
            ax.set_ylim(2.55, -.55)
            ax.axvline(0, color="#888888", linewidth=.8)
            ax.grid(axis="x", color="#e9e9e9", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Q" if column == 0 else "ΔQ (positive favors true correspondence)")
    for column, metric in enumerate(("candidate_Q", "own_control_delta_Q")):
        low, high = min(limits[metric]), max(limits[metric])
        span = max(high-low, .001)
        axes[0, column].set_xlim(low-.07*span, high+.29*span)

    notes = [
        "Same physical electrode/trial budgets; actual acquired inputs are 3E, 1E and 2E. Information and model capacity differ across spectra.",
        "Each ΔQ uses its own matched mean-five target-label control. All 96-electrode identity cells remain included.",
        "Thin ranges describe all eight fixed-prediction session omissions; they are not uncertainty intervals. Session identities and values are in the CSV.",
    ]
    for animal in ANIMALS:
        changes = []
        for policy, _, label, _ in SPECTRA:
            item = values[animal, policy, "own_control_delta_Q"]
            if any((value > 0) != (item["mean"] > 0) for _, value in item["omissions"]):
                changes.append(label)
        notes.append(animal+" own-control positive screen changes after omission: "+(", ".join(changes) if changes else "none")+".")
    notes.append("Exploratory sensitivity in two animals; no confidence interval, new-animal guarantee, spike-contamination or physiological-cause claim.")
    fig.text(.055, .18, "\n".join(notes), ha="left", va="top", fontsize=8.7, color="#444444", linespacing=1.4)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(paths[suffix], dpi=180, bbox_inches="tight")
    plt.close(fig)
    with paths["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"source_report": source, "figure_paths": {key: str(value) for key, value in paths.items()},
                      "csv_rows": len(rows), "new_fits": 0, "held_session_accessed": False}))


if __name__ == "__main__":
    main()
