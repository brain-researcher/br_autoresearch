#!/usr/bin/env python3
"""Plot completed initializer-correspondence evidence without refitting models."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket

OUTPUTS = Path(__file__).resolve().parents[1]
CONTROL = "initial_marginal_target_labels5_v1"
ANIMALS = ("Mihili", "Chewie")
COLORS = {"Mihili": "#2679A5", "Chewie": "#CA6B2C"}
CONTRAST = "mean_delta_Q_from_"+CONTROL
EXPECTED_CELLS = {(n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}


def date_label(filename):
    raw = Path(filename).stem.rsplit("_", 1)[-1]
    return raw[:4]+"-"+raw[4:6]+"-"+raw[6:8]


def true_panel(panels):
    matches = [panel for panel in panels if panel["policy"] == "initial_marginal"]
    if len(matches) != 1:
        raise ValueError("Exactly one reused true-initializer panel required")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=OUTPUTS/"runs/development_search/initializer_target_label_control_attempt001/initializer_target_label_control.json")
    parser.add_argument("--out-prefix", type=Path, default=OUTPUTS/"figures/initializer_target_label_control_attempt001")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Rendering and result extraction require coordinator-owned compute")
    destinations = {suffix: args.out_prefix.with_suffix("."+suffix) for suffix in ("png", "pdf", "csv")}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("Use a fresh figure prefix; existing artifacts are preserved")
    report = json.loads(args.report.read_text())
    if (report["status"] != "completed_one_initializer_target_label_control"
            or report["control"] != CONTROL or report["MC_seeds"] != list(range(5))
            or report["held_session_accessed"] or len(report["sessions"]) != 8):
        raise ValueError("Complete eight-session/five-seed development report required")
    sessions = sorted(report["sessions"], key=lambda item: (ANIMALS.index(item["animal"]), item["filename"]))
    if len({session["filename"] for session in sessions}) != 8 or any(
            sum(session["animal"] == animal for session in sessions) != 4 for animal in ANIMALS):
        raise ValueError("Four distinct original sessions per animal required")
    for session in sessions:
        for policy in ("initial_marginal", CONTROL):
            cells = session["panels"][policy]["cells"]
            if len(cells) != 20 or {(cell["training_trials"], cell["electrodes"]) for cell in cells} != EXPECTED_CELLS:
                raise ValueError("All twenty cells, including the five all96 cells, must remain included")
    panel = true_panel(report["panels"])
    means = {animal["animal"]: animal[CONTRAST] for animal in panel["animals"]}
    seeds = sorted(report["seed_summaries"], key=lambda item: item["mc_seed"])
    if [seed["mc_seed"] for seed in seeds] != list(range(5)) or set(means) != set(ANIMALS):
        raise ValueError("Both animals and all five fixed seed labels required")
    seed_values = {animal: [next(item["true_minus_seed_Q"] for item in seed["animals"]
                                if item["animal"] == animal) for seed in seeds] for animal in ANIMALS}
    for animal in ANIMALS:
        if not math.isclose(sum(seed_values[animal])/5, means[animal], rel_tol=0, abs_tol=1e-12):
            raise ArithmeticError("Seed-mean contrast differs from the reported mean-SSE control")
    values = [session["panels"]["initial_marginal"][CONTRAST] for session in sessions]
    if not all(math.isfinite(value) for value in values+list(means.values())+
               [value for series in seed_values.values() for value in series]):
        raise ValueError("Finite plotted values required")

    source = str(args.report.resolve())
    evidence = "exposed development; two animals; no independent seed replication"
    rows = []
    def retain(kind, animal, value, session="", seed=""):
        rows.append({"row_type": kind, "animal": animal, "session": session, "mc_seed": seed,
                     "true_minus_control_delta_Q": value, "source_report": source, "evidence": evidence})
    for session, value in zip(sessions, values):
        retain("session_mean5_control", session["animal"], value, session["filename"])
    for animal in ANIMALS:
        for seed, value in zip(range(5), seed_values[animal]):
            retain("animal_control_seed", animal, value, seed=seed)
        retain("animal_mean5_control", animal, means[animal])
    retain("equal_animal_mean5_control", "equal animal mean", panel[CONTRAST])
    flips = []
    for omission in report["fixed_prediction_session_omission"]:
        omitted = true_panel(omission["panels"])
        for animal in omitted["animals"]:
            value = animal[CONTRAST]
            retain("fixed_prediction_omission", animal["animal"], value, omission["excluded_session"])
            if (value > 0) != (means[animal["animal"]] > 0):
                flips.append((animal["animal"], date_label(omission["excluded_session"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.9), gridspec_kw={"width_ratios": (1.25, 1)})
    fig.subplots_adjust(left=.18, right=.97, bottom=.25, top=.80, wspace=.31)
    fig.suptitle("True initializer pairing versus target-label control", fontsize=15, y=.98)
    fig.text(.5, .925, "Exposed development corpus · 2 animals · 20 equal-weight resource cells per session",
             ha="center", color="#444444", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=COLORS[animal], label=animal)
                        for animal in ANIMALS], loc="upper center", bbox_to_anchor=(.57, .895), ncol=2,
               frameon=False, handletextpad=.45, columnspacing=1.4)

    ax = axes[0]
    ax.set_title("A  Every development session", loc="left", pad=10, fontweight="bold")
    for position, (session, value) in enumerate(zip(sessions, values)):
        color = COLORS[session["animal"]]
        ax.plot([0, value], [position, position], color=color, linewidth=1.7, alpha=.7)
        ax.scatter(value, position, color=color, s=40, zorder=3)
        ax.annotate(f"{value:+.4f}", (value, position), xytext=(7 if value >= 0 else -7, 0),
                    textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.set_yticks(range(8), [session["animal"]+" "+date_label(session["filename"]) for session in sessions])
    for label, session in zip(ax.get_yticklabels(), sessions):
        label.set_color(COLORS[session["animal"]])
    ax.set_ylim(7.65, -.65)
    ax.axhline(3.5, color="#E3E3E3", linewidth=.8)
    low, high = min(0., min(values)), max(0., max(values))
    span = max(high-low, .001)
    ax.set_xlim(low-.22*span, high+.28*span)
    ax.axvline(0, color="#777777", linewidth=.9)
    ax.set_xlabel("ΔQ: true pairing − mean five-seed control")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", color="#ECECEC", linewidth=.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.set_title("B  Five control seeds and animal means", loc="left", pad=10, fontweight="bold")
    for animal in ANIMALS:
        color = COLORS[animal]
        ax.scatter(range(5), seed_values[animal], color=color, s=40, zorder=3)
        ax.plot([-.2, 5.15], [means[animal], means[animal]], color=color, linestyle=":", linewidth=1.2)
        ax.scatter(5.15, means[animal], color=color, marker="D", s=62, zorder=4)
        ax.annotate(f"{means[animal]:+.5f}", (5.15, means[animal]), xytext=(-2, 9),
                    textcoords="offset points", ha="center", va="bottom", color=color, fontsize=9)
    ax.axhline(0, color="#777777", linewidth=.9)
    ax.axvline(4.55, color="#E3E3E3", linewidth=.9)
    ax.set_xticks([0, 1, 2, 3, 4, 5.15], ["0", "1", "2", "3", "4", "Mean"])
    ax.set_xlim(-.45, 5.8)
    all_seed_values = [value for series in seed_values.values() for value in series]
    low, high = min(0., min(all_seed_values)), max(0., max(all_seed_values))
    span = max(high-low, .001)
    ax.set_ylim(low-.14*span, high+.26*span)
    ax.set_xlabel("Predetermined control seed")
    ax.set_ylabel("ΔQ: true pairing − control")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color="#ECECEC", linewidth=.7)
    ax.set_axisbelow(True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, color="#777777")

    footnote = ("Positive ΔQ favors true initializer correspondence. All 96-electrode identity cells remain included.\n"
                "Points show finite control-seed variation; seeds are not independent animals. No confidence intervals.")
    if flips:
        descriptions = [animal+" mean changes sign when "+date+" is omitted" for animal, date in flips]
        footnote += "\nFixed predictions: "+"; ".join(descriptions)+"."
    fig.text(.18, .14, footnote, ha="left", va="top", fontsize=9, color="#444444", linespacing=1.5)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destinations[suffix], dpi=180, bbox_inches="tight")
    plt.close(fig)
    with destinations["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"source_report": source, "figure_paths": {key: str(path) for key, path in destinations.items()},
                      "csv_rows": len(rows), "refitted": False, "held_session_accessed": False}))


if __name__ == "__main__":
    main()
