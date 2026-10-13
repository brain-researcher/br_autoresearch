#!/usr/bin/env python3
"""Plot completed initializer comparisons from one embedded outcome report."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket

OUTPUTS = Path(__file__).resolve().parents[1]
POLICY = "initial_residual_conditional_v1"
CONTROL = "initial_residual_conditional_target_labels5_v1"
MARGINAL = "initial_marginal"
REPEATABILITY = "initial_direction_repeatability_v1"
PAIRS = ((MARGINAL, "initial_marginal_target_labels5_v1", "Marginal", "Target-label control"),
         (REPEATABILITY, "initial_uniform_order5_v1", "Repeatability", "Uniform-order control"),
         (POLICY, CONTROL, "Conditional", "Target-label control"))
ANIMALS = ("Mihili", "Chewie")
COLORS = {"Mihili": "#2679A5", "Chewie": "#CA6B2C"}
CELLS = {(n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}


def date_label(filename):
    raw = Path(filename).stem.rsplit("_", 1)[-1]
    return raw[:4]+"-"+raw[4:6]+"-"+raw[6:8]


def panel(panels, policy):
    found = [p for p in panels if p["policy"] == policy]
    if len(found) != 1:
        raise ValueError("Exactly one complete embedded policy panel required")
    return found[0]


def animal_values(panels, policy, comparator):
    result = panel(panels, policy)
    key = "mean_delta_Q_from_"+comparator
    values = {a["animal"]: a[key] for a in result["animals"]}
    if set(values) != set(ANIMALS) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Two finite animal contrasts required")
    if not math.isclose(sum(values.values())/2, result[key], rel_tol=0, abs_tol=1e-12):
        raise ArithmeticError("Reported equal-animal contrast differs")
    return values, result[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
        default=OUTPUTS/"runs/development_search/initializer_conditional_attempt001/initializer_conditional_gain.json")
    parser.add_argument("--out-prefix", type=Path, default=OUTPUTS/"figures/initializer_conditional_attempt001")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute required for result extraction and rendering")
    outputs = {suffix: args.out_prefix.with_suffix("."+suffix) for suffix in ("png", "pdf", "csv")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Use a fresh figure prefix; preserve prior figures")
    report = json.loads(args.report.read_text())
    if (report["status"] != "completed_two_initializer_conditional_comparisons"
            or report["policy"] != POLICY or report["control"] != CONTROL
            or report["MC_seeds"] != list(range(5)) or report["held_session_accessed"]
            or len(report["sessions"]) != 8 or len(report["fixed_prediction_session_omission"]) != 8):
        raise ValueError("Complete original eight-session conditional comparison required")
    sessions = sorted(report["sessions"], key=lambda s: (ANIMALS.index(s["animal"]), s["filename"]))
    if (len({s["filename"] for s in sessions}) != 8
            or any(sum(s["animal"] == a for s in sessions) != 4 for a in ANIMALS)):
        raise ValueError("Exactly four original sessions in each animal required")
    for session in sessions:
        for policy, control, _, _ in PAIRS:
            for recipe in (policy, control):
                cells = session["panels"][recipe]["cells"]
                if len(cells) != 20 or {(c["training_trials"], c["electrodes"]) for c in cells} != CELLS:
                    raise ValueError("All twenty equal-weight resource cells must remain included")
    source = str(args.report.resolve())
    evidence = "exposed exploratory development; two animals; four held sessions closed"
    rows = []
    def keep(kind, policy, comparator, animal, value, session=""):
        if not math.isfinite(value):
            raise ValueError("Finite figure values required")
        rows.append({"row_type": kind, "policy": policy, "comparator": comparator,
                     "animal": animal, "session_or_omission": session, "delta_Q": value,
                     "source_report": source, "evidence": evidence})
    controls = {}
    for policy, control, _, _ in PAIRS:
        values, overall = animal_values(report["panels"], policy, control)
        controls[policy] = values
        for animal, value in values.items(): keep("animal_own_control", policy, control, animal, value)
        keep("equal_animal_own_control", policy, control, "equal animal mean", overall)
    challenge, challenge_overall = animal_values(report["panels"], POLICY, MARGINAL)
    differences = {}
    for session in sessions:
        value = session["panels"][POLICY]["mean_Q"]-session["panels"][MARGINAL]["mean_Q"]
        differences[session["filename"]] = value
        keep("session_incumbent_challenge", POLICY, MARGINAL, session["animal"], value, session["filename"])
    for animal in ANIMALS:
        mean = sum(differences[s["filename"]] for s in sessions if s["animal"] == animal)/4
        if not math.isclose(mean, challenge[animal], rel_tol=0, abs_tol=1e-12):
            raise ArithmeticError("Session contrast mean differs from reported animal mean")
        keep("animal_incumbent_challenge", POLICY, MARGINAL, animal, mean)
    keep("equal_animal_incumbent_challenge", POLICY, MARGINAL, "equal animal mean", challenge_overall)
    omission_overall, sign_flips = [], []
    conditional_control_robust = True
    for omission in report["fixed_prediction_session_omission"]:
        values, overall = animal_values(omission["panels"], POLICY, MARGINAL)
        own_control, own_overall = animal_values(omission["panels"], POLICY, CONTROL)
        omitted = omission["excluded_session"]
        omission_overall.append(overall)
        conditional_control_robust &= all(v > 0 for v in own_control.values())
        for animal in ANIMALS:
            keep("fixed_prediction_omission", POLICY, MARGINAL, animal, values[animal], omitted)
            keep("fixed_prediction_omission", POLICY, CONTROL, animal, own_control[animal], omitted)
            if (values[animal] > 0) != (challenge[animal] > 0):
                sign_flips.append(animal+" changes sign without "+date_label(omitted))
        keep("fixed_prediction_omission", POLICY, MARGINAL, "equal animal mean", overall, omitted)
        keep("fixed_prediction_omission", POLICY, CONTROL, "equal animal mean", own_overall, omitted)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.9), gridspec_kw={"width_ratios": (1., 1.35)})
    fig.subplots_adjust(left=.12, right=.965, top=.78, bottom=.27, wspace=.60)
    fig.suptitle("Initializer selectors: controls and the marginal incumbent", fontsize=15, y=.98)
    fig.text(.5, .925, "Exposed development · 2 animals · 20 equal-weight cells per session · 4 held sessions closed",
             ha="center", color="#444444", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=COLORS[a], label=a) for a in ANIMALS],
               loc="upper center", bbox_to_anchor=(.5, .895), ncol=2, frameon=False)
    ax = axes[0]
    ax.set_title("A  Advantage over each selector’s control", loc="left", fontweight="bold", pad=12)
    all_values = []
    for index, (policy, control, label, control_label) in enumerate(PAIRS):
        for offset, animal in zip((-.14, .14), ANIMALS):
            value = controls[policy][animal]
            all_values.append(value)
            y = index+offset
            ax.plot([0, value], [y, y], color=COLORS[animal], alpha=.55, linewidth=1.5)
            ax.scatter(value, y, s=52, color=COLORS[animal], zorder=3)
            ax.annotate(f"{value:+.5f}", (value, y), xytext=(6 if value >= 0 else -6, 0),
                        textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.set_yticks(range(3), [p[2]+"\n"+p[3] for p in PAIRS])
    ax.set_ylim(2.6, -.6)
    low, high = min(0., min(all_values)), max(0., max(all_values))
    span = max(high-low, .001)
    ax.set_xlim(low-.13*span, high+.45*span)
    ax.set_xlabel("ΔQ: selector − its mean five-seed control")

    ax = axes[1]
    ax.set_title("B  Conditional − marginal selection", loc="left", fontweight="bold", pad=12)
    points = []
    for animal in ANIMALS:
        points.extend((animal+" "+date_label(s["filename"]), animal, differences[s["filename"]], False)
                      for s in sessions if s["animal"] == animal)
        points.append((animal+" mean", animal, challenge[animal], True))
    for index, (label, animal, value, is_mean) in enumerate(points):
        ax.plot([0, value], [index, index], color=COLORS[animal], alpha=.5, linewidth=1.5)
        ax.scatter(value, index, s=67 if is_mean else 35, marker="D" if is_mean else "o",
                   color=COLORS[animal], zorder=3)
        ax.annotate(f"{value:+.4f}", (value, index), xytext=(6 if value >= 0 else -6, 0),
                    textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=8.5,
                    fontweight="bold" if is_mean else "normal")
    ax.set_yticks(range(len(points)), [p[0] for p in points])
    for label, point in zip(ax.get_yticklabels(), points):
        label.set_color(COLORS[point[1]])
        if point[3]: label.set_fontweight("bold")
    ax.axhline(4.5, color="#dddddd", linewidth=.8)
    ax.set_ylim(9.6, -.6)
    low, high = min(0., min(p[2] for p in points)), max(0., max(p[2] for p in points))
    span = max(high-low, .001)
    ax.set_xlim(low-.34*span, high+.32*span)
    ax.set_xlabel("ΔQ: conditional − marginal (positive favors conditional)")
    for ax in axes:
        ax.axvline(0, color="#777777", linewidth=.9)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", color="#ececec", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    notes = ["Controls differ across selectors: target-label correspondence for marginal/conditional; uniform physical order for repeatability.",
             "All 96-electrode identity cells remain included. Seeds and sessions are not additional animals; no confidence intervals or causal claim."]
    if conditional_control_robust:
        notes.append("Conditional beats its own control in both animals after every fixed-prediction session omission.")
    if all(v < 0 for v in omission_overall):
        notes.append("Conditional remains below marginal in the equal-animal mean after every single-session omission.")
    if sign_flips:
        notes.append("Animal influence: "+"; ".join(sign_flips)+".")
    fig.text(.06, .17, "\n".join(notes), ha="left", va="top", fontsize=9, color="#444444", linespacing=1.45)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(outputs[suffix], dpi=180, bbox_inches="tight")
    plt.close(fig)
    with outputs["csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"source_report": source, "figure_paths": {k: str(v) for k, v in outputs.items()},
                      "csv_rows": len(rows), "refitted": False, "held_session_accessed": False}))


if __name__ == "__main__":
    main()
