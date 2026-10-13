#!/usr/bin/env python3
"""Compare completed HW, blend and scalar-calibration decompositions only."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time


ANIMALS = ("Mihili", "Chewie")
RECIPES = ("HW", "blend", "calibrated_HW")
LABELS = ("History", "Current/history\nblend", "Calibrated\nhistory")
TERMS = ("mean_gain", "within_gain", "primary_increment", "identity_increment")
COLORS = ("#4477AA", "#228833", "#AA3377")


def read_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in TERMS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite completed decomposition: " + str(path))
        for key in ("bin", "time_ms", "direction"):
            if key in row:
                row[key] = int(row[key])
        if not math.isclose(row["mean_gain"] + row["within_gain"],
                            row["primary_increment"], rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError("Saved components do not reconstruct saved primary score")
    return rows


def mean(rows):
    if not rows:
        raise ValueError("Empty fixed group")
    return {key: sum(row[key] for row in rows) / len(rows) for key in TERMS}


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified Slurm compute allocation")
    if any(os.environ.get(key) != "1" for key in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Use the four existing single-thread limits")
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if not out.is_relative_to(root / "figures"):
        raise ValueError("Use a fresh episode-local outputs/figures directory")
    # Imports and report processing take place only after the allocation guard.
    from source_adapter import M1_SESSIONS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected = {f"{animal}_{day}": animal for animal, _, day in M1_SESSIONS}
    sessions = sorted(expected, key=lambda s: (ANIMALS.index(expected[s]), s))
    old = root / "runs/descriptive_secondary/fixed_prediction_decomposition_attempt001"
    cal = root / "runs/fixed_hw_correction_calibration/attempt001"
    sources = [old / "decomposition.json", old / "sessions.csv", old / "bins.csv",
               cal / "calibration.json", cal / "decomposition_sessions.csv", cal / "direction_bins.csv"]
    old_report = json.loads(sources[0].read_text())
    cal_report = json.loads(sources[3].read_text())
    for report, status in ((old_report, "complete_descriptive_decomposition"),
                           (cal_report, "complete_fixed_hw_correction_calibration")):
        if (report.get("status") != status
                or report.get("original_selected_scores_replayed_exactly") is not True
                or report.get("decomposition_identities_passed") is not True):
            raise ValueError("Require both completed and replayed source reports")
    old_rows, old_bins = read_rows(sources[1]), read_rows(sources[2])
    cal_rows, cal_direction_bins = read_rows(sources[4]), read_rows(sources[5])
    if len(old_rows) != 48 or len(old_bins) != 720 or len(cal_rows) != 16 or len(cal_direction_bins) != 1920:
        raise ValueError("Completed input tables have changed support")
    by_session = {(r["session_id"], r["recipe"]): r for r in old_rows}
    if set(by_session) != {(s, r) for s in sessions for r in ("W", "HW", "blend")}:
        raise ValueError("Require all original sessions and fixed reference recipes")
    for row in cal_rows:
        session = row["session_id"]
        if float(row["amplitude"]) != float(cal_report["selected_amplitudes"][session]):
            raise ValueError("Selected calibration amplitude differs from its completed report")
        key = session, "calibrated_HW"
        if key in by_session:
            raise ValueError("Duplicate calibrated session")
        by_session[key] = row
    session_rows = []
    for session in sessions:
        for recipe in RECIPES:
            row = by_session[session, recipe]
            if row["animal"] != expected[session]:
                raise ValueError("Original biological grouping changed")
            session_rows.append({"session_id": session, "animal": expected[session], "recipe": recipe,
                                 **{key: row[key] for key in TERMS}})

    bin_map = {(r["session_id"], r["recipe"], r["bin"]): r for r in old_bins}
    if set(bin_map) != {(s, r, b) for s in sessions for r in ("W", "HW", "blend") for b in range(15)}:
        raise ValueError("Require every original session/recipe/bin")
    direction_map = {(r["session_id"], r["direction"], r["bin"]): r for r in cal_direction_bins}
    if len(direction_map) != 1920 or set(direction_map) != {(s, d, b) for s in sessions for d in range(8) for b in range(15)}:
        raise ValueError("Require every calibrated direction/bin exactly once")
    for session in sessions:
        for tick in range(15):
            rows = [direction_map[session, d, tick] for d in range(8)]
            if any(r["animal"] != expected[session] or r["time_ms"] != tick * 30 for r in rows):
                raise ValueError("Saved calibrated timing/grouping changed")
            bin_map[session, "calibrated_HW", tick] = {
                "animal": expected[session], "time_ms": tick * 30,
                **{key: sum(r[key] for r in rows) for key in TERMS}}
    bins = []
    for session in sessions:
        for recipe in RECIPES:
            selected = [bin_map[session, recipe, b] for b in range(15)]
            for key in TERMS:
                if not math.isclose(sum(r[key] for r in selected), by_session[session, recipe][key],
                                    rel_tol=1e-11, abs_tol=1e-12):
                    raise ValueError("All fifteen bin contributions must reconstruct saved session totals")
            for tick, row in enumerate(selected):
                if row["animal"] != expected[session] or row["time_ms"] != tick * 30:
                    raise ValueError("Original temporal support/grouping changed")
                bins.append({"session_id": session, "animal": expected[session], "recipe": recipe,
                             "bin": tick, "time_ms": tick * 30, **{key: row[key] for key in TERMS}})

    aggregates = []
    for recipe in RECIPES:
        animal_rows = [{"group": animal, "recipe": recipe,
                        **mean([r for r in session_rows if r["animal"] == animal and r["recipe"] == recipe])}
                       for animal in ANIMALS]
        aggregates.extend(animal_rows)
        aggregates.append({"group": "animal_balanced", "recipe": recipe, **mean(animal_rows)})
        aggregate = aggregates[-1]
        reference = (next(r for r in old_report["animal_balanced"] if r["recipe"] == recipe)
                     if recipe != "calibrated_HW" else cal_report["candidate"])
        for key in ("primary_increment", "identity_increment"):
            if not math.isclose(aggregate[key], reference[key], rel_tol=1e-11, abs_tol=1e-12):
                raise ValueError("Displayed aggregate differs from completed score report")
    contrasts = []
    for first, second in (("blend", "HW"), ("calibrated_HW", "blend")):
        for session in sessions:
            contrasts.append({"session_id": session, "animal": expected[session],
                              "contrast": first + "_minus_" + second,
                              **{key: by_session[session, first][key] - by_session[session, second][key] for key in TERMS}})
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "sessions.csv", session_rows)
    write_csv(out / "bins.csv", bins)
    write_csv(out / "aggregate_components.csv", aggregates)
    write_csv(out / "session_contrasts.csv", contrasts)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for col, animal in enumerate(ANIMALS):
        ax = axes[0, col]
        rows = [next(r for r in aggregates if r["group"] == animal and r["recipe"] == recipe) for recipe in RECIPES]
        ax.bar([x - .16 for x in range(3)], [r["mean_gain"] * 1000 for r in rows], .30,
               color="#CCBB44", label="Direction/time mean correction")
        ax.bar([x + .16 for x in range(3)], [r["within_gain"] * 1000 for r in rows], .30,
               color="#4477AA", label="Net trial-varying correction")
        ax.scatter(range(3), [r["primary_increment"] * 1000 for r in rows], marker="D", color="black", label="Total gain", zorder=3)
        ax.axhline(10, ls="--", color=".45", lw=1, label="Aggregate primary margin (reference)")
        ax.set(xticks=range(3), xticklabels=LABELS, title=f"{animal}: full-window contributions", ylabel="Contribution to delta R² × 1,000")
        ax.legend(fontsize=8, loc="upper left", frameon=False)

    for col, contrast in enumerate(("blend_minus_HW", "calibrated_HW_minus_blend")):
        ax = axes[1, col]
        rows = [r for r in contrasts if r["contrast"] == contrast]
        for key, color, label, marker in (("mean_gain", "#AA9900", "Mean correction", "o"),
                                         ("within_gain", "#4477AA", "Trial-varying correction", "o"),
                                         ("primary_increment", "black", "Total gain", "D")):
            ax.plot(range(16), [r[key] * 1000 for r in rows], color=color, marker=marker, lw=.7, ms=4, label=label)
        ax.axvline(5.5, color=".7", lw=.8)
        ax.set(xticks=range(16), xticklabels=[s[0] + s.split("_")[1][2:] for s in sessions],
               title="Blend − history" if col == 0 else "Calibrated history − blend",
               ylabel="Session contrast × 1,000")
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel("Every original session: M = Mihili, C = Chewie; YYMMDD")

    for col, animal in enumerate(ANIMALS):
        for row_index, term, title in ((2, "mean_gain", "Direction/time mean correction"),
                                      (3, "within_gain", "Net trial-varying correction")):
            ax = axes[row_index, col]
            for recipe, color, label in zip(RECIPES, COLORS, ("History", "Blend", "Calibrated history")):
                values = [mean([r for r in bins if r["animal"] == animal and r["recipe"] == recipe and r["bin"] == tick])[term]
                          for tick in range(15)]
                ax.plot(range(0, 421, 30), [v * 1000 for v in values], color=color, marker="o", ms=3, label=label)
            ax.set(title=f"{animal}: {title}", xlabel="Native movement-aligned time (ms)",
                   ylabel="Per-bin contribution × 1,000", xticks=range(0, 421, 60))
            ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.axhline(0, color=".5", lw=.7)
        ax.grid(axis="y", color=".92", lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Similar total prediction gains can reflect different mean and trial-varying corrections", fontsize=16, y=.995)
    fig.text(.5, .01, "Exposed development data: all 16 sessions, both animals, all 15 bins; no refits or new selection.\n"
             "Bins sum using the original whole-session XY denominator; curves are session means, not per-bin R² or confidence intervals.\n"
             "Original animal-balanced margins remain 0.01 primary / 0.005 identity; no supported primary candidate or neural-mechanism claim.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .98), h_pad=2.5)
    for extension in ("png", "pdf"):
        fig.savefig(out / ("fixed_prediction_components." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)
    report = {"analysis_id": "fixed_prediction_components", "status": "complete",
              "sources": [str(p) for p in sources], "aggregates": aggregates, "sessions": 16, "bins_per_session": 15,
              "session_rows": len(session_rows), "bin_rows": len(bins), "session_contrast_rows": len(contrasts),
              "source_score_reconstruction_passed": True, "new_fits": 0, "new_selection": False,
              "original_search_reopened": False, "primary_margin": .01, "identity_margin": .005,
              "evidence_role": "comparative description of completed exposed-development results",
              "hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
              "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started}
    (out / "components.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Fixed prediction components", "", "The same completed predictions are compared across both animals, all sixteen sessions and all fifteen native bins. No model is fitted or selected here.", "",
             "| Group | Policy | Mean correction | Net trial-varying correction | Total delta R² | Identity |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in aggregates:
        lines.append(f"| {row['group']} | {row['recipe']} | {row['mean_gain']:+.9f} | {row['within_gain']:+.9f} | {row['primary_increment']:+.9f} | {row['identity_increment']:+.9f} |")
    lines += ["", "Mean and trial-varying terms sum to the existing primary result. The trial-varying term is covariance benefit minus prediction energy; identity is a separate whole-trial pairing contrast. Evaluation means are descriptive accounting only. Curves use the unchanged whole-session denominator and are not per-bin R² endpoints.", "",
              "Every original session is shown in both pairwise contrasts; all directions contribute to each of the fifteen time bins. Sessions are averaged within animal, then animals equally. No favorable interval, seed or session is selected, and no uncertainty intervals or biological replication claims are added.", "",
              "The original 45-comparison stop and separate calibration closeout remain unchanged. The 0.01 primary and 0.005 identity practical margins remain in force; these plots cannot promote a candidate. There is no neural-source access, future-session evaluation or spike-population follow-up.", "",
              "![Complete component comparison](fixed_prediction_components.png)", "",
              "[PDF](fixed_prediction_components.pdf) · [Session values](sessions.csv) · [All bins](bins.csv) · [Contrasts](session_contrasts.csv) · [Sources and execution](components.json)"]
    (out / "interpretation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "sessions", "bin_rows", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
