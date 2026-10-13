#!/usr/bin/env python3
"""Summarize only a completed initializer-robustness report; no source interface."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
from statistics import fmean

POLICY = "initial_marginal_initializer_orders3_v1"
CONTROL = "initial_marginal_initializer_orders3_target_labels5_v1"
OLD_CONTROL = "initial_marginal_target_labels5_v1"
CONDITIONS = ("original", "order0", "order1", "order2", "mean3")
ANIMALS = ("Chewie", "Mihili")


def close(actual, expected, message):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected) > 1e-10:
        raise ValueError(message)


def panel_means(panel):
    cells = panel["cells"]
    if len(cells) != 20 or {(c["training_trials"], c["electrodes"]) for c in cells} != {
            (n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}:
        raise ValueError("All twenty declared cells required")
    q = fmean(c["Q"] for c in cells)
    d = fmean(c["direction_only_Q"] for c in cells)
    close(panel["mean_Q"], q, "Stored panel Q disagrees with twenty-cell mean")
    return q, d


def extract(report):
    if (report["status"] != "completed_two_initializer_robustness_comparisons"
            or report["recipe"] != "outputs/initializer_robustness_proposal.md"
            or report["valid_evaluations_charged"] != 2 or report["valid_evaluations_total"] != 41
            or report["held_session_accessed"] or not report["scientific_scores_released"]
            or report["initializer_orders"] != [0, 1, 2] or len(report["sessions"]) != 8
            or report["policy"] != POLICY or report["control"] != CONTROL):
        raise ValueError("Only a released complete three-initializer result may be summarized")
    expected_work = {"outer_training_refits": 2280, "inner_ridge_fits": 45600,
        "capacity_inner_candidate_scores": 228000, "selector_orderings": 720,
        "direction_permutation_maps": 4800, "marginal_score_solves": 69120,
        "residual_normalization_views": 120, "direction_only_models": 120}
    if any(report["fit_progress"][key] != value for key, value in expected_work.items()):
        raise ValueError("Full adopted work must complete before extraction")
    if any(sum(s["animal"] == animal for s in report["sessions"]) != 4 for animal in ANIMALS):
        raise ValueError("Four complete sessions in each animal required")
    rows = []
    true_gain_extrema = []
    for session in report["sessions"]:
        if (not session["all285_target_models_fixed_before_evaluation"]
                or not session["all90_target_orders_fixed_before_evaluation"]
                or [r["initializer_order"] for r in session["order_results"]] != [0, 1, 2]):
            raise ValueError("Complete pre-evaluation conditions required")
        pairs = {"original": (session["original_initializer_context"]["initial_marginal"],
                              session["original_initializer_context"][OLD_CONTROL]),
                 "mean3": (session["panels"][POLICY], session["panels"][CONTROL])}
        for order in session["order_results"]:
            pairs["order"+str(order["initializer_order"])] = (order["panels"][POLICY], order["panels"][CONTROL])
            selected = [r for r in order["selection_records"] if r["mc_seed"] is None]
            if len(selected) != 5:
                raise ValueError("All five true selector views required")
            for view in selected:
                gains = view["marginal_gains_by_physical_channel"]
                if len(gains) != 96 or not all(math.isfinite(v) for v in gains):
                    raise ValueError("Complete finite marginal diagnostics required")
                true_gain_extrema.append((min(gains), max(gains)))
        for condition in CONDITIONS:
            candidate, control = pairs[condition]
            cq, d = panel_means(candidate)
            nq, nd = panel_means(control)
            close(d, nd, "Each condition requires its own common direction null")
            rows.append({"condition": condition, "filename": session["filename"], "animal": session["animal"],
                         "candidate_Q": cq, "control_Q": nq, "contrast": cq-nq,
                         "direction_Q": d, "candidate_minus_direction": cq-d})
        means = {r["condition"]: r for r in rows if r["filename"] == session["filename"]}
        for field in ("candidate_Q", "control_Q", "contrast", "direction_Q", "candidate_minus_direction"):
            close(means["mean3"][field], fmean(means["order"+str(i)][field] for i in range(3)),
                  "Mean-three result differs from equal order-specific Q aggregation")

    metrics = ("candidate_Q", "control_Q", "contrast", "direction_Q", "candidate_minus_direction")
    conditions = []
    for condition in CONDITIONS:
        current = [r for r in rows if r["condition"] == condition]
        animals = []
        for animal in ANIMALS:
            selected = [r for r in current if r["animal"] == animal]
            omission = [{"excluded_session": r["filename"],
                         **{field: fmean(s[field] for s in selected if s["filename"] != r["filename"])
                            for field in metrics}} for r in selected]
            animals.append({"animal": animal, **{field: fmean(s[field] for s in selected) for field in metrics},
                "positive_sessions": sum(s["contrast"] > 0 for s in selected),
                "omission_ranges": {field: [min(o[field] for o in omission), max(o[field] for o in omission)]
                                    for field in metrics}, "session_omissions": omission})
        overall_omission = []
        for omitted in current:
            retained = [s for s in current if s["filename"] != omitted["filename"]]
            overall_omission.append({"excluded_session": omitted["filename"],
                **{field: fmean(fmean(s[field] for s in retained if s["animal"] == a) for a in ANIMALS)
                   for field in metrics}})
        conditions.append({"condition": condition, "animals": animals,
            **{field: fmean(a[field] for a in animals) for field in metrics},
            "overall_omission_ranges": {field: [min(o[field] for o in overall_omission),
                                                  max(o[field] for o in overall_omission)] for field in metrics},
            "overall_session_omissions": overall_omission})
        if condition != "original":
            panels = report["panels"] if condition == "mean3" else report["order_panels"][int(condition[-1])]["panels"]
            panel = next(p for p in panels if p["policy"] == POLICY)
            for animal in animals:
                saved = next(a for a in panel["animals"] if a["animal"] == animal["animal"])
                close(animal["contrast"], saved["mean_delta_Q_from_"+CONTROL], "Reported animal contrast differs")
                close(animal["candidate_Q"], saved["mean_Q"], "Reported animal Q differs")
            close(conditions[-1]["contrast"], panel["mean_delta_Q_from_"+CONTROL], "Reported overall contrast differs")

    averaged = next(c for c in conditions if c["condition"] == "mean3")
    own = all(a["contrast"] > 0 for a in averaged["animals"])
    stable = all(a["contrast"] > 0 for c in conditions if c["condition"].startswith("order") for a in c["animals"])
    screens = report["prospective_directional_screens"]
    if (screens["mean_three_order_correspondence_positive_in_both_animals"] != own
            or screens["correspondence_positive_in_both_animals_in_every_order"] != stable):
        raise ValueError("Saved screen disagrees with declared signs")
    return {"status": report["status"], "valid_evaluations_total": 41,
        "resources": {key: report[key] for key in ("started_utc", "observed_utc", "slurm_job_id", "hostname",
            "elapsed_seconds", "process_CPU_seconds", "maximum_RSS_KiB", "logical_neural_bytes",
            "unique_logical_neural_bytes", "fitting_trial_read_calls", "evaluation_trials_read", "retained_report_bytes")},
        "fit_progress": report["fit_progress"], "conditions": conditions, "session_rows": rows,
        "prospective_directional_screens": screens, "seed_summaries": report["seed_summaries"],
        "true_selector_score_range": [min(v[0] for v in true_gain_extrema), max(v[1] for v in true_gain_extrema)],
        "true_selector_views": len(true_gain_extrema),
        "within_order_full96_replay": report["within_order_full96_score_replay_max_absolute_difference"],
        "held_session_accessed": report["held_session_accessed"],
        "limits": "Original is separate context; changed prefixes/nulls/CV. Orders overlap, two animals, development only; no patience reset."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Completed-result analysis belongs in verified coordinator compute")
    result = extract(json.loads(args.report.read_text()))
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("Preserve existing extraction")
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
