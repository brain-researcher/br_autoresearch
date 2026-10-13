#!/usr/bin/env python3
"""Describe fixed W/HW/blend errors; no source access, fitting or selection."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from primary_history_blend import COMPONENTS, SCORE_KEYS, blend_prediction, prediction_scores
from primary_reference import ANIMALS, SESSIONS
from run_primary_history_blend import SUMMARY_STATUSES, load_components
from source_adapter import compute_only


TERMS = (
    "mean_alignment", "mean_prediction_energy", "mean_gain",
    "within_covariance_benefit", "within_prediction_energy", "within_gain",
    "primary_increment", "identity_increment",
)


def assert_roundoff(actual, expected, label):
    # Only the NEW algebraic decomposition uses roundoff tolerance. Original
    # saved score/support replays below retain their existing exact equality.
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-12):
        raise ArithmeticError(f"Decomposition identity failed: {label}")


def decompose(y, baseline, prediction, labels):
    """Return additive direction/bin terms in units of the original session D.

    For r=P-B and e=Y-B, split e and r into direction/time means plus
    trial deviations: gain = (2*n*mean(e)*mean(r)-n*mean(r)^2)
                           + 2*sum(e_centered*r_centered)-sum(r_centered^2).
    Exact wrong-trial identity is 2*n/(n-1)*sum(e_centered*r_centered).
    Evaluation means are descriptive bookkeeping, never a fitted predictor.
    """
    denominator = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Original joint native-XY denominator must be positive")
    by_direction = []
    for direction in range(8):
        keep = labels == direction
        n = int(keep.sum())
        if n < 2:
            raise ValueError("At least two original trials per direction required")
        b = baseline[keep]
        if not np.array_equal(b, np.broadcast_to(b[0], b.shape)):
            raise ValueError("Direction/time baseline must be common within direction")
        e, r = y[keep]-b, prediction[keep]-b
        em, rm = e.mean(axis=0), r.mean(axis=0)
        ec, rc = e-em, r-rm
        mean_alignment = 2*n*np.sum(em*rm, axis=1)/denominator
        mean_energy = n*np.sum(rm*rm, axis=1)/denominator
        covariance = 2*np.sum(ec*rc, axis=(0, 2))/denominator
        within_energy = np.sum(rc*rc, axis=(0, 2))/denominator
        direct = np.sum(e*e-(e-r)*(e-r), axis=(0, 2))/denominator
        mean_gain = mean_alignment-mean_energy
        within_gain = covariance-within_energy
        assert_roundoff(mean_gain+within_gain, direct, "direction/bin gain")
        for tick in range(15):
            by_direction.append({
                "direction": direction, "bin": tick, "time_ms": tick*30,
                "trials": n, "mean_alignment": float(mean_alignment[tick]),
                "mean_prediction_energy": float(mean_energy[tick]),
                "mean_gain": float(mean_gain[tick]),
                "within_covariance_benefit": float(covariance[tick]),
                "within_prediction_energy": float(within_energy[tick]),
                "within_gain": float(within_gain[tick]),
                "primary_increment": float(direct[tick]),
                "identity_increment": float(covariance[tick]*n/(n-1)),
            })
    totals = {key: float(sum(row[key] for row in by_direction)) for key in TERMS}
    scores = prediction_scores(y, baseline, labels, prediction)
    assert_roundoff(totals["primary_increment"], scores["delta_R2"], "original primary")
    assert_roundoff(totals["identity_increment"], scores["identity_increment"], "derangement identity")
    return totals, by_direction, scores


def average(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in TERMS}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args()
    compute_only()
    if any(os.environ.get(key) != "1" for key in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise RuntimeError("Four explicit single-thread numerical limits required")
    if not 0 < args.max_seconds <= 180:
        raise ValueError("Descriptive artifact computation is bounded at 180 seconds")
    outputs = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if not out.is_relative_to(outputs/"runs"/"descriptive_secondary"):
        raise ValueError("Use a fresh episode-local runs/descriptive_secondary attempt")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "started", "analysis": "fixed_W_HW_blend_SSE_decomposition",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
        "provider_source_values_read": False, "neural_refits": 0,
        "new_candidate_selection": False, "scientific_trial_count_increment": 0,
        "original_comparison_count": 45, "original_stop_preserved": True,
        "primary_margin": .01, "identity_margin": .005,
        "evidence": "descriptive secondary analysis of exposed development predictions",
        "interpretation": "Trial means only decompose fixed errors; no centered predictor is fit or scored. "
            "Identity is pairing covariance, not a within-trial R-squared. "
            "All bins use the original whole-session XY denominator, not per-bin denominators. "
            "Bins/directions/sessions are not additional animals; no physiological lag or causal claim.",
    }
    def timeout(*_):
        raise TimeoutError("180-second artifact-only decomposition limit reached")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        root = outputs/"runs"/"adaptive_search"
        roots = {
            "W": root/"primary_repeatability_weight_control_attempt001",
            "HW": root/"primary_history_repeatability_weight_control_attempt002",
        }
        summary_paths = {
            "W": root/"primary_repeatability_weight_summary_attempt001"/"primary_repeatability_weight.json",
            "HW": root/"primary_history_repeatability_weight_summary_attempt002"/"primary_history_repeatability_weight.json",
            "blend": root/"primary_history_blend_summary_attempt001"/"primary_history_blend.json",
        }
        summaries = {name: json.loads(path.read_text()) for name, path in summary_paths.items()}
        frozen_rows = {}
        for name in COMPONENTS:
            old = summaries[name]
            panels = [p for p in old["panels"] if p["recipe"] == COMPONENTS[name]]
            if old["status"] != SUMMARY_STATUSES[name] or len(panels) != 1:
                raise ValueError("Complete original W/HW selected summaries required")
            frozen_rows[name] = panels[0]["sessions"]
        blend_summary = summaries["blend"]
        if (blend_summary["status"] != "complete_history_blend_panel"
                or blend_summary["all_selected_prediction_scores_replayed"] is not True):
            raise ValueError("Complete original selected blend summary required")
        frozen_rows["blend"] = blend_summary["blend"]["sessions"]
        for name, rows in frozen_rows.items():
            if len(rows) != 16 or {r["session_id"] for r in rows} != set(SESSIONS):
                raise ValueError("All sixteen original selected rows required")
            frozen_rows[name] = {r["session_id"]: r for r in rows}
        session_rows, direction_bin_rows, bin_rows = [], [], []
        for session in SESSIONS:
            y, baseline, labels, ids, banks, _, _ = load_components(roots["W"], roots["HW"], session)
            predictions = {}
            for name in COMPONENTS:
                row = frozen_rows[name][session]
                if row["lambda"] != 1.0 or row["lambda_index"] != 3:
                    raise ValueError("Frozen W/HW selector differs from reported lambda1")
                predictions[name] = banks[name][row["lambda_index"]]
            row = frozen_rows["blend"][session]
            if row["lambda"] != 1.0 or row["alpha"] != .5 or row["lambda_index"] != 3:
                raise ValueError("Frozen original blend differs from reported lambda1/alpha0.5")
            selected = summary_paths["blend"].parent/"selected_predictions"/(session+".npz")
            with np.load(selected, allow_pickle=False) as saved:
                for key, actual in (("actual_xy", y), ("baseline_xy", baseline),
                                    ("directions", labels), ("trial_ids", ids)):
                    if not np.array_equal(saved[key], actual):
                        raise ValueError("Saved blend must share exact original support: "+key)
                if float(saved["selected_lambda"]) != 1.0 or float(saved["selected_alpha"]) != .5:
                    raise ValueError("Stored blend selection metadata changed")
                predictions["blend"] = saved["candidate_xy"].copy()
            if not np.array_equal(predictions["blend"], blend_prediction(predictions["W"], predictions["HW"], .5)):
                raise ValueError("Literal saved half-blend must replay")
            for name, prediction in predictions.items():
                scores = prediction_scores(y, baseline, labels, prediction)
                if any(scores[key] != frozen_rows[name][session][key] for key in SCORE_KEYS):
                    raise ValueError("Original selected score must replay exactly: "+name+"/"+session)
                totals, detail, _ = decompose(y, baseline, prediction, labels)
                identity = {"session_id": session, "animal": ANIMALS[session], "recipe": name}
                session_rows.append({**identity, **totals})
                direction_bin_rows.extend({**identity, **item} for item in detail)
                for tick in range(15):
                    cell = [item for item in detail if item["bin"] == tick]
                    bin_rows.append({**identity, "bin": tick, "time_ms": tick*30,
                                     **{key: float(sum(item[key] for item in cell)) for key in TERMS}})
        animal_rows, aggregate_rows, contrast_rows = [], [], []
        for name in ("W", "HW", "blend"):
            per_animal = []
            for animal in ("Mihili", "Chewie"):
                rows = [r for r in session_rows if r["recipe"] == name and r["animal"] == animal]
                summary = {"animal": animal, "recipe": name, "sessions": len(rows), **average(rows)}
                animal_rows.append(summary)
                per_animal.append(summary)
            aggregate_rows.append({"recipe": name, **average(per_animal)})
        for session in SESSIONS:
            rows = {r["recipe"]: r for r in session_rows if r["session_id"] == session}
            for first, second in (("HW", "W"), ("blend", "W"), ("blend", "HW")):
                contrast_rows.append({"session_id": session, "animal": ANIMALS[session],
                    "contrast": first+"_minus_"+second,
                    **{key: rows[first][key]-rows[second][key] for key in TERMS}})
        report.update(status="complete_descriptive_decomposition",
            original_summary_paths={name: str(path) for name, path in summary_paths.items()},
            original_selected_scores_replayed_exactly=True, decomposition_identities_passed=True,
            animal_balanced=aggregate_rows, animal_summaries=animal_rows,
            sessions=session_rows, session_contrasts=contrast_rows,
            direction_bin_rows=len(direction_bin_rows), bin_rows=len(bin_rows))
        for filename, rows in (("sessions.csv", session_rows), ("animal_summaries.csv", animal_rows),
                               ("session_contrasts.csv", contrast_rows), ("bins.csv", bin_rows),
                               ("direction_bins.csv", direction_bin_rows)):
            write_csv(out/filename, rows)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        (out/"decomposition.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "scientific_trial_count_increment")}))


if __name__ == "__main__":
    main()
