#!/usr/bin/env python3
"""One adopted development-only scalar calibration of frozen HW predictions."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from decompose_fixed_lfp_predictions import assert_roundoff, decompose, write_csv
from primary_elapsed_only import full_model_contrast, panel
from primary_history_blend import COMPONENTS, SCORE_KEYS, blend_prediction, prediction_scores
from primary_reference import ANIMALS, SESSIONS, animal_balanced
from run_primary_history_blend import SUMMARY_STATUSES, load_components
from source_adapter import compute_only

AMPLITUDES = (0., .25, .5, .75, 1.)
ANALYSIS = "fixed_hw_correction_calibration"


def calibrated_prediction(baseline, history, amplitude):
    if (amplitude not in AMPLITUDES or baseline.shape != history.shape
            or baseline.ndim != 3 or baseline.shape[1:] != (15, 2)
            or not np.isfinite(baseline).all() or not np.isfinite(history).all()):
        raise ValueError("Frozen finite native-XY arrays and declared amplitude required")
    if amplitude == 0:
        return baseline
    if amplitude == 1:
        return history
    return baseline+amplitude*(history-baseline)


def score_amplitudes(y, baseline, history, labels):
    return [{"amplitude": amplitude,
             **prediction_scores(y, baseline, labels,
                                 calibrated_prediction(baseline, history, amplitude))}
            for amplitude in AMPLITUDES]


def select_amplitude(banks, held):
    """Only other-session primary scores enter calibration; identity never does."""
    if set(banks) != set(SESSIONS) or held not in SESSIONS:
        raise ValueError("All sixteen original sessions required before calibration")
    for rows in banks.values():
        if ([row["amplitude"] for row in rows] != list(AMPLITUDES)
                or not all(np.isfinite(row["delta_R2"]) for row in rows)):
            raise ValueError("Complete ordered finite five-amplitude score bank required")
    training = [session for session in SESSIONS if session != held]
    scores = [animal_balanced({s: banks[s][index]["delta_R2"] for s in training})[0]
              for index in range(len(AMPLITUDES))]
    index = max(range(len(AMPLITUDES)), key=lambda i: (scores[i], -abs(1-AMPLITUDES[i])))
    return index, training, scores


def load_original_references(root):
    paths = {
        "HW": root/"primary_history_repeatability_weight_summary_attempt002"/"primary_history_repeatability_weight.json",
        "blend": root/"primary_history_blend_summary_attempt001"/"primary_history_blend.json",
    }
    hw = json.loads(paths["HW"].read_text())
    panels = [p for p in hw["panels"] if p["recipe"] == COMPONENTS["HW"]]
    blend = json.loads(paths["blend"].read_text())
    if (hw["status"] != SUMMARY_STATUSES["HW"] or len(panels) != 1
            or blend["status"] != "complete_history_blend_panel"
            or blend["all_selected_prediction_scores_replayed"] is not True):
        raise ValueError("Complete immutable HW and blend references required")
    rows = {"HW": panels[0]["sessions"], "blend": blend["blend"]["sessions"]}
    for name, items in rows.items():
        if len(items) != 16 or {r["session_id"] for r in items} != set(SESSIONS):
            raise ValueError("All original reference sessions required")
        rows[name] = {r["session_id"]: r for r in items}
        for row in items:
            if (row["lambda"] != 1. or row["lambda_index"] != 3
                    or (name == "blend" and row["alpha"] != .5)):
                raise ValueError("References must retain original lambda1 and half-blend")
    return rows, paths


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
        raise ValueError("Adopted calibration has a maximum 180-second internal bound")
    outputs = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if not out.is_relative_to(outputs/"runs"/ANALYSIS):
        raise ValueError("Use a fresh episode-local calibration attempt")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "analysis_id": ANALYSIS, "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
        "amplitude_grid": list(AMPLITUDES), "frozen_HW_lambda": 1.,
        "tie_rule": "maximum other-fifteen-session equal-animal primary, then amplitude nearest1",
        "provider_source_values_read": False, "neural_refits": 0,
        "new_scalar_calibration": True, "future_session_access": False,
        "original_comparison_count": 45, "original_scientific_trial_count_increment": 0,
        "successor_comparison_count": 0, "original_stop_preserved": True,
        "primary_margin": .01, "identity_margin": .005,
        "independent_penalty_amendment_adopted": False,
        "promotion_authorized": False,
        "evidence": "one exploratory development-only calibration comparison on previously exposed sessions",
        "claim_limit": "A scalar learned without each scored session does not undo prior corpus exposure. "
            "Calibration can diagnose correction magnitude, not physiology, neural noise or new-animal transfer.",
    }
    def deadline(*_):
        raise TimeoutError("Adopted 180-second calibration deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        root = outputs/"runs"/"adaptive_search"
        references, summary_paths = load_original_references(root)
        banks, arrays, denominators = {}, {}, {}
        for session in SESSIONS:
            y, b, labels, ids, predictions, _, _ = load_components(
                root/"primary_repeatability_weight_control_attempt001",
                root/"primary_history_repeatability_weight_control_attempt002", session)
            hw = predictions["HW"][3]
            selected = summary_paths["blend"].parent/"selected_predictions"/(session+".npz")
            with np.load(selected, allow_pickle=False) as saved:
                for key, expected in (("actual_xy", y), ("baseline_xy", b),
                                      ("directions", labels), ("trial_ids", ids)):
                    if not np.array_equal(saved[key], expected):
                        raise ValueError("Original blend support changed: "+key)
                if float(saved["selected_lambda"]) != 1. or float(saved["selected_alpha"]) != .5:
                    raise ValueError("Original saved blend selection changed")
                blend = saved["candidate_xy"].copy()
            if not np.array_equal(blend, blend_prediction(predictions["W"][3], hw, .5)):
                raise ValueError("Original half-blend literal prediction failed replay")
            for name, prediction in (("HW", hw), ("blend", blend)):
                scores = prediction_scores(y, b, labels, prediction)
                if any(scores[key] != references[name][session][key] for key in SCORE_KEYS):
                    raise ValueError("Original selected score failed exact replay: "+name+"/"+session)
            banks[session] = score_amplitudes(y, b, hw, labels)
            if any(banks[session][-1][key] != references["HW"][session][key] for key in SCORE_KEYS):
                raise ValueError("Amplitude1 must exactly replay frozen HW")
            if banks[session][0]["delta_R2"] != 0.:
                raise ValueError("Amplitude0 must be the literal baseline")
            denominators[session] = float(np.square(y-y.mean(axis=(0, 1), keepdims=True)).sum())
            arrays[session] = (y, b, labels, ids, hw, blend)
        selected_rows, contrasts = [], {name: [] for name in references}
        selected_arrays, bin_rows, decomposition_rows = {}, [], []
        duplicate = {"HW": True, "blend": True, "baseline": True}
        for session in SESSIONS:
            index, training, training_scores = select_amplitude(banks, session)
            amplitude = AMPLITUDES[index]
            y, b, labels, ids, hw, blend = arrays[session]
            prediction = calibrated_prediction(b, hw, amplitude)
            score = prediction_scores(y, b, labels, prediction)
            if any(score[key] != banks[session][index][key] for key in SCORE_KEYS):
                raise ValueError("Selected calibration score changed")
            identity = {"session_id": session, "animal": ANIMALS[session]}
            selected_rows.append({**identity, **banks[session][index],
                "outer_training_sessions": training, "training_primary_scores": training_scores})
            for name in references:
                contrasts[name].append({**identity,
                    **full_model_contrast(score, references[name][session], denominators[session])})
            totals, detail, _ = decompose(y, b, prediction, labels)
            assert_roundoff(score["identity_increment"],
                            amplitude*references["HW"][session]["identity_increment"], "scaled identity")
            decomposition_rows.append({**identity, "amplitude": amplitude, **totals})
            bin_rows.extend({**identity, "amplitude": amplitude, **row} for row in detail)
            for name, anchor in (("HW", hw), ("blend", blend), ("baseline", b)):
                duplicate[name] = duplicate[name] and np.array_equal(prediction, anchor)
            selected_arrays[session] = prediction
        result = panel(selected_rows)
        report.update(status="complete_fixed_hw_correction_calibration",
            successor_comparison_count=1, original_selected_scores_replayed_exactly=True,
            original_summary_paths={key: str(path) for key, path in summary_paths.items()},
            candidate=result, reference_contrasts={name: panel(rows) for name, rows in contrasts.items()},
            references={name: panel([rows[s] for s in SESSIONS]) for name, rows in references.items()},
            selected_amplitudes={r["session_id"]: r["amplitude"] for r in selected_rows},
            all_five_cell_banks=banks, literal_global_duplicate_of=[k for k, value in duplicate.items() if value],
            original_practical_margin_screens={
                "primary_at_least_0p01": result["primary_increment"] >= .01,
                "identity_at_least_0p005": result["identity_increment"] >= .005},
            calibration_does_not_certify_other_finalist_controls=True,
            decomposition_identities_passed=True, fixed_prediction_omissions_do_not_recalibrate=True)
        selected_dir = out/"selected_predictions"
        selected_dir.mkdir(exist_ok=False)
        for row in selected_rows:
            session = row["session_id"]
            y, b, labels, ids, _, _ = arrays[session]
            np.savez(selected_dir/(session+".npz"), actual_xy=y, baseline_xy=b,
                     directions=labels, trial_ids=ids, candidate_xy=selected_arrays[session],
                     selected_amplitude=row["amplitude"], frozen_lambda=1.)
        write_csv(out/"sessions.csv", [{k: row[k] for k in ("session_id", "animal", "amplitude", *SCORE_KEYS)}
                                       for row in selected_rows])
        write_csv(out/"decomposition_sessions.csv", decomposition_rows)
        write_csv(out/"direction_bins.csv", bin_rows)
        for name, rows in contrasts.items():
            write_csv(out/("contrast_"+name+".csv"), rows)
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}", successor_comparison_count=0)
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
                      maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        (out/"calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "successor_comparison_count")}))


if __name__ == "__main__":
    main()
