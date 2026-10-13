"""Frozen stationary LMP neural-correction offsets; no model fits or source IO."""
import numpy as np

from primary_reference import ANIMALS, LAMBDAS, SESSIONS, wrong_pair_sse
from primary_temporal import aggregate_rows, directional_screens

ANCHOR = "lmp_full_channel_ridge_v1"
RECIPE = "lmp_fixed_residual_circular_offsets14_v1"
OFFSETS = tuple(range(1, 15))
SCORE_KEYS = ("candidate_R2", "candidate_SSE", "wrong_pair_SSE", "delta_R2", "identity_increment")


def offset_prediction(prediction, baseline, offset):
    prediction, baseline = np.asarray(prediction), np.asarray(baseline)
    if (type(offset) is not int or offset not in range(15) or prediction.ndim != 3
            or prediction.shape[1:] != (15, 2) or baseline.shape != prediction.shape
            or not np.isfinite(prediction).all() or not np.isfinite(baseline).all()):
        raise ValueError("Finite complete XY trajectories and offset0..14 required")
    # Zero is reference replay, with no subtraction/addition rounding.
    if offset == 0:
        return prediction.copy()
    donor_time = (np.arange(15)+offset) % 15
    return baseline+(prediction-baseline)[:, donor_time, :]


def prediction_scores(actual, prediction, labels, baseline_sse, denominator):
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Original positive finite joint XY denominator required")
    correct = float(np.square(actual-prediction).sum())
    wrong = wrong_pair_sse(actual, prediction, labels)
    return {"candidate_R2": 1-correct/denominator, "candidate_SSE": correct,
            "wrong_pair_SSE": wrong, "delta_R2": (baseline_sse-correct)/denominator,
            "identity_increment": (wrong-correct)/denominator}


def frozen_selection(session, selected, bank):
    """Use the stored original choice; never select lambda from control outcomes."""
    index = selected["lambda_index"]
    if (session not in SESSIONS or selected["session_id"] != session
            or selected["animal"] != ANIMALS[session]
            or type(index) is not int or index not in range(len(LAMBDAS))
            or selected["lambda"] != LAMBDAS[index]
            or selected["outer_training_sessions"] != [s for s in SESSIONS if s != session]
            or bank["recipe"] != ANCHOR or bank["lambdas"] != list(LAMBDAS)
            or [r["lambda"] for r in bank["lambda_rows"]] != list(LAMBDAS)):
        raise ValueError("Unchanged original LMP selection and outer-session exclusion required")
    for key in SCORE_KEYS:
        if selected[key] != bank["lambda_rows"][index][key]:
            raise ValueError("Preserved selected anchor score row differs; no retuning")
    return index


def replay_selected_anchor(actual, baseline, prediction, labels, bank, selected):
    actual, baseline, prediction, labels = (np.asarray(x) for x in (actual, baseline, prediction, labels))
    if (actual.ndim != 3 or actual.shape[1:] != (15, 2) or baseline.shape != actual.shape
            or prediction.shape != actual.shape or labels.shape != (len(actual),)
            or set(labels.tolist()) != set(range(8))
            or not all(np.isfinite(x).all() for x in (actual, baseline, prediction))):
        raise ValueError("Complete original native XY prediction/target/baseline support required")
    counts = [int(np.count_nonzero(labels == d)) for d in range(8)]
    if counts != bank["direction_counts"] or min(counts) < 2:
        raise ValueError("Original direction counts differ")
    denominator = float(np.square(actual-actual.mean(axis=(0, 1), keepdims=True)).sum())
    baseline_sse = float(np.square(actual-baseline).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Original denominator must be positive")
    replay = {"denominator": denominator, "baseline_SSE": baseline_sse,
              "baseline_R2": 1-baseline_sse/denominator}
    replay.update(prediction_scores(actual, prediction, labels, baseline_sse, denominator))
    for key, value in replay.items():
        expected = bank[key] if key in ("denominator", "baseline_SSE", "baseline_R2") else selected[key]
        if not np.isfinite(expected) or not np.isclose(value, expected, rtol=1e-12, atol=1e-12):
            raise ArithmeticError(f"Selected original anchor {key} replay differs; no offset scoring")
    return replay


def score_session(actual, baseline, prediction, labels, bank, selected, check=lambda: None):
    replay = replay_selected_anchor(actual, baseline, prediction, labels, bank, selected)
    if not np.array_equal(offset_prediction(prediction, baseline, 0), prediction):
        raise ArithmeticError("Zero offset failed exact prediction identity")
    rows = []
    for offset in OFFSETS:
        check()
        shifted = offset_prediction(prediction, baseline, offset)
        rows.append({"offset_bins": offset, "donor_time_indices": ((np.arange(15)+offset) % 15).tolist(),
            **prediction_scores(actual, shifted, labels, replay["baseline_SSE"], replay["denominator"])})
    correct = float(np.mean([r["candidate_SSE"] for r in rows]))
    wrong = float(np.mean([r["wrong_pair_SSE"] for r in rows]))
    return {"recipe": RECIPE, "session_id": selected["session_id"], "animal": selected["animal"],
        "lambda": selected["lambda"], "lambda_index": selected["lambda_index"],
        "outer_training_sessions": selected["outer_training_sessions"],
        "original_inner_animal_balanced_scores": selected["inner_animal_balanced_scores"],
        "offsets": list(OFFSETS), "offset_rows": rows, "direction_counts": bank["direction_counts"],
        "baseline_R2": replay["baseline_R2"], "baseline_SSE": replay["baseline_SSE"],
        "denominator": replay["denominator"], "candidate_R2": 1-correct/replay["denominator"],
        "candidate_SSE": correct, "wrong_pair_SSE": wrong,
        "delta_R2": (replay["baseline_SSE"]-correct)/replay["denominator"],
        "identity_increment": (wrong-correct)/replay["denominator"],
        "anchor_replay": replay, "original_selected_anchor": selected,
        "zero_offset_exact_prediction_replay": True, "new_fits": 0,
        "aggregation": "exact equal mean of14 offset correct/wrong SSEs; not an ensemble"}


def summarize_sessions(sessions):
    if set(sessions) != set(SESSIONS):
        raise ValueError("All sixteen fixed sessions required for timing-control completion")
    rows = []
    for session in SESSIONS:
        row = sessions[session]
        if (row["recipe"] != RECIPE or row["session_id"] != session or row["animal"] != ANIMALS[session]
                or row["offsets"] != list(OFFSETS)
                or [r["offset_bins"] for r in row["offset_rows"]] != list(OFFSETS)
                or not all(np.isfinite(row[k]) for k in SCORE_KEYS)):
            raise ValueError("Complete finite fixed-offset score support required")
        rows.append(row)
    control = {"recipe": RECIPE, **aggregate_rows(rows), "scientific_trial_count_increment": 1,
        "new_fits": 0, "promotion_authorized": False, "search_complete": False,
        "evidence": "exposed exploratory development"}
    anchor = aggregate_rows([r["original_selected_anchor"] for r in rows])
    contrast_rows = [{"session_id": r["session_id"], "animal": r["animal"], "lambda": r["lambda"],
        "delta_R2": r["original_selected_anchor"]["delta_R2"]-r["delta_R2"],
        "identity_increment": r["original_selected_anchor"]["identity_increment"]-r["identity_increment"]}
        for r in rows]
    contrast = aggregate_rows(contrast_rows)
    screens = directional_screens(contrast)
    contrast.update(contrast="fixed aligned LMP minus exact mean of14 nonzero neural-residual offsets",
        directional_timing_screens=screens, all_directional_timing_screens_pass=all(screens.values()),
        scientific_trial_count_increment=0, promotion_authorized=False)
    return {"control": control, "reference_anchor": anchor, "aligned_minus_offset": contrast,
        "offsets": list(OFFSETS), "new_fits": 0, "new_source_values_read": False,
        "scientific_trial_count_increment": 1, "promotion_authorized": False, "search_complete": False,
        "interpretation": "within-trial alignment; circular wrap/phase sensitivity, not a physiological lead or causal test"}
