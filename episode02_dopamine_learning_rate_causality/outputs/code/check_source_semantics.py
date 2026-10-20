#!/usr/bin/env python3
"""Magnitude-blind source-semantics checks on the nine EP02 development controls.

Candidate-role program for the October 16 plan in
`outputs/source_semantics_design.md`. It reads only `trialID`, `seshID`,
`lickState` and `lick` through `development_role_reader.py`, after admitting
exactly the steward-released pack root. It never opens the provider MAT, an
audit record or another field.

Outputs are limited to shapes, dtypes, value sets, code/row/missing counts,
eligible denominators, agreement fractions with off-diagonal counts, onset
indices and agreement-by-alignment profiles. No session lick probability,
learning gain, development centre or scale, and no lick time course, is
written. Sample indices are zero-based; `R` denotes the reward sample.

Stages: `structure` (formats and counts), `format` (single-lick kernel of the
continuous trace, normalized shapes only) and `semantics` (prespecified tests,
with the lick predicate re-specified from the format stage before any agreement test).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import development_role_reader as reader  # noqa: E402

SAMPLES = 701
CUE_OFFSET = 150  # 1.5-s cue-to-reward interval at 100 Hz
WINDOWS = {  # name: (start offset, end offset) relative to R, inclusive
    "w50": (-50, -1),
    "w75": (-75, -1),
    "w100": (-100, -1),
    "w150": (-150, -1),
    "w75_incl": (-75, 0),
    "w75_shift": (-74, 0),
}
POST_REWARD = (0, 199)
MAX_CODES = 64


def admission(released: Path):
    def admit(root: Path) -> None:
        parts = {part.lower() for part in root.parts}
        if root != released or root.name != "development_full" or parts & {"evaluator", "vault", ".evaluator_vault"}:
            raise PermissionError("only the steward-released development_full root is admitted")
        return None
    return admit


def budget(seconds: float):
    start = time.monotonic()

    def check() -> None:
        if time.monotonic() - start > seconds:
            raise TimeoutError("semantics-check time budget exhausted")
    return check


def vector(array, label):
    value = np.asarray(array)
    if value.ndim == 2 and 1 in value.shape:
        return value.reshape(-1)
    if value.ndim == 1:
        return value
    raise ValueError(f"{label} is not a vector: shape {value.shape}")


def trace(array, rows):
    value = np.asarray(array)
    if value.shape == (rows, SAMPLES):
        return value
    if value.shape == (SAMPLES, rows):
        return value.T
    raise ValueError(f"lick shape {value.shape} does not match {rows} rows x {SAMPLES} samples")


def scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load(mouse):
    arrays = mouse.arrays
    trial = vector(arrays["trialID"], "trialID")
    rows = trial.size
    sesh = vector(arrays["seshID"], "seshID")
    state = vector(arrays["lickState"], "lickState")
    if not (sesh.size == state.size == rows):
        raise ValueError("trialID, seshID and lickState lengths differ")
    lick = trace(arrays["lick"], rows)
    return (np.array(trial, dtype=np.float64), np.array(sesh, dtype=np.float64),
            np.array(state, dtype=np.float64), np.array(lick, dtype=np.float64))


def each_mouse(pack):
    for index in reader.DEVELOPMENT_RECORDS:
        with pack.open_mouse(index) as mouse:
            shapes = {field: [list(np.asarray(mouse.arrays[field]).shape), str(np.asarray(mouse.arrays[field]).dtype)]
                      for field in reader.FIELDS}
            yield index, shapes, load(mouse)


def structure(pack):
    out = {}
    for index, shapes, (trial, sesh, state, lick) in each_mouse(pack):
        codes = np.unique(trial[np.isfinite(trial)])
        sessions = np.unique(sesh[np.isfinite(sesh)])
        finite_lick = np.isfinite(lick)
        finite_values = np.unique(lick[finite_lick])
        record = {
            "rows": int(trial.size),
            "shapes_dtypes": shapes,
            "trialID": {"distinct": int(codes.size), "nonfinite": int((~np.isfinite(trial)).sum())},
            "seshID": {
                "distinct_values": [scalar(v) for v in sessions],
                "integer_valued": bool(np.all(np.mod(sessions, 1) == 0)),
                "nonfinite": int((~np.isfinite(sesh)).sum()),
                "order_reversals": int((np.diff(sesh) < 0).sum()),
                "rows_per_session": {str(scalar(s)): int((sesh == s).sum()) for s in sessions},
            },
            "lickState": {
                "distinct_finite_values": [scalar(v) for v in np.unique(state[np.isfinite(state)])][:10],
                "distinct_finite_count": int(np.unique(state[np.isfinite(state)]).size),
                "nonfinite": int((~np.isfinite(state)).sum()),
            },
            "lick": {
                "distinct_finite_count": int(finite_values.size),
                "distinct_finite_values_if_at_most_12": [scalar(v) for v in finite_values] if finite_values.size <= 12 else None,
                "finite_min": scalar(finite_values.min()) if finite_values.size else None,
                "finite_max": scalar(finite_values.max()) if finite_values.size else None,
                "integer_valued": bool(np.all(np.mod(finite_values, 1) == 0)),
                "rows_any_nonfinite": int((~finite_lick).any(axis=1).sum()),
                "rows_all_nonfinite": int((~finite_lick).all(axis=1).sum()),
                "rows_all_nonfinite_per_session": {str(scalar(s)): int((~finite_lick[sesh == s]).all(axis=1).sum()) for s in sessions},
            },
        }
        if codes.size <= MAX_CODES:
            record["trialID"]["values"] = [scalar(v) for v in codes]
            record["trialID"]["counts_per_session"] = {
                str(scalar(s)): {str(scalar(c)): int(((sesh == s) & (trial == c)).sum()) for c in codes} for s in sessions}
            record["lickState"]["nonfinite_per_code"] = {
                str(scalar(c)): int(((trial == c) & ~np.isfinite(state)).sum()) for c in codes}
        out[str(index)] = record
    return {"stage": "structure", "records": out}


def label_codes(mice):
    """Frequency rule (re-specified after the structure stage, before any agreement test) on sessions 1-8 pooled across mice."""
    totals, per_session = {}, {s: {} for s in range(1, 9)}
    for trial, sesh, _, _ in mice.values():
        for code in np.unique(trial[np.isfinite(trial)]):
            totals[code] = totals.get(code, 0) + int((trial == code).sum())
            for s in range(1, 9):
                per_session[s][code] = per_session[s].get(code, 0) + int(((sesh == s) & (trial == code)).sum())
    rows = {s: sum(per_session[s].values()) for s in per_session}
    frac = {code: [per_session[s].get(code, 0) / rows[s] if rows[s] else float("nan") for s in range(1, 9)] for code in totals}
    cued = max(totals, key=totals.get)
    uncued = [c for c in totals if c != cued and all(0.05 <= f <= 0.15 for f in frac[c])]
    omission = [c for c in totals if c != cued and all(f <= 0.01 for f in frac[c][:3]) and all(0.05 <= f <= 0.15 for f in frac[c][3:])]
    labels = {"cued_rewarded": scalar(cued),
              "uncued": scalar(uncued[0]) if len(uncued) == 1 else None,
              "omission": scalar(omission[0]) if len(omission) == 1 else None}
    fractions = {str(scalar(c)): [round(f, 4) for f in v] for c, v in frac.items()}
    return labels, fractions


def local_maxima(x):
    """Right end of every local maximum or plateau, per row."""
    peak = np.zeros(x.shape, dtype=bool)
    peak[:, 1:-1] = (x[:, 1:-1] >= x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])
    peak[:, 0] = x[:, 0] > x[:, 1]
    peak[:, -1] = x[:, -1] >= x[:, -2]
    return peak


def rises(x, theta):
    """Lick onsets of the causal rate trace: a one-sample rise of at least theta."""
    onset = np.zeros(x.shape, dtype=bool)
    onset[:, 1:] = np.diff(x, axis=1) >= theta
    return onset


def format_stage(pack, floor=1.0, isolation=20):
    """Single-lick kernel of the continuous trace: normalized shapes only, no counts."""
    heights, isolated_heights, shapes = [], [], []
    for _, _, (_, _, _, lick) in each_mouse(pack):
        x = lick[np.isfinite(lick).all(axis=1)]
        peak = local_maxima(x) & (x >= floor)
        rows, cols = np.nonzero(peak)
        heights.append(x[rows, cols])
        for row in np.unique(rows):
            c = cols[rows == row]
            gaps = np.diff(c)
            before = np.concatenate([[np.inf], gaps])
            after = np.concatenate([gaps, [np.inf]])
            for col in c[(before > isolation) & (after > isolation) & (c >= isolation) & (c <= SAMPLES - 1 - isolation)]:
                isolated_heights.append(x[row, col])
                shapes.append(x[row, col - isolation:col + isolation + 1] / x[row, col])
    heights = np.concatenate(heights)
    isolated = np.asarray(isolated_heights)
    shape = np.mean(shapes, axis=0)
    edges = np.arange(floor, 30.5, 0.5)
    counts, _ = np.histogram(np.clip(heights, floor, 30.0), bins=edges)
    modal, modal_edges = np.histogram(isolated, bins=np.arange(floor, 30.25, 0.25))
    mode = float(modal_edges[int(np.argmax(modal))] + 0.125)
    above = shape >= 0.5
    left = int(np.argmax(above))
    right = int(len(shape) - 1 - np.argmax(above[::-1]))
    return {
        "stage": "format", "peak_floor": floor, "isolation_samples": isolation,
        "at_least_500_isolated_peaks": bool(isolated.size >= 500),
        "all_peak_height_fraction_by_0p5_bin_from_floor": [round(float(v), 4) for v in counts / counts.sum()],
        "isolated_peak_height_quartiles": [round(float(v), 3) for v in np.percentile(isolated, [25, 50, 75])],
        "isolated_peak_height_mode": round(mode, 3),
        "mean_normalized_isolated_shape_offsets_minus20_to_20": [round(float(v), 4) for v in shape],
        "half_maximum_width_samples": right - left + 1,
        "shape_asymmetry_left_minus_right_area": round(float(shape[:isolation].sum() - shape[isolation + 1:].sum()), 4),
    }


def predicate_matrix(cumulative, start, end):
    """any lick in [R+start, R+end] for every admissible R, as rows x R."""
    first = max(0, -start)
    last = SAMPLES - 1 - max(0, end)
    r = np.arange(first, last + 1)
    counts = cumulative[:, r + end + 1] - cumulative[:, r + start]
    return r, counts > 0


def crossings(curve, fractions=(0.1, 0.5)):
    smooth = np.convolve(curve, np.ones(5) / 5, mode="same")
    peak = int(np.argmax(smooth))
    baseline = float(np.median(smooth[: max(min(100, peak - 20), 5)]))
    rise = float(smooth[peak] - baseline)
    result = {"peak_index": peak}
    for fraction in fractions:
        below = np.nonzero(smooth[: peak + 1] < baseline + fraction * rise)[0]
        result[f"cross_{int(fraction * 100)}"] = int(below[-1] + 1) if below.size else 0
    return result


def admissible(name, R):
    start, end = WINDOWS[name]
    return R + start >= 0 and R + end + 1 <= SAMPLES


def predicate(cumulative, name, R):
    start, end = WINDOWS[name]
    if not admissible(name, R):
        raise ValueError(f"window {name} is outside the axis at R={R}")
    return (cumulative[:, R + end + 1] - cumulative[:, R + start]) > 0


def evaluate(rows, codes):
    """Scan, table, discordance and agreement summaries for one lick predicate."""
    observed_all = np.concatenate([v[0] for v in rows.values()])
    cumulative_all = np.concatenate([v[1] for v in rows.values()])
    trial_all = np.concatenate([v[2] for v in rows.values()])
    usable = int(observed_all.size)
    best, profiles = {}, {}
    for name, (start, end) in WINDOWS.items():
        r, matrix = predicate_matrix(cumulative_all, start, end)
        fraction = (matrix == observed_all[:, None]).mean(axis=0)
        top = int(np.argmax(fraction))
        best[name] = {"R": int(r[top]), "agreement": round(float(fraction[top]), 5),
                      "tied_R": [int(x) for x in r[fraction == fraction[top]]][:10]}
        profiles[name] = {"R_first": int(r[0]), "agreement": [round(float(x), 5) for x in fraction]}
    r_star = best["w75"]["R"]
    r, matrix = predicate_matrix(cumulative_all, *WINDOWS["w75"])
    by_code = {}
    for code in codes:
        mask = trial_all == code
        fraction = (matrix[mask] == observed_all[mask, None]).mean(axis=0)
        by_code[str(code)] = {"R": int(r[int(np.argmax(fraction))]), "agreement": round(float(fraction.max()), 5),
                              "trials": int(mask.sum())}
    primary = predicate(cumulative_all, "w75", r_star)
    table, discord = {}, {}
    for name in WINDOWS:
        if not admissible(name, r_star):
            table[name] = None
            continue
        p = predicate(cumulative_all, name, r_star)
        table[name] = {"trials": int(p.size), "agreement": round(float((p == observed_all).mean()), 5),
                       "state1_video0": int((observed_all & ~p).sum()), "state0_video1": int((~observed_all & p).sum())}
        if name != "w75":
            d = p != primary
            discord[name] = {"discordant_trials": int(d.sum()),
                             "fraction_state_sides_with_w75": round(float((observed_all[d] == primary[d]).mean()), 5) if d.any() else None}
    mouse_agreement = {str(index): round(float((predicate(v[1], "w75", r_star) == v[0]).mean()), 5) for index, v in rows.items()}
    code_agreement = {str(code): round(float((primary[trial_all == code] == observed_all[trial_all == code]).mean()), 5) for code in codes}
    return {"usable_trials": usable, "best_by_window": best, "scan_profiles": profiles, "best_R_w75_by_code": by_code,
            "R_star": r_star, "table_at_R_star": table, "discordance_vs_w75": discord,
            "agreement_w75_by_mouse": mouse_agreement, "agreement_w75_by_code": code_agreement}


def semantics(pack, theta):
    mice = {}
    for index, _, arrays in each_mouse(pack):
        mice[index] = arrays
    labels, fractions = label_codes(mice)
    result = {"stage": "semantics", "theta": theta, "code_labels": labels, "code_fraction_sessions_1_to_8": fractions}
    if None in labels.values():
        result["stopped"] = "code labels not unique under the re-specified rule"
        return result
    values = set()
    for _, _, state, _ in mice.values():
        values |= set(np.unique(state[np.isfinite(state)]).tolist())
    if not values <= {0.0, 1.0}:
        result["stopped"] = {"lickState_values": sorted(values)[:10], "reason": "lickState is not a 0/1 state"}
        return result
    cued, uncued, omission = labels["cued_rewarded"], labels["uncued"], labels["omission"]
    codes = (cued, uncued, omission)
    families = {"onset": lambda x: rises(x, theta),
                "peak": lambda x: local_maxima(x) & (x >= theta), "threshold": lambda x: x >= theta}
    rows = {name: {} for name in families}
    for index, (trial, sesh, state, lick) in mice.items():
        keep = np.isfinite(state) & np.isfinite(lick).all(axis=1)
        for name, events_of in families.items():
            events = events_of(lick[keep]).astype(np.int64)
            cumulative = np.concatenate([np.zeros((events.shape[0], 1), np.int64), np.cumsum(events, axis=1)], axis=1)
            rows[name][index] = (state[keep] == 1.0, cumulative, trial[keep], sesh[keep])
    result["predicates"] = {name: evaluate(rows[name], codes) for name in families}

    r_star = result["predicates"]["onset"]["R_star"]
    cumulative_all = np.concatenate([v[1] for v in rows["onset"].values()])
    trial_all = np.concatenate([v[2] for v in rows["onset"].values()])
    sesh_all = np.concatenate([v[3] for v in rows["onset"].values()])
    events_all = np.diff(cumulative_all, axis=1) > 0
    later = sesh_all >= 4

    def probability(mask):
        return events_all[mask].mean(axis=0)

    e1 = crossings(probability(trial_all == uncued))
    e2 = crossings(probability((trial_all == cued) & later) - probability((trial_all == omission) & later))
    e3 = crossings((probability((trial_all == cued) & later) - probability((trial_all == uncued) & later))[:r_star]) if r_star > 10 else {}
    for marker in (e1, e2, e3):
        marker.update({f"{key}_minus_R": value - r_star for key, value in list(marker.items())})
    lo, hi = POST_REWARD[0], min(POST_REWARD[1], SAMPLES - 1 - r_star)
    window_counts = cumulative_all[:, r_star + hi + 1] - cumulative_all[:, r_star + lo]
    post = {str(code): round(float((window_counts[trial_all == code] > 0).mean()), 2) for code in codes}
    denominators = {}
    for index, (trial, sesh, state, lick) in mice.items():
        eligible = (trial == cued) & np.isfinite(state) & np.isfinite(lick).all(axis=1)
        denominators[str(index)] = {str(scalar(s)): int((eligible & (sesh == s)).sum()) for s in np.unique(sesh[np.isfinite(sesh)])}
    result.update({
        "R_star_primary_onset": r_star, "cue_index_implied": r_star - CUE_OFFSET,
        "E1_uncued_lick_rise": e1, "E2_rewarded_minus_omission_rise": e2,
        "E3_cued_minus_uncued_prereward_rise_descriptive": e3,
        "post_reward_window": [lo, hi], "post_reward_lick_presence_by_code": post,
        "eligible_cued_trials_by_session": denominators,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("structure", "format", "semantics"))
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=1200.0)
    parser.add_argument("--peak-threshold", type=float)
    args = parser.parse_args()
    released = Path(os.environ["EP02_RELEASED_DEVELOPMENT_ROOT"])
    if args.output.exists():
        raise FileExistsError(args.output)
    with reader.open_development_pack(args.development_root, admit_development_root=admission(released),
                                      maximum_mapped_bytes=64 * 1024 * 1024, check=budget(args.budget_seconds)) as pack:
        if args.stage == "structure":
            payload = structure(pack)
        elif args.stage == "format":
            payload = format_stage(pack)
        else:
            if args.peak_threshold is None:
                raise ValueError("semantics requires --peak-threshold from the format stage")
            payload = semantics(pack, args.peak_threshold)
    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=args.output.parent, prefix=".tmp.")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
