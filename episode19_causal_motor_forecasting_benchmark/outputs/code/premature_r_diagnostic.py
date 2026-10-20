#!/usr/bin/env python3
"""Fixed-label EP19 premature-R diagnostic; orchestrator execution only."""
from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import numpy as np
import detector_v2 as D
from detector_v2 import R, RF, V4, F
import self_paced_raw_onset_test_v5 as V5

ROOT = CODE.parent.parent
VIEWS = ROOT / "outputs/runs/source_views_and_timing"
DETECTOR = VIEWS / "detector_v2_job_48261284"
V5RUN = VIEWS / "raw_onset_test_v5_job_48261284"
FREEZE = VIEWS / "detector_freeze_job_48219799"
SEED = 20261017
NEW_SCRIPTS = (
    "premature_r_diagnostic.py", "control_precision_check.py",
    "plot_v5_diagnostics.py", "run_v5_diagnostics.sbatch",
)
_HASHES = None

DECLARATION = """\
### October 17: premature-R diagnostic and control-precision check — pre-declared before execution

This follow-up is post hoc to v5. Added synthetic shapes, cutoffs, coverage requirements and operational choices are assumptions. It asks whether the v5 tail of long lags (L − R ≥ 288 ms) comes from a premature reference onset R produced by low-amplitude activity or drift (H_A), or from a detector that misses substantial movement before its label L (H_B). Accelerometer continuity cannot establish whether a departure belongs to the reach.

1. Use only the fifteen development people and the supplied accelerometer/TRIG readers: no EEG, WAY or held-out data. The D2 labels, the v5 k=5 onsets, all existing artifacts and both v5 UNDERIDENTIFIED outcomes stay unchanged, and nothing is frozen.
2. Verify the input closures and the fixed counts (3,964 events, 3,962 determinate, 291 tail). Write only into fresh diagnostic directories and stages, and re-check the four new scripts' hashes between stages.
3. Run in an actual Slurm compute shell with at most eight workers and one BLAS/OpenMP thread each. Participant subsets are allowed only in health mode. Health mode writes engineering counts and statuses only: no feature values, synthetic outcomes, decisions, provider output or figure.
4. Stage order: synthetic checks; real features (PRD decision); CPC strata/regression; CPC controls (CPC decision). Provider AccStartIndex is read only after both decisions are closed, with verified cue/start joins, explicit matched denominators and no provider flag filtering.
5. Compare tail events (determinate lag ≥ 288 ms) with reference events (0 ≤ lag < 288 ms) within each person. Require native validity over [ws − 720, L + 100], where ws is the first qualifying-window start, and a later-segment amplitude A_F ≥ θ5. Count every exclusion.
6. Fixed quantities:
   - b0 = mean a over [R − 520, R − 20) and the departure d = a − b0;
   - the later bout segment F = [L, L + 100) and A_F = ||mean d over F||;
   - H = L − t½, where t½ is the first native 40-ms window mean reaching half of A_F (searched through L + 60). If half is never reached, H is censored below −60 ms.
   Quarter-amplitude H, ρ_R, Λ and the moving-baseline versions are sensitivities.
7. Continuity: C_E and C_L are the cosines between the F vector and the 40-ms mean departure at R (C_E) or just before L (C_L). Successive non-overlapping 40-ms windows are descriptive. Also report:
   - the pre-R baseline slope and its 0.27-s residual scale;
   - whether removing that trend un-qualifies the first window ("drift explained"), and the trend's alignment with F;
   - anchor occupancy, prior activity, bout gaps, sample occupancy and rise_per_s.
   Neither monotone acceleration nor a link between sensor direction and reach direction is assumed.
8. Statistics: within-person median differences (proportion differences for indicators), requiring at least three tail and twenty reference observations per feature. H censoring bounds are propagated. People are summarized by their median, with 10,000 person-bootstrap resamples (seed 20261017); the intervals are diagnostic and unadjusted.
9. Synthetic known positives use the 32 cached v5 backgrounds per person and orthogonal direction pairs from SeedSequence([20261017, pid, 7]). S = 1920 ms. Labels are actual D2 labels after a valid, quiet 500-ms arming history. Noiseless flat axes are exempt from the noise-validity screen.
10. Shapes: each sustained ramp is shifted so that its noiseless R is 2160 ms (2163 ms at 1 kHz).
    - KP0 (reference) is an 80-ms raised-cosine step to 4·HIGH.
    - KPA (H_A type) adds a perpendicular drift whose residual is ≥ 1.3·θ5 while X stays ≤ 0.85·LOW.
    - KPB (H_B type) is a collinear ramp (residual ≥ 1.05·θ5, X ≤ 0.98·LOW), plus a small step that crosses LOW and a later HIGH step after F. Nine fixed candidates are tried in order (delays {350, 320, 380} ms × LOW factors {1.3, 1.2, 1.4}); the first with a supported 300–400-ms noiseless lag is used, without consulting any feature.
11. Synthetic set rules mirror the real ones: KP0 events need a noisy reference lag, and KPB/KPA events a noisy tail lag. A person is covered with ≥ 24 usable paired backgrounds for both KPB and KPA. Interpret only the decision set (feature-eligible, covered people), which must include at least twelve people. Sub-04 is expected to be KPA-uncovered: its LOW/θ5 is 1.24, while KPA needs about 1.47.
12. Synthetic contrasts (KPx − KP0 per person) use the same function as the real contrasts. KPB_power: a ΔH estimate ≥ 100 ms with CI lower bound > 0, a ΔC_E CI lower bound > −0.2 and a KPB median C_E ≥ 0.5, so the rule returns O1 on H_B-type data. Per-person synthetic values are descriptive, not gates.
13. KPA_power: a ΔH CI inside [−50, 50] ms, a ΔC_E CI upper bound < −0.2 and a KPA median C_E < 0.5, so the rule returns equivalence plus discontinuity on H_A-type data. KPA_drift_power also requires a drift-explained Δ CI lower bound > 0, a KPA median drift cosine < 0.5 and a KPB median drift cosine ≥ 0.5.
14. Weakens premature R from low-amplitude activity/drift (H_A): O1, a ΔH estimate ≥ 100 ms with CI lower bound > 0 and a ΔC_E CI lower bound > −0.2, given KPA_power. Tail departures then reach half the later amplitude at least 100 ms further before L than reference departures do. O1 motivates testing an anchored-baseline or slope-sensitive detector. It does not establish reach identity or a worst-case bound.
15. Weakens the detector missing reach movement (H_B): O2 or O3. In both, the ΔH CI lies inside [−50, 50] ms, so the extra L − R is spent below half the later amplitude.
    - O2 adds either a discontinuity (ΔC_E CI upper bound < −0.2 and tail median C_E < 0.5) or a drift explanation (drift-explained Δ CI lower bound > 0 and tail median drift cosine < 0.5).
    - O3 instead shows continuity (ΔC_E CI lower bound > −0.2, tail median C_E ≥ 0.5) with no positive drift signature.
    These weaken a substantial miss. They do not validate R or L, or prove that drift is absent.
16. Each statement is evaluated separately with its dependencies: H equivalence needs KPB_power; O1, the discontinuity branch and O3's continuity need KPA_power; the drift branch needs KPA_drift_power. A failed dependency makes only that statement UNDERIDENTIFIED. A supported O1 is reported first, then O2, then O3; otherwise the outcome is O4 (UNRESOLVED, or UNDERIDENTIFIED with reasons).
17. CPC keeps v5's 32 selected backgrounds and their IDs first, then appends every remaining eligible inventory row in session/run/start-tone order. One shared control loop must reproduce every first-32 v5 field exactly before any expanded output; a mismatch stops the stage.
18. Development-slope strata (descriptive):
    - Rise rate = (mean r over [R + 40, R + 80) − mean r over [R, R + 40)) / (0.04 s · θ5), from each person's determinate D2 k=5 events.
    - Strata sit at the person's q10 and q50, for linear and quadratic shapes.
    - Amplitudes come from a doubling bracket, 4,000 geometric grid points per phase and thirty local refinements. Choose the closest rate among phase/cap-valid candidates (ties go to the smaller amplitude) and require ≤ 5% relative error.
    - Target and achieved rates are recorded, and infeasible strata stay uncovered.
19. Keep all six v5 families, the exact targets and truths, the phase/cap/anchor/association rules, k ∈ {3, 5, 8}, and seed key 2 for original directions (key 4 for strata). The gates are unchanged: near ≥ 0.90; long and linear/quadratic fast ≥ 0.95; coverage ≥ 0.95; twenty backgrounds per required band; sham ≤ 0.05; all fifteen people.
20. Report, per trace, the target-320 lag error (and the opposite-signed onset error). Also report margin curves for targets 272–320 and gate intervals from 2,000 background resamples (seed keys 20/21 for all/original backgrounds). Population means and curves use 10,000 equal-weight person resamples.
21. CP1 places each interval below, across or above its unchanged gate, or marks it uncovered. An interval crossing a gate is inconclusive: it does not show that the v5 failure was sampling noise. A systematic synthetic precision failure cannot be repaired by detector changes alone.
22. CP2 calls a cell timing-error consistent only if all of the following hold; otherwise it names association/anchor or target-dependence/coverage limits:
    - the target-320 error covers ≥ 95% of traces;
    - it mispredicts ≤ 1% of near rows;
    - ≤ 1% of near rows are unresolved;
    - the predicted near fraction differs from the observed one by ≤ 0.01.
23. CP3 compares development-slope near sensitivity with 0.90 (descriptive). CP4 reports shams, alarm reasons and unanchorable backgrounds. Sub-02's reported v5 sham fraction (0.070) and two unanchorable backgrounds are retained. Larger prospective samples, reference-resolution changes and gate questions remain scientist decisions.
24. Before this declaration the orchestrator had seen the v5 per-person tail counts, noise references and the pooled v5 control-error distribution. Alternatives compared: three outcome patterns, two H_A signatures, nine KPB candidate shapes (chosen on lag/support only), one KPA shape, one KP0 shape and four descriptive slope strata.
25. Report one 300-dpi six-panel figure labelled "development calibration/diagnostic evidence; post hoc", and exactly three summary lines. Design implications:
    - O1: test an anchored or slope-sensitive detector.
    - O2: run a new reference-association/detrending test; no threshold reduction is indicated.
    - O3: the onset-definition question goes to the scientist before a lower or anchored LOW is tested.
    - O4: no automatic change.
    Nothing here grants EEG scoring clearance.
"""

FEATURES = (
    "H", "H25", "CE", "CL", "rho_R", "Lambda", "min_step_cos",
    "D_theta", "drift_explained", "drift_cos", "D_F",
    "anchor_occ_max", "anchor_mean_r_theta", "prior_activity",
    "multiple_bouts", "longest_quiet_ms", "sample_occupancy",
    "moving_H", "moving_CE", "moving_CL", "moving_rho_R", "rise_per_s",
)
KP_FEATURES = ("H", "CE", "drift_explained", "drift_cos")
INDICATORS = {"drift_explained", "prior_activity", "multiple_bouts"}
CENSORED_FEATURES = {"H", "H25", "moving_H"}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(value):
    return math.isfinite(number(value))


def code_hashes():
    global _HASHES
    if _HASHES is None:
        _HASHES = {name: D.digest(CODE / name) for name in NEW_SCRIPTS}
    return dict(_HASHES)


def scope(args):
    if args.participants is not None and not args.health_only:
        raise ValueError("--participants is allowed only with --health-only")
    return D.scope(args)


def check_stage(folder, stage, args=None):
    metadata = D.require_closed(folder, stage)
    if metadata.get("new_script_sha256") != code_hashes():
        raise RuntimeError("New diagnostic code changed after closure")
    if args is not None:
        D.check_scope(metadata, scope(args))
    return metadata


def close_stage(folder, stage, args, **metadata):
    D.close(
        folder, stage, **scope(args),
        new_script_sha256=code_hashes(),
        run_root=str(args.output_dir.resolve()), **metadata,
    )


def output_root(args, kind, *, first):
    job = os.environ["SLURM_JOB_ID"]
    if args.health_only:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            raise RuntimeError("SCRATCH is required for health mode")
        expected = (
            Path(scratch) / "br_autoresearch"
            / "episode19_causal_motor_forecasting_benchmark"
            / "v5_diagnostics_work" / f"smoke_{job}" / kind
        )
    else:
        expected = VIEWS / f"{kind}_job_{job}"
    proposed = args.output_dir
    if proposed.is_symlink() or proposed.resolve() != expected.resolve():
        raise RuntimeError(f"Output must be the new job root: {expected}")
    protected = {
        "detector_freeze_job_48219799", "raw_onset_test_job_48230171",
        "detector_v2_job_48261284", "raw_onset_test_v5_job_48261284",
    }
    if protected.intersection(proposed.resolve().parts):
        raise RuntimeError("Output enters a preserved run")
    if first:
        if proposed.exists():
            raise FileExistsError("Initial diagnostic root must not already exist")
        return D.fresh(proposed)
    if not proposed.is_dir():
        raise FileNotFoundError("Initial diagnostic stage is missing")
    return proposed.resolve()


def inputs():
    """Read fixed closed inputs; never open the provider CSV."""
    for folder, stage in (
        (DETECTOR / "settings", "settings"),
        (DETECTOR / "D2", "labels_D2"),
        (DETECTOR / "labels_stage", "labels"),
        (V5RUN / "calibration", "calibration"),
        (V5RUN / "controls", "controls"),
        (V5RUN / "events", "events"),
    ):
        D.require_closed(folder, stage)
    settings = D.read_json(DETECTOR / "settings/detector_settings.json")
    refs = D.read_json(V5RUN / "calibration/noise_reference.json")
    if len(refs) != 15 or {r["participant"] for r in refs} != set(D.PEOPLE):
        raise RuntimeError("Noise-reference person inventory differs")
    cues = D.read_csv(DETECTOR / "D2/self_paced_cues.csv")
    all_rows = D.read_csv(V5RUN / "events/D2/raw_onsets.csv")
    rows = [r for r in all_rows if int(r["k"]) == 5]
    cue_map = {D.event_key(r): r for r in cues}
    row_map = {D.event_key(r): r for r in rows}
    if len(cues) != 4432 or len(cue_map) != len(cues):
        raise RuntimeError("D2 cue inventory differs")
    if {r["participant"] for r in cues} != set(D.PEOPLE):
        raise RuntimeError("Non-development or missing participant")
    determinate = [r for r in rows if r["status"] == "determinate"]
    tail = [r for r in determinate if float(r["lag_ms"]) >= 288]
    if (len(all_rows), len(rows), len(row_map), len(determinate), len(tail)) != (
        11892, 3964, 3964, 3962, 291
    ):
        raise RuntimeError("Fixed v5 event-count contract differs")
    accepted = {
        key for key, cue in cue_map.items()
        if V4.optional_int(cue["onset_tick"]) is not None
    }
    if set(row_map) != accepted:
        raise RuntimeError("Fixed D2/v5 event keys differ")
    references = {r["participant"]: r for r in refs}
    for key, row in row_map.items():
        cue = cue_map[key]
        if (
            int(row["L_tick"]) != int(cue["onset_tick"])
            or int(row["S_tick"]) != int(cue["start_known_tick"])
            or float(row["theta_g"]) != V4.reference_theta(references[key[0]], 5)
        ):
            raise RuntimeError("Fixed event/cue/reference join differs")
        if row["status"] == "determinate":
            lag = float(row["lag_ms"])
            if lag != int(row["L_tick"]) - int(row["R_tick"]) or lag < 0:
                raise RuntimeError("Fixed lag arithmetic/range differs")
    low = float(settings["candidates"]["D2"]["low_g"])
    high = float(settings["candidates"]["D2"]["high_g"])
    return refs, cues, rows, low, high


def bootstrap_people(values, *, mean=False):
    x = np.asarray([float(v) for v in values if finite(v)], dtype=float)
    if not len(x):
        return {"n_people": 0, "estimate": None, "ci95": [None, None]}
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(x), size=(10000, len(x)))
    statistic = np.mean if mean else np.median
    boot = statistic(x[indices], axis=1)
    return {
        "n_people": len(x), "estimate": float(statistic(x)),
        "ci95": np.quantile(boot, [.025, .975], method="inverted_cdf").tolist(),
    }


def support_ok(ok, fs, ws, L):
    dt = int(round(1000 / fs))
    a, b = (ws - 720) // dt, (L + 100) // dt + 1
    return bool(a >= 0 and b <= len(ok) and np.all(ok[a:b]))


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


def rise_rate(raw, ok, fs, onset, theta):
    dt = int(round(1000 / fs))
    a, m, b = onset // dt, (onset + 40) // dt, (onset + 80) // dt
    if a < 0 or b > len(raw) or not np.all(ok[a:b]):
        return None
    return float((np.mean(raw[m:b]) - np.mean(raw[a:m])) / (.04 * theta))


def extract_features(acc, raw, ok, fs, L, onset, ws, theta, n_bouts,
                     native_vectors=None):
    dt = int(round(1000 / fs))
    if not support_ok(ok, fs, ws, L):
        return {"feature_status": "unsupported"}
    if onset > L:
        return {"feature_status": "unexpected_negative_lag"}
    if native_vectors is None:
        native_vectors = V4.native_residual(acc, fs)

    def section(values, a, b):
        return values[a // dt:b // dt]

    baseline = section(acc, onset - 520, onset - 20)
    b0 = np.mean(baseline, axis=0)
    vf = np.mean(section(acc, L, L + 100), axis=0) - b0
    af = float(np.linalg.norm(vf))
    if not math.isfinite(af) or af < theta:
        return {"feature_status": "weak_later_segment", "A_F_g": af}
    out = {
        "feature_status": "ok", "A_F_g": af,
        "rise_per_s": rise_rate(raw, ok, fs, onset, theta),
    }

    def amplitude_features(values, offset, prefix, later, amplitude):
        if not math.isfinite(amplitude) or amplitude < theta:
            return
        local = section(values, onset, L + 100) - offset
        means = D.S.trailing_mean(local, 40 // dt)[40 // dt - 1:]
        magnitude = np.linalg.norm(means, axis=1)
        out[prefix + "CE"] = cosine(means[0], later)
        late = np.mean(section(values, L - 40, L), axis=0) - offset
        out[prefix + "CL"] = cosine(late, later)
        out[prefix + "rho_R"] = float(magnitude[0] / amplitude)
        levels = ((.5, "H"),) if prefix else ((.5, "H"), (.25, "H25"))
        for level, label in levels:
            hits = np.flatnonzero(magnitude >= level * amplitude)
            name = prefix + label
            if len(hits):
                tick = onset + int(hits[0]) * dt
                value = float(L - tick)
                out.update({
                    name: value, name + "_lo": value, name + "_hi": value,
                    name + "_censored": 0,
                })
                if not prefix and level == .5:
                    out["t50_tick"] = tick
                    out["Lambda"] = (
                        float((tick - onset) / (L - onset)) if L > onset else None
                    )
            else:
                out.update({
                    name: None, name + "_lo": -float("inf"),
                    name + "_hi": -60.0, name + "_censored": 1,
                })
                if not prefix and level == .5:
                    out["t50_tick"], out["Lambda"] = None, None

    amplitude_features(acc, b0, "", vf, af)
    mvf = np.mean(section(native_vectors, L, L + 100), axis=0)
    maf = float(np.linalg.norm(mvf))
    out["moving_A_F_g"] = maf
    amplitude_features(native_vectors, np.zeros(3), "moving_", mvf, maf)

    windows = [
        np.mean(section(acc, t, t + 40), axis=0) - b0
        for t in range(onset, L - 39, 40)
    ]
    adjacent = [cosine(a, b) for a, b in zip(windows[:-1], windows[1:])]
    adjacent = [v for v in adjacent if v is not None]
    out["min_step_cos"] = min(adjacent) if adjacent else None

    bt = np.arange((onset - 520) // dt, (onset - 20) // dt) * dt / 1000
    tc = float(np.mean(bt))
    centered = bt - tc
    delta = np.sum(centered[:, None] * (baseline - b0), axis=0)
    delta /= np.sum(centered ** 2)
    norm_delta = float(np.linalg.norm(delta))
    wt = np.arange(ws // dt, (ws + 40) // dt) * dt / 1000
    trend = b0 + (wt - tc)[:, None] * delta
    detrended = np.linalg.norm(section(acc, ws, ws + 40) - trend, axis=1)
    _, trend_q, _, _ = V4.qualifying_windows(detrended, theta, 40 // dt)
    out.update({
        "D_theta": norm_delta * .27 / theta,
        "drift_explained": int(not bool(trend_q[0])),
        "drift_cos": cosine(delta, vf),
        "D_F": norm_delta * (L - onset) / 1000 / af,
        "baseline_slope_x_g_s": float(delta[0]),
        "baseline_slope_y_g_s": float(delta[1]),
        "baseline_slope_z_g_s": float(delta[2]),
    })

    begin = ws - 720
    local_raw = raw[begin // dt:(L + 100) // dt + 1]
    hot, q, _, _ = V4.qualifying_windows(local_raw, theta, 40 // dt)
    cumulative = np.r_[0, np.cumsum(hot, dtype=np.int64)]
    width = 40 // dt
    occupancy = (cumulative[width:] - cumulative[:-width]) / width

    def q_slice(a, b):
        return slice((a - begin) // dt, (b - begin) // dt)

    anchor = q_slice(ws - 200, ws)
    if np.any(q[anchor]) or not q[(ws - begin) // dt]:
        raise RuntimeError("Closed first-window/anchor qualification differs")
    prior = q[q_slice(ws - 700, ws - 200)]
    quiet = ~q[q_slice(ws, L + dt)]
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    longest = int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0
    sample = section(raw, onset, L)
    out.update({
        "anchor_occ_max": float(np.max(occupancy[anchor])),
        "anchor_mean_r_theta": float(
            np.mean(section(raw, ws - 200, ws)) / theta),
        "prior_activity": int(np.any(prior)),
        "multiple_bouts": int(n_bouts > 1),
        "longest_quiet_ms": longest * dt,
        "sample_occupancy": float(np.mean(sample > theta)) if len(sample) else None,
    })
    return out


def feature_bounds(row, feature):
    if feature in CENSORED_FEATURES:
        lo = number(row.get(feature + "_lo"))
        hi = number(row.get(feature + "_hi"))
        if not math.isnan(lo) and not math.isnan(hi) and lo <= hi:
            return lo, hi
        return None
    if finite(row.get(feature)):
        value = float(row[feature])
        return value, value
    return None


def aggregate_person_items(items, people):
    """Shared person statistic for real data and closed synthetic contrasts."""
    selected = {p: items[p] for p in people}
    included = [item for item in selected.values() if item["included"]]
    lower, upper, tails = [], [], []
    for item in included:
        lo = (
            -float("inf") if item["delta_lower_unbounded"]
            else number(item["delta_lo"])
        )
        hi = (
            float("inf") if item["delta_upper_unbounded"]
            else number(item["delta_hi"])
        )
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            raise RuntimeError("Malformed closed person bounds")
        lower.append(lo)
        upper.append(hi)
        if finite(item.get("tail")):
            tails.append(float(item["tail"]))
    n = len(included)
    result = {
        "people": selected, "n_people": n, "estimate": None,
        "ci95": [None, None], "bounded": False,
        "positive_people": sum(v > 0 for v in lower),
        "tail_median": float(np.median(tails)) if n and len(tails) == n else None,
        "median_lower": None, "median_upper": None,
        "lower_unbounded": False, "upper_unbounded": False,
    }
    if n:
        lo, hi = np.asarray(lower), np.asarray(upper)
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, n, size=(10000, n))
        low_ci = float(np.quantile(
            np.median(lo[indices], axis=1), .025, method="inverted_cdf"))
        high_ci = float(np.quantile(
            np.median(hi[indices], axis=1), .975, method="inverted_cdf"))
        ml, mh = float(np.median(lo)), float(np.median(hi))
        result.update(
            estimate=ml if ml == mh and math.isfinite(ml) else None,
            ci95=[low_ci, high_ci],
            bounded=math.isfinite(low_ci) and math.isfinite(high_ci),
            median_lower=ml, median_upper=mh,
            lower_unbounded=not math.isfinite(low_ci),
            upper_unbounded=not math.isfinite(high_ci),
        )
    return result


def person_feature(rows, people, feature, *, min_tail=3, min_reference=20,
                   tail_only=False):
    items = {}
    for p in people:
        bounds, counts = {}, {}
        for group in ("tail", "reference"):
            values = []
            for row in rows:
                if (row["participant"] == p and row["set"] == group
                        and row.get("feature_status") == "ok"):
                    value = feature_bounds(row, feature)
                    if value is not None:
                        values.append(value)
            counts[group] = len(values)
            if values:
                array = np.asarray(values, dtype=float)
                statistic = np.mean if feature in INDICATORS else np.median
                bounds[group] = (
                    float(statistic(array[:, 0])),
                    float(statistic(array[:, 1])),
                )
        included = (
            counts["tail"] >= min_tail
            and (tail_only or counts["reference"] >= min_reference)
        )
        item = {
            "tail_n": counts["tail"], "reference_n": counts["reference"],
            "included": included,
            "comparison": "tail_median" if tail_only else "tail_minus_reference",
        }
        if included:
            tl, th = bounds["tail"]
            rl, rh = (0.0, 0.0) if tail_only else bounds["reference"]
            dl = tl - rh if math.isfinite(rh) else -float("inf")
            du = th - rl if math.isfinite(rl) else float("inf")
            item.update(
                delta=dl if dl == du and math.isfinite(dl) else None,
                delta_lo=dl, delta_hi=du,
                delta_lower_unbounded=not math.isfinite(dl),
                delta_upper_unbounded=not math.isfinite(du),
                tail=tl if tl == th and math.isfinite(tl) else None,
                reference=(
                    rl if not tail_only and rl == rh and math.isfinite(rl)
                    else None
                ),
                tail_lo=tl, tail_hi=th,
            )
        items[p] = item
    return aggregate_person_items(items, people)


def summarize_features(rows, people):
    return {feature: person_feature(rows, people, feature) for feature in FEATURES}


def synthetic_estimate(acc, fs, theta, low, high, *, noiseless=False):
    raw, ok, departure, ticks, valid = V4.control_features(acc, fs)
    if noiseless:
        valid = np.isfinite(departure)
        ok = np.isfinite(raw)
    start = int(np.searchsorted(ticks, 1920, side="left"))
    begin = start - F.SP_QUIET
    if (
        begin < 0 or not np.all(valid[begin:start])
        or not np.all(departure[begin:start] <= low)
    ):
        return {"status": "unarmed", "reason": "quiet_history"}, raw, ok
    if not np.all(valid[begin:]) or not np.all(np.isfinite(departure[begin:])):
        return {"status": "unsupported", "reason": "detector_support"}, raw, ok
    _, _, cues = D.run_segment_d2(
        departure[begin:, None], np.array([low]), np.array([high]),
        F.SP_CONFIRM, F.SP_QUIET, {start - begin: "supported"}, set(),
    )
    if len(cues) != 1 or cues[0].onset is None:
        return {"status": "no_label", "reason": "no_confirmed_D2"}, raw, ok
    L = int(ticks[begin + int(cues[0].onset[0])])
    est = V5.estimate(raw, ok, fs, L, 1920, theta, evidence_stop=2800)
    row = {
        "status": est["status"], "reason": est["reason"], "L_tick": L,
        "R_tick": est["R_tick"], "lag_ms": est["lag_ms"],
        "ws_tick": est["candidate_window_start_tick"], "n_bouts": est["n_bouts"],
    }
    if est["status"] == "determinate" and not support_ok(
        ok, fs, int(est["candidate_window_start_tick"]), L
    ):
        row["status"], row["reason"] = "unsupported", "diagnostic_support"
    return row, raw, ok


def kp_shapes(reference, low, high):
    fs = float(reference["fs"])
    theta = V4.reference_theta(reference, 5)
    dt = int(round(1000 / fs))
    ticks = np.arange(2800 // dt) * dt
    desired_R = 2163 if fs == 1000 else 2160
    unit = ticks / 1000.0
    residual_scale = float(V5.scalar_residual(unit, fs)[-1])
    unit_acc = np.column_stack((unit, np.zeros_like(unit), np.zeros_like(unit)))
    reduced, _ = R.reduce_to_detector_rate(unit_acc, fs)
    detector_scale = float(D.S.self_paced_departure(reduced)[-1])

    def step(at, width, amplitude):
        u = np.clip((ticks - at) / width, 0, 1)
        return amplitude * (.5 - .5 * np.cos(np.pi * u))

    fast = step(desired_R + 350, 80, 4 * high)

    def ramp(residual_factor, detector_factor):
        lower = residual_factor * theta / residual_scale
        upper = detector_factor * low / detector_scale
        if lower > upper:
            return None, {"reason": "infeasible_plateau_interval"}
        slope = (lower + upper) / 2
        trial = slope * np.maximum((ticks - 1200) / 1000, 0)
        first = V5.first_detectable(
            np.abs(V5.scalar_residual(trial, fs)), fs, theta, 1200)
        if first is None:
            return None, {"reason": "no_noiseless_departure"}
        T0 = 1200 + desired_R - first
        signal = slope * np.maximum((ticks - T0) / 1000, 0)
        return signal, {
            "slope_g_s": slope, "T0_tick": T0,
            "residual_plateau_g": slope * residual_scale,
            "X_plateau_g": slope * detector_scale,
        }

    shapes, records = {}, []

    def inspect(kind, signal, parameters, lower_lag, upper_lag):
        row, _, _ = synthetic_estimate(
            signal, fs, theta, low, high, noiseless=True)
        good = bool(
            row["status"] == "determinate"
            and lower_lag <= float(row["lag_ms"]) <= upper_lag
        )
        record = {
            "type": kind, **parameters, **row,
            "calibration_usable": good, "selected": False,
        }
        records.append(record)
        return good, record

    zero = np.zeros_like(fast)
    good, record = inspect(
        "KP0", np.column_stack((fast, zero, zero)),
        {"candidate": 0}, 0, 199)
    if good:
        shapes["KP0"] = (fast, zero)
        record["selected"] = True

    drift, info = ramp(1.3, .85)
    if drift is None:
        records.append({
            "type": "KPA", "calibration_usable": False,
            "selected": False, **info,
        })
    else:
        good, record = inspect(
            "KPA", np.column_stack((fast, drift, zero)),
            {"candidate": 0, **info}, 300, 400)
        if good:
            shapes["KPA"] = (fast, drift)
            record["selected"] = True

    slow, info = ramp(1.05, .98)
    if slow is None:
        records.append({
            "type": "KPB", "calibration_usable": False,
            "selected": False, **info,
        })
    else:
        candidate = 0
        for delay in (350, 320, 380):
            for overshoot in (1.3, 1.2, 1.4):
                extra = overshoot * low - info["X_plateau_g"]
                signal = (
                    slow + step(desired_R + delay, 40, extra)
                    + step(desired_R + 500, 40, 4 * high)
                )
                good, record = inspect(
                    "KPB", np.column_stack((signal, zero, zero)),
                    {"candidate": candidate, "delay_ms": delay,
                     "LOW_overshoot_factor": overshoot, **info}, 300, 400)
                if good and "KPB" not in shapes:
                    shapes["KPB"] = (signal, zero)
                    record["selected"] = True
                candidate += 1
    return shapes, records


def kp_worker(task):
    reference, low, high, health = task
    D.require_slurm(1)
    p, fs = reference["participant"], float(reference["fs"])
    theta = V4.reference_theta(reference, 5)
    shapes, calibration = kp_shapes(reference, low, high)
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, int(p.split("-")[1]), 7]))
    rows, counts = [], Counter()
    with np.load(V5RUN / "calibration" / f"{p}_backgrounds.npz",
                 allow_pickle=False) as saved:
        backgrounds = saved["acc"]
        if float(saved["fs"]) != fs or len(backgrounds) != 32:
            raise RuntimeError("Known-positive background contract differs")
        for bid, background in enumerate(backgrounds):
            e1 = rng.normal(size=3)
            e1 /= np.linalg.norm(e1)
            e2 = rng.normal(size=3)
            e2 -= np.dot(e2, e1) * e1
            if np.linalg.norm(e2) <= 1e-12:
                raise RuntimeError("Degenerate direction pair")
            e2 /= np.linalg.norm(e2)
            for kind in ("KP0", "KPB", "KPA"):
                identity = {"participant": p, "background_id": bid, "type": kind}
                if kind not in shapes:
                    counts[f"{kind}:uncovered_shape"] += 1
                    if not health:
                        rows.append({**identity, "feature_status": "uncovered_shape"})
                    continue
                along, perpendicular = shapes[kind]
                injected = (
                    background + along[:, None] * e1
                    + perpendicular[:, None] * e2
                )
                event, raw, ok = synthetic_estimate(
                    injected, fs, theta, low, high)
                if event["status"] != "determinate":
                    counts[f"{kind}:{event['status']}:{event['reason']}"] += 1
                    if not health:
                        rows.append({
                            **identity, **event, "feature_status": event["status"],
                        })
                    continue
                if health:
                    counts[f"{kind}:supported_estimate"] += 1
                    continue
                features = extract_features(
                    injected, raw, ok, fs, int(event["L_tick"]),
                    int(event["R_tick"]), int(event["ws_tick"]),
                    theta, int(event["n_bouts"]),
                )
                counts[f"{kind}:{features['feature_status']}"] += 1
                rows.append({**identity, **event, **features})
    return {
        "participant": p, "counts": dict(counts), "rows": rows,
        "calibration": [] if health else calibration,
    }


def kp_set_match(row):
    if not finite(row.get("lag_ms")):
        return False
    lag = float(row["lag_ms"])
    return (0 <= lag < 288) if row["type"] == "KP0" else lag >= 288


def kp_usable(row):
    return bool(
        row.get("feature_status") == "ok" and kp_set_match(row)
        and finite(row.get("CE")) and feature_bounds(row, "H") is not None
    )


def kp_type_counts(rows, kind):
    lags = [float(r["lag_ms"]) for r in rows if finite(r.get("lag_ms"))]
    n = len(rows)
    tail = sum(v >= 288 for v in lags)
    reference = sum(0 <= v < 288 for v in lags)
    matching = reference if kind == "KP0" else tail
    return {
        "attempted_backgrounds": n,
        "lag_available_n": len(lags),
        "median_noisy_lag_ms": float(np.median(lags)) if lags else None,
        "tail_n": tail, "reference_n": reference,
        "fraction_tail": tail / n if n else None,
        "fraction_reference": reference / n if n else None,
        "set_definition": "reference" if kind == "KP0" else "tail",
        "set_definition_fraction": matching / n if n else None,
        "usable_set_n": sum(kp_usable(r) for r in rows),
        "usable_set_fraction": sum(kp_usable(r) for r in rows) / n if n else None,
        "uncovered_n": sum(
            r.get("feature_status") == "uncovered_shape" for r in rows),
        "unsupported_n": sum(
            r.get("feature_status") in ("unsupported", "no_valid_data")
            for r in rows),
        "feature_status_counts": dict(Counter(
            r.get("feature_status", "missing") for r in rows)),
        "fraction_denominator": "all attempted backgrounds, including exclusions",
    }


def kp_population(known, people):
    population = {
        kind: {
            feature: aggregate_person_items(
                known["contrasts"][kind][feature]["people"], people)
            for feature in KP_FEATURES
        } for kind in ("KPB", "KPA")
    }
    n = len(people)

    def ready(kind, feature):
        metric = population[kind][feature]
        return n >= 12 and metric["n_people"] == n and metric["bounded"]

    bh, bc = population["KPB"]["H"], population["KPB"]["CE"]
    ah, ac = population["KPA"]["H"], population["KPA"]["CE"]
    ad = population["KPA"]["drift_explained"]
    adir = population["KPA"]["drift_cos"]
    bdir = population["KPB"]["drift_cos"]
    b_power = bool(
        ready("KPB", "H") and ready("KPB", "CE")
        and finite(bh["estimate"]) and bh["estimate"] >= 100
        and bh["ci95"][0] > 0 and bc["ci95"][0] > -.2
        and finite(bc["tail_median"]) and bc["tail_median"] >= .5
    )
    a_power = bool(
        ready("KPA", "H") and ready("KPA", "CE")
        and ah["ci95"][0] >= -50 and ah["ci95"][1] <= 50
        and ac["ci95"][1] < -.2
        and finite(ac["tail_median"]) and ac["tail_median"] < .5
    )
    drift_power = bool(
        a_power and ready("KPA", "drift_explained")
        and ready("KPA", "drift_cos") and ready("KPB", "drift_cos")
        and ad["ci95"][0] > 0
        and finite(adir["tail_median"]) and adir["tail_median"] < .5
        and finite(bdir["tail_median"]) and bdir["tail_median"] >= .5
    )
    return {
        "people": list(people), "n_people": n, **population,
        "KPB_power": b_power, "KPA_power": a_power,
        "KPA_drift_power": drift_power,
        "minimum_people": 12,
    }


def kp_summary(rows, people, calibration):
    by_person, paired_rows = {}, {"KPB": [], "KPA": []}
    for p in people:
        selected = [r for r in rows if r["participant"] == p]
        usable = {
            kind: {
                int(r["background_id"]): r for r in selected
                if r["type"] == kind and kp_usable(r)
            } for kind in ("KP0", "KPB", "KPA")
        }
        pairs = {}
        for kind in ("KPB", "KPA"):
            ids = sorted(set(usable["KP0"]) & set(usable[kind]))
            pairs[kind] = len(ids)
            for bid in ids:
                paired_rows[kind].append({**usable[kind][bid], "set": "tail"})
                paired_rows[kind].append({**usable["KP0"][bid], "set": "reference"})
        selected_calibration = {
            kind: next(
                (r for r in calibration[p]
                 if r["type"] == kind and r.get("selected")), None)
            for kind in ("KP0", "KPB", "KPA")
        }
        by_person[p] = {
            "covered": pairs["KPB"] >= 24 and pairs["KPA"] >= 24,
            "paired_backgrounds": pairs,
            "types": {
                kind: {
                    **kp_type_counts([r for r in selected if r["type"] == kind], kind),
                    "noiseless_calibration": selected_calibration[kind],
                } for kind in ("KP0", "KPB", "KPA")
            },
            "selected_KPB_candidate": selected_calibration["KPB"],
        }
    contrasts = {
        kind: {
            feature: person_feature(
                paired_rows[kind], people, feature,
                min_tail=24, min_reference=24, tail_only=feature == "drift_cos")
            for feature in KP_FEATURES
        } for kind in ("KPB", "KPA")
    }
    result = {
        "people": by_person, "contrasts": contrasts,
        "pooled_types": {
            kind: kp_type_counts([r for r in rows if r["type"] == kind], kind)
            for kind in ("KP0", "KPB", "KPA")
        },
        "minimum_paired_backgrounds": 24, "minimum_interpreted_people": 12,
        "per_person_values_are_descriptive_not_gates": True,
    }
    covered = [p for p in people if by_person[p]["covered"]]
    result["population"] = kp_population(result, covered)
    return result


def prd_decision(rows, known):
    all_summary = summarize_features(rows, list(D.PEOPLE))
    people = [
        p for p in D.PEOPLE
        if all_summary["H"]["people"][p]["included"]
        and all_summary["CE"]["people"][p]["included"]
        and known["people"][p]["covered"]
    ]
    summaries = summarize_features(rows, people)
    kp = kp_population(known, people)
    h, c = summaries["H"], summaries["CE"]
    dr, dc = summaries["drift_explained"], summaries["drift_cos"]
    primary_ready = bool(
        people and h["n_people"] == len(people) and c["n_people"] == len(people)
        and h["bounded"] and c["bounded"]
    )
    drift_ready = bool(
        people and dr["n_people"] == len(people) and dc["n_people"] == len(people)
        and dr["bounded"] and finite(dc["tail_median"])
    )
    criteria = {
        "large_early_amplitude": bool(
            primary_ready and finite(h["estimate"])
            and h["estimate"] >= 100 and h["ci95"][0] > 0),
        "H_equivalent": bool(
            primary_ready and h["ci95"][0] >= -50 and h["ci95"][1] <= 50),
        "continuity_noninferior": bool(primary_ready and c["ci95"][0] > -.2),
        "discontinuity_signature": bool(
            primary_ready and c["ci95"][1] < -.2
            and finite(c["tail_median"]) and c["tail_median"] < .5),
        "drift_signature": bool(
            drift_ready and dr["ci95"][0] > 0 and dc["tail_median"] < .5),
        "continuous_low_amplitude": bool(
            primary_ready and c["ci95"][0] > -.2
            and finite(c["tail_median"]) and c["tail_median"] >= .5),
    }
    prerequisites = {
        "decision_set_at_least_12": len(people) >= 12,
        "primary_intervals_finite": primary_ready,
        "H_point_identified": finite(h["estimate"]),
        "drift_comparison_available": drift_ready,
        "KPB_power": kp["KPB_power"],
        "KPA_power": kp["KPA_power"],
        "KPA_drift_power": kp["KPA_drift_power"],
    }

    def statement(pattern, additional):
        dependencies = [
            "decision_set_at_least_12", "primary_intervals_finite", *additional]
        failed = [name for name in dependencies if not prerequisites[name]]
        return {
            "pattern": bool(pattern), "dependencies": dependencies,
            "failed_dependencies": failed,
            "status": (
                "UNDERIDENTIFIED" if failed
                else "SUPPORTED" if pattern else "NOT_OBSERVED"),
        }

    statements = {
        "O1": statement(
            criteria["large_early_amplitude"] and criteria["continuity_noninferior"],
            ["H_point_identified", "KPA_power"]),
        "O2_discontinuity": statement(
            criteria["H_equivalent"] and criteria["discontinuity_signature"],
            ["KPB_power", "KPA_power"]),
        "O2_drift": statement(
            criteria["H_equivalent"] and criteria["drift_signature"],
            ["drift_comparison_available", "KPB_power", "KPA_drift_power"]),
        "O3": statement(
            criteria["H_equivalent"] and criteria["continuous_low_amplitude"]
            and not criteria["drift_signature"],
            ["drift_comparison_available", "KPB_power", "KPA_power"]),
    }
    outcome = "O4"
    supported = [name for name, item in statements.items()
                 if item["status"] == "SUPPORTED"]
    if "O1" in supported:
        outcome = "O1"
    elif any(name.startswith("O2_") for name in supported):
        outcome = "O2"
    elif "O3" in supported:
        outcome = "O3"
    unidentified = [
        name for name, item in statements.items()
        if item["status"] == "UNDERIDENTIFIED"
    ]
    status = (
        "DIAGNOSTIC_PATTERN" if outcome != "O4"
        else "UNDERIDENTIFIED" if unidentified else "UNRESOLVED"
    )
    reasons = []
    if outcome == "O4":
        reasons = [
            f"{name}:{dependency}" for name in unidentified
            for dependency in statements[name]["failed_dependencies"]
        ]
        if not reasons:
            reasons = ["declared_amplitude_continuity_signatures_do_not_resolve_alternatives"]
    implications = {
        "O1": "Test an anchored-baseline or slope-sensitive detector in a new declared test.",
        "O2": "Test reference association or detrending in a new declared version; no threshold reduction is indicated here.",
        "O3": "Resolve the onset-definition question before testing lower or anchored LOW with noise guards.",
        "O4": "No automatic detector or reference change is indicated.",
    }
    return {
        "outcome": outcome, "status": status, "reasons": reasons,
        "decision_people": people,
        "omitted_people": [p for p in D.PEOPLE if p not in people],
        "KP": kp, "criteria": criteria, "prerequisites": prerequisites,
        "statements": statements, "supported_statements": supported,
        "underidentified_statements": unidentified,
        "decision_summaries": summaries, "all_people_summaries": all_summary,
        "next_design_implication": implications[outcome],
        "v5_D2_outcome": "UNDERIDENTIFIED", "v5_D1_outcome": "UNDERIDENTIFIED",
        "detector_frozen": False, "provider_access_has_not_occurred": True,
        "claim_scope": "Covered development people; sensor departure, not reach identity.",
        "alternatives_compared": {
            "primary_outcome_patterns": 3, "HA_signatures": 2,
            "KPB_candidates_per_feasible_person": 9,
            "KPA_shapes_per_person": 1, "KP0_shapes_per_person": 1,
            "descriptive_development_slope_strata": 4,
        },
    }


def real_worker(task):
    p, stems, events, reference, health = task
    D.require_slurm(1)
    theta = V4.reference_theta(reference, 5)
    by_run = defaultdict(list)
    for row in events:
        by_run[(row["session"], int(row["run"]))].append(row)
    output, counts = [], Counter()
    for stem in stems:
        _, session, run_number = V4.stem_key(stem)
        selected = by_run.get((session, run_number), [])
        if not selected:
            continue
        run, _, _, _, valid, _, _ = RF.sp_load(stem)
        raw = V4.native_statistic(run.acc_g, run.fs)
        ok = V4.native_validity(raw, valid, run.fs)
        vectors = None if health else V4.native_residual(run.acc_g, run.fs)
        for fixed in selected:
            identity = {
                "participant": p, "session": session, "run": run_number,
                "cue": fixed["cue"],
            }
            if fixed["status"] != "determinate":
                counts["unresolved"] += 1
                if not health:
                    output.append({
                        **identity, "set": "unresolved",
                        "feature_status": "unresolved", "rise_per_s": None,
                    })
                continue
            L, onset = int(fixed["L_tick"]), int(fixed["R_tick"])
            ws = int(fixed["candidate_window_start_tick"])
            group = "tail" if float(fixed["lag_ms"]) >= 288 else "reference"
            if health:
                state = "supported" if support_ok(ok, run.fs, ws, L) else "unsupported"
                counts[f"{group}:{state}"] += 1
                continue
            features = extract_features(
                run.acc_g, raw, ok, run.fs, L, onset, ws, theta,
                int(fixed["n_bouts"]), vectors)
            counts[f"{group}:{features['feature_status']}"] += 1
            output.append({
                **identity, "set": group, "fs": run.fs,
                "L_tick": L, "R_tick": onset, "ws_tick": ws,
                "lag_ms": float(fixed["lag_ms"]), "theta_g": theta,
                "rise_per_s": rise_rate(raw, ok, run.fs, onset, theta),
                **features,
            })
    if sum(counts.values()) != len(events):
        raise RuntimeError("A fixed event was not visited")
    return {"participant": p, "counts": dict(counts), "rows": output}


def known_positive_stage(args):
    root = output_root(args, "premature_r_diagnostic", first=True)
    folder = D.fresh(root / "known_positive")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    refs, _, _, low, high = inputs()
    people = scope(args)["participants"]
    results = D.pool_map(
        kp_worker,
        [(r, low, high, args.health_only) for r in refs if r["participant"] in people],
        args.workers)
    counts = {r["participant"]: r["counts"] for r in results}
    if args.health_only:
        D.write_json(folder / "health.json", {"counts": counts})
    else:
        rows = [row for result in results for row in result["rows"]]
        calibration = {r["participant"]: r["calibration"] for r in results}
        D.write_csv(folder / "synthetic_features.csv", rows)
        D.write_json(folder / "calibration.json", calibration)
        D.write_json(folder / "summary.json", kp_summary(rows, people, calibration))
        D.write_json(folder / "counts.json", counts)
    close_stage(folder, "prd_known_positive", args,
                provider_access_has_not_occurred=True)
    print("Known-positive stage closed", flush=True)


def features_stage(args):
    root = output_root(args, "premature_r_diagnostic", first=False)
    check_stage(root / "known_positive", "prd_known_positive", args)
    folder = D.fresh(root / "features")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    refs, _, events, _, _ = inputs()
    references = {r["participant"]: r for r in refs}
    people = scope(args)["participants"]
    groups = D.stems_by_person()
    results = D.pool_map(
        real_worker,
        [(p, groups[p], [r for r in events if r["participant"] == p],
          references[p], args.health_only) for p in people], args.workers)
    counts = {r["participant"]: r["counts"] for r in results}
    if args.health_only:
        D.write_json(folder / "health.json", {"counts": counts})
    else:
        rows = [row for result in results for row in result["rows"]]
        if len(rows) != 3964:
            raise RuntimeError("Feature stage lost fixed events")
        known = D.read_json(root / "known_positive/summary.json")
        D.write_csv(folder / "event_features.csv", rows)
        D.write_json(folder / "decision.json", prd_decision(rows, known))
        D.write_json(folder / "counts.json", counts)
    close_stage(
        folder, "prd_features", args,
        known_positive_sha256=D.digest(root / "known_positive/closed.json"),
        provider_access_has_not_occurred=True)
    print("PRD feature stage closed", flush=True)


def provider_stage(args):
    if args.health_only:
        print("Provider stage skipped in health mode", flush=True)
        return
    root = output_root(args, "premature_r_diagnostic", first=False)
    meta = check_stage(root / "features", "prd_features", args)
    check_stage(root / "known_positive", "prd_known_positive", args)
    if meta["known_positive_sha256"] != D.digest(root / "known_positive/closed.json"):
        raise RuntimeError("Known-positive predecessor changed")
    if meta.get("provider_access_has_not_occurred") is not True:
        raise RuntimeError("Missing PRD pre-provider boundary")
    if args.cpc_dir is None:
        raise ValueError("Provider stage requires --cpc-dir")
    strata = check_stage(args.cpc_dir / "strata", "cpc_strata", args)
    cpc = check_stage(args.cpc_dir / "controls", "cpc_controls", args)
    feature_hash = D.digest(root / "features/closed.json")
    if (
        cpc["prd_features_sha256"] != feature_hash
        or strata["prd_features_sha256"] != feature_hash
        or cpc["strata_sha256"] != D.digest(args.cpc_dir / "strata/closed.json")
        or cpc.get("provider_access_has_not_occurred") is not True
    ):
        raise RuntimeError("CPC pre-provider boundary differs")
    D.read_json(root / "features/decision.json")
    D.read_json(args.cpc_dir / "controls/decision.json")
    _, cues, _, _, _ = inputs()
    cue_map = {D.event_key(c): c for c in cues}
    providers, seen = {}, set()
    columns = (
        "participant", "session", "run", "cue",
        "cue_tick", "start_known_tick", "acc_start_tick",
    )
    for row in V4.projected_csv(
        FREEZE / "self_paced_cues.csv", columns, allow_provider=True
    ):
        key = D.event_key(row)
        if row["participant"] not in D.PEOPLE or key in seen or key not in cue_map:
            raise RuntimeError("Provider inventory differs")
        seen.add(key)
        for field in ("cue_tick", "start_known_tick"):
            if V4.optional_int(row[field]) != V4.optional_int(cue_map[key][field]):
                raise RuntimeError("Provider cue/start join differs")
        providers[key] = V4.provider_tick(row["acc_start_tick"])
    if len(seen) != 4432 or seen != set(cue_map):
        raise RuntimeError("Provider inventory is incomplete")
    rows = D.read_csv(root / "features/event_features.csv")

    def describe(selected):
        eligible = [r for r in selected if finite(r.get("R_tick"))]
        matched = [r for r in eligible if providers[D.event_key(r)] is not None]
        half = [r for r in matched if finite(r.get("t50_tick"))]

        def difference(group, field):
            return V4.distribution([
                float(r[field]) - providers[D.event_key(r)] for r in group])
        return {
            "events": len(eligible), "provider_matched": len(matched),
            "provider_missing": len(eligible) - len(matched),
            "R_minus_provider": difference(matched, "R_tick"),
            "L_minus_provider": difference(matched, "L_tick"),
            "t50_matched": len(half),
            "t50_missing_or_censored": len(matched) - len(half),
            "same_t50_rows": {
                "R_minus_provider": difference(half, "R_tick"),
                "t50_minus_provider": difference(half, "t50_tick"),
                "L_minus_provider": difference(half, "L_tick"),
            },
        }

    folder = D.fresh(root / "provider")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    D.write_json(folder / "summary.json", {
        "descriptive_only": True, "provider_flag_filtering": False,
        "by_set": {
            group: describe([r for r in rows if r["set"] == group])
            for group in ("tail", "reference")
        },
        "by_person": {
            p: {
                group: describe([
                    r for r in rows if r["participant"] == p and r["set"] == group])
                for group in ("tail", "reference")
            } for p in D.PEOPLE
        },
    })
    close_stage(
        folder, "prd_provider", args,
        prd_features_sha256=feature_hash,
        cpc_controls_sha256=D.digest(args.cpc_dir / "controls/closed.json"))
    print("Post-lock provider stage closed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("known-positive", "features", "provider"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cpc-dir", type=Path)
    parser.add_argument("--participants")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    D.require_slurm(args.workers)
    scope(args)
    code_hashes()
    {
        "known-positive": known_positive_stage,
        "features": features_stage,
        "provider": provider_stage,
    }[args.stage](args)


if __name__ == "__main__":
    main()
