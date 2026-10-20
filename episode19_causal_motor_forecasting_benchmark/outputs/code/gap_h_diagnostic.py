#!/usr/bin/env python3
"""EP19 fixed-label gap/H diagnostic; orchestrator execution only."""
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
from scipy.optimize import brentq
from scipy.special import expit, logit, xlogy
from scipy.stats import rankdata

import detector_v2 as D
import self_paced_raw_onset_test as V4
import self_paced_raw_onset_test_v5 as V5
import premature_r_diagnostic as P

SEED = 20261017
GAPS = (40, 80, 120, 160)
KINDS = ("KPG", "KPB", "KPA")
VIEWS = CODE.parent.parent / "outputs/runs/source_views_and_timing"
PRD = VIEWS / "premature_r_diagnostic_job_48281685"
NEW_SCRIPTS = (
    "gap_h_diagnostic.py",
    "plot_gap_h_diagnostic.py",
    "run_gap_h_diagnostic.sbatch",
)
STAGES = (
    ("known-positive", "known_positive", "ghd_known_positive"),
    ("features", "features", "ghd_features"),
    ("decision", "decision", "ghd_decision"),
    ("provider", "provider", "ghd_provider"),
)
PROTECTED = {
    "detector_freeze_job_48219799", "raw_onset_test_job_48230171",
    "detector_v2_job_48261284", "raw_onset_test_v5_job_48261284",
    "premature_r_diagnostic_job_48281685", "control_precision_job_48281685",
}

DECLARATION = """\
### October 17: fixed-label gap/H diagnostic — pre-declared before execution

This follow-up is post hoc to jobs 48261284 and 48281685; its geometry, synthetic, statistical and operational choices are assumptions. It tests the mixture left by job 48281685: (a) in most tail events (D2 k=5 lag ≥ 288 ms), R is premature, marking intermittent low-amplitude activity joined to the reach across internal non-qualifying runs; (b) in a minority, concentrated in a few people, a substantial departure (H ≥ 288 ms) precedes the label. It tests sensor-departure patterns, not reach identity.

1. Use only the fifteen development people, the fixed D2 labels, the v5 k=5 onsets and the closed PRD features. Re-read accelerometry only through PRD's reader path, for the declared gaps, the pre-split amplitude and the regressions. No EEG, WAY or held-out data.
2. Keep k, every v5 gate, the acceptance rule and the reference estimator unchanged. λ* is a diagnostic property of the fixed R→L interval, never an acceptance reference. Both v5 candidates stay UNDERIDENTIFIED, and nothing is frozen.
3. Before writing, verify the input closures, event joins and fixed counts. Write exclusive files into a new job directory. Run in a Slurm compute shell on AVX-512 nodes, with at most eight workers and one numerical thread each.
4. Rebuild each determinate event's selected bout on [ws − 720, L + 100]. Its R, ws and end must reproduce v5, and supported events must reproduce PRD's longest quiet run, t½ and A_F exactly. PRD's longest quiet run can include a terminal run through L, so it stays separate from the complete internal runs below.
5. An internal run is a complete run of non-qualifying window starts between two consecutive qualifying starts of the selected bout, starting at or before L. Its length is the start count times the native dt (at most 196/199 ms). Its depth is the median native r at those start ticks; the run is deep when that median is ≤ θ3 (the person's k=3 reference).
6. For g = 40/80/120/160 ms (primary 80), split after the last internal run that is both ≥ g long and deep. R* is the first exceedance of the qualifying window that ends that run; without such a run, R* = R. λ* = L − R* and can be negative. The any-run variant (no depth requirement), run details, D2 rejected crossings and the pre-split amplitude are descriptive.
7. Among supported tail events, class B has closed H_lo ≥ 288 and takes priority; censored H is never B. Otherwise class A has λ*_80 < 288 and class C has λ*_80 ≥ 288. These are operational sensor classes, not reach classes.
8. Component (a) predicts that most tail events are class A and that splitting at deep pauses removes the p95 tail. In the decision set, (a) is SUPPORTED when the person-median class-A fraction f_A has CI lower bound > 0.5 and the p95 of λ*_80 (P95*) has CI upper bound < 288. (a) is WEAKENED when f_A has CI upper bound < 0.5 or P95* has CI lower bound ≥ 288 (the p95 tail persists after splitting). Otherwise it is UNRESOLVED, or UNDERIDENTIFIED when a dependency fails.
9. The decision set needs at least twelve people, each with at least three supported tail events and at least 24 usable events of 32 for each of KPG, KPB and KPA. P95* over all 15 people and its CI are reported separately, without extending the subset's controls to excluded people. Every event and the original v5 bounds are kept; the two unresolved events get λ* bounds [−∞, +∞].
10. P95 intervals resample whole people 10,000 times (seed key 40), keeping event multiplicities and separate nearest-rank bounds. Fractions use the PRD person bootstrap. People are the independent units, not events or backgrounds.
11. KPB and KPA are regenerated with PRD's shapes, estimator path and direction stream; any mismatch with the stored rows stops the run. KPG adds two orthogonal flat-top bursts (20-ms edges) before PRD's fast reach step. The first of six ordered plateau/pause candidates that passes the noiseless checks is used: the amplitude rule, the label at the reach, a lag of 300–480 ms, two deep internal runs of 100–180 ms and λ*_80 < 200 ms. All candidates are recorded.
12. KPG_power (sensitivity): the person-median class-A fraction of usable KPG events has CI lower bound > 0.8. KP_specificity: for KPB and for KPA separately, the person-median fraction with λ*_80 < 288 (before B priority) has CI upper bound < 0.2. SUPPORTED needs both checks; WEAKENED needs KPG_power. A failed dependency makes only the affected statement UNDERIDENTIFIED.
13. Component (b) predicts that B events cluster by person and that the clustering repeats across independent halves of each person's runs. T2 applies a within-tail likelihood-ratio statistic G² to the person × B counts, against 10,000 conditional multivariate-hypergeometric draws (seed key 41). Its outcome was largely anticipated from the counts already released.
14. T3 splits each person's runs into halves by alternating position in session/run order and needs at least ten people with at least three supported tail events in each half. It correlates the halves' empirical logits of the B fraction, against 10,000 independent conditional allocations within each half that preserve the person denominators and the half totals (seed key 44).
15. Known positives for T2/T3: 2,000 datasets with shared person logit effects at σ = 0.75 and the observed expected prevalence, each with 2,000 conditional null draws (seed keys 42/43 and 45/46); σ = 0 is simulated too, and its rejection rate's Wilson upper bound must be ≤ 0.075. A non-significant real result counts as NOT_OBSERVED only with power ≥ 0.80; otherwise it is UNDERIDENTIFIED. These check the count pipeline; they do not validate H as reach onset.
16. (b) is SUPPORTED when T2 and T3 are both significant (p < 0.05), WEAKENED when both are qualified NOT_OBSERVED, and otherwise UNRESOLVED or UNDERIDENTIFIED. Descriptive: accepted-event denominators, leave-one-person-out T2, B_abs (also A_F ≥ 2θ5), H thresholds 250/320 and split-half ΔH correlations. The count tests assume exchangeable events within their conditioning; dependence between runs is not modelled.
17. Stage order: known positives, real features, decision, then provider agreement. AccStartIndex is read only after the decision closes, with verified cue/start joins, matched denominators and no flag filtering. Health mode releases engineering counts and statuses only and skips the decision, provider and plot stages.
18. Disclosure: beforehand the orchestrator had seen the job-48281685 summaries, the 54 H ≥ 288 events and three of their per-person counts, and the v5/CPC precision results. Run positions and depths, λ*, class fractions, half values and KPG results were unseen. Alternatives: one class scheme; four gaps, with 80 ms chosen after the PRD's "≥ 80 ms in 61% vs 5%" description; one depth criterion; six noiseless KPG candidates; one σ; one primary H threshold.
19. Report one six-panel 300-dpi figure and three summary lines. Implications: (a) SUPPORTED gives the scientist evidence on a continuity-revised reference (v6); (a) WEAKENED points to continuous or shallow-gap activity and leaves the onset-definition question open. (b) SUPPORTED shows person concentration under the declared count model; (b) WEAKENED does not establish absence or sporadic causes. No outcome changes the observed H exceedances, certifies worst-case timing or grants EEG scoring clearance. v6, gates and p95 versus worst case remain scientist decisions; unresolved outcomes trigger no automatic change.
"""


def num(value):
    return P.number(value)


def truth(value):
    return value is True or str(value).lower() in ("1", "true")


def json_values(value):
    """Preserve the direction of infinite censoring bounds in JSON."""
    if isinstance(value, dict):
        return {str(k): json_values(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def write_json(path, value):
    D.write_json(path, json_values(value))


def code_hashes():
    return {name: D.digest(CODE / name) for name in NEW_SCRIPTS}


def require_compute(workers):
    D.require_slurm(workers)
    features = np.core._multiarray_umath.__cpu_features__
    if not features.get("AVX512F", False):
        raise RuntimeError("Exact regressions require an AVX-512 node")


def scope(args):
    if args.participants is not None and not args.health_only:
        raise ValueError("Participant subsets require --health-only")
    return D.scope(args)


def input_heads():
    folders = {
        "D2_settings": P.DETECTOR / "settings",
        "D2_labels": P.DETECTOR / "D2",
        "labels_stage": P.DETECTOR / "labels_stage",
        "v5_calibration": P.V5RUN / "calibration",
        "v5_controls": P.V5RUN / "controls",
        "v5_events": P.V5RUN / "events",
        "PRD_known_positive": PRD / "known_positive",
        "PRD_features": PRD / "features",
    }
    return {name: D.digest(folder / "closed.json")
            for name, folder in folders.items()}


def check_stage(folder, stage, args=None):
    meta = D.require_closed(folder, stage)
    if meta.get("new_script_sha256") != code_hashes():
        raise RuntimeError("New gap/H scripts changed after closure")
    if args is not None:
        D.check_scope(meta, scope(args))
    return meta


def verify_chain(root, through, args=None):
    root = Path(root).resolve()
    predecessors, result = {}, {}
    first_scope, fixed_inputs = None, None
    for cli, name, stage in STAGES:
        meta = check_stage(root / name, stage, args)
        if Path(meta["run_root"]).resolve() != root:
            raise RuntimeError("Stage belongs to another run")
        if meta.get("predecessors") != predecessors:
            raise RuntimeError("Predecessor closure differs")
        current_scope = {
            k: meta.get(k)
            for k in ("participants", "participant_list", "health_only")
        }
        if first_scope is None:
            first_scope = current_scope
            fixed_inputs = meta["fixed_inputs"]
        elif current_scope != first_scope or meta["fixed_inputs"] != fixed_inputs:
            raise RuntimeError("Stage scope or fixed inputs differ")
        result[name] = meta
        predecessors = {
            **predecessors,
            name: D.digest(root / name / "closed.json"),
        }
        if cli == through:
            return result, predecessors
    raise ValueError("Unknown chain endpoint")


def expected_root(args):
    job = os.environ["SLURM_JOB_ID"]
    if args.health_only:
        if not os.environ.get("SCRATCH"):
            raise RuntimeError("Health mode requires SCRATCH")
        expected = (
            Path(os.environ["SCRATCH"]) / "br_autoresearch"
            / "episode19_causal_motor_forecasting_benchmark"
            / "gap_h_work" / f"smoke_{job}" / "gap_h_diagnostic"
        )
    else:
        expected = VIEWS / f"gap_h_diagnostic_job_{job}"
    proposed = args.output_dir
    if proposed.is_symlink() or proposed.resolve() != expected.resolve():
        raise RuntimeError(f"Output must be the new job root: {expected}")
    if PROTECTED.intersection(proposed.parts) or PROTECTED.intersection(
        proposed.resolve().parts
    ):
        raise RuntimeError("Output enters a preserved run")
    return proposed.resolve()


def close_stage(folder, args, fixed_inputs, predecessors):
    if code_hashes() != args.initial_script_hashes:
        raise RuntimeError("New scripts changed during execution")
    if input_heads() != fixed_inputs:
        raise RuntimeError("Fixed input closure changed during execution")
    stage = next(s for cli, _, s in STAGES if cli == args.stage)
    D.close(
        folder, stage, **scope(args),
        new_script_sha256=args.initial_script_hashes,
        fixed_inputs=fixed_inputs, predecessors=predecessors,
        run_root=str(args.output_dir.resolve()),
        job_id=os.environ["SLURM_JOB_ID"],
        provider_access_has_not_occurred=args.stage != "provider",
        independent_human_unit="participant",
    )


def same(a, b, *, textual=False):
    missing_a = a is None or a == ""
    missing_b = b is None or b == ""
    if missing_a or missing_b:
        return missing_a and missing_b
    return str(a) == str(b) if textual else float(a) == float(b)


def input_contract():
    refs, cues, events, low, high = P.inputs()
    old_scope = {
        "participants": list(D.PEOPLE),
        "participant_list": False, "health_only": False,
    }
    km = P.check_stage(PRD / "known_positive", "prd_known_positive")
    fm = P.check_stage(PRD / "features", "prd_features")
    D.check_scope(km, old_scope)
    D.check_scope(fm, old_scope)
    if fm["known_positive_sha256"] != D.digest(
        PRD / "known_positive/closed.json"
    ):
        raise RuntimeError("PRD predecessor differs")
    features = D.read_csv(PRD / "features/event_features.csv")
    fmap = {D.event_key(r): r for r in features}
    emap = {D.event_key(r): r for r in events}
    if len(features) != 3964 or len(fmap) != 3964 or set(fmap) != set(emap):
        raise RuntimeError("PRD/fixed-event inventory differs")

    counts = Counter()
    for key, event in emap.items():
        old = fmap[key]
        if event["status"] == "determinate":
            group = "tail" if num(event["lag_ms"]) >= 288 else "reference"
            if old["set"] != group:
                raise RuntimeError("PRD set join differs")
            for a, b in (
                ("L_tick", "L_tick"), ("R_tick", "R_tick"),
                ("candidate_window_start_tick", "ws_tick"),
                ("lag_ms", "lag_ms"), ("fs", "fs"),
                ("theta_g", "theta_g"),
            ):
                if not same(event[a], old[b]):
                    raise RuntimeError(f"PRD event join differs: {a}")
            if old["feature_status"] not in ("ok", "weak_later_segment"):
                raise RuntimeError("Unexpected fixed PRD feature status")
            if old["feature_status"] == "ok" and P.feature_bounds(old, "H") is None:
                raise RuntimeError("Supported PRD event lacks H bounds")
        else:
            group = "unresolved"
            if (
                event["status"] != "indeterminate"
                or old["set"] != group
                or old["feature_status"] != "unresolved"
                or any(old.get(f, "") not in ("", None)
                       for f in ("L_tick", "R_tick", "ws_tick", "lag_ms"))
            ):
                raise RuntimeError("Unresolved PRD join differs")
        counts[(group, old["feature_status"])] += 1

    expected = {
        ("tail", "ok"): 289,
        ("tail", "weak_later_segment"): 2,
        ("reference", "ok"): 3661,
        ("reference", "weak_later_segment"): 10,
        ("unresolved", "unresolved"): 2,
    }
    if dict(counts) != expected:
        raise RuntimeError("Fixed PRD count contract differs")
    unresolved = [r for r in events if r["status"] != "determinate"]
    if (
        {r["participant"] for r in unresolved} != {"sub-02"}
        or sorted(num(r["lag_lower_ms"]) for r in unresolved) != [1672, 2484]
        or any(num(r["lag_upper_ms"]) != float("inf") for r in unresolved)
    ):
        raise RuntimeError("Fixed unresolved bounds differ")
    if any(
        V4.nearest_rank([num(r[f]) for r in events]) != 324
        for f in ("lag_lower_ms", "lag_upper_ms")
    ):
        raise RuntimeError("Fixed v5 p95 differs")
    if sum(
        r["feature_status"] == "ok" and num(r.get("H_lo")) >= 288
        for r in features
    ) != 54:
        raise RuntimeError("Released PRD H-count contract differs")

    synthetic = D.read_csv(PRD / "known_positive/synthetic_features.csv")
    skey = lambda r: (r["participant"], int(r["background_id"]), r["type"])
    smap = {skey(r): r for r in synthetic}
    wanted = {
        (p, bid, kind)
        for p in D.PEOPLE for bid in range(32)
        for kind in ("KP0", "KPB", "KPA")
    }
    if len(synthetic) != 1440 or len(smap) != 1440 or set(smap) != wanted:
        raise RuntimeError("PRD synthetic inventory differs")
    return {
        "refs": refs, "cues": cues, "events": events,
        "features": fmap, "synthetic": smap,
        "low": low, "high": high, "fixed_inputs": input_heads(),
    }


def classify(row, *, any_run=False):
    if row.get("feature_status") != "ok":
        return ""
    bounds = P.feature_bounds(row, "H")
    if bounds is None:
        raise RuntimeError("Supported classification lacks H bounds")
    if bounds[0] >= 288:
        return "B"
    field = "lambda_any_80_ms" if any_run else "lambda_80_ms"
    if not P.finite(row.get(field)):
        raise RuntimeError("Supported classification lacks diagnostic interval")
    return "A" if num(row[field]) < 288 else "C"


def gap_features(acc, raw, ok, fs, L, onset, ws, theta, theta3,
                 *, expected_end=None, amplitude=None):
    """Complete internal start-coordinate runs; no new onset estimation."""
    dt = int(round(1000 / fs))
    width = 40 // dt
    if not P.support_ok(ok, fs, ws, L):
        raise RuntimeError("Fixed diagnostic slice lacks native support")
    begin_tick = ws - 720
    begin = begin_tick // dt
    stop = (L + 100) // dt + 1
    hot, q, starts, first_hot = V4.qualifying_windows(
        raw[begin:stop], theta, width)
    bouts = V4.grouped_bouts(
        hot, starts, first_hot, width, 200 // dt, window_quiet=True)
    selected = [
        b for b in bouts
        if (begin + int(b[0])) * dt == onset
        and (begin + int(b[2])) * dt == ws
    ]
    if len(selected) != 1:
        raise RuntimeError("Fixed selected bout R/ws did not reconstruct")
    bout = selected[0]
    end_tick = (begin + int(bout[1])) * dt
    if expected_end is not None and end_tick != expected_end:
        raise RuntimeError("Fixed selected bout end did not reconstruct")
    anchor_start = (ws - 200 - begin_tick) // dt
    first_index = (ws - begin_tick) // dt
    if np.any(q[anchor_start:first_index]) or not q[first_index]:
        raise RuntimeError("Fixed selected bout anchor did not reconstruct")

    positions = np.flatnonzero(
        (starts >= bout[2]) & (starts <= bout[1] - width + 1))
    runs = []
    for left_pos, right_pos in zip(positions[:-1], positions[1:]):
        a, b = int(starts[left_pos]), int(starts[right_pos])
        length = (b - a - 1) * dt
        if length == 0:
            continue
        first_tick = (begin + a + 1) * dt
        if first_tick > L:
            continue
        if not 0 < length <= 200 - dt:
            raise RuntimeError("Internal start-run geometry differs")
        median_g = float(np.median(raw[begin + a + 1:begin + b]))
        next_hot = (begin + int(first_hot[right_pos])) * dt
        if next_hot < onset:
            raise RuntimeError("Post-split exceedance precedes fixed R")
        runs.append({
            "start_tick": first_tick,
            "stop_tick": (begin + b) * dt,
            "previous_qualifying_tick": (begin + a) * dt,
            "next_qualifying_tick": (begin + b) * dt,
            "next_first_hot_tick": next_hot,
            "length_ms": length,
            "median_r_g": median_g,
            "depth_r_theta5": median_g / theta,
            "deep": int(median_g <= theta3),
            "label_inside": int(first_tick <= L < (begin + b) * dt),
        })

    # This deliberately reproduces PRD's potentially truncated quiet statistic.
    quiet = ~q[first_index:(L + dt - begin_tick) // dt]
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    longest_prd = (
        int(np.max(edges[1::2] - edges[::2])) * dt if len(edges) else 0
    )
    longest = max(runs, key=lambda r: r["length_ms"]) if runs else None
    out = {
        "reconstructed_bout_end_tick": end_tick,
        "longest_quiet_ms": longest_prd,
        "n_internal_runs": len(runs),
        "longest_internal_ms": longest["length_ms"] if longest else 0,
        "longest_run_depth": longest["depth_r_theta5"] if longest else None,
        "longest_run_start_tick": longest["start_tick"] if longest else None,
    }

    magnitudes = None
    if P.finite(amplitude) and float(amplitude) >= theta and onset <= L:
        b0 = np.mean(acc[(onset - 520) // dt:(onset - 20) // dt], axis=0)
        local = acc[onset // dt:(L + 100) // dt] - b0
        means = D.S.trailing_mean(local, width)[width - 1:]
        magnitudes = np.linalg.norm(means, axis=1) / float(amplitude)

    for g in GAPS:
        long = [r for r in runs if r["length_ms"] >= g]
        deep = [r for r in long if r["deep"]]
        out[f"n_long_{g}"] = len(long)
        out[f"n_long_deep_{g}"] = len(deep)
        for any_run, eligible in ((False, deep), (True, long)):
            token = f"any_{g}" if any_run else str(g)
            split = eligible[-1] if eligible else None
            Rstar = split["next_first_hot_tick"] if split else onset
            lag = L - Rstar
            if lag > L - onset:
                raise RuntimeError("Diagnostic interval increased after splitting")
            out[f"{'Rany' if any_run else 'Rstar'}_{g}_tick"] = Rstar
            out[f"lambda_{token}_ms"] = lag
            out[f"lambda_{token}_lower_ms"] = lag
            out[f"lambda_{token}_upper_ms"] = lag
            for name, source in (
                ("start_tick", "start_tick"), ("stop_tick", "stop_tick"),
                ("length_ms", "length_ms"), ("depth", "depth_r_theta5"),
            ):
                out[f"split_{token}_{name}"] = split[source] if split else None
            out[f"label_in_{'any' if any_run else 'deep'}_run_{g}"] = (
                split["label_inside"] if split else 0)
            out[f"negative_lambda_{token}"] = int(lag < 0)
            rho = None
            if split and magnitudes is not None:
                nstarts = max(0, (split["start_tick"] - onset + dt - 1) // dt)
                values = magnitudes[:nstarts]
                if len(values):
                    rho = float(np.max(values))
            out[f"rho_pre_{token}"] = rho
    return out, runs


def shift_native(values, shift_ms, dt):
    if shift_ms % dt:
        raise RuntimeError("Template translation is off the native grid")
    indices = np.arange(len(values)) - shift_ms // dt
    return np.asarray(values)[np.clip(indices, 0, len(values) - 1)].copy()


def kpg_shape(reference, low, high, prd_shapes):
    """New bursts; the final fast step is imported from PRD's KP0."""
    fs = float(reference["fs"])
    dt = int(round(1000 / fs))
    theta = V4.reference_theta(reference, 5)
    theta3 = V4.reference_theta(reference, 3)
    ticks = np.arange(2800 // dt) * dt
    desired = 2163 if fs == 1000 else 2160
    candidates = ((50, 110), (40, 110), (50, 100),
                  (60, 100), (40, 120), (50, 120))
    chosen, chosen_info, records = None, None, []
    fast = prd_shapes.get("KP0", (None, None))[0]

    def burst(start, plateau):
        rise = .5 - .5 * np.cos(
            np.pi * np.clip((ticks - start) / 20, 0, 1))
        fall = .5 - .5 * np.cos(
            np.pi * np.clip((ticks - start - 20 - plateau) / 20, 0, 1))
        return rise - fall

    for index, (plateau, pause) in enumerate(candidates):
        record = {
            "type": "KPG", "candidate": index,
            "plateau_ms": plateau, "pause_ms": pause,
            "selected": False, "calibration_usable": False,
        }
        records.append(record)
        if fast is None:
            record["reason"] = "PRD_KP0_step_uncovered"
            continue
        t0 = 1200
        second = t0 + 40 + plateau + pause
        reach = second + 40 + plateau + pause
        unit = np.column_stack((
            np.zeros(len(ticks)), burst(t0, plateau), burst(second, plateau)))
        ur = V4.native_statistic(unit, fs)
        reduced, detector_ticks = D.R.reduce_to_detector_rate(unit, fs)
        ux = D.S.self_paced_departure(reduced)
        plateau_mask = (
            ((ticks >= t0 + 20) & (ticks < t0 + 20 + plateau))
            | ((ticks >= second + 20) & (ticks < second + 20 + plateau))
        )
        before_reach = (detector_ticks >= t0) & (detector_ticks < reach)
        minimum_r = float(np.min(ur[plateau_mask]))
        maximum_x = float(np.max(ux[before_reach]))
        if minimum_r <= 0 or maximum_x <= 0:
            record["reason"] = "invalid_unit_amplitude_support"
            continue
        lower = 1.3 * theta / minimum_r
        upper = .85 * high / maximum_x
        record.update(
            lower_g=lower, upper_g=upper,
            unit_plateau_min_r=minimum_r, unit_burst_max_X=maximum_x)
        if lower > upper:
            record["reason"] = "infeasible_burst_amplitude"
            continue
        amplitude = min(1.15 * lower, upper)
        trial = amplitude * unit
        trial[:, 0] = shift_native(
            fast, reach - (desired + 350), dt)
        first = V5.first_detectable(
            V4.native_statistic(trial, fs), fs, theta, t0)
        if first is None or first >= t0 + 40 + plateau:
            record["reason"] = "first_burst_not_persistently_detectable"
            continue
        shift = desired - first
        signal = shift_native(trial, shift, dt)
        reach_tick = reach + shift
        record.update(amplitude_g=amplitude, shift_ms=shift,
                      reach_start_tick=reach_tick)
        event, raw, ok = P.synthetic_estimate(
            signal, fs, theta, low, high, noiseless=True)
        record.update(event)
        if event["status"] != "determinate":
            record["reason"] = "noiseless_" + event["reason"]
            continue
        L, onset, ws = (int(event[f]) for f in ("L_tick", "R_tick", "ws_tick"))
        geometry, runs = gap_features(
            signal, raw, ok, fs, L, onset, ws, theta, theta3)
        reduced, xticks = D.R.reduce_to_detector_rate(signal, fs)
        departure = D.S.self_paced_departure(reduced)
        before = np.flatnonzero(xticks < reach_tick)
        active_before = (xticks >= 1920) & (xticks < reach_tick)
        crossing = np.flatnonzero(
            (xticks >= reach_tick) & (departure >= low))
        label_ok = bool(
            len(before) and departure[before[-1]] < low
            and np.all(departure[active_before] < high)
            and len(crossing) and L == int(xticks[crossing[0]])
            and reach_tick <= L <= reach_tick + 80
        )
        tests = {
            "R_aligned": onset == desired,
            "reach_label": label_ok,
            "lag_300_480": 300 <= float(event["lag_ms"]) <= 480,
            "two_complete_internal_runs": len(runs) == 2,
            "gap_lengths_100_180": bool(runs) and all(
                100 <= r["length_ms"] <= 180 for r in runs),
            "gap_depth_margin": bool(runs) and all(
                r["median_r_g"] <= .8 * theta3 for r in runs),
            "lambda80_under_200": geometry["lambda_80_ms"] < 200,
        }
        good = all(tests.values())
        record.update(
            checks=tests, internal_runs=runs,
            lambda_80_ms=geometry["lambda_80_ms"],
            calibration_usable=good,
            reason="selected_geometry_eligible" if good
            else "noiseless_geometry_or_label_failed",
        )
        if good and chosen is None:
            chosen, chosen_info = signal, {"reach_start_tick": reach_tick}
            record["selected"] = True
    return chosen, chosen_info, records


def regress_synthetic(new, old):
    if (new.get("feature_status") == "uncovered_shape") != (
        old.get("feature_status") == "uncovered_shape"
    ):
        raise RuntimeError("PRD synthetic uncovered-shape regression failed")
    for field in ("L_tick", "R_tick", "lag_ms", "ws_tick", "status"):
        if not same(new.get(field, ""), old.get(field, ""),
                    textual=field == "status"):
            raise RuntimeError(f"PRD synthetic regression failed: {field}")


def kp_worker(task):
    reference, low, high, old_rows, health = task
    D.require_slurm(1)
    p, fs = reference["participant"], float(reference["fs"])
    theta = V4.reference_theta(reference, 5)
    theta3 = V4.reference_theta(reference, 3)
    shapes, calibration = P.kp_shapes(reference, low, high)
    kpg, kpg_info, kpg_records = kpg_shape(reference, low, high, shapes)
    calibration += kpg_records
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, int(p.split("-")[1]), 7]))
    output, run_output, counts = [], [], Counter()
    counts["KPG_calibration_selected"] = int(kpg is not None)
    with np.load(P.V5RUN / "calibration" / f"{p}_backgrounds.npz",
                 allow_pickle=False) as cache:
        backgrounds = cache["acc"]
        if float(cache["fs"]) != fs or len(backgrounds) != 32:
            raise RuntimeError("Known-positive background inventory differs")
        for bid, background in enumerate(backgrounds):
            # Keep this arithmetic and stream consumption identical to PRD.
            e1 = rng.normal(size=3)
            e1 /= np.linalg.norm(e1)
            e2 = rng.normal(size=3)
            e2 -= np.dot(e2, e1) * e1
            if np.linalg.norm(e2) <= 1e-12:
                raise RuntimeError("Degenerate PRD direction pair")
            e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1, e2)
            for kind in ("KPB", "KPA", "KPG"):
                identity = {
                    "participant": p, "background_id": bid, "type": kind}
                available = kpg is not None if kind == "KPG" else kind in shapes
                if not available:
                    row = {**identity, "feature_status": "uncovered_shape",
                           "usable": False}
                else:
                    if kind == "KPG":
                        injected = (
                            background + kpg[:, 0, None] * e1
                            + kpg[:, 1, None] * e2
                            + kpg[:, 2, None] * e3
                        )
                    else:
                        along, perpendicular = shapes[kind]
                        injected = (
                            background + along[:, None] * e1
                            + perpendicular[:, None] * e2
                        )
                    event, raw, ok = P.synthetic_estimate(
                        injected, fs, theta, low, high)
                    row = {**identity, **event, "feature_status": event["status"]}
                    if event["status"] == "determinate":
                        L, onset, ws = (
                            int(event[f]) for f in ("L_tick", "R_tick", "ws_tick"))
                        features = P.extract_features(
                            injected, raw, ok, fs, L, onset, ws,
                            theta, int(event["n_bouts"]))
                        row.update(features)
                        geometry, runs = gap_features(
                            injected, raw, ok, fs, L, onset, ws, theta, theta3,
                            amplitude=features.get("A_F_g"))
                        row.update(geometry)
                        counts["gap_estimates_checked"] += 1
                        if not health:
                            run_output.extend({**identity, **r} for r in runs)
                    usable = (
                        event["status"] == "determinate" and P.kp_usable(row))
                    if kind == "KPG":
                        reach = kpg_info["reach_start_tick"]
                        reach_label = (
                            P.finite(row.get("L_tick"))
                            and reach <= num(row["L_tick"]) <= reach + 80)
                        row.update(
                            reach_start_tick=reach,
                            reach_label=bool(reach_label))
                        usable = usable and reach_label
                        if not reach_label:
                            counts["KPG_label_outside_reach_step"] += 1
                    row["usable"] = bool(usable)
                    if usable and not health:
                        row["class"] = classify(row)
                        row["class_any"] = classify(row, any_run=True)
                if kind in ("KPB", "KPA"):
                    regress_synthetic(row, old_rows[(p, bid, kind)])
                    counts["regression_rows_passed"] += 1
                counts[f"{kind}:attempted"] += 1
                counts[f"{kind}:status:{row['feature_status']}"] += 1
                counts[f"{kind}:usable"] += int(row["usable"])
                if not health:
                    output.append(row)
    return {
        "participant": p, "rows": output, "runs": run_output,
        "counts": dict(counts),
        "calibration": [] if health else calibration,
    }


def kp_summary(rows, people):
    result = {}
    for p in people:
        result[p] = {}
        for kind in KINDS:
            all_rows = [
                r for r in rows if r["participant"] == p and r["type"] == kind]
            used = [r for r in all_rows if truth(r.get("usable"))]
            n = len(used)
            result[p][kind] = {
                "attempted": len(all_rows), "usable": n, "covered": n >= 24,
                "status_counts": dict(Counter(
                    r["feature_status"] for r in all_rows)),
                "class_fraction": {
                    c: sum(r["class"] == c for r in used) / n if n else None
                    for c in ("A", "B", "C")
                },
                "any_class_fraction": {
                    c: sum(r["class_any"] == c for r in used) / n if n else None
                    for c in ("A", "B", "C")
                },
                "split_fraction": sum(
                    num(r["lambda_80_ms"]) < 288 for r in used) / n if n else None,
                "any_split_fraction": sum(
                    num(r["lambda_any_80_ms"]) < 288 for r in used) / n
                if n else None,
            }
    return {"people": result, "minimum_usable": 24}


def known_positive_stage(args, folder, data):
    people = scope(args)["participants"]
    results = D.pool_map(
        kp_worker,
        [
            (r, data["low"], data["high"], {
                k: v for k, v in data["synthetic"].items()
                if k[0] == r["participant"]
            }, args.health_only)
            for r in data["refs"] if r["participant"] in people
        ],
        args.workers,
    )
    counts = {r["participant"]: r["counts"] for r in results}
    expected = 64 * len(people)
    if sum(c["regression_rows_passed"] for c in counts.values()) != expected:
        raise RuntimeError("Synthetic regression did not visit every required row")
    if args.health_only:
        write_json(folder / "health.json", {
            "status": "PASS", "regression": "PASS", "counts": counts})
        return
    rows = [r for result in results for r in result["rows"]]
    if len(rows) != 96 * len(people):
        raise RuntimeError("Known-positive output inventory differs")
    D.write_csv(folder / "synthetic_features.csv", rows)
    D.write_csv(
        folder / "internal_runs.csv",
        [r for result in results for r in result["runs"]],
        ("participant", "background_id", "type", "start_tick", "stop_tick",
         "previous_qualifying_tick", "next_qualifying_tick",
         "next_first_hot_tick", "length_ms", "median_r_g",
         "depth_r_theta5", "deep", "label_inside"),
    )
    write_json(folder / "calibration.json", {
        r["participant"]: r["calibration"] for r in results})
    write_json(folder / "summary.json", kp_summary(rows, people))
    write_json(folder / "counts.json", {
        "regression": "PASS", "regression_rows": expected, "people": counts})


def run_halves(cues):
    inventories = defaultdict(set)
    for cue in cues:
        inventories[cue["participant"]].add((cue["session"], int(cue["run"])))
    if sum(len(v) for v in inventories.values()) != 155:
        raise RuntimeError("Full development run inventory differs")
    return {
        (p, session, run): 1 + index % 2
        for p in D.PEOPLE
        for index, (session, run) in enumerate(sorted(inventories[p]))
    }


def real_worker(task):
    p, stems, events, old_rows, reference, cues, health = task
    D.require_slurm(1)
    theta = V4.reference_theta(reference, 5)
    theta3 = V4.reference_theta(reference, 3)
    by_run = defaultdict(list)
    for row in events:
        by_run[(row["session"], int(row["run"]))].append(row)
    output, run_output, counts = [], [], Counter()
    for stem in stems:
        _, session, run_number = V4.stem_key(stem)
        selected = by_run.get((session, run_number), [])
        if not selected:
            continue
        run, _, _, _, valid, _, _ = D.RF.sp_load(stem)
        if run.participant != p or run.fs != float(reference["fs"]):
            raise RuntimeError("Native run identity/rate differs")
        raw = V4.native_statistic(run.acc_g, run.fs)
        ok = V4.native_validity(raw, valid, run.fs)
        vectors = V4.native_residual(run.acc_g, run.fs)
        for fixed in selected:
            key = D.event_key(fixed)
            old = old_rows[key]
            identity = {
                "participant": p, "session": session,
                "run": run_number, "cue": fixed["cue"],
            }
            row = {
                **identity, "status": fixed["status"],
                "set": old["set"], "feature_status": old["feature_status"],
                "fs": float(fixed["fs"]), "theta_g": theta,
                "theta_k3_g": theta3,
                "L_tick": int(fixed["L_tick"]),
                "R_tick": fixed["R_tick"],
                "ws_tick": fixed["candidate_window_start_tick"],
                "bout_end_tick": fixed["bout_end_tick"],
                "lag_ms": fixed["lag_ms"],
                "lag_lower_ms": num(fixed["lag_lower_ms"]),
                "lag_upper_ms": num(fixed["lag_upper_ms"]),
                "rejected_crossings": int(cues[key]["rejected_crossings"]),
                **{f: old.get(f, "") for f in
                   ("H", "H_lo", "H_hi", "H_censored", "t50_tick", "A_F_g")},
            }
            counts["events_visited"] += 1
            counts[f"status:{old['feature_status']}"] += 1
            if fixed["status"] != "determinate":
                for g in GAPS:
                    for token in (str(g), f"any_{g}"):
                        row[f"lambda_{token}_lower_ms"] = -float("inf")
                        row[f"lambda_{token}_upper_ms"] = float("inf")
                if not health:
                    output.append(row)
                continue
            L = int(fixed["L_tick"])
            onset = int(fixed["R_tick"])
            ws = int(fixed["candidate_window_start_tick"])
            reproduced = P.extract_features(
                run.acc_g, raw, ok, run.fs, L, onset, ws, theta,
                int(fixed["n_bouts"]), vectors)
            if reproduced["feature_status"] != old["feature_status"]:
                raise RuntimeError("PRD feature-status regression failed")
            geometry, runs = gap_features(
                run.acc_g, raw, ok, run.fs, L, onset, ws, theta, theta3,
                expected_end=int(fixed["bout_end_tick"]),
                amplitude=old.get("A_F_g"))
            counts["bout_regressions_passed"] += 1
            if old["feature_status"] == "ok":
                for field in ("longest_quiet_ms", "t50_tick", "A_F_g"):
                    if not same(reproduced.get(field), old.get(field)):
                        raise RuntimeError(f"PRD exact feature regression failed: {field}")
                if not same(geometry["longest_quiet_ms"], old["longest_quiet_ms"]):
                    raise RuntimeError("PRD quiet-run regression failed")
                counts["feature_regressions_passed"] += 1
            if not health:
                row.update(geometry)
                if old["set"] == "tail" and old["feature_status"] == "ok":
                    row["class"] = classify(row)
                    row["class_any"] = classify(row, any_run=True)
                output.append(row)
                run_output.extend({**identity, **r} for r in runs)
    if counts["events_visited"] != len(events):
        raise RuntimeError("Real feature pass lost a fixed event")
    return {"participant": p, "rows": output, "runs": run_output,
            "counts": dict(counts)}


def features_stage(args, folder, data):
    people = scope(args)["participants"]
    refs = {r["participant"]: r for r in data["refs"]}
    cues = {D.event_key(r): r for r in data["cues"]}
    groups = D.stems_by_person()
    halves = run_halves(data["cues"])
    results = D.pool_map(
        real_worker,
        [
            (p, groups[p],
             [r for r in data["events"] if r["participant"] == p],
             {k: r for k, r in data["features"].items() if k[0] == p},
             refs[p], {k: r for k, r in cues.items() if k[0] == p},
             args.health_only)
            for p in people
        ],
        args.workers,
    )
    counts = {r["participant"]: r["counts"] for r in results}
    if args.health_only:
        write_json(folder / "health.json", {
            "status": "PASS", "regression": "PASS", "counts": counts})
        return
    rows = [r for result in results for r in result["rows"]]
    if len(rows) != 3964 or len({D.event_key(r) for r in rows}) != 3964:
        raise RuntimeError("Real feature inventory differs")
    if (
        sum(c.get("bout_regressions_passed", 0) for c in counts.values()) != 3962
        or sum(c.get("feature_regressions_passed", 0)
               for c in counts.values()) != 3950
    ):
        raise RuntimeError("Real regression denominator differs")
    for row in rows:
        row["half"] = halves[(row["participant"], row["session"], int(row["run"]))]
    rows.sort(key=D.event_key)
    D.write_csv(folder / "event_features.csv", rows)
    D.write_csv(
        folder / "internal_runs.csv",
        [r for result in results for r in result["runs"]],
        ("participant", "session", "run", "cue", "start_tick", "stop_tick",
         "previous_qualifying_tick", "next_qualifying_tick",
         "next_first_hot_tick", "length_ms", "median_r_g",
         "depth_r_theta5", "deep", "label_inside"),
    )
    write_json(folder / "run_halves.json", [
        {"participant": p, "session": session, "run": run, "half": half}
        for (p, session, run), half in sorted(halves.items())
    ])
    write_json(folder / "counts.json", {"regression": "PASS", "people": counts})


def p95_bounds(rows, people, lower, upper, *, bootstrap=False):
    selected = [r for r in rows if r["participant"] in people]
    if not selected or not people:
        return {"n": 0, "n_people": len(people),
                "bounds": [None, None], "ci95": [None, None]}
    arrays = [
        np.asarray([num(r[field]) for r in selected], dtype=float)
        for field in (lower, upper)
    ]
    if any(np.any(np.isnan(x)) for x in arrays) or np.any(arrays[0] > arrays[1]):
        raise RuntimeError("Invalid p95 censoring bounds")
    result = {
        "n": len(selected), "n_people": len(people),
        "bounds": [V4.nearest_rank(x) for x in arrays],
        "ci95": [None, None],
    }
    if not bootstrap:
        return result
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 40]))
    sampled = rng.integers(0, len(people), size=(10000, len(people)))
    multiplicity = np.column_stack([
        np.sum(sampled == j, axis=1) for j in range(len(people))
    ]).astype(np.int64)
    person_sizes = np.asarray([
        sum(r["participant"] == p for r in selected) for p in people
    ], dtype=np.int64)
    targets = np.ceil(.95 * (multiplicity @ person_sizes)).astype(np.int64)
    boot_bounds = []
    for field, values in zip((lower, upper), arrays):
        support = np.unique(values)
        counts = np.zeros((len(people), len(support)), dtype=np.int64)
        for j, p in enumerate(people):
            x = np.asarray([num(r[field]) for r in selected
                            if r["participant"] == p])
            counts[j] = np.bincount(
                np.searchsorted(support, x), minlength=len(support))
        cumulative = np.cumsum(counts, axis=1)
        boot = np.empty(10000)
        for start in range(0, 10000, 256):
            stop = min(start + 256, 10000)
            pooled = multiplicity[start:stop] @ cumulative
            hit = np.argmax(pooled >= targets[start:stop, None], axis=1)
            boot[start:stop] = support[hit]
        boot_bounds.append(boot)
    result["ci95"] = [
        float(np.quantile(boot_bounds[0], .025, method="inverted_cdf")),
        float(np.quantile(boot_bounds[1], .975, method="inverted_cdf")),
    ]
    return result


def tail_people(rows):
    result = {}
    for p in D.PEOPLE:
        group = [
            r for r in rows if r["participant"] == p
            and r["set"] == "tail" and r["feature_status"] == "ok"
        ]
        counts = {c: sum(r["class"] == c for r in group)
                  for c in ("A", "B", "C")}
        result[p] = {
            "n": len(group), "counts": counts,
            "fractions": {
                c: counts[c] / len(group) if group else None for c in counts
            },
            "any_fractions": {
                c: sum(r["class_any"] == c for r in group) / len(group)
                if group else None for c in counts
            },
        }
    return result


def known_population(known, people):
    metrics = {}
    for kind in KINDS:
        metrics[kind] = {}
        for name in ("class_A", "split", "any_class_A", "any_split"):
            values = []
            for p in people:
                item = known["people"][p][kind]
                value = {
                    "class_A": item["class_fraction"]["A"],
                    "split": item["split_fraction"],
                    "any_class_A": item["any_class_fraction"]["A"],
                    "any_split": item["any_split_fraction"],
                }[name]
                values.append(value)
            metrics[kind][name] = P.bootstrap_people(values)
    enough = len(people) >= 12
    power = bool(
        enough and metrics["KPG"]["class_A"]["ci95"][0] is not None
        and metrics["KPG"]["class_A"]["ci95"][0] > .8)
    specificity = bool(
        enough and all(
            metrics[k]["split"]["ci95"][1] is not None
            and metrics[k]["split"]["ci95"][1] < .2
            for k in ("KPB", "KPA")))
    return {
        "people": people, "metrics": metrics,
        "KPG_power": power, "KP_specificity": specificity,
        "specificity_uses_pre_priority_split": True,
    }


def wilson(successes, trials):
    if trials <= 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials
                         + z * z / (4 * trials * trials)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def g_statistic(b, n):
    b = np.asarray(b, dtype=float)
    n = np.asarray(n, dtype=float)
    total = b.sum(axis=-1, keepdims=True)
    expected = n * total / n.sum()
    other = n - b
    terms = (
        xlogy(b, b) - xlogy(b, expected)
        + xlogy(other, other) - xlogy(other, n - expected)
    )
    return np.maximum(0.0, 2 * np.sum(terms, axis=-1))


def hypergeom(rng, n, total, size):
    return rng.multivariate_hypergeometric(
        np.asarray(n, dtype=np.int64), int(total), size=size, method="marginals")


def t2_test(n, b, rng, draws=10000):
    n, b = np.asarray(n, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if len(n) < 2 or n.sum() == 0:
        return {"statistic": None, "p": None, "informative": False}
    observed = float(g_statistic(b, n))
    total = int(b.sum())
    informative = 0 < total < int(n.sum())
    if not informative:
        return {"statistic": observed, "p": 1.0, "informative": False}
    null = hypergeom(rng, n, total, draws)
    exceed = np.count_nonzero(g_statistic(null, n) >= observed - 1e-12)
    return {
        "statistic": observed, "p": (1 + int(exceed)) / (draws + 1),
        "informative": True, "null_draws": draws,
    }


def empirical_logit(b, n):
    return np.log((np.asarray(b, dtype=float) + .5)
                  / (np.asarray(n, dtype=float) - b + .5))


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))
    numerator = np.sum(x * y, axis=-1)
    answer = np.divide(
        numerator, den, out=np.zeros_like(numerator, dtype=float), where=den > 0)
    return np.clip(answer, -1, 1), den > 0


def t3_test(n, b, rng, draws=10000):
    n, b = np.asarray(n, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if n.ndim != 2 or len(n) < 2 or np.any(n.sum(axis=0) == 0):
        return {"r": None, "p": None, "informative": False}
    values = empirical_logit(b, n)
    observed, defined = correlation(values[:, 0], values[:, 1])
    if not bool(defined):
        return {"r": None, "p": 1.0, "informative": False}
    # Each half preserves its observed total and every person's denominator.
    null_halves = [
        hypergeom(rng, n[:, h], b[:, h].sum(), draws) for h in range(2)
    ]
    a = empirical_logit(null_halves[0], n[:, 0])
    c = empirical_logit(null_halves[1], n[:, 1])
    null_r, _ = correlation(a, c)
    exceed = np.count_nonzero(null_r >= float(observed) - 1e-12)
    return {
        "r": float(observed), "p": (1 + int(exceed)) / (draws + 1),
        "informative": True, "null_draws": draws,
        "null": "independent_conditional_allocations_within_each_half",
    }


def h_positive(row, threshold=288, absolute=False):
    bounds = P.feature_bounds(row, "H")
    return bool(
        bounds is not None and bounds[0] >= threshold
        and (not absolute or num(row["A_F_g"]) >= 2 * num(row["theta_g"])))


def count_analysis(rows, *, tail=True, threshold=288, absolute=False, seed_id=0):
    eligible = [
        r for r in rows if r["feature_status"] == "ok"
        and (not tail or r["set"] == "tail")]
    people = []
    for p in D.PEOPLE:
        group = [r for r in eligible if r["participant"] == p]
        half = [[r for r in group if int(r["half"]) == h] for h in (1, 2)]
        n = len(group)
        b = sum(h_positive(r, threshold, absolute) for r in group)
        nh = [len(g) for g in half]
        bh = [sum(h_positive(r, threshold, absolute) for r in g) for g in half]
        people.append({
            "participant": p, "n": n, "B": b,
            "fraction": b / n if n else None, "wilson95": wilson(b, n),
            "half_n": nh, "half_B": bh,
            "T3_included": min(nh) >= 3,
            "half_logits": empirical_logit(np.asarray(bh), np.asarray(nh)).tolist(),
        })
    n = np.asarray([r["n"] for r in people], dtype=np.int64)
    b = np.asarray([r["B"] for r in people], dtype=np.int64)
    included = [r for r in people if r["T3_included"]]
    nh = np.asarray([r["half_n"] for r in included], dtype=np.int64).reshape(-1, 2)
    bh = np.asarray([r["half_B"] for r in included], dtype=np.int64).reshape(-1, 2)
    t2 = t2_test(n, b, np.random.default_rng(
        np.random.SeedSequence([SEED, 41, seed_id])))
    t3 = t3_test(nh, bh, np.random.default_rng(
        np.random.SeedSequence([SEED, 44, seed_id])))
    return {
        "people": people, "N": int(n.sum()), "B": int(b.sum()),
        "pooled_fraction": float(b.sum() / n.sum()) if n.sum() else None,
        "T2": t2, "T3": t3, "T3_n_people": len(included),
        "T3_people": [r["participant"] for r in included],
        "denominator": "supported_tail" if tail else "supported_accepted",
        "H_threshold_ms": threshold, "absolute_amplitude_requirement": absolute,
    }


def matched_intercept(prevalence, sigma):
    if not 0 < prevalence < 1:
        return None
    if sigma == 0:
        return float(logit(prevalence))
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    offsets = math.sqrt(2) * sigma * nodes
    weights = weights / math.sqrt(math.pi)
    return float(brentq(
        lambda alpha: float(np.sum(weights * expit(alpha + offsets))) - prevalence,
        -50, 50))


def power_worker(task):
    name, which, sigma, n, observed_b = task
    D.require_slurm(1)
    n = np.asarray(n, dtype=np.int64)
    observed_b = np.asarray(observed_b, dtype=np.int64)
    mean_key, null_key = (42, 43) if which == "T2" else (45, 46)
    variant = int(sigma == 0)
    generate = np.random.default_rng(
        np.random.SeedSequence([SEED, mean_key, variant]))
    null_rng = np.random.default_rng(
        np.random.SeedSequence([SEED, null_key, variant]))
    if which == "T2":
        prevalence = [float(observed_b.sum() / n.sum())]
    else:
        prevalence = (observed_b.sum(axis=0) / n.sum(axis=0)).tolist()
    intercepts = [matched_intercept(p, sigma) for p in prevalence]
    if any(a is None for a in intercepts):
        return {"name": name, "estimate": None, "ci95": [None, None],
                "reason": "boundary_prevalence", "sigma": sigma}
    u = generate.normal(0, sigma, size=(2000, len(n)))
    if which == "T2":
        probabilities = expit(intercepts[0] + u)
        samples = generate.binomial(n, probabilities)
    else:
        probabilities = expit(np.asarray(intercepts)[None, None, :] + u[:, :, None])
        samples = generate.binomial(n[None, :, :], probabilities)
    rejected = 0
    informative = 0
    for b in samples:
        result = (
            t2_test(n, b, null_rng, 2000) if which == "T2"
            else t3_test(n, b, null_rng, 2000))
        informative += int(result["informative"])
        rejected += int(result["p"] is not None and result["p"] < .05)
    return {
        "name": name, "sigma": sigma,
        "simulations": 2000, "null_draws_per_simulation": 2000,
        "rejections": rejected, "informative_simulations": informative,
        "estimate": rejected / 2000, "ci95": wilson(rejected, 2000),
        "expected_prevalences": prevalence, "intercepts": intercepts,
        "margin_H_ms": 288,
        "interval_meaning": "Monte_Carlo_precision_not_human_uncertainty",
    }


def count_statement(test, power, null_check, *, eligible):
    calibration_ok = (
        null_check.get("ci95", [None, None])[1] is not None
        and null_check["ci95"][1] <= .075)
    power_ok = P.finite(power.get("estimate")) and power["estimate"] >= .80
    reasons = []
    if not eligible:
        reasons.append("insufficient_eligible_people")
    if not test.get("informative") or test.get("p") is None:
        reasons.append("uninformative_statistic")
    if not calibration_ok:
        reasons.append("null_calibration_not_passed")
    if reasons:
        status = "UNDERIDENTIFIED"
    elif test["p"] < .05:
        status = "SIGNIFICANT"
    elif power_ok:
        status = "NOT_OBSERVED"
    else:
        status = "UNDERIDENTIFIED"
        reasons.append("same_margin_power_below_0.80_or_unavailable")
    return {
        "status": status, "reasons": reasons,
        "test": test, "power": power, "null_check": null_check,
        "power_pass": bool(power_ok), "null_calibration_pass": bool(calibration_ok),
    }


def delta_h_halves(rows):
    halves = [
        P.person_feature(
            [r for r in rows if int(r["half"]) == h], list(D.PEOPLE), "H")
        for h in (1, 2)
    ]
    selected, x, y = [], [], []
    for p in D.PEOPLE:
        a, b = [h["people"][p] for h in halves]
        if a["included"] and b["included"] and P.finite(a.get("delta")) \
                and P.finite(b.get("delta")):
            selected.append(p)
            x.append(float(a["delta"]))
            y.append(float(b["delta"]))
    result = {
        "descriptive_only": True, "half_people": [h["people"] for h in halves],
        "point_identified_people": selected,
        "pearson": None, "spearman": None,
    }
    if len(selected) >= 3:
        r, good = correlation(x, y)
        rs, goods = correlation(rankdata(x), rankdata(y))
        result["pearson"] = float(r) if good else None
        result["spearman"] = float(rs) if goods else None
    return result


def survival_summary(rows, lower, upper):
    lo = np.asarray([num(r[lower]) for r in rows])
    hi = np.asarray([num(r[upper]) for r in rows])
    finite = np.r_[lo[np.isfinite(lo)], hi[np.isfinite(hi)]]
    finite = finite[finite >= 200]
    largest = max(288.0, float(np.max(finite)) if len(finite) else 288.0)
    grid = np.unique(np.r_[200.0, 288.0, finite, largest + 4])
    return {
        "x_ms": grid.tolist(), "n": len(rows),
        "lower": (1 - np.searchsorted(np.sort(lo), grid, side="right")
                  / len(rows)).tolist(),
        "upper": (1 - np.searchsorted(np.sort(hi), grid, side="right")
                  / len(rows)).tolist(),
    }


def depth_summary(real, synthetic):
    groups = {
        "Real tail": [r for r in real if r["set"] == "tail"
                      and r["status"] == "determinate"],
        **{kind: [r for r in synthetic if r["type"] == kind
                  and truth(r.get("usable"))] for kind in ("KPG", "KPB")},
    }
    result = {}
    for name, group in groups.items():
        result[name] = []
        for p in D.PEOPLE:
            selected = [r for r in group if r["participant"] == p]
            values = []
            for row in selected:
                v = row.get("split_80_depth")
                if not P.finite(v):
                    v = row.get("longest_run_depth")
                if P.finite(v):
                    values.append(float(v))
            result[name].append({
                "participant": p, "events": len(selected), "with_run": len(values),
                "median": float(np.median(values)) if values else None,
            })
    return result


def decision_stage(args, folder, data):
    root = args.output_dir.resolve()
    rows = D.read_csv(root / "features/event_features.csv")
    synthetic = D.read_csv(root / "known_positive/synthetic_features.csv")
    known = D.read_json(root / "known_positive/summary.json")
    people = list(D.PEOPLE)
    person = tail_people(rows)
    decision_people = [
        p for p in people if person[p]["n"] >= 3
        and all(known["people"][p][k]["covered"] for k in KINDS)]
    fractions = {
        c: P.bootstrap_people([person[p]["fractions"][c] for p in decision_people])
        for c in ("A", "B", "C")
    }
    any_fractions = {
        c: P.bootstrap_people([
            person[p]["any_fractions"][c] for p in decision_people])
        for c in ("A", "B", "C")
    }
    kp = known_population(known, decision_people)
    v5_p95 = p95_bounds(
        rows, people, "lag_lower_ms", "lag_upper_ms", bootstrap=True)
    all_p95 = p95_bounds(
        rows, people, "lambda_80_lower_ms", "lambda_80_upper_ms", bootstrap=True)
    set_p95 = p95_bounds(
        rows, decision_people, "lambda_80_lower_ms", "lambda_80_upper_ms",
        bootstrap=True)
    sensitivity = {}
    for g in GAPS:
        for token in (str(g), f"any_{g}"):
            sensitivity[token] = {
                "all_people": p95_bounds(
                    rows, people, f"lambda_{token}_lower_ms",
                    f"lambda_{token}_upper_ms"),
                "decision_people": p95_bounds(
                    rows, decision_people, f"lambda_{token}_lower_ms",
                    f"lambda_{token}_upper_ms"),
            }

    enough = len(decision_people) >= 12
    fa = fractions["A"]
    ready = enough and all(
        v is not None
        for v in [*fa["ci95"], *set_p95["ci95"]])
    support_pattern = bool(
        ready and fa["ci95"][0] > .5 and set_p95["ci95"][1] < 288)
    weaken_pattern = bool(
        ready and (fa["ci95"][1] < .5 or set_p95["ci95"][0] >= 288))
    prerequisites = {
        "decision_set_at_least_12": enough,
        "intervals_available": ready,
        "KPG_power": kp["KPG_power"],
        "KP_specificity": kp["KP_specificity"],
    }

    def statement(pattern, names):
        failed = [name for name in names if not prerequisites[name]]
        return {
            "pattern": pattern, "dependencies": names, "failed_dependencies": failed,
            "status": "UNDERIDENTIFIED" if failed
            else "SATISFIED" if pattern else "NOT_OBSERVED",
        }

    a_statements = {
        "support": statement(
            support_pattern, ["decision_set_at_least_12", "intervals_available",
                              "KPG_power", "KP_specificity"]),
        "weaken": statement(
            weaken_pattern, ["decision_set_at_least_12", "intervals_available",
                             "KPG_power"]),
    }
    if a_statements["support"]["status"] == "SATISFIED":
        a_status = "a_SUPPORTED"
    elif a_statements["weaken"]["status"] == "SATISFIED":
        a_status = "a_WEAKENED"
    elif any(r["status"] == "UNDERIDENTIFIED" for r in a_statements.values()):
        a_status = "a_UNDERIDENTIFIED"
    else:
        a_status = "a_UNRESOLVED"

    counts = count_analysis(rows)
    n = np.asarray([r["n"] for r in counts["people"]], dtype=np.int64)
    b = np.asarray([r["B"] for r in counts["people"]], dtype=np.int64)
    included = [r for r in counts["people"] if r["T3_included"]]
    nh = np.asarray([r["half_n"] for r in included], dtype=np.int64).reshape(-1, 2)
    bh = np.asarray([r["half_B"] for r in included], dtype=np.int64).reshape(-1, 2)
    tasks = [
        ("T2_power", "T2", .75, n, b),
        ("T2_null", "T2", 0.0, n, b),
    ]
    if len(included) >= 10:
        tasks.extend([
            ("T3_power", "T3", .75, nh, bh),
            ("T3_null", "T3", 0.0, nh, bh),
        ])
    simulated = D.pool_map(
        power_worker, tasks, min(args.workers, len(tasks)))
    powers = {r["name"]: r for r in simulated}
    for name in ("T3_power", "T3_null"):
        powers.setdefault(name, {
            "name": name, "estimate": None, "ci95": [None, None],
            "reason": "fewer_than_10_eligible_people"})
    t2 = count_statement(
        counts["T2"], powers["T2_power"], powers["T2_null"], eligible=True)
    t3 = count_statement(
        counts["T3"], powers["T3_power"], powers["T3_null"],
        eligible=len(included) >= 10)
    b_statements = [t2["status"], t3["status"]]
    if b_statements == ["SIGNIFICANT", "SIGNIFICANT"]:
        b_status = "b_SUPPORTED"
    elif b_statements == ["NOT_OBSERVED", "NOT_OBSERVED"]:
        b_status = "b_WEAKENED"
    elif "UNDERIDENTIFIED" in b_statements:
        b_status = "b_UNDERIDENTIFIED"
    else:
        b_status = "b_UNRESOLVED"

    descriptive = {
        "accepted_B": count_analysis(rows, tail=False, seed_id=1),
        "within_tail_B_abs": count_analysis(rows, absolute=True, seed_id=2),
        "within_tail_H250": count_analysis(rows, threshold=250, seed_id=3),
        "within_tail_H320": count_analysis(rows, threshold=320, seed_id=4),
        "delta_H_halves": delta_h_halves(rows),
        "leave_one_person_out_T2": {},
        "descriptive_only": True,
    }
    for j, p in enumerate(people):
        keep = np.arange(len(people)) != j
        descriptive["leave_one_person_out_T2"][p] = t2_test(
            n[keep], b[keep], np.random.default_rng(
                np.random.SeedSequence([SEED, 41, 100 + j])))

    result = {
        "component_a": {
            "status": a_status, "statements": a_statements,
            "decision_people": decision_people,
            "excluded_people": [p for p in people if p not in decision_people],
            "f": fractions, "any_f": any_fractions,
            "P95_star_decision": set_p95, "P95_star_all": all_p95,
            "v5_P95_all": v5_p95, "gap_sensitivities": sensitivity,
            "known_positive": kp,
            "scope": "covered_development_people_operational_gap_pattern",
        },
        "component_b": {
            "status": b_status, "T2": t2, "T3": t3,
            "scope": "all_development_people_conditional_count_model",
            "counts": counts,
        },
        "mixture_survives_declared_checks": (
            a_status == "a_SUPPORTED" and b_status == "b_SUPPORTED"),
        "descriptive": descriptive,
        "v5_D2_outcome": "UNDERIDENTIFIED",
        "v5_D1_outcome": "UNDERIDENTIFIED",
        "detector_frozen": False,
        "provider_access_has_not_occurred": True,
        "worst_case_certified": False,
        "scoring_clearance": "NOT_GRANTED",
        "limits": (
            "Operational sensor classes; no identified latent reach fractions. "
            "T2/T3 assume conditional event exchangeability, with unmodelled run "
            "dependence. Null qualification is limited to the declared plasmodes."
        ),
    }
    ratios = [
        V4.reference_theta(r, 3) / V4.reference_theta(r, 5)
        for r in data["refs"]]
    figure = {
        "people": people, "tail_people": person,
        "decision_people": decision_people, "known": known,
        "survival": {
            "v5": survival_summary(rows, "lag_lower_ms", "lag_upper_ms"),
            "lambda": survival_summary(
                rows, "lambda_80_lower_ms", "lambda_80_upper_ms"),
        },
        "depth": depth_summary(rows, synthetic),
        "theta3_theta5_band": [min(ratios), max(ratios)],
        "counts": counts,
        "depth_definition": (
            "per-person median: primary splitting-run depth, "
            "otherwise longest complete internal-run depth; no-run rows omitted"),
    }
    write_json(folder / "decision.json", result)
    write_json(folder / "figure_data.json", figure)
    write_json(folder / "plasmodes.json", powers)


def provider_stage(args, folder, data):
    root = args.output_dir.resolve()
    meta = check_stage(root / "decision", "ghd_decision", args)
    if meta.get("provider_access_has_not_occurred") is not True:
        raise RuntimeError("Missing pre-provider decision boundary")
    decision = D.read_json(root / "decision/decision.json")
    if decision.get("provider_access_has_not_occurred") is not True:
        raise RuntimeError("Decision does not certify the provider boundary")
    cues = {D.event_key(r): r for r in data["cues"]}
    providers, seen = {}, set()
    columns = (
        "participant", "session", "run", "cue",
        "cue_tick", "start_known_tick", "acc_start_tick")
    for row in V4.projected_csv(
        P.FREEZE / "self_paced_cues.csv", columns, allow_provider=True
    ):
        key = D.event_key(row)
        if row["participant"] not in D.PEOPLE or key in seen or key not in cues:
            raise RuntimeError("Provider inventory differs")
        seen.add(key)
        for field in ("cue_tick", "start_known_tick"):
            if V4.optional_int(row[field]) != V4.optional_int(cues[key][field]):
                raise RuntimeError("Provider cue/start join differs")
        providers[key] = V4.provider_tick(row["acc_start_tick"])
    if len(seen) != 4432 or seen != set(cues):
        raise RuntimeError("Provider inventory is incomplete")
    rows = D.read_csv(root / "features/event_features.csv")

    def describe(group):
        matched = [r for r in group if providers[D.event_key(r)] is not None]
        core = [r for r in matched
                if all(P.finite(r.get(f)) for f in
                       ("R_tick", "Rstar_80_tick", "L_tick"))]
        half = [r for r in core if P.finite(r.get("t50_tick"))]

        def differences(selected, field):
            return V4.distribution([
                num(r[field]) - providers[D.event_key(r)] for r in selected])

        return {
            "events": len(group), "provider_matched": len(matched),
            "provider_missing": len(group) - len(matched),
            "L_minus_provider_all_matched": differences(matched, "L_tick"),
            "core_matched": len(core),
            "same_core_rows": {
                f: differences(core, f)
                for f in ("R_tick", "Rstar_80_tick", "L_tick")
            },
            "t50_matched": len(half),
            "t50_missing_or_censored_in_core": len(core) - len(half),
            "same_t50_rows": {
                f: differences(half, f)
                for f in ("R_tick", "Rstar_80_tick", "t50_tick", "L_tick")
            },
        }

    def select(name, selected):
        if name in ("A", "B", "C"):
            return [r for r in selected if r.get("class") == name]
        if name == "reference":
            return [r for r in selected if r["set"] == "reference"]
        if name == "reference_supported":
            return [r for r in selected
                    if r["set"] == "reference" and r["feature_status"] == "ok"]
        if name == "unsupported_tail":
            return [r for r in selected
                    if r["set"] == "tail" and r["feature_status"] != "ok"]
        return [r for r in selected if r["set"] == "unresolved"]

    names = ("A", "B", "C", "reference", "reference_supported",
             "unsupported_tail", "unresolved")
    write_json(folder / "summary.json", {
        "descriptive_only": True, "provider_flag_filtering": False,
        "agreement_not_validation": True,
        "by_class_or_set": {
            name: describe(select(name, rows)) for name in names},
        "by_person": {
            p: {
                name: describe(select(
                    name, [r for r in rows if r["participant"] == p]))
                for name in names
            } for p in D.PEOPLE
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=[s[0] for s in STAGES])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participants")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require_compute(args.workers)
    scope(args)
    args.initial_script_hashes = code_hashes()
    if args.health_only and args.stage in ("decision", "provider"):
        print(f"{args.stage}: skipped in health mode", flush=True)
        return
    root = expected_root(args)
    index = next(i for i, item in enumerate(STAGES) if item[0] == args.stage)
    predecessors, previous = {}, {}
    if index == 0:
        if root.exists():
            raise FileExistsError("Initial run root must not already exist")
    else:
        previous, predecessors = verify_chain(
            root, STAGES[index - 1][0], args)
    # All fixed count/join checks precede creation of this stage's artifacts.
    data = input_contract()
    if previous:
        if previous["known_positive"]["fixed_inputs"] != data["fixed_inputs"]:
            raise RuntimeError("Fixed input closure differs from first stage")
    if index == 0:
        D.fresh(root)
    folder = D.fresh(root / STAGES[index][1])
    D.text_file(folder / "predeclaration.md", DECLARATION)
    try:
        {
            "known-positive": known_positive_stage,
            "features": features_stage,
            "decision": decision_stage,
            "provider": provider_stage,
        }[args.stage](args, folder, data)
        close_stage(folder, args, data["fixed_inputs"], predecessors)
    except Exception as error:
        if args.health_only:
            write_json(folder / "failure.json", {
                "status": "FAIL", "closed": False,
                "error_type": type(error).__name__,
                "message": str(error),
            })
        raise
    print(f"{args.stage}: closed", flush=True)


if __name__ == "__main__":
    main()
