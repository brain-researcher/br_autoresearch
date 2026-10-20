#!/usr/bin/env python3
"""EP19 staged raw-onset test v5; development data only."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector_v2 as D
from detector_v2 import R, RF, V4

DECLARATION = """\
### October 17: self-paced detector v2 and raw-onset test v5 — pre-declared before execution

This development-only follow-up is post hoc relative to job 48230171. New template, sham-position, shared-rung and operational choices are assumptions. Existing v1/v4 artifacts remain unchanged.

1. Use only the fifteen named development participants and supplied accelerometer/TRIG readers; never call derivative_onsets or use EEG features.
2. Before candidate output, reproduce D1/v1 non-provider cue fields and every label-array key/value, and reproduce every v4 event/k primary and earliest-sensitivity estimate. A mismatch stops subsequent stages.
3. Fit detector thresholds only from frozen fit-window X[130:250], with v1 quantile scale, rungs {1,1.5,2,3,4}, and conservative HIGH-confirmation window criterion.
4. Apply individual fit-hit ceilings max(1,floor(.05N)), pooled fixed-entry/risk floors .90 and individual risk floor .75. Validation reports only. Shared-rung assumption: advance the common pair when either candidate fails a guard; record every cause.
5. D1 is unchanged v1 replay. D2 suspends at LOW, records that crossing, requires uninterrupted LOW through HIGH and ten following samples, rejects on a drop below LOW, and preserves requalification, cuts, evidence snapshots and exclusive boundaries.
6. Generate and evaluate both candidates. Select the first eligible D2→D1 candidate receiving BELOW_EMPIRICAL_P95_ONLY; otherwise freeze nothing. A common control failure makes both UNDERIDENTIFIED.
7. Stage order is settings; references/backgrounds/templates; controls; labels; event decisions/selection; provider comparison. Each stage requires its predecessor's verified closure and writes fresh artifacts with exclusive creation.
8. Native r(t)=||a(t)-mean(a[t-520,t-20))|| in g, with half-open baseline endpoints and an unsmoothed current sample.
9. Reuse frozen still-window membership and both parities for each person's native reference, using the final 480 ms; require at least twenty windows and finite positive 1.4826*MAD.
10. Primary theta is median+5*1.4826*MAD; k=3 and 8 are locked sensitivities. These person-specific estimator references never initialize or tune the global detector.
11. Search B=max(S-500,L-3000) through L+100 inclusive on available native samples. Require complete validity and support strictly before the admitted evidence endpoint.
12. A complete 40-ms window qualifies when at least 80% of its samples are strictly above theta.
13. Separate bouts by at least 200 ms of nonqualifying window starts. First select the bout with R<=L whose qualifying-window envelope reaches L.
14. Otherwise select the earliest following bout with R>L and first qualifying-window start<=L+60. Its first window may start before L.
15. Require 200 ms of quiet window starts immediately before the selected bout's first qualifying-window start. R is that window's first exceedance; determinate lag is signed L-R.
16. Generic unresolved bounds are [-infinity,+infinity]. A selected bout lacking its left anchor retains [L-candidate_R,+infinity] under the v4 censoring interpretation. Absence of an associated bout is never a successful negative-lag observation.
17. Preserve the v4 earliest-bout, initial 40-ms all-sample-quiet sensitivity unchanged and descriptive. The explicit v4 mode calls the original estimator without the following-bout extension.
18. Backgrounds use the unchanged latest eligible nonoverlapping 2.8-s segment per end-tone+1-s to start-tone interval; require full frozen validity and screen only the final 1.9 s.
19. Select at most 32 backgrounds/person with seed 20261017 and the v4 reservoir stream; require twenty distinct backgrounds. Controls use S=1920 and B=1420 ms.
20. Gated families are linear and quadratic acceleration ramps with theta5 detectability delay 64 ms, plus the continuity-verified pulse_growth family. Descriptive families are 160-ms linear/quadratic ramps and marginal native departure.
21. For each ramp, use u=max((t-T0)/1000,0) and choose the midpoint of the amplitude interval placing the first strict theta5 exceedance at the declared delay. Verify persistent detectability after constructing the full template.
22. Pulse_growth uses a 120-ms raised-cosine pulse of amplitude 4*theta5 and quadratic growth beginning at T0+32 ms. Choose growth amplitude as 1.01 times the nonnegative maximum needed for signed native residual>=2*theta5 from T0+60 through the remaining uncapped template. Verify the pulse alone exceeds 3*theta5 and the complete bout reaches the largest label+40 ms.
23. Marginal departure rises to .99*theta5 immediately before T0+64, reaches 1.05*theta5 there, then increases to 1.10*theta5 over 600 ms. Construct acceleration causally by adding the exact past baseline to this prescribed residual.
24. Use T0=1920 at 250 Hz. At 1 kHz choose the smallest phase in {0,1,2,3} making T_det,5 lie on the detector availability phase. Use one template and one random-direction noisy trace per background/family for all targets.
25. Core targets are fast {0,40,80,120,160}, intermediate {200,240,272}, near {288,292,296,300,304,308,312,316}, and long {320,360,400,448,500,548,600} ms. Early {-40,-20} cells are descriptive.
26. Set synthetic L=T_det,5+target independently of the detector. Cap the shared template at T_det,5+600+100 ms; this common cap preserves every target's existing lookahead.
27. Recompute noiseless detectability after capping. Require exact target equality, detector-clock phase, L>=S, full search support, the 200-ms window-start anchor and the declared associated bout. Invalid cells remain uncovered.
28. Every core and early target is a multiple of 4 ms, so one availability phase serves all targets on a trace (250 Hz ticks 0 mod 4; 1-kHz labels 3 mod 4). Before any output, 448/548 ms replaced a draft's 450/550 ms, which lie off the 4-ms clock. Targets are never rounded or substituted after calibration; an infeasible cell stays uncovered and counts against coverage.
29. At each k, truth is synthetic L minus that k's noiseless detectable onset. Reuse theta5 templates at k=3/8; missing detectable onset means undefined truth. Other-k controls remain descriptive.
30. Consume one deterministic v4-style random direction per background/family, including families with uncovered cells. All targets reuse that trace; close references, background inventory and every calibration row before noisy controls.
31. Evaluate four shams/background at {2096,2296,2496,2696} ms, adding 3 ms at 1 kHz. These equally spaced positions cover early through late supported control labels; their dependence is retained.
32. A sham alarm is a determinate associated R, a selected bout lacking its 200-ms anchor, or invalid/insufficient search support. Valid searches without an associated bout are not sham alarms. Report the v4 any-departure indicator separately.
33. For every person/gated family require at least .95 usable core coverage and twenty distinct backgrounds in each required actual-truth band. Invalid templates/support reduce coverage; unresolved estimates with usable truth fail classification.
34. At k=5 require near [288,320) sensitivity>=.90, long [320,650] sensitivity>=.95, and fast [0,200) specificity>=.95 for linear/quadratic. Intermediate cells affect coverage only; pulse fast and all descriptive-family cells are not additional gates.
35. Require sham alarm fraction<=.05 for every person, using all four positions. All fifteen people must pass; no selected-person subset substitutes for this gate.
36. Use nearest-rank all-event p95 censoring bounds. Failed controls, ineligibility, absent event denominators, >.05 unresolved at any k, a straddling p95 interval, cross-k classification disagreement or participant-list execution imply UNDERIDENTIFIED.
37. Otherwise k=5 lower p95+12>=300 implies FAIL_EMPIRICAL_P95; upper p95+12<300 implies BELOW_EMPIRICAL_P95_ONLY. Keep the separate descriptive 188-ms flag.
38. Report signed U95(L-R)+12, determinate p99/max, per-person p95 bounds, negative lags, totals reaching 300 ms after adding 12, k=3/8 results, exposure, historical backward-LOW support and paired L_D1-L_D2.
39. Close both candidate decisions and selection before projecting acc_start_tick from the frozen v1 cue CSV. Verify cue_tick and start_known_tick by participant/session/run/cue before converting provider values; use no derivative files or provider flag filtering.
40. Health mode writes engineering counts/statuses only for real-event diagnostics, fixes the outcome to UNDERIDENTIFIED, omits real-event lag tables/quantiles/classifications, and skips provider access and plotting. Settings still use all fifteen people.
41. Require an actual Slurm compute shell, at most eight spawn workers within the allocation, and one BLAS/OpenMP thread per process. Refuse nonempty stage directories, changed closure inputs and all artifact replacement.
42. Control failure weakens estimator identification; identified failure weakens the empirical timing argument; BELOW weakens the large-p95-lag alternative only conditionally. People are independent human units; bounds are censoring bounds, not sampling confidence intervals. No outcome certifies worst-case timing or grants EEG scoring clearance.
"""

KS = (3, 5, 8)
CORE = (0, 40, 80, 120, 160, 200, 240, 272,
        288, 292, 296, 300, 304, 308, 312, 316,
        320, 360, 400, 448, 500, 548, 600)
TARGETS = CORE + (-40, -20)
FAMILIES = (
    "linear", "quadratic", "pulse_growth",
    "slow_linear", "slow_quadratic", "marginal",
)
GATED = ("linear", "quadratic", "pulse_growth")
SHAMS = (2096, 2296, 2496, 2696)


def estimate(raw_r, native_ok, fs, L, start_known, theta,
             *, evidence_stop=None, mode="v5"):
    """V4 remains an exact separate path; v5 extends only primary association."""
    if mode == "v4":
        return V4.estimate(raw_r, native_ok, fs, L, start_known, theta)
    if mode != "v5":
        raise ValueError("Unknown estimator mode")
    result = V4.estimate(raw_r, native_ok, fs, L, start_known, theta)
    dt = int(round(1000 / fs))
    last_used_tick = ((int(L) + 100) // dt) * dt
    if evidence_stop is not None and last_used_tick >= int(evidence_stop):
        for prefix in ("", "earliest_"):
            empty = V4.empty_estimate()
            empty.update(status="no_valid_data",
                         reason="incomplete_event_evidence_boundary")
            result.update({prefix + k: v for k, v in empty.items()})
        result["anchor_start_tick"] = result["anchor_end_tick"] = ""
    if result["status"] == "no_valid_data":
        result["lag_lower_ms"] = -float("inf")
        return result
    if result["status"] == "determinate" or result["candidate_tick"] != "":
        return result

    result["lag_lower_ms"] = -float("inf")
    begin = (result["search_start_tick"] + dt - 1) // dt
    end = result["search_end_tick"] // dt + 1
    width, quiet = 40 // dt, 200 // dt
    hot, q, starts, first_hot = V4.qualifying_windows(
        raw_r[begin:end], theta, width)
    bouts = V4.grouped_bouts(
        hot, starts, first_hot, width, quiet, window_quiet=True)
    eligible = [
        bout for bout in bouts
        if (begin + int(bout[0])) * dt > L
        and (begin + int(bout[2])) * dt <= L + 60
    ]
    if not eligible:
        return result
    bout = eligible[0]
    result["n_bouts"] += 1
    V4.record_candidate(result, bout, begin, dt, L)
    first_window = int(bout[2])
    anchor = first_window - quiet
    if anchor < 0 or np.any(q[anchor:first_window]):
        result["reason"] = "missing_200ms_quiet_anchor_left_censored"
        return result
    result["anchor_start_tick"] = int((begin + anchor) * dt)
    result["anchor_end_tick"] = int((begin + first_window) * dt)
    V4.resolve_candidate(result, "anchored_following_bout")
    if result["lag_ms"] >= 0:
        raise RuntimeError("Following-bout association must have negative lag")
    return result


def first_detectable(raw_r, fs, theta, T0):
    if not math.isfinite(theta):
        return None
    dt = int(round(1000 / fs))
    begin = (T0 + dt - 1) // dt
    _, _, starts, first_hot = V4.qualifying_windows(
        raw_r[begin:], theta, 40 // dt)
    return int((begin + first_hot[0]) * dt) if len(starts) else None


def scalar_residual(signal, fs):
    return V4.native_residual(np.asarray(signal)[:, None], fs)[:, 0]


def construct_signal(fs, theta, family, T0):
    dt = int(round(1000 / fs))
    ticks = np.arange(int(2800 / dt), dtype=np.int64) * dt
    u = np.maximum((ticks - T0) / 1000.0, 0)
    info = {"T0_tick": T0, "family": family}
    if family in ("linear", "quadratic", "slow_linear", "slow_quadratic"):
        tau = 160 if family.startswith("slow_") else 64
        power = 2 if "quadratic" in family else 1
        unit = u ** power
        residual = scalar_residual(unit, fs)
        j = (T0 + tau) // dt
        if residual[j] <= 0 or residual[j - 1] <= 0:
            raise RuntimeError("Invalid ramp amplitude interval")
        lower, upper = theta / residual[j], theta / residual[j - 1]
        if not 0 < lower < upper < float("inf"):
            raise RuntimeError("Empty ramp amplitude interval")
        amplitude = (lower + upper) / 2
        signal = amplitude * unit
        info.update(amplitude=amplitude, detectability_delay_ms=tau,
                    amplitude_units="g/s" if power == 1 else "g/s^2")
    elif family == "pulse_growth":
        pulse = np.zeros(len(ticks))
        inside = (u > 0) & (u < .120)
        pulse[inside] = np.sin(np.pi * u[inside] / .120) ** 2
        growth = np.maximum(u - .032, 0) ** 2
        pulse_r = scalar_residual(pulse, fs)
        growth_r = scalar_residual(growth, fs)
        use = ticks >= T0 + 60
        if not np.all(growth_r[use] > 0):
            raise RuntimeError("Pulse-growth sufficient condition is unsupported")
        amplitude = 1.01 * max(
            0.0, float(np.max(
                (2 * theta - 4 * theta * pulse_r[use]) / growth_r[use])))
        signal = 4 * theta * pulse + amplitude * growth
        if np.nanmax(np.abs(4 * theta * pulse_r)) < 3 * theta:
            raise RuntimeError("Pulse-alone detectability guarantee failed")
        if np.any(scalar_residual(signal, fs)[use] < 2 * theta - 1e-12):
            raise RuntimeError("Pulse-growth residual guarantee failed")
        info.update(
            pulse_amplitude_g=4 * theta,
            growth_amplitude_g_s2=amplitude,
            growth_start_tick=T0 + 32,
        )
    elif family == "marginal":
        tau = 64
        relative = ticks - T0
        desired = np.zeros(len(ticks))
        rise = (relative >= 0) & (relative < tau)
        desired[rise] = .99 * theta * relative[rise] / (tau - dt)
        after = relative >= tau
        desired[after] = theta * (
            1.05 + .05 * np.minimum((relative[after] - tau) / 600, 1))
        width, exclusion = int(.5 * fs), int(.02 * fs)
        signal = np.zeros(len(ticks))
        prefix = np.zeros(len(ticks) + 1)
        for j in range(len(ticks)):
            a, b = j - exclusion - width, j - exclusion
            baseline = (prefix[b] - prefix[a]) / width if a >= 0 else 0.0
            signal[j] = desired[j] + baseline
            prefix[j + 1] = prefix[j] + signal[j]
        info["detectability_delay_ms"] = tau
    else:
        raise ValueError(family)
    return ticks, signal, info


def calibrate_person(task):
    reference, folder = task
    D.require_slurm(1)
    p, fs = reference["participant"], float(reference["fs"])
    theta = V4.reference_theta(reference, 5)
    phase = 3 if fs == 1000 else 0
    dt = int(round(1000 / fs))
    arrays, rows, family_info = {}, [], []
    for family in FAMILIES:
        chosen = None
        for phi in ((0,) if fs == 250 else range(4)):
            T0 = 1920 + phi
            ticks, signal, info = construct_signal(fs, theta, family, T0)
            raw = np.abs(scalar_residual(signal, fs))
            detected = first_detectable(raw, fs, theta, T0)
            if detected is None or detected % 4 != phase:
                continue
            if ("detectability_delay_ms" in info
                    and detected - T0 != info["detectability_delay_ms"]):
                raise RuntimeError("Persistent ramp/marginal calibration delay changed")
            cap = detected + 700
            if cap > int(ticks[-1]) or cap % dt:
                raise RuntimeError("Shared template cap lacks native support")
            signal[ticks > cap] = signal[cap // dt]
            raw = np.abs(scalar_residual(signal, fs))
            checked = first_detectable(raw, fs, theta, T0)
            if checked != detected:
                raise RuntimeError("Growth cap changed detectable onset")
            chosen = ticks, signal, raw, detected, T0, cap, info
            break
        if chosen is None:
            raise RuntimeError(f"No availability phase for {p}/{family}")
        ticks, signal, raw, detected, T0, cap, info = chosen
        arrays[family] = signal
        info.update(participant=p, fs=fs, T_det_k5_tick=detected,
                    growth_stop_tick=cap, chosen_phase_ms=T0 - 1920)
        family_info.append(info)
        noiseless_ok = np.isfinite(raw)
        detected_by_k = {
            k: first_detectable(raw, fs, V4.reference_theta(reference, k), T0)
            for k in KS
        }

        # The gated pulse must retain one associated bout through largest L+40.
        if family == "pulse_growth":
            proof = estimate(raw, noiseless_ok, fs, detected + 600,
                             1920, theta)
            if (proof["status"] != "determinate"
                    or proof["R_tick"] != detected
                    or proof["bout_end_tick"] < detected + 640):
                raise RuntimeError("Declared pulse continuity proof failed")

        for target in TARGETS:
            L = detected + target
            reasons = []
            if L % 4 != phase:
                reasons.append("exact_target_off_detector_availability_clock")
            if L < 1920 or L + 100 > int(ticks[-1]):
                reasons.append("insufficient_declared_temporal_support")
            truths = {}
            for k in KS:
                td = detected_by_k[k]
                truth_ok = False
                est = None
                if td is not None and not reasons:
                    est = estimate(raw, noiseless_ok, fs, L, 1920,
                                   V4.reference_theta(reference, k))
                    truth_ok = (est["status"] == "determinate"
                                and est["R_tick"] == td)
                truths[f"T_det_k{k}_tick"] = "" if td is None else td
                truths[f"truth_ok_k{k}"] = bool(truth_ok)
                if k == 5 and not reasons and not truth_ok:
                    reasons.append("noiseless_association_or_anchor_failed")
            if L - detected != target:
                raise RuntimeError("Exact synthetic target arithmetic failed")
            rows.append({
                "participant": p, "fs": fs, "family": family,
                "gated_family": family in GATED, "core": target in CORE,
                "target_detectable_lag_ms": target,
                "T0_tick": T0, "S_tick": 1920, "search_start_tick": 1420,
                "L_tick": L, "growth_stop_tick": cap,
                "calibration_status": "uncovered" if reasons else "calibrated",
                "calibration_reason": ";".join(reasons) if reasons
                else "exact_phase_support_anchor_and_association_verified",
                **truths,
            })
    D.save_npz(Path(folder) / f"{p}_templates.npz", **arrays)
    return rows, family_info


def cache_backgrounds(task):
    p, stems, inventory, folder = task
    D.require_slurm(1)
    rng = np.random.default_rng(
        np.random.SeedSequence([V4.SEED, int(p.split("-")[1]), 1]))
    reservoir = []
    for count, item in enumerate(inventory, start=1):
        if len(reservoir) < 32:
            reservoir.append(item)
        else:
            replacement = int(rng.integers(count))
            if replacement < 32:
                reservoir[replacement] = item
    reservoir.sort(key=lambda r: (
        r["session"], int(r["run"]), int(r["start_tone_tick"])))
    by_run = defaultdict(list)
    for i, item in enumerate(reservoir):
        by_run[(item["session"], int(item["run"]))].append((i, item))
    segments = [None] * len(reservoir)
    fs_expected = 1000.0 if p == "sub-15" else 250.0
    for stem in stems:
        _, session, run_number = V4.stem_key(stem)
        needed = by_run.get((session, run_number), [])
        if not needed:
            continue
        run, acc, ticks, dev, valid, bursts, _ = RF.sp_load(stem)
        if run.fs != fs_expected:
            raise RuntimeError("Background native rate mismatch")
        factor = int(run.fs / 250)
        for i, item in needed:
            first = int(item["background_start_tick"])
            stop = int(item["background_stop_tick"])
            a, b = first // 4, stop // 4
            if first % 4 or stop - first != 2800:
                raise RuntimeError("Invalid closed background interval")
            if not np.all(valid[a:b]):
                raise RuntimeError("Closed background validity changed")
            section = run.acc_g[a * factor:b * factor].copy()
            if (len(section) != int(2.8 * run.fs)
                    or not np.all(np.isfinite(section))):
                raise RuntimeError("Invalid native background")
            segments[i] = section
    if any(segment is None for segment in segments):
        raise RuntimeError("Selected background lacks its raw run")
    length = int(2.8 * fs_expected)
    D.save_npz(
        Path(folder) / f"{p}_backgrounds.npz",
        acc=np.stack(segments) if segments else np.empty((0, length, 3)),
        fs=np.asarray(fs_expected),
    )
    return [{
        **item, "background_id": i,
        "participant": p,
    } for i, item in enumerate(reservoir)]


def control_person(task):
    reference, calibration, inventory, folder = task
    D.require_slurm(1)
    p, fs = reference["participant"], float(reference["fs"])
    rng = np.random.default_rng(
        np.random.SeedSequence([V4.SEED, int(p.split("-")[1]), 2]))
    rows = []
    by_family = {f: [r for r in calibration if r["family"] == f]
                 for f in FAMILIES}
    with np.load(Path(folder) / f"{p}_templates.npz",
                 allow_pickle=False) as template_file:
        templates = {f: template_file[f].copy() for f in FAMILIES}
    with np.load(Path(folder) / f"{p}_backgrounds.npz",
                 allow_pickle=False) as cache:
        if float(cache["fs"]) != fs or len(cache["acc"]) != len(inventory):
            raise RuntimeError("Background cache inventory mismatch")
        for background_id, background in enumerate(cache["acc"]):
            item = inventory[background_id]
            if int(item["background_id"]) != background_id:
                raise RuntimeError("Background order mismatch")
            provenance = {
                "participant": p, "fs": fs, "background_id": background_id,
                "background_session": item["session"],
                "background_run": int(item["run"]),
                "background_start_tick": int(item["background_start_tick"]),
                "background_end_tick": int(item["background_stop_tick"]),
            }
            raw, ok, _, _, _ = V4.control_features(background, fs)
            for sham_position in SHAMS:
                L = sham_position + (3 if fs == 1000 else 0)
                for k in KS:
                    est = estimate(raw, ok, fs, L, 1920,
                                   V4.reference_theta(reference, k))
                    alarm = (
                        est["status"] in ("determinate", "no_valid_data")
                        or est["reason"] == "missing_200ms_quiet_anchor_left_censored"
                    )
                    old_alarm = (
                        est["status"] == "no_valid_data"
                        or est["any_qualifying_before_L"] == 1
                        or est["initial_quiet_40ms"] != 1
                    )
                    rows.append({
                        **provenance, "kind": "sham", "family": "sham",
                        "core": False, "k": k, "L_tick": L, "S_tick": 1920,
                        "control_status": "sham",
                        **V4.estimate_control_fields(est),
                        "sham_alarm": int(alarm),
                        "v4_any_departure_sham_alarm": int(old_alarm),
                    })
            for family in FAMILIES:
                direction = rng.normal(size=3)
                norm = float(np.linalg.norm(direction))
                if norm == 0:
                    raise RuntimeError("Zero random direction")
                direction /= norm
                injected = background + templates[family][:, None] * direction
                raw, ok, _, _, _ = V4.control_features(injected, fs)
                for cell in by_family[family]:
                    for k in KS:
                        td = cell[f"T_det_k{k}_tick"]
                        L = int(cell["L_tick"])
                        if cell["calibration_status"] != "calibrated":
                            status = "uncovered_template"
                            est = V4.missing_control_estimate(
                                cell["calibration_reason"])
                        elif td == "":
                            status = "undefined_noiseless_truth"
                            est = V4.missing_control_estimate(status)
                        elif not cell[f"truth_ok_k{k}"]:
                            status = "truth_not_associated"
                            est = V4.missing_control_estimate(status)
                        else:
                            est = estimate(raw, ok, fs, L, 1920,
                                           V4.reference_theta(reference, k))
                            status = ("invalid_search_support"
                                      if est["status"] == "no_valid_data" else "ok")
                        rows.append({
                            **provenance, "kind": "injected", "family": family,
                            "core": cell["core"], "gated_family": family in GATED,
                            "target_detectable_lag_ms":
                                cell["target_detectable_lag_ms"],
                            "calibration_status": cell["calibration_status"],
                            "calibration_reason": cell["calibration_reason"],
                            "k": k, "L_tick": L, "S_tick": 1920,
                            "T0_tick": cell["T0_tick"], "T_det_tick": td,
                            "true_lag_ms": "" if td == "" else L - int(td),
                            "control_status": status,
                            "direction_x": float(direction[0]),
                            "direction_y": float(direction[1]),
                            "direction_z": float(direction[2]),
                            **V4.estimate_control_fields(est),
                        })
    return rows


def assess_controls(rows, participants):
    by_k = {}
    for k in KS:
        people = {}
        for p in participants:
            group = [r for r in rows if r["participant"] == p and r["k"] == k]
            shams = [r for r in group if r["kind"] == "sham"]
            backgrounds = {r["background_id"] for r in shams}
            if len(shams) != 4 * len(backgrounds):
                raise RuntimeError("Four shams per background are required")
            alarm = (sum(r["sham_alarm"] for r in shams) / len(shams)
                     if shams else None)
            passed = bool(len(backgrounds) >= 20 and alarm is not None and alarm <= .05)
            families = {}
            for family in FAMILIES:
                trials = [r for r in group if r["family"] == family and r["core"]]
                if len(trials) != len(backgrounds) * len(CORE):
                    raise RuntimeError("Missing core controls, including uncovered cells")
                usable = sum(r["control_status"] == "ok" for r in trials)
                coverage = usable / len(trials) if trials else 0.0
                near = V4.control_band(trials, 288, 320, True)
                long = V4.control_band(trials, 320, 650, True, inclusive_upper=True)
                fast = V4.control_band(trials, 0, 200, False)
                checks = [
                    coverage >= .95,
                    near["backgrounds"] >= 20,
                    near["fraction"] is not None and near["fraction"] >= .90,
                    long["backgrounds"] >= 20,
                    long["fraction"] is not None and long["fraction"] >= .95,
                ]
                if family in ("linear", "quadratic"):
                    checks.extend([
                        fast["backgrounds"] >= 20,
                        fast["fraction"] is not None and fast["fraction"] >= .95,
                    ])
                family_pass = bool(all(checks))
                families[family] = {
                    "gated": family in GATED, "pass": family_pass,
                    "trials": len(trials), "usable_crossing_fraction": coverage,
                    "control_status_counts": dict(Counter(
                        r["control_status"] for r in trials)),
                    "near": near, "long": long, "fast": fast,
                    "fast_required": family in ("linear", "quadratic"),
                }
                if family in GATED:
                    passed = passed and family_pass
            people[p] = {
                "pass": bool(passed), "backgrounds": len(backgrounds),
                "sham_n": len(shams), "sham_alarm_fraction": alarm,
                "v4_any_departure_sham_alarm_fraction":
                    sum(r["v4_any_departure_sham_alarm"] for r in shams) / len(shams)
                    if shams else None,
                "families": families,
            }
        by_k[str(k)] = {
            "pass": bool(people) and all(r["pass"] for r in people.values()),
            "participants": people,
        }
    primary = bool(set(participants) == set(D.PEOPLE) and by_k["5"]["pass"])
    return {
        "primary_k": 5, "primary_pass": primary,
        "result": "PASS" if primary else "FAIL", "by_k": by_k,
        "truth": "synthetic L minus person/k-specific noiseless T_det",
        "gated_families": GATED, "descriptive_families": FAMILIES[3:],
        "core_targets_ms": CORE, "early_targets_ms": (-40, -20),
        "all_targets_on_4ms_clock": all(t % 4 == 0 for t in TARGETS),
        "backgrounds_are_reused_not_independent_trials": True,
    }


def controls_stage(args):
    if args.detector_dir is None:
        raise ValueError("Controls require --detector-dir")
    detector = args.detector_dir.resolve()
    D.require_closed(detector / "settings", "settings")
    requested = D.scope(args)
    root = D.fresh(args.output_dir)
    calibration_dir = D.fresh(root / "calibration")
    D.text_file(calibration_dir / "predeclaration.md", DECLARATION)
    settings = D.read_json(detector / "settings" / "detector_settings.json")
    references = [r for r in settings["references"]
                  if r["participant"] in requested["participants"]]
    groups = D.stems_by_person()
    inventory = D.read_csv(detector / "settings" / "background_inventory.csv")
    selected = D.pool_map(
        cache_backgrounds,
        [(p, groups[p], [r for r in inventory if r["participant"] == p],
          str(calibration_dir)) for p in requested["participants"]],
        args.workers,
    )
    selected = [r for person in selected for r in person]
    calibration_results = D.pool_map(
        calibrate_person, [(r, str(calibration_dir)) for r in references],
        args.workers)
    cells = [row for result in calibration_results for row in result[0]]
    family_info = [row for result in calibration_results for row in result[1]]
    if len(cells) != len(references) * len(FAMILIES) * len(TARGETS):
        raise RuntimeError("Incomplete calibration table")
    D.write_json(calibration_dir / "noise_reference.json", references)
    D.write_csv(calibration_dir / "noise_reference.csv", references)
    D.write_json(calibration_dir / "selected_backgrounds.json", selected)
    D.write_csv(calibration_dir / "selected_backgrounds.csv", selected)
    D.write_json(calibration_dir / "control_templates.json", cells)
    D.write_csv(calibration_dir / "control_templates.csv", cells)
    D.write_json(calibration_dir / "template_parameters.json", family_info)
    settings_hash = D.digest(detector / "settings" / "closed.json")
    D.close(calibration_dir, "calibration", **requested,
            settings_sha256=settings_hash)
    D.require_closed(calibration_dir, "calibration")

    control_dir = D.fresh(root / "controls")
    D.text_file(control_dir / "predeclaration.md", DECLARATION)
    results = D.pool_map(
        control_person,
        [(r, [c for c in cells if c["participant"] == r["participant"]],
          [b for b in selected if b["participant"] == r["participant"]],
          str(calibration_dir)) for r in references],
        args.workers,
    )
    rows = [row for result in results for row in result]
    summary = assess_controls(rows, requested["participants"])
    D.write_csv(control_dir / "positive_control.csv", rows)
    D.write_json(control_dir / "positive_control_summary.json", summary)
    completed_references = []
    for reference in references:
        row = dict(reference)
        for k in KS:
            row[f"sham_alarm_fraction_k{k}"] = summary["by_k"][str(k)][
                "participants"][row["participant"]]["sham_alarm_fraction"]
        completed_references.append(row)
    D.write_csv(control_dir / "noise_reference.csv", completed_references)
    D.close(
        control_dir, "controls", **requested, settings_sha256=settings_hash,
        calibration_sha256=D.digest(calibration_dir / "closed.json"))
    print(
        f"Controls closed: {summary['result']}; "
        f"calibrated cells={sum(c['calibration_status'] == 'calibrated' for c in cells)}"
        f"/{len(cells)}", flush=True)


def event_person(task):
    p, stems, candidates, reference, mode = task
    D.require_slurm(1)
    by_run = defaultdict(list)
    for model, cues in candidates.items():
        for cue in cues:
            L = V4.optional_int(cue.get("onset_tick"))
            if L is not None:
                by_run[(cue["session"], int(cue["run"]))].append((model, cue))
    output = {model: [] for model in candidates}
    for stem in stems:
        _, session, run_number = V4.stem_key(stem)
        events = by_run.get((session, run_number), [])
        if not events:
            continue
        run, acc, ticks, dev, valid, bursts, _ = RF.sp_load(stem)
        raw = V4.native_statistic(run.acc_g, run.fs)
        ok = V4.native_validity(raw, valid, run.fs)
        phase = 3 if run.fs == 1000 else 0
        for model, cue in events:
            L = V4.required_int(cue["onset_tick"])
            S = V4.required_int(cue["start_known_tick"])
            stop = (None if mode == "v4"
                    else V4.required_int(cue["end_tick"]))
            if L % 4 != phase or L < S:
                raise RuntimeError("Candidate onset clock/range mismatch")
            for k in KS:
                theta = V4.reference_theta(reference, k)
                output[model].append({
                    "participant": p, "session": session, "run": run_number,
                    "cue": cue["cue"], "fs": run.fs, "k": k,
                    "L_tick": L, "S_tick": S, "theta_g": theta,
                    **estimate(raw, ok, run.fs, L, S, theta,
                               evidence_stop=stop, mode=mode),
                })
    return output


def metric(rows, prefix=""):
    lower = [float(r[prefix + "lag_lower_ms"]) for r in rows]
    upper = [float(r[prefix + "lag_upper_ms"]) for r in rows]
    determined = [r for r in rows if r[prefix + "status"] == "determinate"]
    lags = [float(r[prefix + "lag_ms"]) for r in determined]
    lo = V4.nearest_rank(lower) if rows else -float("inf")
    hi = V4.nearest_rank(upper) if rows else float("inf")
    classification = (
        "FAIL_EMPIRICAL_P95" if lo >= 288 else
        "BELOW_EMPIRICAL_P95_ONLY" if hi < 288 else "STRADDLES_MARGIN"
    )
    return {
        "n": len(rows), "determinate": len(determined),
        "indeterminate": sum(r[prefix + "status"] == "indeterminate" for r in rows),
        "no_valid_data": sum(r[prefix + "status"] == "no_valid_data" for r in rows),
        "unresolved_fraction": 1 - len(determined) / len(rows) if rows else 1.0,
        "multiple_bouts": sum(int(r[prefix + "n_bouts"]) > 1 for r in rows),
        "reason_counts": dict(Counter(r[prefix + "reason"] for r in rows)),
        "p95_determinate_ms": V4.nearest_rank(lags),
        "p99_determinate_ms": V4.nearest_rank(lags, .99),
        "max_determinate_ms": max(lags) if lags else None,
        "p95_lower_ms": lo, "p95_upper_ms": hi,
        "p95_lower_unbounded": not math.isfinite(lo),
        "p95_upper_unbounded": not math.isfinite(hi),
        "negative_lag_count": sum(x < 0 for x in lags),
        "following_bout_count": sum(
            r[prefix + "reason"] == "anchored_following_bout" for r in rows),
        "determinate_total_ge_300_count": sum(x + 12 >= 300 for x in lags),
        "known_total_ge_300_lower_count": sum(x + 12 >= 300 for x in lower),
        "possible_total_ge_300_upper_count": sum(x + 12 >= 300 for x in upper),
        "classification": classification,
    }


def make_decision(rows, controls, requested, eligible):
    participants = requested["participants"]
    by_k, per_person, earliest_k, earliest_people, disagreements = {}, {}, {}, {}, {}
    for k in KS:
        group = [r for r in rows if r["k"] == k]
        by_k[str(k)] = metric(group)
        per_person[str(k)] = {
            p: metric([r for r in group if r["participant"] == p])
            for p in participants}
        earliest_k[str(k)] = metric(group, "earliest_")
        earliest_people[str(k)] = {
            p: metric([r for r in group if r["participant"] == p], "earliest_")
            for p in participants}
        disagreements[str(k)] = V4.estimator_disagreements(group)
    reasons = []
    if requested["participant_list"]:
        reasons.append("participant_list_smoke_run")
    if not eligible:
        reasons.append("noise_admissibility_guard_failed")
    if not controls["primary_pass"]:
        reasons.append("all_participant_primary_controls_not_passed")
    if any(not per_person["5"][p]["n"] for p in participants):
        reasons.append("participant_without_accepted_events")
    if any(m["unresolved_fraction"] > .05 for m in by_k.values()):
        reasons.append("over_5_percent_unresolved_primary_events_at_some_k")
    classes = {m["classification"] for m in by_k.values()}
    if "STRADDLES_MARGIN" in classes:
        reasons.append("primary_p95_censoring_interval_straddles_margin")
    if len(classes) != 1:
        reasons.append("primary_threshold_sensitivity_disagreement")
    main = by_k["5"]
    outcome = "UNDERIDENTIFIED" if reasons else main["classification"]
    if main["p95_lower_ms"] + 12 > 188:
        proxy = "EMPIRICAL_LOWER_BOUND_EXCEEDS_188"
    elif main["p95_upper_ms"] + 12 <= 188:
        proxy = "EMPIRICALLY_COMPATIBLE_CONDITIONAL_ON_IDENTIFICATION"
    else:
        proxy = "UNRESOLVED"
    long_sets = [{
        D.event_key(r) for r in rows
        if r["k"] == k and r["status"] == "determinate" and r["lag_ms"] >= 288
    } for k in KS]
    stable = sorted(set.intersection(*long_sets))
    passing = [p for p in participants
               if controls["by_k"]["5"]["participants"][p]["pass"]]
    restricted = [r for r in rows if r["k"] == 5 and r["participant"] in passing]
    return {
        "outcome": outcome, "reasons": reasons, "primary_k": 5,
        "primary_estimator": "anchored_label_reaching_or_earliest_following_bout",
        "margin_lag_ms": 288, "fixed_terms_ms": 12,
        "quantile": "nearest-rank empirical p95",
        "by_k": by_k, "per_person": per_person,
        "empirical_self_paced_timing_bound_ms": main["p95_upper_ms"] + 12,
        "empirical_bound_unbounded": main["p95_upper_unbounded"],
        "earliest_sensitivity": {
            "used_for_decision": False,
            "quiet_definition": "unchanged strict all-sample quiet",
            "by_k": earliest_k, "per_person": earliest_people,
            "primary_vs_earliest_by_k": disagreements,
        },
        "control_passing_participant_subset": {
            "descriptive_only": True, "used_for_decision": False,
            "participants": passing, "metrics": metric(restricted) if restricted else None,
        },
        "188_ms_proxy": proxy,
        "cross_k_long_lag_event_count": len(stable),
        "cross_k_long_lag_event_keys": stable,
        "worst_case_bound_certified": False,
        "scoring_clearance": "NOT_GRANTED_BY_THIS_TEST",
        "uncertainty": "Censoring bounds, not sampling confidence intervals.",
    }


def engineering_counts(rows):
    return {
        str(k): {
            "n": len(group),
            "determinate": sum(r["status"] == "determinate" for r in group),
            "unresolved": sum(r["status"] != "determinate" for r in group),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "reason_counts": dict(Counter(r["reason"] for r in group)),
        }
        for k in KS
        for group in [[r for r in rows if r["k"] == k]]
    }


def provider_stage(freeze, root, cue_sets, raw_sets, requested):
    metadata = D.require_closed(root / "events", "events")
    if metadata.get("provider_access_has_not_occurred") is not True:
        raise RuntimeError("Missing pre-provider decision closure")
    folder = D.fresh(root / "provider")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    if requested["health_only"]:
        D.write_json(folder / "skipped.json", {"reason": "health_only"})
        D.close(folder, "provider", skipped=True)
        return
    if freeze is None:
        raise ValueError("Provider comparison requires --freeze-dir")
    candidate_keys = {}
    for model, cues in cue_sets.items():
        for cue in cues:
            candidate_keys.setdefault(D.event_key(cue), []).append(cue)
    columns = ("participant", "session", "run", "cue",
               "cue_tick", "start_known_tick", "acc_start_tick")
    providers, seen = {}, set()
    for row in V4.projected_csv(
        freeze / "self_paced_cues.csv", columns, allow_provider=True
    ):
        if row["participant"] not in D.PEOPLE:
            raise RuntimeError("Frozen provider table contains non-development person")
        key = D.event_key(row)
        if key in seen:
            raise RuntimeError("Duplicate frozen provider key")
        seen.add(key)
        if key not in candidate_keys:
            continue
        for cue in candidate_keys[key]:
            for field in ("cue_tick", "start_known_tick"):
                if V4.optional_int(cue.get(field)) != V4.optional_int(row[field]):
                    raise RuntimeError(f"Provider cue join mismatch: {key}, {field}")
        providers[key] = V4.provider_tick(row["acc_start_tick"])
    if len(seen) != 4432 or set(providers) != set(candidate_keys):
        raise RuntimeError("Frozen provider/candidate cue inventory differs")
    for model, rows in raw_sets.items():
        labels = {D.event_key(r): int(r["L_tick"]) for r in rows}
        differences = [L - providers[key] for key, L in labels.items()
                       if providers[key] is not None]
        by_k = {}
        for k in KS:
            matched = [r for r in rows if r["k"] == k
                       and r["status"] == "determinate"
                       and providers[D.event_key(r)] is not None]
            by_k[str(k)] = {
                "R_minus_AccStartIndex": V4.distribution([
                    int(r["R_tick"]) - providers[D.event_key(r)] for r in matched
                ], include_values=True),
                "L_minus_AccStartIndex_same_rows": V4.distribution([
                    int(r["L_tick"]) - providers[D.event_key(r)] for r in matched
                ], include_values=True),
            }
        D.write_json(folder / f"{model}.json", {
            "stage": "post_lock_agreement_only",
            "decision_summary_written_before_provider_access": True,
            "decision_closure_sha256": D.digest(root / "events" / "closed.json"),
            "provider_field_used": "acc_start_tick",
            "provider_flag_filtering": False,
            "accepted_events": len(labels),
            "provider_missing_events": sum(providers[key] is None for key in labels),
            "L_minus_AccStartIndex_all_available":
                V4.distribution(differences, include_values=True),
            "by_k": by_k,
        })
    D.close(folder, "provider", skipped=False)


def events_stage(args):
    if args.detector_dir is None:
        raise ValueError("Events require --detector-dir")
    root, detector = args.output_dir.resolve(), args.detector_dir.resolve()
    requested = D.scope(args)
    if not requested["health_only"] and args.freeze_dir is None:
        raise ValueError("Full events stage requires --freeze-dir")
    D.require_closed(detector / "settings", "settings")
    control_meta = D.require_closed(root / "controls", "controls")
    calibration_meta = D.require_closed(root / "calibration", "calibration")
    label_meta = D.require_closed(detector / "labels_stage", "labels")
    for meta in (control_meta, calibration_meta, label_meta):
        D.check_scope(meta, requested)
    settings_hash = D.digest(detector / "settings" / "closed.json")
    if any(meta["settings_sha256"] != settings_hash
           for meta in (control_meta, calibration_meta, label_meta)):
        raise RuntimeError("Settings closure mismatch")
    if label_meta["controls_sha256"] != D.digest(root / "controls" / "closed.json"):
        raise RuntimeError("Labels used a different control gate")
    if control_meta["calibration_sha256"] != D.digest(
            root / "calibration" / "closed.json"):
        raise RuntimeError("Control calibration closure mismatch")

    folder = D.fresh(root / "events")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    settings = D.read_json(detector / "settings" / "detector_settings.json")
    controls = D.read_json(root / "controls" / "positive_control_summary.json")
    references = D.read_json(root / "calibration" / "noise_reference.json")
    ref = {r["participant"]: r for r in references}
    cue_sets, label_summaries = {}, {}
    for model in D.MODELS:
        meta = D.require_closed(detector / model, f"labels_{model}")
        D.check_scope(meta, requested)
        if D.digest(detector / model / "closed.json") != \
                label_meta["candidate_closures"][model]:
            raise RuntimeError("Candidate label closure changed")
        cue_sets[model] = D.read_csv(detector / model / "self_paced_cues.csv")
        label_summaries[model] = D.read_json(
            detector / model / "label_summary.json")
    groups = D.stems_by_person()
    results = D.pool_map(
        event_person,
        [(p, groups[p], {
            model: [c for c in cue_sets[model] if c["participant"] == p]
            for model in D.MODELS
        }, ref[p], "v5") for p in requested["participants"]],
        args.workers,
    )
    raw_sets = {
        model: sorted([r for result in results for r in result[model]],
                      key=lambda r: (*D.event_key(r), r["k"]))
        for model in D.MODELS
    }
    for model in D.MODELS:
        expected = 3 * sum(
            V4.optional_int(c.get("onset_tick")) is not None
            for c in cue_sets[model])
        if len(raw_sets[model]) != expected:
            raise RuntimeError("Missing candidate event/k rows")

    summaries = {}
    for model in D.MODELS:
        if requested["health_only"]:
            decision = {
                "outcome": "UNDERIDENTIFIED",
                "reasons": ["health_only", "participant_list_smoke_run"]
                if requested["participant_list"] else ["health_only"],
                "engineering_counts": engineering_counts(raw_sets[model]),
                "scoring_clearance": "NOT_GRANTED_BY_THIS_TEST",
            }
        else:
            decision = make_decision(
                raw_sets[model], controls, requested,
                settings["candidates"][model]["eligible"])
        summaries[model] = {
            **requested,
            "scope": "smoke" if requested["participant_list"] else "full_development",
            "accepted_events": len(raw_sets[model]) // 3,
            "candidate": model, "revision_is_post_hoc": True,
            "positive_control_primary_pass": controls["primary_pass"],
            "decision": decision, "label_summary": label_summaries[model],
            "provider_access_has_not_occurred": True,
            "stage3_snapshot_not_rewritten_after_provider_access": True,
        }

    selected = next((
        model for model in D.MODELS
        if summaries[model]["decision"]["outcome"] == "BELOW_EMPIRICAL_P95_ONLY"
        and settings["candidates"][model]["eligible"]
    ), None)
    selection = {
        "order": D.MODELS, "tested": list(D.MODELS),
        "eligible": [m for m in D.MODELS if settings["candidates"][m]["eligible"]],
        "selected": selected, "frozen": selected is not None,
        "descriptive_candidates": [m for m in D.MODELS if m != selected],
        "outcomes": {m: summaries[m]["decision"]["outcome"] for m in D.MODELS},
        "reasons": {m: summaries[m]["decision"]["reasons"] for m in D.MODELS},
        "settings_closure_sha256": settings_hash,
        "candidate_label_closures": label_meta["candidate_closures"],
        "selected_labels": str(detector / selected / "labels") if selected else None,
    }
    D.write_json(folder / "selection.json", selection)

    for model in D.MODELS:
        candidate_dir = D.fresh(folder / model)
        summaries[model]["decision"]["selection"] = selection
        D.write_json(candidate_dir / "raw_onset_summary.json", summaries[model])
        if requested["health_only"]:
            fields = ("participant", "session", "run", "cue", "k",
                      "status", "reason", "earliest_status", "earliest_reason")
            D.write_csv(
                candidate_dir / "event_health.csv",
                [{f: r[f] for f in fields} for r in raw_sets[model]], fields)
        else:
            D.write_csv(candidate_dir / "raw_onsets.csv", raw_sets[model],
                        V4.RAW_FIELDS)

    if requested["health_only"]:
        health = {
            **requested, "outcome": "UNDERIDENTIFIED",
            "control_gates": controls,
            "candidate_counts": {
                model: engineering_counts(raw_sets[model]) for model in D.MODELS},
        }
        D.write_json(folder / "health.json", health)
    else:
        joins = {
            model: {D.event_key(c): c for c in cue_sets[model]}
            for model in D.MODELS
        }
        if set(joins["D1"]) != set(joins["D2"]):
            raise RuntimeError("Paired candidate cue keys differ")
        paired = []
        for key in sorted(joins["D1"]):
            a = V4.optional_int(joins["D1"][key].get("onset_tick"))
            b = V4.optional_int(joins["D2"][key].get("onset_tick"))
            if a is not None and b is not None:
                paired.append({
                    "participant": key[0], "session": key[1],
                    "run": key[2], "cue": key[3],
                    "L_D1_minus_L_D2_ms": a - b,
                })
        D.write_csv(folder / "paired_candidate_onsets.csv", paired,
                    ("participant", "session", "run", "cue", "L_D1_minus_L_D2_ms"))
        values = [r["L_D1_minus_L_D2_ms"] for r in paired]
        D.write_json(folder / "paired_candidate_summary.json", {
            "descriptive_only": True, **V4.distribution(values),
            "p99_ms": V4.nearest_rank(values, .99),
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None,
        })
    D.close(
        folder, "events", **requested,
        provider_access_has_not_occurred=True,
        controls_sha256=D.digest(root / "controls" / "closed.json"),
        labels_sha256=D.digest(detector / "labels_stage" / "closed.json"))
    if requested["health_only"]:
        print("Health events closed: UNDERIDENTIFIED; real-event lag output suppressed",
              flush=True)
    else:
        print(f"Decisions closed: {selection['outcomes']}; selected={selected}",
              flush=True)
    provider_stage(args.freeze_dir, root, cue_sets, raw_sets, requested)


def regression_person(task):
    p, stems, cues, cache_dir = task
    D.require_slurm(1)
    # Reuse v4's exact reference computation; its event pass is empty here.
    reference, unused, cache = V4.stage1_person((p, stems, [], cache_dir))
    if unused:
        raise RuntimeError("Unexpected event output during reference pass")
    result = event_person((p, stems, {"v1": cues}, reference, "v4"))
    return reference, result["v1"]


def regression_v4_stage(args):
    if args.freeze_dir is None or args.v4_dir is None or args.participants is not None:
        raise ValueError("v4 regression requires both reference directories and all people")
    folder = D.fresh(args.output_dir)
    D.text_file(folder / "predeclaration.md", DECLARATION)
    cache = D.fresh(folder / "reference_backgrounds")
    projected = list(V4.projected_csv(
        args.freeze_dir / "self_paced_cues.csv", V4.CUE_COLUMNS))
    if len(projected) != 4432 or sum(
            V4.optional_int(r["onset_tick"]) is not None for r in projected) != 4189:
        raise RuntimeError("Frozen v1 cue/onset contract mismatch")
    groups = D.stems_by_person()
    results = D.pool_map(
        regression_person,
        [(p, groups[p], [c for c in projected if c["participant"] == p],
          str(cache)) for p in D.PEOPLE],
        args.workers,
    )
    references = [result[0] for result in results]
    generated = [row for result in results for row in result[1]]
    for field in ("runs", "start_tones", "still_windows",
                  "fit_windows", "validation_windows"):
        if sum(int(r[field]) for r in references) != V4.EXPECTED_FULL[field]:
            raise RuntimeError(f"v4 reference count mismatch: {field}")
    saved_refs = {
        r["participant"]: r
        for r in D.read_csv(args.v4_dir / "noise_reference.csv")}
    for reference in references:
        for field in ("median_g", "mad_g", "sigma_g",
                      "theta_k3_g", "theta_k5_g", "theta_k8_g"):
            if float(reference[field]) != float(saved_refs[reference["participant"]][field]):
                raise RuntimeError(f"v4 reference differs: {reference['participant']}, {field}")
    saved = D.read_csv(args.v4_dir / "raw_onsets.csv")
    key = lambda row: (*D.event_key(row), int(row["k"]))
    old, new = {key(r): r for r in saved}, {key(r): r for r in generated}
    if (len(old) != len(saved) or len(new) != len(generated)
            or set(old) != set(new) or len(new) != 4189 * 3):
        raise RuntimeError("v4 event/k key contract mismatch")
    fields = V4.ESTIMATE_FIELDS + tuple(
        f"earliest_{f}" for f in V4.ESTIMATE_FIELDS)
    text_fields = {"status", "reason", "earliest_status", "earliest_reason"}
    for event in old:
        for field in fields:
            a, b = old[event][field], new[event][field]
            if field in text_fields:
                equal = a == b
            elif a == "" or b == "":
                equal = a == b
            else:
                equal = float(a) == float(b)
            if not equal:
                raise RuntimeError(f"v4 estimate regression failed: {event}, {field}")
    for k in KS:
        a = V4.metrics([r for r in saved if int(r["k"]) == k])
        b = V4.metrics([r for r in generated if r["k"] == k])
        if a != b:
            raise RuntimeError(f"v4 aggregate regression failed: k={k}")
    report = {
        "pass": True, "events": 4189, "event_k_rows": len(new),
        "fields_compared": fields, "exact_native_references": True,
        "aggregate_metrics_equal": True, "provider_values_used": False,
    }
    D.write_json(folder / "regression.json", report)
    D.close(folder, "regression_v4", **report)
    print("v4 estimator regression PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("controls", "events", "regression-v4"),
                        required=True)
    parser.add_argument("--detector-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--v4-dir", type=Path)
    parser.add_argument("--participants")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    D.require_slurm(args.workers)
    D.scope(args)
    if args.stage == "controls":
        controls_stage(args)
    elif args.stage == "events":
        events_stage(args)
    else:
        regression_v4_stage(args)


if __name__ == "__main__":
    main()
