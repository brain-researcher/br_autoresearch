#!/usr/bin/env python3
"""EP19 development onset/stillness detector freeze and forecast labels (October 16, 2026).

Rules: outputs/component_details.md section 1.2, logged before any detector output.
Data: WAY P1-P12 series 1-7 and AllLifts rows of series 1-7 (HandStart agreement only);
the 15 self-paced development participants and their derivatives (AccStartIndex
agreement only). Closed: WAY series 8-9, held-out people, all EEG. Calibration
evidence for labels/exposure/timing, not a scientific result.
"""
import argparse
import csv
import json
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import detector_freeze as F  # noqa: E402
import detector_readers as R  # noqa: E402
import detector_summaries as S  # noqa: E402
from classical_forecast_backend import GroupIdentity  # noqa: E402
from forecast_labels import AcceptedOnset, Followup, RiskDecision, label_for_decision  # noqa: E402
from way_completed_return import (FrozenReturnReferences, ReturnAdmissions, StartContext,  # noqa: E402
                                  evaluate_completed_return)
from way_risk_stream import WayRiskStream, label_from_snapshots  # noqa: E402

REASONS = {"decision_not_at_risk": 0, "observed_onset": 1, "complete_900ms_without_onset": 2,
           "administratively_censored": 3, "unknown_onset_status": 4, "observation_coverage_unknown": 5}
NOT_AT_RISK = -2
WAY_SUPPORT_MS = {"emg": 1000 * (S.EMG_SUMMARY_SAMPLES - 1) / R.EMG_RATE_HZ,   # 19.75 ms
                  "speed": 1000 * S.SPEED_LAG_TICKS / R.WAY_RATE_HZ}               # 20 ms
SP_SUPPORT_MS = {250.0: 16, 1000.0: 19}


def pct(values, q=(5, 25, 50, 75, 95, 99)):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return {"n": int(len(v)), **({f"p{k}": float(np.percentile(v, k)) for k in q} if len(v) else {}),
            **({"max": float(v.max()), "min": float(v.min()), "mean": float(v.mean())} if len(v) else {})}


def label_tuple(label):
    cat = label.observed_category
    return (NOT_AT_RISK if cat is None else int(cat),
            -1 if label.observed_through_bins is None else int(label.observed_through_bins),
            REASONS[label.reason])


# =============================================================== WAY
def way_noise_and_series(p):
    series, fit_w, val_w = {}, [], []
    ref_pos, ref_vel, ref_force = [], [], []
    still_counts = {}
    for s in R.WAY_SERIES:
        w = R.load_way_series(p, s)
        summ = S.way_summaries(w.emg, w.positions_cm, w.n)
        vel = S.object_velocity(w.positions_cm)
        pos, forces = w.positions_cm, w.forces_n
        digit = np.minimum(np.linalg.norm(pos[:, 1] - pos[:, 0], axis=1), np.linalg.norm(pos[:, 2] - pos[:, 0], axis=1))
        kept = 0
        for k, r in enumerate(w.rises):
            r = int(r)
            a = r - F.WAY_NOISE_TICKS
            if a < 0:
                continue
            seg = slice(a, r)
            if not (np.all(np.isfinite(summ[seg])) and np.all(np.isfinite(vel[seg]))):
                continue
            hand = pos[seg, 1:4]
            spread = np.max(np.linalg.norm(hand - np.median(hand, axis=0), axis=2))
            if spread > F.WAY_STILL_SPREAD_CM or np.min(digit[seg]) < F.WAY_CLEARANCE_CM:
                continue
            kept += 1
            if k % 2 == 0:
                fit_w.append(summ[seg])
                ref_pos.append(pos[seg, 0])
                ref_vel.append(vel[seg])
                ref_force.append(forces[seg])
            else:
                val_w.append(summ[seg])
        still_counts[s] = {"rises": int(len(w.rises)), "still_windows": kept}
        series[s] = {"n": w.n, "summ": summ, "vel": vel, "pos": pos, "forces": forces, "rises": w.rises,
                     "falls": w.falls, "conditions": R.lift_conditions(w), "emg_names": w.emg_names}
        del w
    return series, fit_w, val_w, np.concatenate(ref_pos), np.concatenate(ref_vel), np.concatenate(ref_force), still_counts


def way_references(pos, vel, force, low):
    def tol(dev, floor):
        return np.maximum(F.REF_TOLERANCE_MULTIPLIER * np.percentile(np.abs(dev), 99, axis=0), floor)
    p_ref = np.median(pos, axis=0)
    f_ref = np.median(force, axis=0)
    out = {"object_position_cm": p_ref, "object_position_tolerance_cm": tol(pos - p_ref, F.REF_FLOORS["position_cm"]),
           "object_motion_tolerance_cm_s": tol(vel, F.REF_FLOORS["motion_cm_s"]),
           "plate_force_n": f_ref, "plate_force_tolerance_n": tol(force - f_ref, F.REF_FLOORS["force_n"])}
    frozen = FrozenReturnReferences(
        object_position_m=out["object_position_cm"] / 100.0,
        object_position_tolerance_m=out["object_position_tolerance_cm"] / 100.0,
        object_motion=np.zeros(3), object_motion_tolerance=out["object_motion_tolerance_cm_s"] / 100.0,
        object_motion_unit="m/s", plate_force_n=out["plate_force_n"],
        plate_force_tolerance_n=out["plate_force_tolerance_n"], primary_low=np.asarray(low))
    return out, frozen


ADMISSIONS = ReturnAdmissions(position_unit="m", force_unit="N", object_motion_unit="m/s",
                              physical_units_admitted=True, backward_motion_admitted=True,
                              primary_summaries_causal=True, calibration_admitted=True)


def way_participant(p):
    t0 = time.time()
    series, fit_w, val_w, rpos, rvel, rforce, still_counts = way_noise_and_series(p)
    fits = F.fit_thresholds(fit_w, F.WAY_CONFIRM)
    resolved = all(f.resolved for f in fits)
    low = np.array([f.low for f in fits])
    high = np.array([f.high for f in fits])
    validation = F.validate_thresholds(val_w, low, high, F.WAY_CONFIRM, F.WAY_QUIET, R.WAY_RATE_HZ) if resolved else {}
    refs, frozen = way_references(rpos, rvel, rforce, low)
    handstart = {(r["series"], r["lift"]): r for r in R.way_handstart_rows(p)}
    cue_rows, grids, equivalence = [], {}, {"series": 0, "grid_mismatch": 0, "onset_mismatch": 0, "label_mismatch": 0,
                                           "examples": []}
    departures = []
    for s, d in series.items():
        n, summ = d["n"], d["summ"]
        valid = (np.all(np.isfinite(summ), axis=1) & np.all(np.isfinite(d["vel"]), axis=1)
                 & np.all(np.isfinite(d["pos"]), axis=(1, 2)) & np.all(np.isfinite(d["forces"]), axis=(1, 2)))
        grid_ticks = np.arange(0, n, F.WAY_GRID)
        g_state = np.full(len(grid_ticks), F.STATE_CODES["unknown"], dtype=np.int8)
        g_cue = np.full(len(grid_ticks), -1, dtype=np.int16)
        g_cat = np.full(len(grid_ticks), NOT_AT_RISK, dtype=np.int8)
        g_obs = np.full(len(grid_ticks), -1, dtype=np.int8)
        g_reason = np.zeros(len(grid_ticks), dtype=np.int8)
        if not resolved:
            grids[s] = (grid_ticks, g_state, g_cue, g_cat, g_obs, g_reason)
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
        segments = list(zip(edges[::2], edges[1::2]))
        rises, falls = set(int(x) for x in d["rises"]), set(int(x) for x in d["falls"])
        rise_index = {int(x): k for k, x in enumerate(d["rises"])}
        cue_status = {}
        prior_onset = {}
        for seg_id, (a, b) in enumerate(segments):
            identity = GroupIdentity("WAY", p, s, f"P{p}-S{s}-segment-{seg_id + 1}")
            stream = WayRiskStream(identity, low, high, grid_origin_tick=0)
            open_decisions, way_labels, way_states, way_onsets = [], {}, {}, {}
            previous_rise = previous_fall = None
            previous_reach = None   # previous cue's accepted onset, else its LED rise (reach undetected)
            current_cue = None
            for t in range(a, b):
                start, end = t in rises, t in falls
                support = None
                if start:
                    k = rise_index[t]
                    first = previous_rise is None and k == 0
                    phase = "initial_series" if first else ("after_reach" if previous_rise is not None else None)
                    context = StartContext(start_tick=t, start_authenticated=True, phase=phase,
                                           phase_history_known=phase is not None, series_start_tick=0,
                                           continuous_history_start_tick=int(a),
                                           prior_reach_tick=None if first else previous_reach,
                                           end_tick=None if first else previous_fall,
                                           end_authenticated=previous_fall is not None)
                    support = evaluate_completed_return(
                        evaluation_tick=t, first_tick=0, positions_m=d["pos"] / 100.0, object_motion=d["vel"] / 100.0,
                        plate_forces_n=d["forces"], primary_summaries=summ, primary_history_valid=np.isfinite(summ),
                        object_motion_history_valid=np.all(np.isfinite(d["vel"]), axis=1), sample_valid=valid,
                        admissions=ADMISSIONS, references=frozen, context=context)
                    cue_status[k] = support
                    current_cue = k
                    previous_rise = t
                snap = stream.step(t, summ[t], start=start, end=end, return_support=support, history_valid=True)
                if snap.closed:
                    for dsnap in open_decisions:
                        way_labels[dsnap.decision.decision_tick] = label_tuple(label_from_snapshots(dsnap, snap))
                    if snap.cue is not None:
                        if snap.onset is not None:
                            way_onsets[current_cue] = (snap.onset.crossing_tick, snap.onset.known_tick)
                        previous_reach = snap.onset.crossing_tick if snap.onset is not None else snap.cue.start_tick
                    open_decisions = []
                    previous_fall = t
                if snap.decision is not None:
                    way_states[t] = F.STATE_CODES[snap.decision.state]
                    if snap.decision.state == "at_risk":
                        open_decisions.append(snap)
            snap = stream.close(b, reason="administrative")
            for dsnap in open_decisions:
                way_labels[dsnap.decision.decision_tick] = label_tuple(label_from_snapshots(dsnap, snap))
            if snap.onset is not None and current_cue is not None:
                way_onsets[current_cue] = (snap.onset.crossing_tick, snap.onset.known_tick)
            # Generic replay of the same segment (used for diagnostics and the self-paced stream).
            starts = {t - a: cue_status[rise_index[t]].status for t in rises if a <= t < b}
            cuts = {t - a for t in falls if a <= t < b}
            state, cue_of, cues = F.run_segment(summ[a:b], low, high, F.WAY_CONFIRM, F.WAY_QUIET, starts, cuts)
            equivalence["series"] += 1
            for gi, g in enumerate(grid_ticks):
                if not a <= g < b:
                    continue
                generic = int(state[g - a])
                ci = int(cue_of[g - a])
                g_state[gi] = generic
                if g in way_states:
                    if way_states[g] != generic:
                        equivalence["grid_mismatch"] += 1
                        if len(equivalence["examples"]) < 5:
                            equivalence["examples"].append(["state", p, s, int(g), way_states[g], generic])
                elif generic != F.STATE_CODES["unarmed"]:
                    equivalence["grid_mismatch"] += 1
                if ci >= 0:
                    g_cue[gi] = rise_index[cues[ci].start + a]
                if g in way_labels:
                    g_cat[gi], g_obs[gi], g_reason[gi] = way_labels[g]
            for c in cues:
                k = rise_index[c.start + a]
                generic_onset = None if c.onset is None else (c.onset[0] + a, c.onset[1] + a)
                if generic_onset != way_onsets.get(k):
                    equivalence["onset_mismatch"] += 1
                    if len(equivalence["examples"]) < 5:
                        equivalence["examples"].append(["onset", p, s, k, way_onsets.get(k), generic_onset])
                # generic labels for the same at-risk decisions
                identity = GroupIdentity("WAY", p, s, f"generic-{k}")
                stop = c.stop + a
                unknown = None if c.unknown_from is None else c.unknown_from + a
                onset = None if generic_onset is None else AcceptedOnset(identity, *generic_onset)
                for g in range(((c.start + a + F.WAY_GRID - 1) // F.WAY_GRID) * F.WAY_GRID, stop, F.WAY_GRID):
                    if state[g - a] != F.STATE_CODES["at_risk"]:
                        continue
                    lab = label_for_decision(RiskDecision(identity, g, "at_risk"), Followup(identity, stop, unknown),
                                             onset, ticks_per_second=R.WAY_RATE_HZ, grid_origin_tick=0)
                    if label_tuple(lab) != way_labels.get(g):
                        equivalence["label_mismatch"] += 1
                        if len(equivalence["examples"]) < 5:
                            equivalence["examples"].append(["label", p, s, int(g), way_labels.get(g), label_tuple(lab)])
                cond = d["conditions"][k]
                row = {"source": "WAY", "participant": f"P{p}", "series": s, "cue": k, "lift": k + 1,
                       "cue_tick": c.start + a, "end_tick": stop, "stop_reason": "led_off" if c.stop_reason == "cut" else c.stop_reason,
                       "weight_g": cond["weight_g"], "surface_code": cond["surface_code"],
                       "support": c.status, "support_reasons": ";".join(cue_status[k].reasons),
                       "onset_tick": None, "known_tick": None, "onset_path": None, "rt_ms": None,
                       "confirmation_ms": None, "departure_ms": None, "departure_path": None,
                       "rejected_crossings": len(c.rejected), "unresolved_at_end": bool(c.unknown_from is not None and c.status == "supported"),
                       "handstart_tick": None, "onset_minus_handstart_ms": None, "handstart_cue_check_ticks": None}
                hs = handstart.get((s, k + 1))
                if hs is not None:
                    row["handstart_cue_check_ticks"] = int(round(500 * hs["start_time"])) - (c.start + a)
                    if np.isfinite(hs["handstart_hs_s"]):
                        row["handstart_tick"] = float(500 * hs["handstart_hs_s"])
                if generic_onset is not None:
                    cr, kn = generic_onset
                    dep, dep_path = F.retrospective_departure(summ[a:b], low, cr - a)
                    row.update(onset_tick=cr, known_tick=kn, onset_path=S.WAY_PATHS[c.onset_path],
                               rt_ms=2 * (cr - (c.start + a)), confirmation_ms=2 * (kn - cr),
                               departure_ms=2 * (cr - (dep + a)) + WAY_SUPPORT_MS["emg" if dep_path < 5 else "speed"],
                               departure_path=S.WAY_PATHS[dep_path] if dep_path is not None else None)
                    departures.append(row["departure_ms"])
                    if row["handstart_tick"] is not None:
                        row["onset_minus_handstart_ms"] = 2 * (cr - row["handstart_tick"])
                cue_rows.append(row)
        # cues whose rise fell outside every valid segment
        seen = {r["cue"] for r in cue_rows if r["series"] == s}
        for k, r in enumerate(d["rises"]):
            if k not in seen:
                cue_rows.append({"source": "WAY", "participant": f"P{p}", "series": s, "cue": k, "lift": k + 1,
                                 "cue_tick": int(r), "support": "unknown", "support_reasons": "outside_valid_segment"})
        grids[s] = (grid_ticks, g_state, g_cue, g_cat, g_obs, g_reason)
    settings = {"participant": f"P{p}", "paths": list(S.WAY_PATHS), "emg_channel_names": series[1]["emg_names"],
                "resolved": resolved, "fits": [f.__dict__ for f in fits], "low": low.tolist(), "high": high.tolist(),
                "noise_windows": {"fit": len(fit_w), "validation": len(val_w), "by_series": still_counts},
                "validation": validation,
                "return_references": {k: np.asarray(v).tolist() for k, v in refs.items()}}
    return {"participant": f"P{p}", "settings": settings, "cues": cue_rows, "grids": grids,
            "equivalence": equivalence, "seconds": time.time() - t0}


# ========================================================== self-paced
def sp_load(stem):
    run = R.load_self_paced_run(stem)
    acc, ticks = R.reduce_to_detector_rate(run.acc_g, run.fs)
    dev = S.self_paced_departure(acc)
    sd = R.trailing_sd(acc, R.SP_DETECTOR_HZ)
    with np.errstate(invalid="ignore"):
        axis_ok = np.all(sd >= F.SP_FLAT_SD_G, axis=1)
    valid = np.isfinite(dev) & axis_ok
    bursts, tone_info = F.detect_tones(run.trig_uv, run.fs)
    tps = run.ticks_per_sample
    for b in bursts:
        b["onset_tick"], b["known_tick"] = b["onset"] * tps, b["known"] * tps
    return run, acc, ticks, dev, valid, bursts, tone_info


def sp_noise(stem):
    run, acc, ticks, dev, valid, bursts, _ = sp_load(stem)
    fit, val, kept, total = [], [], 0, 0
    starts = [b for b in bursts if b.get("tone") and b["class"] == "start"]
    for k, b in enumerate(starts):
        on = int(np.searchsorted(ticks, b["onset_tick"]))
        a = on - F.SP_NOISE_SAMPLES
        total += 1
        if a < 0 or not np.all(valid[a:on]):
            continue
        blocks = acc[a:on].reshape(-1, F.SP_BLOCK_SAMPLES, 3).mean(axis=1)
        if np.max(np.linalg.norm(blocks - acc[a:on].mean(axis=0), axis=1)) > F.SP_STILL_BLOCK_G:
            continue
        kept += 1
        (fit if k % 2 == 0 else val).append(dev[a:on, None])
    return {"stem": run.stem, "participant": run.participant, "fit": fit, "val": val, "start_tones": total,
            "still_windows": kept, "fs": run.fs}


def sp_participant(args):
    participant, low, high = args
    t0 = time.time()
    stems = [s for s in R.development_stems() if os.path.basename(s).startswith(participant + "_")]
    by_session = {}
    for stem in stems:
        by_session.setdefault(os.path.basename(stem).split("_")[1], []).append(stem)
    cue_rows, grids, departures, tone_rows = [], {}, [], []
    deriv_info = {}
    for session, session_stems in sorted(by_session.items()):
        loaded = {}
        for stem in session_stems:
            loaded[stem] = sp_load(stem)
        frames = {loaded[st][0].run: loaded[st][0].frames for st in session_stems}
        fs_raw = {loaded[st][0].run: loaded[st][0].fs for st in session_stems}
        try:
            deriv = R.derivative_onsets(participant, session, frames)
        except KeyError as error:
            deriv = {"fs": None, "rows": [], "binding": {}, "error": str(error)}
        deriv_info[session] = {"fs": deriv["fs"], "binding": {str(k): v for k, v in deriv["binding"].items()},
                               "rows": len(deriv["rows"])}
        # Bind by frame count allowing a resampled derivative.
        if deriv["fs"] and deriv["rows"]:
            ratio = {run: fs_raw[run] / deriv["fs"] for run in frames}
            binding = {}
            for k, count in enumerate(deriv["derivative_frames"], start=1):
                hits = [run for run, raw in frames.items() if abs(raw - count * ratio[run]) <= ratio[run]]
                binding[k] = hits[0] if len(hits) == 1 else None
            use = {run: sum(v == run for v in binding.values()) for run in frames}
            binding = {k: v if v is not None and use[v] == 1 else None for k, v in binding.items()}
            deriv_info[session]["binding"] = {str(k): v for k, v in binding.items()}
        else:
            binding = {}
        for stem in session_stems:
            run, acc, ticks, dev, valid, bursts, tone_info = loaded[stem]
            rows_here = [r for r in deriv["rows"] if binding.get(r["derivative_run"]) == run.run]
            tick_per_deriv = None if not deriv["fs"] else 1000.0 / deriv["fs"]
            m = len(dev)
            values = dev[:, None]
            tones = [b for b in bursts if b.get("tone")]
            # spurious-burst test case and tone bookkeeping
            for b in bursts:
                tone_rows.append({"participant": participant, "session": session, "run": run.run,
                                  "onset_s": b["onset"] / run.fs, "duration_s": b["duration_s"],
                                  "class": b["class"], "known_s": b["known"] / run.fs})
            starts, cuts, evidence = {}, set(), {}
            start_meta = {}
            last_classified = None
            for b in tones:
                kn = int(np.searchsorted(ticks, b["known_tick"]))
                on = int(np.searchsorted(ticks, b["onset_tick"]))
                if kn >= m:
                    continue
                cuts.add(kn)
                evidence[kn] = on
                if b["class"] == "start":
                    S_tick = b["known_tick"]
                    lo = int(np.searchsorted(ticks, S_tick - 500))
                    reasons = []
                    if last_classified is None or last_classified["class"] != "end":
                        status = "unknown"
                        reasons.append("previous_end_not_observed")
                    elif last_classified["onset_tick"] >= S_tick - 500:
                        status = "unknown"
                        reasons.append("end_inside_return_window")
                    elif lo <= 0 or not np.all(valid[lo:kn]):
                        status = "unknown"
                        reasons.append("sensor_history_incomplete")
                    elif np.any(dev[lo:kn] > low[0]):
                        status = "not_supported"
                        reasons.append("departure_exceeds_low_threshold")
                    else:
                        status = "supported"
                    starts[kn] = status
                    start_meta[kn] = (b, status, reasons)
                if b["class"] != "unclassified":
                    last_classified = b
            # segments of valid detector samples
            edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
            segments = list(zip(edges[::2], edges[1::2]))
            state_all = np.full(m, F.STATE_CODES["unknown"], dtype=np.int8)
            cue_all = np.full(m, -1, dtype=np.int32)
            all_cues = []
            endpoint_cuts = {}
            for a, bnd in segments:
                st = {k - a: v for k, v in starts.items() if a <= k < bnd}
                ct = {k - a for k in cuts if a <= k < bnd}
                ev = {k - a: max(v, a) - a for k, v in evidence.items() if a <= k < bnd}
                boundary_tick = int(ticks[bnd]) if bnd < m else int(ticks[bnd - 1]) + 1
                # A burst still unresolved at the segment end (or a tone whose cut falls at or
                # after it) limits the open cue's evidence to its onset.
                tail = [x for x in bursts if ticks[a] <= x["onset_tick"] < boundary_tick
                        and (x["known_tick"] >= boundary_tick
                             or (x.get("tone") and np.searchsorted(ticks, x["known_tick"]) >= bnd))]
                if tail:
                    x = min(tail, key=lambda y: y["onset_tick"])
                    ev[bnd - a] = int(np.searchsorted(ticks, x["onset_tick"])) - a
                    endpoint_cuts[bnd] = (int(x["onset_tick"]),
                                          "tone_" + x["class"] if x["known_tick"] < boundary_tick else "unresolved_burst")
                state, cue_of, cues = F.run_segment(values[a:bnd], low, high, F.SP_CONFIRM, F.SP_QUIET, st, ct, ev)
                state_all[a:bnd] = state
                base = len(all_cues)
                cue_all[a:bnd] = np.where(cue_of >= 0, cue_of + base, -1)
                for c in cues:
                    all_cues.append((a, bnd, c))
            last_tick = int(ticks[-1]) if m else 0
            grid_ticks = np.arange(0, last_tick + 1, F.SP_GRID_TICKS)
            idx = np.searchsorted(ticks, grid_ticks, side="right") - 1
            g_state = np.where(idx >= 0, state_all[np.maximum(idx, 0)], F.STATE_CODES["unknown"]).astype(np.int8)
            g_cue = np.where(idx >= 0, cue_all[np.maximum(idx, 0)], -1).astype(np.int32)
            # tick-based suspension while any TRIG burst is unresolved inside an open cue
            for b in bursts:
                inside = (grid_ticks >= b["onset_tick"]) & (grid_ticks < b["known_tick"]) & (g_cue >= 0)
                g_state[inside & (g_state == F.STATE_CODES["at_risk"])] = F.STATE_CODES["pending"]
            g_cat = np.full(len(grid_ticks), NOT_AT_RISK, dtype=np.int8)
            g_obs = np.full(len(grid_ticks), -1, dtype=np.int8)
            g_reason = np.zeros(len(grid_ticks), dtype=np.int8)
            out_cue = np.full(len(grid_ticks), -1, dtype=np.int16)
            for ci, (a, bnd, c) in enumerate(all_cues):
                kn_abs = c.start + a
                meta = start_meta.get(kn_abs)
                tone, status, reasons = meta if meta else (None, c.status, [])
                identity = GroupIdentity("self_paced", participant, f"{session}/run-{run.run:04d}", f"cue-{ci}")
                if c.stop_reason == "cut":
                    closing = next(b for b in tones if int(np.searchsorted(ticks, b["known_tick"])) == c.stop + a)
                    stop_tick = int(closing["onset_tick"])
                    stop_reason = "tone_" + closing["class"]
                else:
                    boundary = c.stop + a
                    stop_tick = int(ticks[boundary]) if boundary < m else int(ticks[boundary - 1]) + 1
                    stop_reason = "gap" if boundary < m else "run_end"
                    if bnd in endpoint_cuts and endpoint_cuts[bnd][0] > int(ticks[kn_abs]):
                        stop_tick, stop_reason = endpoint_cuts[bnd]
                unknown = None if c.unknown_from is None else int(ticks[c.unknown_from + a])
                onset = None
                if c.onset is not None:
                    onset = (int(ticks[c.onset[0] + a]), int(ticks[c.onset[1] + a]))
                    if onset[0] >= stop_tick:
                        onset = None
                accepted = None if onset is None else AcceptedOnset(identity, *onset)
                sel = np.flatnonzero(g_cue == ci)
                out_cue[sel] = ci
                for gi in sel:
                    if g_state[gi] != F.STATE_CODES["at_risk"] or grid_ticks[gi] >= stop_tick:
                        if g_state[gi] == F.STATE_CODES["at_risk"]:
                            g_state[gi] = F.STATE_CODES["pending"]
                        continue
                    lab = label_for_decision(RiskDecision(identity, int(grid_ticks[gi]), "at_risk"),
                                             Followup(identity, stop_tick, unknown), accepted,
                                             ticks_per_second=R.SP_TICK_HZ, grid_origin_tick=0)
                    g_cat[gi], g_obs[gi], g_reason[gi] = label_tuple(lab)
                row = {"source": "self_paced", "participant": participant, "session": session, "run": run.run,
                       "cue": ci, "cue_tick": int(tone["onset_tick"]) if tone else None,
                       "start_known_tick": int(ticks[kn_abs]), "end_tick": stop_tick, "stop_reason": stop_reason,
                       "support": status, "support_reasons": ";".join(reasons), "onset_tick": None, "known_tick": None,
                       "rt_ms": None, "confirmation_ms": None, "departure_ms": None,
                       "rejected_crossings": len(c.rejected), "unresolved_at_end": bool(unknown is not None and status == "supported"),
                       "acc_start_tick": None, "onset_minus_accstart_ms": None, "trial_start_minus_cue_ms": None}
                if onset is not None:
                    dep, _ = F.retrospective_departure(values[a:bnd], low, c.onset[0])
                    row.update(onset_tick=onset[0], known_tick=onset[1],
                               rt_ms=onset[0] - row["cue_tick"] if row["cue_tick"] is not None else None,
                               confirmation_ms=onset[1] - onset[0],
                               departure_ms=onset[0] - int(ticks[dep + a]) + SP_SUPPORT_MS[run.fs])
                    departures.append(row["departure_ms"])
                if tone is not None and rows_here and tick_per_deriv:
                    cand = [r for r in rows_here if np.isfinite(r["trial_start_index"])]
                    if cand:
                        best = min(cand, key=lambda r: abs((r["trial_start_index"] - 1) * tick_per_deriv - tone["onset_tick"]))
                        tdiff = tone["onset_tick"] - (best["trial_start_index"] - 1) * tick_per_deriv
                        if abs(tdiff) <= 500:
                            row["trial_start_minus_cue_ms"] = -tdiff
                            if np.isfinite(best["acc_start_index"]) and best["acc_start_index"] > 0:
                                row["acc_start_tick"] = (best["acc_start_index"] - 1) * tick_per_deriv
                                row["acc_flag"] = best["acc_flag"]
                                row["discard"] = best["discard"]
                                if onset is not None:
                                    row["onset_minus_accstart_ms"] = onset[0] - row["acc_start_tick"]
                cue_rows.append(row)
            seen_starts = {a + c.start for a, _, c in all_cues}
            for extra, (k, (tone, _, reasons)) in enumerate(
                    [(k, meta) for k, meta in sorted(start_meta.items()) if k not in seen_starts], start=len(all_cues)):
                cue_rows.append({"source": "self_paced", "participant": participant, "session": session,
                                 "run": run.run, "cue": extra, "cue_tick": int(tone["onset_tick"]),
                                 "start_known_tick": int(ticks[k]), "support": "unknown",
                                 "support_reasons": ";".join([*reasons, "outside_valid_segment"]), "onset_tick": None})
            grids[f"{session}/run-{run.run:04d}"] = (grid_ticks, g_state, out_cue, g_cat, g_obs, g_reason)
        del loaded
    return {"participant": participant, "cues": cue_rows, "grids": grids, "tones": tone_rows,
            "derivatives": deriv_info, "seconds": time.time() - t0}


# =============================================================== reports
def exposure_row(source, participant, grids, cues, tick_hz, grid_ms=50):
    states = np.concatenate([g[1] for g in grids.values()]) if grids else np.zeros(0, np.int8)
    cats = np.concatenate([g[3] for g in grids.values()]) if grids else np.zeros(0, np.int8)
    obs = np.concatenate([g[4] for g in grids.values()]) if grids else np.zeros(0, np.int8)
    reasons = np.concatenate([g[5] for g in grids.values()]) if grids else np.zeros(0, np.int8)
    at = states == F.STATE_CODES["at_risk"]
    row = {"source": source, "participant": participant, "streams": len(grids),
           "cues": len(cues), "cues_supported": sum(c.get("support") == "supported" for c in cues),
           "cues_not_supported": sum(c.get("support") == "not_supported" for c in cues),
           "cues_unknown_support": sum(c.get("support") == "unknown" for c in cues),
           "cues_with_onset": sum(c.get("onset_tick") is not None for c in cues),
           "armed_cues_without_onset": sum(c.get("support") == "supported" and c.get("onset_tick") is None for c in cues),
           "grid_decisions": int(len(states)),
           **{f"decisions_{k}": int(np.sum(states == v)) for k, v in F.STATE_CODES.items()},
           "eligible_stillness_hours": float(at.sum() * grid_ms / 1000 / 3600),
           "pending_hours": float(np.sum(states == F.STATE_CODES["pending"]) * grid_ms / 1000 / 3600)}
    for name, code in REASONS.items():
        if code:
            row[f"labels_{name}"] = int(np.sum(at & (reasons == code)))
    for name, (lo, hi) in {"0_300": (0, 6), "300_600": (6, 12), "600_900": (12, 18)}.items():
        # Status of "first onset in [lo, hi)" is known for any observed category (0-18)
        # or when censored follow-up covers the horizon's upper edge.
        scorable = at & ((cats >= 0) | (obs >= hi))
        positive = at & (cats >= lo) & (cats < hi)
        row[f"scorable_{name}"] = int(scorable.sum())
        row[f"positive_{name}"] = int(positive.sum())
    return row


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--way", default=",".join(str(p) for p in R.WAY_PARTICIPANTS))
    ap.add_argument("--self-paced", default=",".join(R.SP_DEVELOPMENT))
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Run inside a Slurm allocation on a compute node")
    way_people = [int(x) for x in args.way.split(",") if x]
    sp_people = [x for x in args.self_paced.split(",") if x]
    if any(x not in R.SP_DEVELOPMENT for x in sp_people):
        raise SystemExit("held-out self-paced participants are closed")
    out = args.output_dir
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        way = list(pool.map(way_participant, way_people))
        sp_stems = [s for s in R.development_stems() if os.path.basename(s).split("_")[0] in sp_people]
        noise = list(pool.map(sp_noise, sp_stems))
        fit_w = [w for r in noise for w in r["fit"]]
        val_w = [w for r in noise for w in r["val"]]
        sp_fit = F.fit_thresholds(fit_w, F.SP_CONFIRM)[0]
        if not sp_fit.resolved:
            raise ValueError("unresolved self-paced thresholds; refusing label generation")
        low, high = np.array([sp_fit.low]), np.array([sp_fit.high])
        sp_validation = F.validate_thresholds(val_w, low, high, F.SP_CONFIRM, F.SP_QUIET, R.SP_DETECTOR_HZ)
        per_person = {}
        for person in sp_people:
            w = [x for r in noise if r["participant"] == person for x in r["fit"] + r["val"]]
            v = np.concatenate(w)[:, 0] if w else np.zeros(0)
            per_person[person] = {"still_windows": len(w), "median": float(np.median(v)) if len(v) else None,
                                  "q999": float(np.percentile(v, 99.9)) if len(v) else None,
                                  "validation": F.validate_thresholds([x for r in noise if r["participant"] == person for x in r["val"]],
                                                                      low, high, F.SP_CONFIRM, F.SP_QUIET, R.SP_DETECTOR_HZ)}
        sp = list(pool.map(sp_participant, [(person, low, high) for person in sp_people]))
    # ---- outputs
    for r in way:
        np.savez_compressed(os.path.join(out, "labels", f"way_{r['participant']}.npz"),
                            **{f"S{s}_{name}": arr for s, g in r["grids"].items()
                               for name, arr in zip(("tick", "state", "cue", "category", "observed_bins", "reason"), g)})
    for r in sp:
        np.savez_compressed(os.path.join(out, "labels", f"self_paced_{r['participant']}.npz"),
                            **{f"{key.replace('/', '_')}_{name}": arr for key, g in r["grids"].items()
                               for name, arr in zip(("tick", "state", "cue", "category", "observed_bins", "reason"), g)})
    cues = [c for r in way + sp for c in r["cues"]]
    write_csv(os.path.join(out, "way_cues.csv"), [c for r in way for c in r["cues"]])
    write_csv(os.path.join(out, "self_paced_cues.csv"), [c for r in sp for c in r["cues"]])
    write_csv(os.path.join(out, "self_paced_tones.csv"), [t for r in sp for t in r["tones"]])
    exposure = [exposure_row("WAY", r["participant"], r["grids"], r["cues"], 500) for r in way]
    exposure += [exposure_row("self_paced", r["participant"], r["grids"], r["cues"], 1000) for r in sp]
    write_csv(os.path.join(out, "exposure_by_participant.csv"), exposure)
    settings = {"rules": "outputs/component_details.md#12-detector-freeze-rules-october-16-written-before-any-detector-output",
                "constants": {k: getattr(F, k) for k in dir(F) if k.isupper()},
                "way": [r["settings"] for r in way],
                "self_paced": {"fit": sp_fit.__dict__, "low_g": float(low[0]), "high_g": float(high[0]),
                               "noise_windows": {"fit": len(fit_w), "validation": len(val_w),
                                                 "start_tones": sum(r["start_tones"] for r in noise),
                                                 "still": sum(r["still_windows"] for r in noise)},
                               "validation": sp_validation, "per_person_noise": per_person}}
    with open(os.path.join(out, "detector_settings.json"), "w") as f:
        json.dump(settings, f, indent=1, default=float)
    summary = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(), "elapsed_s": time.time() - t0,
               "way_equivalence_generic_vs_WayRiskStream": {
                   k: sum(r["equivalence"][k] for r in way) for k in ("series", "grid_mismatch", "onset_mismatch", "label_mismatch")},
               "way_equivalence_examples": [e for r in way for e in r["equivalence"]["examples"]][:10],
               "self_paced_derivatives": {r["participant"]: r["derivatives"] for r in sp},
               "held_out_or_eeg_opened": False}
    with open(os.path.join(out, "run_report.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary["way_equivalence_generic_vs_WayRiskStream"]), "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
