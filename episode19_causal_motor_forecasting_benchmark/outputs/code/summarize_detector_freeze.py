#!/usr/bin/env python3
"""Summarize the EP19 detector freeze run: crossing term, timing bounds, exposure, agreement.

Reads only the run directory written by run_detector_freeze.py. Timing-bound terms
other than the measured crossing delay come from the October 16 source-fact
classification (source_timing_handoff.md): WAY position acquisition <= 110 ms (p95),
clock <= 2 ms, drift 100 ppm x 660 s = 66 ms, sampling 2 ms; self-paced shared clock 0,
accelerometer module <= 5 ms, sampling 4 ms, causal 1-kHz reduction <= 3 ms.
"""
import argparse
import csv
import json
import os

import numpy as np

WAY_TERMS = {"position_acquisition_p95": 110, "clock": 2, "drift_100ppm_660s": 66, "sampling": 2}
EMG_TERMS = {"emg_acquisition": 5, "clock": 2, "drift_100ppm_660s": 66, "sampling": 2}
SP_TERMS = {"shared_amplifier_clock": 0, "accelerometer_module": 5, "sampling_250hz": 4, "causal_1khz_reduction": 3}


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def dist(values, q=(5, 25, 50, 75, 95, 99)):
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return {"n": 0}
    out = {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
    out.update({f"p{k}": float(np.percentile(v, k)) for k in q})
    return out


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    d = args.run_dir
    settings = json.load(open(os.path.join(d, "detector_settings.json")))
    report = json.load(open(os.path.join(d, "run_report.json")))
    way = read(os.path.join(d, "way_cues.csv"))
    sp = read(os.path.join(d, "self_paced_cues.csv"))
    exposure = read(os.path.join(d, "exposure_by_participant.csv"))
    tones = read(os.path.join(d, "self_paced_tones.csv"))
    out = {"run_dir": d, "job_id": report["job_id"], "status": "calibration evidence, not a scientific result",
           "equivalence_generic_vs_WayRiskStream": report["way_equivalence_generic_vs_WayRiskStream"]}

    # ---------------- crossing delay and bounds
    way_dep = [num(r.get("departure_ms")) for r in way]
    sp_dep = [num(r.get("departure_ms")) for r in sp]
    emg_first = [num(r.get("departure_ms")) for r in way if (r.get("departure_path") or "").startswith("emg")]
    out["crossing_delay_ms"] = {"WAY": dist(way_dep), "WAY_departure_path_emg": dist(emg_first),
                                "self_paced": dist(sp_dep)}
    per_person = {}
    for src, rows in (("WAY", way), ("self_paced", sp)):
        for person in sorted({r["participant"] for r in rows}):
            per_person[f"{src}:{person}"] = dist([num(r.get("departure_ms")) for r in rows if r["participant"] == person],
                                                 q=(50, 95))
    out["crossing_delay_ms_by_participant"] = per_person
    bounds = {}
    for label, q in (("p95", "p95"), ("p99", "p99"), ("max", "max")):
        w = out["crossing_delay_ms"]["WAY"].get(q)
        e = out["crossing_delay_ms"]["WAY_departure_path_emg"].get(q)
        s = out["crossing_delay_ms"]["self_paced"].get(q)
        bounds[label] = {"WAY_ms": None if w is None else sum(WAY_TERMS.values()) + w,
                         "WAY_emg_departure_ms": None if e is None else sum(EMG_TERMS.values()) + e,
                         "self_paced_ms": None if s is None else sum(SP_TERMS.values()) + s}
    out["timing_bound_ms"] = {"terms_WAY": WAY_TERMS, "terms_WAY_emg": EMG_TERMS, "terms_self_paced": SP_TERMS,
                              "crossing_term": "pooled p95 of the backward supra-low run plus 20-ms support (pre-declared)",
                              "primary_p95": bounds["p95"], "sensitivity_p99": bounds["p99"], "sensitivity_max": bounds["max"],
                              "below_300_ms": {k: (v is not None and v < 300) for k, v in bounds["p95"].items()}}
    out["confirmation_delay_ms"] = {"WAY": dist([num(r.get("confirmation_ms")) for r in way]),
                                    "self_paced": dist([num(r.get("confirmation_ms")) for r in sp])}
    out["reaction_time_ms"] = {"WAY_from_led_rise": dist([num(r.get("rt_ms")) for r in way]),
                               "self_paced_from_tone_onset": dist([num(r.get("rt_ms")) for r in sp])}
    out["early_onsets"] = {"WAY_rt_below_150ms": sum(num(r.get("rt_ms")) < 150 for r in way),
                           "self_paced_rt_below_300ms": sum(num(r.get("rt_ms")) < 300 for r in sp)}

    # ---------------- agreement (diagnostic only)
    hs = [num(r.get("onset_minus_handstart_ms")) for r in way]
    acc = [num(r.get("onset_minus_accstart_ms")) for r in sp]
    out["agreement_ms"] = {
        "WAY_label_minus_HandStart": dist(hs),
        "WAY_within_100ms_of_HandStart": float(np.mean([abs(x) <= 100 for x in hs if np.isfinite(x)])) if any(np.isfinite(hs)) else None,
        "WAY_label_before_HandStart": float(np.mean([x < 0 for x in hs if np.isfinite(x)])) if any(np.isfinite(hs)) else None,
        "WAY_HandStart_cue_mapping_ticks": dist([num(r.get("handstart_cue_check_ticks")) for r in way]),
        "self_paced_label_minus_AccStartIndex": dist(acc),
        "self_paced_within_100ms_of_AccStartIndex": float(np.mean([abs(x) <= 100 for x in acc if np.isfinite(x)])) if any(np.isfinite(acc)) else None,
        "self_paced_label_after_AccStartIndex": float(np.mean([x > 0 for x in acc if np.isfinite(x)])) if any(np.isfinite(acc)) else None,
        "self_paced_TrialStart_minus_tone_onset": dist([num(r.get("trial_start_minus_cue_ms")) for r in sp]),
        "by_participant_median": {
            **{f"WAY:{p}": dist([num(r.get("onset_minus_handstart_ms")) for r in way if r["participant"] == p], q=(50,)).get("p50")
               for p in sorted({r["participant"] for r in way})},
            **{f"self_paced:{p}": dist([num(r.get("onset_minus_accstart_ms")) for r in sp if r["participant"] == p], q=(50,)).get("p50")
               for p in sorted({r["participant"] for r in sp})}}}
    sp_lag95 = out["agreement_ms"]["self_paced_label_minus_AccStartIndex"].get("p95")
    out["timing_alternative_if_provider_onset_were_true_onset_ms"] = {
        "self_paced_p95_lag": sp_lag95,
        "self_paced_bound": None if sp_lag95 is None else sum(SP_TERMS.values()) + sp_lag95,
        "note": "AccStartIndex uses zero-phase 10-Hz smoothing and manual review, which can place it earlier "
                "than any causal detector; reported as the strongest alternative to the pre-declared crossing term"}

    # ---------------- exposure
    def total(src, key):
        return sum(num(r[key]) for r in exposure if r["source"] == src)
    tot = {}
    for src in ("WAY", "self_paced"):
        keys = [k for k in exposure[0] if k not in ("source", "participant")]
        tot[src] = {k: total(src, k) for k in keys}
        tot[src]["participants"] = sum(r["source"] == src for r in exposure)
        tot[src]["armed_cue_fraction"] = tot[src]["cues_supported"] / tot[src]["cues"] if tot[src]["cues"] else None
        tot[src]["onset_fraction_of_armed"] = tot[src]["cues_with_onset"] / tot[src]["cues_supported"] if tot[src]["cues_supported"] else None
    out["exposure_totals"] = tot
    out["support_reasons"] = {}
    for src, rows in (("WAY", way), ("self_paced", sp)):
        counts = {}
        for r in rows:
            if r.get("support") != "supported":
                for reason in (r.get("support_reasons") or "unknown").split(";")[:1]:
                    counts[reason] = counts.get(reason, 0) + 1
        out["support_reasons"][src] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    out["onset_paths_WAY"] = {}
    for r in way:
        if r.get("onset_path"):
            out["onset_paths_WAY"][r["onset_path"]] = out["onset_paths_WAY"].get(r["onset_path"], 0) + 1

    # ---------------- noise validation and thresholds
    wv = [s["validation"] for s in settings["way"] if s.get("validation")]
    out["noise_validation"] = {
        "WAY": {"windows": sum(v["windows"] for v in wv),
                "windows_with_any_path_crossing": sum(v["windows_with_any_path_crossing"] for v in wv),
                "noise_minutes": sum(v["noise_minutes"] for v in wv),
                "quiet_entry_available": float(np.average([v["quiet_entry_available_fraction"] for v in wv],
                                                          weights=[v["windows"] for v in wv]))},
        "self_paced": settings["self_paced"]["validation"]}
    nv = out["noise_validation"]["WAY"]
    nv["any_path_crossing_fraction"] = nv["windows_with_any_path_crossing"] / nv["windows"]
    nv["crossings_per_minute"] = nv["windows_with_any_path_crossing"] / nv["noise_minutes"]
    out["thresholds"] = {
        "WAY": {s["participant"]: {"rungs": [f["rung"] for f in s["fits"]],
                                   "low": s["low"], "high": s["high"], "resolved": s["resolved"]} for s in settings["way"]},
        "WAY_paths": settings["way"][0]["paths"],
        "self_paced": {"low_g": settings["self_paced"]["low_g"], "high_g": settings["self_paced"]["high_g"],
                       "rung": settings["self_paced"]["fit"]["rung"], "median_g": settings["self_paced"]["fit"]["median"],
                       "sigma_q_g": settings["self_paced"]["fit"]["sigma_q"],
                       "noise_windows": settings["self_paced"]["noise_windows"]}}
    # ---------------- cues
    counts = {}
    for t in tones:
        counts[t["class"]] = counts.get(t["class"], 0) + 1
    out["tones"] = {"classes": counts,
                    "spurious_burst_test_case": [t for t in tones if t["participant"] == "sub-15" and t["session"] == "ses-01"
                                                 and t["run"] == "1" and 216 < num(t["onset_s"]) < 230]}
    with open(os.path.join(d, "detector_freeze_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    keys = ("crossing_delay_ms", "timing_bound_ms", "agreement_ms", "exposure_totals", "noise_validation",
            "early_onsets", "tones", "equivalence_generic_vs_WayRiskStream")
    print(json.dumps({k: out[k] for k in keys}, indent=1)[:12000])


if __name__ == "__main__":
    main()
