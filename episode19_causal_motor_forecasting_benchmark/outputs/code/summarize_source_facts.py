#!/usr/bin/env python3
"""Apply the pre-declared October 16 rules (W1-W9, S1-S5) to the source-fact reports.

Usage: summarize_source_facts.py RUN_DIR  -> writes RUN_DIR/source_facts_summary.json
Rules: outputs/source_timing_handoff.md, October 16 pre-declaration.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

G = 9.81
MASS_KG = {"165g": 0.165, "330g": 0.330, "660g": 0.660}
G_VOLT = 1.45e6


def q(values, probs=(10, 25, 50, 75, 90)):
    v = np.asarray([x for x in values if x is not None], float)
    return {"n": int(v.size), **({f"p{p}": float(np.percentile(v, p)) for p in probs} if v.size else {})}


def frac(values, test):
    v = list(values)
    return float(np.mean([x is not None and test(x) for x in v])) if v else None


def way_summary(way):
    files = way["files"]
    trials = [dict(t, participant=f["participant"], series=f["series"]) for f in files for t in f["trials"]]
    out = {"files": len(files), "windows": len(trials)}

    # W1 LED polarity and cue position.
    w1 = []
    for f in files:
        ts = sorted((t for t in f["trials"] if "hs_high_s" in t), key=lambda t: t["hs_rise"])
        dur_ok = sum(abs(t["hs_high_s"] - t["ws_led_s"]) <= 0.004 for t in ts)
        pos_ok = sum(abs(t["rise_minus_n0_kin"] - 1000) <= 2 for t in ts)
        low = [(b["hs_rise"] - a["hs_fall"]) / 500 for a, b in zip(ts[:-1], ts[1:])]
        low_match = sum(abs(l - b["ws_led_s"]) <= 0.004 for l, b in zip(low, ts[1:]))
        w1.append({"P": f["participant"], "S": f["series"], "windows": f["ws_windows"], "hs_rises": f["hs_rises"],
                   "duration_matches": dur_ok, "position_matches": pos_ok, "low_reading_matches": low_match,
                   "led_exact_0_1_fraction": f["led_exact_0_1_fraction"],
                   "polarity_high_lit": dur_ok == f["ws_windows"] == pos_ok and low_match <= 0.1 * max(1, len(low))})
    out["W1"] = {"files_passing": sum(r["polarity_high_lit"] for r in w1),
                 "extra_hs_rises_total": sum(r["hs_rises"] - r["windows"] for r in w1),
                 "failures": [r for r in w1 if not r["polarity_high_lit"]],
                 "led_exact_0_1_fraction_min": min(r["led_exact_0_1_fraction"] for r in w1)}

    # W2 WS -> HS origin binding.
    off = defaultdict(int)
    eeg_kin, emg_kin, exact = defaultdict(int), defaultdict(int), defaultdict(int)
    for t in trials:
        off[t["n0_kin_minus_500tst"]] += 1
        eeg_kin[t["n0_eeg"] - t["n0_kin"]] += 1
        emg_kin[t["n0_emg"] - 8 * t["n0_kin"]] += 1
        exact[(bool(t["kin_match"] and t["kin_match"]["exact"]), bool(t["eeg_match"] and t["eeg_match"]["exact"]),
               bool(t["emg_match"] and t["emg_match"]["exact"]))] += 1
    eeg_t0_unique = [len({round(t["eeg_t0"], 6) for t in f["trials"]}) for f in files]
    out["W2"] = {"n0_minus_500tst": dict(off), "eeg_minus_kin": dict(eeg_kin), "emg_minus_8kin": dict(emg_kin),
                 "exact_kin_eeg_emg": {str(k): v for k, v in exact.items()},
                 "min_r": {s: min(t[f"r_{s}"] for t in trials) for s in ("kin", "eeg", "emg")},
                 "rise_minus_n0": dict(sorted({k: sum(1 for t in trials if t.get("rise_minus_n0_kin") == k)
                                               for k in {t.get("rise_minus_n0_kin") for t in trials}}.items(),
                                              key=lambda kv: (kv[0] is None, kv[0]))),
                 "fall_minus_expected": {str(k): sum(1 for t in trials if t.get("fall_minus_expected") == k)
                                         for k in {t.get("fall_minus_expected") for t in trials}},
                 "files_with_single_eeg_t0": sum(u == 1 for u in eeg_t0_unique),
                 "ws_header_years": {y: sum(y in f["ws_header"] for f in files) for y in ("2013", "2014")},
                 "hs_header_years": {y: sum(y in f["hs_header"] for f in files) for y in ("2013", "2014")}}

    # W3 HS LED vs HS wrist motion.
    lat = [t.get("wrist_motion_latency_s") for t in trials if "hs_high_s" in t]
    pre = [t.get("wrist_speed_max_pre_rise") for t in trials if "hs_high_s" in t]
    out["W3"] = {"latency_s": q(lat), "fraction_latency_0p1_to_1p5": frac(lat, lambda x: 0.1 <= x <= 1.5),
                 "fraction_pre_rise_speed_below_10": frac(pre, lambda x: x < 10.0),
                 "per_participant_median_latency": {p: float(np.median([t["wrist_motion_latency_s"] for t in trials
                                                                        if t["participant"] == p and t.get("wrist_motion_latency_s") is not None]))
                                                    for p in range(1, 13)}}

    # W4 force units and weight codebook.
    w4 = {}
    for wid, m in MASS_KG.items():
        rows = [t for t in trials if t.get("weight_id") == wid and "LF_hold_net" in t]
        if not rows:
            continue
        per_p = {p: float(np.median([t["LF_hold_net"] for t in rows if t["participant"] == p]))
                 for p in sorted({t["participant"] for t in rows})}
        target = m * G
        w4[wid] = {"target_N": target, "codes": sorted({t["weight"] for t in rows}),
                   "env_at_rise": q([t["weight_env_at_rise"] for t in rows], (50,)),
                   "LF_hold_net": q([t["LF_hold_net"] for t in rows]), "LF_hold_raw_p50": float(np.median([t["LF_hold_raw"] for t in rows])),
                   "pooled_within_10pct": abs(np.median([t["LF_hold_net"] for t in rows]) / target - 1) <= 0.10,
                   "participant_medians": per_p,
                   "participants_within_15pct": sum(abs(v / target - 1) <= 0.15 for v in per_p.values()),
                   "participants": len(per_p)}
    out["W4"] = {"by_weight": w4, "LF_rest": q([t.get("LF_rest") for t in trials if "LF_rest" in t])}

    # W5 surface codebook via GF/LF in friction series.
    kinds = {}
    for f in files:
        ws_w = {t["weight"] for t in f["trials"]}
        ws_s = {t["surf"] for t in f["trials"]}
        kinds[(f["participant"], f["series"])] = ("weight" if len(ws_w) > 1 and len(ws_s) == 1 else
                                                  "friction" if len(ws_s) > 1 and len(ws_w) == 1 else
                                                  "mixed" if len(ws_s) > 1 and len(ws_w) > 1 else "constant")
    weight_series_surf = sorted({(t["surf"], t["surf_id"]) for t in trials if kinds[(t["participant"], t["series"])] == "weight"})
    ratio = defaultdict(list)
    for t in trials:
        if kinds[(t["participant"], t["series"])] == "friction" and t.get("LF_hold_net", 0) > 0.5:
            ratio[(t["surf"], t["surf_id"])].append(t["GF_hold_net"] / t["LF_hold_net"])
    out["W5"] = {"series_kinds": {k: sum(v == k for v in kinds.values()) for k in set(kinds.values())},
                 "series_kind_by_file": {f"P{p}S{s}": v for (p, s), v in sorted(kinds.items())},
                 "weight_series_surfaces": weight_series_surf,
                 "friction_GF_over_LF": {f"{k[0]}:{k[1]}": q(v, (25, 50, 75)) for k, v in sorted(ratio.items())},
                 "surface_codes_env_vs_ws": sorted({(t["surf"], t["surf_id"], round(t["surf_env_at_rise"]))
                                                    for t in trials if "surf_env_at_rise" in t}),
                 "weight_codes_env_vs_ws": sorted({(t["weight"], t["weight_id"], round(t["weight_env_at_rise"]))
                                                   for t in trials if "weight_env_at_rise" in t}),
                 "weight_code_change_before_rise_s": q([t.get("weight_code_last_change_before_rise_s") for t in trials]),
                 "surf_code_change_before_rise_s": q([t.get("surf_code_last_change_before_rise_s") for t in trials])}

    # W6 position units and frame.
    auto = [t for t in trials if "hs_high_s" in t and not t["button_press_since_prev_fall"]]
    out["W6"] = {"lift_height": q([t.get("lift_height") for t in trials]),
                 "lift_height_participant_medians": {p: float(np.median([t["lift_height"] for t in trials
                                                                          if t["participant"] == p and "lift_height" in t]))
                                                     for p in range(1, 13)},
                 "aperture_hold": q([t.get("aperture_hold") for t in trials]),
                 "d_digit_min_last1s": q([t.get("d_digit_min_last1s") for t in trials]),
                 "no_button_trials": len(auto),
                 **{f"auto_fraction_1to3s_{lab}": frac([t.get(f"delay_since_{lab}_below15_s") for t in auto],
                                                     lambda x: 1.0 <= x <= 3.0) for lab in ("min", "index", "thumb")},
                 "auto_delay_min": q([t.get("delay_since_min_below15_s") for t in auto], (1, 5, 50, 95, 99)),
                 "auto_min_dist_last1s": q([t.get("d_digit_min_last1s") for t in auto], (1, 5, 50)),
                 "wrist_disp_hold_xyz_median": [float(np.median([t["wrist_disp_hold_xyz"][i] for t in trials
                                                                 if "wrist_disp_hold_xyz" in t])) for i in range(3)]}

    # W7 FASTRAK update structure; W8 magnet marker; W9 lift-off lag.
    out["W7"] = {"px4_equal_run_median": q([f["px4_equal_run_median"] for f in files], (10, 50, 90)),
                 "px4_kink_spacing_median": q([f["px4_kink_spacing_median"] for f in files], (10, 50, 90)),
                 "px4_kink_spacing_mode": q([f["px4_kink_spacing_mode"] for f in files], (10, 50, 90)),
                 "px4_zero_diff_fraction": q([f["px4_zero_diff_fraction"] for f in files], (10, 50, 90))}
    mags = [f["magnet_check"] for f in files if f["magnet_check"]["edges"] >= 3]
    out["W8"] = {"files_with_3plus_edges": len(mags),
                 "files_peak_over_5sd": sum(m["peak_over_baseline_sd"] > 5 for m in mags),
                 "peak_over_sd": q([m["peak_over_baseline_sd"] for m in mags]),
                 "peak_lag_samples": q([m["peak_lag_samples"] for m in mags])}
    out["W8"]["usable"] = out["W8"]["files_peak_over_5sd"] >= 0.5 * max(1, len(mags))
    out["W9"] = {k: q([t.get(k) for t in trials]) for k in ("liftoff_lag_02_s", "liftoff_lag_05_s")}
    out["durations_s"] = q([f["duration_s"] for f in files], (0, 50, 100))
    return out


def label_cues(onsets, gaps, kinds):
    labels = [None] * len(onsets)
    for i, k in enumerate(kinds):
        if k == "start_end":
            labels[i], labels[i + 1] = labels[i] or "start", "end"
        elif k == "end_start":
            labels[i], labels[i + 1] = labels[i] or "end", "start"
    return labels


def self_paced_summary(sp):
    runs = sp["runs"]
    out = {"runs": len(runs)}
    se = [g for r in runs for g, k in zip(r["cue_gaps_s"], r["cue_gap_kinds"]) if k == "start_end"]
    es = [g for r in runs for g, k in zip(r["cue_gaps_s"], r["cue_gap_kinds"]) if k == "end_start"]
    med = {"start_end": float(np.median(se)), "end_start": float(np.median(es))}
    flags, irregular, cycles = [], [], []
    for r in runs:
        for i, (g, k) in enumerate(zip(r["cue_gaps_s"], r["cue_gap_kinds"])):
            if k in med and abs(g - med[k]) > 0.06:
                flags.append({"run": r["stem"], "index": i, "kind": k, "gap_s": g, "deviation_s": g - med[k]})
            if k == "irregular":
                irregular.append({"run": r["stem"], "index": i, "gap_s": g, "of": len(r["cue_gaps_s"])})
        labels = label_cues(r["cue_onset_s"], r["cue_gaps_s"], r["cue_gap_kinds"])
        starts = [(i, t) for i, (t, l) in enumerate(zip(r["cue_onset_s"], labels)) if l == "start"]
        for (i, a), (j, b) in zip(starts[:-1], starts[1:]):
            if j == i + 2 and labels[i + 1] == "end":
                cycles.append({"run": r["stem"], "start_index": i, "cycle_s": b - a})
    cyc_med = float(np.median([c["cycle_s"] for c in cycles]))
    cyc_flags = [dict(c, deviation_s=c["cycle_s"] - cyc_med) for c in cycles if abs(c["cycle_s"] - cyc_med) > 0.06]
    out["S4"] = {"class_medians_s": med, "start_end": q(se, (1, 50, 99)), "end_start": q(es, (1, 50, 99)),
                 "intervals": len(se) + len(es), "flagged_intervals": flags, "irregular_intervals": irregular,
                 "cycle_median_s": cyc_med, "cycles": len(cycles), "cycle_s": q([c["cycle_s"] for c in cycles], (1, 50, 99)),
                 "flagged_cycles_post_hoc": cyc_flags,
                 "longest_constant_ge25": [(r["stem"], r["longest_constant_samples"]) for r in runs
                                           if max(r["longest_constant_samples"].values()) >= 25],
                 "longest_constant_max": {k: max(r["longest_constant_samples"][k] for r in runs) for k in ("eeg_ch1", "TRIG", "X")}}
    fits = sp["accelerometer_sphere_fits"]
    out["S3"] = {"pooled": fits.get("all_development_sessions"),
                 "sessions": {k: {kk: v.get(kk) for kk in ("radius_over_1g_spec", "rms_resid_over_radius",
                                                           "max_angle_from_mean_deg", "vectors")} | (
                     {"center_over_1g": float(np.linalg.norm(v["center_uV"]) / G_VOLT)} if "center_uV" in v else {})
                     for k, v in fits.items() if k != "all_development_sessions"},
                 "zero_offset_norm_over_1g": q([r["acc_norm_median_uV"] / G_VOLT for r in runs], (0, 10, 50, 90, 100)),
                 "acc_abs_max_V": q([r["acc_abs_max_uV"] / 1e6 for r in runs], (50, 90, 100)),
                 "axis_std_over_rest_std_min": float(min(min(a / max(b, 1e-9) for a, b in zip(r["axis_std_uV"], r["axis_rest_std_uV"]))
                                                         for r in runs))}
    tl = sp["timeline"]
    out["S5"] = {"sessions": len(tl), "negative_gaps": [(k, v["runs"], v["between_run_gaps_s"]) for k, v in tl.items()
                                                       if any(g is not None and g < 0 for g in v["between_run_gaps_s"])],
                 "between_run_gap_s": q([g for v in tl.values() for g in v["between_run_gaps_s"] if g is not None], (0, 50, 100))}
    out["S1"] = {"timeline": tl.get("sub-03_ses-02"),
                 "derivative": {"numberOfRuns": sp["sub03_ses02_derivative"]["numberOfRuns"],
                                "runs": [{"derivative_run": r["derivative_run"], "frames": r["frames"], "trials": r["trials"],
                                          "discarded": r["discarded_trials"],
                                          "matched_raw_runs": [c["raw_run"] for c in r["raw_candidates"] if c["frames_equal"]],
                                          "max_scaled_diff_uV": max((v for c in r["raw_candidates"] if c["frames_equal"]
                                                                     for kk, v in c.items() if kk.endswith("_scaled")), default=None)}
                                         for r in sp["sub03_ses02_derivative"]["runs"]]},
                 "vmrk_datafile_mismatches": [r["stem"] for r in runs if not r["vmrk_datafile_ok"]],
                 "vhdr_datafile_mismatches": [r["stem"] for r in runs if not r["vhdr_datafile_ok"]]}
    out["S1"]["derivative"]["kept_trials"] = sum(r["trials"] - r["discarded"] for r in out["S1"]["derivative"]["runs"])
    out["fs_counts"] = {str(k): sum(r["fs"] == k for r in runs) for k in {r["fs"] for r in runs}}
    out["new_segment_counts"] = {str(k): sum(r["new_segments"] == k for r in runs) for k in {r["new_segments"] for r in runs}}
    return out


def main():
    run = sys.argv[1]
    with open(os.path.join(run, "way_source_facts.json")) as f:
        way = json.load(f)
    with open(os.path.join(run, "self_paced_source_facts.json")) as f:
        sp = json.load(f)
    summary = {"way": way_summary(way), "self_paced": self_paced_summary(sp),
               "jobs": [way["job_id"], sp["job_id"]], "held_out_touched": False}
    with open(os.path.join(run, "source_facts_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
