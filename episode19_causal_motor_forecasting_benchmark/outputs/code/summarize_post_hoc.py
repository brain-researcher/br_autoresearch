#!/usr/bin/env python3
"""Post hoc summaries selected after job 48183086 (labelled post hoc in every record).

Usage: summarize_post_hoc.py RUN_DIR -> RUN_DIR/post_hoc_summary.json
"""
import json
import os
import sys

import numpy as np


def q(v, probs=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    v = np.asarray([x for x in v if x is not None], float)
    return {"n": int(v.size), **{f"p{p}": float(np.percentile(v, p)) for p in probs}} if v.size else {"n": 0}


def kind(f):
    w = {t["weight"] for t in f["trials"]}
    s = {t["surf"] for t in f["trials"]}
    return "weight" if len(w) > 1 and len(s) == 1 else "friction" if len(s) > 1 and len(w) == 1 else "mixed"


def cue_cycles(runs, med_se, med_es, cycle_med):
    out = {"cold_start_first_cycle": [], "spurious_burst": [], "ambiguous_mid_run": [], "short_compensated": 0, "cycles": 0}
    for r in runs:
        on, kinds, gaps = r["cue_onset_s"], r["cue_gap_kinds"], r["cue_gaps_s"]
        lab = [None] * len(on)
        for i, k in enumerate(kinds):
            if k == "start_end":
                lab[i], lab[i + 1] = lab[i] or "start", "end"
            elif k == "end_start":
                lab[i], lab[i + 1] = lab[i] or "end", "start"
        starts = [i for i, l in enumerate(lab) if l == "start"]
        cyc = [(i, on[j] - on[i] if (j == i + 2 and lab[i + 1] == "end") else None) for i, j in zip(starts[:-1], starts[1:])]
        out["cycles"] += sum(c is not None for _, c in cyc)
        for n, (i, c) in enumerate(cyc):
            if c is None or c - cycle_med >= -0.06:
                continue
            nb = [cyc[m][1] - cycle_med for m in (n - 1, n + 1) if 0 <= m < len(cyc) and cyc[m][1] is not None]
            if any(abs((c - cycle_med) + x) <= 0.06 for x in nb):
                out["short_compensated"] += 1
                continue
            rec = {"run": r["stem"], "start_cue_index": i, "start_cue_s": on[i], "cycle_dev_s": c - cycle_med,
                   "start_end_dev_s": gaps[i] - med_se, "end_start_dev_s": gaps[i + 1] - med_es}
            if abs(gaps[i] - med_se) > 1.0 or abs(gaps[i + 1] - med_es) > 1.0:
                out["spurious_burst"].append(rec)
            elif i == 0:
                out["cold_start_first_cycle"].append(rec)
            else:
                out["ambiguous_mid_run"].append(rec)
    return out


def main():
    run = sys.argv[1]
    way = json.load(open(os.path.join(run, "way_source_facts.json")))
    sp = json.load(open(os.path.join(run, "self_paced_source_facts.json")))
    fu = json.load(open(os.path.join(run, "followups_post_hoc.json")))
    summ = json.load(open(os.path.join(run, "source_facts_summary.json")))
    rule = {}
    for k in ("weight", "friction", "mixed"):
        rows = [t for f in way["files"] if kind(f) == k for t in f["trials"] if "hs_high_s" in t]
        rule[k] = {"trials": len(rows)}
        for lab in ("min", "index", "thumb"):
            d = [t.get(f"delay_since_{lab}_below15_s") for t in rows]
            rule[k][lab] = {"fraction_1_to_3s": float(np.mean([x is not None and 1.0 <= x <= 3.0 for x in d])),
                            "delay_s": q(d)}
    trials = [t for f in way["files"] for t in f["trials"]]
    s4 = summ["self_paced"]["S4"]
    cyc = cue_cycles(sp["runs"], s4["class_medians_s"]["start_end"], s4["class_medians_s"]["end_start"], s4["cycle_median_s"])
    spec = {k: {m: q([f["spectra"][k][m] for f in fu["way"]], (10, 50, 90)) for m in ("f95_hz", "f99_hz", "power_above_60hz")}
            for k in ("Px4", "Pz1")}
    env = {}
    for f in fu["way"]:
        for k, v in f["env_code_timing"].items():
            env[k] = env.get(k, 0) + v
    n = env.pop("n")
    sp_fu = fu["self_paced"]
    out = {"labelled": "post hoc (selected after job 48183086)",
           "W6_cue_rule_by_series_type": rule,
           "W9_liftoff_lag_s": {k: q([t.get(k) for t in trials]) for k in ("liftoff_lag_02_s", "liftoff_lag_05_s")},
           "W7_velocity_spectra": spec,
           "env_code_agreement_with_ws": {k: v / n for k, v in env.items()} | {"trials": n},
           "button_channel": {"high_fraction": q([f["button_high_fraction"] for f in fu["way"]], (0, 50, 100)),
                              "falling_edges_minus_windows": sorted({f["button_falling_edges"] - len(w["trials"])
                                                                     for f, w in zip(fu["way"], way["files"])})},
           "S4_cycle_decomposition": {"cycles": cyc["cycles"], "short_compensated": cyc["short_compensated"],
                                      "cold_start_first_cycle": len(cyc["cold_start_first_cycle"]),
                                      "spurious_burst": cyc["spurious_burst"],
                                      "ambiguous_mid_run": cyc["ambiguous_mid_run"],
                                      "ambiguous_dev_s": q([c["cycle_dev_s"] for c in cyc["ambiguous_mid_run"]], (0, 50, 100)),
                                      "ambiguous_sessions": sorted({c["run"][:13] for c in cyc["ambiguous_mid_run"]})},
           "S4_mains_phase_test": {"snr60_median": q([r["snr60_median"] for r in sp_fu], (0, 10, 50, 90)),
                                   "phase_resid_sd_deg": q([r["resid_sd_deg"] for r in sp_fu if r["resid_sd_deg"] is not None], (10, 50, 90)),
                                   "verdict": "uninformative: 60-Hz SNR too low for sample-level phase continuity"}}
    with open(os.path.join(run, "post_hoc_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "S4_cycle_decomposition"}, indent=1)[:6000])
    d = out["S4_cycle_decomposition"]
    print({k: (len(v) if isinstance(v, list) else v) for k, v in d.items()})


if __name__ == "__main__":
    main()
