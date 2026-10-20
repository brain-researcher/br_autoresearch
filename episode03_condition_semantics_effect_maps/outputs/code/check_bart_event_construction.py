#!/usr/bin/env python3
"""Check how the ds000001 BART `_demean` event columns were constructed.

Reads only BIDS events TSVs, which are model inputs: the copy FitLins used
(modified in place by the producer) and the pristine OpenNeuro source copy.
Opens no images, statistical maps, reports or scores. Standard library only.
"""

import argparse
import csv
import json
import math
import os
import socket
import statistics
import time
from collections import Counter
from pathlib import Path

PARAM = ["pumps_demean", "control_pumps_demean", "cash_demean", "explode_demean"]
FILLED = PARAM + ["response_time"]
TERMINAL = ("cash_demean", "explode_demean")
STEP_TOL = 0.01   # TSV values are rounded; a unit step must be within 0.01
FIT_TOL = 0.01


def num(text):
    text = (text or "").strip()
    if text.lower() in ("", "n/a", "nan", "na"):
        return math.nan
    return float(text)


def read_events(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return reader.fieldnames, rows


def compare_used_to_source(used_cols, used, src_cols, src):
    """The documented producer change is NaN -> 0 in FILLED columns only."""
    out = {"same_columns": used_cols == src_cols, "same_row_count": len(used) == len(src),
           "zero_filled_cells": Counter(), "other_differences": []}
    if not (out["same_columns"] and out["same_row_count"]):
        return out
    for i, (u, s) in enumerate(zip(used, src)):
        for col in src_cols:
            if col == "trial_type":
                if u[col] != s[col]:
                    out["other_differences"].append([i, col, s[col], u[col]])
                continue
            sv, uv = num(s[col]), num(u[col])
            if math.isnan(sv):
                if col in FILLED and uv == 0.0:
                    out["zero_filled_cells"][col] += 1
                elif not math.isnan(uv):
                    out["other_differences"].append([i, col, s[col], u[col]])
            elif math.isnan(uv) or abs(sv - uv) > 1e-9:
                out["other_differences"].append([i, col, s[col], u[col]])
    out["zero_filled_cells"] = dict(out["zero_filled_cells"])
    out["other_differences"] = out["other_differences"][:20]
    return out


def linear_fit(xs, ys):
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = max(abs(y - (intercept + slope * x)) for x, y in zip(xs, ys))
    return {"n": n, "slope": slope, "intercept": intercept, "max_abs_residual": resid}


def segments_by_steps(values, breaks):
    """Split a sequence of modulator values into runs separated by `breaks`."""
    segs, cur = [], []
    for i, v in enumerate(values):
        if cur and breaks[i]:
            segs.append(cur)
            cur = []
        cur.append(v)
    if cur:
        segs.append(cur)
    return segs


def step_summary(segs):
    steps = [b - a for seg in segs for a, b in zip(seg, seg[1:])]
    return {"segments": len(segs), "steps": len(steps),
            "unit_steps": sum(abs(s - 1) <= STEP_TOL for s in steps),
            "zero_steps": sum(abs(s) <= STEP_TOL for s in steps),
            "negative_steps": sum(s < -STEP_TOL for s in steps),
            "other_positive_steps": sum(s > STEP_TOL and abs(s - 1) > STEP_TOL for s in steps)}


def analyze_run(cols, rows):
    rows = sorted(rows, key=lambda r: num(r["onset"]))
    tt = [r["trial_type"] for r in rows]
    res = {"rows": len(rows), "columns": cols, "trial_type_counts": dict(Counter(tt))}

    # Where each modulator carries a value.
    occupancy = {}
    for p in PARAM:
        own = [num(r[p]) for r in rows if r["trial_type"] == p]
        other = [num(r[p]) for r in rows if r["trial_type"] != p]
        vals = [v for v in own if not math.isnan(v)]
        occupancy[p] = {
            "own_rows": len(own), "own_nan": len(own) - len(vals),
            "other_rows_non_nan": sum(not math.isnan(v) for v in other),
            "other_rows_nonzero": sum((not math.isnan(v)) and v != 0 for v in other),
            "own_mean": statistics.fmean(vals) if vals else None,
            "own_min": min(vals, default=None), "own_max": max(vals, default=None)}
    res["occupancy"] = occupancy

    # Reward balloons: consecutive reward-pump rows closed by a cash-out or explosion row.
    balloons, pumps = [], []
    for r in rows:
        t = r["trial_type"]
        if t == "pumps_demean":
            pumps.append(num(r["pumps_demean"]))
        elif t in TERMINAL:
            balloons.append({"terminal": t, "terminal_value": num(r[t]), "pump_values": pumps})
            pumps = []
    res["reward_balloons"] = len(balloons)
    res["trailing_unterminated_reward_pumps"] = len(pumps)
    reward_segs = [b["pump_values"] for b in balloons if b["pump_values"]]
    res["reward_pump_steps_within_balloon"] = step_summary(reward_segs)
    res["reward_balloon_offsets"] = {
        # offset = 1-based pump index minus modulator value; constant across
        # balloons means centering across the run, (K+1)/2 means within balloon.
        "offset_spread": None, "offset_equals_within_balloon_mean_index": None}
    offsets, within = [], []
    for seg in reward_segs:
        k = len(seg)
        offs = [(i + 1) - v for i, v in enumerate(seg)]
        offsets.append(statistics.fmean(offs))
        within.append(abs(statistics.fmean(offs) - (k + 1) / 2) <= FIT_TOL)
    if offsets:
        res["reward_balloon_offsets"] = {
            "offset_spread": max(offsets) - min(offsets),
            "offset_mean": statistics.fmean(offsets),
            "offset_equals_within_balloon_mean_index": sum(within),
            "balloons_with_pumps": len(offsets)}

    # Terminal modulators against the number of reward pumps in their balloon.
    terminal = {}
    for t in TERMINAL:
        pts = [(len(b["pump_values"]), b["terminal_value"]) for b in balloons
               if b["terminal"] == t and not math.isnan(b["terminal_value"])]
        terminal[t] = {"events": len(pts), "fit_value_on_pump_count": linear_fit(
            [p[0] for p in pts], [p[1] for p in pts]),
            "mean_value": statistics.fmean([p[1] for p in pts]) if pts else None}
    res["terminal_vs_pump_count"] = terminal

    # Control balloons have no terminal event. Split control-pump rows only where
    # another event intervenes; back-to-back control balloons then appear as
    # negative resets, which are counted separately from within-balloon steps.
    ctrl_vals, ctrl_breaks, prev_ctrl = [], [], False
    for r in rows:
        if r["trial_type"] == "control_pumps_demean":
            ctrl_breaks.append(not prev_ctrl)
            ctrl_vals.append(num(r["control_pumps_demean"]))
            prev_ctrl = True
        else:
            prev_ctrl = False
    res["control_pump_steps_within_segment"] = step_summary(segments_by_steps(ctrl_vals, ctrl_breaks))

    # Response times and durations by event type (duration feeds the RT regressor).
    rt = {}
    for t in sorted(set(tt)):
        sub = [r for r in rows if r["trial_type"] == t]
        rts = [num(r["response_time"]) for r in sub]
        durs = [num(r["duration"]) for r in sub]
        rt[t] = {"events": len(sub), "response_time_nan": sum(math.isnan(v) for v in rts),
                 "response_time_zero": sum(v == 0 for v in rts),
                 "duration_equals_response_time": sum(
                     (not math.isnan(a)) and (not math.isnan(b)) and abs(a - b) <= 1e-6
                     for a, b in zip(durs, rts)),
                 "duration_min": min(durs), "duration_max": max(durs)}
    res["response_time_and_duration"] = rt
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--used-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    runs, errors = [], []
    for sub in range(1, 17):
        for run in (1, 2, 3):
            rel = f"sub-{sub:02d}/func/sub-{sub:02d}_task-balloonanalogrisktask_run-{run:02d}_events.tsv"
            try:
                ucols, used = read_events(args.used_root / rel)
                scols, src = read_events(args.source_root / rel)
                runs.append({"events": rel,
                             "used_vs_source": compare_used_to_source(ucols, used, scols, src),
                             "source_construction": analyze_run(scols, src),
                             "used_construction": analyze_run(ucols, used)})
            except Exception as exc:  # report and continue; nothing is retried
                errors.append({"events": rel, "error": repr(exc)})

    def total(path):
        acc = Counter()
        for r in runs:
            node = r
            for key in path:
                node = node[key]
            acc.update({k: v for k, v in node.items() if isinstance(v, int)})
        return dict(acc)

    report = {
        "analysis_id": "source_handoff_qualification",
        "check": "bart_event_column_construction",
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "used_root": str(args.used_root), "source_root": str(args.source_root),
        "runs_checked": len(runs), "errors": errors,
        "images_maps_reports_or_scores_opened": False,
        "summary": {
            "runs_with_other_used_vs_source_differences": sum(
                bool(r["used_vs_source"]["other_differences"]) or not r["used_vs_source"]["same_columns"]
                or not r["used_vs_source"]["same_row_count"] for r in runs),
            "zero_filled_cells_total": total(["used_vs_source", "zero_filled_cells"]),
            "reward_pump_steps": total(["source_construction", "reward_pump_steps_within_balloon"]),
            "control_pump_steps": total(["source_construction", "control_pump_steps_within_segment"]),
            "own_row_nan_by_column": {p: sum(r["source_construction"]["occupancy"][p]["own_nan"] for r in runs) for p in PARAM},
            "other_row_nonzero_by_column": {p: sum(r["source_construction"]["occupancy"][p]["other_rows_nonzero"] for r in runs) for p in PARAM},
            "max_abs_run_mean_by_column": {p: max(abs(r["source_construction"]["occupancy"][p]["own_mean"] or 0) for r in runs) for p in PARAM},
            "reward_offset_spread_max": max((r["source_construction"]["reward_balloon_offsets"].get("offset_spread") or 0) for r in runs),
            "reward_balloons_within_balloon_centered": sum(r["source_construction"]["reward_balloon_offsets"].get("offset_equals_within_balloon_mean_index") or 0 for r in runs),
            "reward_balloons_with_pumps": sum(r["source_construction"]["reward_balloon_offsets"].get("balloons_with_pumps") or 0 for r in runs),
            "terminal_fits": {t: [r["source_construction"]["terminal_vs_pump_count"][t]["fit_value_on_pump_count"] for r in runs] for t in TERMINAL},
        },
        "runs": runs,
        "elapsed_seconds": time.time() - t0,
    }
    for t in TERMINAL:
        fits = [f for f in report["summary"]["terminal_fits"][t] if f]
        report["summary"][f"{t}_runs_with_fit"] = len(fits)
        report["summary"][f"{t}_slope_range"] = [min(f["slope"] for f in fits), max(f["slope"] for f in fits)] if fits else None
        report["summary"][f"{t}_max_abs_residual"] = max((f["max_abs_residual"] for f in fits), default=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
