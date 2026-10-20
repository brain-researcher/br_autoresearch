#!/usr/bin/env python3
"""Fix EP16's T11-derived numbers and run the known-positive control on T11 (development only).

Stages, in order, each written before the next starts: packets and realized-design gate; ridge lambda
per feature; J_min from the block-preserving shifted-label null; observed T11 primary statistics and
the raw-score decline margin; known-positive control (uninjected, extra-loss and local-signal
injections, 9,999 draws each). Rules are those of config/analysis_specification.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np

import ep16_branches as br
import ep16_control as pc
from ep16_data import assemble, load_feature_dir, packet_summary, passing_component_days, realized_design, structural_records
from ep16_panel import Panel, bootstrap, one_sided_bound, primary_statistics
from ep16_roles import authorize_protected_read

PARTICIPANT = "T11"
GRID = [10 ** (-2 + 0.25 * k) for k in range(21)]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write(path, payload):
    path.write_text(json.dumps(jsonable(payload), indent=1) + "\n")


def select_lambda(sessions, feature):
    means = {}
    grid = list(GRID)
    for lam in grid:
        means[lam] = float(np.mean(Panel(sessions, feature, lam, 0.0).observed().local["L"]))
    best = max(grid, key=lambda lam: (means[lam], lam))
    extended = None
    if best in (grid[0], grid[-1]):  # an edge maximum extends the grid one decade once
        step = -1 if best == grid[0] else 1
        extra = [best * 10 ** (step * 0.25 * k) for k in range(1, 5)]
        for lam in extra:
            means[lam] = float(np.mean(Panel(sessions, feature, lam, 0.0).observed().local["L"]))
        best = max(means, key=lambda lam: (means[lam], lam))
        extended = "down" if step < 0 else "up"
    return best, {"grid": sorted(means), "mean_local_score": [means[k] for k in sorted(means)], "selected": best,
                  "edge_extension": extended}


def bounds(draws):
    return {"adjusted": one_sided_bound(draws["adjusted"], 0.0125, "lower"),
            "robust": one_sided_bound(draws["robust"], 0.0125, "lower"),
            "fractional_local": one_sided_bound(draws["fractional_local"], 0.0125, "upper"),
            "raw_transported": one_sided_bound(draws["raw_transported"], 0.0125, "upper"),
            "nonestimable": {k: int((~np.isfinite(v)).sum()) for k, v in draws.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--support-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=9999)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")))
    parser.add_argument("--stop-after", choices=("packets", "lambda", "readiness", "observed", "control"), default="control")
    parser.add_argument("--normalization", choices=("block", "none"), default="block")
    args = parser.parse_args(argv)
    if args.draws <= 0 or args.workers <= 0:
        parser.error("--draws and --workers must be positive")
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Compute allocation required")
    authorize_protected_read(PARTICIPANT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "workers": args.workers,
            "participant": PARTICIPANT, "role": "development", "normalization": args.normalization,
            "readiness": "strict", "draws": args.draws}

    days = passing_component_days(args.support_report, PARTICIPANT)
    records = structural_records(args.structural_jsonl, PARTICIPANT, days)
    features = load_feature_dir(args.feature_dir, PARTICIPANT, days)
    sessions, packets, axis = assemble(records, features, normalization=args.normalization)
    design = realized_design(sessions, json.loads(args.support_config.read_text()))
    write(args.output_dir / "packets.json", {**meta, "summary": packet_summary(packets), "axis_channels": len(axis),
          "realized_design": {k: v for k, v in design.items() if k != "contrast_node_shares_by_day"},
          "sessions": [{k: v for k, v in p.items() if k not in ("sides", "eligible_rows")} | {"sides": {s: {"blocks": p["sides"][s]["blocks"],
                        "configuration_composition": p["sides"][s]["configuration_composition"]} for s in p.get("sides", {})}}
                       for p in packets.values()]})
    print(json.dumps({"stage": "packets", "resolved": len(sessions), "design_pass": design["conditional_numerical_pass"],
                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    if not design["conditional_numerical_pass"] or args.stop_after == "packets":
        return

    lam_sbp, sel_sbp = select_lambda(sessions, "sbp")
    lam_tx, sel_tx = select_lambda(sessions, "tx_4_5")
    write(args.output_dir / "lambda_selection.json", {**meta, "sbp": sel_sbp, "tx_4_5": sel_tx})
    print(json.dumps({"stage": "lambda", "sbp": lam_sbp, "tx_4_5": lam_tx, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    if args.stop_after == "lambda":
        return

    panel = Panel(sessions, "sbp", lam_sbp, 0.0)
    null = br.permutation_null(panel)
    shifted = br.shifted_label_null(panel)
    q99 = float(np.quantile(null, 0.99, method="higher"))
    J_min = max(0.01, math.ceil(q99 * 100 - 1e-9) / 100)
    J_obs = panel.observed().local["J"]
    days_sorted = [s.day for s in sessions]
    write(args.output_dir / "readiness.json", {**meta, "permutation_null": {"count": int(len(null)), "q99_higher": q99,
          "max": float(null.max()), "median": float(np.median(null))},
          "shifted_label_null_control": {"count": int(len(shifted)), "q99": float(np.quantile(shifted, 0.99)),
                                         "max": float(shifted.max()), "median": float(np.median(shifted))},
          "J_min": J_min, "D_min": J_min,
          "observed_J": {"min": float(J_obs.min()), "median": float(np.median(J_obs)), "max": float(J_obs.max()),
                         "sessions_at_or_below_J_min": int((J_obs <= J_min).sum()),
                         "days_at_or_below_J_min": [d for d, j in zip(days_sorted, J_obs) if j <= J_min]}})
    print(json.dumps({"stage": "readiness", "J_min": J_min, "J_min_observed": float(J_obs.min())}), flush=True)
    if args.stop_after == "readiness":
        return

    panel = Panel(sessions, "sbp", lam_sbp, J_min)
    observed_draw = panel.observed()
    observed, info = primary_statistics(observed_draw, J_min)
    m_near = (info.get("raw_means") or {}).get(False)
    if m_near is None or not np.isfinite(m_near) or m_near <= 0:
        raise RuntimeError("Cannot freeze the raw-score decline margin: near mean F-B must be finite and positive")
    raw_margin = math.floor(-0.15 * m_near * 1000) / 1000
    tx_stats, tx_info = primary_statistics(Panel(sessions, "tx_4_5", lam_tx, J_min).observed(), J_min)
    lam_frozen, _ = select_lambda(assemble(records, features, normalization="none")[0], "sbp")
    values = {"ridge_lambda_sbp": lam_sbp, "ridge_lambda_tx_4_5": lam_tx, "ridge_lambda_sbp_frozen_scaling": lam_frozen,
              "J_min": J_min, "D_min": J_min,
              "raw_score_decline_margin": raw_margin, "raw_margin_basis_near_mean_F_minus_B": m_near,
              "fixed_utc": datetime.now(timezone.utc).isoformat()}
    write(args.output_dir / "calibration_values.json", {**meta, **values})
    write(args.output_dir / "observed_t11.json", {**meta, "sbp": {"point": observed, "info": info},
          "tx_4_5_description": {"point": tx_stats, "info": tx_info}})
    print(json.dumps({"stage": "observed", **values, "point": observed}), flush=True)
    if args.stop_after == "observed":
        return

    stat = lambda d: primary_statistics(d, J_min)[0]
    support_config = json.loads(args.support_config.read_text())
    variants = {"strict_frozen_readiness": panel}
    kept = [s for s, j in zip(sessions, J_obs) if j > J_min]
    screened_design = realized_design(kept, support_config)
    if screened_design["conditional_numerical_pass"]:
        variants["screened_diagnostic"] = Panel(kept, "sbp", lam_sbp, J_min)
    runs, timing, passes = {}, {}, {}
    for vname, vpanel in variants.items():
        observed_v = primary_statistics(vpanel.observed(), J_min)[0]
        inj_R, report_R = pc.extra_loss_injection(vpanel)
        inj_J, calibration = pc.local_signal_injection(vpanel)
        runs[vname] = {}
        for name, variant in (("uninjected", None), ("extra_loss_injected", inj_R), ("local_signal_injected", inj_J)):
            t0 = time.monotonic()
            if variant is not None and variant.get("status") == "unresolved":
                runs[vname][name] = {"status": "unresolved", "reason": variant["reason"]}
                timing[f"{vname}/{name}"] = time.monotonic() - t0
                print(json.dumps({"stage": "control", "variant": vname, "run": name,
                                  **runs[vname][name]}), flush=True)
                continue
            point = primary_statistics(vpanel.observed(variant), J_min)[0]
            draws = bootstrap(vpanel, stat, args.draws, seed=16, workers=args.workers, variant=variant)
            runs[vname][name] = {"status": "evaluated", "point": point, "bounds": bounds(draws)}
            timing[f"{vname}/{name}"] = time.monotonic() - t0
            np.savez_compressed(args.output_dir / f"draws_{vname}_{name}.npz", **draws)
            print(json.dumps({"stage": "control", "variant": vname, "run": name, "point": point,
                              "bounds": runs[vname][name]["bounds"], "seconds": round(timing[f"{vname}/{name}"], 1)}), flush=True)
        extra_pass = ({"pass": False, "status": "unresolved", "reason": inj_R["reason"]}
                      if inj_R.get("status") == "unresolved" else
                      pc.extra_loss_pass(observed_v, runs[vname]["extra_loss_injected"]["point"],
                                         runs[vname]["extra_loss_injected"]["bounds"]))
        local_pass = ({"pass": False, "status": "unresolved", "reason": inj_J["reason"]}
                      if inj_J.get("status") == "unresolved" else
                      pc.local_signal_pass(observed_v, runs[vname]["local_signal_injected"]["point"],
                                           runs[vname]["local_signal_injected"]["bounds"], raw_margin))
        passes[vname] = {"extra_loss_pass": extra_pass, "local_signal_pass": local_pass,
                         "local_signal_calibration": calibration, "extra_loss_injection": report_R,
                         "sessions": len(vpanel.sessions)}
        passes[vname]["pass"] = bool(extra_pass["pass"] and local_pass["pass"])
    passes["screened_design"] = {k: v for k, v in screened_design.items() if k != "contrast_node_shares_by_day"}
    result = {**meta, "values": values, "runs": runs, "seconds": timing, "variants": passes,
              "elapsed_seconds": time.monotonic() - started,
              "frozen_rule": "strict_frozen_readiness; target readiness gates adjusted/robust, source readiness gates fractional_local, finite scores gate raw_transported, and rank gates adjusted",
              "screened_diagnostic": "sessions with observed J <= J_min removed before pairing; NOT the frozen rule; informs the scientist's decision"}
    result["pass"] = bool(passes["strict_frozen_readiness"]["pass"])
    write(args.output_dir / "positive_control.json", result)
    print(json.dumps({"stage": "done", "pass": result["pass"], "elapsed": round(result["elapsed_seconds"], 1)}), flush=True)


if __name__ == "__main__":
    main()
