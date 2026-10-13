#!/usr/bin/env python3
"""Describe exposed cached R0/R3 predictions; no input payloads or model fitting.

Execute in the coordinator's Slurm compute shell. Reads only the named frozen
run's outer JSON and reports. Preserves every cached row, domain, role and view.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket

OUTPUTS = Path(__file__).resolve().parents[1]
SOURCE = OUTPUTS / "runs/fixed_development_comparison/attempt001"
DESTINATION = OUTPUTS / "runs/cached_projection_diagnostics/attempt001"
ATLASES = ("desikan", "destrieux")
ARMS = ("R0", "R3")
ROLES = ("external", "internal")
EXPECTED_COUNTS = ({"internal": 289, "external": 316},
                   {"internal": 290, "external": 318},
                   {"internal": 292, "external": 320})


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: {actual} != {expected}")


def load_rows(atlas, arm):
    rows = []
    for view in range(3):
        path = SOURCE / "outer" / f"{atlas}_{arm}_view{view}.json"
        cached = read_json(path)
        if (cached["atlas"], cached["arm"], cached["view"]) != (atlas, arm, view):
            raise ValueError(f"Unexpected cached identity: {path}")
        appearances = cached["appearances"]
        if {r["role"] for r in appearances} != set(ROLES):
            raise ValueError(f"Incomplete cached roles: {path}")
        if dict(Counter(r["role"] for r in appearances)) != EXPECTED_COUNTS[view]:
            raise ValueError(f"Incomplete cached role counts: {path}")
        for row in appearances:
            if row["view"] != view:
                raise ValueError(f"Mismatched appearance view: {path}")
            if not all(math.isfinite(row[k]) for k in ("age", "prediction", "absolute_error")):
                raise ValueError(f"Nonfinite cached value: {path}")
            close(abs(row["prediction"] - row["age"]), row["absolute_error"], "cached error")
        rows.extend(appearances)
    keys = [(r["id"], r["view"], r["role"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate cached appearance")
    identities = {}
    for row in rows:
        identity = (str(row["domain"]), row["age"])
        if identities.setdefault(row["id"], identity) != identity:
            raise ValueError("Repeated participant age/domain disagrees")
    return rows


def domain_metrics(rows):
    counts = Counter(r["id"] for r in rows)
    weights = [1.0 / (len(counts) * counts[r["id"]]) for r in rows]
    close(math.fsum(weights), 1.0, "participant-balanced weights")
    residuals = [r["prediction"] - r["age"] for r in rows]
    avg = lambda values: math.fsum(w * x for w, x in zip(weights, values))
    offset = avg(residuals)
    mse = avg([e * e for e in residuals])
    centered_mse = avg([(e - offset) ** 2 for e in residuals])
    close(mse, offset * offset + centered_mse, "weighted MSE decomposition")
    mean_age = avg([r["age"] for r in rows])
    return {
        "participants": len(counts), "appearances": len(rows),
        "appearances_by_view": dict(sorted(Counter(str(r["view"]) for r in rows).items())),
        "age_min": min(r["age"] for r in rows), "age_max": max(r["age"] for r in rows),
        "age_mean": mean_age,
        "age_sd": math.sqrt(avg([(r["age"] - mean_age) ** 2 for r in rows])),
        "mean_residual": offset, "mean_residual_squared": offset * offset,
        "mae": avg([abs(e) for e in residuals]),
        "centered_mae": avg([abs(e - offset) for e in residuals]),
        "mse": mse, "rmse": math.sqrt(mse),
        "centered_mse": centered_mse, "centered_rmse": math.sqrt(centered_mse),
        "decomposition_discrepancy": mse - offset * offset - centered_mse,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["domain"])].append(row)
    domains = {d: domain_metrics(v) for d, v in sorted(groups.items())}
    macro = {k: math.fsum(v[k] for v in domains.values()) / len(domains)
             for k in ("mae", "centered_mae", "mse", "mean_residual_squared", "centered_mse")}
    macro["rmse_from_domain_macro_mse"] = math.sqrt(macro["mse"])
    macro["centered_rmse_from_domain_macro_mse"] = math.sqrt(macro["centered_mse"])
    close(macro["mse"], macro["mean_residual_squared"] + macro["centered_mse"], "macro MSE decomposition")
    return {"domain_count": len(domains), "participants": len({r["id"] for r in rows}),
            "appearances": len(rows), "domain_macro": macro, "domains": domains}


def compare(base, projected):
    if set(base["domains"]) != set(projected["domains"]):
        raise ValueError("Paired domains disagree")
    keys = ("mae", "centered_mae", "mse", "mean_residual_squared", "centered_mse")
    delta = lambda a, b: {k: b[k] - a[k] for k in keys}
    macro = delta(base["domain_macro"], projected["domain_macro"])
    close(macro["mse"], macro["mean_residual_squared"] + macro["centered_mse"], "paired MSE decomposition")
    return {"R3_minus_R0_domain_macro": macro,
            "R3_minus_R0_by_domain": {d: delta(base["domains"][d], projected["domains"][d])
                                      for d in base["domains"]}}


def markdown(report):
    lines = ["# EP14 cached projection diagnostics", "", "Exploratory secondary description of exposed development predictions; no fitting or confirmation.", "",
             "Residual = prediction minus age. Positive values indicate overprediction. All cached ages, domains, roles and views are retained.", "",
             "For pooled-domain summaries, each participant has total weight 1 within its domain, divided equally among that participant's appearances. Residuals are centered using that same pooled-domain weighting. Per-view summaries instead center within view × domain. Predictions are never averaged across views.", "",
             "The exact weighted identity is MSE = squared mean residual + centered residual variance. Domain-macro components average domains equally. Centered MAE is a separate diagnostic: MAE has no such additive decomposition. Centering uses evaluation ages and is not a deployable correction. These components do not identify a causal explanation.", "",
             "JSON includes every domain in both roles and all three per-view summaries. No intervals or significance claims are added.", ""]
    for atlas in ATLASES:
        data = report["atlases"][atlas]
        lines += [f"## {atlas.capitalize()} external domains", "", "| Domain | N / appearances | Age range | R0 offset | R3 offset | R0 MAE | R3 MAE | R0 centered MAE | R3 centered MAE |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        a, b = [data["pooled"]["external"][arm] for arm in ARMS]
        for domain, row in a["domains"].items():
            other = b["domains"][domain]
            lines.append(f"| {domain} | {row['participants']} / {row['appearances']} | {row['age_min']:.2f}–{row['age_max']:.2f} | {row['mean_residual']:.4f} | {other['mean_residual']:.4f} | {row['mae']:.4f} | {other['mae']:.4f} | {row['centered_mae']:.4f} | {other['centered_mae']:.4f} |")
        lines += ["", "| Role / view | Δ MAE (years) | Δ centered MAE (years) | Δ MSE (years²) | Δ squared domain offset (years²) | Δ centered variance (years²) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        panels = [(f"{role} / pooled", data["pooled"][role]) for role in ROLES]
        panels += [(f"{role} / {view}", data["per_view"][str(view)][role]) for view in range(3) for role in ROLES]
        for label, panel in panels:
            d = panel["contrast"]["R3_minus_R0_domain_macro"]
            lines.append(f"| {label} | {d['mae']:.6f} | {d['centered_mae']:.6f} | {d['mse']:.6f} | {d['mean_residual_squared']:.6f} | {d['centered_mse']:.6f} |")
        lines.append("")
    lines += ["The frozen R0–R3 selection and its primary absolute-MAE endpoint are unchanged. Destrieux and repeated RAMP views are sensitivity descriptions, not independent replications. Mean-offset and centered-error differences can reflect several correlated mechanisms; model geometry, support, and biological explanations remain unresolved.", "",
              "Sources: the twelve R0/R3 outer JSON files in `../../fixed_development_comparison/attempt001/outer/`, plus the existing development and controls reports used to verify exact reproduction of their external/internal MAEs.", ""]
    return "\n".join(lines)


def main():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Run only in the coordinator's actual Slurm compute shell")
    if DESTINATION.exists():
        raise FileExistsError(f"Preserve existing descriptive attempt: {DESTINATION}")
    references = {"desikan": read_json(SOURCE / "development_report.json")["results"],
                  "destrieux": read_json(SOURCE / "controls_report.json")["destrieux_sensitivity"]}
    report = {"analysis_id": "cached_projection_diagnostics", "evidence": "exploratory_development_secondary",
              "created_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "source": str(SOURCE),
              "model_fitting": False, "holdout_access": False, "row_filtering": False,
              "centering": "pooled role-domain with participant-balanced appearance weights; per-view role-domain separately",
              "residual_sign": "prediction_minus_age", "independent_replications": False, "atlases": {}}
    all_signatures = []
    for atlas in ATLASES:
        rows = {arm: load_rows(atlas, arm) for arm in ARMS}
        for arm in ARMS:
            all_signatures.append({(r["id"], r["view"], r["role"]): (str(r["domain"]), r["age"]) for r in rows[arm]})
        if all_signatures[-1] != all_signatures[0] or all_signatures[-2] != all_signatures[0]:
            raise ValueError("Participant/view/role support differs across arms or atlases")
        data = {"pooled": {}, "per_view": {}}
        for view in (None, 0, 1, 2):
            panels = {}
            for role in ROLES:
                panel = {arm: summarize([r for r in rows[arm] if r["role"] == role and (view is None or r["view"] == view)]) for arm in ARMS}
                panel["contrast"] = compare(panel["R0"], panel["R3"])
                panels[role] = panel
                if view is None:
                    reference_key = "external_age" if role == "external" else "internal_age"
                    for arm in ARMS:
                        close(panel[arm]["domain_macro"]["mae"], references[atlas][arm][reference_key]["domain_macro"], f"reproduced {atlas} {arm} {role} MAE")
            if view is None:
                data["pooled"] = panels
            else:
                data["per_view"][str(view)] = panels
        report["atlases"][atlas] = data
    report["checks"] = {"all_views_and_roles_present": True, "paired_support_identical": True,
                        "original_MAEs_reproduced": True, "weighted_MSE_identities_passed": True}
    body = markdown(report)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    DESTINATION.mkdir(parents=True, exist_ok=False)
    (DESTINATION / "cached_projection_report.json").write_text(serialized)
    (DESTINATION / "cached_projection_report.md").write_text(body)
    print(json.dumps({"complete": True, "output": str(DESTINATION), "checks": report["checks"]}))


if __name__ == "__main__":
    main()
