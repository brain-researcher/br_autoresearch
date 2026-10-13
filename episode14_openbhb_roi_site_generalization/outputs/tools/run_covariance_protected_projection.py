#!/usr/bin/env python3
"""EP14 exact fit-covariance protection; fixed development-only counterfactual.

Original scale/directions and raw/original predictions are borrowed from the
completed age-adjusted diagnostic. Only two protected arms are newly fitted.
Protecting fitting covariance is not a guarantee about ridge or held-out age.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from threadpoolctl import threadpool_limits

import run_age_adjusted_projection as source
import run_development as legacy
import run_discarded_direction_diagnostics as prior

SEED = 14014
ALPHAS = source.ALPHAS.copy()
ATLASES = ("desikan", "destrieux")
ARMS = ("raw", "original", "covariance_protected", "sham_protected")
PAIRS = (("covariance_protected", "original"), ("sham_protected", "original"),
         ("covariance_protected", "sham_protected"), ("covariance_protected", "raw"),
         ("original", "raw"))
model_name = source.model_name
MODELS = tuple(model_name(a, alpha) for a in ARMS for alpha in ALPHAS)
REPLICATES = 10000
OUTPUTS = Path(__file__).resolve().parents[1]
EXPOSED = OUTPUTS.parent / "inputs/openbhb/exposed"
SOURCE = OUTPUTS / "runs/age_adjusted_projection/attempt001"
RUN_ROOT = OUTPUTS / "runs/covariance_protected_projection"
ANALYSIS_ID = "covariance_protected_projection"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariance(z, ages):
    ages = np.asarray(ages, dtype=float)
    return z.T @ (ages - ages.mean()) / len(ages)


def protected_directions(original, c):
    """Project four original rows into c-perp; fail rather than change rank."""
    original, c = np.asarray(original), np.asarray(c)
    if original.shape[0] != 4 or original.shape[1] != c.size:
        raise prior.SupportFailure("Original directions must have four rows matching covariance")
    prior.require_close(original @ original.T, np.eye(4), "Original row orthonormality")
    norm = float(np.linalg.norm(c))
    if not np.isfinite(norm) or norm <= 0:
        raise prior.SupportFailure("Protected fitting covariance has zero/nonfinite norm")
    unit = c / norm
    rows = original - (original @ unit)[:, None] * unit[None, :]
    _, singular, vt = np.linalg.svd(rows, full_matrices=False)
    tolerance = max(rows.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank != 4:
        raise prior.SupportFailure(f"Covariance-perpendicular original row space has rank {rank}, needs 4")
    directions = vt[:4]
    prior.require_close(directions @ directions.T, np.eye(4), "Protected row orthonormality",
                        atol=1e-10, rtol=1e-10)
    prior.require_close(directions @ unit, np.zeros(4), "Protected unit covariance orthogonality",
                        atol=1e-10, rtol=1e-10)
    prior.require_close(c - directions.T @ (directions @ c), c, "Pc=c",
                        atol=1e-9, rtol=1e-10)
    return directions, {"projected_row_rank": rank, "projected_row_singular_values": singular.tolist(),
                        "rank_tolerance": float(tolerance),
                        "protected_covariance_norm": norm,
                        "protected_covariance_residual_norm": float(np.linalg.norm(directions @ c)),
                        "pc_minus_c_max_abs": float(np.max(np.abs(directions.T @ (directions @ c))))}


def project(z, directions):
    """Features only, in the borrowed original standardized metric."""
    return z - (z @ directions.T) @ directions


def direction_metadata(z, directions, original, actual_c):
    singular = np.clip(np.linalg.svd(directions @ original.T, compute_uv=False), 0, 1)
    removed_energy = float(np.sum((z @ directions.T)**2))
    age_energy = float(actual_c @ actual_c)
    removed_age_energy = float(np.sum((directions @ actual_c)**2))
    return {"removed_rank": 4,
            "principal_angles_degrees_vs_original": np.degrees(np.arccos(singular)).tolist(),
            "projector_frobenius_displacement_vs_original": float(np.sqrt(max(0, 8 - 2 * np.sum(singular**2)))),
            "fit_removed_frobenius_norm": float(np.sqrt(removed_energy)),
            "fit_removed_energy": removed_energy,
            "fit_removed_energy_fraction": removed_energy / float(np.sum(z**2)),
            "fit_marginal_age_covariance_norm": float(np.sqrt(age_energy)),
            "removed_age_covariance_norm": float(np.sqrt(removed_age_energy)),
            "removed_age_covariance_squared_norm_fraction": removed_age_energy / age_energy
            if age_energy else None}


def verify_cached_rows(record, bundle, view, support):
    """Exact evaluated identities/order and fit complement; no target-based fitting."""
    split = bundle.splits[view]
    evaluated = np.concatenate((split["internal"], split["external"]))
    rows = record["appearances"]
    if record["view"] != view or len(rows) != len(evaluated) or record["support"] != support:
        raise ValueError("Borrowed view, row count or support disagrees")
    bins = prior.age_bins(bundle.age[evaluated])
    for j, i in enumerate(evaluated):
        expected = {"id": str(bundle.ids[i]), "domain": str(bundle.sites[i]), "view": view,
                    "role": "internal" if j < len(split["internal"]) else "external",
                    "age": float(bundle.age[i]), "age_bin": int(bins[j]),
                    "on_common_support": bool(bins[j] in support["eligible_bins"])}
        if any(rows[j][key] != value for key, value in expected.items()):
            raise ValueError("Borrowed evaluated identity/domain/age/support mismatch")
    fit_ids = set(map(str, bundle.ids[split["fit"]]))
    if fit_ids != set(map(str, bundle.ids)) - {r["id"] for r in rows}:
        raise ValueError("Borrowed evaluation complement does not exactly identify fit participants")
    return evaluated


def outer_fold(bundle, x, atlas, view, out):
    path = out / "outer" / f"{atlas}_view{view}.json"
    if path.exists():
        if not path.with_suffix(".npz").exists():
            raise ValueError("Checkpoint missing fitted state")
        return legacy.read_json(path)
    started = time.monotonic()
    cached_path = SOURCE / "outer" / path.name
    cached = legacy.read_json(cached_path)
    fit = bundle.splits[view]["fit"]
    y, sites = bundle.age[fit], bundle.sites[fit]
    support = prior.fit_support(y, sites)
    evaluated = verify_cached_rows(cached, bundle, view, support)
    metadata = cached["direction_metadata"]
    if (cached["atlas"] != atlas or metadata["fit_rows"] != len(fit)
            or metadata["input_columns"] != x.shape[1]
            or metadata["fit_domain_count"] != len(set(sites))
            or metadata["age_mean"] != float(y.mean()) or metadata["age_scale"] != float(y.std())
            or cached["alphas"] != ALPHAS.tolist() or cached["anchor_alpha"] != 1000):
        raise ValueError("Borrowed fit/atlas/penalty metadata differs")
    with np.load(cached_path.with_suffix(".npz"), allow_pickle=False) as state:
        mean, scale = state["pre_mean"].copy(), state["pre_scale"].copy()
        original = state["directions_original"].copy()
    recomputed = legacy.Scale().fit(x[fit])
    if not np.array_equal(mean, recomputed.mean) or not np.array_equal(scale, recomputed.scale):
        raise ValueError("Borrowed original scale is not exactly the current fit-only scale")
    if original.shape != (4, x.shape[1]) or metadata["arms"]["original"]["removed_rank"] != 4:
        raise prior.SupportFailure("Borrowed original directions are not rank four")
    z = (x[fit] - mean) / scale
    held = (x[evaluated] - mean) / scale
    true_c = covariance(z, y)
    sham_y = source.within_site_sham(y, sites, seed=SEED)
    sham_c = covariance(z, sham_y)
    arms_metadata = {"original": direction_metadata(z, original, original, true_c)}
    prior.require_close(arms_metadata["original"]["fit_removed_energy_fraction"],
                        metadata["arms"]["original"]["fit_removed_energy_fraction"],
                        "Borrowed original removed energy", atol=1e-12, rtol=1e-10)
    prior.require_close(arms_metadata["original"]["removed_age_covariance_squared_norm_fraction"],
                        metadata["arms"]["original"]["removed_age_covariance_squared_norm_fraction"],
                        "Borrowed original age covariance", atol=1e-12, rtol=1e-10)
    predictions, correctness, directions, ranks = {}, {}, {}, {}
    n_internal = len(bundle.splits[view]["internal"])
    for arm, target, c in (("covariance_protected", y, true_c), ("sham_protected", sham_y, sham_c)):
        direction, checks = protected_directions(original, c)
        directions[arm] = direction
        train, test = project(z, direction), project(held, direction)
        preserved = covariance(train, target)
        prior.require_close(preserved, c, f"{arm} training covariance", atol=1e-9, rtol=1e-10)
        checks["transformed_training_covariance_max_abs_error"] = float(np.max(np.abs(preserved - c)))
        arms_metadata[arm] = {**direction_metadata(z, direction, original, true_c), **checks}
        predicted, rank = legacy.ridge_grid(train, y, test, ALPHAS)
        if rank > min(len(fit) - 1, x.shape[1] - 4):
            raise prior.SupportFailure(f"{arm} retained rank exceeds rank-four removal")
        ranks[arm] = rank
        for column, alpha in enumerate(ALPHAS):
            predictions[model_name(arm, alpha)] = predicted[:, column]
        probe = legacy.fit_classifier(train, sites)
        correctness[arm] = probe.predict(test[:n_internal]) == bundle.sites[bundle.splits[view]["internal"]]
    appearances = []
    for j, old in enumerate(cached["appearances"]):
        row = {k: v for k, v in old.items() if k not in ("predictions", "site_correctness")}
        row["predictions"] = {model_name(a, alpha): old["predictions"][model_name(a, alpha)]
                              for a in ARMS[:2] for alpha in ALPHAS}
        row["predictions"].update({name: float(value[j]) for name, value in predictions.items()})
        row["site_correctness"] = None if j >= n_internal else {
            **{a: old["site_correctness"][a] for a in ARMS[:2]},
            **{a: bool(value[j]) for a, value in correctness.items()}}
        appearances.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), pre_mean=mean, pre_scale=scale,
                        fit_ids=bundle.ids[fit].astype(str), evaluated_ids=bundle.ids[evaluated].astype(str),
                        true_fit_covariance=true_c, sham_fit_covariance=sham_c,
                        directions_original=original,
                        **{f"directions_{a}": v for a, v in directions.items()})
    record = {"atlas": atlas, "view": view, "appearances": appearances, "support": support,
              "alphas": ALPHAS.tolist(), "anchor_alpha": 1000, "alpha_selection": "none_fixed_curve",
              "direction_metadata": {"fit_rows": len(fit), "input_columns": x.shape[1],
                                     "fit_domain_count": len(set(sites)), "arms": arms_metadata,
                                     "true_and_sham_rotation_can_differ": True},
              "retained_effective_ranks": {**{a: cached["retained_effective_ranks"][a] for a in ARMS[:2]}, **ranks},
              "cached_source": str(cached_path),
              "cached_sha256": {p.name: fingerprint(p) for p in (cached_path, cached_path.with_suffix(".npz"))},
              "baseline_predictions_and_correctness_reused_exactly": True,
              "borrowed_scale_verified_exactly": True, "borrowed_fit_complement_verified_exactly": True,
              "evaluation_transform_uses_ages": False, "post_projection_scaling": False,
              "site_probe": "C1_balanced_unrescaled_same_representation_no_age_input",
              "seconds": time.monotonic() - started}
    legacy.write_json(path, record)
    print(json.dumps({"stage": "outer", "atlas": atlas, "view": view, "seconds": record["seconds"]}), flush=True)
    return record


def score_panel(groups, centered, common, intervals, restrict=False):
    chosen = {d: g for d, g in groups.items()
              if not (common and (centered or restrict)) or g["coverage"]["common_centered_eligible"]}
    if not chosen:
        return {"status": "insufficient_support", "included_domains": [], "domain_macro_mae": None,
                "contrasts": {}, "bootstrap": {"status": "unavailable_no_eligible_domains"}}
    point = np.mean([g["centered" if centered else "absolute"] for g in chosen.values()], axis=0)
    boot = source.bootstrap_models(list(chosen.values()), centered, models=MODELS,
                                   replicates=REPLICATES, seed=SEED) if intervals else None
    contrasts = {}
    for alpha in ALPHAS:
        for a, b in PAIRS:
            other, base = model_name(a, alpha), model_name(b, alpha)
            ai, bi = MODELS.index(other), MODELS.index(base)
            item = {"other": other, "reference": base, "delta_mae_years": float(point[ai] - point[bi])}
            if boot is not None:
                item["descriptive_percentiles_2_5_97_5"] = np.quantile(boot[:, ai] - boot[:, bi], [.025, .975]).tolist()
            contrasts[f"{other}_minus_{base}"] = item
    return {"status": "descriptive", "included_domains": list(chosen),
            "participants": sum(g["n"] for g in chosen.values()),
            "domain_macro_mae": dict(zip(MODELS, map(float, point))), "contrasts": contrasts,
            "bootstrap": {"status": "complete" if boot is not None else
                          ("unavailable_fewer_than_two_domains" if intervals else "not_requested_per_view"),
                          "replicates": REPLICATES if boot is not None else 0, "seed": SEED,
                          "scope": "conditional_on_observed_fixed_support_and_eligible_domains",
                          "participant_appearances_kept_together": True,
                          "centering_recomputed_each_resampled_domain": centered}}


def summarize_population(rows, common, intervals):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    groups, domains = {}, {}
    for domain, all_rows in sorted(by_domain.items()):
        selected = [r for r in all_rows if not common or r["on_common_support"]]
        coverage = {"total_participants": len({r["id"] for r in all_rows}), "total_appearances": len(all_rows)}
        if not selected:
            domains[domain] = {**coverage, "status": "no_rows_on_fit_defined_support", "participants": 0,
                               "appearances": 0, "common_centered_eligible": False,
                               "absolute_mae": None, "centered_mae": None}
            continue
        group = source.group_arrays(selected, models=MODELS)
        groups[domain] = group
        domains[domain] = {**coverage, **group["coverage"],
                           "status": "descriptive" if not common or group["coverage"]["common_centered_eligible"]
                           else "insufficient_common_centered_support",
                           "absolute_mae": dict(zip(MODELS, map(float, group["absolute"]))),
                           "centered_mae": dict(zip(MODELS, map(float, group["centered"])))}
    panels = {"absolute": score_panel(groups, False, common, intervals),
              "centered": score_panel(groups, True, common, intervals)}
    if common:
        panels["absolute_on_centered_eligible_domains"] = score_panel(groups, False, True, intervals, restrict=True)
    for panel in panels.values():
        panel["excluded_domains"] = [d for d in domains if d not in panel["included_domains"]]
    return {"population": "fit_defined_common_support" if common else "all_rows", "domains": domains,
            "centering": "within_panel_domain_participant_balanced_appearances", **panels}


def site_summary(rows, intervals):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["role"] != "internal" or row["site_correctness"] is None:
            raise ValueError("Site summary requires internal correctness")
        grouped[row["domain"]][row["id"]].append(row)
    groups, domains = [], {}
    for domain, people in sorted(grouped.items()):
        correct = np.array([[np.mean([r["site_correctness"][a] for r in appearances]) for a in ARMS]
                            for appearances in people.values()])
        domains[domain] = {"participants": len(people), "appearances": sum(map(len, people.values())),
                           "class_recall": dict(zip(ARMS, map(float, correct.mean(axis=0))))}
        groups.append({"n": len(people), "person_abs": correct})
    point = {a: float(np.mean([d["class_recall"][a] for d in domains.values()])) for a in ARMS}
    boot = source.bootstrap_models(groups, False, models=ARMS, replicates=REPLICATES, seed=SEED) if intervals else None
    contrasts = {}
    for a, b in PAIRS:
        item = {"delta_balanced_accuracy": point[a] - point[b]}
        if boot is not None:
            item["descriptive_percentiles_2_5_97_5"] = np.quantile(
                boot[:, ARMS.index(a)] - boot[:, ARMS.index(b)], [.025, .975]).tolist()
        contrasts[f"{a}_minus_{b}"] = item
    return {"domains": domains, "balanced_accuracy": point, "contrasts": contrasts,
            "bootstrap": {"replicates": REPLICATES if boot is not None else 0, "seed": SEED,
                          "status": "complete" if boot is not None else "not_requested_per_view",
                          "scope": "conditional_on_observed_internal_domains"},
            "scope": "ordinary_unconditional_site_probe_descriptive_no_new_guardrail"}


def validate_rows(records):
    rows = [r for record in records for r in record["appearances"]]
    seen, identities = set(), {}
    for r in rows:
        key = (r["id"], r["view"], r["role"])
        if key in seen or identities.setdefault(r["id"], (r["domain"], r["age"])) != (r["domain"], r["age"]):
            raise ValueError("Duplicate appearance or inconsistent participant")
        seen.add(key)
        if set(r["predictions"]) != set(MODELS) or not all(np.isfinite(v) for v in r["predictions"].values()):
            raise ValueError("Incomplete finite paired predictions")
        if r["role"] == "internal":
            if r["site_correctness"] is None or set(r["site_correctness"]) != set(ARMS):
                raise ValueError("Incomplete paired internal correctness")
        elif r["site_correctness"] is not None:
            raise ValueError("External correctness must remain unavailable")
    for view in range(3):
        expected = dict(zip(("internal", "external"), legacy.EXPECTED_SPLITS[view][1:]))
        if Counter(r["role"] for r in rows if r["view"] == view) != expected:
            raise ValueError("Appearance population differs from original views")
    return rows


def aggregate(out):
    report = {"analysis_id": ANALYSIS_ID, "evidence": "exploratory_development_only", "atlases": {},
              "fixed_alphas": ALPHAS.tolist(), "anchor_alpha": 1000, "alpha_selection": "none",
              "public_validation_access": False, "original_selection_changed": False,
              "fit_covariance_protection_guaranteed_to_numerical_tolerance": True,
              "heldout_age_or_ridge_prediction_protection_guaranteed": False,
              "true_and_sham_rotation_can_differ": True, "independent_replications": False,
              "conditional_site_probe": False, "single_within_site_sham_seed": SEED}
    signature = None
    for atlas in ATLASES:
        records = [legacy.read_json(out / "outer" / f"{atlas}_view{v}.json") for v in range(3)]
        rows = validate_rows(records)
        current = {(r["id"], r["view"], r["role"]): (r["domain"], r["age"], r["on_common_support"]) for r in rows}
        if signature is not None and current != signature:
            raise ValueError("Atlas population/support differs")
        signature = current
        result = {"direction_metadata": [r["direction_metadata"] for r in records],
                  "fit_defined_support": [r["support"] for r in records],
                  "pooled": {}, "per_view": {}, "site_probe": {"per_view_internal": {}}}
        for role in ("external", "internal"):
            selected = [r for r in rows if r["role"] == role]
            result["pooled"][role] = {key: summarize_population(selected, common, True)
                                      for key, common in (("all_rows", False), ("common_support", True))}
        result["site_probe"]["pooled_internal"] = site_summary([r for r in rows if r["role"] == "internal"], True)
        for v in range(3):
            result["per_view"][str(v)] = {role: {key: summarize_population(
                [r for r in rows if r["role"] == role and r["view"] == v], common, False)
                for key, common in (("all_rows", False), ("common_support", True))} for role in ("external", "internal")}
            result["site_probe"]["per_view_internal"][str(v)] = site_summary(
                [r for r in rows if r["role"] == "internal" and r["view"] == v], False)
        report["atlases"][atlas] = result
        print(json.dumps({"stage": "aggregate", "atlas": atlas}), flush=True)
    legacy.write_json(out / "covariance_protected_report.json", report)
    lines = ["# Exact fitting-covariance protection", "",
             "Exploratory exposed-development counterfactual. Original fit scale and four site directions are reused. Projecting those directions into the true or within-site-shuffled fitting covariance's orthogonal complement preserves rank four. No final scaling. All downstream ages are true fitting ages; evaluation transforms receive features only.", "",
             "Raw/original predictions and ordinary seen-site correctness are copied exactly from the completed source. Protecting fitting covariance does not preserve ridge predictions or held-out utility. True and sham rotate the original subspace by potentially different amounts; principal angles, projector displacement and removed feature energy are reported in JSON.", "",
             "Fixed alpha1000 anchor; all five penalties retained without selection. Equal participants within domain, then equal domains. Descriptive 95% intervals use 10,000 paired hierarchical domain/participant draws with centers recomputed. Common-support eligibility stays fixed; zero-support domains remain visible. Final holdout stays closed and original no-candidate unchanged.", ""]
    for atlas, data in report["atlases"].items():
        lines += [f"## {atlas}", "", "| Population | Metric | Comparison | Δ MAE years | Descriptive 95% interval | Domains |", "| --- | --- | --- | ---: | --- | ---: |"]
        for population, panels in data["pooled"]["external"].items():
            for metric in ("absolute", "centered", "absolute_on_centered_eligible_domains"):
                if metric not in panels:
                    continue
                score = panels[metric]
                for a, b in PAIRS:
                    c = score["contrasts"].get(f"{model_name(a, 1000)}_minus_{model_name(b, 1000)}")
                    if c is None:
                        continue
                    ci = c.get("descriptive_percentiles_2_5_97_5")
                    interval = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "unavailable"
                    lines.append(f"| {population} | {metric} | {a} − {b} | {c['delta_mae_years']:.4f} | {interval} | {len(score['included_domains'])} |")
        lines.append("")
    (out / "covariance_protected_report.md").write_text("\n".join(lines))
    return report


def initialize_attempt(out, resume):
    dependencies = (Path(__file__), Path(source.__file__), Path(legacy.__file__), Path(prior.__file__))
    code = {p.name: fingerprint(p) for p in dependencies}
    source_manifest = legacy.read_json(SOURCE / "attempt.json")
    if legacy.read_json(SOURCE / "completion.json")["status"] != "complete":
        raise ValueError("Borrowed prerequisite is incomplete")
    for name, value in source_manifest["code_sha256"].items():
        if code.get(name) != value:
            raise ValueError(f"Borrowed source code changed: {name}")
    source_files = [SOURCE / "attempt.json", SOURCE / "completion.json"]
    source_files += [SOURCE / "outer" / f"{a}_view{v}.{ext}" for a in ATLASES for v in range(3) for ext in ("json", "npz")]
    hashes = {str(p.relative_to(SOURCE)): fingerprint(p) for p in source_files}
    manifest = out / "attempt.json"
    if resume:
        old = legacy.read_json(manifest)
        if old["code_sha256"] != code or old["source_sha256"] != hashes:
            raise ValueError("Resume requires identical code and borrowed states/predictions")
    else:
        out.mkdir(parents=True, exist_ok=False)
        legacy.write_json(manifest, {"analysis_id": ANALYSIS_ID, "code_sha256": code, "source_sha256": hashes,
                                    "borrowed_run": str(SOURCE), "exposed_source": str(EXPOSED),
                                    "created_utc": datetime.now(timezone.utc).isoformat(),
                                    "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
                                    "public_validation_access": False, "seed": SEED, "anchor_alpha": 1000})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN_ROOT / "attempt001")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--representative-one-fold", action="store_true")
    args = parser.parse_args()
    legacy.compute_only()
    if not 1 <= args.threads <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Threads exceed allocation")
    out = args.out.resolve()
    if not out.is_relative_to(RUN_ROOT):
        raise ValueError("Output outside this analysis run root")
    initialize_attempt(out, args.resume)
    started = time.monotonic()
    try:
        with threadpool_limits(limits=args.threads):
            bundle = legacy.Bundle(EXPOSED)
            if args.representative_one_fold:
                x, _ = bundle.atlas("destrieux")
                outer_fold(bundle, x, "destrieux", 0, out)
                return
            for atlas in ATLASES:
                x, _ = bundle.atlas(atlas)
                for view in range(3):
                    outer_fold(bundle, x, atlas, view, out)
            aggregate(out)
    except prior.SupportFailure as error:
        legacy.write_json(out / "support_failure.json", {"status": "support_failure", "reason": str(error),
                                                         "no_capacity_substitution": True})
        raise
    legacy.write_json(out / "completion.json", {"status": "complete", "analysis_id": ANALYSIS_ID,
                                               "seconds": time.monotonic() - started,
                                               "job_id": os.environ.get("SLURM_JOB_ID"),
                                               "hostname": socket.gethostname(),
                                               "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"complete": True, "analysis_id": ANALYSIS_ID, "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
