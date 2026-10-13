#!/usr/bin/env python3
"""Bounded EP14 exposed-development bilateral-prior comparison.

This never changes the completed R0--R3 comparison or opens public validation.
The coordinator records the prospective amendment before launching this worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import run_development as base

ALPHAS = np.array([.1, 1., 10., 100., 1000., 10000., 100000.])
RATIOS = np.array([.01, .1, 1., 10., 100.])
SHAM_SEEDS = (14015, 14016, 14017)
ARMS = ("raw", "homologous", "sham14015", "sham14016", "sham14017")
EPISODE = Path(__file__).resolve().parents[2]
EXPOSED = EPISODE / "inputs/openbhb/exposed"
PROVIDER_FAMILIES = (
    "surface_area_mm^2", "gray_matter_volume_mm^3", "average_thickness_mm",
    "thickness_stddev_mm", "integrated_rectified_mean_curvature_mm^-1",
    "integrated_rectified_gaussian_curvature_mm^-2", "intrinsic_curvature_index",
)
SPEC = {
    "analysis_id": "bilateral_regularization", "atlas": "desikan",
    "alphas": ALPHAS.tolist(), "asymmetric_shared_penalty_ratios": RATIOS.tolist(),
    "sham_seeds": list(SHAM_SEEDS), "arms": list(ARMS),
    "transform": "fit_raw_standardization_then_orthonormal_pairs_then_asymmetry_divide_sqrt_ratio",
    "tuning": "inner_leave_one_siteXacq_out_equal_domain_MAE",
    "tie_break": ["ratio_closest_to_one_on_log_scale", "larger_alpha", "smaller_ratio"],
    "scoring": "average_appearance_losses_per_participant_then_equal_domains",
    "holdout_access": False, "candidate_promotion": False,
    "original_frozen_disposition": "development_no_candidate_unchanged",
}


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise DeadlineReached("bounded worker deadline reached; preserve completed folds")


def authenticate_pairs(names, declared_families):
    """Require explicit provider measurement suffixes; never infer anatomical groups."""
    pairs, unpaired = base.bilateral_mapping(names)
    if len(names) != 476 or len(pairs) != 238 or unpaired:
        raise ValueError("Desikan requires the previously admitted 238 bilateral pairs")
    if tuple(declared_families) != PROVIDER_FAMILIES:
        raise ValueError("declared measurement families differ from provider channel reference")
    families = {family: [] for family in declared_families}
    for i, (left, right) in enumerate(pairs):
        matches = [family for family in families if names[left].endswith("_" + family)]
        if len(matches) != 1 or names[left][3:] != names[right][3:]:
            raise ValueError(f"ambiguous or unmatched provider family: {names[left]}")
        families[matches[0]].append(i)
    if any(len(indices) != 34 for indices in families.values()):
        raise ValueError("each of seven provider measurement families must contain 34 pairs")
    mappings = {"homologous": np.asarray(pairs, dtype=int)}
    for seed in SHAM_SEEDS:
        rng = np.random.RandomState(seed)
        shuffled = np.asarray(pairs, dtype=int).copy()
        for family in sorted(families):
            indices = families[family]
            shuffled[indices, 1] = rng.permutation(shuffled[indices, 1])
        mappings[f"sham{seed}"] = shuffled
    return mappings, {family: len(indices) for family, indices in families.items()}


def admitted_metadata():
    reference = EXPOSED / "reference/freesurfer_channels.txt"
    families = tuple(line.strip() for line in reference.read_text().splitlines() if line.strip())
    if families != PROVIDER_FAMILIES:
        raise ValueError("provider channel reference differs from prospectively declared seven families")
    names = base.read_json(EXPOSED / "feature_names_desikan.json")
    mappings, counts = authenticate_pairs(names, families)
    return names, mappings, counts


def rotate(z, pairs, ratio=1.):
    left, right = pairs[:, 0], pairs[:, 1]
    return np.column_stack(((z[:, left] + z[:, right]) / np.sqrt(2.),
                            (z[:, left] - z[:, right]) / np.sqrt(2. * ratio)))


def choose(losses, ratios=RATIOS):
    return min(np.ndindex(losses.shape), key=lambda ij: (
        float(losses[ij]), abs(float(np.log10(ratios[ij[0]]))),
        -float(ALPHAS[ij[1]]), float(ratios[ij[0]])))


def inner_fold(x, age, sites, fit, mappings, view, domain_number, out, deadline):
    path = out / "inner" / f"view{view}_domain{domain_number}.json"
    if path.exists():
        return base.read_json(path)
    check_deadline(deadline)
    start = time.monotonic()
    domain = np.unique(sites[fit])[domain_number]
    train, valid = fit[sites[fit] != domain], fit[sites[fit] == domain]
    scale = base.Scale().fit(x[train])
    z, held = scale.transform(x[train]), scale.transform(x[valid])
    raw, rank = base.ridge_grid(z, age[train], held, ALPHAS)
    raw_loss = np.mean(np.abs(raw - age[valid, None]), axis=0)
    losses = {"raw": raw_loss[None, :].tolist()}
    for arm, pairs in mappings.items():
        by_ratio = []
        for ratio in RATIOS:
            check_deadline(deadline)
            if ratio == 1:
                # An orthonormal change of basis preserves isotropic ridge.
                by_ratio.append(raw_loss.tolist())
            else:
                prediction, _ = base.ridge_grid(rotate(z, pairs, ratio), age[train],
                                                 rotate(held, pairs, ratio), ALPHAS)
                by_ratio.append(np.mean(np.abs(prediction - age[valid, None]), axis=0).tolist())
        losses[arm] = by_ratio
    record = {"view": view, "domain": str(domain), "train_rows": len(train),
              "validation_rows": len(valid), "raw_rank": rank, "losses": losses,
              "seconds": time.monotonic() - start}
    base.write_json(path, record)
    print(json.dumps({"stage": "inner", **{k: v for k, v in record.items() if k != "losses"}}), flush=True)
    return record


def run_view(view, out_string, deadline):
    with threadpool_limits(limits=1):
        out = Path(out_string)
        path = out / "outer" / f"view{view}.json"
        if path.exists():
            return {"view": view, "complete": True, "reused": True}
        try:
            check_deadline(deadline)
            # Header-based family admission precedes feature matrix access.
            names, mappings, counts = admitted_metadata()
            bundle = base.Bundle(EXPOSED)
            x, checked_names = bundle.atlas("desikan")
            if checked_names != names:
                raise ValueError("feature headers changed during admission")
            split, age, sites = bundle.splits[view], bundle.age, bundle.sites
            fit = split["fit"]
            inner = [inner_fold(x, age, sites, fit, mappings, view, i, out, deadline)
                     for i in range(len(np.unique(sites[fit])))]
            scale = base.Scale().fit(x[fit])
            both = np.concatenate((split["internal"], split["external"]))
            z, held = scale.transform(x[fit]), scale.transform(x[both])
            raw_grid, _ = base.ridge_grid(z, age[fit], held, ALPHAS)
            records = {}
            for arm in ARMS:
                check_deadline(deadline)
                means = np.mean([r["losses"][arm] for r in inner], axis=0)
                ratios = np.array([1.]) if arm == "raw" else RATIOS
                ri, ai = choose(means, ratios)
                ratio, alpha = float(ratios[ri]), float(ALPHAS[ai])
                if arm == "raw":
                    transformed, evaluation = z, held
                else:
                    # Check every pairing, rather than trusting the reuse identity.
                    orthogonal, _ = base.ridge_grid(rotate(z, mappings[arm]), age[fit],
                                                    rotate(held, mappings[arm]), ALPHAS)
                    np.testing.assert_allclose(orthogonal, raw_grid, rtol=1e-8, atol=1e-8)
                    transformed = rotate(z, mappings[arm], ratio)
                    evaluation = rotate(held, mappings[arm], ratio)
                prediction, rank = base.ridge_grid(transformed, age[fit], evaluation, np.array([alpha]))
                check_deadline(deadline)
                probe = base.fit_classifier(transformed, sites[fit])
                n_internal = len(split["internal"])
                site_prediction = probe.predict(evaluation[:n_internal])
                appearances = []
                for j, row in enumerate(both):
                    internal = j < n_internal
                    item = {"id": str(bundle.ids[row]), "domain": str(sites[row]),
                            "age": float(age[row]), "prediction": float(prediction[j, 0]),
                            "absolute_error": float(abs(prediction[j, 0] - age[row])),
                            "median_error": float(abs(np.median(age[fit]) - age[row])),
                            "role": "internal" if internal else "external", "view": view}
                    if internal:
                        item["site_correct"] = int(site_prediction[j] == sites[row])
                    appearances.append(item)
                records[arm] = {"view": view, "arm": arm, "alpha": alpha, "ratio": ratio,
                                "effective_rank": rank, "final_dimension": transformed.shape[1],
                                "inner_domain_macro_mae": means.tolist(),
                                "appearances": appearances}
                print(json.dumps({"stage": "outer", "view": view, "arm": arm,
                                  "alpha": alpha, "ratio": ratio}), flush=True)
            base.write_json(path, {"view": view, "family_pair_counts": counts, "arms": records,
                                   "orthonormal_raw_prediction_parity": True})
            return {"view": view, "complete": True, "reused": False}
        except DeadlineReached as error:
            return {"view": view, "complete": False, "reason": str(error)}


def paired_bootstrap(base_values, other_values, deadline, relative=False):
    """Same declared hierarchical bootstrap, with bounded batched arithmetic."""
    keys = sorted(base_values)
    if keys != sorted(other_values):
        raise ValueError("unpaired participants")
    domains = sorted({base_values[k]["domain"] for k in keys})
    groups = []
    for domain in domains:
        members = [k for k in keys if base_values[k]["domain"] == domain]
        if any(other_values[k]["domain"] != domain for k in members):
            raise ValueError("paired domain mismatch")
        groups.append(np.array([[base_values[k]["loss"], other_values[k]["loss"]] for k in members]))
    rng = np.random.RandomState(base.SEED)
    values = []
    for b in range(10000):
        if b % 100 == 0:
            check_deadline(deadline)
        means = []
        for di in rng.randint(len(groups), size=len(groups)):
            group = groups[di]
            means.append(group[rng.randint(len(group), size=len(group))].mean(axis=0))
        reference, alternative = np.mean(means, axis=0)
        values.append((alternative - reference) / reference if relative else alternative - reference)
    return {"descriptive_percentiles_2_5_97_5": np.quantile(values, [.025, .975]).tolist(),
            "upper_one_sided_95": float(np.quantile(values, .95)), "replicates": 10000,
            "seed": base.SEED, "scope": "observed_domains_only_exploratory_not_multiplicity_adjusted"}


def aggregate(out, deadline):
    outer = [base.read_json(out / "outer" / f"view{view}.json") for view in range(3)]
    records = {arm: [o["arms"][arm] for o in outer] for arm in ARMS}
    age = {arm: base.collapse(records[arm], "external", "absolute_error") for arm in ARMS}
    site = {arm: base.collapse(records[arm], "internal", "site_correct") for arm in ARMS}
    results = {}
    for arm in ARMS:
        results[arm] = {
            "external_age": base.summarize(age[arm]), "internal_site": base.summarize(site[arm]),
            "internal_age": base.summarize(base.collapse(records[arm], "internal", "absolute_error")),
            "per_view_external": [base.summarize(base.collapse([record], "external", "absolute_error"))
                                  for record in records[arm]],
            "per_view_internal_site": [base.summarize(base.collapse([record], "internal", "site_correct"))
                                       for record in records[arm]],
            "selected_alpha": [r["alpha"] for r in records[arm]],
            "selected_ratio": [r["ratio"] for r in records[arm]],
            "effective_rank": [r["effective_rank"] for r in records[arm]],
        }
    contrasts = {}
    for reference in ("raw", "sham14015", "sham14016", "sham14017"):
        ref, alt = results[reference], results["homologous"]
        delta = alt["external_age"]["domain_macro"] - ref["external_age"]["domain_macro"]
        contrasts[reference] = {
            "direction": "homologous_minus_reference", "delta_age": delta,
            "relative_age_change": delta / ref["external_age"]["domain_macro"],
            "delta_site": alt["internal_site"]["domain_macro"] - ref["internal_site"]["domain_macro"],
            "domain_age_changes": {domain: item["mean"] - ref["external_age"]["domains"][domain]["mean"]
                                   for domain, item in alt["external_age"]["domains"].items()},
            "leave_one_domain_out_age_changes": {
                domain: value - ref["external_age"]["leave_one_domain_out"][domain]
                for domain, value in alt["external_age"]["leave_one_domain_out"].items()},
            "age_interval": paired_bootstrap(age[reference], age["homologous"], deadline),
            "relative_age_interval": paired_bootstrap(age[reference], age["homologous"], deadline, True),
            "site_interval": paired_bootstrap(site[reference], site["homologous"], deadline),
        }
    report = {**SPEC, "complete": True, "results": results, "homologous_contrasts": contrasts,
              "mean_sham_MAE": float(np.mean([results[a]["external_age"]["domain_macro"] for a in ARMS[2:]])),
              "reference_margins_only": {"relative_age_change": -.02, "site_BAcc_change": .01},
              "claim": "exploratory_development_comparison_no_candidate_selection_or_confirmation"}
    base.write_json(out / "bilateral_regularization_report.json", report)
    lines = ["# Bilateral regularization: exposed development only", "",
             "All 757 public-validation rows remain closed. Frozen R0–R3 disposition is unchanged.", "",
             "| Arm | External domain-macro MAE | Internal site BAcc | α by view | ρ by view |",
             "| --- | ---: | ---: | --- | --- |"]
    for arm, result in results.items():
        lines.append(f"| {arm} | {result['external_age']['domain_macro']:.6f} | "
                     f"{result['internal_site']['domain_macro']:.6f} | {result['selected_alpha']} | "
                     f"{result['selected_ratio']} |")
    lines += ["", "All three fixed sham pairings are retained; none is selected by its result.",
              "Intervals are descriptive, conditional on observed domains, and not multiplicity adjusted.",
              "Raw has seven penalty settings; each paired arm has the same 35-setting allowance.",
              "Site-probe changes include the effect of feature metric on fixed probe regularization.",
              "Overlapping views are robustness descriptions, not independent replications.", ""]
    (out / "bilateral_regularization_report.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=3300)
    args = parser.parse_args()
    base.compute_only()
    if not 1 <= args.workers <= 3 or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) < args.workers:
        raise ValueError("requires 1--3 workers within actual allocation")
    if not 1 <= args.max_seconds <= 3300:
        raise ValueError("worker deadline must be at most 3300 seconds")
    out = Path(args.out).resolve()
    if not out.is_relative_to(EPISODE / "outputs/runs/bilateral_regularization"):
        raise ValueError("output must be this analysis's episode-local run directory")
    deadline = time.monotonic() + args.max_seconds
    spec = {**SPEC, "provider_family_suffixes": list(PROVIDER_FAMILIES)}
    path = out / "specification.json"
    if path.exists() and base.read_json(path) != spec:
        raise ValueError("existing attempt belongs to a different specification")
    names, mappings, counts = admitted_metadata()
    base.write_json(path, spec)
    base.write_json(out / "pairing_metadata.json", {
        "family_pair_counts": counts, "pairs_by_arm": {key: value.tolist() for key, value in mappings.items()},
        "source": "provider_feature_names_desikan_and_declared_measurement_suffixes"})
    started = time.monotonic()
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_view, view, str(out), deadline): view for view in range(3)}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {"view": futures[future], "complete": False,
                          "unexpected_error": f"{type(error).__name__}: {error}"}
            completed.append(result)
            print(json.dumps(result), flush=True)
    status = {"analysis_id": SPEC["analysis_id"], "views": completed,
              "seconds": time.monotonic() - started, "complete": False, "holdout_access": False}
    if all(result["complete"] for result in completed):
        try:
            with threadpool_limits(limits=1):
                aggregate(out, deadline)
            status["complete"] = True
        except DeadlineReached as error:
            status["reason"] = str(error)
        except Exception as error:
            status["unexpected_error"] = f"{type(error).__name__}: {error}"
    status["seconds"] = time.monotonic() - started
    base.write_json(out / "run_status.json", status)
    print(json.dumps(status), flush=True)
    if status["complete"]:
        return 0
    return 1 if "unexpected_error" in status or any("unexpected_error" in r for r in completed) else 2


if __name__ == "__main__":
    raise SystemExit(main())
