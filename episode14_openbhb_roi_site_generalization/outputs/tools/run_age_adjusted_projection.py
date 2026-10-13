#!/usr/bin/env python3
"""Fixed EP14 linear-age-adjusted direction diagnostic; development only.

All fitted state uses outer-training rows. Evaluation transforms take features
only. Actual-age versus within-site-shuffled-age pairing is the comparison;
residualizing training ROIs does not guarantee age-protected directions.
Execute only in the coordinator's bounded Slurm allocation.
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

import run_development as legacy
import run_discarded_direction_diagnostics as prior

SEED = 14014
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
ARMS = ("raw", "original", "age_adjusted", "sham_adjusted")
ATLASES = ("desikan", "destrieux")
ROLES = ("external", "internal")
REPLICATES = 10000
OUTPUTS = Path(__file__).resolve().parents[1]
EXPOSED = OUTPUTS.parent / "inputs/openbhb/exposed"
RUN_ROOT = OUTPUTS / "runs/age_adjusted_projection"
ANALYSIS_ID = "age_adjusted_projection"


def model_name(arm, alpha):
    return f"{arm}_alpha{alpha:g}".replace(".", "p")


MODELS = tuple(model_name(arm, alpha) for arm in ARMS for alpha in ALPHAS)
PAIRS = (("age_adjusted", "original"), ("sham_adjusted", "original"),
         ("age_adjusted", "sham_adjusted"), ("age_adjusted", "raw"),
         ("original", "raw"))


def within_site_sham(ages, sites, seed=SEED):
    """One seeded permutation, sorted domains and original fit-row order."""
    result = np.asarray(ages, dtype=float).copy()
    rng = np.random.RandomState(seed)
    for domain in sorted(set(sites)):
        rows = np.flatnonzero(sites == domain)
        result[rows] = rng.permutation(result[rows])
    return result


def regress_age(z, ages):
    """OLS ROI~intercept+standardized age in the original standardized metric."""
    ages = np.asarray(ages, dtype=float)
    mean, scale = float(np.mean(ages)), float(np.std(ages, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise prior.SupportFailure("Training age has no finite positive variation")
    a = (ages - mean) / scale
    design = np.column_stack((np.ones(len(a)), a))
    coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank != 2:
        raise prior.SupportFailure("Training age design lacks rank two")
    residual = z - design @ coefficients
    prior.require_close(design.T @ residual / len(a), np.zeros((2, z.shape[1])),
                        "OLS residual orthogonality", atol=1e-10, rtol=1e-10)
    return residual, {"age_mean": mean, "age_scale": scale,
                      "coefficients": coefficients, "design_rank": int(rank)}


class DirectionMaps:
    def fit(self, x, sites, ages):
        if not (np.isfinite(x).all() and np.isfinite(ages).all()):
            raise ValueError("Nonfinite fit features or ages")
        if len(x) != len(sites) or len(x) != len(ages):
            raise ValueError("Fit inputs are not aligned")
        self.pre = legacy.Scale().fit(x)
        z = self.pre.transform(x)
        sham_age = within_site_sham(ages, sites)
        actual, self.actual_regression = regress_age(z, ages)
        sham, self.sham_regression = regress_age(z, sham_age)
        self.directions = {}
        self.metadata = {"fit_rows": len(x), "input_columns": x.shape[1],
                         "fit_domain_count": len(set(sites)),
                         "age_scale": self.actual_regression["age_scale"],
                         "age_mean": self.actual_regression["age_mean"],
                         "sham_rows_changed": int(np.count_nonzero(sham_age != ages)),
                         "sham_preserves_each_site_age_multiset": True,
                         "arms": {}}
        covariance = z.T @ (ages - np.mean(ages)) / len(ages)
        covariance_energy = float(covariance @ covariance)
        for arm, matrix in (("original", z), ("age_adjusted", actual),
                            ("sham_adjusted", sham)):
            directions, rank = legacy.site_directions(matrix, sites)
            if rank != 4:
                raise prior.SupportFailure(f"{arm} removal requires four directions; got {rank}")
            prior.require_close(directions @ directions.T, np.eye(4), f"{arm} directions")
            self.directions[arm] = directions
            removed = (z @ directions.T) @ directions
            projected_covariance = directions @ covariance
            self.metadata["arms"][arm] = {
                "removed_rank": rank,
                "fit_removed_frobenius_norm": float(np.linalg.norm(removed, ord="fro")),
                "fit_removed_energy_fraction": float(np.sum(removed**2) / np.sum(z**2)),
                "fit_marginal_age_covariance_norm": float(np.sqrt(covariance_energy)),
                "removed_age_covariance_norm": float(np.linalg.norm(projected_covariance)),
                "removed_age_covariance_squared_norm_fraction":
                    float(projected_covariance @ projected_covariance / covariance_energy)
                    if covariance_energy > 0 else None,
                "direction_signed_cosines_with_age_covariance":
                    (projected_covariance / np.sqrt(covariance_energy)).tolist()
                    if covariance_energy > 0 else None,
                "training_residual_covariance_with_actual_age_norm":
                    float(np.linalg.norm(matrix.T @ (ages - np.mean(ages)) / len(ages))),
            }
        return self

    def transform(self, x):
        """Features only: do not residualize evaluation features using ages."""
        z = self.pre.transform(x)
        representations = {"raw": z}
        for arm, directions in self.directions.items():
            representations[arm] = z - (z @ directions.T) @ directions
        if not all(np.isfinite(value).all() for value in representations.values()):
            raise ValueError("Nonfinite transformed features")
        return representations

    def save_state(self, path):
        np.savez_compressed(path, pre_mean=self.pre.mean, pre_scale=self.pre.scale,
                            actual_age_coefficients=self.actual_regression["coefficients"],
                            sham_age_coefficients=self.sham_regression["coefficients"],
                            **{f"directions_{k}": v for k, v in self.directions.items()})


def outer_fold(bundle, x, atlas, view, out):
    path = out / "outer" / f"{atlas}_view{view}.json"
    if path.exists():
        return legacy.read_json(path)
    started = time.monotonic()
    split = bundle.splits[view]
    fit = split["fit"]
    if len(np.unique(bundle.ids[fit])) != len(fit):
        raise ValueError("Training must contain one row per participant")
    evaluated = np.concatenate((split["internal"], split["external"]))
    maps = DirectionMaps().fit(x[fit], bundle.sites[fit], bundle.age[fit])
    train = maps.transform(x[fit])
    held = maps.transform(x[evaluated])
    predictions, correct, ranks = {}, {}, {}
    n_internal = len(split["internal"])
    for arm in ARMS:
        predicted, rank = legacy.ridge_grid(train[arm], bundle.age[fit], held[arm], ALPHAS)
        if arm != "raw" and rank > min(len(fit) - 1, x.shape[1] - 4):
            raise prior.SupportFailure(f"{arm}: retained rank includes a removed direction")
        ranks[arm] = rank
        for column, alpha in enumerate(ALPHAS):
            predictions[model_name(arm, alpha)] = predicted[:, column]
        probe = legacy.fit_classifier(train[arm], bundle.sites[fit])
        if not set(bundle.sites[split["internal"]]).issubset(set(probe.classes_)):
            raise ValueError("Internal probe row belongs to an unseen fitting domain")
        correct[arm] = probe.predict(held[arm][:n_internal]) == bundle.sites[split["internal"]]
    support = prior.fit_support(bundle.age[fit], bundle.sites[fit])
    bins = prior.age_bins(bundle.age[evaluated])
    selected = np.isin(bins, support["eligible_bins"])
    appearances = []
    for j, row in enumerate(evaluated):
        appearances.append({"id": str(bundle.ids[row]), "domain": str(bundle.sites[row]),
                            "view": view, "role": "internal" if j < n_internal else "external",
                            "age": float(bundle.age[row]), "age_bin": int(bins[j]),
                            "on_common_support": bool(selected[j]),
                            "predictions": {name: float(predictions[name][j]) for name in MODELS},
                            "site_correctness": {arm: bool(correct[arm][j]) for arm in ARMS}
                            if j < n_internal else None})
    path.parent.mkdir(parents=True, exist_ok=True)
    maps.save_state(path.with_suffix(".npz"))
    record = {"atlas": atlas, "view": view, "direction_metadata": maps.metadata,
              "retained_effective_ranks": ranks, "support": support,
              "alphas": ALPHAS.tolist(), "anchor_alpha": 1000.0,
              "alpha_selection": "none_fixed_curve", "sham_seed": SEED,
              "site_probe": "C1_balanced_unrescaled_same_representation_no_age_input",
              "appearances": appearances, "seconds": time.monotonic() - started}
    legacy.write_json(path, record)
    print(json.dumps({"stage": "outer", "atlas": atlas, "view": view,
                      "seconds": record["seconds"]}), flush=True)
    return record


def group_arrays(rows, models=MODELS):
    ids = sorted({r["id"] for r in rows})
    lookup = {key: i for i, key in enumerate(ids)}
    index = np.array([lookup[r["id"]] for r in rows], dtype=int)
    counts = np.bincount(index, minlength=len(ids))
    weights = 1.0 / (len(ids) * counts[index])
    residuals = np.array([[r["predictions"][name] - r["age"] for name in models] for r in rows])
    ages = np.array([r["age"] for r in rows])
    mean_age = float(weights @ ages)
    person_abs = np.zeros((len(ids), len(models)))
    np.add.at(person_abs, index, np.abs(residuals) / counts[index, None])
    offset = weights @ residuals
    age_sd = float(np.sqrt(weights @ ((ages - mean_age)**2)))
    return {"n": len(ids), "index": index, "counts": counts, "residuals": residuals,
            "person_abs": person_abs, "absolute": weights @ np.abs(residuals),
            "centered": np.sum(weights[:, None] * np.abs(residuals - offset), axis=0),
            "coverage": {"participants": len(ids), "appearances": len(rows),
                         "age_min": float(np.min(ages)), "age_max": float(np.max(ages)),
                         "age_mean": mean_age, "age_sd": age_sd,
                         "common_centered_eligible": len(ids) >= 10 and age_sd >= 2.5}}


def bootstrap_models(groups, centered, models=MODELS, replicates=REPLICATES, seed=SEED):
    """Same paired draws as the prior diagnostic, explicit model dimension."""
    if len(groups) < 2:
        return None
    rng = np.random.RandomState(seed)
    domain_draws = rng.randint(len(groups), size=(replicates, len(groups)))
    estimates = np.zeros((replicates, len(models)))
    for domain_index, group in enumerate(groups):
        positions = np.nonzero(domain_draws == domain_index)[0]
        n = group["n"]
        for start in range(0, len(positions), 64):
            selected_positions = positions[start:start + 64]
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=len(selected_positions))
            if centered:
                weights = counts[:, group["index"]] / (n * group["counts"][group["index"]])
                means = weights @ group["residuals"]
                values = np.sum(weights[:, :, None] *
                                np.abs(group["residuals"][None, :, :] - means[:, None, :]), axis=1)
            else:
                values = counts @ group["person_abs"] / n
            np.add.at(estimates, selected_positions, values / len(groups))
    return estimates


def contrast_definitions():
    return [(model_name(a, alpha), model_name(b, alpha)) for alpha in ALPHAS for a, b in PAIRS]


def score_panel(groups_by_domain, centered, common, intervals, restrict_to_centered_eligible=False):
    chosen = {d: g for d, g in groups_by_domain.items()
              if not ((centered or restrict_to_centered_eligible) and common)
              or g["coverage"]["common_centered_eligible"]}
    metric = "centered" if centered else "absolute"
    if not chosen:
        return {"status": "insufficient_support", "included_domains": [],
                "excluded_domains": list(groups_by_domain), "domain_macro_mae": None,
                "contrasts": {}, "bootstrap": {"status": "unavailable_no_eligible_domains"}}
    point = np.mean([g[metric] for g in chosen.values()], axis=0)
    boot = bootstrap_models(list(chosen.values()), centered, models=MODELS) if intervals else None
    contrasts = {}
    for other, base in contrast_definitions():
        a, b = MODELS.index(other), MODELS.index(base)
        item = {"other": other, "reference": base, "delta_mae_years": float(point[a] - point[b])}
        if boot is not None:
            item["descriptive_percentiles_2_5_97_5"] = np.quantile(boot[:, a] - boot[:, b], [0.025, 0.975]).tolist()
        contrasts[f"{other}_minus_{base}"] = item
    return {"status": "descriptive", "included_domains": list(chosen),
            "excluded_domains": [d for d in groups_by_domain if d not in chosen],
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
        selected = [r for r in all_rows if r["on_common_support"] or not common]
        coverage = {"total_participants": len({r["id"] for r in all_rows}), "total_appearances": len(all_rows)}
        if not selected:
            domains[domain] = {**coverage, "status": "no_rows_on_fit_defined_support", "participants": 0,
                               "appearances": 0, "common_centered_eligible": False,
                               "absolute_mae": None, "centered_mae": None}
            continue
        group = group_arrays(selected, models=MODELS)
        groups[domain] = group
        domains[domain] = {**coverage, **group["coverage"],
                           "status": "descriptive" if not common or group["coverage"]["common_centered_eligible"]
                           else "insufficient_common_centered_support",
                           "absolute_mae": dict(zip(MODELS, map(float, group["absolute"]))),
                           "centered_mae": dict(zip(MODELS, map(float, group["centered"])))}
    panels = {"absolute": score_panel(groups, False, common, intervals),
              "centered": score_panel(groups, True, common, intervals)}
    if common:
        panels["absolute_on_centered_eligible_domains"] = score_panel(
            groups, False, True, intervals, restrict_to_centered_eligible=True)
    for panel in panels.values():
        panel["excluded_domains"] = [d for d in domains if d not in panel["included_domains"]]
    return {"population": "fit_defined_common_support" if common else "all_rows",
            "centering": "within_panel_domain_participant_balanced_appearances",
            "domains": domains, **panels}


def site_probe_summary(rows, intervals=False, replicates=REPLICATES):
    by_domain = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["role"] != "internal" or row["site_correctness"] is None:
            raise ValueError("Site probe summary accepts internal correctness only")
        by_domain[row["domain"]][row["id"]].append(row)
    domains, groups = {}, []
    for domain, persons in sorted(by_domain.items()):
        person_correct = np.array([[np.mean([r["site_correctness"][arm] for r in appearances])
                                    for arm in ARMS] for appearances in persons.values()])
        domains[domain] = {"participants": len(persons),
                           "appearances": sum(map(len, persons.values())),
                           "class_recall": dict(zip(ARMS, map(float, person_correct.mean(axis=0))))}
        # The non-centered paired bootstrap accepts participant-level values;
        # correctness is averaged before class recall exactly as age losses are.
        groups.append({"n": len(persons), "person_abs": person_correct})
    accuracy = {arm: float(np.mean([d["class_recall"][arm] for d in domains.values()])) for arm in ARMS}
    boot = bootstrap_models(groups, centered=False, models=ARMS, replicates=replicates) if intervals else None
    contrasts = {}
    for a, b in PAIRS:
        item = {"delta_balanced_accuracy": accuracy[a] - accuracy[b]}
        if boot is not None:
            item["descriptive_percentiles_2_5_97_5"] = np.quantile(
                boot[:, ARMS.index(a)] - boot[:, ARMS.index(b)], [0.025, 0.975]).tolist()
        contrasts[f"{a}_minus_{b}"] = item
    return {"domains": domains, "balanced_accuracy": accuracy,
            "contrasts": contrasts,
            "bootstrap": {"replicates": replicates if boot is not None else 0, "seed": SEED,
                          "status": "complete" if boot is not None else
                          ("unavailable_fewer_than_two_domains" if intervals else "not_requested_per_view"),
                          "scope": "conditional_on_observed_internal_domains",
                          "participant_appearances_kept_together": True},
            "scope": "ordinary_unconditional_site_probe_descriptive_no_new_guardrail",
            "weighting": "correctness_first_then_appearances_within_participant_then_class_recall"}


def validate_rows(records):
    rows = [row for record in records for row in record["appearances"]]
    identities, seen = {}, set()
    for row in rows:
        key = (row["id"], row["view"], row["role"])
        if key in seen:
            raise ValueError("Repeated identical appearance")
        seen.add(key)
        value = (row["domain"], row["age"])
        if identities.setdefault(row["id"], value) != value:
            raise ValueError("Participant age/domain changed across appearances")
        if set(row["predictions"]) != set(MODELS) or not all(np.isfinite(v) for v in row["predictions"].values()):
            raise ValueError("Missing paired finite predictions")
        if row["role"] == "internal":
            if row["site_correctness"] is None or set(row["site_correctness"]) != set(ARMS):
                raise ValueError("Missing internal site correctness")
        elif row["site_correctness"] is not None:
            raise ValueError("Unseen domains must not receive a seen-domain site score")
    for view in range(3):
        counts = Counter(r["role"] for r in rows if r["view"] == view)
        expected = dict(zip(("internal", "external"), legacy.EXPECTED_SPLITS[view][1:]))
        if counts != expected:
            raise ValueError(f"Unexpected cached role counts in view{view}")
    return rows


def aggregate(out):
    report = {"analysis_id": ANALYSIS_ID, "evidence": "exploratory_development_only",
              "created_utc": datetime.now(timezone.utc).isoformat(), "public_validation_access": False,
              "original_selection_changed": False, "anchor_alpha": 1000.0,
              "fixed_alphas": ALPHAS.tolist(), "alpha_selection": "none",
              "age_covariance_protection_guaranteed": False, "independent_replications": False,
              "conditional_site_probe": False, "atlases": {}}
    first_signature = None
    for atlas in ATLASES:
        records = [legacy.read_json(out / "outer" / f"{atlas}_view{view}.json") for view in range(3)]
        rows = validate_rows(records)
        signature = {(r["id"], r["view"], r["role"]):
                     (r["age"], r["domain"], r["on_common_support"]) for r in rows}
        if first_signature is None:
            first_signature = signature
        elif signature != first_signature:
            raise ValueError("Atlas support/participants disagree")
        result = {"direction_metadata": [r["direction_metadata"] for r in records],
                  "fit_defined_support": [r["support"] for r in records],
                  "pooled": {}, "per_view": {}, "site_probe": {}}
        for role in ROLES:
            subset = [r for r in rows if r["role"] == role]
            result["pooled"][role] = {key: summarize_population(subset, common, intervals=True)
                                      for key, common in (("all_rows", False), ("common_support", True))}
        result["site_probe"]["pooled_internal"] = site_probe_summary(
            [r for r in rows if r["role"] == "internal"], intervals=True)
        result["site_probe"]["per_view_internal"] = {}
        for view in range(3):
            result["per_view"][str(view)] = {}
            for role in ROLES:
                subset = [r for r in rows if r["role"] == role and r["view"] == view]
                result["per_view"][str(view)][role] = {key: summarize_population(subset, common, intervals=False)
                                                       for key, common in (("all_rows", False), ("common_support", True))}
            result["site_probe"]["per_view_internal"][str(view)] = site_probe_summary(
                [r for r in rows if r["role"] == "internal" and r["view"] == view])
        report["atlases"][atlas] = result
        print(json.dumps({"stage": "aggregate", "atlas": atlas}), flush=True)
    legacy.write_json(out / "age_adjusted_report.json", report)
    (out / "age_adjusted_report.md").write_text(markdown(report))
    return report


def markdown(report):
    lines = ["# Linear-age-adjusted direction diagnostic", "",
             "Exploratory exposed-development evidence. Original no-candidate decision and public holdout remain unchanged.", "",
             "Training ROI~linear age OLS precedes direction estimation only. Evaluation transforms project original standardized ROIs with no ages or post-projection scaling. The sham permutes fitting ages within each site once. Downstream ridge always fits actual training ages.", "",
             "Alpha1000 is fixed; all five penalties are sensitivity descriptions without outcome selection. Negative age contrasts favor the first arm. Intervals use 10,000 paired domain/participant bootstrap draws with recentering, conditional on observed domains. Common-support eligibility and evaluator-only centering match the preceding diagnostic. No biological, causal, equivalence or age-preservation conclusion follows automatically.", ""]
    for atlas in ATLASES:
        data = report["atlases"][atlas]
        lines += [f"## {atlas.capitalize()}", ""]
        for population in ("all_rows", "common_support"):
            panel = data["pooled"]["external"][population]
            lines += [f"### External {population.replace('_', ' ')}", "",
                      "| Metric | Comparison, alpha1000 | Δ MAE, years | Descriptive 95% interval | Domains |",
                      "| --- | --- | ---: | --- | ---: |"]
            metrics = ("absolute", "absolute_on_centered_eligible_domains", "centered") if population == "common_support" else ("absolute", "centered")
            for metric in metrics:
                score = panel[metric]
                for a, b in PAIRS:
                    key = f"{model_name(a, 1000)}_minus_{model_name(b, 1000)}"
                    value = score["contrasts"].get(key)
                    if value is None:
                        continue
                    interval = value.get("descriptive_percentiles_2_5_97_5")
                    ci = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval is not None else "unavailable"
                    lines.append(f"| {metric} | {a} − {b} | {value['delta_mae_years']:.4f} | {ci} | {len(score['included_domains'])} |")
            lines += ["", "| Domain | Supported/all participants | Supported/all appearances | Age range | Age SD | Centered eligible |",
                      "| --- | ---: | ---: | --- | ---: | --- |"]
            for domain, coverage in panel["domains"].items():
                age_range = f"{coverage['age_min']:.2f}–{coverage['age_max']:.2f}" if coverage["participants"] else "none"
                age_sd = f"{coverage['age_sd']:.3f}" if coverage["participants"] else "—"
                lines.append(f"| {domain} | {coverage['participants']}/{coverage['total_participants']} | {coverage['appearances']}/{coverage['total_appearances']} | {age_range} | {age_sd} | {coverage['common_centered_eligible']} |")
            lines.append("")
        lines += ["### Ordinary seen-site probe", "", "| Representation | Balanced accuracy |", "| --- | ---: |"]
        for arm, value in data["site_probe"]["pooled_internal"]["balanced_accuracy"].items():
            lines.append(f"| {arm} | {value:.6f} |")
        lines += ["", "| Comparison | Δ balanced accuracy | Descriptive 95% interval |",
                  "| --- | ---: | --- |"]
        for contrast, value in data["site_probe"]["pooled_internal"]["contrasts"].items():
            interval = value.get("descriptive_percentiles_2_5_97_5")
            ci = f"[{interval[0]:.6f}, {interval[1]:.6f}]" if interval is not None else "unavailable"
            lines.append(f"| {contrast} | {value['delta_balanced_accuracy']:.6f} | {ci} |")
        lines += ["", "### Direction alignment", "", "Squared fraction of training marginal age-covariance norm removed; this is a diagnostic, not an age-preservation guarantee.", "",
                  "| View | Original | Actual-age adjusted | Within-site sham |", "| --- | ---: | ---: | ---: |"]
        for view, metadata in enumerate(data["direction_metadata"]):
            values = [metadata["arms"][arm]["removed_age_covariance_squared_norm_fraction"] for arm in ARMS[1:]]
            labels = [f"{v:.6f}" if v is not None else "undefined" for v in values]
            lines.append(f"| {view} | " + " | ".join(labels) + " |")
        lines.append("")
    lines += ["The JSON preserves all domains, roles, overlapping views, fixed penalties, paired contrasts, support and site-class recalls. Per-row predictions and correctness and fitted direction state are retained in the six outer checkpoints. The single sham realization conditions this comparison; it is not a permutation test or independent replication.", ""]
    return "\n".join(lines)


def initialize_attempt(out, resume):
    fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__).resolve(), Path(legacy.__file__).resolve(), Path(prior.__file__).resolve())}
    manifest = out / "attempt.json"
    if resume:
        if not manifest.exists() or legacy.read_json(manifest)["code_sha256"] != fingerprint:
            raise ValueError("Resume requires the identical runner and dependency code")
    else:
        out.mkdir(parents=True, exist_ok=False)
        legacy.write_json(manifest, {"analysis_id": ANALYSIS_ID, "code_sha256": fingerprint,
                                    "created_utc": datetime.now(timezone.utc).isoformat(),
                                    "source": str(EXPOSED), "hostname": socket.gethostname(),
                                    "job_id": os.environ.get("SLURM_JOB_ID"), "seed": SEED,
                                    "anchor_alpha": 1000.0, "public_validation_access": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN_ROOT / "attempt001")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--representative-one-fold", action="store_true",
                        help="Checkpoint Destrieux view0 only; reuse it with --resume")
    args = parser.parse_args()
    legacy.compute_only()
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if not 1 <= args.threads <= allocated:
        raise ValueError("Thread request must fit SLURM_CPUS_PER_TASK")
    out = args.out.resolve()
    if not out.is_relative_to(RUN_ROOT):
        raise ValueError("Output must be under this analysis's episode-local run root")
    initialize_attempt(out, args.resume)
    started = time.monotonic()
    try:
        with threadpool_limits(limits=args.threads):
            bundle = legacy.Bundle(EXPOSED)
            if args.representative_one_fold:
                x, _ = bundle.atlas("destrieux")
                record = outer_fold(bundle, x, "destrieux", 0, out)
                print(json.dumps({"representative_complete": True, "atlas": "destrieux", "view": 0,
                                  "seconds": record["seconds"], "output": str(out)}), flush=True)
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
    legacy.write_json(out / "completion.json", {"analysis_id": ANALYSIS_ID, "status": "complete",
                                               "seconds": time.monotonic() - started,
                                               "job_id": os.environ.get("SLURM_JOB_ID"),
                                               "hostname": socket.gethostname(),
                                               "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"complete": True, "analysis_id": ANALYSIS_ID,
                      "seconds": time.monotonic() - started, "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
