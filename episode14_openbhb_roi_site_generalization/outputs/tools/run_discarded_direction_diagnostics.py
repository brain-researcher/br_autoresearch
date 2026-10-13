#!/usr/bin/env python3
"""EP14 development-only discarded-score and matched-geometry diagnostics.

Every fitted quantity uses its assigned fit role. No public-holdout interface.
Run and test only in the coordinator's actual Slurm allocation.
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

SEED = 14014
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
KINDS = ("site", "random", "permuted")
ATLASES = ("desikan", "destrieux")
ROLES = ("external", "internal")
REPLICATES = 10000
OUTPUTS = Path(__file__).resolve().parents[1]
EPISODE = OUTPUTS.parent
EXPOSED = EPISODE / "inputs/openbhb/exposed"
DEFAULT_OUT = OUTPUTS / "runs/discarded_direction_diagnostics/attempt001"


class SupportFailure(RuntimeError):
    """A declared rank/capacity condition failed; never silently substitute."""


def alpha_name(alpha):
    return f"{alpha:g}".replace(".", "p")


def geometry_name(kind, alpha):
    return f"geometry_{kind}_alpha{alpha_name(alpha)}"


MODELS = ("age_median", "centered_zero") + tuple(f"discarded_{k}" for k in KINDS) + tuple(
    geometry_name(kind, alpha)
    for kind in ("raw", "retained", "rescaled", "restored") for alpha in ALPHAS
)


def require_close(actual, expected, label, atol=1e-8, rtol=1e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        difference = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise ValueError(f"{label}: maximum absolute difference {difference}")


class DirectionMaps:
    """Fit scales/directions without ages; transform accepts only feature rows."""

    def fit(self, x, sites):
        self.pre = legacy.Scale().fit(x)
        z = self.pre.transform(x)
        site, rank = legacy.site_directions(z, sites)
        permuted, permuted_rank = legacy.site_directions(z, np.random.RandomState(SEED).permutation(sites))
        if rank != 4 or permuted_rank != 4:
            raise SupportFailure(f"Four matched directions required; site={rank}, permuted={permuted_rank}")
        random = np.random.RandomState(SEED).normal(size=(z.shape[1], rank))
        random = np.linalg.qr(random, mode="reduced")[0].T
        self.directions = {"site": site, "random": random, "permuted": permuted}
        self.score_scales = {}
        for kind, directions in self.directions.items():
            require_close(directions @ directions.T, np.eye(4), f"{kind} orthonormal directions")
            scale = legacy.Scale().fit(z @ directions.T)
            if np.count_nonzero(scale.keep) != 4:
                raise SupportFailure(f"{kind} discarded-score variance has fewer than four nonzero columns")
            self.score_scales[kind] = scale
        retained = z - (z @ site.T) @ site
        self.retained_scale = legacy.Scale().fit(retained)
        return self

    def transform(self, x, include_geometry=False):
        z = self.pre.transform(x)
        raw_scores = {kind: z @ directions.T for kind, directions in self.directions.items()}
        scores = {kind: self.score_scales[kind].transform(d, discard=True) for kind, d in raw_scores.items()}
        if not include_geometry:
            if not all(np.isfinite(v).all() for v in scores.values()):
                raise ValueError("Nonfinite discarded scores")
            return scores, None
        retained = z - raw_scores["site"] @ self.directions["site"]
        geometry = {"raw": z, "retained": retained,
                    "rescaled": self.retained_scale.transform(retained, discard=True),
                    "restored": np.column_stack((retained, raw_scores["site"]))}
        if not all(np.isfinite(v).all() for v in (*scores.values(), *geometry.values())):
            raise ValueError("Nonfinite transformed values")
        return scores, geometry


def age_bins(ages):
    ages = np.asarray(ages)
    bins = np.floor(ages / 5.0).astype(int)
    bins[ages == 100.0] = 19
    return np.where((ages >= 0) & (ages <= 100), bins, -1)


def fit_support(ages, sites):
    bins = age_bins(ages)
    domains = sorted(set(sites))
    counts = {str(b): {str(d): int(np.count_nonzero((bins == b) & (sites == d))) for d in domains}
              for b in range(20)}
    eligible = [b for b in range(20) if sum(n >= 5 for n in counts[str(b)].values()) >= 5]
    return {"eligible_bins": eligible, "training_counts_by_bin_and_domain": counts,
            "bin_rule": "[0,5),...,[95,100]; age100 included; at least5 participants in each of at least5 fit domains",
            "outside_declared_age_range": int(np.count_nonzero(bins < 0))}


def choose_alpha(fold_losses):
    mean = np.mean(np.asarray(fold_losses), axis=0)
    index = min(range(len(ALPHAS)), key=lambda i: (mean[i], -ALPHAS[i]))
    return index, mean


def inner_fold(bundle, x, atlas, view, domain_number, out):
    path = out / "inner" / f"{atlas}_view{view}_domain{domain_number}.json"
    if path.exists():
        return legacy.read_json(path)
    started = time.monotonic()
    fit = bundle.splits[view]["fit"]
    domain = np.unique(bundle.sites[fit])[domain_number]
    train = fit[bundle.sites[fit] != domain]
    valid = fit[bundle.sites[fit] == domain]
    maps = DirectionMaps().fit(x[train], bundle.sites[train])
    scores, _ = maps.transform(x[train])
    held, _ = maps.transform(x[valid])
    losses = {}
    for kind in KINDS:
        predictions, rank = legacy.ridge_grid(scores[kind], bundle.age[train], held[kind], ALPHAS)
        if rank != 4:
            raise SupportFailure(f"{atlas} view{view} inner {domain}: {kind} score rank={rank}")
        losses[kind] = np.mean(np.abs(predictions - bundle.age[valid, None]), axis=0).tolist()
    record = {"atlas": atlas, "view": view, "held_domain": str(domain),
              "fit_rows": len(train), "validation_rows": len(valid), "mae_by_alpha": losses,
              "seconds": time.monotonic() - started}
    legacy.write_json(path, record)
    print(json.dumps({"stage": "inner", "atlas": atlas, "view": view,
                      "domain_number": domain_number, "seconds": record["seconds"]}), flush=True)
    return record


def outer_fold(bundle, x, atlas, view, out):
    path = out / "outer" / f"{atlas}_view{view}.json"
    if path.exists():
        return legacy.read_json(path)
    started = time.monotonic()
    split = bundle.splits[view]
    fit = split["fit"]
    inner = [inner_fold(bundle, x, atlas, view, i, out)
             for i in range(len(np.unique(bundle.sites[fit])))]
    maps = DirectionMaps().fit(x[fit], bundle.sites[fit])
    scores, geometry = maps.transform(x[fit], include_geometry=True)
    evaluated = np.concatenate((split["internal"], split["external"]))
    held_scores, held_geometry = maps.transform(x[evaluated], include_geometry=True)
    age = bundle.age[fit]
    predictions = {"age_median": np.full(len(evaluated), np.median(age)),
                   "centered_zero": np.zeros(len(evaluated))}
    tuning = {}
    for kind in KINDS:
        index, mean = choose_alpha([fold["mae_by_alpha"][kind] for fold in inner])
        predicted, rank = legacy.ridge_grid(scores[kind], age, held_scores[kind], ALPHAS[index:index + 1])
        if rank != 4:
            raise SupportFailure(f"{atlas} view{view} outer {kind}: score rank={rank}")
        predictions[f"discarded_{kind}"] = predicted[:, 0]
        tuning[kind] = {"alpha": float(ALPHAS[index]), "inner_domain_macro_mae": mean.tolist()}
    geometry_predictions = {}
    for kind in ("raw", "retained", "rescaled", "restored"):
        predicted, rank = legacy.ridge_grid(geometry[kind], age, held_geometry[kind], ALPHAS)
        geometry_predictions[kind] = predicted
        for column, alpha in enumerate(ALPHAS):
            predictions[geometry_name(kind, alpha)] = predicted[:, column]
    require_close(geometry_predictions["restored"], geometry_predictions["raw"], "restored/raw predictions at every alpha")
    support = fit_support(age, bundle.sites[fit])
    bins = age_bins(bundle.age[evaluated])
    selected = np.isin(bins, support["eligible_bins"])
    appearances = []
    for j, row in enumerate(evaluated):
        appearances.append({"id": str(bundle.ids[row]), "domain": str(bundle.sites[row]),
                            "view": view, "role": "internal" if j < len(split["internal"]) else "external",
                            "age": float(bundle.age[row]), "age_bin": int(bins[j]),
                            "on_common_support": bool(selected[j]),
                            "predictions": {name: float(predictions[name][j]) for name in MODELS}})
    record = {"atlas": atlas, "view": view, "tuning": tuning, "support": support,
              "geometry_alphas": ALPHAS.tolist(), "anchor_alpha": 1000.0,
              "restored_raw_max_abs_difference": float(np.max(np.abs(geometry_predictions["restored"] - geometry_predictions["raw"]))),
              "appearances": appearances, "seconds_including_uncached_inner": time.monotonic() - started}
    legacy.write_json(path, record)
    print(json.dumps({"stage": "outer", "atlas": atlas, "view": view,
                      "equivalence_max_error": record["restored_raw_max_abs_difference"]}), flush=True)
    return record


def group_arrays(rows):
    ids = sorted({r["id"] for r in rows})
    lookup = {key: i for i, key in enumerate(ids)}
    index = np.array([lookup[r["id"]] for r in rows], dtype=int)
    counts = np.bincount(index, minlength=len(ids))
    weights = 1.0 / (len(ids) * counts[index])
    residuals = np.array([[r["predictions"][name] - r["age"] for name in MODELS] for r in rows])
    ages = np.array([r["age"] for r in rows])
    mean_age = float(weights @ ages)
    person_abs = np.zeros((len(ids), len(MODELS)))
    np.add.at(person_abs, index, np.abs(residuals) / counts[index, None])
    offset = weights @ residuals
    absolute = weights @ np.abs(residuals)
    centered = np.sum(weights[:, None] * np.abs(residuals - offset), axis=0)
    age_sd = float(np.sqrt(weights @ ((ages - mean_age) ** 2)))
    return {"n": len(ids), "index": index, "counts": counts, "residuals": residuals,
            "person_abs": person_abs, "absolute": absolute, "centered": centered,
            "coverage": {"participants": len(ids), "appearances": len(rows),
                         "age_min": float(np.min(ages)), "age_max": float(np.max(ages)),
                         "age_mean": mean_age, "age_sd": age_sd,
                         "common_centered_eligible": len(ids) >= 10 and age_sd >= 2.5}}


def contrast_definitions(centered):
    reference = "centered_zero" if centered else "age_median"
    pairs = [(f"discarded_{kind}", reference) for kind in KINDS]
    pairs += [("discarded_site", "discarded_random"), ("discarded_site", "discarded_permuted")]
    for alpha in ALPHAS:
        pairs += [(geometry_name(a, alpha), geometry_name(b, alpha))
                  for a, b in (("retained", "raw"), ("rescaled", "retained"),
                               ("restored", "retained"), ("restored", "raw"))]
    return pairs


def bootstrap_models(groups, centered, replicates=REPLICATES, seed=SEED):
    """Pair all models; resample domains then participant bundles and recenter."""
    if len(groups) < 2:
        return None
    rng = np.random.RandomState(seed)
    domain_draws = rng.randint(len(groups), size=(replicates, len(groups)))
    estimates = np.zeros((replicates, len(MODELS)))
    for domain_index, group in enumerate(groups):
        positions = np.nonzero(domain_draws == domain_index)[0]
        n = group["n"]
        for start in range(0, len(positions), 64):
            selected_positions = positions[start:start + 64]
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=len(selected_positions))
            if centered:
                appearance_weights = counts[:, group["index"]] / (n * group["counts"][group["index"]])
                means = appearance_weights @ group["residuals"]
                values = np.sum(appearance_weights[:, :, None] *
                                np.abs(group["residuals"][None, :, :] - means[:, None, :]), axis=1)
            else:
                values = counts @ group["person_abs"] / n
            np.add.at(estimates, selected_positions, values / len(groups))
    return estimates


def score_panel(groups_by_domain, centered, common, intervals, restrict_to_centered_eligible=False):
    chosen = {d: g for d, g in groups_by_domain.items()
              if not ((centered or restrict_to_centered_eligible) and common) or g["coverage"]["common_centered_eligible"]}
    metric = "centered" if centered else "absolute"
    names = [n for n in MODELS if n != ("age_median" if centered else "centered_zero")]
    if not chosen:
        return {"status": "insufficient_support", "included_domains": [],
                "excluded_domains": list(groups_by_domain), "domain_macro_mae": None,
                "contrasts": {}, "bootstrap": {"status": "unavailable_no_eligible_domains"}}
    point = np.mean([g[metric] for g in chosen.values()], axis=0)
    boot = bootstrap_models(list(chosen.values()), centered) if intervals else None
    contrasts = {}
    for other, base in contrast_definitions(centered):
        a, b = MODELS.index(other), MODELS.index(base)
        item = {"other": other, "reference": base, "delta_mae_years": float(point[a] - point[b])}
        if boot is not None:
            delta = boot[:, a] - boot[:, b]
            item["descriptive_percentiles_2_5_97_5"] = np.quantile(delta, [0.025, 0.975]).tolist()
            item["upper_one_sided_95"] = float(np.quantile(delta, 0.95))
        contrasts[f"{other}_minus_{base}"] = item
    return {"status": "descriptive", "included_domains": list(chosen),
            "excluded_domains": [d for d in groups_by_domain if d not in chosen],
            "participants": sum(g["n"] for g in chosen.values()),
            "domain_macro_mae": {n: float(point[MODELS.index(n)]) for n in names},
            "contrasts": contrasts,
            "bootstrap": {"status": "complete" if boot is not None else ("unavailable_fewer_than_two_domains" if intervals else "not_requested_per_view"),
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
        group = group_arrays(selected)
        groups[domain] = group
        domains[domain] = {**coverage, **group["coverage"],
                           "status": "descriptive" if not common or group["coverage"]["common_centered_eligible"] else "insufficient_common_centered_support",
                           "absolute_mae": {n: float(group["absolute"][i]) for i, n in enumerate(MODELS) if n != "centered_zero"},
                           "centered_mae": {n: float(group["centered"][i]) for i, n in enumerate(MODELS) if n != "age_median"}}
    absolute = score_panel(groups, centered=False, common=common, intervals=intervals)
    centered = score_panel(groups, centered=True, common=common, intervals=intervals)
    panels = {"absolute": absolute, "centered": centered}
    if common:
        panels["absolute_on_centered_eligible_domains"] = score_panel(
            groups, centered=False, common=True, intervals=intervals, restrict_to_centered_eligible=True)
    for panel in panels.values():
        panel["excluded_domains"] = [d for d in domains if d not in panel["included_domains"]]
    return {"population": "fit_defined_common_support" if common else "all_rows",
            "centering": "within_this_panel_domain_with_participant_balanced_appearance_weights",
            "domains": domains, **panels}


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
        if set(row["predictions"]) != set(MODELS):
            raise ValueError("Missing paired model prediction")
        if not all(np.isfinite(v) for v in row["predictions"].values()):
            raise ValueError("Nonfinite prediction")
    for view in range(3):
        counts = Counter(r["role"] for r in rows if r["view"] == view)
        expected = dict(zip(("internal", "external"), legacy.EXPECTED_SPLITS[view][1:]))
        if counts != expected:
            raise ValueError(f"Unexpected cached role counts in view{view}")
    return rows


def aggregate(out):
    report = {"analysis_id": "discarded_direction_diagnostics", "evidence": "exploratory_development_only",
              "created_utc": datetime.now(timezone.utc).isoformat(), "holdout_access": False,
              "original_selection_changed": False, "geometry_anchor_alpha": 1000.0,
              "negative_contrast_means_lower_error": True, "near_zero_or_equivalence_claim": False,
              "independent_replications": False, "atlases": {}}
    atlas_signatures = []
    for atlas in ATLASES:
        records = [legacy.read_json(out / "outer" / f"{atlas}_view{view}.json") for view in range(3)]
        rows = validate_rows(records)
        atlas_signatures.append({(r["id"], r["view"], r["role"]): (r["age"], r["domain"], r["on_common_support"]) for r in rows})
        if atlas_signatures[-1] != atlas_signatures[0]:
            raise ValueError("Atlas support/participants disagree")
        result = {"outer_tuning": [r["tuning"] for r in records],
                  "fit_defined_support": [r["support"] for r in records],
                  "restored_raw_max_abs_difference": max(r["restored_raw_max_abs_difference"] for r in records),
                  "pooled": {}, "per_view": {}}
        for role in ROLES:
            subset = [r for r in rows if r["role"] == role]
            result["pooled"][role] = {key: summarize_population(subset, common, intervals=True)
                                      for key, common in (("all_rows", False), ("common_support", True))}
        for view in range(3):
            result["per_view"][str(view)] = {}
            for role in ROLES:
                subset = [r for r in rows if r["role"] == role and r["view"] == view]
                result["per_view"][str(view)][role] = {key: summarize_population(subset, common, intervals=False)
                                                       for key, common in (("all_rows", False), ("common_support", True))}
        report["atlases"][atlas] = result
        print(json.dumps({"stage": "aggregate", "atlas": atlas}), flush=True)
    legacy.write_json(out / "diagnostics_report.json", report)
    (out / "diagnostics_report.md").write_text(markdown(report))
    return report


def markdown(report):
    lines = ["# EP14 discarded-direction diagnostics", "", "Exploratory development evidence. Original no-candidate selection and closed public holdout are unchanged.", "",
             "Discarded-score predictors use fold-local directions/scales and nested leave-one-site-out tuning. Geometry uses five fixed penalties without outcome selection; alpha1000 is the prespecified anchor. Every evaluation row is retained in the all-row panels.", "",
             "Negative contrasts favor the first named model. Absolute utility compares against the outer-training age median. Centered utility compares against zero prediction of centered age; centering is evaluator-only and is recomputed within each bootstrap sample. It is not a deployable correction. Domains are equally weighted after participant-balanced appearance losses; predictions are not averaged across views.", "",
             "Intervals are 10,000 paired hierarchical bootstrap percentiles, conditional on observed support/domains. They are descriptive, with no equivalence claim or omnibus success criterion. All paired models use identical participants. Common-support centered summaries require at least10 unique participants and ageSD>=2.5years per domain; all exclusions are explicit. The absolute_on_centered_eligible_domains companion uses exactly those same domains/participants for joint absolute/centered interpretation; the broader absolute panel remains descriptive of all covered domains. Per-view panels are reported without intervals in JSON.", ""]
    for atlas in ATLASES:
        data = report["atlases"][atlas]
        lines += [f"## {atlas.capitalize()}", "", f"Maximum restored/raw prediction discrepancy across alphas: {data['restored_raw_max_abs_difference']:.3g} years.", ""]
        for population in ("all_rows", "common_support"):
            panel = data["pooled"]["external"][population]
            lines += [f"### External {population.replace('_', ' ')}", "", "| Metric | Comparison | Δ MAE, years | Descriptive 95% interval | Domains |", "| --- | --- | ---: | --- | ---: |"]
            metrics = ("absolute", "absolute_on_centered_eligible_domains", "centered") if population == "common_support" else ("absolute", "centered")
            for metric in metrics:
                scores = panel[metric]
                pairs = contrast_definitions(metric == "centered")[:5]
                pairs += [(geometry_name(a, 1000), geometry_name(b, 1000))
                          for a, b in (("retained", "raw"), ("rescaled", "retained"), ("restored", "retained"))]
                if not scores["contrasts"]:
                    lines.append(f"| {metric} | insufficient support | — | unavailable | 0 |")
                for a, b in pairs:
                    item = scores["contrasts"].get(f"{a}_minus_{b}")
                    if item is None:
                        continue
                    interval = item.get("descriptive_percentiles_2_5_97_5")
                    ci = f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval is not None else "unavailable"
                    lines.append(f"| {metric} | {a} − {b} | {item['delta_mae_years']:.4f} | {ci} | {len(scores['included_domains'])} |")
            lines += ["", "| Domain | Supported / all participants | Supported / all appearances | Age range | Age SD | Common centered eligible |", "| --- | ---: | ---: | --- | ---: | --- |"]
            for domain, coverage in panel["domains"].items():
                age_range = f"{coverage['age_min']:.2f}–{coverage['age_max']:.2f}" if coverage["participants"] else "none"
                age_sd = f"{coverage['age_sd']:.3f}" if coverage["participants"] else "—"
                lines.append(f"| {domain} | {coverage['participants']} / {coverage['total_participants']} | {coverage['appearances']} / {coverage['total_appearances']} | {age_range} | {age_sd} | {coverage['common_centered_eligible']} |")
            lines.append("")
    lines += ["The complete JSON retains both roles, every domain, all view-specific results, all fixed-alpha curves, tuning choices, support counts and paired contrasts. No support fallback or age/domain selection was made from prediction outcomes. Composition, regularization geometry, redundancy and biological explanations remain distinct; predictive utility alone does not identify causality or a biological aging mechanism.", ""]
    return "\n".join(lines)


def initialize_attempt(out, resume):
    fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (Path(__file__).resolve(), Path(legacy.__file__).resolve())}
    manifest = out / "attempt.json"
    if resume:
        if not manifest.exists() or legacy.read_json(manifest)["code_sha256"] != fingerprint:
            raise ValueError("Resume requires an existing attempt with exactly the same runner/dependency code")
    else:
        out.mkdir(parents=True, exist_ok=False)
        legacy.write_json(manifest, {"analysis_id": "discarded_direction_diagnostics", "code_sha256": fingerprint,
                                    "created_utc": datetime.now(timezone.utc).isoformat(),
                                    "source": str(EXPOSED), "hostname": socket.gethostname(),
                                    "job_id": os.environ.get("SLURM_JOB_ID"), "seed": SEED})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--representative-one-fold", action="store_true",
                        help="Fit and checkpoint Destrieux view0 inner domain0, then stop; --resume reuses it")
    args = parser.parse_args()
    legacy.compute_only()
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if not 1 <= args.threads <= allocated:
        raise ValueError("Thread request must fit SLURM_CPUS_PER_TASK")
    out = args.out.resolve()
    if not out.is_relative_to(OUTPUTS / "runs/discarded_direction_diagnostics"):
        raise ValueError("Attempt must be under this diagnostic's episode-local run directory")
    initialize_attempt(out, args.resume)
    started = time.monotonic()
    try:
        with threadpool_limits(limits=args.threads):
            bundle = legacy.Bundle(EXPOSED)
            if args.representative_one_fold:
                x, _ = bundle.atlas("destrieux")
                record = inner_fold(bundle, x, "destrieux", 0, 0, out)
                print(json.dumps({"representative_complete": True, "seconds": record["seconds"],
                                  "fit_rows": record["fit_rows"], "validation_rows": record["validation_rows"]}), flush=True)
                return
            for atlas in ATLASES:
                x, _ = bundle.atlas(atlas)
                for view in range(3):
                    outer_fold(bundle, x, atlas, view, out)
            aggregate(out)
    except SupportFailure as error:
        legacy.write_json(out / "support_failure.json", {"status": "support_failure", "reason": str(error),
                                                         "no_capacity_substitution": True})
        raise
    print(json.dumps({"complete": True, "analysis_id": "discarded_direction_diagnostics",
                      "seconds": time.monotonic() - started, "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
